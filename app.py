import os
import math
import json
import base64
import asyncio
import logging
import audioop
import numpy as np
import aiohttp
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

###############################################################################
# CONFIG
###############################################################################

PUBLIC_BASE_URL = "https://riteway-ai-agent.onrender.com"
WS_MEDIA_URL = "wss://" + PUBLIC_BASE_URL.replace("https://", "").replace("http://", "") + "/media"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = "gpt-4o-realtime-preview"

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Riteway AI Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# AUDIO HELPERS
###############################################################################

def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
    """
    Convert PCM16 numpy array -> G.711 μ-law bytes.
    Twilio expects 8kHz μ-law, 20ms frames (160 bytes).
    """
    BIAS = 0x84
    CLIP = 32635
    out = bytearray()
    for s in pcm16.astype(np.int32):
        sign = 0x80 if s < 0 else 0x00
        if s < 0:
            s = -s
        if s > CLIP:
            s = CLIP
        s = s + BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (s & mask):
            mask >>= 1
            exponent -= 1
        mantissa = (s >> (exponent + 3)) & 0x0F
        ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        out.append(ulaw_byte)
    return bytes(out)

def generate_beep_ulaw_chunks(duration_sec=0.8, freq_hz=440.0, sample_rate=8000):
    """
    Safety tone, in case we ever need to send audio while AI is booting.
    We won't actively play this now unless something is badly wrong.
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.arange(total_samples) / sample_rate
    pcm16 = (10000 * np.sin(2 * math.pi * freq_hz * t)).astype(np.int16)
    ulaw_bytes = pcm16_to_ulaw(pcm16)

    frame_size = 160  # 20ms @ 8kHz
    chunks_b64 = []
    for i in range(0, len(ulaw_bytes), frame_size):
        frame = ulaw_bytes[i:i+frame_size]
        if not frame:
            continue
        b64_payload = base64.b64encode(frame).decode("ascii")
        chunks_b64.append(b64_payload)

    return chunks_b64

async def send_ulaw_chunks_to_twilio(ws: WebSocket, stream_sid: str, chunks_b64: list):
    """
    Send pre-encoded μ-law 20ms frames to Twilio with ~20ms pacing.
    """
    logging.info(f"🔊 sending {len(chunks_b64)} fallback frames to Twilio (sid={stream_sid})")
    for frame_b64 in chunks_b64:
        await ws.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": frame_b64
            }
        })
        await asyncio.sleep(0.02)
    logging.info("🔊 finished sending fallback frames")

###############################################################################
# ROUTES
###############################################################################

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "model": REALTIME_MODEL,
        "has_api_key": bool(OPENAI_API_KEY)
    })

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """
    Twilio calls this first.
    We return TwiML that IMMEDIATELY starts streaming the call audio
    to our /media WebSocket. We REMOVED the Twilio <Say> so callers
    do NOT hear 'Connecting you...' first.
    """
    logging.info("☎ Twilio hit /voice")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

@app.websocket("/media")
async def media(ws: WebSocket):
    """
    Live call bridge:
      Twilio <-> (this server) <-> OpenAI Realtime Voice API
    """
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None
    openai_connected = False
    oai_ws_handle = None

    # μ-law audio chunks from OpenAI waiting to go back to Twilio
    playback_queue = deque()
    playback_running = True
    playback_task = None

    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY in environment.")
    else:
        logging.info("🔑 OPENAI_API_KEY is present")

    async def playback_loop():
        """
        Take μ-law frames from playback_queue and drip them to Twilio
        every ~20ms so the caller hears smooth audio.
        """
        await asyncio.sleep(0.1)  # small delay so Twilio is fully ready
        nonlocal playback_running
        while playback_running:
            if playback_queue and stream_sid:
                frame_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": frame_b64},
                })
                await asyncio.sleep(0.02)  # ~20ms per 160-byte μ-law frame
            else:
                await asyncio.sleep(0.005)

    async def forward_twilio_to_openai(oai_ws):
        """
        1. Read incoming 'media' events from Twilio = caller's voice.
        2. Convert μ-law 8kHz -> PCM16 24kHz.
        3. Send audio to OpenAI as input_audio_buffer.append.
        """
        nonlocal stream_sid
        nonlocal openai_connected

        while True:
            try:
                msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Twilio websocket disconnected")
                break
            except Exception:
                logging.exception("💥 Error receiving from Twilio WebSocket")
                break

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                logging.warning(f"⚠ got non-JSON from Twilio: {msg}")
                continue

            event = data.get("event")

            if event == "connected":
                logging.info(f"ℹ Twilio event connected: {data}")

            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

            elif event == "media":
                # Caller audio chunk
                payload_b64 = data["media"]["payload"]
                ulaw_bytes = base64.b64decode(payload_b64)

                # μ-law -> PCM16 @8kHz
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                # upsample 8k -> 24k (OpenAI expects 24k PCM16)
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)

                # forward to OpenAI
                if oai_ws and openai_connected:
                    try:
                        b64_for_openai = base64.b64encode(pcm16_24k).decode("ascii")
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": b64_for_openai,
                        })
                    except Exception:
                        logging.exception("💥 Error sending audio to OpenAI")

            elif event == "stop":
                logging.info("📴 Twilio sent stop (caller hung up)")
                break

            else:
                logging.info(f"ℹ Twilio event {event}: {data}")

        logging.info("🚪 forward_twilio_to_openai exiting")

    async def forward_openai_to_twilio(oai_ws):
        """
        Read streamed events from OpenAI.
        On response.audio.delta -> push μ-law chunk to playback_queue.
        """
        nonlocal openai_connected
        try:
            async for raw in oai_ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")
                logging.info(f"🤖 OAI event: {oai_type}")

                openai_connected = True  # we heard from OpenAI

                if oai_type == "response.audio.delta":
                    ulaw_chunk_b64 = data.get("delta")
                    if ulaw_chunk_b64:
                        playback_queue.append(ulaw_chunk_b64)

                elif oai_type == "response.completed":
                    logging.info("✅ AI finished a spoken response")

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI internal error event: {data}")

        except Exception:
            logging.exception("💥 Error while reading from OpenAI ws (forward_openai_to_twilio)")
        logging.info("🚪 forward_openai_to_twilio exiting")

    # MAIN: connect to OpenAI Realtime
    if not OPENAI_API_KEY:
        logging.error("❌ OPENAI_API_KEY missing. Skipping OpenAI connect.")
        await forward_twilio_to_openai(None)
        logging.info("🔚 /media connection closed (no OpenAI)")
        return

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(
                f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as oai_ws:
                logging.info("✅ Connected to OpenAI Realtime successfully!")
                openai_connected = True
                oai_ws_handle = oai_ws

                # Tell OpenAI how to act on this call
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "g711_ulaw",
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "silence_duration_ms": 300,
                            "create_response": True
                        },
                        "instructions": (
                            "You are the live phone receptionist for Riteway Landscape Supply.\n\n"
                            "TONE:\n"
                            "- Speak warm, professional, confident, and efficient.\n"
                            "- Sound like a real dispatcher at a busy landscape yard.\n"
                            "- Be friendly and helpful, but also direct and firm.\n"
                            "- Keep each answer under 30 seconds.\n\n"
                            "BUSINESS INFO:\n"
                            "- Business: Riteway Landscape Supply.\n"
                            "- We sell bulk landscape material by the cubic yard.\n"
                            "- We are open Monday–Friday, 9 AM to 5 PM. No after-hours or weekend scheduling.\n"
                            "- We mainly serve Tooele Valley and surrounding areas.\n"
                            "- Most callers already know which product they want and how many yards.\n\n"
                            "PRICING (ALWAYS say 'per yard' or 'per ton'):\n"
                            "ROCK (price per yard unless noted):\n"
                            "- Washed Pea Gravel: $42 per yard.\n"
                            "- Desert Sun 7/8\" Crushed Rock: $40 per yard.\n"
                            "- 7/8\" Crushed Rock: $25 per yard.\n"
                            "- Desert Sun 1.5\" Crushed Rock: $40 per yard.\n"
                            "- 1.5\" Crushed Rock: $25 per yard.\n"
                            "- Commercial Road Base: $20 per yard.\n"
                            "- 3/8\" Minus Fines: $12 per yard.\n"
                            "- Desert Sun 1–3\" Cobble: $40 per yard.\n"
                            "- 8\" Landscape Cobble: $40 per yard.\n"
                            "- Desert Sun Boulders: $75 per ton (not per yard).\n\n"
                            "DIRT / SOIL / SAND (price per yard):\n"
                            "- Fill Dirt: $12 per yard.\n"
                            "- Top Soil: $26 per yard.\n"
                            "- Screened Premium Top Soil: $40 per yard.\n"
                            "- Washed Sand: $65 per yard.\n\n"
                            "BARK / MULCH (price per yard):\n"
                            "- Premium Mulch: $44 per yard.\n"
                            "- Colored Shredded Bark: $76 per yard.\n\n"
                            "DELIVERY RULES:\n"
                            "- We can haul up to 16 yards per load. If they need more than 16 yards, that's multiple loads.\n"
                            "- Delivery pricing inside Tooele Valley:\n"
                            "  - $75 delivery fee to Grantsville.\n"
                            "  - $115 delivery fee to the rest of Tooele Valley (Tooele, Stansbury, etc.).\n"
                            "- For deliveries OUTSIDE Tooele Valley (for example Magna):\n"
                            "  1. Ask for the full delivery address.\n"
                            "  2. Repeat the address back and confirm it's correct.\n"
                            "  3. Tell them: 'We charge $7 per mile from our yard in Grantsville, Utah.'\n"
                            "  4. Say: 'We'll confirm the final total when dispatch calls you back.'\n\n"
                            "BOOKING / TAKING AN ORDER:\n"
                            "If they want to schedule a delivery or place an order, gather:\n"
                            "1. Material they want.\n"
                            "2. How many yards they want (remind them: up to 16 yards per load).\n"
                            "3. Delivery address.\n"
                            "4. When they want it delivered.\n"
                            "5. Their name and callback number.\n"
                            "After you get that info, tell them: 'We'll confirm timing and reach back out to lock this in.'\n\n"
                            "COVERAGE / YARDAGE ESTIMATES:\n"
                            "- One cubic yard of material covers roughly 100 square feet at about 3 inches deep. Use that as a rule of thumb if they ask.\n"
                            "- You can also help them estimate yards needed from dimensions.\n"
                            "- To estimate yards needed:\n"
                            "  1. Ask for length in feet.\n"
                            "  2. Ask for width in feet.\n"
                            "  3. Ask for depth in inches.\n"
                            "  4. Compute: yards_needed = (length_ft * width_ft * depth_inches / 12) / 27.\n"
                            "     - Explanation: length * width = square feet. depth_inches/12 = depth in feet.\n"
                            "       Multiply to get cubic feet, then divide by 27 to get cubic yards.\n"
                            "  5. Round to one decimal place and tell them: 'You need about X yards. We can haul up to 16 yards per load.'\n"
                            "  6. After estimating, offer delivery.\n\n"
                            "POLICY / WHAT NOT TO DO:\n"
                            "- Do NOT promise exact arrival times. Say: 'We'll confirm the delivery window with dispatch.'\n"
                            "- Do NOT offer after-hours or weekend service. We are Monday–Friday, 9 AM to 5 PM.\n"
                            "- Do NOT invent products or prices.\n"
                            "- If you're not sure, or it sounds unusual, collect their info and say someone will call them back.\n\n"
                            "CALL FLOW / HOW YOU TALK:\n"
                            "- First line when the call starts should be: 'Hi, this is Riteway Landscape Supply. How can I help you today?'\n"
                            "- Answer pricing questions clearly using 'per yard' or 'per ton'.\n"
                            "- If they sound ready to order, move immediately into collecting delivery details.\n"
                            "- Stay calm, helpful, and decisive.\n"
                        )
                    }
                })

                # First spoken line the caller hears (our greeting, warm + professional)
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "instructions": "Hi, this is Riteway Landscape Supply. How can I help you today?"
                    }
                })

                # Start the playback loop to drip audio back to Twilio
                playback_task = asyncio.create_task(playback_loop())

                # Run both pipelines (Twilio->OpenAI and OpenAI->Twilio) at the same time
                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws_handle),
                    forward_openai_to_twilio(oai_ws_handle),
                )

        except Exception:
            logging.exception(
                "❌ Failed to connect to OpenAI Realtime! "
                "Possible causes: invalid OPENAI_API_KEY or missing model access."
            )
            # If OpenAI dies, still drain Twilio events so call ends clean
            await forward_twilio_to_openai(None)

    # Cleanup playback loop
    playback_running = False
    if playback_task:
        await asyncio.sleep(0.1)
        playback_task.cancel()

    logging.info("🔚 /media connection closed")

# local dev entrypoint (Render normally calls uvicorn directly)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
