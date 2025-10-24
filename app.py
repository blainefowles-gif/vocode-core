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

def generate_beep_ulaw_chunks(duration_sec=1.0, freq_hz=440.0, sample_rate=8000):
    """
    Make a short 'beep' in μ-law sliced into 20ms frames and base64 encode.
    We use this as a sanity check / filler so Twilio doesn't think we're dead.
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
    logging.info(f"🔊 sending {len(chunks_b64)} fallback beep frames to Twilio (sid={stream_sid})")
    for frame_b64 in chunks_b64:
        await ws.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": frame_b64
            }
        })
        await asyncio.sleep(0.02)
    logging.info("🔊 finished sending fallback beep frames")

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
    Twilio calls this first. We respond with TwiML telling Twilio:
    1. Say something to the caller
    2. Start streaming the live call audio to /media over WebSocket
    """
    logging.info("☎ Twilio hit /voice")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Connecting you to the Riteway assistant.</Say>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

@app.websocket("/media")
async def media(ws: WebSocket):
    """
    This is the live audio bridge during a phone call.

    Twilio <-> (this server) <-> OpenAI Realtime
    """
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None
    openai_connected = False
    oai_ws_handle = None

    # Queue of μ-law frames from OpenAI that we drip back to Twilio.
    playback_queue = deque()
    playback_running = True
    playback_task = None

    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY in environment.")
    else:
        logging.info("🔑 OPENAI_API_KEY is present")

    async def play_fallback_beep_once():
        """
        Play a short beep to prove Twilio can hear *something*
        if OpenAI isn't ready yet. Helps avoid silence.
        """
        if not stream_sid:
            return
        chunks = generate_beep_ulaw_chunks(duration_sec=1.0, freq_hz=440.0)
        await send_ulaw_chunks_to_twilio(ws, stream_sid, chunks)

    async def playback_loop():
        """
        Pull μ-law audio frames from playback_queue
        and send them to Twilio every ~20ms.
        """
        await asyncio.sleep(0.1)  # tiny delay so Twilio is fully ready
        nonlocal playback_running
        while playback_running:
            if playback_queue and stream_sid:
                frame_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": frame_b64},
                })
                await asyncio.sleep(0.02)  # ~20ms pacing
            else:
                await asyncio.sleep(0.005)

    async def forward_twilio_to_openai(oai_ws):
        """
        1. Receive caller audio from Twilio ("media" events).
        2. Convert Twilio μ-law 8k -> PCM16 24k.
        3. Send that audio into OpenAI Realtime using input_audio_buffer.append.
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

                # If OpenAI isn't ready yet, play the beep so caller hears *something*
                if not openai_connected:
                    logging.warning("⚠ OpenAI not connected yet. Playing fallback beep.")
                    await play_fallback_beep_once()

            elif event == "media":
                # Caller audio chunk from Twilio
                payload_b64 = data["media"]["payload"]
                ulaw_bytes = base64.b64decode(payload_b64)

                # μ-law 8kHz -> PCM16 16-bit
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                # resample 8k -> 24k because we feed 24k PCM16 to OpenAI
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)

                # send to OpenAI if connected
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
        Read events from OpenAI.
        For each 'response.audio.delta', we get μ-law audio and queue it
        for playback_loop() to drip to Twilio.
        """
        nonlocal openai_connected
        try:
            async for raw in oai_ws:
                # We mostly care about TEXT frames from OpenAI WS
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")
                logging.info(f"🤖 OAI event: {oai_type}")

                openai_connected = True  # we heard from OpenAI

                if oai_type == "response.audio.delta":
                    # base64-encoded μ-law audio chunk from OpenAI
                    audio_chunk_b64 = data.get("delta")
                    if audio_chunk_b64:
                        playback_queue.append(audio_chunk_b64)

                elif oai_type == "response.completed":
                    logging.info("✅ AI finished a spoken response")

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI internal error event: {data}")

        except Exception:
            logging.exception("💥 Error while reading from OpenAI ws (forward_openai_to_twilio)")
        logging.info("🚪 forward_openai_to_twilio exiting")

    # MAIN CONNECT TO OPENAI REALTIME
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

                # Tell OpenAI how to behave in this call
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
                            "You are Riteway Landscape Supply's live phone assistant.\n\n"
                            "ABOUT THE BUSINESS:\n"
                            "- Business name: Riteway Landscape Supply.\n"
                            "- We supply bulk landscape material (rock, dirt, bark, sand) by the yard.\n"
                            "- We are open Monday through Friday, 9 AM to 5 PM. No after-hours or weekend scheduling.\n"
                            "- We primarily serve the Tooele Valley and surrounding areas.\n"
                            "- Most callers already know what product they want and how many yards.\n\n"
                            "PRODUCTS AND PRICING (price per yard unless noted):\n"
                            "ROCK:\n"
                            "- Washed Pea Gravel – $42.00 per yard\n"
                            "- Desert Sun 7/8\" Crushed Rock – $40.00 per yard\n"
                            "- 7/8\" Crushed Rock – $25.00 per yard\n"
                            "- Desert Sun 1.5\" Crushed Rock – $40.00 per yard\n"
                            "- 1.5\" Crushed Rock – $25.00 per yard\n"
                            "- Commercial Road Base – $20.00 per yard\n"
                            "- 3/8\" Minus Fines – $12.00 per yard\n"
                            "- Desert Sun 1–3\" Cobble – $40.00 per yard\n"
                            "- 8\" Landscape Cobble – $40.00 per yard\n"
                            "- Desert Sun Boulders – $75.00 per ton (NOT per yard)\n\n"
                            "DIRT / SOIL / SAND:\n"
                            "- Fill Dirt – $12.00 per yard\n"
                            "- Top Soil – $26.00 per yard\n"
                            "- Screened Premium Top Soil – $40.00 per yard\n"
                            "- Washed Sand – $65.00 per yard\n\n"
                            "BARK / MULCH:\n"
                            "- Premium Mulch – $44.00 per yard\n"
                            "- Colored Shredded Bark – $76.00 per yard\n\n"
                            "DELIVERY / TRUCK INFO:\n"
                            "- We can haul up to 16 yards per load. If someone needs more than 16 yards, that is multiple loads.\n"
                            "- Delivery pricing inside Tooele Valley:\n"
                            "  - $75 delivery fee to Grantsville.\n"
                            "  - $115 delivery fee to the rest of Tooele Valley (Tooele, Stansbury, etc.).\n"
                            "- For deliveries OUTSIDE Tooele Valley (for example Magna):\n"
                            "  1. Ask for the full delivery address.\n"
                            "  2. Repeat the address back and confirm it's correct.\n"
                            "  3. Tell them: 'We charge $7 per mile from our yard in Grantsville, Utah.'\n"
                            "  4. Say: 'We'll confirm the exact total when dispatch calls you back.'\n\n"
                            "HOW TO BOOK A DELIVERY:\n"
                            "If the caller wants to place an order or schedule a delivery, do this:\n"
                            "1. Ask what material they want.\n"
                            "2. Ask how many yards they want (remind them we haul up to 16 yards per load).\n"
                            "3. Ask for the full delivery address.\n"
                            "4. Ask when they would like it delivered.\n"
                            "5. Ask for their name and callback number.\n"
                            "6. If they are in Tooele Valley, tell them the delivery fee ($75 Grantsville, $115 rest of Tooele Valley).\n"
                            "7. If they are outside Tooele Valley, explain the $7/mile from Grantsville and say we'll confirm final pricing.\n"
                            "8. Tell them: 'We'll confirm timing and reach back out to lock this in.'\n\n"
                            "WHAT NOT TO DO:\n"
                            "- Do NOT promise an exact delivery time window. Say: 'We'll confirm timing with dispatch.'\n"
                            "- Do NOT offer after-hours or weekend service. We are Monday–Friday 9 AM to 5 PM.\n"
                            "- Do NOT invent products or prices that are not listed above.\n"
                            "- If you're not sure or they ask something unusual, take their info and say someone will call them back.\n\n"
                            "TONE / STYLE:\n"
                            "- Be fast, polite, confident, and friendly.\n"
                            "- Keep each response short, under 30 seconds.\n"
                            "- Most callers just want price and delivery. Answer directly.\n"
                            "- If they are ready to schedule, collect details so we can actually deliver.\n"
                        )
                    }
                })

                # Send initial spoken greeting
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "instructions": "Hi, this is Riteway Landscape Supply. How can I help you today?"
                    }
                })

                # Start the audio-out pacing loop
                playback_task = asyncio.create_task(playback_loop())

                # Run both pipelines at once:
                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws_handle),
                    forward_openai_to_twilio(oai_ws_handle),
                )

        except Exception:
            logging.exception(
                "❌ Failed to connect to OpenAI Realtime! "
                "Common causes: bad/expired OPENAI_API_KEY or model access."
            )
            # if OpenAI failed completely, still drain Twilio so call ends cleanly
            await forward_twilio_to_openai(None)

    # Cleanup playback loop
    playback_running = False
    if playback_task:
        await asyncio.sleep(0.1)
        playback_task.cancel()

    logging.info("🔚 /media connection closed")

# local dev entrypoint (Render uses uvicorn directly so this is mostly backup)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
