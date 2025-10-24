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
# AUDIO UTILS (for fallback beep)
###############################################################################

def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
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
    logging.info(f"🔊 sending {len(chunks_b64)} fallback beep frames to Twilio (sid={stream_sid})")
    for frame_b64 in chunks_b64:
        await ws.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": frame_b64
            }
        })
        # pace ~20ms per frame
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
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None
    openai_connected = False
    oai_ws_handle = None

    # queue of audio chunks from OpenAI waiting to send to Twilio
    playback_queue = deque()
    playback_task = None
    playback_running = True

    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY in environment.")
    else:
        logging.info("🔑 OPENAI_API_KEY is present")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY if OPENAI_API_KEY else ''}",
        "OpenAI-Beta": "realtime=v1"
    }

    async def play_fallback_beep_once():
        if not stream_sid:
            return
        chunks = generate_beep_ulaw_chunks(duration_sec=1.0, freq_hz=440.0)
        await send_ulaw_chunks_to_twilio(ws, stream_sid, chunks)

    async def playback_loop():
        """
        Take audio chunks from playback_queue and drip them to Twilio
        every ~20ms so Twilio hears clear speech instead of static.
        """
        nonlocal playback_running
        while playback_running:
            if playback_queue and stream_sid:
                frame_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": frame_b64},
                })
                # pace it so Twilio plays smoothly
                await asyncio.sleep(0.02)
            else:
                # nothing to play right now
                await asyncio.sleep(0.005)

    async def forward_twilio_to_openai(oai_ws):
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
                # Twilio sometimes sends this first now
                logging.info(f"ℹ Twilio event connected: {data}")

            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

                if not openai_connected:
                    logging.warning("⚠ OpenAI not connected yet. Playing fallback beep.")
                    await play_fallback_beep_once()

            elif event == "media":
                payload_b64 = data["media"]["payload"]
                ulaw_bytes = base64.b64decode(payload_b64)

                # Twilio -> PCM16 8kHz
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                # Resample 8kHz -> 24kHz
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)

                if oai_ws and openai_connected:
                    b64_for_openai = base64.b64encode(pcm16_24k).decode("ascii")
                    try:
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
        Push audio chunks into playback_queue.
        playback_loop() will drip them back to Twilio in real time.
        """
        nonlocal openai_connected
        try:
            async for raw in oai_ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(raw.data)
                oai_type = data.get("type")
                logging.info(f"🤖 OAI event: {oai_type}")

                openai_connected = True

                if oai_type == "response.audio.delta":
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

    # main OpenAI connect
    if not OPENAI_API_KEY:
        logging.error("❌ OPENAI_API_KEY missing. Skipping OpenAI connect.")
        # even without OpenAI we still run forward_twilio_to_openai to keep call alive
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

                # FIX 1: update modalities to include "text" too
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "input_audio_format": {
                            "type": "pcm16",
                            "sample_rate_hz": 24000
                        },
                        "output_audio_format": {
                            "type": "g711_ulaw",
                            "sample_rate_hz": 8000
                        },
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "silence_duration_ms": 300,
                            "create_response": True
                        },
                        "instructions": (
                            "You are the Riteway answering assistant for incoming business calls. "
                            "You greet the caller, gather what they need, take messages, "
                            "and be polite and professional. Keep answers short and clear."
                        )
                    }
                })

                # Kick off greeting
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "instructions": "Hi, this is Riteway. How can I help you today?"
                    }
                })

                # start playback loop in background
                playback_task = asyncio.create_task(playback_loop())

                # run Twilio->OpenAI and OpenAI->Twilio in parallel
                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws_handle),
                    forward_openai_to_twilio(oai_ws_handle),
                )

        except Exception:
            logging.exception(
                "❌ Failed to connect to OpenAI Realtime! "
                "Most common causes: (1) bad/expired OPENAI_API_KEY, "
                "(2) account not allowed to use gpt-4o-realtime-preview yet."
            )
            # still let Twilio talk a little / hear beep
            await forward_twilio_to_openai(None)

    # clean up playback loop
    playback_running = False
    if playback_task:
        await asyncio.sleep(0.1)
        playback_task.cancel()

    logging.info("🔚 /media connection closed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
