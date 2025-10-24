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
    Twilio hits this first.
    Add a short pause before the AI connects to ensure OpenAI loads
    so the greeting isn't clipped.
    """
    logging.info("☎ Twilio hit /voice")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Pause length="1.5"/>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

###############################################################################
# WEBSOCKET MEDIA HANDLER
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None
    playback_queue = deque()
    playback_running = True
    playback_task = None

    async def playback_loop():
        """Send AI audio back to Twilio smoothly."""
        await asyncio.sleep(0.1)
        while playback_running:
            if playback_queue and stream_sid:
                frame_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": frame_b64}
                })
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    async def forward_twilio_to_openai(oai_ws):
        """Forward Twilio’s caller audio to OpenAI."""
        nonlocal stream_sid
        while True:
            try:
                msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Twilio disconnected")
                break

            data = json.loads(msg)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                logging.info(f"📞 streamSid={stream_sid}")

            elif event == "media":
                ulaw_bytes = base64.b64decode(data["media"]["payload"])
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)
                if oai_ws:
                    b64 = base64.b64encode(pcm16_24k).decode("ascii")
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": b64,
                    })

            elif event == "stop":
                logging.info("📴 Caller hung up")
                break

    async def forward_openai_to_twilio(oai_ws):
        """Send AI-generated audio to Twilio."""
        async for raw in oai_ws:
            if raw.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(raw.data)
            if data.get("type") == "response.audio.delta":
                delta = data.get("delta")
                if delta:
                    playback_queue.append(delta)

    if not OPENAI_API_KEY:
        logging.error("❌ Missing OPENAI_API_KEY")
        return

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as oai_ws:
            logging.info("✅ Connected to OpenAI Realtime")

            # Configure assistant personality + knowledge
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
                        "You are the phone receptionist for Riteway Landscape Products.\n\n"
                        "GREETING:\n"
                        "- Begin every call with: 'Hello, thanks for calling Riteway Landscape Products! How can I help you?'\n"
                        "- Start promptly, with a warm and professional tone.\n\n"
                        "TONE:\n"
                        "- Be friendly, confident, knowledgeable, and efficient.\n"
                        "- Keep answers short and professional.\n\n"
                        "BUSINESS INFO:\n"
                        "- Business: Riteway Landscape Products.\n"
                        "- We sell bulk landscape materials by the cubic yard.\n"
                        "- Open Monday–Friday, 9 AM to 5 PM. No after-hours or weekend scheduling.\n"
                        "- Serve Tooele Valley and nearby areas.\n\n"
                        "PRICING (say 'per yard' or 'per ton'):\n"
                        "- Washed Pea Gravel: $42 per yard.\n"
                        "- Desert Sun 7/8\" Crushed Rock: $40 per yard.\n"
                        "- 7/8\" Crushed Rock: $25 per yard.\n"
                        "- Desert Sun 1.5\" Crushed Rock: $40 per yard.\n"
                        "- 1.5\" Crushed Rock: $25 per yard.\n"
                        "- Commercial Road Base: $20 per yard.\n"
                        "- 3/8\" Minus Fines: $12 per yard.\n"
                        "- Desert Sun 1–3\" Cobble: $40 per yard.\n"
                        "- 8\" Landscape Cobble: $40 per yard.\n"
                        "- Desert Sun Boulders: $75 per ton.\n"
                        "- Fill Dirt: $12 per yard.\n"
                        "- Top Soil: $26 per yard.\n"
                        "- Screened Premium Top Soil: $40 per yard.\n"
                        "- Washed Sand: $65 per yard.\n"
                        "- Premium Mulch: $44 per yard.\n"
                        "- Colored Shredded Bark: $76 per yard.\n\n"
                        "DELIVERY INFO:\n"
                        "- Haul up to 16 yards per load.\n"
                        "- $75 delivery fee to Grantsville.\n"
                        "- $115 delivery fee to Tooele or Stansbury.\n"
                        "- For outside areas (like Magna): ask for address, confirm, then quote $7 per mile from Grantsville.\n\n"
                        "YARDAGE ESTIMATES:\n"
                        "- 1 cubic yard covers about 100 sq ft at 3 inches deep.\n"
                        "- To calculate: yards = (length_ft * width_ft * depth_in / 12) / 27.\n"
                        "- Round to one decimal and say 'You’ll need about X yards. We can haul up to 16 per load.'\n\n"
                        "REMINDERS:\n"
                        "- Always mention 'per yard' or 'per ton'.\n"
                        "- Never promise exact delivery times.\n"
                        "- No weekend or after-hours service.\n"
                    )
                }
            })

            # Greeting
            await oai_ws.send_json({
                "type": "response.create",
                "response": {
                    "instructions": (
                        "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
                    )
                }
            })

            playback_task = asyncio.create_task(playback_loop())
            await asyncio.gather(
                forward_twilio_to_openai(oai_ws),
                forward_openai_to_twilio(oai_ws),
            )

    playback_running = False
    if playback_task:
        playback_task.cancel()
    logging.info("🔚 /media connection closed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
