import os, math, json, base64, asyncio, logging, audioop
import numpy as np
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, JSONResponse

###############################################################################
# CONFIG
###############################################################################
PUBLIC_BASE_URL = "https://riteway-ai-agent.onrender.com"
WS_MEDIA_URL = "wss://" + PUBLIC_BASE_URL.replace("https://", "") + "/media"
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
async def health(): return JSONResponse({"ok": True})

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """TwiML to start audio streaming"""
    logging.info("☎ Twilio hit /voice")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Connecting you to the Riteway assistant.</Say>
  <Connect><Stream url="{WS_MEDIA_URL}" /></Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media")
    stream_sid = None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers
        ) as oai_ws:
            # initialize OpenAI realtime session
            await oai_ws.send_json({
                "type": "session.update",
                "session": {
                    "modalities": ["audio"],
                    "input_audio_format": {"type": "pcm16", "sample_rate_hz": 24000},
                    "output_audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "voice": "alloy",
                    "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                       "silence_duration_ms": 200, "create_response": True},
                    "instructions": "You are the Riteway answering assistant. Be concise and professional."
                }
            })
            await oai_ws.send_json({
                "type": "response.create",
                "response": {"instructions": "Hi, this is Riteway! How can I help you today?"}
            })

            async def forward_twilio_to_openai():
                """caller audio → OpenAI"""
                nonlocal stream_sid
                while True:
                    try:
                        msg = await ws.receive_text()
                        data = json.loads(msg)
                        event = data.get("event")
                        if event == "start":
                            stream_sid = data["start"]["streamSid"]
                            logging.info(f"streamSid={stream_sid}")
                        elif event == "media":
                            payload = data["media"]["payload"]
                            ulaw = base64.b64decode(payload)
                            pcm16 = audioop.ulaw2lin(ulaw, 2)
                            pcm24, _ = audioop.ratecv(pcm16, 2, 1, 8000, 24000, None)
                            b64 = base64.b64encode(pcm24).decode("ascii")
                            await oai_ws.send_json({"type": "input_audio_buffer.append", "audio": b64})
                        elif event == "stop":
                            break
                    except WebSocketDisconnect:
                        break
                    except Exception:
                        logging.exception("Error forwarding audio")

            async def forward_openai_to_twilio():
                """AI audio → Twilio"""
                async for msg in oai_ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if data.get("type") == "response.audio.delta":
                        chunk = data.get("delta")
                        if chunk and stream_sid:
                            await ws.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": chunk}
                            })

            await asyncio.gather(forward_twilio_to_openai(), forward_openai_to_twilio())

    logging.info("🔚 media closed")
