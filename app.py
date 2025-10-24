# app.py — Riteway AI Voice Agent (Twilio <-> OpenAI Realtime)
# Super-simple FastAPI app that:
#  1) Returns TwiML with <Connect><Stream> (bidirectional audio)
#  2) Bridges Twilio Media Streams <-> OpenAI Realtime via two websockets
#
# Requirements:
#  - Environment variable OPENAI_API_KEY set in Render
#  - Start command: uvicorn app:app --host 0.0.0.0 --port $PORT

import os
import json
import asyncio
import base64

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import aiohttp

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")  # default

app = FastAPI(title="Riteway AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) Twilio hits this when a call arrives.
#    We return TwiML that tells Twilio to open a *bidirectional* websocket to /media
#    so we can both receive and send audio on the same call.
@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""
    return twiml

# 2) Twilio connects here with a websocket. We then open a websocket to OpenAI Realtime.
#    We forward audio chunks from Twilio -> OpenAI, and AI audio from OpenAI -> Twilio.
@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    print("✅ Twilio connected to /media")

    if not OPENAI_API_KEY:
        await ws.close()
        return

    # Open a websocket to OpenAI Realtime
    # (The Realtime API is full-duplex audio over WS.)
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    # Build an initial "session" config with a short system prompt for Riteway
    # and request that the model returns audio (μ-law/PCMU 8k) compatible with Twilio.
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["audio"],
            "instructions": (
                "You are the phone receptionist for Riteway Landscape Materials in Grantsville, Utah. "
                "Be brief, friendly, and professional. Collect: caller name, callback number, address, "
                "material, quantity (yards/tons), and preferred delivery window. If unsure on pricing, "
                "give a range and say you will text a confirmed quote."
            ),
            # Twilio Media Streams uses 8kHz mulaw (audio/pcmu). Ask Realtime to output that.
            "voice": "alloy",
            "audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
        },
    }

    # Connect to OpenAI Realtime over websocket
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
            timeout=0,
        ) as oai_ws:

            # Send initial session settings (instructions + audio format)
            await oai_ws.send_json(session_update)

            # Helper: forward audio from Twilio -> OpenAI
            async def twilio_to_openai():
                async for message in ws.iter_text():
                    data = json.loads(message)

                    event = data.get("event")
                    if event == "start":
                        # Twilio gives us a streamSid we can use for marks if needed
                        pass

                    elif event == "media":
                        # Twilio sends base64 audio payload (PCMU 8k)
                        audio_b64 = data["media"]["payload"]
                        # Tell OpenAI we're appending audio to the input buffer
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,  # keep as base64
                            "audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                        })

                    elif event == "mark":
                        # Optional: barge-in timing controls could go here
                        pass

                    elif event == "stop":
                        # Commit and ask for a response (speak back)
                        await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                        await oai_ws.send_json({"type": "response.create", "response": {"instructions": ""}})

            # Helper: forward audio from OpenAI -> Twilio
            async def openai_to_twilio():
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)

                        # OpenAI will stream audio chunks; look for deltas
                        # Some SDKs send "output_audio.delta" or "output_audio_buffer.delta"
                        typ = payload.get("type", "")
                        if typ in ("output_audio.delta", "output_audio_buffer.delta"):
                            # payload["audio"] is base64-encoded μ-law when we request g711_ulaw
                            audio_chunk_b64 = payload.get("audio")
                            if audio_chunk_b64:
                                await ws.send_json({
                                    "event": "media",
                                    "media": {"payload": audio_chunk_b64}
                                })

                        # When the model signals the end of its turn, ask Twilio for more input naturally.
                        if typ in ("response.completed", "output_audio_buffer.commit"):
                            # No-op; Twilio keeps streaming caller audio automatically.
                            pass

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break

            try:
                await asyncio.gather(twilio_to_openai(), openai_to_twilio())
            except WebSocketDisconnect:
                print("❌ Twilio websocket disconnected")
            finally:
                await oai_ws.close()

# Optional quick check
@app.get("/health")
def health():
    return {"ok": True, "realtime_model": REALTIME_MODEL}
