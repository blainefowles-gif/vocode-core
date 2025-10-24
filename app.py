# app.py — Riteway AI Voice Agent (Twilio <-> OpenAI Realtime)
# Works with Twilio <Connect><Stream> (bidirectional) and OpenAI Realtime.
# IMPORTANT:
#  - Render env var: OPENAI_API_KEY must be set.
#  - Start command on Render: uvicorn app:app --host 0.0.0.0 --port $PORT
#
# What this does:
#  1) /voice returns TwiML that tells Twilio to open a WSS stream to /media
#  2) /media bridges Twilio audio <-> OpenAI Realtime
#  3) Audio formats are set ONCE in session.update (no audio_format on append)

import os
import json
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import aiohttp

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

app = FastAPI(title="Riteway AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) Twilio hits this on inbound call.
#    We instruct it to open a BIDIRECTIONAL websocket to /media.
@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""
    return twiml

# 2) Twilio connects here with a websocket. We bridge to OpenAI Realtime over another websocket.
@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    print("✅ Twilio connected to /media")

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY missing")
        await ws.close()
        return

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with aiohttp.ClientSession() as session:
        # Connect to OpenAI Realtime API (WebSocket)
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
            timeout=0,
        ) as oai_ws:

            # Tell OpenAI the audio formats ONCE (input & output = μ-law 8k)
            await oai_ws.send_json({
                "type": "session.update",
                "session": {
                    "modalities": ["audio"],
                    "instructions": (
                        "You are the phone receptionist for Riteway Landscape Materials in Grantsville, Utah. "
                        "Be brief, friendly, and professional. Collect: caller name, callback number, address, "
                        "material, quantity (yards/tons), and preferred delivery window. If unsure on pricing, "
                        "give a range and say you will text a confirmed quote."
                    ),
                    "input_audio_format":  {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "output_audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "voice": "alloy"
                }
            })

            # Optional: greet immediately so we know output works
            await oai_ws.send_json({
                "type": "response.create",
                "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
            })

            # ---------- Twilio -> OpenAI ----------
            # ---------- Twilio -> OpenAI (batch frames before commit + keep greeting) ----------
async def twilio_to_openai():
    # Send initial greeting immediately after connection
    await oai_ws.send_json({
        "type": "response.create",
        "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
    })

    # Twilio usually sends 20ms per frame, so wait ~5 frames before committing (≈100ms)
    frames_since_commit = 0

    async for message in ws.iter_text():
        data = json.loads(message)
        event = data.get("event")

        if event == "media":
            audio_b64 = data["media"]["payload"]

            # Append audio to OpenAI’s buffer (no audio_format here!)
            await oai_ws.send_json({
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            })

            frames_since_commit += 1

            # Only commit once ~100ms of audio has been received
            if frames_since_commit >= 5:
                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {"instructions": ""}
                })
                frames_since_commit = 0

        elif event == "stop":
            # If call ended but there’s uncommitted audio, flush it
            if frames_since_commit > 0:
                try:
                    await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {"instructions": ""}
                    })
                except Exception as e:
                    print("Flush error:", e)
            break


            # ---------- OpenAI -> Twilio ----------
            async def openai_to_twilio():
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        typ = payload.get("type", "")

                        # Current streamed audio event from Realtime
                        if typ == "response.audio.delta":
                            # 'delta' contains base64 μ-law 8k audio (because of output_audio_format)
                            chunk_b64 = payload.get("delta")
                            if chunk_b64:
                                await ws.send_json({
                                    "event": "media",
                                    "media": {"payload": chunk_b64}
                                })

                        elif typ == "error":
                            print("OpenAI error:", payload)

            try:
                await asyncio.gather(twilio_to_openai(), openai_to_twilio())
            except WebSocketDisconnect:
                print("❌ Twilio websocket disconnected")
            finally:
                await oai_ws.close()

# Simple health check
@app.get("/health")
def health():
    return {"ok": True, "realtime_model": REALTIME_MODEL}
