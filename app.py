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

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with aiohttp.ClientSession() as session:
        # 1) Connect to OpenAI Realtime
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
        ) as oai_ws:

            # 2) Tell OpenAI what formats we use (CRITICAL):
            # - Twilio sends 8kHz G.711 μ-law (mulaw) to us
            # - We want OpenAI to SEND BACK the same (so Twilio can play it)
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

            # Optional: have the AI greet immediately so you know output works
            await oai_ws.send_json({"type": "response.create",
                                    "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}})

            async def twilio_to_openai():
                async for message in ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "media":
                        # Twilio gives us base64 mulaw @ 8k
                        audio_b64 = data["media"]["payload"]
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                            "audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                        })
                        # OPTIONAL: Commit frequently so the AI responds quickly
                        await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                        await oai_ws.send_json({"type": "response.create", "response": {"instructions": ""}})

                    elif event == "stop":
                        # End of stream (usually hangup)
                        try:
                            await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                            await oai_ws.send_json({"type": "response.create", "response": {"instructions": ""}})
                        except Exception:
                            pass
                        break

            async def openai_to_twilio():
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        typ = payload.get("type", "")

                        # OpenAI streams audio chunks as base64 when output format is g711_ulaw
                        if typ in ("output_audio.delta", "output_audio_buffer.delta"):
                            chunk_b64 = payload.get("audio")
                            if chunk_b64:
                                await ws.send_json({
                                    "event": "media",
                                    "media": {"payload": chunk_b64}
                                })

                        # You can also watch for errors to log
                        if typ == "error":
                            print("OpenAI error:", payload)

            try:
                await asyncio.gather(twilio_to_openai(), openai_to_twilio())
            except WebSocketDisconnect:
                print("❌ Twilio websocket disconnected")
            finally:
                await oai_ws.close()


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
