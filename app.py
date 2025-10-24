# app.py — Riteway AI Voice Agent (Twilio <-> OpenAI Realtime)
# Env vars on Render:
#   OPENAI_API_KEY   (required)
#   REALTIME_MODEL   (optional, default: gpt-4o-realtime-preview)
#
# Render start command:
#   uvicorn app:app --host 0.0.0.0 --port $PORT

import os
import json
import base64
import audioop  # stdlib
import asyncio
import traceback

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse, JSONResponse
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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64: str) -> str:
    """Twilio -> OpenAI: μ-law@8k (b64) -> PCM16@24k (b64)."""
    ulaw_bytes = base64.b64decode(ulaw_b64)       # μ-law, 8k, mono
    pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)    # -> PCM16 8k
    pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)
    return base64.b64encode(pcm16_24k).decode("ascii")


# -----------------------------------------------------------------------------
# 1) Twilio webhook -> TwiML (MUST return valid XML with 200 OK)
# -----------------------------------------------------------------------------
@app.post("/voice")
async def voice_post(_: Request):
    try:
        print("📞 /voice POST hit by Twilio")
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""
        # IMPORTANT: Twilio expects XML content type
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        print("❌ /voice error:", e)
        traceback.print_exc()
        # Twilio treats non-200 as "Application Error"
        return Response(content="<Response></Response>", media_type="application/xml", status_code=200)

# Optional GET so you can open /voice in a browser for a quick check
@app.get("/voice")
async def voice_get():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# -----------------------------------------------------------------------------
# 2) Media WebSocket: Twilio <-> OpenAI Realtime bridge
# -----------------------------------------------------------------------------
@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    print("✅ Twilio connected to /media")

    if not OPENAI_API_KEY:
        print("❌ Missing OPENAI_API_KEY — closing socket")
        await ws.close()
        return

    twilio_stream_sid = None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
                headers=headers,
                autoping=True,
                heartbeat=20,
                timeout=0,
            ) as oai_ws:
                # Configure realtime session (server-side VAD = auto-commit + auto-response)
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio"],
                        "input_audio_format":  {"type": "pcm16", "sample_rate_hz": 24000},
                        "output_audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "silence_duration_ms": 200,
                            "create_response": True
                        },
                        "instructions": (
                            "You are the phone receptionist for Riteway Landscape Materials "
                            "in Grantsville, Utah. Be brief, friendly, and professional. "
                            "Collect: caller name, callback number, address, material, "
                            "quantity (yards/tons), and preferred delivery window. "
                            "If unsure on pricing, give a range and say you will text "
                            "a confirmed quote."
                        ),
                    }
                })

                # Initial greeting
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
                })

                async def twilio_to_openai():
                    nonlocal twilio_stream_sid
                    async for message in ws.iter_text():
                        data = json.loads(message)
                        event = data.get("event")

                        if event == "start":
                            twilio_stream_sid = data.get("start", {}).get("streamSid")
                            print("Twilio start: streamSid =", twilio_stream_sid)

                        elif event == "media":
                            try:
                                ulaw_b64 = data["media"]["payload"]
                                pcm24_b64 = ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64)
                                await oai_ws.send_json({
                                    "type": "input_audio_buffer.append",
                                    "audio": pcm24_b64
                                })
                            except Exception as e:
                                print("media convert/append error:", e)
                                traceback.print_exc()

                        elif event == "stop":
                            print("Twilio stop event")
                            break

                async def openai_to_twilio():
                    nonlocal twilio_stream_sid
                    async for msg in oai_ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(msg.data)
                            typ = payload.get("type", "")
                            print("OAI event:", typ)

                            if typ == "response.audio.delta":
                                # Already μ-law@8k as requested
                                chunk_b64 = payload.get("delta")
                                if chunk_b64 and twilio_stream_sid:
                                    await ws.send_json({
                                        "event": "media",
                                        "streamSid": twilio_stream_sid,
                                        "media": {"payload": chunk_b64}
                                    })

                            elif typ == "error":
                                print("OpenAI error:", payload)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print("OpenAI WS error frame")
                            break

                try:
                    await asyncio.gather(twilio_to_openai(), openai_to_twilio())
                except WebSocketDisconnect:
                    print("❌ Twilio websocket disconnected")
                finally:
                    await oai_ws.close()

    except Exception as e:
        # If anything goes wrong above, log it instead of crashing the process.
        print("❌ /media fatal error:", e)
        traceback.print_exc()
        try:
            await ws.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return JSONResponse({"ok": True, "realtime_model": REALTIME_MODEL})
