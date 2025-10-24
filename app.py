# app.py — Riteway AI Voice Agent (Twilio <-> OpenAI Realtime)
# Assumes:
#   - Env var OPENAI_API_KEY set in Render
#   - (optional) REALTIME_MODEL (default gpt-4o-realtime-preview)
# Start command (Render): uvicorn app:app --host 0.0.0.0 --port $PORT

import os
import json
import asyncio
import base64
import audioop  # stdlib
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

# --- 1) Twilio webhook: open a BIDIRECTIONAL stream to /media ---
@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""
    return twiml

# --- 2) Media WebSocket: Twilio <-> OpenAI Realtime bridge ---
@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    print("✅ Twilio connected to /media")

    if not OPENAI_API_KEY:
        print("❌ Missing OPENAI_API_KEY")
        await ws.close()
        return

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
            timeout=0,
        ) as oai_ws:
            # IMPORTANT: Tell OpenAI we will send PCM16 at 24 kHz, but we want μ-law 8 kHz back for Twilio.
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
                    # INPUT to OpenAI (what we send): PCM16 24kHz
                    "input_audio_format":  {"type": "pcm16", "sample_rate_hz": 24000},
                    # OUTPUT from OpenAI (what we need Twilio to play): μ-law 8kHz
                    "output_audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "voice": "alloy"
                }
            })

            # Initial greeting (kept per your request)
            await oai_ws.send_json({
                "type": "response.create",
                "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
            })

            # ---------- Twilio -> OpenAI ----------
            # We'll buffer frames and commit on a timer (≈120ms) so we never commit an empty buffer.
            frames_buffered = 0
            producer_running = True

            def ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64: str) -> str:
                """Convert Twilio μ-law@8k (base64) -> PCM16@24k (base64) for OpenAI input."""
                ulaw_bytes = base64.b64decode(ulaw_b64)          # μ-law 8k, 8-bit mono
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)       # -> PCM16 8k, 16-bit mono
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)  # upsample to 24k
                return base64.b64encode(pcm16_24k).decode("ascii")

            async def twilio_to_openai():
                nonlocal frames_buffered, producer_running
                async for message in ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "start":
                        # stream started
                        continue

                    if event == "media":
                        # Twilio sends μ-law@8k base64
                        ulaw_b64 = data["media"]["payload"]
                        # Convert to PCM16@24k for OpenAI
                        pcm24_b64 = ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64)

                        # Append ONLY (no audio_format here)
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": pcm24_b64
                        })
                        frames_buffered += 1
                        if frames_buffered % 3 == 0:
                            print(f"frames buffered: {frames_buffered}")

                    elif event == "stop":
                        producer_running = False
                        break

            async def commit_timer():
                nonlocal frames_buffered, producer_running
                try:
                    while producer_running:
                        await asyncio.sleep(0.12)  # ~120 ms
                        if frames_buffered > 0:
                            print(f"COMMIT ({frames_buffered} frames)")
                            await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                            await oai_ws.send_json({
                                "type": "response.create",
                                "response": {"instructions": ""}
                            })
                            frames_buffered = 0
                except Exception as e:
                    print("commit_timer error:", e)

                # final flush
                if frames_buffered > 0:
                    print(f"FINAL COMMIT ({frames_buffered} frames)")
                    try:
                        await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                        await oai_ws.send_json({
                            "type": "response.create",
                            "response": {"instructions": ""}
                        })
                    except Exception as e:
                        print("final flush error:", e)
                    frames_buffered = 0

            # ---------- OpenAI -> Twilio ----------
            async def openai_to_twilio():
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        typ = payload.get("type", "")
                        print("OAI event:", typ)

                        if typ == "response.audio.delta":
                            # Because we requested output g711_ulaw@8k, 'delta' should already be μ-law 8k base64.
                            chunk_b64 = payload.get("delta")
                            if chunk_b64:
                                await ws.send_json({
                                    "event": "media",
                                    "media": {"payload": chunk_b64}
                                })

                        elif typ in ("response.done", "response.completed", "response.audio.done"):
                            pass

                        elif typ == "error":
                            print("OpenAI error:", payload)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print("OpenAI WS error frame")
                        break

            try:
                await asyncio.gather(
                    twilio_to_openai(),
                    commit_timer(),
                    openai_to_twilio()
                )
            except WebSocketDisconnect:
                print("❌ Twilio websocket disconnected")
            finally:
                await oai_ws.close()

# Health check
@app.get("/health")
def health():
    return {"ok": True, "realtime_model": REALTIME_MODEL}
