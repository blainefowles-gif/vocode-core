# app.py — Riteway AI Voice Agent (Twilio <-> OpenAI Realtime)
# Requirements:
#   - Render env var: OPENAI_API_KEY
#   - (optional) REALTIME_MODEL (default: gpt-4o-realtime-preview)
#   - (optional) FORCE_PCM_CONVERT="1" to force PCM->μ-law conversion if you still hear silence
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
FORCE_PCM_CONVERT = os.getenv("FORCE_PCM_CONVERT", "0") == "1"

app = FastAPI(title="Riteway AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1) Twilio webhook: return TwiML that opens a BIDIRECTIONAL stream to /media ---
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
        "OpenAI-Beta": "realtime=v1",  # enables Realtime WS
    }

    async with aiohttp.ClientSession() as session:
        # Connect to OpenAI Realtime WS
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
            timeout=0,
        ) as oai_ws:
            # Configure audio I/O formats ONCE (CRITICAL)
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
                    # Twilio uses 8k μ-law (G.711)
                    "input_audio_format":  {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "output_audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "voice": "alloy"
                }
            })

            # Initial greeting (kept, per your request)
            await oai_ws.send_json({
                "type": "response.create",
                "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
            })

            # ---------- Twilio -> OpenAI (batch ~100ms before commit) ----------
            async def twilio_to_openai():
                # Twilio typically sends ~20ms per "media" frame; gather 5 frames ≈ 100ms
                frames_since_commit = 0

                async for message in ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "media":
                        # Base64 μ-law 8k from Twilio
                        audio_b64 = data["media"]["payload"]

                        # Append ONLY (do not include 'audio_format' here)
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64
                        })
                        frames_since_commit += 1

                        # Commit after ~100ms of audio
                        if frames_since_commit >= 5:
                            await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                            await oai_ws.send_json({
                                "type": "response.create",
                                "response": {"instructions": ""}
                            })
                            frames_since_commit = 0

                    elif event == "stop":
                        # Flush any remaining audio
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

            # ---------- OpenAI -> Twilio (stream audio back; fallback converter available) ----------
            def to_mulaw_8k_b64(delta_b64: str) -> str:
                """
                Convert PCM16@24k (if that's what the model sent) to μ-law@8k.
                If FORCE_PCM_CONVERT is False, just passthrough (expects μ-law@8k already).
                Toggle FORCE_PCM_CONVERT=1 in Render env if you still hear silence.
                """
                if not FORCE_PCM_CONVERT:
                    # Assume it's already μ-law@8k (per session.update)
                    return delta_b64

                # Forced conversion path: treat incoming as PCM16@24k mono
                try:
                    pcm_bytes = base64.b64decode(delta_b64)  # 16-bit PCM little-endian @ 24k
                    # Downsample 24k -> 8k
                    pcm16_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, 24000, 8000, None)
                    # Convert linear PCM16 -> μ-law
                    mulaw_8k = audioop.lin2ulaw(pcm16_8k, 2)
                    return base64.b64encode(mulaw_8k).decode("ascii")
                except Exception as e:
                    print("Fallback conversion failed:", e)
                    # As a last resort, pass through
                    return delta_b64

            async def openai_to_twilio():
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        typ = payload.get("type", "")
                        print("OAI event:", typ)

                        # Realtime audio stream from OpenAI
                        if typ == "response.audio.delta":
                            chunk_b64 = payload.get("delta")
                            if chunk_b64:
                                safe_chunk_b64 = to_mulaw_8k_b64(chunk_b64)
                                await ws.send_json({
                                    "event": "media",
                                    "media": {"payload": safe_chunk_b64}
                                })

                        elif typ == "response.audio.done":
                            # End of assistant turn
                            pass

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

# Health check
@app.get("/health")
def health():
    return {"ok": True, "realtime_model": REALTIME_MODEL, "force_pcm_convert": FORCE_PCM_CONVERT}
