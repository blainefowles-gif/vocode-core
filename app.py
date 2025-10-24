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
FORCE_PCM_CONVERT = os.getenv("FORCE_PCM_CONVERT", "0") == "1"

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
            # Configure audio formats ONCE (Twilio uses μ-law 8k)
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

            # Initial greeting (kept per your request)
            await oai_ws.send_json({
                "type": "response.create",
                "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
            })

            # ------------- Twilio -> OpenAI (producer) -------------
            # We will *not* commit here. We only append frames and count them.
            frames_buffered = 0
            producer_running = True

            async def twilio_to_openai():
                nonlocal frames_buffered, producer_running
                async for message in ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "start":
                        # stream started
                        continue

                    if event == "media":
                        audio_b64 = data["media"]["payload"]
                        # Append ONLY (no audio_format field here)
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64
                        })
                        frames_buffered += 1
                        if frames_buffered % 3 == 0:
                            print(f"frames buffered: {frames_buffered}")

                    elif event == "stop":
                        # mark producer as done
                        producer_running = False
                        break

            # ------------- Timer task: commit every ~120ms if we have audio -------------
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

                # Final flush at the very end if anything is left
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

            # ------------- OpenAI -> Twilio (consumer) -------------
            def to_mulaw_8k_b64(delta_b64: str) -> str:
                """
                If FORCE_PCM_CONVERT=1, convert incoming PCM16@24k -> μ-law@8k.
                Otherwise, passthrough (assumes μ-law@8k already per session.update).
                """
                if not FORCE_PCM_CONVERT:
                    return delta_b64
                try:
                    pcm_bytes = base64.b64decode(delta_b64)  # assume PCM16@24k
                    pcm16_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, 24000, 8000, None)
                    mulaw_8k = audioop.lin2ulaw(pcm16_8k, 2)
                    return base64.b64encode(mulaw_8k).decode("ascii")
                except Exception as e:
                    print("Fallback conversion failed:", e)
                    return delta_b64

            async def openai_to_twilio():
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        typ = payload.get("type", "")
                        print("OAI event:", typ)

                        if typ == "response.audio.delta":
                            chunk_b64 = payload.get("delta")
                            if chunk_b64:
                                safe_chunk_b64 = to_mulaw_8k_b64(chunk_b64)
                                await ws.send_json({
                                    "event": "media",
                                    "media": {"payload": safe_chunk_b64}
                                })

                        elif typ in ("response.done", "response.completed", "response.audio.done"):
                            # end of an assistant turn
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
    return {"ok": True, "realtime_model": REALTIME_MODEL, "force_pcm_convert": FORCE_PCM_CONVERT}
