# app.py — Riteway AI Voice Agent (Twilio <-> OpenAI Realtime)
# Required:
#   OPENAI_API_KEY (Render env var)
# Optional:
#   REALTIME_MODEL (default: gpt-4o-realtime-preview)
# Start command (Render): uvicorn app:app --host 0.0.0.0 --port $PORT

import os
import json
import base64
import audioop  # stdlib
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import asyncio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

app = FastAPI(title="Riteway AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- 1) Twilio webhook -> TwiML ---------------------------

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    # Bidirectional media stream to /media
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""
    return twiml

# ----------------------- helper: audio conversions (μ-law<->PCM) ------------------

def ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64: str) -> str:
    """Twilio → OpenAI: μ-law@8k (base64) -> PCM16@24k (base64)."""
    ulaw_bytes = base64.b64decode(ulaw_b64)     # μ-law 8k mono
    pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)  # -> PCM16 8k mono (16-bit)
    pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)  # -> PCM16 24k mono
    return base64.b64encode(pcm16_24k).decode("ascii")

# --------------------- 2) Media WebSocket: Twilio <-> OpenAI bridge ----------------

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

    # Will capture Twilio streamSid from "start"
    twilio_stream_sid = None

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
            timeout=0,
        ) as oai_ws:

            # Tell OpenAI what we send in and what we want back
            await oai_ws.send_json({
                "type": "session.update",
                "session": {
                    "modalities": ["audio"],
                    # We SEND PCM16@24k to OpenAI
                    "input_audio_format":  {"type": "pcm16", "sample_rate_hz": 24000},
                    # We want μ-law@8k BACK (so Twilio can play it)
                    "output_audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "voice": "alloy",
                    "instructions": (
                        "You are the phone receptionist for Riteway Landscape Materials in Grantsville, Utah. "
                        "Be brief, friendly, and professional. Collect: caller name, callback number, address, "
                        "material, quantity (yards/tons), and preferred delivery window. If unsure on pricing, "
                        "give a range and say you will text a confirmed quote."
                    ),
                }
            })

            # Initial greeting so you hear something right away
            await oai_ws.send_json({
                "type": "response.create",
                "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
            })

            # -------- state for batching ----------
            frames_since_last_commit = 0      # how many Twilio frames are appended since last commit
            MIN_FRAMES_PER_COMMIT = 8         # ~8 x 20ms ≈ 160ms (safely > 100ms)
            producer_running = True

            # -------------- Twilio -> OpenAI (append only, event-driven commit) --------------
            async def twilio_to_openai():
                nonlocal twilio_stream_sid, frames_since_last_commit, producer_running
                async for message in ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "start":
                        twilio_stream_sid = data.get("start", {}).get("streamSid")
                        print("Twilio start: streamSid =", twilio_stream_sid)

                    elif event == "media":
                        # Convert Twilio μ-law@8k -> PCM16@24k and append to OpenAI buffer
                        try:
                            ulaw_b64 = data["media"]["payload"]
                            pcm24_b64 = ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64)

                            await oai_ws.send_json({
                                "type": "input_audio_buffer.append",
                                "audio": pcm24_b64
                            })
                            frames_since_last_commit += 1
                            if frames_since_last_commit % 4 == 0:
                                print(f"frames appended since last commit: {frames_since_last_commit}")

                            # Only COMMIT when we have enough audio (>= ~160ms)
                            if frames_since_last_commit >= MIN_FRAMES_PER_COMMIT:
                                print(f"COMMIT ({frames_since_last_commit} frames)  # >=160ms")
                                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                                await oai_ws.send_json({
                                    "type": "response.create",
                                    "response": {"instructions": ""}  # continue the turn
                                })
                                frames_since_last_commit = 0

                        except Exception as e:
                            print("media convert/append error:", e)

                    elif event == "stop":
                        # Flush ONLY if we have >=100ms worth (avoid empty-commit error)
                        if frames_since_last_commit >= MIN_FRAMES_PER_COMMIT:
                            print(f"FINAL COMMIT ({frames_since_last_commit} frames)  # >=160ms")
                            try:
                                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                                await oai_ws.send_json({
                                    "type": "response.create",
                                    "response": {"instructions": ""}
                                })
                            except Exception as e:
                                print("final commit error:", e)
                        else:
                            print(f"Call stopping with {frames_since_last_commit} frames (<{MIN_FRAMES_PER_COMMIT}); skipping final commit.")
                        producer_running = False
                        break

            # -------------------- OpenAI -> Twilio (stream μ-law audio back) -------------------
            async def openai_to_twilio():
                nonlocal twilio_stream_sid
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        typ = payload.get("type", "")
                        print("OAI event:", typ)

                        if typ == "response.audio.delta":
                            # Because we requested output g711_ulaw@8k, 'delta' is μ-law@8k base64.
                            chunk_b64 = payload.get("delta")
                            if chunk_b64 and twilio_stream_sid:
                                await ws.send_json({
                                    "event": "media",
                                    "streamSid": twilio_stream_sid,   # include streamSid for Twilio playback
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
                await asyncio.gather(twilio_to_openai(), openai_to_twilio())
            except WebSocketDisconnect:
                print("❌ Twilio websocket disconnected")
            finally:
                await oai_ws.close()

# ------------------------------ Health check ---------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "realtime_model": REALTIME_MODEL}
