# app.py — Riteway AI Voice Agent (Twilio <-> OpenAI Realtime)
# Required env var on Render:
#   OPENAI_API_KEY
# Optional:
#   REALTIME_MODEL (default: gpt-4o-realtime-preview)
# Start command on Render:
#   uvicorn app:app --host 0.0.0.0 --port $PORT

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
    return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""

# ----------------------- helpers: audio conversions (μ-law<->PCM) -----------------

def ulaw8k_b64_to_pcm16_24k(ulaw_b64: str) -> bytes:
    """
    Twilio -> OpenAI: μ-law@8k (b64) -> PCM16@24k (raw bytes).
    - μ-law is 8-bit, 8000 samples/sec mono
    - We convert to 16-bit PCM mono at 24000 Hz (what OpenAI likes)
    """
    ulaw_bytes = base64.b64decode(ulaw_b64)         # μ-law 8k mono
    pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)      # -> PCM16 8k mono (16-bit)
    pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)  # -> 24k mono
    return pcm16_24k  # raw bytes (not b64)

def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

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

    # Capture Twilio streamSid from "start"
    twilio_stream_sid = None

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
            timeout=0,
        ) as oai_ws:

            # Configure formats:
            #  - We SEND PCM16@24k to OpenAI
            #  - We want μ-law@8k BACK (so Twilio can play it)
            await oai_ws.send_json({
                "type": "session.update",
                "session": {
                    "modalities": ["audio"],
                    "input_audio_format":  {"type": "pcm16", "sample_rate_hz": 24000},
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

            # Greeting so you hear something even before caller speaks
            await oai_ws.send_json({
                "type": "response.create",
                "response": {"instructions": "Hi! This is Riteway. How can I help you today?"}
            })

            # ---------------- state for *byte-accurate* batching ----------------
            # OpenAI minimum: >= 100ms of audio per commit.
            # At PCM16@24k mono: 24000 samples/sec * 2 bytes/sample = 48000 bytes/sec
            # 100ms = 0.1 sec -> 4800 bytes. We'll use a SAFE threshold = 150ms -> 7200 bytes.
            BYTES_PER_SEC_PCM16_24K = 24000 * 2  # 48000
            MIN_COMMIT_BYTES = int(0.150 * BYTES_PER_SEC_PCM16_24K)  # 7200 bytes (~150ms)
            bytes_since_last_commit = 0
            producer_running = True

            async def twilio_to_openai():
                """
                Append caller audio to OpenAI buffer as PCM16@24k.
                Commit ONLY when we've accumulated >= MIN_COMMIT_BYTES.
                """
                nonlocal twilio_stream_sid, bytes_since_last_commit, producer_running

                async for message in ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "start":
                        twilio_stream_sid = data.get("start", {}).get("streamSid")
                        print("Twilio start: streamSid =", twilio_stream_sid)

                    elif event == "media":
                        try:
                            ulaw_b64 = data["media"]["payload"]
                            pcm24_bytes = ulaw8k_b64_to_pcm16_24k(ulaw_b64)

                            # Append (base64) to OpenAI buffer
                            await oai_ws.send_json({
                                "type": "input_audio_buffer.append",
                                "audio": bytes_to_b64(pcm24_bytes)
                            })

                            bytes_since_last_commit += len(pcm24_bytes)
                            if bytes_since_last_commit >= MIN_COMMIT_BYTES:
                                ms = int(1000 * bytes_since_last_commit / BYTES_PER_SEC_PCM16_24K)
                                print(f"COMMIT by bytes: {bytes_since_last_commit} bytes (~{ms} ms)")
                                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                                await oai_ws.send_json({
                                    "type": "response.create",
                                    "response": {"instructions": ""}
                                })
                                bytes_since_last_commit = 0

                        except Exception as e:
                            print("media convert/append error:", e)

                    elif event == "stop":
                        # Final flush ONLY if we truly have >=100ms worth (by bytes)
                        if bytes_since_last_commit >= MIN_COMMIT_BYTES:
                            ms = int(1000 * bytes_since_last_commit / BYTES_PER_SEC_PCM16_24K)
                            print(f"FINAL COMMIT by bytes: {bytes_since_last_commit} bytes (~{ms} ms)")
                            try:
                                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                                await oai_ws.send_json({
                                    "type": "response.create",
                                    "response": {"instructions": ""}
                                })
                            except Exception as e:
                                print("final commit error:", e)
                        else:
                            ms = int(1000 * bytes_since_last_commit / BYTES_PER_SEC_PCM16_24K)
                            print(f"Stop with only {bytes_since_last_commit} bytes (~{ms} ms); skipping final commit.")
                        producer_running = False
                        break

            async def openai_to_twilio():
                """
                Relay OpenAI audio back to Twilio.
                We requested g711_ulaw@8k, so 'delta' is already μ-law@8k base64.
                Always include streamSid — without it, Twilio can be silent.
                """
                nonlocal twilio_stream_sid
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        typ = payload.get("type", "")
                        print("OAI event:", typ)

                        if typ == "response.audio.delta":
                            chunk_b64 = payload.get("delta")
                            if chunk_b64 and twilio_stream_sid:
                                await ws.send_json({
                                    "event": "media",
                                    "streamSid": twilio_stream_sid,
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
