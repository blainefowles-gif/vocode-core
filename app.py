import os
import json
import asyncio
import base64
import logging
import time
from collections import deque

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

###############################################################################
# CONFIG
###############################################################################

PUBLIC_BASE_URL = "https://riteway-ai-agent.onrender.com"
WS_MEDIA_URL = "wss://" + PUBLIC_BASE_URL.replace("https://", "").replace("http://", "") + "/media"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

# Twilio Media Streams uses g711 u-law @ 8kHz.
# We'll set OpenAI input/output to match and pass-through base64 frames.
OAI_AUDIO_FORMAT = "g711_ulaw"

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

@app.get("/")
async def root():
    return JSONResponse({"ok": True, "message": "Riteway AI Agent running. Twilio should POST /voice."})

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "model": REALTIME_MODEL,
        "has_api_key": bool(OPENAI_API_KEY)
    })

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    logging.info("☎ Twilio hit /voice")
    # Keep a small pause so caller doesn't miss the first word.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Pause length="1"/>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

###############################################################################
# /media  (Twilio <-> us <-> OpenAI Realtime)
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY set")
        await ws.close()
        return

    # Twilio streamSid (set on "start")
    stream_sid = None

    # AI->Twilio audio queue (base64 g711_ulaw chunks)
    playback_queue = deque()

    # State flags
    playback_running = True
    ai_busy = False
    greeted = False

    # Call lifecycle flags
    call_active = False
    received_any_media = False

    # Audio timing / commit gating
    caller_ulaw_bytes_since_commit = 0
    last_media_ts = 0.0

    # g711_ulaw @ 8kHz = 8000 bytes/sec
    # Require >= 250ms to avoid edge cases (OpenAI wants >=100ms, but we over-shoot).
    MIN_BYTES_FOR_COMMIT = 2000  # 0.25s * 8000 = 2000 bytes
    SILENCE_GAP_SECONDS = 0.45   # require a real pause before committing

    stop_evt = asyncio.Event()

    async def safe_send_twilio(obj: dict):
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    async def cancel_ai(oai_ws):
        nonlocal ai_busy
        playback_queue.clear()
        ai_busy = False
        try:
            await oai_ws.send_json({"type": "response.cancel"})
        except Exception:
            pass

    async def playback_loop():
        """
        Drip audio back to Twilio.
        IMPORTANT: We buffer even before stream_sid exists.
        As soon as stream_sid is set, we flush queued audio.
        """
        await asyncio.sleep(0.05)
        while playback_running and not stop_evt.is_set():
            if playback_queue and stream_sid:
                chunk_b64 = playback_queue.popleft()
                await safe_send_twilio({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk_b64},
                })
                # 20ms pacing
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    async def request_greeting(oai_ws):
        """
        Request the exact greeting ONCE.
        NOTE: We do this ASAP after OpenAI session.update.
        Audio will buffer into playback_queue until streamSid exists.
        """
        nonlocal greeted, ai_busy
        if greeted or stop_evt.is_set():
            return
        greeted = True
        ai_busy = True
        logging.info("👋 Requesting greeting from OpenAI NOW (buffering until streamSid exists)")
        await oai_ws.send_json({
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": "Say EXACTLY: Hello, thanks for calling Riteway Landscape Products! How can I help you?"
            }
        })

    async def forward_twilio_to_openai(oai_ws):
        nonlocal stream_sid, call_active, received_any_media
        nonlocal caller_ulaw_bytes_since_commit, last_media_ts, ai_busy

        while not stop_evt.is_set():
            try:
                raw_msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Twilio disconnected")
                stop_evt.set()
                break
            except Exception:
                logging.exception("💥 Error receiving from Twilio WS")
                stop_evt.set()
                break

            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                call_active = True
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

            elif event == "media":
                if not call_active:
                    continue

                received_any_media = True
                last_media_ts = time.time()

                ulaw_b64 = data["media"]["payload"]

                # If AI is talking and caller speaks -> barge-in cancel
                if ai_busy:
                    logging.info("🛑 BARGE-IN: caller audio while AI talking -> cancel AI")
                    await cancel_ai(oai_ws)

                # Count bytes for commit gating
                try:
                    ulaw_bytes = base64.b64decode(ulaw_b64)
                    caller_ulaw_bytes_since_commit += len(ulaw_bytes)
                except Exception:
                    pass

                # Send caller audio to OpenAI
                try:
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": ulaw_b64,
                    })
                except Exception:
                    logging.exception("⚠ error sending audio chunk to OpenAI")

            elif event == "stop":
                logging.info("📴 Twilio sent stop (caller hung up)")
                call_active = False
                stop_evt.set()
                try:
                    await cancel_ai(oai_ws)
                except Exception:
                    pass
                break

        logging.info("🚪 forward_twilio_to_openai exiting")

    async def forward_openai_to_twilio(oai_ws):
        nonlocal ai_busy, playback_running

        try:
            async for raw in oai_ws:
                if stop_evt.is_set():
                    break
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")

                if oai_type in ("session.created", "rate_limits.updated"):
                    continue

                if oai_type == "response.audio.delta":
                    delta_b64 = data.get("delta")
                    if delta_b64:
                        playback_queue.append(delta_b64)
                        # Helpful log to confirm we ARE receiving audio from OpenAI
                        logging.info(f"🔊 OpenAI audio.delta received (queue={len(playback_queue)})")

                elif oai_type == "response.done":
                    ai_busy = False
                    logging.info("✅ OpenAI response.done (AI finished speaking)")

                elif oai_type == "response.interrupted":
                    ai_busy = False
                    playback_queue.clear()
                    logging.info("⛔ OpenAI response.interrupted (cleared playback)")

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI error event: {data}")
                    ai_busy = False

        except Exception:
            logging.exception("💥 Error in forward_openai_to_twilio loop")

        playback_running = False
        stop_evt.set()
        logging.info("🚪 forward_openai_to_twilio exiting")

    async def speech_drive_loop(oai_ws):
        """
        Commit only when:
        - call_active and received_any_media
        - enough audio bytes
        - caller has paused (silence gap)
        - AI is not already talking
        """
        nonlocal caller_ulaw_bytes_since_commit, ai_busy

        while not stop_evt.is_set():
            await asyncio.sleep(0.10)

            if not call_active or not received_any_media:
                continue

            if last_media_ts and (time.time() - last_media_ts) < SILENCE_GAP_SECONDS:
                continue

            if caller_ulaw_bytes_since_commit < MIN_BYTES_FOR_COMMIT:
                continue

            if ai_busy:
                continue

            try:
                ai_busy = True
                logging.info(
                    f"🗣 COMMIT {caller_ulaw_bytes_since_commit} ulaw-bytes (~{caller_ulaw_bytes_since_commit/8000:.2f}s)"
                )
                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {"modalities": ["audio", "text"]}
                })
            except Exception:
                logging.exception("⚠ commit/response.create failed")
                ai_busy = False
            finally:
                caller_ulaw_bytes_since_commit = 0

    ###############################################################################
    # CONNECT TO OPENAI REALTIME
    ###############################################################################
    async with aiohttp.ClientSession() as session_http:
        try:
            async with session_http.ws_connect(
                f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as oai_ws:
                logging.info("✅ Connected to OpenAI Realtime successfully!")

                # Configure the session FIRST
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "input_audio_format": OAI_AUDIO_FORMAT,
                        "output_audio_format": OAI_AUDIO_FORMAT,
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.45,
                            "silence_duration_ms": 250,
                            "create_response": False
                        },
                        "instructions": (
                            "You are Riteway Landscape Products' phone assistant for a bulk landscape materials yard.\n"
                            "\n"
                            "TONE:\n"
                            "- Warm, professional, and firm. Short direct answers.\n"
                            "- If the caller interrupts, STOP talking immediately.\n"
                            "\n"
                            "BUSINESS INFO:\n"
                            "- Name: Riteway Landscape Products.\n"
                            "- Hours: Monday–Friday, 9 AM–5 PM. No after-hours, no weekends.\n"
                            "- We sell bulk landscape materials by the cubic yard (and some boulders by the ton).\n"
                            "- We serve the Tooele Valley and surrounding areas.\n"
                            "\n"
                            "PRICING (always say 'per yard' or 'per ton'):\n"
                            "- Washed Pea Gravel: $42 per yard\n"
                            "- Desert Sun 7/8\" Crushed Rock: $40 per yard\n"
                            "- 7/8\" Crushed Rock: $25 per yard\n"
                            "- Desert Sun 1.5\" Crushed Rock: $40 per yard\n"
                            "- 1.5\" Crushed Rock: $25 per yard\n"
                            "- Commercial Road Base: $20 per yard\n"
                            "- 3/8\" Minus Fines: $12 per yard\n"
                            "- Desert Sun 1–3\" Cobble: $40 per yard\n"
                            "- 8\" Landscape Cobble: $40 per yard\n"
                            "- Desert Sun Boulders: $75 per ton\n"
                            "- Fill Dirt: $12 per yard\n"
                            "- Top Soil: $26 per yard\n"
                            "- Screened Premium Top Soil: $40 per yard\n"
                            "- Washed Sand: $65 per yard\n"
                            "- Premium Mulch: $44 per yard\n"
                            "- Colored Shredded Bark: $76 per yard\n"
                            "\n"
                            "DELIVERY:\n"
                            "- Max 16 yards per load.\n"
                            "- $75 delivery fee to Grantsville.\n"
                            "- $115 delivery fee to the rest of Tooele Valley.\n"
                            "- Outside Tooele Valley (ex: Magna): ask full address, repeat it back, and say:\n"
                            "  \"We charge seven dollars per mile from our yard in Grantsville, Utah. We’ll confirm the final total.\"\n"
                            "\n"
                            "ORDER TAKING:\n"
                            "Collect: material, yards, address, preferred day/time, name + callback.\n"
                            "Then say you’ll confirm dispatch and timing.\n"
                            "\n"
                            "COVERAGE + YARDAGE MATH:\n"
                            "- 1 cubic yard covers ~100 sq ft at ~3 inches deep.\n"
                            "- yards = (length_ft * width_ft * (depth_in/12)) / 27\n"
                            "- Round to 1 decimal place.\n"
                            "\n"
                            "IMPORTANT:\n"
                            "- Do NOT talk about nurseries, succulents, flowers, or unrelated items.\n"
                            "- Assume callers usually know what they want; ask short clarifying questions.\n"
                        )
                    }
                })

                # NOW request greeting immediately (buffered until streamSid arrives)
                await request_greeting(oai_ws)

                playback_task = asyncio.create_task(playback_loop())
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws),
                    forward_openai_to_twilio(oai_ws),
                )

                stop_evt.set()
                playback_task.cancel()
                driver_task.cancel()

        except Exception:
            logging.exception("❌ Failed talking to OpenAI realtime. Check API key/billing/model access.")

    try:
        await ws.close()
    except Exception:
        pass

    logging.info("🔚 /media connection closed")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
