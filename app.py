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

# WebSocket endpoint for media streaming; notice we use the public URL
WS_MEDIA_URL = "wss://riteway-ai-agent.onrender.com/media"
# HTTP endpoint to receive status callbacks from Twilio <Stream>
STREAM_STATUS_URL = "https://riteway-ai-agent.onrender.com/stream-status"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

# Twilio Media Streams use μ-law (G.711) at 8 kHz
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
        "has_api_key": bool(OPENAI_API_KEY),
    })

@app.post("/stream-status")
async def stream_status(req: Request):
    """Log Twilio <Stream> status events for debugging."""
    body = await req.body()
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        data = {"raw": body.decode("utf-8", errors="ignore")}
    logging.info(f"📡 Twilio Stream Status Callback: {data}")
    return JSONResponse({"ok": True})

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """Twilio hits this URL first; we return TwiML to connect to /media WebSocket."""
    logging.info("☎ Twilio hit /voice")
    # We include a <Say> for debug so callers hear something while connecting.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">Hello, connecting you now.</Say>
  <Pause length="0.5"/>
  <Connect>
    <Stream
      url="{WS_MEDIA_URL}"
      track="inbound_track"
      statusCallback="{STREAM_STATUS_URL}"
      statusCallbackMethod="POST"
    />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

###############################################################################
# /media – Twilio ↔ our server ↔ OpenAI Realtime
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    """Handle the bidirectional audio conversation."""
    # Accept Twilio’s subprotocol if requested (needed to avoid 31924 errors)
    requested = ws.headers.get("sec-websocket-protocol")
    if requested:
        offered = [p.strip() for p in requested.split(",") if p.strip()]
        try:
            await ws.accept(subprotocol=offered[0])
            logging.info(f"✅ Twilio connected (accepted subprotocol={offered[0]})")
        except TypeError:
            await ws.accept()
            logging.info("✅ Twilio connected (server couldn’t accept subprotocol arg)")
    else:
        await ws.accept()
        logging.info("✅ Twilio connected (no subprotocol requested)")

    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY set")
        await ws.close()
        return

    # Per-call state
    stream_sid = None
    playback_queue = deque()
    playback_running = True
    ai_busy = False
    greeted = False

    call_active = False
    received_any_media = False
    caller_ulaw_bytes_since_commit = 0
    last_media_ts = 0.0

    # We require at least ~250 ms (~2000 bytes) of caller audio and a pause before committing
    MIN_BYTES_FOR_COMMIT = 2000
    SILENCE_GAP_SECONDS = 0.45

    stop_evt = asyncio.Event()
    session_updated_evt = asyncio.Event()
    first_audio_delta_evt = asyncio.Event()

    async def safe_send_twilio(obj: dict):
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    async def cancel_ai(oai_ws):
        """Cancel current AI response if caller speaks during AI speech (barge-in)."""
        nonlocal ai_busy
        playback_queue.clear()
        ai_busy = False
        try:
            await oai_ws.send_json({"type": "response.cancel"})
        except Exception:
            pass

    async def playback_loop():
        """Send AI audio back to Twilio as a drip of 20 ms μ-law frames."""
        await asyncio.sleep(0.05)
        while playback_running and not stop_evt.is_set():
            if playback_queue and stream_sid:
                chunk_b64 = playback_queue.popleft()
                await safe_send_twilio({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk_b64},
                })
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    async def send_greeting(oai_ws):
        """Send our fixed greeting once the session is ready."""
        nonlocal greeted, ai_busy
        if greeted or stop_evt.is_set():
            return

        # Wait for OpenAI session.updated so the model is ready
        await session_updated_evt.wait()

        greeted = True
        ai_busy = True
        logging.info("👋 Sending greeting to OpenAI")

        # Our greeting uses only ‘audio’ and ‘text’ modalities – no response.text!
        await oai_ws.send_json({
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": "Say EXACTLY: Hello, thanks for calling Riteway Landscape Products! How can I help you?"
            }
        })

        # Watchdog: if we see no response.audio.delta in 2 seconds, resend greeting with only audio
        async def watchdog():
            try:
                await asyncio.wait_for(first_audio_delta_evt.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                if stop_evt.is_set():
                    return
                logging.warning("⚠ Greeting didn’t produce audio; retrying with audio only.")
                try:
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio"],
                            "instructions": "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
                        }
                    })
                except Exception:
                    pass

        asyncio.create_task(watchdog())

    async def forward_twilio_to_openai(oai_ws):
        """Forward Twilio events and audio to OpenAI."""
        nonlocal stream_sid, call_active, received_any_media
        nonlocal caller_ulaw_bytes_since_commit, last_media_ts, ai_busy

        while not stop_evt.is_set():
            try:
                raw_msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Twilio disconnected from /media")
                break
            except Exception:
                logging.exception("💥 Error receiving from Twilio WS")
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
                await send_greeting(oai_ws)

            elif event == "media":
                # Ignore audio before call officially started
                if not call_active:
                    continue

                received_any_media = True
                last_media_ts = time.time()

                ulaw_b64 = data["media"]["payload"]

                # Barge-in cancel if AI is talking
                if ai_busy:
                    await cancel_ai(oai_ws)

                # Track bytes for commit gating
                try:
                    ulaw_bytes = base64.b64decode(ulaw_b64)
                    caller_ulaw_bytes_since_commit += len(ulaw_bytes)
                except Exception:
                    pass

                # Forward ulaw chunk to OpenAI
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
                await cancel_ai(oai_ws)
                break

        logging.info("🚪 forward_twilio_to_openai exiting")

    async def forward_openai_to_twilio(oai_ws):
        """Listen to OpenAI events and send audio deltas back to Twilio."""
        nonlocal ai_busy, playback_running

        try:
            async for raw in oai_ws:
                if stop_evt.is_set():
                    break
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")
                # Log most events – helps debug silence
                if oai_type not in ("rate_limits.updated",):
                    logging.info(f"🤖 OAI event: {oai_type}")

                if oai_type == "session.updated":
                    session_updated_evt.set()

                if oai_type == "response.audio.delta":
                    # We’ve received audio – mark first_audio_delta_evt
                    first_audio_delta_evt.set()
                    delta_b64 = data.get("delta")
                    if delta_b64:
                        playback_queue.append(delta_b64)

                elif oai_type == "response.done":
                    ai_busy = False

                elif oai_type == "response.interrupted":
                    ai_busy = False
                    playback_queue.clear()

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI error event: {data}")
                    ai_busy = False

        except Exception:
            logging.exception("💥 Error in forward_openai_to_twilio loop")

        playback_running = False
        stop_evt.set()
        logging.info("🚪 forward_openai_to_twilio exiting")

    async def speech_drive_loop(oai_ws):
        """Periodically check whether to commit caller audio and request a response."""
        nonlocal caller_ulaw_bytes_since_commit, ai_busy

        while not stop_evt.is_set():
            await asyncio.sleep(0.10)

            if not call_active or not received_any_media:
                continue

            # Wait for a pause in caller speech
            if last_media_ts and (time.time() - last_media_ts) < SILENCE_GAP_SECONDS:
                continue

            # Require enough audio so we don’t commit an empty buffer
            if caller_ulaw_bytes_since_commit < MIN_BYTES_FOR_COMMIT:
                continue

            # Don’t overlap AI responses
            if ai_busy:
                continue

            try:
                ai_busy = True
                logging.info(f"🗣 COMMIT {caller_ulaw_bytes_since_commit} ulaw bytes")
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

    ###########################################################################
    # CONNECT TO OPENAI REALTIME
    ###########################################################################
    async with aiohttp.ClientSession() as session_http:
        try:
            async with session_http.ws_connect(
                f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as oai_ws:
                logging.info("✅ Connected to OpenAI Realtime")

                # Start listening to OpenAI events
                oai_task = asyncio.create_task(forward_openai_to_twilio(oai_ws))

                # Initial session update: specify formats, voice, turn detection, instructions
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
                            "Tone: warm, professional, firm. Keep answers short and direct. If the caller interrupts, stop immediately.\n"
                            "Hours: Monday–Friday, 9 AM to 5 PM. We are closed after hours and weekends.\n"
                            "We sell bulk material by the cubic yard and some boulders by the ton; no nursery plants.\n"
                            "Pricing: Washed Pea Gravel $42/yd; Desert Sun 7/8\" Crushed Rock $40/yd; 7/8\" Crushed Rock $25/yd; Desert Sun 1.5\" Crushed Rock $40/yd; 1.5\" Crushed Rock $25/yd; Commercial Road Base $20/yd; 3/8\" Minus Fines $12/yd; Desert Sun 1–3\" Cobble $40/yd; 8\" Landscape Cobble $40/yd; Desert Sun Boulders $75/ton; Fill Dirt $12/yd; Top Soil $26/yd; Screened Premium Top Soil $40/yd; Washed Sand $65/yd; Premium Mulch $44/yd; Colored Shredded Bark $76/yd.\n"
                            "Delivery: Up to 16 yards per load. $75 delivery to Grantsville, $115 to the rest of Tooele Valley. Outside the valley (e.g. Magna) ask the full address, repeat it back, and explain we charge $7 per mile from Grantsville; dispatch will confirm the final total.\n"
                            "Coverage: One yard covers roughly 100 square feet at 3 inches deep. To compute yardage: yards = (length_ft × width_ft × (depth_in/12)) / 27, round to one decimal place.\n"
                            "If the caller wants to place an order, collect: (1) material, (2) yards, (3) delivery address, (4) preferred day/time, (5) their name and callback number, and tell them we’ll confirm the schedule.\n"
                            "Do not discuss flowers, succulents, nursery plants, or unrelated products."
                        )
                    }
                })

                # Wait a bit for session.updated; if it never comes, we’ll still proceed
                try:
                    await asyncio.wait_for(session_updated_evt.wait(), timeout=3.0)
                    logging.info("✅ OpenAI session.updated received")
                except asyncio.TimeoutError:
                    logging.warning("⚠ Did not receive session.updated within 3 seconds (continuing anyway)")

                # Launch concurrency tasks: playback, Twilio→OpenAI, commit driver
                playback_task = asyncio.create_task(playback_loop())
                twilio_task = asyncio.create_task(forward_twilio_to_openai(oai_ws))
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                # Wait for Twilio and driver tasks to finish
                await asyncio.gather(twilio_task, driver_task)

                # Clean up
                stop_evt.set()
                playback_task.cancel()
                oai_task.cancel()

        except Exception:
            logging.exception("❌ Failed talking to OpenAI realtime. Check API key/billing/model access.")

    # Always close Twilio WebSocket
    try:
        await ws.close()
    except Exception:
        pass

    logging.info("🔚 /media connection closed")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
