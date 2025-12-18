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

# CONFIG
PUBLIC_BASE_URL = "https://riteway-ai-agent.onrender.com"
WS_MEDIA_URL = "wss://riteway-ai-agent.onrender.com/media"
STREAM_STATUS_URL = "https://riteway-ai-agent.onrender.com/stream-status"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

# Twilio Media Streams use g711 μ‑law at 8 kHz.  OpenAI can accept it directly.
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
    """Log Twilio stream status events for debugging."""
    body = await req.body()
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        data = {"raw": body.decode("utf-8", errors="ignore")}
    logging.info(f"📡 Twilio Stream Status Callback: {data}")
    return JSONResponse({"ok": True})

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """
    When a call comes in, Twilio hits this endpoint.
    We respond with TwiML telling Twilio to stream audio to /media and post status.
    The small greeting <Say> is optional; remove it later if not needed.
    """
    logging.info("☎ Twilio hit /voice")
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
# /media  (Twilio <-> us <-> OpenAI Realtime)
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    """
    Handle the Twilio Media Stream WebSocket.
    Forward incoming μ‑law audio to OpenAI and send the AI’s audio back to Twilio.
    """
    # Accept the WebSocket and honour any subprotocol if Twilio asked for one.
    requested_proto = ws.headers.get("sec-websocket-protocol")
    if requested_proto:
        subprotocol = [p.strip() for p in requested_proto.split(",") if p.strip()][0]
        try:
            await ws.accept(subprotocol=subprotocol)
            logging.info(f"✅ Twilio connected (accepted subprotocol={subprotocol})")
        except TypeError:
            # If FastAPI/Starlette doesn’t support the argument, just accept
            await ws.accept()
            logging.info("✅ Twilio connected (without explicit subprotocol)")
    else:
        await ws.accept()
        logging.info("✅ Twilio connected (no subprotocol requested)")

    if not OPENAI_API_KEY:
        logging.error("❌ OPENAI_API_KEY not set")
        await ws.close()
        return

    # Track Twilio’s stream ID and queue of AI audio to send back
    stream_sid = None
    playback_queue = deque()

    # Flags to manage call state
    playback_running = True
    ai_busy = False
    greeted = False
    call_active = False
    received_audio = False

    # Buffer control
    caller_bytes_since_commit = 0
    last_media_ts = 0.0
    MIN_BYTES_FOR_COMMIT = 2000   # require ≈250ms of audio
    SILENCE_GAP_SECONDS = 0.45

    stop_evt = asyncio.Event()

    # Events to manage OpenAI session ready/greeting state
    session_updated_evt = asyncio.Event()
    first_audio_delta_evt = asyncio.Event()

    async def safe_send_twilio(obj: dict):
        """Safely send JSON back to Twilio (ignore failures)."""
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    async def cancel_ai(oai_ws):
        """Cancel any active AI response and clear the playback queue."""
        nonlocal ai_busy
        playback_queue.clear()
        ai_busy = False
        try:
            await oai_ws.send_json({"type": "response.cancel"})
        except Exception:
            pass

    async def playback_loop():
        """
        Drip the AI’s μ‑law audio back to Twilio in ~20ms chunks.
        """
        await asyncio.sleep(0.05)  # small startup delay
        while playback_running and not stop_evt.is_set():
            if playback_queue and stream_sid:
                chunk = playback_queue.popleft()
                await safe_send_twilio({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk},
                })
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    async def send_greeting(oai_ws):
        """
        Send a single greeting after session.update completes.
        Use the 'text' field to generate both text and audio.
        """
        nonlocal greeted, ai_busy
        if greeted or stop_evt.is_set():
            return
        await session_updated_evt.wait()
        greeted = True
        ai_busy = True
        logging.info("👋 Sending greeting to OpenAI")
        await oai_ws.send_json({
            "type": "response.create",
            "response": {
                "text": "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
            }
        })

    async def forward_twilio_to_openai(oai_ws):
        """
        Read WebSocket messages from Twilio and forward audio to OpenAI.
        Also detect start/stop events and handle barge‑ins.
        """
        nonlocal stream_sid, call_active, received_audio
        nonlocal caller_bytes_since_commit, last_media_ts, ai_busy

        while not stop_evt.is_set():
            try:
                raw_msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Twilio disconnected from /media")
                break
            except Exception:
                logging.exception("💥 Error reading from Twilio WS")
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
                if not call_active:
                    continue
                received_audio = True
                last_media_ts = time.time()
                ulaw_b64 = data["media"]["payload"]

                # If AI is talking and user speaks, cancel AI’s response (barge‑in)
                if ai_busy:
                    await cancel_ai(oai_ws)

                try:
                    ulaw_bytes = base64.b64decode(ulaw_b64)
                    caller_bytes_since_commit += len(ulaw_bytes)
                except Exception:
                    pass

                try:
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": ulaw_b64,
                    })
                except Exception:
                    logging.exception("⚠ Error sending audio to OpenAI")

            elif event == "stop":
                logging.info("📴 Twilio stop: caller hung up")
                call_active = False
                stop_evt.set()
                await cancel_ai(oai_ws)
                break

        logging.info("🚪 forward_twilio_to_openai exiting")

    async def forward_openai_to_twilio(oai_ws):
        """
        Listen for events from OpenAI and queue AI audio to send back.
        """
        nonlocal ai_busy, playback_running

        try:
            async for raw in oai_ws:
                if stop_evt.is_set():
                    break
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")

                # Log important events for debugging
                if oai_type not in ("rate_limits.updated",):
                    logging.info(f"🤖 OAI event: {oai_type}")

                if oai_type == "session.updated":
                    session_updated_evt.set()

                if oai_type == "response.audio.delta":
                    # We have AI audio to play
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
        """
        Decide when to commit the caller’s audio and ask the AI to respond.
        """
        nonlocal caller_bytes_since_commit, ai_busy

        while not stop_evt.is_set():
            await asyncio.sleep(0.1)

            if not call_active or not received_audio:
                continue

            # Wait for a short pause (user stops talking)
            if last_media_ts and (time.time() - last_media_ts) < SILENCE_GAP_SECONDS:
                continue

            # Ensure we have at least ~250ms of audio before committing
            if caller_bytes_since_commit < MIN_BYTES_FOR_COMMIT:
                continue

            # Only start a new response if the AI isn’t talking
            if ai_busy:
                continue

            try:
                ai_busy = True
                logging.info(f"🗣 Committing {caller_bytes_since_commit} μ‑law bytes")
                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                # Start the AI response (no modalities field: session settings apply)
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {}
                })
            except Exception:
                logging.exception("⚠ Error on commit/response.create")
                ai_busy = False
            finally:
                caller_bytes_since_commit = 0

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

                # Start OpenAI listener right away
                oai_listener = asyncio.create_task(forward_openai_to_twilio(oai_ws))

                # Configure the session once at start
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
                            "Tone: warm, professional, firm. Short direct answers. If caller interrupts, stop.\n"
                            "Hours: Mon–Fri 9–5. No after hours.\n"
                            "We sell bulk by the yard; boulders by the ton.\n"
                            "Pricing:\n"
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
                            "Delivery: max 16 yards per load. $75 Grantsville; $115 rest of Tooele Valley.\n"
                            "Outside valley: ask address, repeat it, say $7 per mile from Grantsville yard; confirm final total.\n"
                            "Coverage: 1 yard ~100 sq ft at 3 inches. Formula yards=(L*W*(D/12))/27, round to one decimal.\n"
                            "Do NOT talk about nurseries, succulents, flowers or unrelated items.\n"
                        )
                    }
                })

                # Wait up to 3 seconds for session.updated; proceed anyway if missing
                try:
                    await asyncio.wait_for(session_updated_evt.wait(), timeout=3.0)
                    logging.info("✅ OpenAI session.updated received")
                except asyncio.TimeoutError:
                    logging.warning("⚠ Did not receive session.updated; continuing anyway")

                # Start the playback and other loops
                playback_task = asyncio.create_task(playback_loop())
                twilio_task = asyncio.create_task(forward_twilio_to_openai(oai_ws))
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                # Wait until Twilio reading loops finish (call ends)
                await asyncio.gather(twilio_task, driver_task)

                # Stop everything and cancel tasks
                stop_evt.set()
                playback_task.cancel()
                oai_listener.cancel()

        except Exception:
            logging.exception("❌ Failed connecting to OpenAI Realtime. Check API key or model access.")

    # Close Twilio WebSocket
    try:
        await ws.close()
    except Exception:
        pass

    logging.info("🔚 /media connection closed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
