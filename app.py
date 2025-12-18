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

WS_MEDIA_URL = "wss://riteway-ai-agent.onrender.com/media"
STREAM_STATUS_URL = "https://riteway-ai-agent.onrender.com/stream-status"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

# Twilio Media Streams = G.711 u-law @ 8kHz
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
    """
    Twilio will POST here with stream events + errors.
    """
    body = await req.body()
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        data = {"raw": body.decode("utf-8", errors="ignore")}
    logging.info(f"📡 Twilio Stream Status Callback: {data}")
    return JSONResponse({"ok": True})

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    logging.info("☎ Twilio hit /voice")

    # Debug Say so we KNOW Twilio is executing the TwiML.
    # You can remove later after everything is stable.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">Hello, connecting you now.</Say>
  <Pause length="0.4"/>
  <Connect>
    <Stream
      url="{WS_MEDIA_URL}"
      statusCallback="{STREAM_STATUS_URL}"
      statusCallbackMethod="POST"
    />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

###############################################################################
# /media
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    """
    Twilio Media Streams sometimes includes Sec-WebSocket-Protocol.
    IMPORTANT: Only accept a subprotocol if the client requested one.
    """
    requested = ws.headers.get("sec-websocket-protocol")
    if requested:
        offered = [p.strip() for p in requested.split(",") if p.strip()]
        # Prefer "audio" if offered, else use first offered
        chosen = "audio" if "audio" in offered else offered[0]
        try:
            await ws.accept(subprotocol=chosen)
            logging.info(f"✅ Twilio connected to /media (accepted subprotocol={chosen})")
        except TypeError:
            await ws.accept()
            logging.info("✅ Twilio connected to /media (server can't accept subprotocol arg)")
    else:
        await ws.accept()
        logging.info("✅ Twilio connected to /media (no subprotocol requested)")

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

    # Commit gating
    caller_ulaw_bytes_since_commit = 0
    last_media_ts = 0.0

    # 8k ulaw = 8000 bytes/sec
    # Require >= 250ms to avoid edge cases (OpenAI needs >=100ms)
    MIN_BYTES_FOR_COMMIT = 2000
    SILENCE_GAP_SECONDS = 0.45

    stop_evt = asyncio.Event()

    # OpenAI readiness + greeting watchdog
    session_updated_evt = asyncio.Event()
    first_audio_delta_evt = asyncio.Event()

    async def safe_send_twilio(obj: dict):
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    async def cancel_ai(oai_ws):
        """
        Cancel active AI response (barge-in) + clear queued audio.
        """
        nonlocal ai_busy
        playback_queue.clear()
        ai_busy = False
        try:
            await oai_ws.send_json({"type": "response.cancel"})
        except Exception:
            pass

    async def playback_loop():
        """
        Drip base64 g711_ulaw back to Twilio at ~20ms pacing.
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
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    async def send_greeting(oai_ws):
        """
        Say your exact greeting ONCE, only after session.updated confirmed.
        Also includes a watchdog retry that is ALWAYS audio+text (never audio-only).
        """
        nonlocal greeted, ai_busy
        if greeted or stop_evt.is_set():
            return

        # Wait until OpenAI confirms session.update is active
        await session_updated_evt.wait()

        greeted = True
        ai_busy = True
        logging.info("👋 Sending greeting response.create to OpenAI")

        await oai_ws.send_json({
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "voice": "alloy",  # IMPORTANT: set at response-time too
                "instructions": "Say EXACTLY: Hello, thanks for calling Riteway Landscape Products! How can I help you?"
            }
        })

        async def watchdog():
            nonlocal ai_busy
            try:
                await asyncio.wait_for(first_audio_delta_evt.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                if stop_evt.is_set():
                    return
                logging.warning("⚠ No response.audio.delta after greeting. Retrying greeting (audio+text).")
                try:
                    ai_busy = True
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                            "voice": "alloy",
                            "instructions": "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
                        }
                    })
                except Exception:
                    pass

        asyncio.create_task(watchdog())

    async def forward_twilio_to_openai(oai_ws):
        """
        Twilio -> OpenAI
        """
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

                # Greet as soon as session.updated is confirmed
                await send_greeting(oai_ws)

            elif event == "media":
                if not call_active:
                    continue

                received_any_media = True
                last_media_ts = time.time()

                ulaw_b64 = data["media"]["payload"]

                # Barge-in: if AI is talking and caller speaks, cancel AI
                if ai_busy:
                    await cancel_ai(oai_ws)

                # Track bytes (decoded ulaw)
                try:
                    ulaw_bytes = base64.b64decode(ulaw_b64)
                    caller_ulaw_bytes_since_commit += len(ulaw_bytes)
                except Exception:
                    pass

                # Send audio chunk to OpenAI (base64 ulaw)
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
        """
        OpenAI -> Twilio
        Also logs text deltas so we can detect "text but no audio" situations.
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

                # Log most events (helps debug silence)
                if oai_type not in ("rate_limits.updated",):
                    logging.info(f"🤖 OAI event: {oai_type}")

                if oai_type == "session.updated":
                    session_updated_evt.set()
                    logging.info("✅ OpenAI session.updated received")

                if oai_type == "response.audio.delta":
                    first_audio_delta_evt.set()
                    delta_b64 = data.get("delta")
                    if delta_b64:
                        playback_queue.append(delta_b64)

                elif oai_type == "response.output_text.delta":
                    # Debug only: tells us if model is generating text but not audio
                    delta = data.get("delta")
                    if delta:
                        logging.info(f"📝 OAI text delta: {delta}")

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
        Commit only when:
        - caller has spoken
        - we see a pause (silence gap)
        - we have enough audio buffered (prevents commit_empty)
        - AI is not currently busy
        """
        nonlocal caller_ulaw_bytes_since_commit, ai_busy

        while not stop_evt.is_set():
            await asyncio.sleep(0.10)

            if not call_active or not received_any_media:
                continue

            # Require a pause
            if last_media_ts and (time.time() - last_media_ts) < SILENCE_GAP_SECONDS:
                continue

            # Require enough audio
            if caller_ulaw_bytes_since_commit < MIN_BYTES_FOR_COMMIT:
                continue

            # Don't overlap responses
            if ai_busy:
                continue

            try:
                ai_busy = True
                logging.info(f"🗣 COMMIT {caller_ulaw_bytes_since_commit} ulaw bytes")
                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        "voice": "alloy",  # IMPORTANT: set at response-time too
                    }
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
                logging.info("✅ Connected to OpenAI Realtime successfully!")

                # Start OpenAI listener immediately so we catch session.updated
                oai_task = asyncio.create_task(forward_openai_to_twilio(oai_ws))

                # Configure the session
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
                            "\n"
                            "Pricing (always say 'per yard' or 'per ton'):\n"
                            "- Washed Pea Gravel: $42 per yard\n"
                            "- Desert Sun 7/8 Crushed Rock: $40 per yard\n"
                            "- 7/8 Crushed Rock: $25 per yard\n"
                            "- Desert Sun 1.5 Crushed Rock: $40 per yard\n"
                            "- 1.5 Crushed Rock: $25 per yard\n"
                            "- Commercial Road Base: $20 per yard\n"
                            "- 3/8 Minus Fines: $12 per yard\n"
                            "- Desert Sun 1–3 Cobble: $40 per yard\n"
                            "- 8 Landscape Cobble: $40 per yard\n"
                            "- Desert Sun Boulders: $75 per ton\n"
                            "- Fill Dirt: $12 per yard\n"
                            "- Top Soil: $26 per yard\n"
                            "- Screened Premium Top Soil: $40 per yard\n"
                            "- Washed Sand: $65 per yard\n"
                            "- Premium Mulch: $44 per yard\n"
                            "- Colored Shredded Bark: $76 per yard\n"
                            "\n"
                            "Delivery: max 16 yards per load. $75 Grantsville, $115 rest of Tooele Valley.\n"
                            "Outside valley: ask address, repeat it, say $7/mile from Grantsville yard; confirm final total.\n"
                            "\n"
                            "Coverage: 1 yard ~100 sq ft at 3 inches. Formula yards=(L*W*(D/12))/27 round 1 decimal.\n"
                            "Do NOT talk about nurseries/succulents/flowers.\n"
                        )
                    }
                })

                # Wait briefly for session.updated (so greeting doesn't race)
                try:
                    await asyncio.wait_for(session_updated_evt.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    logging.warning("⚠ Did not receive session.updated within 3s (continuing anyway)")

                playback_task = asyncio.create_task(playback_loop())
                twilio_task = asyncio.create_task(forward_twilio_to_openai(oai_ws))
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                # Run until Twilio task ends (hangup)
                await twilio_task

                stop_evt.set()
                playback_task.cancel()
                driver_task.cancel()
                oai_task.cancel()

        except Exception:
            logging.exception("❌ Failed talking to OpenAI realtime. Check API key/billing/model access.")

    try:
        await ws.close()
    except Exception:
        pass

    logging.info("🔚 /media connection closed")

###############################################################################
# LOCAL DEV
###############################################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
