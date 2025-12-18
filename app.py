import os
import json
import asyncio
import base64
import logging
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

# Use your model; keep as-is if this is what you already have access to
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

# Twilio Media Streams audio is g711 u-law @ 8kHz, base64 payloads
# We'll set OpenAI Realtime to ALSO use g711_ulaw so we can pass-through.
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
# BASIC PAGES (optional but helps avoid confusion)
###############################################################################

@app.get("/")
async def root():
    return JSONResponse({"ok": True, "message": "Riteway AI Agent is running. Use /voice for Twilio."})

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "model": REALTIME_MODEL,
        "has_api_key": bool(OPENAI_API_KEY)
    })

###############################################################################
# /voice - Twilio webhook to start call
###############################################################################

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """
    Twilio hits this first.
    We add a short pause so the greeting doesn't get clipped,
    then we <Stream> back to our /media websocket.
    """
    logging.info("☎ Twilio hit /voice")

    # Use whole seconds (Twilio is happier this way)
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
        logging.error("❌ No OPENAI_API_KEY set in environment")
        await ws.close()
        return

    stream_sid = None

    # Queue of audio chunks (base64 g711_ulaw) to send back to Twilio
    playback_queue = deque()

    # State flags
    playback_running = True
    ai_busy = False
    greeted = False

    # Track caller audio so we only commit once we have enough (>= 100ms)
    frames_since_commit = 0
    MIN_FRAMES_FOR_COMMIT = 5  # Twilio sends ~20ms per frame => 5 frames = 100ms

    # We also want to cancel AI speech when caller talks (barge-in)
    # This prevents rambling and prevents "active response" errors.
    async def cancel_ai_response(oai_ws):
        nonlocal ai_busy
        try:
            await oai_ws.send_json({"type": "response.cancel"})
        except Exception:
            # If cancel isn't supported or fails, ignore; we'll still gate on ai_busy.
            pass
        # Clear queued audio immediately so caller isn't hearing stale speech
        playback_queue.clear()
        ai_busy = False

    ###########################################################################
    # playback_loop: drip audio back to caller
    ###########################################################################
    async def playback_loop():
        """
        Twilio expects base64 g711_ulaw payloads, ~20ms pacing.
        We'll pace the queue out.
        """
        await asyncio.sleep(0.05)
        while playback_running:
            if playback_queue and stream_sid:
                chunk_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk_b64},
                })
                await asyncio.sleep(0.02)  # ~20ms
            else:
                await asyncio.sleep(0.005)

    ###########################################################################
    # Greeting: force exact wording at start
    ###########################################################################
    async def send_greeting(oai_ws):
        """
        Make the agent speak the exact greeting once.
        IMPORTANT: Only do this after session.update is sent.
        """
        nonlocal greeted, ai_busy
        if greeted:
            return
        greeted = True
        ai_busy = True

        # We force it explicitly here.
        await oai_ws.send_json({
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": "Say EXACTLY: Hello, thanks for calling Riteway Landscape Products! How can I help you?",
            }
        })

    ###########################################################################
    # Twilio -> OpenAI
    ###########################################################################
    async def forward_twilio_to_openai(oai_ws, session_ready_evt: asyncio.Event):
        nonlocal stream_sid, frames_since_commit, ai_busy

        while True:
            try:
                raw_msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Twilio disconnected")
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
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

                # Wait until OpenAI session.update is done before greeting
                await session_ready_evt.wait()
                await send_greeting(oai_ws)

            elif event == "media":
                # Caller audio frame (base64 g711_ulaw)
                ulaw_b64 = data["media"]["payload"]

                # If AI is talking and caller starts speaking -> cancel AI immediately (barge-in)
                # This stops rambling + prevents active response errors.
                if ai_busy:
                    await cancel_ai_response(oai_ws)

                # Send audio directly to OpenAI in SAME format (g711_ulaw)
                try:
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": ulaw_b64,
                    })
                    frames_since_commit += 1
                except Exception:
                    logging.exception("⚠ error sending audio chunk to OpenAI")

            elif event == "stop":
                logging.info("📴 Twilio sent stop (caller hung up)")
                break

        logging.info("🚪 forward_twilio_to_openai exiting")

    ###########################################################################
    # OpenAI -> Twilio
    ###########################################################################
    async def forward_openai_to_twilio(oai_ws):
        nonlocal ai_busy, playback_running

        try:
            async for raw in oai_ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")

                if oai_type in ("session.created", "rate_limits.updated"):
                    continue

                if oai_type == "response.audio.delta":
                    # This is base64 audio in our output_audio_format (g711_ulaw)
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
        logging.info("🚪 forward_openai_to_twilio exiting")

    ###########################################################################
    # Drive loop: commit + ask model to respond after caller speaks enough
    ###########################################################################
    async def speech_drive_loop(oai_ws):
        nonlocal frames_since_commit, ai_busy

        while playback_running:
            await asyncio.sleep(0.25)

            # Only create a response if:
            # - caller has at least 100ms buffered
            # - AI isn't currently speaking
            if frames_since_commit >= MIN_FRAMES_FOR_COMMIT and not ai_busy:
                try:
                    ai_busy = True
                    await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"]
                            # No extra instructions here; uses session instructions
                        }
                    })
                except Exception:
                    logging.exception("⚠ commit/response.create failed")
                    ai_busy = False

                frames_since_commit = 0

    ###########################################################################
    # CONNECT to OpenAI Realtime
    ###########################################################################
    session_ready_evt = asyncio.Event()

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

                # Session config (IMPORTANT: audio+text, g711_ulaw in/out)
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "input_audio_format": OAI_AUDIO_FORMAT,
                        "output_audio_format": OAI_AUDIO_FORMAT,
                        "voice": "alloy",

                        # We'll manually trigger responses (so create_response False)
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
                            "When giving a price, format like:\n"
                            "\"Pea gravel is forty-two dollars per yard.\"\n"
                            "\n"
                            "DELIVERY:\n"
                            "- Max 16 yards per load.\n"
                            "- $75 delivery fee to Grantsville.\n"
                            "- $115 delivery fee to the rest of Tooele Valley.\n"
                            "- Outside Tooele Valley (ex: Magna): ask for full address, repeat it back, and say:\n"
                            "  \"We charge seven dollars per mile from our yard in Grantsville, Utah. We’ll confirm the final total.\"\n"
                            "\n"
                            "ORDER TAKING (when caller wants delivery): collect\n"
                            "1) material 2) yards 3) address 4) preferred day/time 5) name + callback\n"
                            "Then say we will confirm dispatch and timing.\n"
                            "\n"
                            "COVERAGE + YARDAGE MATH:\n"
                            "- 1 cubic yard covers ~100 sq ft at ~3 inches deep.\n"
                            "- yards = (length_ft * width_ft * (depth_in/12)) / 27\n"
                            "- Round to 1 decimal place.\n"
                            "If caller gives measurements, calculate yards and tell them how many loads (16 yd max).\n"
                            "\n"
                            "IMPORTANT:\n"
                            "- Do NOT talk about nurseries, succulents, flowers, or unrelated items.\n"
                            "- Assume callers usually know what they want; ask short clarifying questions.\n"
                        )
                    }
                })

                # Mark session ready so greeting can fire after Twilio "start"
                session_ready_evt.set()

                # Start tasks
                playback_task = asyncio.create_task(playback_loop())
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws, session_ready_evt),
                    forward_openai_to_twilio(oai_ws),
                )

                driver_task.cancel()
                playback_task.cancel()

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
