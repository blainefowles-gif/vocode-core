import os
import json
import base64
import asyncio
import logging
import audioop
import aiohttp
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

###############################################################################
# BASIC CONFIG
###############################################################################

PUBLIC_BASE_URL = "https://riteway-ai-agent.onrender.com"
WS_MEDIA_URL = "wss://" + PUBLIC_BASE_URL.replace("https://", "").replace("http://", "") + "/media"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = "gpt-4o-realtime-preview"

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
# HEALTHCHECK
###############################################################################

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "model": REALTIME_MODEL,
        "has_api_key": bool(OPENAI_API_KEY)
    })

###############################################################################
# TWILIO /voice
###############################################################################

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """
    Twilio hits this first.
    We add ~1.2s pause to avoid clipping first word.
    """
    logging.info("☎ Twilio hit /voice")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Pause length="1.2"/>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

###############################################################################
# /media: TWILIO <-> OPENAI BRIDGE
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None

    # queue of μ-law audio chunks (base64) that we will drip back to Twilio
    playback_queue = deque()
    playback_running = True
    playback_task = None

    # We will track:
    # - caller_audio_seen: did we get new caller audio recently?
    # - ai_busy: is OpenAI already in the middle of giving a response?
    caller_audio_seen = False
    ai_busy = False  # we'll flip this True when we ask AI to respond, then False on "response.done"

    ###########################################################################
    # playback_loop
    ###########################################################################
    async def playback_loop():
        """
        Send AI audio chunks back to the caller at ~20ms pacing.
        """
        await asyncio.sleep(0.1)
        while playback_running:
            if playback_queue and stream_sid:
                chunk_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk_b64}
                })
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    ###########################################################################
    # speech_drive_loop
    ###########################################################################
    async def speech_drive_loop(oai_ws):
        """
        Every 0.5 seconds:
        - if we've heard caller audio recently
        - and AI is not already talking
        => commit the buffer and ask OpenAI to respond ONCE.

        This prevents spamming response.create constantly.
        """
        nonlocal caller_audio_seen, ai_busy
        while playback_running:
            await asyncio.sleep(0.5)

            if caller_audio_seen and not ai_busy:
                # lock so we don't spam
                ai_busy = True
                caller_audio_seen = False

                try:
                    # tell OpenAI "the user finished a thought"
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.commit"
                    })
                    # now ask it to talk
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            # empty instructions => continue conversation naturally
                        }
                    })
                except Exception:
                    logging.exception("⚠ commit/response.create in speech_drive_loop failed")

    ###########################################################################
    # forward_twilio_to_openai
    ###########################################################################
    async def forward_twilio_to_openai(oai_ws):
        """
        - Receive Twilio audio events.
        - Convert μ-law 8kHz -> PCM16 24kHz.
        - Append to OpenAI's input buffer.
        - Mark that we've got fresh caller audio (caller_audio_seen = True).
        We DO NOT ask for a response here anymore to avoid overlap.
        """
        nonlocal stream_sid, caller_audio_seen, playback_running

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
                logging.warning(f"⚠ Non-JSON from Twilio: {raw_msg}")
                continue

            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

            elif event == "media":
                # caller audio base64 μ-law (8 kHz mono)
                ulaw_bytes = base64.b64decode(data["media"]["payload"])

                # μ-law -> PCM16 @8kHz
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                # Upsample 8k -> 24k PCM16 mono
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)

                # Send audio chunk to OpenAI
                if oai_ws is not None:
                    try:
                        b64_audio = base64.b64encode(pcm16_24k).decode("ascii")
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": b64_audio,
                        })
                        caller_audio_seen = True
                    except Exception:
                        logging.exception("⚠ error sending audio chunk to OpenAI")

            elif event == "stop":
                logging.info("📴 Twilio sent stop (caller hung up)")
                break

        logging.info("🚪 forward_twilio_to_openai exiting")
        playback_running = False  # stop loops when caller hangs up

    ###########################################################################
    # forward_openai_to_twilio
    ###########################################################################
    async def forward_openai_to_twilio(oai_ws):
        """
        Read streaming events from OpenAI realtime:
        - response.audio.delta    -> push μ-law chunks to playback_queue
        - response.done           -> AI finished, so ai_busy = False
        - response.interrupted    -> AI stopped itself because caller talked
        """
        nonlocal ai_busy, playback_running
        try:
            async for raw in oai_ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")
                logging.info(f"🤖 OAI event: {oai_type}")

                if oai_type == "response.audio.delta":
                    # base64 ulaw audio from OpenAI
                    ulaw_chunk_b64 = data.get("delta")
                    if ulaw_chunk_b64:
                        playback_queue.append(ulaw_chunk_b64)

                elif oai_type == "response.done":
                    # AI finished that reply. Safe to let it talk again next time.
                    ai_busy = False
                    logging.info("✅ AI finished speaking")

                elif oai_type == "response.interrupted":
                    # AI self-stopped because caller barged in
                    ai_busy = False
                    logging.info("⛔ AI interrupted due to caller talking")

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI error event: {data}")

        except Exception:
            logging.exception("💥 Error in forward_openai_to_twilio loop")

        logging.info("🚪 forward_openai_to_twilio exiting")
        playback_running = False  # stop loops if OpenAI dies

    ###########################################################################
    # connect to OpenAI realtime
    ###########################################################################
    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY set in environment")
        return

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(
                f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as oai_ws:
                logging.info("✅ Connected to OpenAI Realtime successfully!")

                # Configure the assistant's brain, tone, pricing, delivery rules, math, etc.
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],

                        # we send caller audio to OpenAI as raw 24k PCM16
                        "input_audio_format": "pcm16",

                        # we want OpenAI to send us ulaw 8k so we can hand it straight to Twilio
                        "output_audio_format": "g711_ulaw",

                        "voice": "alloy",

                        # Turn detection to reduce rambling and allow interruption
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.4,
                            "silence_duration_ms": 200,
                            "create_response": True,
                            "allow_agent_interrupt": True
                        },

                        "instructions": (
                            # GREETING RULE
                            "GREETING RULE:\n"
                            "The VERY FIRST THING you say on the call MUST be exactly:\n"
                            "\"Hello, thanks for calling Riteway Landscape Products! How can I help you?\"\n"
                            "Do not rephrase that opening line. Do not add anything before it.\n"
                            "After that first line, act normally.\n\n"

                            # VOICE
                            "TONE & STYLE:\n"
                            "- You are a real dispatcher at a bulk landscape supply yard.\n"
                            "- Warm, confident, efficient. Local vibe, not corporate.\n"
                            "- Keep answers under ~20 seconds unless giving prices or doing coverage math.\n"
                            "- If caller starts talking while you are talking, STOP immediately.\n"
                            "- Do not over-apologize. Just keep helping.\n\n"

                            # BUSINESS
                            "BUSINESS INFO:\n"
                            "- Business: Riteway Landscape Products.\n"
                            "- We sell bulk landscape material by the cubic yard.\n"
                            "- Hours: Monday–Friday, 9 AM to 5 PM. No after-hours, no weekends.\n"
                            "- We mainly serve Tooele Valley and surrounding areas.\n"
                            "- Most callers already know what they want.\n\n"

                            # MATERIAL KNOWLEDGE
                            "MATERIAL EXPERTISE:\n"
                            "- Pea gravel: small rounded gravel, great for play areas, dog runs, walking areas.\n"
                            "- Desert Sun decorative rock: tan/brown rock for xeriscape / front yard look.\n"
                            "- Road base / crushed rock: compacts hard, good for driveways and parking pads.\n"
                            "- Top soil / screened premium top soil: lawns, gardens, leveling.\n"
                            "- Bark / mulch: helps with weed suppression and moisture around plants.\n"
                            "- Cobble / boulders: decorative borders, dry creek look, accent rocks.\n"
                            "If caller asks 'what should I use for ___?', answer like an experienced yard guy.\n\n"

                            # PRICING
                            "PRICING: Always say 'per yard' or 'per ton'.\n"
                            "- Washed Pea Gravel: $42 per yard.\n"
                            "- Desert Sun 7/8\" Crushed Rock: $40 per yard.\n"
                            "- 7/8\" Crushed Rock: $25 per yard.\n"
                            "- Desert Sun 1.5\" Crushed Rock: $40 per yard.\n"
                            "- 1.5\" Crushed Rock: $25 per yard.\n"
                            "- Commercial Road Base: $20 per yard.\n"
                            "- 3/8\" Minus Fines: $12 per yard.\n"
                            "- Desert Sun 1–3\" Cobble: $40 per yard.\n"
                            "- 8\" Landscape Cobble: $40 per yard.\n"
                            "- Desert Sun Boulders: $75 per ton (say 'per ton', not per yard).\n"
                            "- Fill Dirt: $12 per yard.\n"
                            "- Top Soil: $26 per yard.\n"
                            "- Screened Premium Top Soil: $40 per yard.\n"
                            "- Washed Sand: $65 per yard.\n"
                            "- Premium Mulch: $44 per yard.\n"
                            "- Colored Shredded Bark: $76 per yard.\n"
                            "When you say a price, include 'per yard' or 'per ton'.\n"
                            "Example: 'Washed pea gravel is forty-two dollars per yard.'\n\n"

                            # DELIVERY / FEES
                            "DELIVERY RULES:\n"
                            "- We can haul up to 16 yards per load.\n"
                            "- $75 delivery fee to Grantsville.\n"
                            "- $115 delivery fee to Tooele / Stansbury / rest of Tooele Valley.\n"
                            "- Outside Tooele Valley (for example Magna):\n"
                            "   1. Ask for full delivery address.\n"
                            "   2. Repeat it back and confirm it's correct.\n"
                            "   3. Say: 'We charge seven dollars per mile from our yard in Grantsville, Utah.'\n"
                            "   4. Say: 'We'll confirm that total when dispatch calls you back.'\n"
                            "- Never promise an exact delivery time. Say:\n"
                            "\"We'll confirm the delivery window with dispatch.\"\n\n"

                            # ORDER CAPTURE
                            "IF THEY WANT TO PLACE AN ORDER / DELIVERY:\n"
                            "Get these details:\n"
                            "1. Material they want.\n"
                            "2. How many yards (remind them: up to 16 yards per load).\n"
                            "3. Delivery address.\n"
                            "4. When they want it dropped.\n"
                            "5. Their name and callback number.\n"
                            "Then say:\n"
                            "\"Perfect, we'll confirm timing and call you back to lock this in.\"\n\n"

                            # COVERAGE MATH
                            "COVERAGE / YARDAGE ESTIMATES:\n"
                            "- One cubic yard covers roughly 100 square feet at about 3 inches deep.\n"
                            "- Formula (don't read the formula unless they ask):\n"
                            "  yards = (length_ft * width_ft * (depth_in / 12)) / 27\n"
                            "- Round to one decimal place.\n"
                            "- Say it like:\n"
                            "\"You’re looking at about 4.5 yards. We can haul up to 16 yards per load.\"\n\n"

                            # HOURS POLICY
                            "HOURS / WEEKENDS:\n"
                            "- Hours are Monday–Friday, 9 AM to 5 PM.\n"
                            "- We do not deliver after hours or weekends.\n"
                            "- If they ask for weekend, say:\n"
                            "\"We’re not running weekends, but I can take your info and dispatch will call you during business hours.\"\n\n"

                            # HOW TO TALK
                            "CALL FLOW:\n"
                            "1. FIRST LINE: Say the exact greeting line.\n"
                            "2. Ask how you can help.\n"
                            "3. Answer their question.\n"
                            "4. If they sound ready to buy, collect delivery/order details.\n"
                            "5. If they interrupt you, stop talking immediately and listen.\n"
                            "6. Do not ramble. Keep it tight, like a dispatcher.\n"
                        )
                    }
                })

                # Send FIRST EVER greeting manually, ONCE.
                # (This starts the conversation.)
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
                        )
                    }
                })

                # Start background loops
                playback_task = asyncio.create_task(playback_loop())
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                # Run both forwarders (these block until hangup / error)
                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws),
                    forward_openai_to_twilio(oai_ws),
                )

                # when either forwarder exits, stop the helper loops
                driver_task.cancel()

        except Exception:
            logging.exception("❌ Failed talking to OpenAI realtime. Check API key/billing/model access.")

    # Cleanup
    global playback_running  # just being explicit
    playback_running = False
    if playback_task:
        playback_task.cancel()
    logging.info("🔚 /media connection closed")

###############################################################################
# LOCAL DEV
###############################################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
