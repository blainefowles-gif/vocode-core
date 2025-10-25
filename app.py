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
# /voice  (Twilio hits this first)
###############################################################################

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """
    Twilio calls this URL first.
    We respond with TwiML telling Twilio:
    - wait ~1.2 seconds (so we don't clip the greeting),
    - then open a bidirectional audio Stream to /media.
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
# /media  (live audio websocket Twilio <-> our server <-> OpenAI Realtime)
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None

    # queue of μ-law audio chunks (base64 strings) to send back to Twilio
    playback_queue = deque()
    playback_running = True

    # control flags
    caller_audio_seen = False   # did we receive caller speech recently?
    ai_busy = False             # is OpenAI currently talking?

    ###########################################################################
    # playback_loop
    ###########################################################################
    async def playback_loop():
        """
        Send AI audio chunks back to the caller at ~20ms pace.
        Twilio wants μ-law (G.711 u-law) 8kHz mono.
        We assume items in playback_queue are ALREADY base64 u-law @8kHz.
        """
        await asyncio.sleep(0.1)
        while playback_running:
            if playback_queue and stream_sid:
                ulaw_chunk_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": ulaw_chunk_b64}
                })
                await asyncio.sleep(0.02)  # ~20ms pacing
            else:
                await asyncio.sleep(0.005)

    ###########################################################################
    # speech_drive_loop
    ###########################################################################
    async def speech_drive_loop(oai_ws):
        """
        Every 0.5 seconds:
        - If caller_audio_seen == True and ai_busy == False:
              commit audio buffer + request response
        - Then set ai_busy = True, caller_audio_seen = False
        - When we later get response.done from OpenAI, we'll set ai_busy = False
        This stops us from spamming response.create.
        """
        nonlocal caller_audio_seen, ai_busy, playback_running
        while playback_running:
            await asyncio.sleep(0.5)

            if caller_audio_seen and not ai_busy:
                ai_busy = True
                caller_audio_seen = False
                try:
                    # Tell OpenAI: "that's the end of the caller's turn"
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.commit"
                    })
                    # Ask it to talk back
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            # letting it respond normally now
                        }
                    })
                except Exception:
                    logging.exception("⚠ commit/response.create in speech_drive_loop failed")

    ###########################################################################
    # forward_twilio_to_openai
    ###########################################################################
    async def forward_twilio_to_openai(oai_ws):
        """
        Read audio from Twilio:
        - Twilio sends G.711 μ-law 8kHz mono chunks (base64).
        Convert to PCM16 24kHz mono before feeding to OpenAI input buffer.
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
                # Twilio gives μ-law 8kHz mono (base64)
                ulaw_b64 = data["media"]["payload"]
                ulaw_bytes = base64.b64decode(ulaw_b64)

                # μ-law -> 16-bit PCM @8kHz
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                # upsample 8kHz -> 24kHz
                pcm16_24k, _ = audioop.ratecv(
                    pcm16_8k,  # audio bytes
                    2,         # width (2 bytes = 16-bit)
                    1,         # mono
                    8000,      # from sample rate
                    24000,     # to sample rate
                    None
                )

                # send to OpenAI input buffer
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
        playback_running = False

    ###########################################################################
    # forward_openai_to_twilio
    ###########################################################################
    async def forward_openai_to_twilio(oai_ws):
        """
        Listen for OpenAI events.
        For response.audio.delta:
          - OpenAI will send us base64 PCM16 audio (24kHz mono).
          - We convert 24kHz PCM16 -> 8kHz PCM16 -> μ-law -> base64.
          - Push that into playback_queue so playback_loop can send to Twilio.
        When OpenAI says response.done / response.interrupted:
          - Set ai_busy = False so we can generate the next answer later.
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
                    pcm16_24k_b64 = data.get("delta")
                    if pcm16_24k_b64:
                        # decode PCM16 24kHz mono from OpenAI
                        pcm16_24k = base64.b64decode(pcm16_24k_b64)

                        # downsample 24kHz -> 8kHz PCM16 mono
                        pcm16_8k, _ = audioop.ratecv(
                            pcm16_24k,
                            2,      # width=2 bytes (16-bit)
                            1,      # mono
                            24000,  # from 24kHz
                            8000,   # to 8kHz
                            None
                        )

                        # PCM16 -> μ-law bytes (8kHz mono)
                        ulaw_bytes = audioop.lin2ulaw(pcm16_8k, 2)

                        # base64 so Twilio will play it
                        ulaw_b64 = base64.b64encode(ulaw_bytes).decode("ascii")

                        # queue for playback_loop
                        playback_queue.append(ulaw_b64)

                elif oai_type == "response.done":
                    ai_busy = False
                    logging.info("✅ AI finished speaking")

                elif oai_type == "response.interrupted":
                    ai_busy = False
                    logging.info("⛔ AI interrupted due to caller talking")

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI error event: {data}")

        except Exception:
            logging.exception("💥 Error in forward_openai_to_twilio loop")

        logging.info("🚪 forward_openai_to_twilio exiting")
        playback_running = False

    ###########################################################################
    # connect to OpenAI Realtime, set session, greet
    ###########################################################################
    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY set in environment")
        await ws.close()
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
                logging.info(f"🔑 OPENAI_API_KEY is present")

                # Tell OpenAI how to behave
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        # We want speech and text
                        "modalities": ["audio", "text"],

                        # Caller audio we send in is 24k PCM16 mono
                        "input_audio_format": "pcm16",

                        # We will handle conversion ourselves, so we want PCM16 out
                        "output_audio_format": "pcm16",

                        # Voice to synthesize
                        "voice": "alloy",

                        # Barge-in / turn-taking
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.4,
                            "silence_duration_ms": 200,
                            "create_response": True,
                            "allow_agent_interrupt": True
                        },

                        # Business rules, tone, prices, math, etc.
                        "instructions": (
                            # === GREETING RULE ===
                            "GREETING RULE:\n"
                            "The VERY FIRST THING you say on the call MUST be exactly:\n"
                            "\"Hello, thanks for calling Riteway Landscape Products! How can I help you?\"\n"
                            "Do not rephrase that opening line. Do not add anything before it.\n"
                            "After that first line, act normally.\n\n"

                            # === VOICE / TONE ===
                            "TONE & STYLE:\n"
                            "- You are a real dispatcher at a bulk landscape supply yard.\n"
                            "- Warm, confident, efficient. Local vibe, not corporate.\n"
                            "- Keep answers under ~20 seconds unless giving prices or doing coverage math.\n"
                            "- If caller starts talking while you are talking, STOP immediately.\n"
                            "- Do not over-apologize. Just keep helping.\n\n"

                            # === BUSINESS INFO ===
                            "BUSINESS INFO:\n"
                            "- Business: Riteway Landscape Products.\n"
                            "- We sell bulk landscape material by the cubic yard.\n"
                            "- Hours: Monday–Friday, 9 AM to 5 PM. No after-hours, no weekends.\n"
                            "- We mainly serve Tooele Valley and surrounding areas.\n"
                            "- Most callers already know what they want.\n\n"

                            # === MATERIAL KNOWLEDGE ===
                            "MATERIAL EXPERTISE:\n"
                            "- Washed pea gravel: small rounded gravel, great for play areas, dog runs, walking areas.\n"
                            "- Desert Sun decorative rock: tan/brown rock for xeriscape / front yard curb appeal.\n"
                            "- Road base / crushed rock: compacts hard, good for driveways and parking pads.\n"
                            "- Top soil / screened premium top soil: lawns, gardens, leveling low spots.\n"
                            "- Bark / mulch: helps with weed suppression and moisture around plants.\n"
                            "- Cobble / boulders: borders, dry creek beds, accent rock.\n"
                            "If caller asks 'what should I use for ___?', answer like an experienced yard guy.\n\n"

                            # === PRICING (ALWAYS SAY 'PER YARD' or 'PER TON') ===
                            "PRICING:\n"
                            "- Washed Pea Gravel: $42 per yard.\n"
                            "- Desert Sun 7/8\" Crushed Rock: $40 per yard.\n"
                            "- 7/8\" Crushed Rock: $25 per yard.\n"
                            "- Desert Sun 1.5\" Crushed Rock: $40 per yard.\n"
                            "- 1.5\" Crushed Rock: $25 per yard.\n"
                            "- Commercial Road Base: $20 per yard.\n"
                            "- 3/8\" Minus Fines: $12 per yard.\n"
                            "- Desert Sun 1–3\" Cobble: $40 per yard.\n"
                            "- 8\" Landscape Cobble: $40 per yard.\n"
                            "- Desert Sun Boulders: $75 per ton (per ton, not per yard).\n"
                            "- Fill Dirt: $12 per yard.\n"
                            "- Top Soil: $26 per yard.\n"
                            "- Screened Premium Top Soil: $40 per yard.\n"
                            "- Washed Sand: $65 per yard.\n"
                            "- Premium Mulch: $44 per yard.\n"
                            "- Colored Shredded Bark: $76 per yard.\n"
                            "When you say a price, include 'per yard' or 'per ton'. Example:\n"
                            "\"Washed pea gravel is forty-two dollars per yard.\"\n\n"

                            # === DELIVERY RULES ===
                            "DELIVERY:\n"
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

                            # === ORDER CAPTURE ===
                            "IF THEY WANT TO PLACE AN ORDER / DELIVERY:\n"
                            "Collect:\n"
                            "1. Material they want.\n"
                            "2. How many yards (remind them: up to 16 yards per load).\n"
                            "3. Delivery address.\n"
                            "4. When they want it dropped.\n"
                            "5. Their name and callback number.\n"
                            "Then say:\n"
                            "\"Perfect, we'll confirm timing and call you back to lock this in.\"\n\n"

                            # === COVERAGE / YARDAGE MATH ===
                            "COVERAGE ESTIMATES:\n"
                            "- One cubic yard covers roughly 100 square feet at about 3 inches deep.\n"
                            "- Formula to figure yards from dimensions:\n"
                            "  yards = (length_ft * width_ft * (depth_in / 12)) / 27\n"
                            "- Round to one decimal place when you answer.\n"
                            "- Example style:\n"
                            "\"That area is about 4.5 yards. We can haul up to 16 yards per load.\"\n"
                            "If they ask 'how much will a yard cover?', answer:\n"
                            "\"One yard covers roughly 100 square feet at about three inches deep.\"\n\n"

                            # === HOURS / AFTER HOURS ===
                            "HOURS POLICY:\n"
                            "- Hours are Monday–Friday, 9 AM to 5 PM.\n"
                            "- No weekends, no after-hours.\n"
                            "- If they ask for weekend delivery:\n"
                            "\"We're not running weekends, but I can take your info and dispatch will call you during business hours.\"\n\n"

                            # === HOW TO TALK ===
                            "CALL FLOW:\n"
                            "1. FIRST LINE: Say the exact greeting line.\n"
                            "2. Ask how you can help.\n"
                            "3. Answer directly.\n"
                            "4. If they sound ready to buy, collect delivery/order details.\n"
                            "5. If they interrupt you, STOP immediately and listen.\n"
                            "6. Keep it tight and helpful, like a dispatcher.\n"
                        )
                    }
                })

                # Force the very first spoken line so we don't rely on VAD timing:
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
                        )
                    }
                })

                # Start helper tasks
                playback_task = asyncio.create_task(playback_loop())
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                # Run pump loops until call ends
                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws),
                    forward_openai_to_twilio(oai_ws),
                )

                # cleanup after hangup
                driver_task.cancel()
                playback_task.cancel()

        except Exception:
            logging.exception("❌ Failed talking to OpenAI realtime. Check API key/billing/model access.")

    # finally close Twilio socket if still open
    try:
        await ws.close()
    except Exception:
        pass

    logging.info("🔚 /media connection closed")

###############################################################################
# LOCAL DEV ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
