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
# CONFIG
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
# HEALTH
###############################################################################

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
    We:
    - Pause ~1.2s so we don't clip the greeting.
    - Open <Stream> back to our /media websocket.
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

    # Twilio stream ID
    stream_sid = None

    # Queue of audio chunks (already ulaw 8kHz base64) we will send back to Twilio
    playback_queue = deque()

    # Control flags/state
    playback_running = True
    ai_busy = False  # True while AI is in the middle of generating a response
    greeted = False  # True after we play the initial greeting
    caller_audio_since_last_commit_bytes = 0  # how many bytes we buffered since last "turn"
    caller_has_spoken_once = False  # becomes True after first human audio

    ###########################################################################
    # playback_loop: drip audio back to caller
    ###########################################################################
    async def playback_loop():
        """
        Twilio expects 8kHz mono μ-law (G.711 u-law), base64 payload,
        dripping ~20ms chunks.
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
                await asyncio.sleep(0.02)  # about 20ms pacing
            else:
                await asyncio.sleep(0.005)

    ###########################################################################
    # helper: push PCM16 24k -> μ-law 8kHz into playback_queue
    ###########################################################################
    def enqueue_pcm16_from_openai(pcm16_24k: bytes):
        """
        OpenAI gives us PCM16 mono 24kHz in response.audio.delta chunks.
        We:
          1. downsample 24k -> 8kHz PCM16 mono
          2. convert PCM16 -> μ-law
          3. base64 encode
          4. append to playback_queue
        """
        # 24k -> 8k PCM16
        pcm16_8k, _ = audioop.ratecv(
            pcm16_24k,
            2,      # 16-bit
            1,      # mono
            24000,
            8000,
            None
        )
        # PCM16 -> μ-law @8kHz
        ulaw_bytes = audioop.lin2ulaw(pcm16_8k, 2)
        ulaw_b64 = base64.b64encode(ulaw_bytes).decode("ascii")
        playback_queue.append(ulaw_b64)

    ###########################################################################
    # say_greeting_once: queue our custom greeting audio
    ###########################################################################
    async def say_greeting_once(oai_ws):
        """
        We send response.create with a fixed intro line,
        but only ONCE, and only after Twilio stream is ready.
        """
        nonlocal greeted, ai_busy
        if greeted:
            return
        greeted = True
        ai_busy = True  # AI is about to talk
        logging.info("👋 Sending greeting request to OpenAI")
        await oai_ws.send_json({
            "type": "response.create",
            "response": {
                "instructions": (
                    # Must match EXACT wording we want callers to hear:
                    "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
                )
            }
        })

    ###########################################################################
    # forward_twilio_to_openai: user's audio -> OpenAI
    ###########################################################################
    async def forward_twilio_to_openai(oai_ws):
        """
        We receive Twilio messages:
          - 'start': contains streamSid
          - 'media': 20ms chunks of μ-law 8kHz mono audio from caller
          - 'stop': call ended

        We:
          - convert μ-law 8k -> PCM16 24k
          - append to OpenAI input_audio_buffer
          - track how many bytes we've sent since last commit
        """
        nonlocal stream_sid
        nonlocal caller_audio_since_last_commit_bytes
        nonlocal caller_has_spoken_once

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

                # as soon as Twilio is ready, queue the greeting for the caller
                await say_greeting_once(oai_ws)

            elif event == "media":
                # Twilio gives μ-law 8kHz mono in base64
                ulaw_b64 = data["media"]["payload"]
                ulaw_bytes = base64.b64decode(ulaw_b64)

                # μ-law -> PCM16 mono @8kHz
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                # upsample 8k -> 24k PCM16 mono for OpenAI
                pcm16_24k, _ = audioop.ratecv(
                    pcm16_8k,
                    2,      # width=2 bytes (16-bit)
                    1,      # mono
                    8000,
                    24000,
                    None
                )

                # Send chunk to OpenAI
                if oai_ws is not None:
                    try:
                        b64_audio = base64.b64encode(pcm16_24k).decode("ascii")
                        await oai_ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": b64_audio,
                        })
                        # Track that caller talked
                        caller_has_spoken_once = True
                        caller_audio_since_last_commit_bytes += len(pcm16_24k)
                    except Exception:
                        logging.exception("⚠ error sending audio chunk to OpenAI")

            elif event == "stop":
                logging.info("📴 Twilio sent stop (caller hung up)")
                break

        logging.info("🚪 forward_twilio_to_openai exiting")

    ###########################################################################
    # forward_openai_to_twilio: AI audio -> caller
    ###########################################################################
    async def forward_openai_to_twilio(oai_ws):
        """
        Listen to OpenAI Realtime events.
        - response.audio.delta -> queue PCM16 audio (after convert to μ-law)
        - response.done / response.interrupted -> mark ai_busy False
        - error -> log
        """
        nonlocal ai_busy
        nonlocal playback_running

        try:
            async for raw in oai_ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(raw.data)
                oai_type = data.get("type")
                logging.info(f"🤖 OAI event: {oai_type}")

                if oai_type == "response.audio.delta":
                    # base64 PCM16 mono @24kHz from OpenAI
                    pcm16_24k_b64 = data.get("delta")
                    if pcm16_24k_b64:
                        pcm16_24k = base64.b64decode(pcm16_24k_b64)
                        enqueue_pcm16_from_openai(pcm16_24k)

                elif oai_type == "response.done":
                    ai_busy = False
                    logging.info("✅ AI finished speaking")

                elif oai_type == "response.interrupted":
                    ai_busy = False
                    logging.info("⛔ AI interrupted (barge-in)")

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI error event: {data}")

        except Exception:
            logging.exception("💥 Error in forward_openai_to_twilio loop")

        logging.info("🚪 forward_openai_to_twilio exiting")
        playback_running = False

    ###########################################################################
    # speech_drive_loop: decide when to ask OpenAI to answer
    ###########################################################################
    async def speech_drive_loop(oai_ws):
        """
        Every 0.5s:
        - If caller has spoken, we have >100ms audio (about ~4800 bytes at 24kHz 16-bit),
          and AI is not currently busy, we:
            1. commit the audio buffer (end caller turn)
            2. ask OpenAI to respond
          then reset counter.

        This prevents:
        - committing when buffer is empty  -> fixes 'buffer too small' error
        - blasting multiple responses while it's already talking
        """
        nonlocal caller_audio_since_last_commit_bytes
        nonlocal ai_busy

        MIN_BYTES_FOR_COMMIT = 4800  # ~100ms of 24kHz 16-bit mono audio

        while playback_running:
            await asyncio.sleep(0.5)

            # conditions to trigger AI reply
            if (
                caller_audio_since_last_commit_bytes >= MIN_BYTES_FOR_COMMIT
                and not ai_busy
            ):
                logging.info(
                    f"🗣 committing {caller_audio_since_last_commit_bytes} bytes of caller audio to OpenAI"
                )
                ai_busy = True  # we're about to make it talk
                try:
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.commit"
                    })
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            # no custom instructions here -> it will use session instructions
                        }
                    })
                except Exception:
                    logging.exception("⚠ commit/response.create failed")

                # reset
                caller_audio_since_last_commit_bytes = 0

    ###########################################################################
    # CONNECT to OpenAI Realtime
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
                logging.info("🔑 OPENAI_API_KEY is present")

                # Tell OpenAI how to behave for the WHOLE CALL.
                # Strong rules: greeting, tone, pricing, delivery logic, math, barge-in rules.
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        # we want both speech and text
                        "modalities": ["audio", "text"],

                        # AUDIO IN we send: PCM16 mono 24k
                        "input_audio_format": "pcm16",

                        # AUDIO OUT we want: PCM16 mono 24k
                        # (we will downsample + mu-law ourselves)
                        "output_audio_format": "pcm16",

                        "voice": "alloy",

                        # Let model stop talking if caller interrupts
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.4,
                            "silence_duration_ms": 200,
                            "create_response": False,
                            "allow_agent_interrupt": True
                        },

                        "instructions": (
                            # *** CRITICAL TOP RULES ***
                            "TOP PRIORITY RULES:\n"
                            "1. Do not start talking until either (a) you are explicitly asked to greet "
                            "or (b) you are asked to answer after the caller talks.\n"
                            "2. When you are told to greet, your FIRST spoken line must be EXACTLY:\n"
                            "\"Hello, thanks for calling Riteway Landscape Products! How can I help you?\"\n"
                            "Do not change that wording. Do not add anything before it.\n"
                            "Say it one time at the start of the call.\n"
                            "After that, speak naturally.\n"
                            "3. Keep answers under about 20 seconds unless giving prices or doing coverage math.\n"
                            "4. If the caller interrupts you, stop talking immediately.\n\n"

                            "ABOUT RITEWAY:\n"
                            "- Business name: Riteway Landscape Products.\n"
                            "- We sell bulk landscape material by the cubic yard.\n"
                            "- Hours: Monday–Friday, 9 AM to 5 PM. No weekends, no after-hours.\n"
                            "- We mainly serve the Tooele Valley and surrounding areas.\n"
                            "- Most callers already know what they want, like pea gravel or road base.\n\n"

                            "MATERIAL EXPERTISE:\n"
                            "- Washed pea gravel: small rounded gravel, great for play areas, dog runs, walk paths.\n"
                            "- Desert Sun decorative rock: tan/brown, looks good for xeriscape/front yard.\n"
                            "- Road base / crushed rock: compacts hard, great for driveways and parking pads.\n"
                            "- Top soil / screened premium top soil: lawns, gardens, leveling low spots.\n"
                            "- Bark / mulch: weed suppression and moisture control around plants.\n"
                            "- Cobble / boulders: borders, dry creek beds, accents.\n"
                            "If they ask \"what should I use for ___?\", answer like an experienced landscape yard person.\n"
                            "Do NOT talk about succulents, flowers, design services, irrigation kits, etc. "
                            "We are a bulk rock / soil / bark yard, not a nursery.\n\n"

                            "PRICING (ALWAYS SAY 'PER YARD' or 'PER TON'):\n"
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
                            "When giving a price, ALWAYS say 'per yard' or 'per ton'. Example:\n"
                            "\"Washed pea gravel is forty-two dollars per yard.\"\n\n"

                            "DELIVERY:\n"
                            "- We can haul up to 16 yards per load.\n"
                            "- $75 delivery fee to Grantsville.\n"
                            "- $115 delivery fee to Tooele / Stansbury / rest of Tooele Valley.\n"
                            "- Outside Tooele Valley, like Magna:\n"
                            "   1. Ask for full delivery address.\n"
                            "   2. Repeat it back and confirm.\n"
                            "   3. Say: \"We charge seven dollars per mile from our yard in Grantsville, Utah. "
                            "Dispatch will confirm the final total with you.\" \n"
                            "- Never promise an exact time window. Say:\n"
                            "\"We'll confirm the delivery window with dispatch.\"\n\n"

                            "IF THEY WANT TO PLACE AN ORDER:\n"
                            "Collect:\n"
                            "1. Material name.\n"
                            "2. How many yards (remind them we haul up to 16 yards per load).\n"
                            "3. Delivery address.\n"
                            "4. When they want it dropped.\n"
                            "5. Their name and call-back number.\n"
                            "Then say:\n"
                            "\"Perfect, we'll confirm timing and call you back to lock this in.\"\n\n"

                            "COVERAGE / YARDAGE MATH:\n"
                            "- One cubic yard covers ~100 square feet at ~3 inches deep.\n"
                            "- Formula for how many yards they need:\n"
                            "  yards = (length_ft * width_ft * (depth_in / 12)) / 27\n"
                            "- Round to one decimal place.\n"
                            "Example answer style:\n"
                            "\"That area is about 4.5 yards. We can haul up to 16 yards per load.\"\n"
                            "If they ask \"how much does a yard cover?\":\n"
                            "\"One yard covers roughly 100 square feet at about three inches deep.\"\n\n"

                            "HOURS / AFTER HOURS:\n"
                            "- If they want weekend or after 5 PM:\n"
                            "\"We're Monday through Friday, nine to five. I can take your info "
                            "and dispatch will call you back during business hours.\"\n\n"

                            "CONVERSATION STYLE:\n"
                            "- Sound like a real dispatcher in Tooele Valley.\n"
                            "- Be warm, professional, and firm.\n"
                            "- Stop talking immediately if the caller interrupts.\n"
                            "- Do not free-style about topics we don't sell.\n"
                        )
                    }
                })

                # Now spin up helper coroutines
                playback_task = asyncio.create_task(playback_loop())
                driver_task = asyncio.create_task(speech_drive_loop(oai_ws))

                # Bridge Twilio<->OpenAI until hangup
                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws),
                    forward_openai_to_twilio(oai_ws),
                )

                # cleanup
                driver_task.cancel()
                playback_task.cancel()

        except Exception:
            logging.exception("❌ Failed talking to OpenAI realtime. Check API key/billing/model access.")

    # close Twilio socket if still open
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
