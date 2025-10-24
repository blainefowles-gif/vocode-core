import os
import json
import base64
import asyncio
import logging
import audioop
import aiohttp
import numpy as np
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
    We add ~1.2s pause before streaming so OpenAI boots
    and the very first word doesn't get chopped.
    No 'ring', no 'connecting you', just clean handoff.
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
# /media: BRIDGE TWILIO <-> OPENAI REALTIME
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None

    # queue of mulaw audio chunks (base64) waiting to go TO Twilio
    playback_queue = deque()
    playback_running = True
    playback_task = None

    #
    # playback_loop
    #
    async def playback_loop():
        """
        Take G.711 u-law audio chunks from OpenAI and drip them back to Twilio
        in ~20ms frames so caller hears smooth audio.
        """
        await asyncio.sleep(0.1)  # tiny delay to let Twilio finish setup
        while playback_running:
            if playback_queue and stream_sid:
                chunk_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk_b64}
                })
                # 20ms pacing (Twilio expects ~160-byte ulaw frames / 8kHz)
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    #
    # forward_twilio_to_openai
    #
    async def forward_twilio_to_openai(oai_ws):
        """
        1. Receive Twilio 'media' events (caller voice in μ-law @8kHz)
        2. Convert μ-law -> PCM16 8k -> upsample to 24k PCM16
        3. Send to OpenAI via input_audio_buffer.append
        4. On caller 'stop', break
        """
        nonlocal stream_sid
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
                logging.warning(f"⚠ Non-JSON message from Twilio: {raw_msg}")
                continue

            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

            elif event == "media":
                # caller audio as base64 μ-law
                ulaw_bytes = base64.b64decode(data["media"]["payload"])
                # μ-law -> linear PCM16 @8kHz (2 bytes/sample, mono)
                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)
                # upsample 8k -> 24k
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)

                if oai_ws is not None:
                    b64_audio = base64.b64encode(pcm16_24k).decode("ascii")
                    # send chunk to OpenAI buffer
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": b64_audio,
                    })

            elif event == "stop":
                logging.info("📴 Twilio sent stop (caller hung up)")
                break

            # after each caller chunk / at pauses, we "commit" so AI knows it's the caller's turn
            # NOTE: we commit after each loop tick. This helps the AI respond fast,
            #       and lets it stop talking sooner if caller interrupts.
            if oai_ws is not None:
                try:
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.commit"
                    })
                    # ask OpenAI to create a response, BUT we use VAD settings so
                    # it will cut itself off if caller starts talking.
                    await oai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            # empty instructions here means:
                            # "continue the normal conversation flow"
                        }
                    })
                except Exception:
                    # we don't want to kill the whole call if one commit fails
                    logging.exception("⚠ commit/response.create failed")

        logging.info("🚪 forward_twilio_to_openai exiting")

    #
    # forward_openai_to_twilio
    #
    async def forward_openai_to_twilio(oai_ws):
        """
        1. Read streaming events from OpenAI realtime
        2. Each response.audio.delta => mulaw frame (base64) to playback_queue
        3. If OpenAI sends 'response.interrupted'/'barge-in', we just keep rolling.
        """
        try:
            async for raw in oai_ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(raw.data)
                oai_type = data.get("type")
                logging.info(f"🤖 OAI event: {oai_type}")

                if oai_type == "response.audio.delta":
                    ulaw_chunk_b64 = data.get("delta")
                    if ulaw_chunk_b64:
                        playback_queue.append(ulaw_chunk_b64)

                elif oai_type == "response.completed":
                    logging.info("✅ AI finished speaking")

                elif oai_type == "response.interrupted":
                    # OpenAI decided to stop talking because caller started.
                    logging.info("⛔ AI speech interrupted due to caller barge-in")

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI error event: {data}")

        except Exception:
            logging.exception("💥 Error in forward_openai_to_twilio loop")

        logging.info("🚪 forward_openai_to_twilio exiting")

    #
    # connect to OpenAI realtime
    #
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

                ################################################################
                # IMPORTANT BEHAVIOR + KNOWLEDGE SETUP
                ################################################################
                # We will:
                #  - lock the exact greeting
                #  - tune turn_detection so it stops when the caller barges in
                #  - load Riteway product/pricing/delivery/yardage knowledge
                #  - train it to sound like an expert in landscape materials
                ################################################################
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],

                        # We're sending caller audio to OpenAI as 24k PCM16
                        "input_audio_format": "pcm16",

                        # We want μ-law 8k audio back for Twilio
                        "output_audio_format": "g711_ulaw",

                        # Voice to use
                        "voice": "alloy",

                        # TURN / INTERRUPTION CONTROL ============================
                        # We ask OpenAI to listen for silence and also to STOP
                        # when the caller starts talking ("barge-in").
                        # Lower threshold + short silence_duration_ms
                        # = more responsive / less rambling.
                        "turn_detection": {
                            "type": "server_vad",
                            # how sensitive it is to new voice activity:
                            # lower = more sensitive (cuts agent off faster)
                            "threshold": 0.4,
                            # how long of silence (ms) counts as "your turn is done"
                            "silence_duration_ms": 200,
                            # if true, OpenAI will auto start responding
                            "create_response": True,
                            # let barge-in interrupt assistant speech
                            "allow_agent_interrupt": True
                        },

                        # BRAINS =================================================
                        "instructions": (
                            # --- GREETING RULE (lock this line) ---
                            "GREETING RULE:\n"
                            "The VERY FIRST THING you say on the call MUST be exactly:\n"
                            "\"Hello, thanks for calling Riteway Landscape Products! How can I help you?\"\n"
                            "Do not rephrase that opening line. Do not add anything before it.\n"
                            "After that first line, act normally.\n\n"

                            # --- CONVERSATION STYLE ---
                            "VOICE / PERSONALITY:\n"
                            "- You are a real dispatcher at a bulk landscape supply yard.\n"
                            "- Warm, confident, knowledgeable, professional, local.\n"
                            "- You are helpful, but also keep things moving. Don't ramble.\n"
                            "- Keep each answer under 20 seconds unless you're actively quoting prices or doing a coverage calculation.\n"
                            "- If the caller starts talking while you're talking, STOP immediately and listen.\n"
                            "- If you get interrupted, don't apologize over and over. Just continue helping.\n\n"

                            # --- BUSINESS INFO ---
                            "BUSINESS INFO:\n"
                            "- Business name: Riteway Landscape Products.\n"
                            "- We sell bulk landscape material by the cubic yard.\n"
                            "- We are open Monday–Friday, 9 AM to 5 PM. We do not do after-hours or weekends.\n"
                            "- We mainly serve Tooele Valley and surrounding areas.\n"
                            "- Most callers already know what they want and how many yards.\n"
                            "- You can absolutely answer product questions and help them choose rock, mulch, soil, etc.\n\n"

                            # --- EXPERT HELP ---
                            "MATERIAL KNOWLEDGE:\n"
                            "You are an expert in gravel, rock, soil, mulch, and general landscape coverage.\n"
                            "Explain things in plain English like a pro in the yard, not like a sales script.\n"
                            "Example guidance:\n"
                            "- Pea gravel: smooth, small, good for walking areas, dog runs, play areas.\n"
                            "- Desert Sun rock: decorative tan/brown rock, nice for front yards and xeriscape.\n"
                            "- Road base / crushed rock: compacts hard, great for driveways and parking pads.\n"
                            "- Top soil / premium screened top soil: for lawns, garden beds, leveling yards.\n"
                            "- Bark / mulch: weed suppression, moisture retention around plants.\n"
                            "- Cobble / boulders: decorative accents, edging, dry creek look.\n"
                            "If they ask 'what should I use for ___?' answer like a landscape supply expert.\n\n"

                            # --- PRICING ---
                            "PRICING (always say 'per yard' or 'per ton'):\n"
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
                            "- Colored Shredded Bark: $76 per yard.\n\n"

                            "When you say a price, INCLUDE 'per yard' or 'per ton'. Example:\n"
                            "\"Washed pea gravel is forty-two dollars per yard.\"\n\n"

                            # --- DELIVERY RULES ---
                            "DELIVERY:\n"
                            "- We can haul up to 16 yards per load.\n"
                            "- Delivery fee: $75 to Grantsville.\n"
                            "- Delivery fee: $115 to Tooele / Stansbury / rest of Tooele Valley.\n"
                            "- For anything outside Tooele Valley (for example Magna):\n"
                            "  1. Ask for the full delivery address.\n"
                            "  2. Repeat it back and confirm it's correct.\n"
                            "  3. Say: 'We charge $7 per mile from our yard in Grantsville, Utah.'\n"
                            "  4. Tell them: 'We'll confirm that total when dispatch calls you back.'\n"
                            "- Never promise exact delivery time. Instead say:\n"
                            "\"We'll confirm the delivery window with dispatch.\"\n\n"

                            # --- SCHEDULING / TAKING AN ORDER ---
                            "IF THEY WANT TO PLACE AN ORDER OR SCHEDULE A DELIVERY:\n"
                            "Collect these in a calm, natural way:\n"
                            "1. Material they want.\n"
                            "2. How many yards they want (remind them: up to 16 yards per load).\n"
                            "3. Delivery address.\n"
                            "4. When they want it dropped.\n"
                            "5. Their name and callback number.\n"
                            "Then say:\n"
                            "\"Perfect, we'll confirm timing and call you back to lock this in.\"\n\n"

                            # --- COVERAGE / YARDAGE MATH ---
                            "COVERAGE ESTIMATES:\n"
                            "- One cubic yard of material covers about 100 square feet at roughly 3 inches deep.\n"
                            "- If they give dimensions, calculate yards like this:\n"
                            "  yards_needed = (length_ft * width_ft * (depth_inches / 12)) / 27\n"
                            "  1) Multiply length_ft * width_ft to get square feet.\n"
                            "  2) depth_inches/12 = depth in feet.\n"
                            "  3) square feet * depth(feet) = cubic feet.\n"
                            "  4) Divide cubic feet by 27 to get cubic yards.\n"
                            "- Round to one decimal place.\n"
                            "- Tell them: 'You’re looking at about X yards. We can haul up to 16 yards per load.'\n"
                            "- Then ask if they’d like it delivered.\n\n"

                            # --- HOURS / POLICIES ---
                            "HOURS & RULES:\n"
                            "- Hours: Monday–Friday, 9 AM to 5 PM.\n"
                            "- We do not do after-hours or weekend runs.\n"
                            "- If they ask for weekend or after 5 PM, say you can't schedule that, but you can take their info and dispatch will follow up during business hours.\n\n"

                            # --- CALL FLOW ---
                            "CALL FLOW:\n"
                            "1. FIRST LINE: Say the exact required greeting line.\n"
                            "2. Ask how you can help.\n"
                            "3. Answer questions simply.\n"
                            "4. If they sound ready to buy, shift into gathering delivery details.\n"
                            "5. If they talk over you, STOP and listen.\n"
                            "6. Never read a giant speech when a short answer will do.\n"
                        )
                    }
                })

                # Send the greeting manually right now. This forces the first outbound audio
                # to be EXACTLY what you want.
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Hello, thanks for calling Riteway Landscape Products! How can I help you?"
                        )
                    }
                })

                # start sending audio back to Twilio
                playback_task = asyncio.create_task(playback_loop())

                # run both directions at once
                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws),
                    forward_openai_to_twilio(oai_ws),
                )

        except Exception:
            logging.exception(
                "❌ Failed talking to OpenAI realtime. "
                "Check API key, billing, or model access."
            )

    # cleanup
    playback_running = False
    if playback_task:
        playback_task.cancel()

    logging.info("🔚 /media connection closed")


###############################################################################
# LOCAL DEV ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
