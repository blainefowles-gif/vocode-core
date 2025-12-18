import os
import json
import base64
import asyncio
import logging
import time
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
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

# Pick a made-up name here (or set env var AGENT_NAME on Render)
AGENT_NAME = os.getenv("AGENT_NAME", "Mason")

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

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "model": REALTIME_MODEL,
        "has_api_key": bool(OPENAI_API_KEY)
    })

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """
    Twilio hits this first, we return TwiML that streams to /media.
    """
    logging.info("☎ Twilio hit /voice")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
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
        logging.error("❌ No OPENAI_API_KEY in environment.")
        await ws.close()
        return

    stream_sid = None

    # AI->Twilio audio queue (g711_ulaw base64)
    playback_queue = deque()
    playback_running = True

    # Conversation state
    greeted = False
    ai_speaking = False         # True if we are currently receiving/playing AI audio
    ai_busy = False             # True if we have requested a response and are waiting
    stop_evt = asyncio.Event()

    # Caller audio gating (prevents committing mid-sentence + prevents commit_empty)
    caller_audio_since_commit_bytes = 0
    last_media_ts = 0.0

    # OpenAI needs >= 100ms audio to commit.
    # At 24kHz PCM16 mono:
    # 100ms = 2400 samples * 2 bytes = 4800 bytes.
    MIN_BYTES_FOR_COMMIT = 4800
    SILENCE_GAP_SECONDS = 0.35

    async def playback_loop():
        """
        Drip queued g711_ulaw frames to Twilio at ~20ms pace.
        """
        await asyncio.sleep(0.05)
        while playback_running and not stop_evt.is_set():
            if playback_queue and stream_sid:
                frame_b64 = playback_queue.popleft()
                try:
                    await ws.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": frame_b64},
                    })
                except Exception:
                    pass
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.005)

    async def cancel_ai(oai_ws):
        """
        Barge-in: stop AI immediately and clear any queued audio.
        """
        nonlocal ai_speaking, ai_busy
        playback_queue.clear()
        ai_speaking = False
        ai_busy = False
        try:
            await oai_ws.send_json({"type": "response.cancel"})
        except Exception:
            pass

    async def send_greeting_once(oai_ws):
        """
        Greet ONE TIME after Twilio start.
        """
        nonlocal greeted, ai_busy
        if greeted:
            return
        greeted = True
        ai_busy = True
        greeting = f"Hey, I'm {AGENT_NAME} with Riteway Landscape Products. How can I help you?"
        logging.info("👋 Sending greeting to OpenAI")
        await oai_ws.send_json({
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": greeting
            }
        })

    async def forward_twilio_to_openai(oai_ws):
        """
        Twilio -> OpenAI:
        - Twilio sends ulaw 8k base64
        - Convert to PCM16 24k base64 and append to OpenAI input buffer
        - If caller talks while AI talks => cancel AI (barge-in)
        """
        nonlocal stream_sid, caller_audio_since_commit_bytes, last_media_ts

        while not stop_evt.is_set():
            try:
                msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Twilio websocket disconnected")
                break
            except Exception:
                logging.exception("💥 Error receiving from Twilio WebSocket")
                break

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

                # tiny delay helps avoid greeting clipping on some setups
                await asyncio.sleep(0.15)
                await send_greeting_once(oai_ws)

            elif event == "media":
                payload_b64 = data["media"]["payload"]
                last_media_ts = time.time()

                # If caller starts speaking while AI is speaking -> barge-in cancel
                if ai_speaking or ai_busy:
                    await cancel_ai(oai_ws)

                # Twilio ulaw 8k -> PCM16 8k
                try:
                    ulaw_bytes = base64.b64decode(payload_b64)
                    pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)
                except Exception:
                    continue

                # 8k -> 24k PCM16
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)

                # track bytes to know when we have enough to commit
                caller_audio_since_commit_bytes += len(pcm16_24k)

                # send to OpenAI input buffer
                try:
                    b64_for_openai = base64.b64encode(pcm16_24k).decode("ascii")
                    await oai_ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": b64_for_openai,
                    })
                except Exception:
                    logging.exception("💥 Error sending audio to OpenAI")

            elif event == "stop":
                logging.info("📴 Twilio sent stop (caller hung up)")
                stop_evt.set()
                break

        logging.info("🚪 forward_twilio_to_openai exiting")

    async def forward_openai_to_twilio(oai_ws):
        """
        OpenAI -> Twilio:
        - response.audio.delta contains g711_ulaw base64 (because we request that output format)
        """
        nonlocal ai_speaking, ai_busy

        try:
            async for raw in oai_ws:
                if stop_evt.is_set():
                    break
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")

                # Useful events for debugging
                if oai_type in ("error", "response.done", "response.audio.delta"):
                    logging.info(f"🤖 OAI event: {oai_type}")

                if oai_type == "response.audio.delta":
                    delta_b64 = data.get("delta")
                    if delta_b64:
                        ai_speaking = True
                        playback_queue.append(delta_b64)

                elif oai_type in ("response.done", "response.completed"):
                    ai_speaking = False
                    ai_busy = False

                elif oai_type == "response.interrupted":
                    ai_speaking = False
                    ai_busy = False
                    playback_queue.clear()

                elif oai_type == "error":
                    logging.error(f"❌ OpenAI error event: {data}")
                    ai_speaking = False
                    ai_busy = False

        except Exception:
            logging.exception("💥 Error while reading from OpenAI ws")

        logging.info("🚪 forward_openai_to_twilio exiting")
        stop_evt.set()

    async def speech_drive_loop(oai_ws):
        """
        Decide WHEN to ask the AI to respond:
        - only after caller has spoken enough audio (>=100ms)
        - and only after a real pause (silence gap)
        - prevents AI responding while you keep talking
        """
        nonlocal caller_audio_since_commit_bytes, ai_busy, ai_speaking

        while not stop_evt.is_set():
            await asyncio.sleep(0.10)

            # don't create a new response if AI is still talking / queued
            if ai_speaking or ai_busy:
                continue

            # need a pause
            if last_media_ts and (time.time() - last_media_ts) < SILENCE_GAP_SECONDS:
                continue

            # need enough audio to commit (prevents commit_empty)
            if caller_audio_since_commit_bytes < MIN_BYTES_FOR_COMMIT:
                continue

            try:
                ai_busy = True
                logging.info(f"🗣 committing caller audio bytes={caller_audio_since_commit_bytes}")
                await oai_ws.send_json({"type": "input_audio_buffer.commit"})
                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"]
                    }
                })
            except Exception:
                logging.exception("⚠ commit/response.create failed")
                ai_busy = False
            finally:
                caller_audio_since_commit_bytes = 0

    # CONNECT to OpenAI Realtime
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

                # Session behavior
                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "g711_ulaw",
                        "voice": "alloy",

                        # IMPORTANT:
                        # We set create_response=False so the model doesn't talk while you're still talking.
                        # We control turns with speech_drive_loop (silence gap + min audio).
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "silence_duration_ms": int(SILENCE_GAP_SECONDS * 1000),
                            "create_response": False,
                            "allow_agent_interrupt": True
                        },

                        "instructions": (
                            f"You are {AGENT_NAME}, the live phone assistant for Riteway Landscape Products.\n"
                            "Be warm, professional, confident, and efficient.\n"
                            "Very important: If the caller starts talking, stop speaking immediately.\n"
                            "Keep answers under 20–30 seconds unless giving prices or doing quick yardage math.\n"
                            "\n"
                            "Business:\n"
                            "- Riteway Landscape Products (bulk landscape materials)\n"
                            "- Open Mon–Fri 9am–5pm. No weekends/after-hours.\n"
                            "- Sell by the cubic yard; boulders by the ton.\n"
                            "\n"
                            "Pricing (always say per yard / per ton):\n"
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
                            "Delivery:\n"
                            "- Up to 16 yards per load\n"
                            "- $75 to Grantsville\n"
                            "- $115 to rest of Tooele Valley\n"
                            "- Outside valley: ask address, repeat it, say $7/mile from Grantsville yard; dispatch confirms total.\n"
                            "\n"
                            "Yardage math:\n"
                            "- 1 yard covers ~100 sq ft at ~3 inches\n"
                            "- yards = (L_ft * W_ft * (D_in/12)) / 27, round 1 decimal\n"
                        )
                    }
                })

                # Start tasks
                playback_task = asyncio.create_task(playback_loop())
                drive_task = asyncio.create_task(speech_drive_loop(oai_ws))

                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws),
                    forward_openai_to_twilio(oai_ws),
                )

                stop_evt.set()
                playback_task.cancel()
                drive_task.cancel()

        except Exception:
            logging.exception("❌ Failed to connect to OpenAI Realtime. Check API key/model access.")
            stop_evt.set()

    # Cleanup
    playback_running = False
    try:
        await ws.close()
    except Exception:
        pass

    logging.info("🔚 /media connection closed")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
