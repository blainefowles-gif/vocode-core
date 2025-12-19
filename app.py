import os
import math
import json
import base64
import asyncio
import logging
import audioop
import time
import numpy as np
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
# AUDIO HELPERS
###############################################################################

def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
    """
    Convert PCM16 numpy array -> G.711 μ-law bytes.
    Twilio expects 8kHz μ-law, 20ms frames (160 bytes).
    """
    BIAS = 0x84
    CLIP = 32635
    out = bytearray()
    for s in pcm16.astype(np.int32):
        sign = 0x80 if s < 0 else 0x00
        if s < 0:
            s = -s
        if s > CLIP:
            s = CLIP
        s = s + BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (s & mask):
            mask >>= 1
            exponent -= 1
        mantissa = (s >> (exponent + 3)) & 0x0F
        ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        out.append(ulaw_byte)
    return bytes(out)

def generate_beep_ulaw_chunks(duration_sec=0.8, freq_hz=440.0, sample_rate=8000):
    """
    Safety tone, in case we ever need to send audio while AI is booting.
    We won't actively play this now unless something is badly wrong.
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.arange(total_samples) / sample_rate
    pcm16 = (10000 * np.sin(2 * math.pi * freq_hz * t)).astype(np.int16)
    ulaw_bytes = pcm16_to_ulaw(pcm16)

    frame_size = 160  # 20ms @ 8kHz
    chunks_b64 = []
    for i in range(0, len(ulaw_bytes), frame_size):
        frame = ulaw_bytes[i:i+frame_size]
        if not frame:
            continue
        b64_payload = base64.b64encode(frame).decode("ascii")
        chunks_b64.append(b64_payload)

    return chunks_b64

async def send_ulaw_chunks_to_twilio(ws: WebSocket, stream_sid: str, chunks_b64: list):
    """
    Send pre-encoded μ-law 20ms frames to Twilio with ~20ms pacing.
    """
    logging.info(f"🔊 sending {len(chunks_b64)} fallback frames to Twilio (sid={stream_sid})")
    for frame_b64 in chunks_b64:
        await ws.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": frame_b64}
        })
        await asyncio.sleep(0.02)
    logging.info("🔊 finished sending fallback frames")

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
    logging.info("☎ Twilio hit /voice")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media")

    stream_sid = None
    openai_connected = False
    oai_ws_handle = None

    playback_queue = deque()
    playback_running = True
    playback_task = None

    ai_speaking = False

    # -------------------------
    # BARGE-IN (FAST + ROBUST)
    # -------------------------
    # Key fixes for the “~3 second delay”:
    # 1) Send Twilio "clear" immediately on barge-in (flushes Twilio jitter/playout buffer)
    # 2) Use an adaptive noise floor (so we ignore background noise but trigger quickly on real speech)
    # 3) Lower required frames by default + add a fast-trigger for obvious speech
    #
    # You can still override any of these with Render env vars.
    BARGE_RMS_MIN = int(os.getenv("BARGE_RMS_MIN", "650"))                 # absolute floor
    BARGE_NOISE_MULT = float(os.getenv("BARGE_NOISE_MULT", "3.0"))         # noise * mult -> threshold
    BARGE_FRAMES_REQUIRED = int(os.getenv("BARGE_FRAMES_REQUIRED", "2"))   # 2 frames ~ 40ms
    BARGE_PEAK_MIN = int(os.getenv("BARGE_PEAK_MIN", "2200"))              # ignore very soft hum
    BARGE_FAST_TRIGGER_RATIO = float(os.getenv("BARGE_FAST_TRIGGER_RATIO", "1.8"))  # instant if rms > thr*ratio
    BARGE_CANCEL_COOLDOWN_MS = int(os.getenv("BARGE_CANCEL_COOLDOWN_MS", "120"))

    # Optional ZCR gate (off by default; ZCR can be device-dependent)
    BARGE_USE_ZCR = os.getenv("BARGE_USE_ZCR", "0") == "1"
    BARGE_ZCR_MIN = float(os.getenv("BARGE_ZCR_MIN", "0.02"))
    BARGE_ZCR_MAX = float(os.getenv("BARGE_ZCR_MAX", "0.22"))

    barge_speech_frames = 0
    last_cancel_ts = 0.0

    # Adaptive noise floor (EMA)
    noise_rms_ema = float(os.getenv("NOISE_RMS_INIT", "200"))  # starting guess
    NOISE_EMA_ALPHA = float(os.getenv("NOISE_EMA_ALPHA", "0.05"))
    NOISE_UPDATE_RMS_CAP = int(os.getenv("NOISE_UPDATE_RMS_CAP", "900"))   # don't learn on loud speech
    NOISE_UPDATE_WHEN_AI_SPEAKING = os.getenv("NOISE_UPDATE_WHEN_AI_SPEAKING", "0") == "1"

    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY in environment.")
    else:
        logging.info("🔑 OPENAI_API_KEY is present")

    async def playback_loop():
        await asyncio.sleep(0.1)
        nonlocal playback_running, ai_speaking
        while playback_running:
            if playback_queue and stream_sid:
                frame_b64 = playback_queue.popleft()
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": frame_b64},
                })
                await asyncio.sleep(0.02)
            else:
                ai_speaking = False
                await asyncio.sleep(0.005)

    async def twilio_clear_audio():
        """
        Ask Twilio to immediately flush any buffered outbound audio.
        This is what removes that “keeps talking for seconds” feeling.
        """
        if not stream_sid:
            return
        try:
            await ws.send_json({"event": "clear", "streamSid": stream_sid})
        except Exception:
            pass

    async def cancel_openai_response_if_real_speech(ulaw_bytes: bytes):
        nonlocal ai_speaking, barge_speech_frames, last_cancel_ts, noise_rms_ema

        # Always compute RMS/Peak so we can keep learning the noise floor (when appropriate)
        try:
            pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)
            rms = audioop.rms(pcm16_8k, 2)
            peak = audioop.max(pcm16_8k, 2)
        except Exception:
            barge_speech_frames = 0
            return

        # Update noise floor when not speaking (or if you explicitly allow learning during AI speech)
        can_learn_noise = (not ai_speaking) or NOISE_UPDATE_WHEN_AI_SPEAKING
        if can_learn_noise and rms <= NOISE_UPDATE_RMS_CAP:
            noise_rms_ema = (1.0 - NOISE_EMA_ALPHA) * noise_rms_ema + NOISE_EMA_ALPHA * float(rms)

        # If AI isn't speaking, don't barge-cancel anything.
        if not oai_ws_handle or not ai_speaking:
            barge_speech_frames = 0
            return

        now = time.time()
        if (now - last_cancel_ts) * 1000 < BARGE_CANCEL_COOLDOWN_MS:
            return

        # Dynamic threshold derived from noise floor + absolute floor
        dyn_thr = max(BARGE_RMS_MIN, int(noise_rms_ema * BARGE_NOISE_MULT))

        # Optional ZCR test (off by default)
        zcr_ok = True
        zcr_val = None
        if BARGE_USE_ZCR:
            try:
                samples = np.frombuffer(pcm16_8k, dtype=np.int16)
                if samples.size >= 2:
                    signs = np.sign(samples)
                    zcr_val = float(np.mean(signs[1:] != signs[:-1]))
                    zcr_ok = (BARGE_ZCR_MIN <= zcr_val <= BARGE_ZCR_MAX)
            except Exception:
                zcr_ok = False

        speech_like = (rms >= dyn_thr) and (peak >= BARGE_PEAK_MIN) and zcr_ok

        # Fast trigger: if it's clearly above threshold, cancel immediately (no multi-frame wait)
        fast_trigger = (rms >= int(dyn_thr * BARGE_FAST_TRIGGER_RATIO)) and (peak >= int(BARGE_PEAK_MIN * 1.1))

        if fast_trigger:
            barge_speech_frames = BARGE_FRAMES_REQUIRED
        elif speech_like:
            barge_speech_frames += 1
        else:
            barge_speech_frames = max(0, barge_speech_frames - 1)

        if barge_speech_frames < BARGE_FRAMES_REQUIRED:
            return

        # Confirmed barge-in: stop audio NOW
        barge_speech_frames = 0
        playback_queue.clear()
        ai_speaking = False
        last_cancel_ts = now

        # Critical: flush Twilio playout buffer
        await twilio_clear_audio()

        # Stop OpenAI response generation
        try:
            await oai_ws_handle.send_json({"type": "response.cancel"})
        except Exception:
            pass

        logging.info(
            f"🛑 BARGE-IN: cancel (rms={rms}, peak={peak}, dyn_thr={dyn_thr}, "
            f"noise_ema={noise_rms_ema:.1f}, frames={BARGE_FRAMES_REQUIRED}, "
            f"fast={fast_trigger}, zcr={'on' if BARGE_USE_ZCR else 'off'}"
            f"{'' if zcr_val is None else f', zcr={zcr_val:.3f}'}"
            f")"
        )

    async def forward_twilio_to_openai(oai_ws):
        nonlocal stream_sid
        nonlocal openai_connected
        nonlocal oai_ws_handle

        while True:
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
                logging.warning(f"⚠ got non-JSON from Twilio: {msg}")
                continue

            event = data.get("event")

            if event == "connected":
                logging.info(f"ℹ Twilio event connected: {data}")

            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                logging.info(f"📞 Twilio start: streamSid={stream_sid} callSid={call_sid}")

            elif event == "media":
                payload_b64 = data["media"]["payload"]
                ulaw_bytes = base64.b64decode(payload_b64)

                await cancel_openai_response_if_real_speech(ulaw_bytes)

                pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)
                pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 24000, None)

                if oai_ws and openai_connected:
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
                break

            else:
                logging.info(f"ℹ Twilio event {event}: {data}")

        logging.info("🚪 forward_twilio_to_openai exiting")

    async def forward_openai_to_twilio(oai_ws):
        nonlocal openai_connected
        nonlocal ai_speaking

        try:
            async for raw in oai_ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(raw.data)
                oai_type = data.get("type")
                logging.info(f"🤖 OAI event: {oai_type}")

                openai_connected = True

                if oai_type == "response.audio.delta":
                    ulaw_chunk_b64 = data.get("delta")
                    if ulaw_chunk_b64:
                        ai_speaking = True
                        playback_queue.append(ulaw_chunk_b64)

                elif oai_type in ("response.done", "response.completed"):
                    ai_speaking = False
                    logging.info("✅ AI finished a spoken response")

                elif oai_type == "response.interrupted":
                    ai_speaking = False
                    playback_queue.clear()
                    logging.info("🛑 AI response interrupted")

                elif oai_type == "error":
                    ai_speaking = False
                    logging.error(f"❌ OpenAI internal error event: {data}")

        except Exception:
            logging.exception("💥 Error while reading from OpenAI ws (forward_openai_to_twilio)")
        logging.info("🚪 forward_openai_to_twilio exiting")

    if not OPENAI_API_KEY:
        logging.error("❌ OPENAI_API_KEY missing. Skipping OpenAI connect.")
        await forward_twilio_to_openai(None)
        logging.info("🔚 /media connection closed (no OpenAI)")
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
                openai_connected = True
                oai_ws_handle = oai_ws

                await oai_ws.send_json({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "g711_ulaw",
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "silence_duration_ms": 300,
                            "create_response": True
                        },
                        "instructions": (
                            "LANGUAGE:\n"
                            "- Speak ENGLISH only. Do not switch languages.\n\n"

                            "ROLE:\n"
                            "You are Tammy, the live phone receptionist for Riteway Landscape Products.\n"
                            "You are speaking to callers on the phone.\n"
                            "Never talk as if you are the caller.\n"
                            "If you greet, greet the caller (do not greet Tammy).\n\n"

                            "SCOPE (IMPORTANT):\n"
                            "- You are NOT a general purpose assistant.\n"
                            "- ONLY answer questions related to Riteway Landscape Products and landscape materials.\n"
                            "- Allowed topics: product/pricing, delivery areas/fees, scheduling within business hours,\n"
                            "  loading limits (16 yards), and yardage/material estimates for landscaping.\n"
                            "- If asked about anything else (history, trivia, politics, random questions, etc.),\n"
                            "  politely refuse and redirect back to landscape materials.\n"
                            "  Example refusal: \"I can only help with Riteway landscape materials, pricing, delivery, or yardage estimates. What material are you working with?\"\n\n"

                            "STYLE:\n"
                            "- Speak warm, professional, confident, and efficient.\n"
                            "- Keep each answer under 30 seconds.\n"
                            "- If the caller starts talking, stop and let them finish.\n"
                            "- When doing calculations, do NOT explain your steps or reasoning.\n"
                            "  Give ONLY the final result and one short helpful sentence.\n\n"

                            "BUSINESS INFO:\n"
                            "- Business: Riteway Landscape Products.\n"
                            "- We sell bulk landscape material by the cubic yard.\n"
                            "- We are open Monday–Friday, 9 AM to 5 PM. No after-hours or weekend scheduling.\n"
                            "- We mainly serve Tooele Valley and surrounding areas.\n\n"

                            "PRICING (ALWAYS say 'per yard' or 'per ton'):\n"
                            "- Washed Pea Gravel: $42 per yard.\n"
                            "- Desert Sun 7/8\" Crushed Rock: $40 per yard.\n"
                            "- 7/8\" Crushed Rock: $25 per yard.\n"
                            "- Desert Sun 1.5\" Crushed Rock: $40 per yard.\n"
                            "- 1.5\" Crushed Rock: $25 per yard.\n"
                            "- Commercial Road Base: $20 per yard.\n"
                            "- 3/8\" Minus Fines: $12 per yard.\n"
                            "- Desert Sun 1–3\" Cobble: $40 per yard.\n"
                            "- 8\" Landscape Cobble: $40 per yard.\n"
                            "- Desert Sun Boulders: $75 per ton.\n"
                            "- Fill Dirt: $12 per yard.\n"
                            "- Top Soil: $26 per yard.\n"
                            "- Screened Premium Top Soil: $40 per yard.\n"
                            "- Washed Sand: $65 per yard.\n"
                            "- Premium Mulch: $44 per yard.\n"
                            "- Colored Shredded Bark: $76 per yard.\n\n"

                            "DELIVERY:\n"
                            "- Up to 16 yards per load.\n"
                            "- $75 to Grantsville.\n"
                            "- $115 to rest of Tooele Valley.\n"
                            "- Outside valley: ask address, repeat it, say $7/mile from Grantsville yard; dispatch confirms total.\n\n"

                            "YARDAGE ESTIMATES (no step-by-step):\n"
                            "- Use: yards = (L_ft * W_ft * (D_in/12)) / 27, rounded to 1 decimal.\n"
                            "- Also: 1 yard covers ~100 sq ft at ~3 inches.\n"
                        )
                    }
                })

                # Wait briefly for session.updated before greeting
                session_updated = False
                start = time.time()
                while time.time() - start < 2.0:
                    try:
                        msg = await oai_ws.receive(timeout=2.0)
                    except Exception:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    t = data.get("type")
                    logging.info(f"🤖 OAI event (pre-loop): {t}")
                    if t == "session.updated":
                        session_updated = True
                        break
                if not session_updated:
                    logging.warning("⚠ Did not confirm session.updated before greeting (continuing anyway).")

                await oai_ws.send_json({
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        "instructions": (
                            "You are speaking to the caller. "
                            "Say EXACTLY and ONLY this sentence, then stop: "
                            "Hey, I’m Tammy with Riteway Landscape Products. How can I help you?"
                        )
                    }
                })

                playback_task = asyncio.create_task(playback_loop())

                await asyncio.gather(
                    forward_twilio_to_openai(oai_ws_handle),
                    forward_openai_to_twilio(oai_ws_handle),
                )

        except Exception:
            logging.exception(
                "❌ Failed to connect to OpenAI Realtime! "
                "Possible causes: invalid OPENAI_API_KEY or missing model access."
            )
            await forward_twilio_to_openai(None)

    playback_running = False
    if playback_task:
        await asyncio.sleep(0.1)
        playback_task.cancel()

    logging.info("🔚 /media connection closed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
