import os
import math
import json
import base64
import asyncio
import logging

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    Response,
    PlainTextResponse,
    JSONResponse,
)

###############################################################################
# CONFIG
###############################################################################

# PUBLIC_BASE_URL should be the HTTPS URL that Twilio can reach.
# This MUST match what you see in the browser for your Render app.
# Example: "https://riteway-ai-agent.onrender.com"
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://riteway-ai-agent.onrender.com"
).rstrip("/")

# We'll build the wss:// URL Twilio should use for streaming audio.
WS_MEDIA_URL = "wss://" + PUBLIC_BASE_URL.replace("https://", "").replace("http://", "") + "/media"

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Riteway Voice Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# HELPERS: audio generation + mu-law conversion
###############################################################################

def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
    """
    Convert 16-bit signed PCM samples (numpy int16) -> 8-bit μ-law bytes.
    This matches G.711 u-law which Twilio expects for playback on <Stream>. 
    (8 kHz, mono, μ-law) 
    """
    BIAS = 0x84
    CLIP = 32635
    out = bytearray()

    # ensure we're dealing with regular Python ints
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


def generate_beep_ulaw_chunks(duration_sec=1.0, freq_hz=440.0, sample_rate=8000):
    """
    Make about 1 second of "beeeeep" audio so we can prove we can talk back.
    Steps:
    1. Make a sine wave at 440 Hz, mono, 8kHz.
    2. Convert to μ-law bytes.
    3. Slice into 20ms frames (160 samples @ 8000 Hz).
    4. Base64 each frame so we can send them to Twilio.
    """
    total_samples = int(duration_sec * sample_rate)

    # sine wave, not too loud so it doesn't clip
    t = np.arange(total_samples) / sample_rate
    pcm16 = (10000 * np.sin(2 * math.pi * freq_hz * t)).astype(np.int16)

    # convert PCM16 -> μ-law bytes
    ulaw_bytes = pcm16_to_ulaw(pcm16)

    # 20ms of audio @8kHz = 160 samples (160 bytes after μ-law)
    frame_size = 160
    chunks_b64 = []
    for i in range(0, len(ulaw_bytes), frame_size):
        frame = ulaw_bytes[i:i+frame_size]
        if len(frame) == 0:
            continue
        b64_payload = base64.b64encode(frame).decode("ascii")
        chunks_b64.append(b64_payload)

    return chunks_b64  # list of base64-encoded 20ms frames


async def send_ulaw_chunks_to_twilio(ws: WebSocket, stream_sid: str, chunks_b64: list):
    """
    Send a sequence of μ-law frames back to Twilio in the JSON format Twilio
    expects. We'll send them with ~20ms spacing so they play like audio.

    Twilio expects:
    {
      "event": "media",
      "streamSid": "<sid we got from 'start'>",
      "media": { "payload": "<base64 ulaw frame>" }
    }
    """
    logging.info(f"🔊 sending {len(chunks_b64)} beep frames to Twilio for streamSid={stream_sid}")
    for frame_b64 in chunks_b64:
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": frame_b64
            }
        }

        await ws.send_json(msg)
        # sleep ~20ms so Twilio plays them in real-time order
        await asyncio.sleep(0.02)

    logging.info("🔊 finished sending beep frames")


###############################################################################
# ROUTES
###############################################################################

@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "ws_media_url": WS_MEDIA_URL})


@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    """
    Twilio sends us a POST here when someone calls the number.
    We respond with TwiML saying:
      1. Say a short message,
      2. Start <Stream> to our WebSocket at /media.
    """
    logging.info("☎ /voice got POST from Twilio")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while I connect you to the Riteway A I assistant.</Say>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""

    # Twilio wants XML content back. PlainTextResponse with media_type is fine.
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.get("/voice")
async def voice_get():
    """
    Optional: lets you hit /voice in your browser and see the TwiML.
    This helps confirm Render is returning valid XML.
    """
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while I connect you to the Riteway A I assistant.</Say>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media")
async def media(ws: WebSocket):
    """
    Twilio connects here over WebSocket after it gets our TwiML.

    Flow:
    - Twilio sends JSON messages:
        event: "start"  → includes streamSid
        event: "media"  → includes base64 μ-law audio from the caller (20ms chunks)
        event: "stop"   → call ended
    - We log everything.
    - As soon as we get "start", we generate a beep and send it back to Twilio
      using the matching streamSid. The caller should hear the beep.
    """

    await ws.accept()
    logging.info("📞 Twilio connected to /media WebSocket ✅")

    stream_sid = None
    beep_sent = False

    try:
        while True:
            try:
                raw_msg = await ws.receive_text()
            except WebSocketDisconnect:
                logging.info("❌ Caller hung up (WebSocketDisconnect)")
                break
            except Exception as e:
                logging.exception("💥 error receiving from Twilio WS")
                break

            # Parse Twilio JSON
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                logging.warning(f"⚠ got non-JSON WS frame from Twilio: {raw_msg}")
                continue

            event_type = data.get("event")
            logging.info(f"📞 Twilio event: {event_type}")

            if event_type == "start":
                # Twilio tells us the streamSid here. We MUST include this
                # when we send audio back.
                start_info = data.get("start", {})
                stream_sid = start_info.get("streamSid")
                call_sid = start_info.get("callSid")
                logging.info(f"📞 streamSid={stream_sid} callSid={call_sid}")

                # send test beep once after we know the streamSid
                if stream_sid and not beep_sent:
                    beep_sent = True
                    chunks = generate_beep_ulaw_chunks(duration_sec=1.0, freq_hz=440.0)
                    # fire-and-forget the beep so we don't block receiving
                    asyncio.create_task(send_ulaw_chunks_to_twilio(ws, stream_sid, chunks))

            elif event_type == "media":
                # Caller voice frames arrive here.
                payload_b64 = data.get("media", {}).get("payload", "")
                logging.info(f"🎤 Caller audio frame base64 len={len(payload_b64)}")

                # Later: we'll forward this audio to OpenAI Realtime.
                # Right now we just log it.

            elif event_type == "stop":
                logging.info("📞 Twilio says stop (caller hung up normally)")
                break

            elif event_type == "mark":
                # 'mark' is Twilio acknowledging audio we sent finished playing
                logging.info(f"📍 Twilio mark: {data}")

            else:
                logging.info(f"ℹ other Twilio event payload: {data}")

    except Exception:
        logging.exception("💥 Fatal error inside /media loop")

    finally:
        # close socket nicely
        try:
            await ws.close()
        except Exception:
            pass

        logging.info("🔚 /media websocket closed")


###############################################################################
# LOCAL DEV ENTRYPOINT (for running python app.py locally if you want)
###############################################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
