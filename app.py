import os
import asyncio
import logging
import json
import base64
import math
import numpy as np
from aiohttp import web, WSMsgType

#############################################################################
# CONFIG
#############################################################################

# 🔐 This MUST be your live Render URL (the https one you open in a browser).
# I'm filling in what you told me. Change it if your URL is different.
PUBLIC_BASE_URL = "https://riteway-ai-agent.onrender.com"

# Twilio will call /voice (HTTP) first.
# /voice will respond with <Connect><Stream url="wss://.../media" />
# That tells Twilio "open a live audio WebSocket to /media".
#
# Twilio sends caller audio to /media as base64 μ-law 8kHz chunks.
# We are going to send audio BACK on that same socket in the SAME format.
#
# IMPORTANT: Twilio requires audio you send back to be:
#   - mu-law (a.k.a. μ-law / G.711 u-law)
#   - 8000 Hz sample rate
#   - base64 encoded
#   - wrapped in { "event":"media", "streamSid":"...", "media":{"payload":"..."} }
# or it won't play to the caller. :contentReference[oaicite:1]{index=1}
#
# We'll generate a 1-second "beep" tone in that format and stream it back.
# If you hear the beep when you call, we know outbound audio works. Then we add AI.


#############################################################################
# LITTLE HELPERS
#############################################################################

def http_to_ws_url(base_url: str) -> str:
    """
    Convert your https://something.onrender.com
    into wss://something.onrender.com/media
    (Twilio needs wss:// for the live audio stream)
    """
    base_url = base_url.rstrip("/")
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://"):]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://"):]
    else:
        # fallback, assume secure
        ws_base = "wss://" + base_url
    return ws_base + "/media"

WS_MEDIA_URL = http_to_ws_url(PUBLIC_BASE_URL)


def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
    """
    Convert raw 16-bit PCM samples (int16) to 8-bit μ-law bytes (G.711 u-law).
    This is the classic phone codec Twilio expects. 8 kHz μ-law mono. :contentReference[oaicite:2]{index=2}
    """
    BIAS = 0x84
    CLIP = 32635
    ulaw_bytes = bytearray()

    # make sure we are working with Python ints, not numpy scalar overflow weirdness
    for sample in pcm16.astype(np.int32):
        # Get sign bit
        sign = 0x80 if sample < 0 else 0x00
        if sample < 0:
            sample = -sample

        # clip
        if sample > CLIP:
            sample = CLIP

        # apply bias
        sample = sample + BIAS

        # figure out exponent
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (sample & mask):
            mask >>= 1
            exponent -= 1

        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        ulaw_bytes.append(ulaw_byte)

    return bytes(ulaw_bytes)


def generate_beep_ulaw_chunks(duration_sec: float = 1.0,
                              freq_hz: float = 440.0,
                              sample_rate: int = 8000):
    """
    Make a simple test tone (beeeeep) so we can prove audio is going OUT.
    Steps:
      1. Generate a sine wave at 440 Hz (A tone).
      2. 8kHz sample rate so it's phone-quality.
      3. Convert to μ-law.
      4. Split into 20ms chunks (160 samples @ 8000 Hz).
      5. Base64 each chunk so Twilio can play it.
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.arange(total_samples) / sample_rate
    # make a sine wave, not too loud (10000 out of int16 max ~32767)
    pcm16 = (10000 * np.sin(2 * math.pi * freq_hz * t)).astype(np.int16)

    ulaw_bytes = pcm16_to_ulaw(pcm16)

    frame_size = 160  # 20ms of audio @8kHz = 160 samples
    chunks_b64 = []
    for i in range(0, len(ulaw_bytes), frame_size):
        frame = ulaw_bytes[i:i+frame_size]
        b64_payload = base64.b64encode(frame).decode("ascii")
        chunks_b64.append(b64_payload)

    return chunks_b64  # list of base64 strings


#############################################################################
# AIOHTTP APP + ROUTES
#############################################################################

logging.basicConfig(level=logging.INFO)
routes = web.RouteTableDef()


@routes.post("/voice")
async def voice_handler(request: web.Request):
    """
    Twilio hits this FIRST on every inbound call.
    We answer with TwiML.
    TwiML tells Twilio:
      - say a quick message to the caller
      - then <Connect><Stream> audio to our /media websocket
    """

    logging.info("☎ /voice got hit by Twilio")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while I connect you to the Riteway A I assistant.</Say>
  <Connect>
    <Stream url="{WS_MEDIA_URL}" />
  </Connect>
</Response>"""

    # Send TwiML back
    return web.Response(text=twiml, content_type="text/xml")


@routes.get("/media")
async def media_ws(request: web.Request):
    """
    Twilio opens a secure WebSocket (wss://...) here.
    After that:
      - Twilio sends us JSON messages:
          { "event": "start", ... }
          { "event": "media", "media": {"payload": "..."} }
          { "event": "stop", ... }
        "media.payload" is the caller's live mic audio in μ-law 8kHz, base64. :contentReference[oaicite:3]{index=3}

      - We can send audio BACK by sending JSON:
          {
            "event": "media",
            "streamSid": "theSidFromStart",
            "media": { "payload": "base64-ULAW-audio" }
          }
        Twilio will play that audio to the caller in real time. :contentReference[oaicite:4]{index=4}
    """

    logging.info("📞 Twilio is opening the /media websocket")

    # Accept the WebSocket upgrade
    twilio_ws = web.WebSocketResponse()
    await twilio_ws.prepare(request)
    logging.info("📞 Twilio WebSocket handshake complete ✅")

    stream_sid_holder = {"sid": None}
    played_test_beep = asyncio.Event()

    async def send_audio_frame(b64_payload: str):
        """
        Send one frame of μ-law audio back to Twilio so the caller hears it.
        """
        if stream_sid_holder["sid"] is None:
            logging.warning("⚠ Tried to send audio frame but we don't have streamSid yet.")
            return

        msg = {
            "event": "media",
            "streamSid": stream_sid_holder["sid"],
            "media": {
                "payload": b64_payload
            }
        }

        await twilio_ws.send_json(msg)

    async def play_test_beep_to_caller():
        """
        Send ~1 second of beep frames after the call starts.
        This proves outbound audio works.
        """
        chunks = generate_beep_ulaw_chunks(duration_sec=1.0, freq_hz=440.0)
        logging.info(f"🔊 Sending {len(chunks)} beep frames back to Twilio...")

        # Send frames at ~20ms spacing so it plays smoothly
        for b64_payload in chunks:
            await send_audio_frame(b64_payload)
            await asyncio.sleep(0.02)

        logging.info("🔊 Finished sending test beep frames (caller should have heard a tone).")
        played_test_beep.set()

    async def handle_twilio_messages():
        """
        Main loop: read every message from Twilio until it says "stop".
        """
        async for ws_msg in twilio_ws:
            if ws_msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(ws_msg.data)
                except json.JSONDecodeError:
                    logging.error("❌ got non-JSON text from Twilio: %s", ws_msg.data)
                    continue

                event_type = data.get("event")
                logging.info("📞 Twilio event: %s", event_type)

                if event_type == "start":
                    # Save the streamSid. We must include this when we send audio back.
                    stream_sid_holder["sid"] = data["start"]["streamSid"]
                    logging.info("📞 streamSid = %s", stream_sid_holder["sid"])

                    # Kick off the beep (only once)
                    if not played_test_beep.is_set():
                        asyncio.create_task(play_test_beep_to_caller())

                elif event_type == "media":
                    # Twilio is giving us the caller's voice audio here.
                    payload = data.get("media", {}).get("payload", "")
                    logging.info("🎤 Caller audio frame size (base64 chars)=%d", len(payload))

                    # Later: we'll forward this to OpenAI Realtime so AI can listen & talk back.
                    # For now: just logging.

                elif event_type == "stop":
                    logging.info("📞 Twilio says stop (caller hung up)")
                    break

                elif event_type == "mark":
                    # Mark events are Twilio acknowledging audio we sent finished playing.
                    logging.info("📍 Twilio mark: %s", data.get("mark"))

                else:
                    logging.info("ℹ other Twilio event payload: %s", data)

            elif ws_msg.type == WSMsgType.ERROR:
                logging.error("❌ Twilio ws error: %s", twilio_ws.exception())
                break

        logging.info("👋 handle_twilio_messages finished")

    # run the Twilio read loop
    await handle_twilio_messages()

    # cleanup
    await twilio_ws.close()
    logging.info("🔚 /media websocket closed cleanly")
    return twilio_ws


#############################################################################
# START THE SERVER ON RENDER
#############################################################################

app = web.Application()
app.add_routes(routes)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    logging.info(f"🚀 Starting Riteway voice bridge on 0.0.0.0:{port}")
    web.run_app(app, host="0.0.0.0", port=port)
