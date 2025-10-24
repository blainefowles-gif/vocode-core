import os
import json
import base64
import audioop
import asyncio
import traceback
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse, JSONResponse
import aiohttp

###############################################################################
# 0. SETUP
###############################################################################

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

app = FastAPI(title="Riteway AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# 1. SIMPLE HEALTH CHECK
###############################################################################

@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "realtime_model": REALTIME_MODEL})


###############################################################################
# 2. TWILIO VOICE WEBHOOK (/voice)
#
# Twilio hits this FIRST, every time someone calls your Twilio number.
# We respond with TwiML telling Twilio:
#   "create a live audio stream to wss://.../media"
###############################################################################

@app.post("/voice", response_class=PlainTextResponse)
async def voice(_: Request):
    logging.info("📞 /voice POST hit by Twilio")

    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""

    # Twilio wants XML back. 200 OK + this TwiML = good.
    return PlainTextResponse(content=twiml, media_type="application/xml")


# Bonus: so you can visit /voice in the browser to confirm it's returning XML
@app.get("/voice")
async def voice_get():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://riteway-ai-agent.onrender.com/media" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


###############################################################################
# 3. AUDIO CONVERSION
#
# Twilio sends audio chunks as base64 μ-law (g711_ulaw) at 8kHz mono, 20ms each.
# OpenAI Realtime expects PCM16 at 24kHz mono.
#
# We:
#  - decode Twilio μ-law b64 → raw μ-law bytes
#  - convert μ-law 8kHz → PCM16 8kHz
#  - resample PCM16 8kHz → PCM16 24kHz
#  - base64 that, send to OpenAI
###############################################################################

def ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64: str) -> str:
    # 1) base64 decode Twilio's payload
    ulaw_bytes = base64.b64decode(ulaw_b64)  # 8kHz, 8-bit μ-law mono

    # 2) convert μ-law -> PCM16 16-bit @ 8kHz
    pcm16_8k = audioop.ulaw2lin(ulaw_bytes, 2)  # width=2 bytes/sample

    # 3) resample 8kHz -> 24kHz
    pcm16_24k, _ = audioop.ratecv(
        pcm16_8k,        # audio data
        2,               # width=2 bytes/sample (16-bit)
        1,               # channels=1 (mono)
        8000,            # from 8kHz
        24000,           # to 24kHz
        None
    )

    # 4) return as base64 because OpenAI wants base64 PCM chunks
    return base64.b64encode(pcm16_24k).decode("ascii")


###############################################################################
# 4. /media WEBSOCKET
#
# Twilio opens a live WebSocket to this endpoint to send us audio in real time.
# We must:
#   - accept Twilio's WS
#   - open OpenAI Realtime WS
#   - forward caller audio -> OpenAI
#   - forward AI audio -> caller
#
# Notes:
#  - We use "server_vad" (voice activity detection) so we DON'T have to manually
#    commit audio buffers. OpenAI will figure out "the caller finished talking"
#    and then auto-generate a response.
#
#  - We ALWAYS send streamSid when we send audio back to Twilio, or Twilio won't play it.
###############################################################################

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    logging.info("✅ Twilio connected to /media WebSocket")

    if not OPENAI_API_KEY:
        logging.error("❌ No OPENAI_API_KEY set in environment")
        # We still keep the socket alive but we can't talk back.
        # The call will be silent, but won't crash.
        try:
            while True:
                msg_text = await ws.receive_text()
                data = json.loads(msg_text)
                logging.info(f"📞 Twilio event (no AI mode): {data.get('event')}")
                if data.get("event") == "stop":
                    logging.info("Caller hung up (stop)")
                    break
        except WebSocketDisconnect:
            logging.info("❌ Caller disconnected WebSocket (no AI mode)")
        except Exception:
            logging.exception("💥 error in /media (no AI mode)")
        finally:
            try:
                await ws.close()
            except Exception:
                pass
        return

    # If we DO have an API key, spin up OpenAI Realtime
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    # We'll share these between tasks
    twilio_stream_sid = {"value": None}
    websocket_open = True  # to help both loops know when to stop

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            headers=headers,
            autoping=True,
            heartbeat=20,
            timeout=0
        ) as oai_ws:

            # Tell OpenAI how we want audio in/out
            # - input: PCM16 24kHz, base64 chunks
            # - output: μ-law 8kHz so Twilio can play it directly
            # - turn_detection: server_vad so OpenAI auto-commits and auto-responds
            await oai_ws.send_json({
                "type": "session.update",
                "session": {
                    "modalities": ["audio"],
                    "input_audio_format":  {"type": "pcm16", "sample_rate_hz": 24000},
                    "output_audio_format": {"type": "g711_ulaw", "sample_rate_hz": 8000},
                    "voice": "alloy",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 200,
                        "create_response": True
                    },
                    "instructions": (
                        "You are the phone receptionist for Riteway Landscape Materials "
                        "in Grantsville, Utah. Be brief, friendly, and professional. "
                        "Collect: caller name, callback number, delivery address, material type, "
                        "quantity (yards or tons), and preferred delivery window. "
                        "If unsure on pricing, give a range and say you will text "
                        "a confirmed quote."
                    ),
                }
            })

            # Send initial greeting so the caller hears us right away
            await oai_ws.send_json({
                "type": "response.create",
                "response": {
                    "instructions": "Hi! This is Riteway. How can I help you today?"
                }
            })

            async def pump_twilio_to_openai():
                """
                Read audio events from Twilio WebSocket (ws),
                convert to PCM16/24k, append to OpenAI's input buffer.
                """
                nonlocal websocket_open
                while websocket_open:
                    try:
                        msg_text = await ws.receive_text()
                    except WebSocketDisconnect:
                        logging.info("❌ Twilio WebSocket disconnected (caller hung up)")
                        websocket_open = False
                        break
                    except Exception:
                        logging.exception("💥 Error receiving from Twilio WebSocket")
                        websocket_open = False
                        break

                    # Parse Twilio event
                    try:
                        data = json.loads(msg_text)
                    except json.JSONDecodeError:
                        logging.warning(f"Got non-JSON from Twilio: {msg_text}")
                        continue

                    event_type = data.get("event")
                    logging.info(f"📞 Twilio event: {event_type}")

                    if event_type == "start":
                        # Save streamSid - we need to echo it back when we send audio TO Twilio
                        sid = data.get("streamSid") or data.get("start", {}).get("streamSid")
                        if sid:
                            twilio_stream_sid["value"] = sid
                            logging.info(f"Twilio streamSid set: {sid}")

                    elif event_type == "media":
                        # Caller audio chunk
                        ulaw_b64 = data.get("media", {}).get("payload")
                        if ulaw_b64:
                            # Convert to PCM16@24k -> base64
                            pcm24_b64 = ulaw8k_b64_to_pcm16_24k_b64(ulaw_b64)
                            # Send to OpenAI
                            try:
                                await oai_ws.send_json({
                                    "type": "input_audio_buffer.append",
                                    "audio": pcm24_b64
                                })
                            except Exception:
                                logging.exception("💥 Error sending audio chunk to OpenAI")

                    elif event_type == "stop":
                        logging.info("📞 Twilio says stop (caller hung up)")
                        websocket_open = False
                        break

                    else:
                        # connected / mark / etc.
                        pass

                # End while loop
                logging.info("pump_twilio_to_openai finished")

            async def pump_openai_to_twilio():
                """
                Read AI responses from OpenAI Realtime WS.
                When we get audio delta, stream it back to Twilio so caller hears it.
                """
                nonlocal websocket_open
                async for msg in oai_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            payload = json.loads(msg.data)
                        except json.JSONDecodeError:
                            logging.warning(f"Non-JSON from OpenAI: {msg.data}")
                            continue

                        typ = payload.get("type", "")
                        logging.info(f"OAI event: {typ}")

                        if typ == "response.audio.delta":
                            # OpenAI is giving us audio in μ-law 8kHz base64 (g711_ulaw)
                            # We forward this straight to Twilio.
                            chunk_b64 = payload.get("delta")
                            sid_val = twilio_stream_sid["value"]
                            if chunk_b64 and sid_val:
                                try:
                                    await ws.send_json({
                                        "event": "media",
                                        "streamSid": sid_val,
                                        "media": {
                                            "payload": chunk_b64
                                        }
                                    })
                                except Exception:
                                    logging.exception("💥 Error sending audio back to Twilio")

                        elif typ == "response.done":
                            # end of one spoken response
                            pass

                        elif typ == "error":
                            logging.error(f"OpenAI error: {payload}")

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logging.error("❌ OpenAI WSMsgType.ERROR")
                        break

                # OpenAI socket ended
                logging.info("pump_openai_to_twilio finished")
                websocket_open = False

            # Run both directions at once
            try:
                await asyncio.gather(
                    pump_twilio_to_openai(),
                    pump_openai_to_twilio()
                )
            except Exception:
                logging.exception("💥 Fatal error in gather Twilio<->OpenAI bridge")

    # Clean up Twilio socket
    try:
        await ws.close()
    except Exception:
        pass

    logging.info("🔚 /media websocket closed cleanly")


###############################################################################
# 5. LOCAL DEV ENTRYPOINT (optional)
# Render ignores this because we tell Render to run uvicorn manually.
###############################################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
