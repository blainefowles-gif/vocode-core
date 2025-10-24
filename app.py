import os
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

###############################################################################
# 0. BASIC SETUP
#
# - We create a FastAPI server.
# - We add CORS just in case (lets Twilio / browsers hit us).
# - We add logging so Render logs are useful.
###############################################################################

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow anything for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# 1. HEALTH CHECK
#
# Render pings this so it knows your service is alive.
# You can also hit this in your browser: 
# https://riteway-ai-agent.onrender.com/health
###############################################################################

@app.get("/health")
async def health():
    return {"status": "ok", "service": "riteway"}


###############################################################################
# 2. TWILIO VOICE WEBHOOK (/voice)
#
# Twilio will POST here when someone calls your Twilio number.
#
# We respond with TwiML (Twilio’s little XML language) that tells Twilio:
#   "connect the caller’s audio to my websocket at /media"
#
# IMPORTANT:
# - <Connect><Stream> = bidirectional audio. Twilio will try to open a
#   secure WebSocket (wss://...) back to /media below.
#
# - If /media crashes or refuses the WebSocket handshake, the caller hears
#   "An application error has occurred" and Twilio hangs up.
#
# So fixing /media stability fixes that message.
###############################################################################

@app.post("/voice", response_class=PlainTextResponse)
async def voice():
    logging.info("Twilio hit /voice webhook")

    # <<IMPORTANT>>
    # Twilio expects valid XML, content-type: application/xml.
    # FastAPI will send text, which Twilio accepts fine if it's valid TwiML.
    #
    # NOTE: While this stream is active Twilio does NOT say anything to the caller.
    # So the caller just hears silence right now.
    # That's normal for this step. We'll add talking later.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://riteway-ai-agent.onrender.com/media" />
    </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")


###############################################################################
# 3. MEDIA STREAM ENDPOINT (/media)
#
# This is the MOST important part.
#
# Twilio will try to open a secure WebSocket to:
#   wss://riteway-ai-agent.onrender.com/media
#
# Then Twilio will start sending JSON messages like:
#
#   {"event":"connected", ...}
#   {"event":"start", "streamSid":"XYZ", ...}
#   {"event":"media", "media":{"payload":"...base64 audio..."}, ... }
#   {"event":"stop", ...}
#
# What we do in this version:
#   - Accept the WebSocket.
#   - Listen and log what Twilio sends.
#   - Keep the socket open so Twilio stays happy.
#   - DO NOT crash. (Crashing = Twilio says "application error".)
#
# What we are NOT doing yet:
#   - Sending audio back to Twilio (so caller hears silence).
#   - Sending audio to OpenAI.
#
# Why?
#   Because first we need a rock-solid connection with ZERO crashes.
#   Once that's stable, we add the AI brain + voice on top.
###############################################################################

@app.websocket("/media")
async def media_stream(websocket: WebSocket):
    # Step 1. Accept the websocket upgrade from Twilio.
    #
    # If we FAIL here (throw an error before accept), Twilio's going to
    # immediately bail with error 31920 and your caller hears
    # "an application error has occurred".
    #
    await websocket.accept()
    logging.info("✅ Twilio connected to /media WebSocket")

    # We'll store the Twilio streamSid so we know which call we're dealing with.
    stream_sid = None

    try:
        # Step 2. Main receive loop.
        #
        # We sit in a while True and keep reading messages from Twilio.
        # Each message is text JSON.
        #
        while True:
            # Wait for Twilio to send us something.
            # If caller hangs up, this will raise WebSocketDisconnect and jump to `except` below.
            raw_msg = await websocket.receive_text()

            # Parse it so we can see what Twilio said.
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                logging.warning(f"Got non-JSON message from Twilio: {raw_msg}")
                continue

            event_type = data.get("event")
            logging.info(f"📞 Twilio event: {event_type}")

            # --- Handle specific events Twilio sends us ----------------------

            # 3a. "connected" happens when the media stream is first opened.
            if event_type == "connected":
                logging.info("Twilio says: connected")

                # We *could* reply to Twilio here in future with audio,
                # but for now we stay quiet. Silence = no crash.

            # 3b. "start" tells us the call officially started streaming.
            elif event_type == "start":
                stream_sid = data.get("streamSid")
                call_sid = data.get("start", {}).get("callSid")
                logging.info(f"Stream started. streamSid={stream_sid} callSid={call_sid}")

                # LATER:
                #   - Here is where we’ll tell OpenAI “the call began”.
                #   - We'll also send the first spoken line from Riteway to the caller.

            # 3c. "media" is the actual audio chunks from the caller.
            elif event_type == "media":
                # Twilio gives us 20ms chunks of the caller's audio
                # (base64-encoded 8kHz mu-law).
                payload_b64 = data.get("media", {}).get("payload")

                # For debugging only:
                if payload_b64:
                    logging.info(f"✅ got audio chunk len={len(payload_b64)} (base64 chars)")

                # LATER:
                #   - We'll forward this audio to OpenAI realtime.
                #   - We'll get OpenAI's spoken answer back.
                #   - We'll send that answer audio back to Twilio so the caller hears it.

                # RIGHT NOW:
                #   We are *not* sending anything back to Twilio, on purpose,
                #   just keeping the stream alive.

            # 3d. "mark" events can show playback markers if we send audio
            #     back to Twilio. We're not doing that yet, so just log.
            elif event_type == "mark":
                logging.info(f"Mark event: {data}")

            # 3e. "stop" means Twilio is ending the stream (call ended or hangup).
            elif event_type == "stop":
                logging.info("Twilio says: stop (caller probably hung up)")
                break

            # Anything else, just log for now.
            else:
                logging.info(f"Other event from Twilio: {data}")

        # End while True loop

    except WebSocketDisconnect:
        # This means Twilio hung up / closed the socket. Totally normal.
        logging.info("❌ WebSocketDisconnect (caller hung up)")

    except Exception as e:
        # ANY other exception means our code blew up.
        # If we let this kill the socket instantly, Twilio
        # will tell the caller "an application error has occurred"
        # and hang up.
        #
        # We log the error but DO NOT re-raise, so the process
        # stays alive on Render.
        logging.exception("💥 ERROR inside /media loop (we caught it so Twilio won't insta-fail)")

    finally:
        # Step 3. Cleanup.
        #
        # Close websocket nicely if it's still open.
        try:
            await websocket.close()
        except Exception:
            pass

        logging.info(f"🔚 /media websocket closed for streamSid={stream_sid}")


###############################################################################
# 4. LOCAL DEV ENTRYPOINT (OPTIONAL)
#
# Render will IGNORE this because you told Render to run uvicorn directly:
#   uvicorn app:app --host 0.0.0.0 --port $PORT
#
# But this lets you run locally with:
#   python app.py
###############################################################################

if __name__ == "__main__":
    # Use PORT env if present (Render sets this), else default to 10000
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
