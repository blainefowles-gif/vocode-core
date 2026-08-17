"""Riteway's Twilio Media Streams to OpenAI Realtime voice agent."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from html import escape as xml_escape
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

import aiohttp
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.request_validator import RequestValidator

from riteway_knowledge import BUSINESS_FACTS, build_agent_instructions, load_catalog


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("riteway_voice_agent")


def _public_base_url() -> str:
    configured = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    render_hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    if render_hostname:
        return f"https://{render_hostname}"
    return "https://riteway-ai-agent.onrender.com"


PUBLIC_BASE_URL = _public_base_url()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-1.5").strip()
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "marin").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
MEDIA_STREAM_TOKEN = os.getenv("MEDIA_STREAM_TOKEN", "").strip()
INQUIRY_WEBHOOK_URL = os.getenv(
    "INQUIRY_WEBHOOK_URL", "https://formspree.io/f/mkovqjzg"
).strip()
INVENTORY_URL = os.getenv(
    "RITEWAY_INVENTORY_URL", "https://ritewaylandscapeproducts.com/api/inventory"
).strip()
INVENTORY_CACHE_SECONDS = int(os.getenv("INVENTORY_CACHE_SECONDS", "60"))
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "3300"))

CATALOG = load_catalog()


def _websocket_media_url() -> str:
    parts = urlsplit(PUBLIC_BASE_URL)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, "/media", "", ""))


WS_MEDIA_URL = _websocket_media_url()

app = FastAPI(title="Riteway AI Voice Agent", version="2.0.0")


INQUIRY_TOOL = {
    "type": "function",
    "name": "record_inquiry",
    "description": (
        "Send a Riteway caller's quote, order, delivery, pickup, hauling, disposal, "
        "schedule, or callback request to the Riteway team. Call this during the call "
        "as soon as callback identity and a useful request summary are available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Caller's name"},
            "callback_phone": {
                "type": "string",
                "description": "Best phone number for the Riteway team to call or text",
            },
            "email": {"type": "string", "description": "Email, only when volunteered"},
            "request_type": {
                "type": "string",
                "enum": [
                    "quote",
                    "order",
                    "delivery",
                    "pickup",
                    "hauling",
                    "disposal",
                    "schedule",
                    "callback",
                    "other",
                ],
            },
            "material": {
                "type": "string",
                "description": "Requested material or project type",
            },
            "quantity": {
                "type": "string",
                "description": "Requested yards, tons, loads, dimensions, or quantity",
            },
            "fulfillment": {
                "type": "string",
                "enum": ["delivery", "pickup", "unsure", "not_applicable"],
            },
            "city_or_zip": {"type": "string"},
            "delivery_address": {"type": "string"},
            "summary": {
                "type": "string",
                "description": "Concise request summary and any timing or access notes",
            },
        },
        "required": ["name", "callback_phone", "request_type", "summary"],
        "additionalProperties": False,
    },
}


_inventory_cache: dict[str, Any] = {"expires_at": 0.0, "inventory": {}}
_inventory_lock = asyncio.Lock()


def build_twiml(caller_phone: str = "") -> str:
    parameters = []
    if MEDIA_STREAM_TOKEN:
        parameters.append(
            f'      <Parameter name="token" value="{xml_escape(MEDIA_STREAM_TOKEN, quote=True)}" />'
        )
    if caller_phone:
        parameters.append(
            f'      <Parameter name="callerPhone" value="{xml_escape(caller_phone, quote=True)}" />'
        )
    parameter_xml = f"\n{chr(10).join(parameters)}\n    " if parameters else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        "  <Connect>\n"
        f'    <Stream url="{xml_escape(WS_MEDIA_URL, quote=True)}">'
        f"{parameter_xml}</Stream>\n"
        "  </Connect>\n"
        "</Response>"
    )


def build_unavailable_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        "  <Say>Riteway's virtual receptionist is temporarily unavailable. "
        f"Please text {BUSINESS_FACTS['phone']} or use the inquiry form on our website.</Say>\n"
        "</Response>"
    )


def build_session_update(instructions: str) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 450,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": OPENAI_VOICE,
                },
            },
            "instructions": instructions,
            "tools": [INQUIRY_TOOL],
            "tool_choice": "auto",
        },
    }


def _public_request_url(request: Request) -> str:
    url = f"{PUBLIC_BASE_URL}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    return url


async def _validated_twilio_form(request: Request) -> dict[str, str]:
    raw_body = await request.body()
    params = dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
    if not TWILIO_AUTH_TOKEN:
        return params

    signature = request.headers.get("x-twilio-signature", "")
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    if not signature or not validator.validate(_public_request_url(request), params, signature):
        logger.warning("Rejected a request with an invalid Twilio signature")
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    return params


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Riteway AI Voice Agent",
        "health": "/health",
        "twilio_voice_webhook": "/voice",
    }


@app.get("/health")
async def health() -> JSONResponse:
    ready = bool(OPENAI_API_KEY and CATALOG.get("products"))
    body = {
        "ok": ready,
        "version": app.version,
        "model": REALTIME_MODEL,
        "voice": OPENAI_VOICE,
        "catalog_products": CATALOG.get("orderable_product_count", 0),
        "openai_configured": bool(OPENAI_API_KEY),
        "twilio_signature_validation": bool(TWILIO_AUTH_TOKEN),
        "media_stream_authentication": bool(MEDIA_STREAM_TOKEN),
        "inquiry_capture_configured": bool(INQUIRY_WEBHOOK_URL),
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/voice", response_class=PlainTextResponse)
async def voice(request: Request) -> PlainTextResponse:
    params = await _validated_twilio_form(request)
    logger.info("Twilio requested call instructions call_sid=%s", params.get("CallSid", "unknown"))
    twiml = build_twiml(params.get("From", "")) if OPENAI_API_KEY else build_unavailable_twiml()
    return PlainTextResponse(twiml, media_type="application/xml")


async def get_inventory(http_session: aiohttp.ClientSession) -> dict[str, Any]:
    if not INVENTORY_URL:
        return {}

    now = time.monotonic()
    if _inventory_cache["inventory"] and now < _inventory_cache["expires_at"]:
        return dict(_inventory_cache["inventory"])

    async with _inventory_lock:
        now = time.monotonic()
        if _inventory_cache["inventory"] and now < _inventory_cache["expires_at"]:
            return dict(_inventory_cache["inventory"])
        try:
            timeout = aiohttp.ClientTimeout(total=3)
            async with http_session.get(INVENTORY_URL, timeout=timeout) as response:
                response.raise_for_status()
                payload = await response.json()
                inventory = payload.get("inventory", {})
                if not isinstance(inventory, dict):
                    raise ValueError("Inventory response did not contain an inventory object")
                _inventory_cache["inventory"] = inventory
                _inventory_cache["expires_at"] = now + INVENTORY_CACHE_SECONDS
                logger.info("Loaded live inventory for %d products", len(inventory))
                return dict(inventory)
        except Exception as exc:
            logger.warning("Live inventory unavailable: %s", type(exc).__name__)
            return dict(_inventory_cache.get("inventory") or {})


async def _wait_for_twilio_start(websocket: WebSocket) -> tuple[str, str, str]:
    async with asyncio.timeout(10):
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            if message.get("event") != "start":
                continue

            start = message.get("start") or {}
            custom = start.get("customParameters") or {}
            received_token = str(custom.get("token") or "")
            if MEDIA_STREAM_TOKEN and not hmac.compare_digest(
                received_token, MEDIA_STREAM_TOKEN
            ):
                raise PermissionError("Invalid media stream token")

            stream_sid = str(start.get("streamSid") or message.get("streamSid") or "")
            if not stream_sid:
                raise ValueError("Twilio start event did not include a streamSid")
            return (
                stream_sid,
                str(start.get("callSid") or ""),
                str(custom.get("callerPhone") or ""),
            )


async def _configure_realtime(oai_websocket: aiohttp.ClientWebSocketResponse, instructions: str) -> None:
    await oai_websocket.send_json(build_session_update(instructions))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        message = await asyncio.wait_for(
            oai_websocket.receive(), timeout=max(0.1, deadline - time.monotonic())
        )
        if message.type != aiohttp.WSMsgType.TEXT:
            if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise RuntimeError("OpenAI closed before the Realtime session was configured")
            continue
        event = json.loads(message.data)
        event_type = event.get("type")
        if event_type == "session.updated":
            return
        if event_type == "error":
            error = event.get("error") or event
            raise RuntimeError(f"OpenAI session error: {error.get('message', 'unknown error')}")
    raise TimeoutError("OpenAI did not confirm session.updated")


def _clean(value: Any, maximum: int = 1_000) -> str:
    return " ".join(str(value or "").split())[:maximum]


async def _record_inquiry(
    http_session: aiohttp.ClientSession,
    arguments: dict[str, Any],
    call_sid: str,
    caller_phone: str,
) -> dict[str, Any]:
    callback_phone = _clean(arguments.get("callback_phone") or caller_phone, 80)
    payload = {
        "_subject": "New Riteway AI phone inquiry",
        "source": "Riteway AI phone agent",
        "name": _clean(arguments.get("name"), 120),
        "phone": callback_phone,
        "email": _clean(arguments.get("email"), 200),
        "request_type": _clean(arguments.get("request_type"), 80),
        "material": _clean(arguments.get("material"), 200),
        "quantity": _clean(arguments.get("quantity"), 160),
        "fulfillment": _clean(arguments.get("fulfillment"), 80),
        "city_or_zip": _clean(arguments.get("city_or_zip"), 160),
        "delivery_address": _clean(arguments.get("delivery_address"), 300),
        "message": _clean(arguments.get("summary"), 2_000),
        "call_sid": _clean(call_sid, 80),
        "caller_phone": _clean(caller_phone, 80),
    }

    if not INQUIRY_WEBHOOK_URL:
        return {"saved": False, "message": "The inquiry webhook is not configured."}

    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with http_session.post(
            INQUIRY_WEBHOOK_URL,
            json=payload,
            headers={"Accept": "application/json"},
            timeout=timeout,
        ) as response:
            if 200 <= response.status < 300:
                logger.info("Recorded Riteway inquiry call_sid=%s", call_sid or "unknown")
                return {
                    "saved": True,
                    "message": "The inquiry was sent to the Riteway team for follow-up.",
                }
            logger.error("Inquiry webhook returned HTTP %s", response.status)
    except Exception as exc:
        logger.error("Inquiry webhook failed: %s", type(exc).__name__)
    return {
        "saved": False,
        "message": "The inquiry could not be sent. Ask the caller to text Riteway or use the website form.",
    }


async def _handle_function_calls(
    event: dict[str, Any],
    oai_websocket: aiohttp.ClientWebSocketResponse,
    http_session: aiohttp.ClientSession,
    call_sid: str,
    caller_phone: str,
) -> None:
    outputs = (event.get("response") or {}).get("output") or []
    handled = False
    for item in outputs:
        if item.get("type") != "function_call" or item.get("name") != "record_inquiry":
            continue
        call_id = item.get("call_id")
        if not call_id:
            continue
        try:
            arguments = json.loads(item.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("Function arguments must be an object")
            result = await _record_inquiry(http_session, arguments, call_sid, caller_phone)
        except Exception as exc:
            logger.warning("Invalid record_inquiry call: %s", type(exc).__name__)
            result = {"saved": False, "message": "The inquiry details were invalid."}

        await oai_websocket.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        handled = True

    if handled:
        await oai_websocket.send_json(
            {"type": "response.create", "response": {"output_modalities": ["audio"]}}
        )


async def _twilio_to_openai(
    twilio_websocket: WebSocket,
    oai_websocket: aiohttp.ClientWebSocketResponse,
) -> None:
    while True:
        try:
            raw_message = await twilio_websocket.receive_text()
        except WebSocketDisconnect:
            return
        message = json.loads(raw_message)
        event_type = message.get("event")
        if event_type == "media":
            audio = (message.get("media") or {}).get("payload")
            if audio:
                await oai_websocket.send_json(
                    {"type": "input_audio_buffer.append", "audio": audio}
                )
        elif event_type == "stop":
            return


async def _openai_to_twilio(
    oai_websocket: aiohttp.ClientWebSocketResponse,
    twilio_websocket: WebSocket,
    http_session: aiohttp.ClientSession,
    stream_sid: str,
    call_sid: str,
    caller_phone: str,
) -> None:
    async for message in oai_websocket:
        if message.type != aiohttp.WSMsgType.TEXT:
            if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                return
            continue

        event = json.loads(message.data)
        event_type = event.get("type")
        if event_type in ("response.output_audio.delta", "response.audio.delta"):
            audio = event.get("delta")
            if audio:
                await twilio_websocket.send_json(
                    {"event": "media", "streamSid": stream_sid, "media": {"payload": audio}}
                )
        elif event_type == "input_audio_buffer.speech_started":
            await twilio_websocket.send_json({"event": "clear", "streamSid": stream_sid})
        elif event_type == "response.done":
            await _handle_function_calls(
                event,
                oai_websocket,
                http_session,
                call_sid,
                caller_phone,
            )
        elif event_type == "error":
            error = event.get("error") or event
            logger.error(
                "OpenAI Realtime error code=%s message=%s",
                error.get("code", "unknown"),
                _clean(error.get("message"), 300),
            )


async def _run_call_bridge(
    twilio_websocket: WebSocket,
    stream_sid: str,
    call_sid: str,
    caller_phone: str,
) -> None:
    async with aiohttp.ClientSession() as http_session:
        inventory = await get_inventory(http_session)
        instructions = build_agent_instructions(CATALOG, inventory, caller_phone)
        model = quote(REALTIME_MODEL, safe="")
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        if caller_phone:
            safety_id = hashlib.sha256(caller_phone.encode("utf-8")).hexdigest()[:32]
            headers["OpenAI-Safety-Identifier"] = safety_id

        async with http_session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={model}",
            headers=headers,
            heartbeat=20,
            max_msg_size=0,
        ) as oai_websocket:
            await _configure_realtime(oai_websocket, instructions)
            await oai_websocket.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                        "instructions": (
                            "Greet the caller in one sentence. Say: Hi, you've reached Riteway "
                            "Landscape Products. I'm Tammy, the virtual receptionist. How can I "
                            "help with your project?"
                        ),
                    },
                }
            )

            tasks = {
                asyncio.create_task(
                    _twilio_to_openai(twilio_websocket, oai_websocket),
                    name="twilio_to_openai",
                ),
                asyncio.create_task(
                    _openai_to_twilio(
                        oai_websocket,
                        twilio_websocket,
                        http_session,
                        stream_sid,
                        call_sid,
                        caller_phone,
                    ),
                    name="openai_to_twilio",
                ),
            }
            done, pending = await asyncio.wait(
                tasks, timeout=MAX_CALL_SECONDS, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.cancelled():
                    continue
                error = task.exception()
                if error and not isinstance(
                    error, (WebSocketDisconnect, asyncio.CancelledError)
                ):
                    raise error


@app.websocket("/media")
async def media(websocket: WebSocket) -> None:
    await websocket.accept()
    call_sid = ""
    try:
        if not OPENAI_API_KEY:
            await websocket.close(code=1011, reason="Voice agent is not configured")
            return
        stream_sid, call_sid, caller_phone = await _wait_for_twilio_start(websocket)
        logger.info("Started Riteway voice call call_sid=%s", call_sid or "unknown")
        await _run_call_bridge(websocket, stream_sid, call_sid, caller_phone)
    except PermissionError:
        logger.warning("Rejected unauthorized media stream")
        await websocket.close(code=1008, reason="Unauthorized media stream")
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        logger.exception(
            "Voice call failed call_sid=%s error=%s", call_sid or "unknown", type(exc).__name__
        )
        try:
            await websocket.close(code=1011, reason="Voice agent connection failed")
        except Exception:
            pass
    finally:
        logger.info("Ended Riteway voice call call_sid=%s", call_sid or "unknown")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
