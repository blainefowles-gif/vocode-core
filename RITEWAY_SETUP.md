# Riteway AI phone agent setup

This repository contains Riteway's production Twilio Media Streams bridge in `app.py`.
It connects Twilio's bidirectional PCMU audio directly to the OpenAI Realtime API, loads
Riteway's public product catalog, checks live website inventory, and can send quote/order
requests to Riteway's existing Formspree inquiry channel.

## 1. Render configuration

The included `render.yaml` matches the existing service name, `riteway-ai-agent`. For an
existing service, keep its current instance plan and apply these settings in Render:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`
- Python: `3.12.13`

Required secrets:

- `OPENAI_API_KEY`: a project API key with Realtime API access and billing enabled.
- `TWILIO_AUTH_TOKEN`: the Auth Token for the Twilio account that owns the phone number.
- `MEDIA_STREAM_TOKEN`: a long random secret. Render can generate this from the Blueprint.

The Blueprint also sets these non-secret values:

- `PUBLIC_BASE_URL=https://riteway-ai-agent.onrender.com`
- `OPENAI_REALTIME_MODEL=gpt-realtime-1.5`
- `OPENAI_VOICE=marin`
- `INQUIRY_WEBHOOK_URL=https://formspree.io/f/mkovqjzg`
- `RITEWAY_INVENTORY_URL=https://ritewaylandscapeproducts.com/api/inventory`

If the Render service uses a different hostname, change `PUBLIC_BASE_URL`. The value must
exactly match the public URL configured in Twilio or Twilio signature validation will reject
the call.

After deployment, verify:

```bash
curl -sS https://riteway-ai-agent.onrender.com/health
```

`ok` must be `true`, `catalog_products` must be `54`, and the three configuration booleans
for OpenAI, media authentication, and inquiry capture should be `true`. Twilio signature
validation is `true` when `TWILIO_AUTH_TOKEN` is set.

For phone calls, use an always-on Render instance. A sleeping instance can take too long to
wake up after Twilio starts a call.

## 2. Twilio phone number

In Twilio Console, open **Phone Numbers → Manage → Active Numbers**, select Riteway's number,
and set **A call comes in** to:

- Webhook URL: `https://riteway-ai-agent.onrender.com/voice`
- Method: `HTTP POST`

Save, call the number, and confirm Tammy identifies herself as Riteway's virtual receptionist.
The webhook is `/voice`; do not use the stock Vocode `/inbound_call` route for this custom app.

## 3. Inquiry capture test

During a test call, ask for a quote and provide a name, callback number, material, quantity,
and delivery city. Tammy should confirm that the inquiry was sent. Confirm the message arrives
in the same Formspree destination used by Riteway's website.

If Riteway changes the website form provider, update `INQUIRY_WEBHOOK_URL`. When this value is
blank or the endpoint fails, Tammy is instructed not to claim the request was saved.

## 4. Catalog and inventory

`riteway_catalog.json` contains the 54 orderable products and website prices. Inventory is
pulled live from the website once per minute. To refresh the checked-in catalog after changing
`products-data.js`, run from this repository:

```bash
node scripts/sync_riteway_catalog.mjs https://ritewaylandscapeproducts.com/products-data.js riteway_catalog.json
```

Review and commit both the website change and regenerated catalog so phone and web pricing stay
aligned.

## 5. Local verification

Create a Python 3.12 virtual environment, install dependencies, and run the focused tests:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest tests.test_riteway_voice_agent
```

Run locally with:

```bash
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 10000
```

Local phone testing still requires a public HTTPS/WSS tunnel and a matching `PUBLIC_BASE_URL`.
