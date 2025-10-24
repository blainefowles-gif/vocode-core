# app.py  -- simple test server
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Riteway AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# Twilio test endpoint
@app.post("/voice", response_class=PlainTextResponse)
async def voice(request: Request):
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Riteway AI line is online. This is a test message.</Say>
  <Hangup/>
</Response>"""
    return twiml
