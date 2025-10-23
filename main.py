import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream

# -----------------------------
# OpenAI API Key Configuration
# -----------------------------
OPENAI_API_KEY = None
_OPENAI_KEY_SOURCE = None

# Try Streamlit Cloud secrets first (safe on Streamlit Cloud)
try:
    import streamlit as st  # streamlit may not be installed locally; this will fail locally if not present
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        _OPENAI_KEY_SOURCE = "streamlit"
except Exception:
    # If import fails or key missing, we'll fall back to .env below
    pass

# Fallback to local .env for local development/testing
if not OPENAI_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            _OPENAI_KEY_SOURCE = "env"
    except Exception:
        # dotenv may not be installed; not fatal here
        pass

if not OPENAI_API_KEY:
    # Clear, actionable error message
    raise ValueError(
        "Missing the OpenAI API key. "
        "Set it in Streamlit Cloud Secrets (OPENAI_API_KEY) or in a local .env file as OPENAI_API_KEY."
    )

# Small non-sensitive confirmation for logs (DOES NOT print the key)
print(f"OPENAI_API_KEY loaded from: {_OPENAI_KEY_SOURCE}")

# -----------------------------
# Configuration
# -----------------------------
PORT = int(os.getenv("PORT", 5050))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
VOICE = "verse"
SHOW_TIMING_MATH = False
LOG_EVENT_TYPES = [
    "error",
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
    "session.updated",
]

# -----------------------------
# OB/GYN Subspecialties
# -----------------------------
SUBSPECIALTIES = {
    "maternal_fetal": "Maternal-Fetal Medicine (High-Risk Pregnancy)",
    "urogynecology": "Urogynecology & Pelvic Reconstructive Medicine",
    "gynecologic_oncology": "Gynecologic Oncology",
    "reproductive_endo": "Reproductive Endocrinology & Infertility",
    "minimally_invasive": "Complex/Minimally Invasive Gynecologic Surgery",
    "general_obgyn": "General OB/GYN",
    "emergency": "Emergency OB/GYN",
}

# -----------------------------
# System Message for AI
# -----------------------------
SYSTEM_MESSAGE = f"""You are an AI medical assistant for an OB/GYN clinic's triage hotline. Your role is to:

1. Collect patient information efficiently and compassionately
2. Assess urgency and recommend the appropriate subspecialty
3. Provide clear, professional guidance, and give the suggestion of which subspecialty to refer to

SUBSPECIALTY CATEGORIES:
{json.dumps(SUBSPECIALTIES, indent=2)}

TRIAGE PROTOCOL:
1. Listen carefully as the patient describes their situation
2. If EMERGENCY symptoms detected (severe hemorrhage, chest pain, severe abdominal pain, difficulty breathing, seizures, vision changes):
   - Immediately recommend calling 911 or going to the nearest Emergency Room
   - Classify as: "Emergency OB/GYN"

3. For non-emergency cases, classify based on:
   - Pregnancy-related concerns → "Maternal-Fetal Medicine"
   - Cancer screening, abnormal pap, postmenopausal bleeding → "Gynecologic Oncology"
   - Urinary incontinence, pelvic prolapse → "Urogynecology"
   - Infertility, PCOS, hormonal issues → "Reproductive Endocrinology & Infertility"
   - Fibroids, endometriosis, complex surgical needs → "Complex/Minimally Invasive Gynecologic Surgery"
   - Routine checkups, general concerns → "General OB/GYN"

RESPONSE FORMAT:
After collecting information, provide:
1. Brief summary of patient info
2. Recommended subspecialty with clear reasoning
3. Urgency level (Emergency/Urgent/Routine)
4. Next steps (call 911, schedule appointment, etc.)

Keep your tone warm, professional, and reassuring. Speak clearly and avoid medical jargon when possible.
"""

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say(
        "Hello, wetrophia at your service. Please state your name and symptoms, and we will assist you shortly. "
        "If you are in an emergency, please hang up and call 911 immediately.",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    response.pause(length=1)
    response.say(
        "O.K. you can start talking!",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    # Open a connection to the OpenAI Realtime API WebSocket
    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model=gpt-realtime&temperature={TEMPERATURE}",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data.get('event') == 'media' and openai_ws.state.name == 'OPEN':
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data.get('event') == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        nonlocal_vars = None
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data.get('event') == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.state.name == 'OPEN':
                    await openai_ws.close()
            except Exception as e:
                print(f"Error in receive_from_twilio: {e}")

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response.get('type') in LOG_EVENT_TYPES:
                        print(f"Received event: {response.get('type')}", response)

                    if response.get('type') == 'response.output_audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response.get("item_id") and response["item_id"] != last_assistant_item:
                            response_start_timestamp_twilio = latest_media_timestamp
                            last_assistant_item = response["item_id"]
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        await send_mark(websocket, stream_sid)

                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

# -----------------------------
# Helper functions for OpenAI session
# -----------------------------
async def send_initial_conversation_item(openai_ws):
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {"format": {"type": "audio/pcmu"}, "turn_detection": {"type": "server_vad"}},
                "output": {"format": {"type": "audio/pcmu"}, "voice": VOICE}
            },
            "instructions": SYSTEM_MESSAGE,
        }
    }
    print("Sending session update:", json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))
    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
