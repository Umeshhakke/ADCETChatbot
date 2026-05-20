from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import asyncio
import time
import os
import json
import logging
import sys
from pathlib import Path

# ── Paths & env ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# Ensure server/ is on sys.path so `from core.rag import ...` resolves
SERVER_DIR = Path(__file__).parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

HOST         = os.getenv("HOST", "0.0.0.0")
PORT         = int(os.getenv("PORT", "8001"))
RATE_LIMIT   = int(os.getenv("RATE_LIMIT", "10"))
RATE_WINDOW  = int(os.getenv("RATE_WINDOW", "60"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE     = os.getenv("LOG_FILE", "logs/server.log")

log_path = ROOT / LOG_FILE
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Import RAG (after sys.path fix) ─────────────────────────────────────────
from core.rag import query_rag, stream_rag  # noqa: E402

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="ADCET Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = ROOT / "public"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

sessions         = defaultdict(lambda: {"messages": [], "created_at": datetime.now()})
rate_limit_store = defaultdict(list)


class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str


def check_rate_limit(session_id: str) -> bool:
    now = time.time()
    rate_limit_store[session_id] = [t for t in rate_limit_store[session_id] if now - t < RATE_WINDOW]
    if len(rate_limit_store[session_id]) >= RATE_LIMIT:
        return False
    rate_limit_store[session_id].append(now)
    return True

def _record(session_id: str, role: str, content: str) -> None:
    sessions[session_id]["messages"].append(
        {"role": role, "content": content, "timestamp": datetime.now()}
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Blocking chat – full answer returned at once."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not check_rate_limit(request.session_id):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Max {RATE_LIMIT} per {RATE_WINDOW}s.")

    _record(request.session_id, "user", request.message)
    try:
        # Run blocking RAG call in a thread so we don't block the event loop
        response = await asyncio.to_thread(query_rag, request.message)
    except Exception as exc:
        logger.error(f"RAG error: {exc}")
        response = "Sorry, I could not retrieve an answer right now."
    _record(request.session_id, "bot", response)
    logger.info(f"Session {request.session_id}: processed")
    return ChatResponse(response=response, session_id=request.session_id)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming SSE – tokens arrive as:  data: {"token": "..."}\n\n
    Final event:                        data: {"done": true}\n\n
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not check_rate_limit(request.session_id):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Max {RATE_LIMIT} per {RATE_WINDOW}s.")

    _record(request.session_id, "user", request.message)
    message = request.message
    session_id = request.session_id

    async def event_generator():
        full = []
        _sentinel = object()
        try:
            loop = asyncio.get_event_loop()
            gen = stream_rag(message)

            while True:
                token = await loop.run_in_executor(None, next, gen, _sentinel)
                if token is _sentinel:
                    break
                full.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.error(f"Stream error: {exc}")
            yield f"data: {json.dumps({'token': 'Sorry, I could not retrieve an answer right now.'})}\n\n"
        finally:
            _record(session_id, "bot", "".join(full))
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "message_count": len(sessions[session_id]["messages"]),
        "created_at": sessions[session_id]["created_at"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions), "rate_limit": f"{RATE_LIMIT}/{RATE_WINDOW}s"}


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting ADCET Chatbot on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
