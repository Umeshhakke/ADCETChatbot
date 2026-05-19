from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import time
import os
import logging
from pathlib import Path

# Load environment variables from root
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration from .env
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/server.log")

# Setup logging with absolute path
log_path = Path(__file__).parent.parent / LOG_FILE
log_path.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="College Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with absolute path
static_path = Path(__file__).parent.parent / "public"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Session storage
sessions = defaultdict(lambda: {"messages": [], "created_at": datetime.now()})

# Rate limiting storage
rate_limit_store = defaultdict(list)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

def check_rate_limit(session_id: str) -> bool:
    """Check if session has exceeded rate limit"""
    now = time.time()
    requests = rate_limit_store[session_id]
    
    # Remove old requests outside the window
    rate_limit_store[session_id] = [req_time for req_time in requests if now - req_time < RATE_WINDOW]
    
    if len(rate_limit_store[session_id]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for session: {session_id}")
        return False
    
    rate_limit_store[session_id].append(now)
    return True

def get_bot_response(message: str, session_id: str) -> str:
    """
    Placeholder for RAG pipeline integration.
    Replace this function with your core.rag module.
    """
    session = sessions[session_id]
    session["messages"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now()
    })
    
    # TODO: Replace with your RAG pipeline from core folder
    # from core.rag import query_rag
    # response = query_rag(message, session["messages"])
    
    response = f"Thank you for your message: '{message}'. This is a placeholder response. Your RAG pipeline will be integrated here."
    
    session["messages"].append({
        "role": "bot",
        "content": response,
        "timestamp": datetime.now()
    })
    
    logger.info(f"Session {session_id}: User message processed")
    return response

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages with rate limiting"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not check_rate_limit(request.session_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW} seconds."
        )
    
    response = get_bot_response(request.message, request.session_id)
    
    return ChatResponse(response=response, session_id=request.session_id)

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "message_count": len(sessions[session_id]["messages"]),
        "created_at": sessions[session_id]["created_at"]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "active_sessions": len(sessions),
        "rate_limit": f"{RATE_LIMIT} requests per {RATE_WINDOW}s"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
