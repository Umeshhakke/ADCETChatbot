# 🎓 College Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready chatbot interface for college websites with FastAPI backend, intelligent session management, rate limiting, and ChatGPT-like streaming responses.

## Features

### Frontend
- **Modern UI**: Sliding chat window with smooth animations
- **Streaming Responses**: ChatGPT-like typing effect for natural interaction
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Intuitive UX**: Click-to-open chat icon at bottom-right corner

### Backend
- **Session Management**: Unique session tracking without authentication
- **Rate Limiting**: Configurable request throttling per session
- **Logging**: Comprehensive logging to file and console
- **RAG Ready**: Easy integration point for RAG pipeline
- **CORS Enabled**: Ready for cross-origin requests
- **Environment Config**: All settings via `.env` file

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ADCETChatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Start the backend server**
   
   **Option A: Direct Python**
   ```bash
   python -m server.server
   ```
   
   **Option B: Docker (Recommended)**
   ```bash
   docker-compose up -d
   ```
   
   Server will start on `http://localhost:8001`

5. **Start the frontend**
   ```bash
   python3 -m http.server 3000 --directory public
   ```
   Frontend available at `http://localhost:3000`

## Project Structure

```
ADCETChatbot/
├── server/                # Backend application
│   ├── core/             # RAG and database logic
│   │   └── __init__.py
│   ├── server.py         # FastAPI backend
│   └── __init__.py
├── public/               # Frontend files
│   ├── index.html       # Main HTML page
│   ├── style.css        # Styling and animations
│   └── script.js        # Client-side logic
├── logs/                # Application logs
│   └── server.log       # Server logs
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── requirements.txt     # Python dependencies
├── .env                 # Environment configuration
└── README.md           # This file
```

## Configuration

All configuration is managed through the `.env` file:

```env
# Server Configuration
HOST=0.0.0.0              # Server host
PORT=8001                 # Server port

# Rate Limiting
RATE_LIMIT=10             # Max requests per window
RATE_WINDOW=60            # Time window in seconds

# CORS Settings
CORS_ORIGINS=*            # Allowed origins (comma-separated)

# Logging
LOG_LEVEL=INFO            # Logging level
LOG_FILE=logs/server.log  # Log file path

# Session Configuration
SESSION_TIMEOUT=3600      # Session timeout in seconds
```

## API Endpoints

### `POST /chat`
Send a chat message and receive a response.

**Request:**
```json
{
  "message": "What programs do you offer?",
  "session_id": "session_123"
}
```

**Response:**
```json
{
  "response": "We offer B.Tech, M.Tech, BBA, and BCA programs...",
  "session_id": "session_123"
}
```

**Rate Limit:** Configurable via `.env` (default: 10 requests/minute)

### `GET /session/{session_id}`
Retrieve session information.

**Response:**
```json
{
  "session_id": "session_123",
  "message_count": 5,
  "created_at": "2026-05-19T17:30:00"
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "active_sessions": 42,
  "rate_limit": "10 requests per 60s"
}
```

## RAG Pipeline Integration

The chatbot is designed for easy integration with RAG (Retrieval-Augmented Generation) pipelines.

### Integration Point

Replace the `get_bot_response()` function in `server/server.py`:

```python
def get_bot_response(message: str, session_id: str) -> str:
    session = sessions[session_id]
    session["messages"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now()
    })
    
    # YOUR RAG PIPELINE HERE
    from core.rag import query_rag
    response = query_rag(
        query=message,
        conversation_history=session["messages"]
    )
    
    session["messages"].append({
        "role": "bot",
        "content": response,
        "timestamp": datetime.now()
    })
    
    return response
```

### Available Context

- `message`: Current user query
- `session_id`: Unique session identifier
- `session["messages"]`: Full conversation history with timestamps

## Testing

### Test Chat Endpoint
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test_123"}'
```

### Test Rate Limiting
```bash
for i in {1..11}; do
  curl -X POST http://localhost:8001/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"Test $i\", \"session_id\": \"rate_test\"}"
done
```

### Check Health
```bash
curl http://localhost:8001/health
```

## Customization

### Adjust Streaming Speed
Edit `public/script.js`:
```javascript
await streamText(botMessageDiv, data.response, 20);  // 20ms per character
```

### Change Rate Limits
Edit `.env`:
```env
RATE_LIMIT=20      # 20 requests
RATE_WINDOW=60     # per 60 seconds
```

### Modify UI Colors
Edit `public/style.css`:
```css
.chat-icon {
    background: #007bff;  /* Change to your brand color */
}
```

## Session Management

- **Storage**: In-memory (use Redis for production)
- **Tracking**: All messages with timestamps per session
- **Persistence**: Session IDs stored in browser localStorage
- **Expiration**: Configurable via `SESSION_TIMEOUT` in `.env`

## Rate Limiting

- **Algorithm**: Sliding window
- **Scope**: Per-session (not IP-based)
- **Response**: HTTP 429 when exceeded
- **Configuration**: Via `.env` file

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Build and Run with Docker

```bash
# Build the image
docker build -t college-chatbot:latest .

# Run the container
docker run -d \
  -p 8001:8001 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env \
  --name college-chatbot \
  college-chatbot:latest

# View logs
docker logs -f college-chatbot

# Stop the container
docker stop college-chatbot
docker rm college-chatbot
```

### Docker Features

- Health checks enabled
- Automatic restart on failure
- Volume mounts for logs and configuration
- Optimized multi-stage build
- Minimal image size with Python 3.11-slim

## Production Deployment

### Recommendations

1. **Use Redis** for session storage
2. **Enable HTTPS** with SSL certificates
3. **Set specific CORS origins** instead of `*`
4. **Use a process manager** (e.g., systemd, supervisor)
5. **Set up log rotation** for `logs/server.log`
6. **Add authentication** if needed
7. **Monitor with tools** like Prometheus/Grafana

### Example with Gunicorn

```bash
pip install gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```
