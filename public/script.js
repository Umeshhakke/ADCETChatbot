const chatIcon     = document.getElementById('chatIcon');
const chatWindow   = document.getElementById('chatWindow');
const closeChat    = document.getElementById('closeChat');
const chatMessages = document.getElementById('chatMessages');
const chatInput    = document.getElementById('chatInput');
const sendBtn      = document.getElementById('sendBtn');

const API_BASE = 'http://localhost:8001';

let sessionId = localStorage.getItem('chatSessionId') || generateSessionId();
localStorage.setItem('chatSessionId', sessionId);

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

chatIcon.addEventListener('click', () => {
    chatWindow.classList.add('open');
    chatIcon.style.display = 'none';
    chatInput.focus();
});

closeChat.addEventListener('click', () => {
    chatWindow.classList.remove('open');
    chatIcon.style.display = 'flex';
});

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) sendMessage();
});

function addMessage(text, isUser) {
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user' : 'bot'}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message bot typing-indicator';
    div.id = 'typingIndicator';
    div.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    chatInput.value = '';
    sendBtn.disabled = true;
    showTypingIndicator();

    // Create bot message bubble (hidden until first token)
    const botDiv = document.createElement('div');
    botDiv.className = 'message bot';
    botDiv.style.display = 'none';
    chatMessages.appendChild(botDiv);

    try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            removeTypingIndicator();
            botDiv.remove();
            addMessage(err.detail || 'Error occurred. Please try again.', false);
            sendBtn.disabled = false;
            return;
        }

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = JSON.parse(line.slice(6));

                if (payload.done) break;

                if (payload.token) {
                    // Show bubble on first token
                    if (botDiv.style.display === 'none') {
                        removeTypingIndicator();
                        botDiv.style.display = '';
                    }
                    botDiv.textContent += payload.token;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }
        }

        // Fallback: if nothing was streamed, remove empty bubble
        if (botDiv.style.display === 'none') {
            removeTypingIndicator();
            botDiv.remove();
            addMessage('No response received.', false);
        }

    } catch (error) {
        removeTypingIndicator();
        botDiv.remove();
        addMessage('Connection error. Is the server running?', false);
    }

    sendBtn.disabled = false;
    chatInput.focus();
}
