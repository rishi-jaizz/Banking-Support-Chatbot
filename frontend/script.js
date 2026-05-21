/**
 * BankAssist AI — Chat Interface Logic
 * Handles messaging, session management, and UI interactions.
 */

// ============================================
// State Management
// ============================================
const state = {
    sessionId: null,
    isLoading: false,
    messages: [],
};

// API base URL (same origin)
const API_BASE = window.location.origin;

// ============================================
// DOM Elements
// ============================================
const elements = {
    chatContainer: document.getElementById('chatContainer'),
    messages: document.getElementById('messages'),
    welcomeScreen: document.getElementById('welcomeScreen'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    statusIndicator: document.getElementById('statusIndicator'),
    sidebar: document.getElementById('sidebar'),
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Generate session ID
    state.sessionId = generateSessionId();

    // Set up input monitoring
    elements.messageInput.addEventListener('input', () => {
        elements.sendBtn.disabled = !elements.messageInput.value.trim();
    });

    // Check API health
    checkHealth();

    // Focus input
    elements.messageInput.focus();
});

function generateSessionId() {
    return 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).substr(2, 9);
}

// ============================================
// API Communication
// ============================================
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        updateStatus(data.rag_status === 'ready');
    } catch (err) {
        console.error('Health check failed:', err);
        updateStatus(false);
    }
}

function updateStatus(isOnline) {
    const dot = elements.statusIndicator.querySelector('.status-dot');
    const text = elements.statusIndicator;
    if (isOnline) {
        dot.className = 'status-dot online';
        text.innerHTML = '<span class="status-dot online"></span> Online';
    } else {
        dot.className = 'status-dot offline';
        text.innerHTML = '<span class="status-dot offline"></span> Connecting...';
    }
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || state.isLoading) return;

    // Hide welcome screen
    elements.welcomeScreen.classList.add('hidden');

    // Clear input
    elements.messageInput.value = '';
    elements.sendBtn.disabled = true;
    autoResize(elements.messageInput);

    // Add user message to UI
    addMessage('user', message);

    // Show typing indicator
    state.isLoading = true;
    showTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: state.sessionId,
            }),
        });

        if (!res.ok) {
            throw new Error(`API error: ${res.status}`);
        }

        const data = await res.json();
        state.sessionId = data.session_id;

        // Remove typing indicator and add response
        removeTypingIndicator();
        addMessage('assistant', data.response, data.sources);

    } catch (err) {
        console.error('Send error:', err);
        removeTypingIndicator();
        addMessage('assistant', 'I apologize, but I encountered an error processing your request. Please check that the server is running and try again.');
    } finally {
        state.isLoading = false;
        elements.messageInput.focus();
    }
}

// ============================================
// Message Rendering
// ============================================
function addMessage(role, content, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = role === 'assistant' ? 'AI' : 'You';
    const formattedContent = role === 'assistant' ? formatMarkdown(content) : escapeHtml(content);

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const sourceItems = sources.map(s => {
            const score = Math.round(s.relevance_score * 100);
            const sourceName = s.source.replace('.md', '').replace(/_/g, ' ');
            return `
                <div class="source-item">
                    <span class="source-name">${escapeHtml(sourceName)}</span>
                    <div class="source-bar">
                        <div class="source-bar-fill" style="width: ${score}%"></div>
                    </div>
                    <span class="source-score">${score}%</span>
                </div>`;
        }).join('');

        sourcesHtml = `
            <div class="message-sources">
                <button class="sources-toggle" onclick="toggleSources(this)">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    ${sources.length} source${sources.length > 1 ? 's' : ''} referenced
                </button>
                <div class="sources-list hidden">
                    ${sourceItems}
                </div>
            </div>`;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${formattedContent}</div>
            ${sourcesHtml}
        </div>`;

    elements.messages.appendChild(messageDiv);
    scrollToBottom();

    // Animate source bars after a slight delay
    if (sources && sources.length > 0) {
        setTimeout(() => {
            messageDiv.querySelectorAll('.source-bar-fill').forEach(bar => {
                bar.style.width = bar.style.width; // trigger reflow
            });
        }, 100);
    }
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="message-avatar" style="background: var(--gradient-primary); color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700;">AI</div>
        <div class="typing-bubble">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>`;
    elements.messages.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

// ============================================
// Markdown Formatting
// ============================================
function formatMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold and italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Unordered lists
    html = html.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Numbered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Tables (basic)
    html = html.replace(/\|(.+)\|/g, (match) => {
        const cells = match.split('|').filter(c => c.trim());
        if (cells.every(c => /^[-:]+$/.test(c.trim()))) return ''; // separator row
        const tag = 'td';
        const row = cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('');
        return `<tr>${row}</tr>`;
    });
    html = html.replace(/((?:<tr>.*<\/tr>\n?)+)/g, '<table>$1</table>');

    // Line breaks (double newline = paragraph, single = br)
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>(<h[1-3]>)/g, '$1');
    html = html.replace(/(<\/h[1-3]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<\/p>/g, '$1');
    html = html.replace(/<p>(<table>)/g, '$1');
    html = html.replace(/(<\/table>)<\/p>/g, '$1');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// UI Interactions
// ============================================
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function scrollToBottom() {
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

function toggleSources(btn) {
    const list = btn.nextElementSibling;
    btn.classList.toggle('expanded');
    list.classList.toggle('hidden');
}

function toggleSidebar() {
    elements.sidebar.classList.toggle('open');

    // Manage overlay
    let overlay = document.querySelector('.sidebar-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        overlay.onclick = () => {
            elements.sidebar.classList.remove('open');
            overlay.classList.remove('active');
        };
        document.body.appendChild(overlay);
    }
    overlay.classList.toggle('active');
}

function startNewChat() {
    state.sessionId = generateSessionId();
    state.messages = [];
    elements.messages.innerHTML = '';
    elements.welcomeScreen.classList.remove('hidden');
    elements.messageInput.value = '';
    elements.sendBtn.disabled = true;
    elements.messageInput.focus();

    // Close sidebar on mobile
    elements.sidebar.classList.remove('open');
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.classList.remove('active');
}

function clearChat() {
    if (state.messages.length === 0 && elements.messages.children.length === 0) return;

    // Delete session on server
    fetch(`${API_BASE}/api/sessions/${state.sessionId}`, { method: 'DELETE' }).catch(() => {});

    startNewChat();
}

function sendQuickTopic(topic) {
    elements.messageInput.value = topic;
    elements.sendBtn.disabled = false;
    sendMessage();

    // Close sidebar on mobile
    elements.sidebar.classList.remove('open');
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.classList.remove('active');
}
