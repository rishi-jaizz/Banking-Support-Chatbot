/**
 * BankAssist AI — Production Chat Interface
 * Handles messaging, session management, file uploads,
 * streaming SSE events, toast notifications, search, relative timestamps,
 * and UI interactions.
 */

// ============================================
// State Management
// ============================================
const state = {
    sessionId: null,
    isLoading: false,
    messages: [],
    lastConfidence: 0,
    loadedDocuments: [],
};

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || !window.location.hostname
    ? 'http://localhost:10000'
    : window.location.origin;

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
    confidenceBadge: document.getElementById('confidenceBadge'),
    confidenceText: document.getElementById('confidenceText'),
    toastContainer: document.getElementById('toastContainer'),
    docList: document.getElementById('docList'),
    docCount: document.getElementById('docCount'),
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Load existing session ID or generate new
    const savedSessionId = localStorage.getItem('bankassist_session_id');
    if (savedSessionId) {
        state.sessionId = savedSessionId;
        loadConversationHistory(savedSessionId);
    } else {
        state.sessionId = generateSessionId();
        localStorage.setItem('bankassist_session_id', state.sessionId);
    }

    elements.messageInput.addEventListener('input', () => {
        elements.sendBtn.disabled = !elements.messageInput.value.trim();
    });

    checkHealth();
    loadDocumentList();
    elements.messageInput.focus();
    
    // Auto-refresh document list every 10 seconds to sync changes
    setInterval(loadDocumentList, 10000);
});

function generateSessionId() {
    return 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).substr(2, 9);
}

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'info', duration = 4000) {
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || ''}</span><span>${message}</span>`;
    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============================================
// API Communication & History
// ============================================
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        updateStatus(data.rag_status === 'ready');
        const poweredText = document.getElementById('poweredByText');
        if (poweredText && data.llm_provider) {
            poweredText.textContent = `RAG + ${data.llm_provider.charAt(0).toUpperCase() + data.llm_provider.slice(1)} AI`;
        }
    } catch (err) {
        console.error('Health check failed:', err);
        updateStatus(false);
    }
}

function updateStatus(isOnline) {
    if (isOnline) {
        elements.statusIndicator.innerHTML = '<span class="status-dot online"></span> Online';
    } else {
        elements.statusIndicator.innerHTML = '<span class="status-dot offline"></span> Connecting...';
    }
}

async function loadConversationHistory(sessionId) {
    try {
        const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/history`);
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.messages && data.messages.length > 0) {
            elements.welcomeScreen.classList.add('hidden');
            elements.messages.innerHTML = '';
            
            data.messages.forEach(msg => {
                const role = msg.role;
                const content = msg.content;
                const sources = msg.sources || [];
                const suggestions = msg.suggested_questions || [];
                
                addMessage(role, content, sources, suggestions);
            });
            
            // Show last confidence from the last assistant message
            const assistantMsgs = data.messages.filter(m => m.role === 'assistant');
            if (assistantMsgs.length > 0) {
                const lastAssistantMsg = assistantMsgs[assistantMsgs.length - 1];
                if (lastAssistantMsg.sources && lastAssistantMsg.sources.length > 0) {
                    const avgConf = lastAssistantMsg.sources.reduce((acc, s) => acc + s.relevance_score, 0) / lastAssistantMsg.sources.length;
                    const pct = Math.round(avgConf * 100);
                    elements.confidenceBadge.style.display = 'inline-flex';
                    elements.confidenceText.textContent = `${pct}% confidence`;
                } else {
                    elements.confidenceBadge.style.display = 'none';
                }
            }
            scrollToBottom();
        }
    } catch (err) {
        console.error('Failed to load conversation history:', err);
    }
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || state.isLoading) return;

    elements.welcomeScreen.classList.add('hidden');
    elements.messageInput.value = '';
    elements.sendBtn.disabled = true;
    autoResize(elements.messageInput);

    addMessage('user', message);

    state.isLoading = true;
    showSkeletonLoader();

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: state.sessionId,
                stream: true
            }),
        });

        if (!response.ok) {
            if (response.status === 429) {
                let errorMsg = 'Rate limit exceeded. Please wait a minute before trying again.';
                try {
                    const errJson = await response.json();
                    if (errJson && errJson.detail) {
                        errorMsg = errJson.detail;
                    }
                } catch (_) {}
                const rateLimitError = new Error(errorMsg);
                rateLimitError.status = 429;
                throw rateLimitError;
            }
            const errorText = await response.text();
            throw new Error(errorText || `API error: ${response.status}`);
        }

        removeSkeletonLoader();
        await readSSEStream(response);

    } catch (err) {
        console.error('Send error:', err);
        removeSkeletonLoader();
        if (err.status === 429) {
            addMessage('assistant', `I apologize, but you are sending messages too quickly. ${err.message}`);
            showToast(err.message, 'error');
        } else if (err instanceof TypeError || err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            addMessage('assistant', 'I apologize, but I encountered a connection issue. Please check that the backend server is running and try again.');
            showToast('Unable to connect to server', 'error');
        } else {
            addMessage('assistant', 'I apologize, but the AI service is temporarily unavailable. Please try again in a moment.');
            showToast('AI service temporarily unavailable', 'error');
        }
    } finally {
        state.isLoading = false;
        elements.messageInput.focus();
    }
}

async function regenerateLastResponse() {
    if (state.isLoading) return;
    
    // Find the last assistant message in DOM
    const messageDivs = elements.messages.querySelectorAll('.message');
    let lastAssistantDiv = null;
    for (let i = messageDivs.length - 1; i >= 0; i--) {
        if (messageDivs[i].classList.contains('assistant')) {
            lastAssistantDiv = messageDivs[i];
            break;
        }
    }
    
    if (!lastAssistantDiv) {
        showToast('No response available to regenerate', 'error');
        return;
    }
    
    // Remove the last assistant message from DOM
    lastAssistantDiv.remove();
    
    state.isLoading = true;
    showSkeletonLoader();
    
    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: "",
                session_id: state.sessionId,
                stream: true,
                regenerate: true
            }),
        });
        
        if (!response.ok) {
            if (response.status === 429) {
                let errorMsg = 'Rate limit exceeded. Please wait a minute before trying again.';
                try {
                    const errJson = await response.json();
                    if (errJson && errJson.detail) {
                        errorMsg = errJson.detail;
                    }
                } catch (_) {}
                const rateLimitError = new Error(errorMsg);
                rateLimitError.status = 429;
                throw rateLimitError;
            }
            const errorText = await response.text();
            throw new Error(errorText || `API error: ${response.status}`);
        }
        
        removeSkeletonLoader();
        await readSSEStream(response);
        
    } catch (err) {
        console.error('Regeneration error:', err);
        removeSkeletonLoader();
        if (err.status === 429) {
            addMessage('assistant', `I apologize, but you are sending messages too quickly. ${err.message}`);
            showToast(err.message, 'error');
        } else if (err instanceof TypeError || err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            addMessage('assistant', 'I apologize, but I encountered a connection issue. Please check that the backend server is running and try again.');
            showToast('Unable to connect to server', 'error');
        } else {
            addMessage('assistant', 'I apologize, but the AI service is temporarily unavailable. Please try again in a moment.');
            showToast('AI service temporarily unavailable', 'error');
        }
    } finally {
        state.isLoading = false;
        elements.messageInput.focus();
    }
}

// ============================================
// SSE Streaming Parser
// ============================================
async function readSSEStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const messageDiv = document.createElement('div');
    messageDiv.className = `message assistant`;
    
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="message-bubble streaming-cursor" id="streamingBubble"></div>
            <div class="message-sources-wrapper"></div>
            <div class="message-suggestions-wrapper"></div>
            <div class="message-actions-wrapper"></div>
        </div>`;
    
    elements.messages.appendChild(messageDiv);
    scrollToBottom();

    const bubble = messageDiv.querySelector('#streamingBubble');
    const sourcesWrapper = messageDiv.querySelector('.message-sources-wrapper');
    const suggestionsWrapper = messageDiv.querySelector('.message-suggestions-wrapper');
    const actionsWrapper = messageDiv.querySelector('.message-actions-wrapper');
    
    let fullResponseText = "";
    let sources = [];
    let confidence = 0.0;
    
    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            
            for (const line of lines) {
                const cleanLine = line.trim();
                if (!cleanLine.startsWith('data: ')) continue;
                
                const dataStr = cleanLine.slice(6);
                try {
                    const data = JSON.parse(dataStr);
                    
                    if (data.event === 'metadata') {
                        sources = data.sources || [];
                        confidence = data.confidence || 0.0;
                        
                        const pct = Math.round(confidence * 100);
                        if (sources.length > 0) {
                            elements.confidenceBadge.style.display = 'inline-flex';
                            elements.confidenceText.textContent = `${pct}% confidence`;
                        } else {
                            elements.confidenceBadge.style.display = 'none';
                        }
                        
                        sourcesWrapper.innerHTML = buildSourcesHtml(sources);
                    } 
                    else if (data.event === 'token') {
                        const token = data.text || "";
                        fullResponseText += token;
                        bubble.innerHTML = formatMarkdown(fullResponseText);
                        scrollToBottom();
                    } 
                    else if (data.event === 'done') {
                        const suggestions = data.suggested_questions || [];
                        const finalResponse = data.text || fullResponseText;
                        
                        bubble.classList.remove('streaming-cursor');
                        bubble.innerHTML = formatMarkdown(finalResponse);
                        bubble.removeAttribute('id');
                        
                        suggestionsWrapper.innerHTML = buildSuggestionsHtml(suggestions);
                        actionsWrapper.innerHTML = `
                            <div class="message-actions">
                                <button class="action-btn copy-btn" onclick="copyMessageText(this)" title="Copy response">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                    </svg>
                                    <span>Copy</span>
                                </button>
                                <button class="action-btn regen-btn" onclick="regenerateLastResponse()" title="Regenerate response">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path>
                                    </svg>
                                    <span>Regenerate</span>
                                </button>
                            </div>`;
                        
                        scrollToBottom();
                    } 
                    else if (data.event === 'error') {
                        throw new Error(data.text || 'Error during streaming generation.');
                    }
                } catch (pe) {
                    console.error('Error parsing SSE line:', pe);
                }
            }
        }
    } catch (streamErr) {
        console.error('Stream read error:', streamErr);
        bubble.classList.remove('streaming-cursor');
        bubble.innerHTML = 'I apologize, but I encountered an error during response generation.';
        showToast('Generation interrupted', 'error');
    }
}

// ============================================
// Message & Element Rendering Helper
// ============================================
function addMessage(role, content, sources = [], suggestions = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = role === 'assistant' ? 'AI' : 'You';
    const formattedContent = role === 'assistant' ? formatMarkdown(content) : escapeHtml(content);

    let sourcesHtml = buildSourcesHtml(sources);
    let suggestionsHtml = role === 'assistant' ? buildSuggestionsHtml(suggestions) : '';
    let actionsHtml = role === 'assistant' ? `
        <div class="message-actions">
            <button class="action-btn copy-btn" onclick="copyMessageText(this)" title="Copy response">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span>Copy</span>
            </button>
            <button class="action-btn regen-btn" onclick="regenerateLastResponse()" title="Regenerate response">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path>
                </svg>
                <span>Regenerate</span>
            </button>
        </div>` : '';

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${formattedContent}</div>
            ${sourcesHtml}
            ${suggestionsHtml}
            ${actionsHtml}
        </div>`;

    elements.messages.appendChild(messageDiv);
    scrollToBottom();
}

function buildSourcesHtml(sources) {
    if (!sources || sources.length === 0) return '';

    const sourceItems = sources.map(s => {
        const score = Math.round(s.relevance_score * 100);
        // Extract file extension and format name nicely for tooltip
        const ext = s.source.split('.').pop().toLowerCase();
        const typeEmoji = ext === 'pdf' ? '📕' : (ext === 'txt' ? '📝' : '📄');
        const formattedName = s.source.replace('.md', '').replace('.txt', '').replace('.pdf', '').replace(/_/g, ' ');
        
        return `
            <div class="source-item" title="${escapeHtml(s.source)}">
                <span style="font-size: 0.8rem; margin-right: 2px;">${typeEmoji}</span>
                <span class="source-name">${escapeHtml(formattedName)}</span>
                <div class="source-bar">
                    <div class="source-bar-fill" style="width: ${score}%"></div>
                </div>
                <span class="source-score">${score}%</span>
            </div>`;
    }).join('');

    return `
        <div class="message-sources">
            <button class="sources-toggle" onclick="toggleSources(this)">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                <span>${sources.length} source${sources.length > 1 ? 's' : ''} referenced</span>
            </button>
            <div class="sources-list hidden">
                ${sourceItems}
            </div>
        </div>`;
}

function buildSuggestionsHtml(suggestions) {
    if (!suggestions || suggestions.length === 0) return '';
    const pills = suggestions.map(q => 
        `<button class="suggested-pill" onclick="sendSuggestedQuestion('${escapeHtml(q.replace(/'/g, "\\'"))}')">${escapeHtml(q)}</button>`
    ).join('');
    
    return `
        <div class="suggested-questions-container">
            <div class="suggested-title">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A5 5 0 0 0 8 8c0 1 .2 1.5.7 2.2a4 4 0 0 1 .8 2.3v1"></path><line x1="9" y1="18" x2="15" y2="18"></line><line x1="10" y1="22" x2="14" y2="22"></line></svg>
                Suggested Questions
            </div>
            ${pills}
        </div>`;
}

function showSkeletonLoader() {
    const loader = document.createElement('div');
    loader.className = 'skeleton-loader-msg';
    loader.id = 'skeletonLoader';
    loader.innerHTML = `
        <div class="message-avatar" style="background: var(--gradient-primary); color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700;">AI</div>
        <div class="skeleton-bubble">
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line short"></div>
        </div>
    `;
    elements.messages.appendChild(loader);
    scrollToBottom();
}

function removeSkeletonLoader() {
    const loader = document.getElementById('skeletonLoader');
    if (loader) loader.remove();
}

// ============================================
// Markdown Formatting
// ============================================
function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Headings
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    // Bold / Italic / Code
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Unordered & Ordered Lists
    html = html.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Tables
    html = html.replace(/\|(.+)\|/g, (match) => {
        const cells = match.split('|').filter(c => c.trim());
        if (cells.every(c => /^[-:]+$/.test(c.trim()))) return '';
        const row = cells.map(c => `<td>${c.trim()}</td>`).join('');
        return `<tr>${row}</tr>`;
    });
    html = html.replace(/((?:<tr>.*<\/tr>\n?)+)/g, '<table>$1</table>');

    // Paragraph splits
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = '<p>' + html + '</p>';

    // Formatting Cleanups
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
// Relative Timestamps
// ============================================
function formatRelativeTime(isoString) {
    if (!isoString) return '';
    try {
        const date = new Date(isoString);
        const now = new Date();
        const seconds = Math.floor((now - date) / 1000);
        
        if (seconds < 5) return 'Just now';
        if (seconds < 60) return `${seconds}s ago`;
        
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        
        const days = Math.floor(hours / 24);
        if (days === 1) return 'Yesterday';
        if (days < 7) return `${days}d ago`;
        
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    } catch (e) {
        return '';
    }
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
    requestAnimationFrame(() => {
        elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    });
}

function toggleSources(btn) {
    const list = btn.nextElementSibling;
    btn.classList.toggle('expanded');
    list.classList.toggle('hidden');
}

function toggleSidebar() {
    elements.sidebar.classList.toggle('open');
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
    localStorage.setItem('bankassist_session_id', state.sessionId);
    state.messages = [];
    state.lastConfidence = 0;
    elements.messages.innerHTML = '';
    elements.welcomeScreen.classList.remove('hidden');
    elements.messageInput.value = '';
    elements.sendBtn.disabled = true;
    elements.confidenceBadge.style.display = 'none';
    elements.messageInput.focus();

    elements.sidebar.classList.remove('open');
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.classList.remove('active');
    showToast('New conversation started', 'info', 2000);
}

function clearChat() {
    if (elements.messages.children.length === 0) return;
    fetch(`${API_BASE}/api/sessions/${state.sessionId}`, { method: 'DELETE' }).catch(() => {});
    startNewChat();
}

function sendQuickTopic(topic) {
    elements.messageInput.value = topic;
    elements.sendBtn.disabled = false;
    sendMessage();
    elements.sidebar.classList.remove('open');
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.classList.remove('active');
}

function copyMessageText(btn) {
    const messageContent = btn.closest('.message-content');
    const bubble = messageContent.querySelector('.message-bubble');
    if (!bubble) return;
    
    const textToCopy = bubble.innerText || bubble.textContent;
    
    navigator.clipboard.writeText(textToCopy).then(() => {
        showToast('Response copied to clipboard', 'success', 2000);
        const span = btn.querySelector('span');
        const originalText = span.textContent;
        span.textContent = 'Copied!';
        btn.style.color = 'var(--success)';
        btn.style.borderColor = 'rgba(16, 185, 129, 0.2)';
        
        setTimeout(() => {
            span.textContent = originalText;
            btn.style.color = '';
            btn.style.borderColor = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text:', err);
        showToast('Failed to copy text', 'error');
    });
}

function sendSuggestedQuestion(question) {
    elements.messageInput.value = question;
    elements.sendBtn.disabled = false;
    sendMessage();
}

// ============================================
// Document List Search & Filter
// ============================================
function filterDocuments() {
    const input = document.getElementById('docSearchInput');
    const filter = input.value.toLowerCase().trim();
    const clearBtn = document.getElementById('docSearchClearBtn');
    const docItems = elements.docList.querySelectorAll('.doc-item');
    
    if (filter) {
        clearBtn.style.display = 'block';
    } else {
        clearBtn.style.display = 'none';
    }
    
    let visibleCount = 0;
    docItems.forEach(item => {
        const nameSpan = item.querySelector('.doc-item-name');
        if (nameSpan) {
            const name = nameSpan.textContent.toLowerCase();
            if (name.includes(filter)) {
                item.style.display = 'flex';
                visibleCount++;
            } else {
                item.style.display = 'none';
            }
        }
    });
    
    let emptyMsg = elements.docList.querySelector('.doc-search-empty');
    if (visibleCount === 0 && docItems.length > 0) {
        if (!emptyMsg) {
            emptyMsg = document.createElement('div');
            emptyMsg.className = 'doc-search-empty';
            emptyMsg.style.cssText = 'text-align: center; color: var(--text-tertiary); font-size: 0.7rem; padding: 12px 0;';
            emptyMsg.textContent = 'No matching documents';
            elements.docList.appendChild(emptyMsg);
        }
    } else {
        if (emptyMsg) emptyMsg.remove();
    }
}

function clearDocSearch() {
    const input = document.getElementById('docSearchInput');
    input.value = '';
    filterDocuments();
    input.focus();
}

// ============================================
// File Upload & Indexing
// ============================================
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Prevent duplicate uploads
    const isDuplicate = state.loadedDocuments.some(d => d.filename.toLowerCase() === file.name.toLowerCase());
    if (isDuplicate) {
        // Highlight the matching item in the sidebar
        if (elements.docList) {
            const docItems = elements.docList.querySelectorAll('.doc-item');
            docItems.forEach(item => {
                const nameEl = item.querySelector('.doc-item-name');
                if (nameEl && nameEl.textContent.trim().toLowerCase() === file.name.toLowerCase()) {
                    item.classList.add('highlight-duplicate');
                    item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    setTimeout(() => {
                        item.classList.remove('highlight-duplicate');
                    }, 6000); // Pulse highlight for 6 seconds
                }
            });
        }

        // Show the warning toast with options
        showDuplicateToast(file);
        event.target.value = '';
        return;
    }

    uploadFileDirectly(file);
    event.target.value = '';
}

async function deleteDocumentSilently(filename) {
    try {
        const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'Failed to delete');
        }
        return true;
    } catch (err) {
        console.error('Failed to delete document silently:', err);
        return false;
    }
}

function showDuplicateToast(file) {
    // Softer warning-style UX toast
    const toast = document.createElement('div');
    toast.className = 'toast warning';
    
    let dismissTimeout;
    const startDismissTimer = () => {
        dismissTimeout = setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 6000); // Smooth auto-dismiss after 6s
    };
    
    const cancelDismissTimer = () => {
        if (dismissTimeout) clearTimeout(dismissTimeout);
    };

    toast.innerHTML = `
        <span style="font-size: 1.15rem; flex-shrink: 0; line-height: 1;">⚠️</span>
        <div style="display: flex; flex-direction: column; gap: 4px; flex: 1;">
            <div style="font-weight: 600; font-size: 0.78rem; line-height: 1.25; color: white;">Document already exists in the knowledge base.</div>
            <div style="font-size: 0.7rem; color: rgba(255, 255, 255, 0.85); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 250px;">"${escapeHtml(file.name)}"</div>
            <div class="toast-actions" style="display: flex; gap: 8px; margin-top: 4px;">
                <button class="toast-action-btn replace-btn" style="background: white; color: #111827; border: none; padding: 4px 8px; border-radius: 4px; font-size: 0.68rem; cursor: pointer; font-weight: 700; transition: background-color 0.2s;">Replace Existing</button>
                <button class="toast-action-btn skip-btn" style="background: rgba(255, 255, 255, 0.15); color: white; border: 1px solid rgba(255, 255, 255, 0.25); padding: 4px 8px; border-radius: 4px; font-size: 0.68rem; cursor: pointer; font-weight: 500; transition: background-color 0.2s;">Skip Upload</button>
            </div>
        </div>
    `;

    toast.addEventListener('mouseenter', cancelDismissTimer);
    toast.addEventListener('mouseleave', startDismissTimer);

    const replaceBtn = toast.querySelector('.replace-btn');
    const skipBtn = toast.querySelector('.skip-btn');

    replaceBtn.addEventListener('click', async () => {
        cancelDismissTimer();
        toast.remove();
        
        showToast(`Replacing "${file.name}"...`, 'info');
        const success = await deleteDocumentSilently(file.name);
        if (success) {
            uploadFileDirectly(file);
        } else {
            showToast(`Failed to replace "${file.name}"`, 'error');
        }
    });

    skipBtn.addEventListener('click', () => {
        cancelDismissTimer();
        toast.remove();
        showToast(`Upload of "${file.name}" skipped`, 'info', 2000);
    });

    elements.toastContainer.appendChild(toast);
    startDismissTimer();
}

async function uploadFileDirectly(file) {
    const uploadText = document.getElementById('uploadText');
    const progressContainer = document.getElementById('uploadProgressContainer');
    const progressBar = document.getElementById('uploadProgressBar');
    const fileInput = document.getElementById('fileInput');

    const tempId = 'temp_' + Date.now();
    const tempItem = document.createElement('div');
    tempItem.className = 'doc-item pending';
    tempItem.id = tempId;
    
    const ext = file.name.split('.').pop().toLowerCase();
    const typeIcon = ext === 'pdf' ? '📕' : (ext === 'txt' ? '📝' : '📄');
    
    tempItem.innerHTML = `
        <span class="doc-item-icon">${typeIcon}</span>
        <div class="doc-item-info-wrap">
            <span class="doc-item-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
            <div class="doc-item-meta">
                <span class="doc-status-badge uploading">
                    <span class="doc-spinner"></span>
                    Uploading...
                </span>
            </div>
        </div>
    `;
    
    if (elements.docList) {
        // Remove empty state if it's there
        const emptyList = elements.docList.querySelector('.doc-list-empty');
        if (emptyList) emptyList.remove();
        elements.docList.insertBefore(tempItem, elements.docList.firstChild);
    }

    const originalText = uploadText.textContent;
    uploadText.textContent = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/api/upload`);

    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            if (progressContainer && progressBar) {
                progressContainer.style.display = 'block';
                progressBar.style.width = `${pct * 0.8}%`; // Max 80% for upload, remaining for backend indexing
            }
            
            const statusBadge = tempItem.querySelector('.doc-status-badge');
            if (statusBadge) {
                if (pct < 100) {
                    statusBadge.className = 'doc-status-badge uploading';
                    statusBadge.innerHTML = `<span class="doc-spinner"></span> Uploading (${pct}%)`;
                } else {
                    statusBadge.className = 'doc-status-badge processing';
                    statusBadge.innerHTML = `<span class="doc-spinner"></span> Indexing...`;
                    uploadText.textContent = `Indexing ${file.name}...`;
                    if (progressBar) progressBar.style.width = '90%';
                }
            }
        }
    };

    xhr.onload = async () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                if (progressBar) progressBar.style.width = '100%';

                const statusBadge = tempItem.querySelector('.doc-status-badge');
                if (statusBadge) {
                    statusBadge.className = 'doc-status-badge success';
                    statusBadge.innerHTML = 'Indexed ✅';
                }

                showToast(`📂 ${data.filename} indexed successfully`, 'success');

                elements.welcomeScreen.classList.add('hidden');
                addMessage('assistant', `📂 **Document Indexed Successfully**\n\n**${data.filename}** has been added to the knowledge base.\n\n- **Chunks created:** ${data.chunks_added}\n- **Total knowledge base:** ${data.total_indexed_chunks} chunks\n\nYou can now ask questions about this document!`);

                setTimeout(() => {
                    loadDocumentList();
                    resetUploaderUI();
                }, 1000);

            } catch (err) {
                handleError('Failed to parse indexing response');
            }
        } else {
            let errMsg = 'Upload failed';
            try {
                const errData = JSON.parse(xhr.responseText);
                errMsg = errData.detail || errMsg;
            } catch (e) {}
            handleError(errMsg);
        }
    };

    xhr.onerror = () => {
        handleError('Network error occurred during upload.');
    };

    function handleError(message) {
        const statusBadge = tempItem.querySelector('.doc-status-badge');
        if (statusBadge) {
            statusBadge.className = 'doc-status-badge failed';
            statusBadge.innerHTML = 'Failed ❌';
        }
        showToast(message, 'error');

        setTimeout(() => {
            tempItem.remove();
            resetUploaderUI();
            loadDocumentList(); // Reload in case list is empty to show empty indicator
        }, 3000);
    }

    function resetUploaderUI() {
        uploadText.textContent = originalText;
        if (progressContainer) progressContainer.style.display = 'none';
        if (progressBar) progressBar.style.width = '0%';
        fileInput.value = '';
    }

    xhr.send(formData);
}

// ============================================
// Document List Retrieval
// ============================================
async function loadDocumentList() {
    try {
        const res = await fetch(`${API_BASE}/api/documents`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.documents) {
            state.loadedDocuments = data.documents;
            
            if (elements.docCount) {
                elements.docCount.textContent = data.documents.length;
            }

            if (elements.docList) {
                // If currently searching, don't overwrite if it might disrupt typing,
                // but if we are not searching, we update the HTML.
                const searchInput = document.getElementById('docSearchInput');
                if (searchInput && searchInput.value.trim() !== '') {
                    // Just update the metadata of existing DOM elements if needed, or skip to avoid disrupting search filter
                    return;
                }

                if (data.documents.length === 0) {
                    elements.docList.innerHTML = `<div class="doc-list-empty" style="text-align: center; color: var(--text-tertiary); font-size: 0.7rem; padding: 12px 0;">No documents indexed</div>`;
                    return;
                }

                const icons = { md: '📄', txt: '📝', pdf: '📕' };
                elements.docList.innerHTML = data.documents.map(d => {
                    const relativeTime = formatRelativeTime(d.uploaded_at);
                    const timeDisplay = relativeTime ? `Uploaded ${relativeTime}` : '';
                    const chunksDisplay = d.chunks ? `${d.chunks} chunks` : '';
                    const metaParts = [timeDisplay, chunksDisplay].filter(Boolean);
                    const metaHtml = metaParts.length > 0 ? `
                        <div class="doc-item-meta">
                            <span>${metaParts.join(' <span class="doc-item-meta-separator">&bull;</span> ')}</span>
                        </div>` : '';
                    
                    return `
                        <div class="doc-item ${d.type}">
                            <span class="doc-item-icon">${icons[d.type] || '📄'}</span>
                            <div class="doc-item-info-wrap">
                                <span class="doc-item-name" title="${d.filename}">${d.filename}</span>
                                ${metaHtml}
                            </div>
                            <span class="doc-item-size">${d.size_kb}KB</span>
                            <button class="doc-delete-btn" onclick="event.stopPropagation(); deleteDocument('${d.filename}')" title="Delete document">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="3 6 5 6 21 6"></polyline>
                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                </svg>
                                Delete
                            </button>
                        </div>`;
                }).join('');
            }
        }
    } catch (err) {
        console.error('Failed to load document list:', err);
    }
}

// ============================================
// Document Deletion
// ============================================
async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"? This will permanently remove it from the knowledge base and RAG search index.`)) {
        return;
    }

    showToast(`Deleting ${filename}...`, 'info');

    try {
        const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'Failed to delete');
        }

        const data = await res.json();
        showToast(data.message || `Deleted ${filename}`, 'success');

        loadDocumentList();

        elements.welcomeScreen.classList.add('hidden');
        addMessage('assistant', `🗑️ **Document Deleted**\n\n**${filename}** has been removed from the knowledge base and its chunks deleted from the search index.`);

    } catch (err) {
        console.error('Failed to delete document:', err);
        showToast(`Failed to delete document: ${err.message}`, 'error');
    }
}
