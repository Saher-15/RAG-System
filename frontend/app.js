/**
 * Mini RAG System — Frontend
 *
 * Communicates with the FastAPI backend:
 *   POST /api/documents/upload    — ingest a file
 *   GET  /api/documents           — list documents
 *   DELETE /api/documents/:id     — remove a document
 *   POST /api/chat/stream         — SSE streaming chat
 */

// When served by FastAPI (same origin), use relative URLs.
// For local file:// development, fall back to localhost.
const API = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  documents: [],          // [{document_id, filename, chunk_count}]
  conversation: [],       // [{role, content}]
  streaming: false,
};

// ── DOM references ────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const uploadZone    = $('uploadZone');
const fileInput     = $('fileInput');
const uploadProgress = $('uploadProgress');
const progressBar   = $('progressBar');
const progressStatus = $('progressStatus');
const docList       = $('docList');
const docEmpty      = $('docEmpty');
const messages      = $('messages');
const welcome       = $('welcome');
const questionInput = $('questionInput');
const sendBtn       = $('sendBtn');
const modelBadge    = $('modelBadge');
const toast         = $('toast');

// ── Toast ─────────────────────────────────────────────────────────────────────

let toastTimer;
function showToast(msg, type = 'info') {
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── Document list ──────────────────────────────────────────────────────────────

async function loadDocuments(retries = 5) {
  try {
    const res = await fetch(`${API}/api/documents`);
    if (res.status === 503 && retries > 0) {
      setTimeout(() => loadDocuments(retries - 1), 3000);
      return;
    }
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.documents = data.documents;
    renderDocuments();
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

function renderDocuments() {
  // Remove existing doc items (keep the empty placeholder)
  docList.querySelectorAll('.doc-item').forEach(el => el.remove());

  docEmpty.style.display = state.documents.length ? 'none' : '';

  for (const doc of state.documents) {
    const icon = doc.filename.endsWith('.pdf') ? '📕' :
                 doc.filename.endsWith('.md')  ? '📝' : '📄';

    const item = document.createElement('div');
    item.className = 'doc-item';
    item.innerHTML = `
      <span class="doc-icon">${icon}</span>
      <div class="doc-info">
        <div class="doc-name" title="${esc(doc.filename)}">${esc(doc.filename)}</div>
        <div class="doc-meta">${doc.chunk_count} chunk${doc.chunk_count !== 1 ? 's' : ''}</div>
      </div>
      <button class="doc-delete" title="Remove document" data-id="${esc(doc.document_id)}">✕</button>
    `;
    item.querySelector('.doc-delete').addEventListener('click', e => {
      e.stopPropagation();
      deleteDocument(doc.document_id, doc.filename);
    });
    docList.appendChild(item);
  }
}

async function deleteDocument(docId, filename) {
  try {
    const res = await fetch(`${API}/api/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    showToast(`Removed "${filename}"`, 'success');
    await loadDocuments();
  } catch (err) {
    showToast(`Failed to delete: ${err.message}`, 'error');
  }
}

// ── File upload ───────────────────────────────────────────────────────────────

function setupUpload() {
  fileInput.addEventListener('change', () => handleFiles(fileInput.files));

  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
  });
}

async function handleFiles(files) {
  if (!files.length) return;
  const fileArr = Array.from(files);

  uploadProgress.classList.add('active');
  progressBar.style.width = '0%';
  progressStatus.textContent = `Uploading 0 / ${fileArr.length}…`;

  let done = 0;
  for (const file of fileArr) {
    progressStatus.textContent = `Uploading "${file.name}" (${done + 1}/${fileArr.length})…`;
    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(`${API}/api/documents/upload`, {
        method: 'POST',
        body: form,
      });

      if (res.status === 503) {
        throw new Error('Server is still loading, please wait a moment and try again.');
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }

      const data = await res.json();
      showToast(`✅ ${data.message}`, 'success');
    } catch (err) {
      showToast(`❌ "${file.name}": ${err.message}`, 'error');
    }

    done++;
    progressBar.style.width = `${(done / fileArr.length) * 100}%`;
  }

  progressStatus.textContent = 'Done!';
  setTimeout(() => uploadProgress.classList.remove('active'), 1000);
  fileInput.value = '';
  await loadDocuments();
}

// ── Chat ──────────────────────────────────────────────────────────────────────

function setupChat() {
  sendBtn.addEventListener('click', sendMessage);

  questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-grow textarea
  questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = questionInput.scrollHeight + 'px';
  });
}

async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || state.streaming) return;

  if (state.documents.length === 0) {
    showToast('Upload at least one document first.', 'error');
    return;
  }

  // Hide welcome
  welcome.style.display = 'none';

  // Append user bubble
  appendMessage('user', question);
  state.conversation.push({ role: 'user', content: question });

  // Clear input
  questionInput.value = '';
  questionInput.style.height = 'auto';

  // Create assistant placeholder
  const { bubble, sourcesEl, typingEl } = appendAssistantPlaceholder();

  state.streaming = true;
  sendBtn.disabled = true;

  let fullAnswer = '';
  let sources = [];

  try {
    const res = await fetch(`${API}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        conversation_history: state.conversation.slice(0, -1), // exclude current turn
        top_k: 5,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    typingEl.remove(); // remove typing dots once stream starts

    // Add cursor
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.appendChild(cursor);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload) continue;

        let event;
        try { event = JSON.parse(payload); } catch { continue; }

        switch (event.type) {
          case 'sources':
            sources = event.sources || [];
            break;

          case 'token':
            fullAnswer += event.content;
            // Render markdown before the cursor
            bubble.innerHTML = renderMarkdown(fullAnswer);
            bubble.appendChild(cursor);
            scrollToBottom();
            break;

          case 'done':
            cursor.remove();
            renderSources(sourcesEl, sources);
            break;

          case 'error':
            cursor.remove();
            bubble.innerHTML = `<span style="color:var(--red)">⚠ ${esc(event.error)}</span>`;
            break;
        }
      }
    }

  } catch (err) {
    typingEl.remove();
    bubble.innerHTML = `<span style="color:var(--red)">⚠ ${esc(err.message)}</span>`;
  }

  // Save assistant turn to history
  if (fullAnswer) {
    state.conversation.push({ role: 'assistant', content: fullAnswer });
  }

  state.streaming = false;
  sendBtn.disabled = false;
  questionInput.focus();
  scrollToBottom();
}

// ── Message rendering ─────────────────────────────────────────────────────────

function appendMessage(role, content) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? '👤' : '🤖';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = role === 'user' ? esc(content) : renderMarkdown(content);

  body.appendChild(bubble);
  msg.appendChild(avatar);
  msg.appendChild(body);
  messages.appendChild(msg);
  scrollToBottom();
  return bubble;
}

function appendAssistantPlaceholder() {
  const msg = document.createElement('div');
  msg.className = 'message assistant';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = '🤖';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  // Typing indicator
  const typingEl = document.createElement('div');
  typingEl.className = 'typing-indicator';
  typingEl.innerHTML = '<span></span><span></span><span></span>';
  bubble.appendChild(typingEl);

  // Sources placeholder
  const sourcesEl = document.createElement('div');
  sourcesEl.className = 'sources';

  body.appendChild(bubble);
  body.appendChild(sourcesEl);
  msg.appendChild(avatar);
  msg.appendChild(body);
  messages.appendChild(msg);
  scrollToBottom();

  return { bubble, sourcesEl, typingEl };
}

function renderSources(container, sources) {
  if (!sources || sources.length === 0) return;

  const toggle = document.createElement('button');
  toggle.className = 'sources-toggle';
  toggle.innerHTML = `📎 ${sources.length} source${sources.length !== 1 ? 's' : ''} <span>▾</span>`;

  const list = document.createElement('div');
  list.className = 'sources-list';

  for (let i = 0; i < sources.length; i++) {
    const s = sources[i];
    const score = ((1 - s.distance) * 100).toFixed(1);
    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-header">
        <span class="source-name">Source ${i + 1} · ${esc(s.source)}</span>
        <span class="source-score">relevance ${score}%</span>
      </div>
      <div class="source-excerpt">${esc(s.text)}</div>
    `;
    list.appendChild(card);
  }

  toggle.addEventListener('click', () => {
    list.classList.toggle('open');
    toggle.querySelector('span').textContent = list.classList.contains('open') ? '▴' : '▾';
  });

  container.appendChild(toggle);
  container.appendChild(list);
}

// ── Markdown renderer (minimal, no deps) ──────────────────────────────────────

function renderMarkdown(md) {
  return md
    // Escape HTML first
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // Code blocks
    .replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Headings
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
    // Ordered list
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, m => `<ol>${m}</ol>`)
    // Unordered list
    .replace(/^[*\-] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, m => {
      if (m.includes('<ol>')) return m;
      return `<ul>${m}</ul>`;
    })
    // Paragraphs
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(.+)$/, '<p>$1</p>');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

// ── Fetch model name from health endpoint ─────────────────────────────────────

async function fetchHealth() {
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      const data = await res.json();
      modelBadge.textContent = data.model || 'claude-opus-4-6';
    }
  } catch { /* server not up yet */ }
}

// ── Init ──────────────────────────────────────────────────────────────────────

setupUpload();
setupChat();
loadDocuments();
fetchHealth();
