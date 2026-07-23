// ─── API Base ───────────────────────────────────────────────────────────────
const API_BASE = '';

// ─── Tab switching ───────────────────────────────────────────────────────────
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

function showTab(tabName) {
    tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });
}

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
});

// ─── Health Check ────────────────────────────────────────────────────────────
async function checkHealth() {
    const indicator = document.getElementById('healthIndicator');
    if (!indicator) return;
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        const ok = data.status === 'ok' && data.elasticsearch;
        indicator.querySelector('.dot').style.background = ok ? '#4ade80' : '#f87171';
        indicator.querySelector('span:last-child').textContent = ok ? 'Connected' : 'Disconnected';
    } catch {
        indicator.querySelector('.dot').style.background = '#f87171';
        indicator.querySelector('span:last-child').textContent = 'Offline';
    }
}
checkHealth();

// ─── Ingest Tab ──────────────────────────────────────────────────────────────
let selectedFile = null;
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const fileList = document.getElementById('fileList');
const ingestBtn = document.getElementById('ingestBtn');
const ingestProgress = document.getElementById('ingestProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const ingestResult = document.getElementById('ingestResult');

if (browseBtn) browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
if (dropZone) {
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });
}
if (fileInput) fileInput.addEventListener('change', (e) => { handleFile(e.target.files[0]); fileInput.value = ''; });

function handleFile(file) {
    if (!file) return;
    if (!file.name.endsWith('.pdf')) {
        showIngestResult('Only PDF files are supported.', true);
        return;
    }
    selectedFile = file;
    renderFileItem(file);
    updateIngestButton();
}

function renderFileItem(file) {
    if (!fileList) return;
    const size = formatFileSize(file.size);
    fileList.innerHTML = `
        <div class="file-item">
            <div class="file-info">
                <div class="file-icon pdf">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                    </svg>
                </div>
                <div class="file-details">
                    <div class="file-name">${escapeHtml(file.name)}</div>
                    <div class="file-size">${size}</div>
                </div>
            </div>
            <button class="file-remove" id="removeFileBtn" aria-label="Remove file">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        </div>
    `;
    document.getElementById('removeFileBtn').addEventListener('click', () => {
        selectedFile = null;
        fileList.innerHTML = '';
        updateIngestButton();
    });
}

function updateIngestButton() {
    if (!ingestBtn) return;
    const hasFile = !!selectedFile;
    const titleVal = document.getElementById('ingestTitle')?.value.trim();
    const authorsVal = document.getElementById('ingestAuthors')?.value.trim();
    const categoryVal = document.getElementById('ingestCategory')?.value.trim();
    ingestBtn.disabled = !(hasFile && titleVal && authorsVal && categoryVal);
}

['ingestTitle', 'ingestAuthors', 'ingestCategory', 'ingestAllCategories'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateIngestButton);
});

function showIngestResult(message, isError = false) {
    if (!ingestResult) return;
    ingestResult.textContent = message;
    ingestResult.className = `result ${isError ? 'error' : ''}`;
    ingestResult.classList.remove('hidden');
}

async function ingestFiles() {
    if (!selectedFile) return;

    const title = document.getElementById('ingestTitle')?.value.trim();
    const authors = document.getElementById('ingestAuthors')?.value.trim();
    const primaryCategory = document.getElementById('ingestCategory')?.value.trim();
    const allCategories = document.getElementById('ingestAllCategories')?.value.trim() || primaryCategory;
    const url = document.getElementById('ingestUrl')?.value.trim() || '';

    if (!title || !authors || !primaryCategory) {
        showIngestResult('Please fill in all required fields (Title, Authors, Category).', true);
        return;
    }

    ingestBtn.disabled = true;
    ingestProgress.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading document...';
    ingestResult.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('title', title);
    formData.append('authors', authors);
    formData.append('primary_category', primaryCategory);
    formData.append('all_categories', allCategories);
    if (url) formData.append('url', url);

    try {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = percent + '%';
                progressText.textContent = `Uploading... ${percent}%`;
            }
        });

        const response = await new Promise((resolve, reject) => {
            xhr.open('POST', `${API_BASE}/api/v1/ingest`);
            xhr.onload = () => {
                try {
                    resolve({ status: xhr.status, data: JSON.parse(xhr.responseText) });
                } catch {
                    reject(new Error('Invalid JSON response'));
                }
            };
            xhr.onerror = () => reject(new Error('Network error'));
            xhr.send(formData);
        });

        progressFill.style.width = '100%';
        progressText.textContent = 'Processing complete!';

        if (response.status === 200) {
            const msg = response.data.message || `"${title}" ingested`;
            showIngestResult(`✓ ${msg} — ${response.data.chunks_indexed} chunks indexed.`);
            selectedFile = null;
            if (fileList) fileList.innerHTML = '';
            ['ingestTitle', 'ingestAuthors', 'ingestCategory', 'ingestAllCategories', 'ingestUrl'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            updateIngestButton();
        } else {
            showIngestResult(response.data.detail || 'Ingestion failed', true);
        }
    } catch (error) {
        showIngestResult('Error: ' + error.message, true);
    } finally {
        setTimeout(() => {
            ingestProgress.classList.add('hidden');
            updateIngestButton();
        }, 1500);
    }
}

if (ingestBtn) ingestBtn.addEventListener('click', ingestFiles);

// ─── Search Tab ───────────────────────────────────────────────────────────────
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchResults = document.getElementById('searchResults');

if (searchBtn) searchBtn.addEventListener('click', runSearch);
if (searchInput) searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(); });

async function runSearch() {
    const query = searchInput?.value.trim();
    if (!query) return;

    const mode = document.querySelector('input[name="searchMode"]:checked')?.value || 'keyword';

    if (searchBtn) searchBtn.disabled = true;
    if (searchResults) searchResults.innerHTML = '<div class="search-loading"><div class="spinner"></div><span>Searching...</span></div>';

    try {
        let results = [];

        if (mode === 'hybrid') {
            const [kwRes, semRes] = await Promise.all([
                fetch(`${API_BASE}/search/keyword?q=${encodeURIComponent(query)}&limit=5`).then(r => r.json()),
                fetch(`${API_BASE}/search/semantic?q=${encodeURIComponent(query)}&limit=5`).then(r => r.json())
            ]);
            results = mergeResults(kwRes.results || [], semRes.results || []);
        } else {
            const endpoint = mode === 'semantic' ? 'semantic' : 'keyword';
            const res = await fetch(`${API_BASE}/search/${endpoint}?q=${encodeURIComponent(query)}&limit=10`);
            const data = await res.json();
            if (!res.ok) {
                if (searchResults) searchResults.innerHTML = `<div class="error">${escapeHtml(data.detail || 'Search failed')}</div>`;
                return;
            }
            results = data.results || [];
        }

        renderSearchResults(results);
    } catch (error) {
        if (searchResults) searchResults.innerHTML = `<div class="error">Error: ${escapeHtml(error.message)}</div>`;
    } finally {
        if (searchBtn) searchBtn.disabled = false;
    }
}

function mergeResults(kwResults, semResults) {
    const seen = new Set();
    const merged = [];
    [...kwResults, ...semResults].forEach(r => {
        const key = r.chunk_id || r.title;
        if (!seen.has(key)) { seen.add(key); merged.push(r); }
    });
    return merged;
}

function renderSearchResults(results) {
    if (!searchResults) return;
    if (results.length === 0) {
        searchResults.innerHTML = '<div class="no-results">No results found.</div>';
        return;
    }
    searchResults.innerHTML = results.map(r => {
        const hasPdf = !!r.paper_id;
        const pageNum = r.page_number || 1;
        const chunkParam = r.chunk_id ? `?chunk_id=${encodeURIComponent(r.chunk_id)}` : '';
        const pdfUrl = hasPdf ? `/api/v1/pdf/${encodeURIComponent(r.paper_id)}${chunkParam}#page=${pageNum}` : null;
        const clickable = hasPdf ? `style="cursor:pointer;" onclick="window.open('${pdfUrl}','_blank')" title="Open PDF at page ${pageNum}"` : '';
        const pageBadge = hasPdf
            ? `<span class="source-page-badge">p.${pageNum} <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:10px;height:10px;vertical-align:middle;"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></span>`
            : '';
        return `
        <div class="source-item${hasPdf ? ' source-item--clickable' : ''}" ${clickable}>
            <div class="source-header">
                <span class="source-title">${escapeHtml(r.title || 'Untitled')}</span>
                <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                    ${pageBadge}
                    <span class="source-score">${typeof r.score === 'number' ? r.score.toFixed(3) : ''}</span>
                </div>
            </div>
            ${r.authors ? `<div class="source-meta">${escapeHtml(Array.isArray(r.authors) ? r.authors.join(', ') : r.authors)}</div>` : ''}
            <div class="source-text">${escapeHtml(r.text || '')}</div>
        </div>
    `;
    }).join('');
}

// ─── Query (RAG) Chat Tab ─────────────────────────────────────────────────────
const questionInput = document.getElementById('questionInput');
const queryBtn = document.getElementById('queryBtn');
const chatMessages = document.getElementById('chatMessages');

if (questionInput) {
    questionInput.addEventListener('input', () => {
        if (queryBtn) queryBtn.disabled = !questionInput.value.trim();
        autoResizeTextarea(questionInput);
    });
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (queryBtn && !queryBtn.disabled) queryBtn.click();
        }
    });
}

function autoResizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// Configure marked once at load time
if (typeof marked !== 'undefined') {
    marked.setOptions({ gfm: true, breaks: true });
}

function renderMarkdown(md) {
    const mathBlocks = [];
    const placeholder = (i) => `\x02MATH${i}\x03`;
    let processed = md;

    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
        const id = mathBlocks.length;
        mathBlocks.push({ display: true, math: math.trim() });
        return placeholder(id);
    });
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
        const id = mathBlocks.length;
        mathBlocks.push({ display: true, math: math.trim() });
        return placeholder(id);
    });
    processed = processed.replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => {
        const id = mathBlocks.length;
        mathBlocks.push({ display: false, math: math.trim() });
        return placeholder(id);
    });
    processed = processed.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
        const id = mathBlocks.length;
        mathBlocks.push({ display: false, math: math.trim() });
        return placeholder(id);
    });

    let html;
    if (typeof marked !== 'undefined') {
        html = marked.parse(processed);
    } else {
        const div = document.createElement('div');
        div.textContent = processed;
        html = '<p>' + div.innerHTML.replace(/\n/g, '<br>') + '</p>';
    }

    html = html.replace(/\x02MATH(\d+)\x03/g, (_, idx) => {
        const block = mathBlocks[parseInt(idx)];
        if (!block) return '';
        if (typeof katex !== 'undefined') {
            try {
                return katex.renderToString(block.math, {
                    displayMode: block.display,
                    throwOnError: false,
                    trust: false,
                });
            } catch {
                return `<code class="math-error">${escapeHtml(block.math)}</code>`;
            }
        }
        return `<code>${escapeHtml(block.math)}</code>`;
    });

    return html;
}

function addChatMessage(role, content) {
    if (!chatMessages) return null;
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = `chat-message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    if (role === 'assistant') bubble.classList.add('prose');
    bubble.innerHTML = content;
    msg.appendChild(bubble);

    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function addTypingIndicator() {
    if (!chatMessages) return null;
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = 'chat-message assistant';
    msg.id = 'typingIndicator';

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble chat-typing';
    bubble.innerHTML = '<span class="chat-typing-dot"></span><span class="chat-typing-dot"></span><span class="chat-typing-dot"></span>';
    msg.appendChild(bubble);

    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

function addChatError(message) {
    if (!chatMessages) return;
    const err = document.createElement('div');
    err.className = 'chat-error';
    err.textContent = message;
    chatMessages.appendChild(err);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderChatSources(sources) {
    if (!sources || sources.length === 0) return null;

    const container = document.createElement('div');
    container.className = 'chat-sources';

    const toggle = document.createElement('div');
    toggle.className = 'chat-sources-toggle';
    toggle.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg> <span>${sources.length} source${sources.length !== 1 ? 's' : ''}</span>`;

    const list = document.createElement('div');
    list.className = 'chat-sources-list collapsed';

    sources.forEach(src => {
        const pageNum = src.page_number || 1;
        const hasPdf = !!src.paper_id;
        const chunkParam = src.anchor_chunk_id ? `?chunk_id=${encodeURIComponent(src.anchor_chunk_id)}` : '';
        const pdfUrl = hasPdf ? `/api/v1/pdf/${encodeURIComponent(src.paper_id)}${chunkParam}#page=${pageNum}` : null;

        const item = document.createElement('div');
        item.className = 'chat-source-item' + (hasPdf ? ' chat-source-item--clickable' : '');
        if (hasPdf) {
            item.title = `Open PDF at page ${pageNum}`;
            item.style.cursor = 'pointer';
            item.addEventListener('click', () => window.open(pdfUrl, '_blank'));
        }
        item.innerHTML = `
            <span class="chat-source-title">${escapeHtml(src.title || 'Document')}</span>
            <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                ${hasPdf ? `<span class="source-page-badge">p.${pageNum}</span>` : ''}
                <span class="chat-source-score">${typeof src.matching_score === 'number' ? src.matching_score.toFixed(3) : ''}</span>
            </div>
        `;
        list.appendChild(item);
    });

    toggle.addEventListener('click', () => {
        list.classList.toggle('collapsed');
        toggle.classList.toggle('open');
    });

    container.appendChild(toggle);
    container.appendChild(list);
    return container;
}

if (queryBtn) {
    queryBtn.addEventListener('click', async () => {
        const question = questionInput?.value.trim();
        if (!question) return;

        questionInput.value = '';
        questionInput.style.height = 'auto';
        queryBtn.disabled = true;

        addChatMessage('user', escapeHtml(question).replace(/\n/g, '<br>'));
        addTypingIndicator();

        let mdBuffer = '';
        let sourcesData = null;

        try {
            const res = await fetch(`${API_BASE}/rag/ask/stream?q=${encodeURIComponent(question)}&limit=3&window_size=1`);

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let sseBuffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                sseBuffer += decoder.decode(value, { stream: true });
                const parts = sseBuffer.split('\n\n');
                sseBuffer = parts.pop();

                for (const part of parts) {
                    const lines = part.trim().split('\n');
                    let eventType = 'message';
                    let dataLine = '';

                    for (const line of lines) {
                        if (line.startsWith('event:')) eventType = line.slice(6).trim();
                        else if (line.startsWith('data:')) dataLine = line.slice(5).trim();
                    }

                    if (!dataLine) continue;

                    if (eventType === 'sources') {
                        try { sourcesData = JSON.parse(dataLine); } catch {}

                    } else if (eventType === 'token') {
                        try {
                            const token = JSON.parse(dataLine);
                            mdBuffer += token;

                            removeTypingIndicator();
                            let assistantMsg = chatMessages?.querySelector('.chat-message.assistant:last-child');
                            if (!assistantMsg) {
                                assistantMsg = addChatMessage('assistant', '');
                            }
                            const bubble = assistantMsg.querySelector('.chat-bubble');
                            if (bubble) {
                                bubble.innerHTML = renderMarkdown(mdBuffer) + '<span class="stream-cursor"></span>';
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } catch {}

                    } else if (eventType === 'error') {
                        try {
                            removeTypingIndicator();
                            const msg = JSON.parse(dataLine);
                            addChatError(msg);
                        } catch {}

                    } else if (eventType === 'done') {
                        removeTypingIndicator();
                        if (mdBuffer) {
                            let assistantMsg = chatMessages?.querySelector('.chat-message.assistant:last-child');
                            if (!assistantMsg) {
                                assistantMsg = addChatMessage('assistant', '');
                            }
                            const bubble = assistantMsg.querySelector('.chat-bubble');
                            if (bubble) {
                                bubble.innerHTML = renderMarkdown(mdBuffer);
                            }

                            if (sourcesData && sourcesData.length > 0) {
                                const sourcesEl = renderChatSources(sourcesData);
                                if (sourcesEl) assistantMsg.appendChild(sourcesEl);
                            }

                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }
                        break;
                    }
                }
            }
        } catch (error) {
            removeTypingIndicator();
            addChatError('Error: ' + error.message);
        } finally {
            if (queryBtn) queryBtn.disabled = !questionInput?.value.trim();
            chatMessages?.querySelectorAll('.stream-cursor').forEach(el => el.remove());
        }
    });
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text ?? '');
    return div.innerHTML;
}
