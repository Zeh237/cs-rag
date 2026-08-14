// ─── API Base & Auth State ────────────────────────────────────────────────────
const API_BASE = '';
let authToken = localStorage.getItem('csc_token') || null;
let currentUser = JSON.parse(localStorage.getItem('csc_user') || 'null');
let activeChatId = null;
let capabilities = null;

// ─── Auth Helpers ─────────────────────────────────────────────────────────────
function authHeaders() {
    return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` };
}

function saveSession(token, user) {
    authToken = token;
    currentUser = user;
    localStorage.setItem('csc_token', token);
    localStorage.setItem('csc_user', JSON.stringify(user));
}

function clearSession() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem('csc_token');
    localStorage.removeItem('csc_user');
}

// ─── Auth Overlay Logic ───────────────────────────────────────────────────────
const authOverlay  = document.getElementById('authOverlay');
const appShell     = document.getElementById('appShell');
const loginPanel   = document.getElementById('loginPanel');
const signupPanel  = document.getElementById('signupPanel');
const loginForm    = document.getElementById('loginForm');
const signupForm   = document.getElementById('signupForm');
const loginError   = document.getElementById('loginError');
const signupError  = document.getElementById('signupError');
const goToSignup   = document.getElementById('goToSignup');
const goToLogin    = document.getElementById('goToLogin');
const logoutBtn    = document.getElementById('logoutBtn');
const userPillName = document.getElementById('userPillName');

function showAuthPanel(panel) {
    loginPanel.classList.add('hidden');
    signupPanel.classList.add('hidden');
    loginError.classList.add('hidden');
    signupError.classList.add('hidden');
    loginError.textContent = '';
    signupError.textContent = '';
    panel.classList.remove('hidden');
}

function showApp() {
    authOverlay.classList.add('hidden');
    appShell.classList.remove('hidden');
    if (currentUser) userPillName.textContent = currentUser.username;
    checkHealth();
    loadChatList();
}

function showAuthOverlay() {
    authOverlay.classList.remove('hidden');
    appShell.classList.add('hidden');
    showAuthPanel(loginPanel);
}

function setAuthBtnLoading(btn, loading) {
    const label   = btn.querySelector('.auth-btn-label');
    const spinner = btn.querySelector('.auth-btn-spinner');
    btn.disabled = loading;
    label.classList.toggle('hidden', loading);
    spinner.classList.toggle('hidden', !loading);
}

function showFormError(el, msg) {
    el.textContent = msg;
    el.classList.remove('hidden');
}

// Switch panels
goToSignup?.addEventListener('click', () => showAuthPanel(signupPanel));
goToLogin?.addEventListener('click',  () => showAuthPanel(loginPanel));

// Login
loginForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    loginError.classList.add('hidden');
    const btn = document.getElementById('loginBtn');
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    if (!username || !password) { showFormError(loginError, 'Please fill in all fields.'); return; }
    setAuthBtnLoading(btn, true);
    try {
        const res = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (!res.ok) { showFormError(loginError, data.detail || 'Login failed'); return; }
        // fetch /me to get user object
        const meRes = await fetch(`${API_BASE}/auth/me`, {
            headers: { 'Authorization': `Bearer ${data.access_token}` }
        });
        const user = await meRes.json();
        saveSession(data.access_token, user);
        showApp();
    } catch (err) {
        showFormError(loginError, 'Network error. Please try again.');
    } finally {
        setAuthBtnLoading(btn, false);
    }
});

// Signup
signupForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    signupError.classList.add('hidden');
    const btn = document.getElementById('signupBtn');
    const username = document.getElementById('signupUsername').value.trim();
    const password = document.getElementById('signupPassword').value;
    if (!username || !password) { showFormError(signupError, 'Please fill in all fields.'); return; }
    if (password.length < 6) { showFormError(signupError, 'Password must be at least 6 characters.'); return; }
    setAuthBtnLoading(btn, true);
    try {
        const res = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (!res.ok) { showFormError(signupError, data.detail || 'Registration failed'); return; }
        // Auto-login after registration
        const loginRes = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const loginData = await loginRes.json();
        saveSession(loginData.access_token, data);
        showApp();
    } catch (err) {
        showFormError(signupError, 'Network error. Please try again.');
    } finally {
        setAuthBtnLoading(btn, false);
    }
});

// Logout
logoutBtn?.addEventListener('click', () => {
    clearSession();
    activeChatId = null;
    showAuthOverlay();
});

// Password visibility toggles
document.querySelectorAll('.auth-eye-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const input = document.getElementById(btn.dataset.target);
        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';
        btn.querySelector('.eye-icon').classList.toggle('hidden', isPassword);
        btn.querySelector('.eye-off-icon').classList.toggle('hidden', !isPassword);
    });
});

// ─── Bootstrap ────────────────────────────────────────────────────────────────
if (authToken && currentUser) {
    showApp();
} else {
    showAuthOverlay();
}

// ─── Tab Switching ────────────────────────────────────────────────────────────
const tabBtns     = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

function showTab(tabName) {
    tabBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabName));
    tabContents.forEach(content => content.classList.toggle('active', content.id === `${tabName}-tab`));
}
tabBtns.forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));

// ─── Health Check ─────────────────────────────────────────────────────────────
async function checkHealth() {
    const indicator = document.getElementById('healthIndicator');
    if (!indicator) return;
    try {
        const res  = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        capabilities = data;
        const ok   = data.status === 'ok' && data.elasticsearch;
        indicator.querySelector('.dot').style.background = ok ? '#4ade80' : '#f87171';
        indicator.querySelector('span:last-child').textContent = ok ? 'Connected' : 'Disconnected';
        updateModeBar();
    } catch {
        indicator.querySelector('.dot').style.background = '#f87171';
        indicator.querySelector('span:last-child').textContent = 'Offline';
    }
}

function updateModeBar() {
    const researchLabel = document.getElementById('researchModeLabel');
    const hintEl        = document.getElementById('chatModeHint');
    const researchRadio = researchLabel?.querySelector('input[value="research"]');

    if (capabilities?.vision_enabled) {
        researchLabel?.classList.remove('disabled');
        researchLabel?.setAttribute('title', `Vision model reads PDF pages as images (${capabilities.vision_model || 'vision model'})`);
        if (hintEl) hintEl.textContent = `Vision: ${capabilities.vision_model || 'on'} · LLM: ${capabilities.llm_model || 'n/a'}`;
    } else {
        researchLabel?.classList.add('disabled');
        researchLabel?.setAttribute('title', 'Vision model not configured — research mode unavailable');
        if (researchRadio) researchRadio.disabled = true;
        if (hintEl) hintEl.textContent = capabilities?.llm_model ? `LLM: ${capabilities.llm_model}` : '';
    }
}

// ─── Chat List (Sidebar) ──────────────────────────────────────────────────────
const chatListEl      = document.getElementById('chatList');
const newChatModal    = document.getElementById('newChatModal');
const newChatTitleEl  = document.getElementById('newChatTitle');
const chatNoSelection = document.getElementById('chatNoSelection');
const chatActive      = document.getElementById('chatActive');
const chatTitleLabel  = document.getElementById('chatTitleLabel');
const chatMessages    = document.getElementById('chatMessages');
const chatWelcome     = document.getElementById('chatWelcome');

let chatListData = [];

async function loadChatList() {
    if (!authToken) return;
    try {
        const res   = await fetch(`${API_BASE}/chats`, { headers: authHeaders() });
        if (res.status === 401) { clearSession(); showAuthOverlay(); return; }
        chatListData = await res.json();
        renderChatList();
    } catch (e) {
        console.error('Failed to load chats', e);
    }
}

function renderChatList() {
    if (!chatListEl) return;
    if (chatListData.length === 0) {
        chatListEl.innerHTML = '<p class="chat-list-empty">No chats yet.<br>Click + to start one.</p>';
        return;
    }
    chatListEl.innerHTML = chatListData.map(c => `
        <div class="chat-list-item${c.id === activeChatId ? ' active' : ''}" data-id="${c.id}" id="chatItem-${c.id}">
            <span class="chat-list-item-name" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</span>
        </div>
    `).join('');
    chatListEl.querySelectorAll('.chat-list-item').forEach(item => {
        item.addEventListener('click', () => selectChat(Number(item.dataset.id)));
    });
}

function selectChat(chatId) {
    const chat = chatListData.find(c => c.id === chatId);
    if (!chat) return;
    activeChatId = chatId;
    renderChatList();
    chatNoSelection?.classList.add('hidden');
    chatActive?.classList.remove('hidden');
    if (chatTitleLabel) chatTitleLabel.textContent = chat.title;
    // Clear messages area and reset to welcome
    if (chatMessages) {
        chatMessages.innerHTML = '';
        const welcome = document.createElement('div');
        welcome.className = 'chat-welcome';
        welcome.id = 'chatWelcome';
        welcome.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="40" height="40">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                <line x1="12" y1="22.08" x2="12" y2="12"/>
            </svg>
            <p>Ask me anything about your indexed documents.</p>`;
        chatMessages.appendChild(welcome);
    }
    // Load existing history from API
    loadChatHistory(chatId);
}

async function loadChatHistory(chatId) {
    try {
        const res  = await fetch(`${API_BASE}/chats/${chatId}`, { headers: authHeaders() });
        const data = await res.json();
        if (data.history && data.history.length > 0) {
            // Remove welcome
            const welcome = chatMessages?.querySelector('.chat-welcome');
            if (welcome) welcome.remove();
            // Render history
            data.history.forEach(msg => {
                if (msg.role === 'user') {
                    addChatMessage('user', escapeHtml(msg.content).replace(/\n/g, '<br>'));
                } else {
                    addChatMessage('assistant', renderMarkdown(msg.content));
                }
            });
        }
    } catch (e) {
        console.error('Failed to load chat history', e);
    }
}

// ─── New Chat Modal ───────────────────────────────────────────────────────────
function openNewChatModal() {
    if (newChatTitleEl) newChatTitleEl.value = '';
    newChatModal?.classList.remove('hidden');
    setTimeout(() => newChatTitleEl?.focus(), 50);
}

function closeNewChatModal() {
    newChatModal?.classList.add('hidden');
}

document.getElementById('newChatBtn')?.addEventListener('click', openNewChatModal);
document.getElementById('newChatBtnAlt')?.addEventListener('click', openNewChatModal);
document.getElementById('cancelNewChat')?.addEventListener('click', closeNewChatModal);
newChatModal?.addEventListener('click', (e) => { if (e.target === newChatModal) closeNewChatModal(); });
newChatTitleEl?.addEventListener('keydown', (e) => { if (e.key === 'Enter') document.getElementById('confirmNewChat')?.click(); });

document.getElementById('confirmNewChat')?.addEventListener('click', async () => {
    const title = newChatTitleEl?.value.trim() || 'New Chat';
    try {
        const res  = await fetch(`${API_BASE}/chats`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({ title }),
        });
        if (res.status === 401) { clearSession(); showAuthOverlay(); return; }
        const newChat = await res.json();
        chatListData.unshift(newChat);
        closeNewChatModal();
        renderChatList();
        selectChat(newChat.id);
        showTab('query');
    } catch (e) {
        console.error('Failed to create chat', e);
    }
});

// ─── Delete Chat ──────────────────────────────────────────────────────────────
document.getElementById('deleteChatBtn')?.addEventListener('click', async () => {
    if (!activeChatId) return;
    const chat = chatListData.find(c => c.id === activeChatId);
    if (!confirm(`Delete "${chat?.title}"? This will clear all its history.`)) return;
    try {
        await fetch(`${API_BASE}/chats/${activeChatId}`, { method: 'DELETE', headers: authHeaders() });
        chatListData = chatListData.filter(c => c.id !== activeChatId);
        activeChatId = null;
        renderChatList();
        chatNoSelection?.classList.remove('hidden');
        chatActive?.classList.add('hidden');
    } catch (e) {
        console.error('Failed to delete chat', e);
    }
});

// ─── Ingest Tab ───────────────────────────────────────────────────────────────
let selectedFile = null;
const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const fileList     = document.getElementById('fileList');
const ingestBtn    = document.getElementById('ingestBtn');
const ingestProgress = document.getElementById('ingestProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const ingestResult = document.getElementById('ingestResult');

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
    const ok = ['.pdf', '.md', '.txt'].some(ext => file.name.toLowerCase().endsWith(ext));
    if (!ok) { showIngestResult('Only PDF, Markdown, or Text files are supported.', true); return; }
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
    const hasFile   = !!selectedFile;
    const titleVal  = document.getElementById('ingestTitle')?.value.trim();
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
    const title           = document.getElementById('ingestTitle')?.value.trim();
    const authors         = document.getElementById('ingestAuthors')?.value.trim();
    const primaryCategory = document.getElementById('ingestCategory')?.value.trim();
    const allCategories   = document.getElementById('ingestAllCategories')?.value.trim() || primaryCategory;
    const url             = document.getElementById('ingestUrl')?.value.trim() || '';
    const useVision       = document.getElementById('ingestUseVision')?.checked || false;

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
    formData.append('use_vision', useVision);
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
            xhr.setRequestHeader('Authorization', `Bearer ${authToken}`);
            xhr.onload = () => {
                try { resolve({ status: xhr.status, data: JSON.parse(xhr.responseText) }); }
                catch { reject(new Error('Invalid JSON response')); }
            };
            xhr.onerror = () => reject(new Error('Network error'));
            xhr.send(formData);
        });
        progressFill.style.width = '100%';
        progressText.textContent = 'Upload complete. Indexing in background...';
        if (response.status === 202) {
            const final = await pollIngestStatus(response.data.job_id);
            if (final.status === 'completed') {
                showIngestResult(`✓ ${final.message} — ${final.chunks_indexed} chunks indexed.`);
                selectedFile = null;
                if (fileList) fileList.innerHTML = '';
                ['ingestTitle', 'ingestAuthors', 'ingestCategory', 'ingestAllCategories', 'ingestUrl'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.value = '';
                });
                const visionEl = document.getElementById('ingestUseVision');
                if (visionEl) visionEl.checked = false;
            } else {
                showIngestResult(final.error || final.message || 'Ingestion failed', true);
            }
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

async function pollIngestStatus(jobId) {
    for (let attempt = 0; attempt < 300; attempt++) {
        try {
            const res = await fetch(`${API_BASE}/api/v1/ingest/status/${jobId}`, { headers: authHeaders() });
            const data = await res.json();
            if (data.status === 'completed' || data.status === 'failed' || data.status === 'error') {
                return data;
            }
            progressText.textContent = `Indexing... ${data.message || ''}`;
        } catch { /* transient error, keep polling */ }
        await new Promise(resolve => setTimeout(resolve, 1500));
    }
    return { status: 'failed', error: 'Timed out waiting for indexing to finish.' };
}

if (ingestBtn) ingestBtn.addEventListener('click', ingestFiles);

// ─── Search Tab ───────────────────────────────────────────────────────────────
const searchInput   = document.getElementById('searchInput');
const searchBtn     = document.getElementById('searchBtn');
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
            const res  = await fetch(`${API_BASE}/search/${endpoint}?q=${encodeURIComponent(query)}&limit=10`);
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
    if (results.length === 0) { searchResults.innerHTML = '<div class="no-results">No results found.</div>'; return; }
    searchResults.innerHTML = results.map(r => {
        const hasLocalPdf  = !!r.local_pdf;
        const hasSourceUrl = !!r.source_url;
        const hasPdf       = !!r.paper_id;
        const pageNum      = r.page_number || 1;
        const chunkParam   = r.chunk_id ? `?chunk_id=${encodeURIComponent(r.chunk_id)}` : '';
        let pdfUrl = null, openUrl = null, badge = '';
        if (hasLocalPdf) {
            pdfUrl = `/api/v1/pdf/${encodeURIComponent(r.paper_id)}${chunkParam}#page=${pageNum}`;
            badge = `<span class="source-page-badge">PDF <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:10px;height:10px;vertical-align:middle;"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></span>`;
        } else if (hasSourceUrl) {
            openUrl = r.source_url;
            badge = `<span class="source-page-badge">source <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:10px;height:10px;vertical-align:middle;"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></span>`;
        } else if (hasPdf) {
            pdfUrl = `/api/v1/pdf/${encodeURIComponent(r.paper_id)}${chunkParam}#page=${pageNum}`;
            badge = `<span class="source-page-badge">p.${pageNum} <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:10px;height:10px;vertical-align:middle;"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></span>`;
        }
        const clickable = openUrl || pdfUrl
            ? `style="cursor:pointer;" onclick="window.open('${escapeHtml(openUrl || pdfUrl)}','_blank')" title="${openUrl ? 'Open source' : `Open PDF`}"`
            : '';
        return `
        <div class="source-item${(openUrl || pdfUrl) ? ' source-item--clickable' : ''}" ${clickable}>
            <div class="source-header">
                <span class="source-title">${escapeHtml(r.title || 'Untitled')}</span>
                <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                    ${badge}
                    <span class="source-score">${typeof r.score === 'number' ? r.score.toFixed(3) : ''}</span>
                </div>
            </div>
            ${r.authors ? `<div class="source-meta">${escapeHtml(Array.isArray(r.authors) ? r.authors.join(', ') : r.authors)}</div>` : ''}
            <div class="source-text">${escapeHtml(r.text || '')}</div>
        </div>
    `;
    }).join('');
}

// ─── Chat (RAG) ───────────────────────────────────────────────────────────────
const questionInput = document.getElementById('questionInput');
const queryBtn      = document.getElementById('queryBtn');

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

// Configure marked
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
                return katex.renderToString(block.math, { displayMode: block.display, throwOnError: false, trust: false });
            } catch {
                return `<code class="math-error">${escapeHtml(block.math)}</code>`;
            }
        }
        return `<code>${escapeHtml(block.math)}</code>`;
    });
    return html;
}

function linkifyCitations(md, citations) {
    if (!citations || !citations.length) return md;
    return md.replace(/\[(\d+)\](?!\()/g, (match, idx) => {
        const n = parseInt(idx, 10);
        const c = citations[n - 1];
        if (!c) return match;
        const url = `/api/v1/pdf/${encodeURIComponent(c.paper_id)}?chunk_id=${encodeURIComponent(c.chunk_id)}#page=${c.page}`;
        return `[${idx}](${url})`;
    });
}

function linkifyPageRefs(md, paperId) {
    if (!paperId) return md;
    return md.replace(/\(p{1,2}\.\s*(\d+(?:\s*[-–]\s*\d+)?)\)/g, (match, body) => {
        const page = (body.match(/\d+/) || [])[0];
        if (!page) return match;
        const url = `/api/v1/pdf/${encodeURIComponent(paperId)}#page=${page}`;
        return `[(p. ${body})](${url})`;
    });
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
    bubble.innerHTML = '<span class="chat-thinking-text">Thinking</span><span class="chat-typing-dot"></span><span class="chat-typing-dot"></span><span class="chat-typing-dot"></span>';
    msg.appendChild(bubble);
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function removeTypingIndicator() {
    document.getElementById('typingIndicator')?.remove();
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
        const hasLocalPdf  = !!src.local_pdf;
        const hasSourceUrl = !!src.source_url;
        const hasPdf  = !!src.paper_id;
        const chunkParam = src.anchor_chunk_id ? `?chunk_id=${encodeURIComponent(src.anchor_chunk_id)}` : '';
        const pdfUrl  = (hasLocalPdf || hasPdf) ? `/api/v1/pdf/${encodeURIComponent(src.paper_id)}${chunkParam}#page=${pageNum}` : null;
        const openUrl = (!hasLocalPdf && hasSourceUrl) ? src.source_url : null;
        const item = document.createElement('div');
        item.className = 'chat-source-item' + ((openUrl || pdfUrl) ? ' chat-source-item--clickable' : '');
        if (openUrl) {
            item.title = 'Open source';
            item.style.cursor = 'pointer';
            item.addEventListener('click', () => window.open(openUrl, '_blank'));
        } else if (pdfUrl) {
            item.title = 'Open PDF';
            item.style.cursor = 'pointer';
            item.addEventListener('click', () => window.open(pdfUrl, '_blank'));
        }
        item.innerHTML = `
            <span class="chat-source-title">${escapeHtml(src.title || 'Document')}</span>
            <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                ${hasLocalPdf ? '<span class="source-page-badge">PDF</span>' : (openUrl ? '<span class="source-page-badge">source</span>' : (hasPdf ? `<span class="source-page-badge">p.${pageNum}</span>` : ''))}
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
        if (!activeChatId) {
            addChatError('Please select or create a chat first.');
            return;
        }
        const question = questionInput?.value.trim();
        if (!question) return;

        questionInput.value = '';
        questionInput.style.height = 'auto';
        queryBtn.disabled = true;

        addChatMessage('user', escapeHtml(question).replace(/\n/g, '<br>'));
        addTypingIndicator();

        let mdBuffer   = '';
        let sourcesData = null;
        let citationsData = null;
        let researchInfo = null;
        let assistantMsg = null;
        let bubble = null;

        try {
            const modeRadio = document.querySelector('input[name="answerMode"]:checked');
            const mode = modeRadio ? modeRadio.value : 'auto';
            const modeParam = mode === 'auto' ? '' : `&mode=${mode}`;
            const url = `${API_BASE}/rag/ask/stream?q=${encodeURIComponent(question)}&chat_id=${activeChatId}&limit=4&window_size=2${modeParam}`;
            const res = await fetch(url, { headers: { 'Authorization': `Bearer ${authToken}` } });

            if (res.status === 401) { clearSession(); showAuthOverlay(); return; }
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${res.status}`);
            }

            const reader  = res.body.getReader();
            const decoder = new TextDecoder();
            let sseBuffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                sseBuffer += decoder.decode(value, { stream: true });
                const parts = sseBuffer.split('\n\n');
                sseBuffer = parts.pop();
                for (const part of parts) {
                    if (!part.trim()) continue;
                    const lines = part.split('\n');
                    let eventType = 'message';
                    let dataLine  = '';
                    for (const line of lines) {
                        if (line.startsWith('event:')) eventType = line.slice(6).trim();
                        else if (line.startsWith('data:')) dataLine = line.slice(5).trim();
                    }
                    if (!dataLine) continue;

                    if (eventType === 'sources') {
                        try { sourcesData = JSON.parse(dataLine); } catch (e) { console.error('Sources parse error', e); }
                    } else if (eventType === 'citations') {
                        try { citationsData = JSON.parse(dataLine); } catch (e) { console.error('Citations parse error', e); }
                    } else if (eventType === 'research') {
                        try {
                            researchInfo = JSON.parse(dataLine);
                            const t = document.querySelector('#typingIndicator .chat-thinking-text');
                            if (t) t.textContent = `Researching ${researchInfo.title || 'the document'}…`;
                        } catch (e) { console.error('Research parse error', e); }
                    } else if (eventType === 'token') {
                        try {
                            const token = JSON.parse(dataLine);
                            mdBuffer += token;
                            if (!assistantMsg) {
                                removeTypingIndicator();
                                assistantMsg = addChatMessage('assistant', '');
                                bubble = assistantMsg.querySelector('.chat-bubble');
                            }
                            if (bubble) {
                                bubble.innerHTML = renderMarkdown(linkifyCitations(linkifyPageRefs(mdBuffer, researchInfo?.paper_id), citationsData)) + '<span class="stream-cursor"></span>';
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } catch (e) { console.error('Token parse error', e); }
                    } else if (eventType === 'error') {
                        removeTypingIndicator();
                        try { addChatError(JSON.parse(dataLine)); } catch { addChatError('Stream error'); }
                    } else if (eventType === 'done') {
                        removeTypingIndicator();
                        if (mdBuffer && bubble) {
                            bubble.innerHTML = renderMarkdown(linkifyCitations(linkifyPageRefs(mdBuffer, researchInfo?.paper_id), citationsData));
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
            console.error('Stream fetch error:', error);
        } finally {
            if (queryBtn) queryBtn.disabled = !questionInput?.value.trim();
            chatMessages?.querySelectorAll('.stream-cursor').forEach(el => el.remove());
        }
    });
}

// ─── Utilities ─────────────────────────────────────────────────────────────────
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