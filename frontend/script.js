/* Multilingual AI Assistant — frontend.
 *
 * Two things here are worth reading closely:
 *
 * 1. SSE over fetch(). EventSource cannot issue a POST and cannot send a JSON
 *    body, so we POST and parse the text/event-stream ourselves off a
 *    ReadableStream reader. Frames arrive as "event: <name>\ndata: <json>\n\n".
 *
 * 2. Web Speech API support is genuinely uneven. Speech recognition is
 *    Chrome/Edge/Safari only (webkit-prefixed), and Indian-language coverage is
 *    patchy — te-IN recognition and Telugu TTS voices are missing on most
 *    platforms. Every voice control therefore feature-detects and hides itself
 *    rather than presenting a button that silently does nothing.
 */

const els = {
  app: document.getElementById('app'),
  sidebar: document.getElementById('sidebar'),
  scrim: document.getElementById('scrim'),
  sidebarToggle: document.getElementById('sidebarToggle'),
  sidebarClose: document.getElementById('sidebarClose'),

  messages: document.getElementById('messages'),
  form: document.getElementById('chatForm'),
  input: document.getElementById('input'),
  sendBtn: document.getElementById('sendBtn'),
  micBtn: document.getElementById('micBtn'),
  speakBtn: document.getElementById('speakBtn'),
  clearChatBtn: document.getElementById('clearChatBtn'),

  fileInput: document.getElementById('fileInput'),
  dropzone: document.getElementById('dropzone'),
  docList: document.getElementById('docList'),
  docEmpty: document.getElementById('docEmpty'),
  docCount: document.getElementById('docCount'),
  clearDocsBtn: document.getElementById('clearDocsBtn'),

  langPair: document.getElementById('langPair'),
  langProfile: document.getElementById('langProfile'),
  langSelect: document.getElementById('langSelect'),
  langField: document.querySelector('.side-field'),
  langNote: document.getElementById('langNote'),
  langSummary: document.getElementById('langSummary'),
  cacheStat: document.getElementById('cacheStat'),

  convList: document.getElementById('convList'),
  convEmpty: document.getElementById('convEmpty'),
  convCount: document.getElementById('convCount'),
  newChatBtn: document.getElementById('newChatBtn'),

  langBadge: document.getElementById('langBadge'),
  docBadge: document.getElementById('docBadge'),
  statusBar: document.getElementById('statusBar'),
};

/* The two top-bar badges wrap their value in a .badge-val span so the "lang"/"docs"
 * key label stays put while only the value updates. */
const badgeVal = (badge) => badge.querySelector('.badge-val') || badge;

/* Upload constraints — mirrored from the backend purely for instant client-side
 * feedback. The server remains the authority (see documents.py). */
const ALLOWED_EXTENSIONS = ['.txt', '.md', '.pdf', '.csv', '.json'];
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

/* Stable per-browser id so the backend can remember this user's language pair
 * across sessions without requiring accounts. */
const USER_ID = (() => {
  let id = localStorage.getItem('maa_user_id');
  if (!id) {
    id = 'u_' + Math.random().toString(36).slice(2, 12);
    localStorage.setItem('maa_user_id', id);
  }
  return id;
})();

let history = [];
let speakEnabled = false;
let busy = false;
/* Last detection result, used to pick a TTS voice matching the reply language. */
let lastDetection = null;

/* --- Conversation state ----------------------------------------------------
 * The id is generated here rather than fetched, the same way USER_ID is, so a new
 * chat is usable immediately. The backend creates the row lazily on the first
 * successful message — which is why `conversationSaved` matters: until it flips,
 * there is no row to PATCH a language change onto. */
const newConversationId = () => 'c_' + Math.random().toString(36).slice(2, 12) + Date.now().toString(36);

let conversationId = newConversationId();
let conversationSaved = false;
/* null = auto-detect. Otherwise a key from the server's whitelist. Held per
 * conversation, so switching chats switches language mode with it. */
let languageOverride = null;

/* --- UI helpers ----------------------------------------------------------- */

function setStatus(text, kind = '') {
  if (!text) {
    els.statusBar.hidden = true;
    return;
  }
  els.statusBar.hidden = false;
  els.statusBar.textContent = text;
  els.statusBar.className = 'status-bar ' + kind;
}

function clearWelcome() {
  const welcome = els.messages.querySelector('.welcome');
  if (welcome) welcome.remove();
}

/* Replaces the transcript with an empty-state panel. Built as nodes rather than an
 * HTML string so conversation titles can never be injected into markup. */
function showWelcome(heading, body) {
  els.messages.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'welcome';

  const glyph = document.createElement('div');
  glyph.className = 'welcome-glyph';
  glyph.setAttribute('aria-hidden', 'true');
  glyph.textContent = '◈';

  const title = document.createElement('h2');
  title.textContent = heading;

  const text = document.createElement('p');
  text.textContent = body;

  wrap.append(glyph, title, text);
  els.messages.appendChild(wrap);
}

function addMessage(role, text = '') {
  clearWelcome();
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const label = document.createElement('div');
  label.className = 'msg-role';
  label.textContent = role === 'user' ? 'you' : 'assistant';
  wrap.appendChild(label);

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);

  els.messages.appendChild(wrap);
  scrollToBottom();
  return { wrap, bubble };
}

/* Simple detection/cache chips, plus — when the answer used documents — an
 * expandable citation block listing each retrieved excerpt with its source and a
 * preview of the text. rag_chunks carries {source, score, preview} per chunk.
 *
 * Also attaches this message's own play button. Returns that button (or null when
 * the browser has no speech synthesis) so the caller can drive its state when the
 * global auto-read toggle speaks the message. */
function addMeta(wrap, chips, ragChunks, detection) {
  let playButton = null;

  // The row is worth building for the play button alone, even with no chips.
  if (chips.length || TTS_AVAILABLE) {
    const meta = document.createElement('div');
    meta.className = 'msg-meta';

    if (TTS_AVAILABLE) {
      playButton = buildSpeakButton(wrap, detection);
      meta.appendChild(playButton);
    }

    for (const chip of chips) {
      const span = document.createElement('span');
      span.className = 'chip ' + (chip.kind || '');
      span.textContent = chip.text;
      meta.appendChild(span);
    }
    wrap.appendChild(meta);
  }

  if (ragChunks && ragChunks.length) {
    const details = document.createElement('details');
    details.className = 'sources';

    const summary = document.createElement('summary');
    const sources = [...new Set(ragChunks.map((c) => c.source))];
    summary.textContent = `${ragChunks.length} source${ragChunks.length > 1 ? 's' : ''} · ${sources.join(', ')}`;
    details.appendChild(summary);

    const list = document.createElement('div');
    list.className = 'source-list';
    ragChunks.forEach((chunk, i) => {
      const item = document.createElement('div');
      item.className = 'source-item';

      const head = document.createElement('div');
      head.className = 'source-item-head';
      const name = document.createElement('span');
      name.className = 'src-name';
      name.textContent = `[${i + 1}] ${chunk.source}`;
      const score = document.createElement('span');
      score.className = 'src-score';
      if (typeof chunk.score === 'number') score.textContent = `score ${chunk.score.toFixed(2)}`;
      head.appendChild(name);
      head.appendChild(score);

      const preview = document.createElement('div');
      preview.className = 'source-preview';
      preview.textContent = chunk.preview || '';

      item.appendChild(head);
      item.appendChild(preview);
      list.appendChild(item);
    });
    details.appendChild(list);
    wrap.appendChild(details);
  }

  scrollToBottom();
  return playButton;
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setBusy(value) {
  busy = value;
  els.sendBtn.disabled = value;
  els.sendBtn.firstChild.textContent = value ? '…' : 'Send';
}

/* --- Document status + list ----------------------------------------------- */

async function refreshDocs() {
  try {
    const response = await fetch('/documents/status');
    const data = await response.json();
    renderDocList(data);
  } catch {
    /* The document panel is cosmetic; a failure here must not break the chat. */
  }
}

function renderDocList(data) {
  const docs = data.documents || [];
  els.docList.innerHTML = '';

  // Top-bar badge + sidebar count.
  els.docCount.textContent = String(docs.length);
  if (data.has_documents) {
    badgeVal(els.docBadge).textContent = `${docs.length} · ${data.chunks} chunks`;
    els.docBadge.classList.add('active');
  } else {
    badgeVal(els.docBadge).textContent = '0 · 0 chunks';
    els.docBadge.classList.remove('active');
  }

  els.docEmpty.hidden = docs.length > 0;
  els.clearDocsBtn.hidden = docs.length === 0;

  for (const doc of docs) {
    const li = document.createElement('li');
    li.className = 'doc-row';

    const glyph = document.createElement('span');
    glyph.className = 'doc-glyph';
    glyph.textContent = '▸';

    const name = document.createElement('span');
    name.className = 'doc-name';
    name.textContent = doc.filename;
    name.title = doc.filename;

    const chunks = document.createElement('span');
    chunks.className = 'doc-chunks';
    chunks.textContent = `${doc.chunks}ch`;

    const del = document.createElement('button');
    del.className = 'doc-del';
    del.type = 'button';
    del.textContent = '✕';
    del.title = `Remove ${doc.filename}`;
    del.setAttribute('aria-label', `Remove ${doc.filename}`);
    // doc_id may be empty for legacy chunks stored before it was tracked; those
    // can only be removed via "Clear all".
    if (doc.doc_id) {
      del.addEventListener('click', () => deleteDoc(doc.doc_id, doc.filename, del));
    } else {
      del.disabled = true;
      del.title = 'Legacy document — use "Clear all documents"';
    }

    li.append(glyph, name, chunks, del);
    els.docList.appendChild(li);
  }
}

async function deleteDoc(docId, filename, button) {
  button.disabled = true;
  try {
    const response = await fetch(`/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `Delete failed (${response.status})`);
    }
    setStatus(`Removed ${filename}.`, 'ok');
    refreshDocs();
  } catch (error) {
    button.disabled = false;
    setStatus(error.message, 'error');
  }
}

async function clearAllDocs() {
  if (!confirm('Remove every indexed document? This cannot be undone.')) return;
  try {
    const response = await fetch('/documents', { method: 'DELETE' });
    if (!response.ok) throw new Error(`Clear failed (${response.status})`);
    setStatus('All documents removed.', 'ok');
    refreshDocs();
  } catch (error) {
    setStatus(error.message, 'error');
  }
}

/* --- Conversations -------------------------------------------------------- */

/* SQLite hands back "YYYY-MM-DD HH:MM:SS" in UTC with no zone marker, which JS
 * would otherwise parse as local time and show hours out. */
function parseUtc(stamp) {
  return new Date(String(stamp || '').replace(' ', 'T') + 'Z');
}

function relativeTime(stamp) {
  const then = parseUtc(stamp);
  if (Number.isNaN(then.getTime())) return '';
  const seconds = Math.max(0, (Date.now() - then.getTime()) / 1000);
  if (seconds < 60) return 'now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d`;
  return then.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

async function refreshConversations() {
  try {
    const response = await fetch(`/chat/conversations?user_id=${encodeURIComponent(USER_ID)}`);
    const data = await response.json();
    renderConvList(data.conversations || []);
  } catch {
    /* The conversation panel is navigational; a failure must not break the chat. */
  }
}

function renderConvList(items) {
  els.convList.innerHTML = '';
  els.convCount.textContent = String(items.length);
  els.convEmpty.hidden = items.length > 0;

  for (const item of items) {
    const li = document.createElement('li');
    li.className = 'conv-row' + (item.conversation_id === conversationId ? ' active' : '');
    li.tabIndex = 0;
    li.setAttribute('role', 'button');

    const title = document.createElement('span');
    title.className = 'conv-title';
    title.textContent = item.title;
    title.title = `${item.title} · ${item.message_count} messages`;

    const time = document.createElement('span');
    time.className = 'conv-time';
    time.textContent = relativeTime(item.updated_at);

    const del = document.createElement('button');
    del.className = 'conv-del';
    del.type = 'button';
    del.textContent = '✕';
    del.title = `Delete "${item.title}"`;
    del.setAttribute('aria-label', del.title);
    del.addEventListener('click', (event) => {
      // Without this the row's own click handler would also load the chat.
      event.stopPropagation();
      deleteConversation(item.conversation_id, item.title, del);
    });

    const open = () => loadConversation(item.conversation_id);
    li.addEventListener('click', open);
    li.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        open();
      }
    });

    li.append(title, time, del);
    els.convList.appendChild(li);
  }
}

/* Rebuilds the transcript of a saved conversation. Replayed assistant turns get
 * the same per-message play button as live ones; the conversation's own language
 * mode is passed through so an old Telugu chat still reads in a Telugu voice. */
function renderLoadedMessages(messages, override) {
  els.messages.innerHTML = '';
  const detection = override && LANGUAGE_CODES[override]
    ? { languages: LANGUAGE_CODES[override] }
    : null;

  for (const message of messages) {
    const role = message.role === 'user' ? 'user' : 'assistant';
    const { wrap } = addMessage(role, message.content);
    if (role === 'assistant') addMeta(wrap, [], [], detection);
  }
  if (!messages.length) showWelcome('Conversation cleared.',
    'Your uploaded documents and remembered language preference are still active.');
}

async function loadConversation(id) {
  if (busy) {
    setStatus('Wait for the current reply to finish before switching chats.', 'error');
    return;
  }
  try {
    const response = await fetch(
      `/chat/conversations/${encodeURIComponent(id)}?user_id=${encodeURIComponent(USER_ID)}`);
    if (!response.ok) throw new Error(`Could not open that conversation (${response.status})`);
    const data = await response.json();

    stopSpeaking();
    conversationId = data.conversation_id;
    conversationSaved = true;
    // Restore the language mode this conversation was using, not the one the
    // session happens to be on.
    applyLanguageOverride(data.language_override || null, { persist: false });

    history = data.messages.map((m) => ({ role: m.role, content: m.content }));
    renderLoadedMessages(data.messages, data.language_override || null);

    setStatus('');
    refreshConversations();
    closeDrawer();
  } catch (error) {
    setStatus(error.message, 'error');
  }
}

function newChat() {
  stopSpeaking();
  conversationId = newConversationId();
  conversationSaved = false;
  history = [];
  // A new chat starts on the default mode rather than inheriting the last one.
  applyLanguageOverride(null, { persist: false });
  showWelcome('New chat.', 'Ask anything, in whichever language you think in. Your other conversations are saved in the sidebar.');
  setStatus('');
  refreshConversations();
  closeDrawer();
  els.input.focus();
}

async function deleteConversation(id, title, button) {
  if (!confirm(`Delete "${title}"? This cannot be undone.`)) return;
  button.disabled = true;
  try {
    const response = await fetch(
      `/chat/conversations/${encodeURIComponent(id)}?user_id=${encodeURIComponent(USER_ID)}`,
      { method: 'DELETE' });
    if (!response.ok) throw new Error(`Delete failed (${response.status})`);
    setStatus(`Deleted "${title}".`, 'ok');
    // Deleting the chat you are reading leaves you on a fresh one.
    if (id === conversationId) newChat();
    else refreshConversations();
  } catch (error) {
    button.disabled = false;
    setStatus(error.message, 'error');
  }
}

async function refreshCacheStat() {
  try {
    const response = await fetch('/chat/cache/stats');
    const data = await response.json();
    els.cacheStat.textContent = `${data.total} entr${data.total === 1 ? 'y' : 'ies'}`;
  } catch {
    /* Cosmetic. */
  }
}

/* --- Chat streaming ------------------------------------------------------- */

async function sendMessage(text) {
  if (!text.trim() || busy) return;
  setBusy(true);
  setStatus('');

  addMessage('user', text);
  history.push({ role: 'user', content: text });

  const { wrap, bubble } = addMessage('assistant', '');
  wrap.classList.add('streaming');

  let full = '';
  let metaChips = [];
  let ragChunks = [];

  try {
    const response = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: history.slice(0, -1),
        user_id: USER_ID,
        conversation_id: conversationId,
        language_override: languageOverride,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Server returned ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      /* SSE frames are separated by a blank line. Keep the trailing partial
       * frame in the buffer until the rest of it arrives. */
      const frames = buffer.split('\n\n');
      buffer = frames.pop();

      for (const frame of frames) {
        if (!frame.trim()) continue;

        let eventName = 'message';
        let dataLine = '';
        for (const line of frame.split('\n')) {
          if (line.startsWith('event: ')) eventName = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataLine += line.slice(6);
        }
        if (!dataLine) continue;

        let payload;
        try {
          payload = JSON.parse(dataLine);
        } catch {
          continue;
        }

        if (eventName === 'meta') {
          lastDetection = payload.detection;
          badgeVal(els.langBadge).textContent = payload.detection.label;
          els.langBadge.classList.add('active');
          updateLangPair(`${payload.detection.label} · ${payload.detection.register}`);
          setLangSummary(payload.detection.languages);

          metaChips = [{ text: `${payload.detection.label} · ${payload.detection.register}` }];
          if (payload.detection.code_mixed) metaChips.push({ text: 'code-mixed' });
          if (payload.detection.method === 'llm') metaChips.push({ text: 'LLM-classified' });
          // Makes it visible that this reply bypassed detection entirely.
          if (payload.detection.method === 'manual') metaChips.push({ text: 'manual' });
          if (payload.cache_hit) {
            metaChips.push({ kind: 'cache', text: `cached (${payload.cache_similarity} similar)` });
          }
          ragChunks = payload.rag_chunks || [];
        } else if (eventName === 'token') {
          /* Appended immediately, one frame at a time — this is what makes the
           * reply visibly stream rather than appear all at once. */
          full += payload.t;
          bubble.textContent = full;
          scrollToBottom();
        } else if (eventName === 'error') {
          wrap.classList.add('error');
          bubble.textContent = payload.message;
          setStatus(payload.message, 'error');
        }
      }
    }
  } catch (error) {
    wrap.classList.add('error');
    bubble.textContent = `Could not reach the assistant: ${error.message}`;
  } finally {
    wrap.classList.remove('streaming');
    setBusy(false);
  }

  if (full) {
    history.push({ role: 'assistant', content: full });
    const playButton = addMeta(wrap, metaChips, ragChunks, lastDetection);
    // Auto-read routes through the message's own button so its playing state and
    // stop control work exactly as if it had been clicked.
    if (speakEnabled) speak(full, { languages: lastDetection?.languages, button: playButton });
    // A cache write may have happened server-side; keep the counter fresh.
    refreshCacheStat();
    // The server persists the turn before sending `done`, so by now the row
    // exists — which is what makes a later language change PATCH-able.
    conversationSaved = true;
    refreshConversations();
  }
}

/* --- Document upload ------------------------------------------------------ */

async function uploadFile(file) {
  // Instant client-side validation so obvious rejects don't cost a round-trip.
  const dot = file.name.lastIndexOf('.');
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    setStatus(`Unsupported file type ${ext || '(none)'}. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`, 'error');
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    setStatus(`File is ${Math.round(file.size / 1024)} KB; the limit is ${MAX_UPLOAD_BYTES / 1024} KB.`, 'error');
    return;
  }

  setStatus(`Uploading and indexing ${file.name}…`);
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/documents/upload', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Upload failed (${response.status})`);
    setStatus(`Indexed ${data.filename}: ${data.chunks} chunks. Ask about it in any language.`, 'ok');
    refreshDocs();
  } catch (error) {
    setStatus(error.message, 'error');
  }
}

/* --- Voice input (speech-to-text) ----------------------------------------- */

/* Chrome/Edge/Safari only, and always webkit-prefixed in practice. */
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
  els.micBtn.hidden = false;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;

  /* en-IN is the most reliable choice for this audience: it handles Indian-accented
   * English AND the English words in code-mixed speech, whereas te-IN recognition
   * is unavailable on most platforms. If the browser rejects the locale it falls
   * back to the system default rather than failing outright. */
  recognition.lang = 'en-IN';

  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    els.input.value = transcript;
    autoResize();
  };
  recognition.onerror = (event) => {
    setStatus(
      event.error === 'not-allowed'
        ? 'Microphone permission denied.'
        : `Speech recognition error: ${event.error}`,
      'error'
    );
    els.micBtn.classList.remove('recording');
  };
  recognition.onend = () => els.micBtn.classList.remove('recording');

  els.micBtn.addEventListener('click', () => {
    if (els.micBtn.classList.contains('recording')) {
      recognition.stop();
      return;
    }
    try {
      recognition.start();
      els.micBtn.classList.add('recording');
      setStatus('Listening… speak now.');
    } catch {
      /* start() throws if already running; harmless. */
    }
  });
}
/* else: the mic button stays hidden. No dead control is shown. */

/* --- Voice output (text-to-speech) ---------------------------------------- */

const TTS_AVAILABLE = 'speechSynthesis' in window;

if (TTS_AVAILABLE) {
  els.speakBtn.hidden = false;
  els.speakBtn.addEventListener('click', () => {
    speakEnabled = !speakEnabled;
    els.speakBtn.setAttribute('aria-pressed', String(speakEnabled));
    if (!speakEnabled) stopSpeaking();
    setStatus(speakEnabled ? 'Replies will be read aloud.' : '');
  });
}

/* Exactly one utterance may be audible at a time, so playback state is tracked
 * globally rather than per button: starting a new message implicitly stops
 * whatever was already speaking, and the previous button has to be reset. */
let activeUtterance = null;
let activeSpeakButton = null;

function setSpeakingState(button, speaking) {
  if (!button) return;
  button.classList.toggle('speaking', speaking);
  button.setAttribute('aria-pressed', String(speaking));
  button.textContent = speaking ? '⏹' : '🔊';
  button.title = speaking ? 'Stop reading' : 'Read this message aloud';
  button.setAttribute('aria-label', button.title);
}

function stopSpeaking() {
  if (!TTS_AVAILABLE) return;
  /* Cleared BEFORE cancel(): cancel() synchronously fires the pending utterance's
   * onend, and that handler must not mistake a newer utterance for its own. */
  activeUtterance = null;
  setSpeakingState(activeSpeakButton, false);
  activeSpeakButton = null;
  window.speechSynthesis.cancel();
}

/* Builds the play control for one message. Deliberately reads the bubble text at
 * click time rather than capturing it, so a message rendered earlier in the
 * session still speaks its current contents. */
function buildSpeakButton(wrap, detection) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'speak-btn';
  setSpeakingState(button, false);

  /* This message's OWN detected language, not the conversation's latest: replaying
   * an older Telugu reply must still pick a Telugu voice after the chat has moved
   * on to English. */
  const languages = detection?.languages || null;

  button.addEventListener('click', () => {
    if (button === activeSpeakButton) {
      stopSpeaking();
      return;
    }
    const bubble = wrap.querySelector('.bubble');
    speak(bubble ? bubble.textContent : '', { languages, button });
  });

  return button;
}

/* Map detected language codes to BCP-47 tags for voice selection. `codes` lets a
 * specific message override the conversation-wide latest detection. */
function preferredVoiceLang(codes = null) {
  const list = codes || lastDetection?.languages || [];
  if (list.includes('te') || list.includes('te-rom')) return 'te-IN';
  if (list.includes('hi') || list.includes('hi-rom')) return 'hi-IN';
  return 'en-IN';
}

/* `button` is the per-message control to reflect playing state on, if any; the
 * global auto-read toggle passes the same button so both entry points stay in
 * sync. Voice selection below is shared by both. */
function speak(text, { languages = null, button = null } = {}) {
  if (!TTS_AVAILABLE || !text.trim()) return;
  // Replaces whatever is currently speaking, and resets that message's button.
  stopSpeaking();

  const wanted = preferredVoiceLang(languages);
  const voices = window.speechSynthesis.getVoices();

  /* Graceful degradation, in order of preference:
   *   exact locale -> same base language -> any English -> browser default.
   * Telugu (te-IN) voices in particular are absent on most desktop platforms,
   * so this commonly lands on Hindi or English rather than going silent. */
  let voice =
    voices.find((v) => v.lang === wanted) ||
    voices.find((v) => v.lang.startsWith(wanted.split('-')[0])) ||
    voices.find((v) => v.lang.startsWith('en'));

  const utterance = new SpeechSynthesisUtterance(text);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
    if (!voice.lang.startsWith(wanted.split('-')[0])) {
      setStatus(`No ${wanted} voice installed in this browser; reading with ${voice.lang}.`);
    }
  } else {
    utterance.lang = wanted;
  }

  /* Identity guard: cancel() fires onend for the utterance it interrupted, which
   * would otherwise clear the state of the message that just replaced it. */
  const finish = () => {
    if (utterance !== activeUtterance) return;
    activeUtterance = null;
    setSpeakingState(button, false);
    activeSpeakButton = null;
  };
  utterance.onend = finish;
  utterance.onerror = finish;

  activeUtterance = utterance;
  activeSpeakButton = button;
  // Set eagerly rather than in onstart, which can lag noticeably behind the click.
  setSpeakingState(button, true);
  window.speechSynthesis.speak(utterance);
}

/* Voice list loads asynchronously in Chrome; this fires once it is ready. */
if (TTS_AVAILABLE) {
  window.speechSynthesis.onvoiceschanged = () => {};
}

/* --- Language profile display --------------------------------------------- */

function updateLangPair(text) {
  els.langPair.textContent = text;
  els.langProfile.classList.add('active');
}

/* Badge on the collapsed Language header, so the active mode is readable without
 * expanding. Codes rather than full names ("te-rom + en", not "Telugu (romanized)
 * + English") because the row is narrow. */
function setLangSummary(codes) {
  const list = (codes || []).filter(Boolean);
  els.langSummary.textContent = list.length ? list.join(' + ') : 'auto';
  els.langSummary.title = languageOverride
    ? 'Language pinned manually for this conversation'
    : 'Detected automatically';
  els.langSummary.classList.toggle('manual', Boolean(languageOverride));
}

/* --- Manual language override --------------------------------------------- */

/* value -> code array, filled from /chat/languages. Used to give a loaded
 * conversation's TTS the right voice without re-deriving the mapping here. */
const LANGUAGE_CODES = {};

async function loadLanguageOptions() {
  try {
    const response = await fetch('/chat/languages');
    const data = await response.json();
    for (const option of data.options || []) {
      LANGUAGE_CODES[option.value] = option.codes;
      const element = document.createElement('option');
      element.value = option.value;
      element.textContent = option.label;
      els.langSelect.appendChild(element);
    }
  } catch {
    /* Leaves just "Auto-detect", which is exactly today's behaviour. */
  }
}

/* Single place that moves the override between the control, the request payload
 * and the server, so the three cannot disagree. */
function applyLanguageOverride(value, { persist = true } = {}) {
  languageOverride = value || null;
  els.langSelect.value = languageOverride || '';
  els.langField.classList.toggle('manual', Boolean(languageOverride));

  if (languageOverride) {
    const label = els.langSelect.selectedOptions[0]?.textContent || languageOverride;
    updateLangPair(`${label} · manual`);
    els.langNote.textContent =
      'Detection is off for this chat — replies use the language you picked. The setting is saved with this conversation.';
    setLangSummary(LANGUAGE_CODES[languageOverride] || [languageOverride]);
  } else {
    els.langPair.textContent = '—';
    els.langProfile.classList.remove('active');
    els.langNote.textContent =
      'Code-mixed input is detected automatically; your usual pair is remembered across sessions.';
    // Back to auto: the badge resets until the next reply reports a detection.
    setLangSummary(null);
  }

  // A chat with no messages has no row yet, so there is nothing to PATCH; the
  // value rides along with its first message instead.
  if (persist && conversationSaved) {
    fetch(`/chat/conversations/${encodeURIComponent(conversationId)}?user_id=${encodeURIComponent(USER_ID)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language_override: languageOverride }),
    }).catch(() => {});
  }
}

els.langSelect.addEventListener('change', () => applyLanguageOverride(els.langSelect.value));

/* --- Collapsible sidebar sections ------------------------------------------
 * <details> handles opening and closing itself; this only remembers which were
 * open. Persisted rather than session-scoped because it costs the same and
 * surviving a reload is the friendlier default. Absent state = collapsed, which
 * keeps the sidebar to three compact rows on first load. */

const PANEL_STATE_KEY = 'maa_panels_open';

function initCollapsibles() {
  let open = [];
  try {
    open = JSON.parse(localStorage.getItem(PANEL_STATE_KEY) || '[]');
  } catch {
    open = [];
  }

  for (const panel of document.querySelectorAll('.side-collapse')) {
    panel.open = open.includes(panel.id);
    panel.addEventListener('toggle', () => {
      const ids = [...document.querySelectorAll('.side-collapse')]
        .filter((element) => element.open)
        .map((element) => element.id);
      try {
        localStorage.setItem(PANEL_STATE_KEY, JSON.stringify(ids));
      } catch {
        /* Private-mode storage failures must not break the panel. */
      }
    });
  }
}

initCollapsibles();

/* --- Sidebar drawer (mobile) ---------------------------------------------- */

function openDrawer() {
  els.app.classList.add('drawer-open');
  els.scrim.hidden = false;
}
function closeDrawer() {
  els.app.classList.remove('drawer-open');
  els.scrim.hidden = true;
}

els.sidebarToggle.addEventListener('click', openDrawer);
els.sidebarClose.addEventListener('click', closeDrawer);
els.scrim.addEventListener('click', closeDrawer);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeDrawer();
});

/* --- Composer wiring ------------------------------------------------------ */

function autoResize() {
  els.input.style.height = 'auto';
  els.input.style.height = Math.min(els.input.scrollHeight, 168) + 'px';
}

els.input.addEventListener('input', autoResize);

els.input.addEventListener('keydown', (event) => {
  /* Enter sends; Shift+Enter makes a newline. */
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

els.form.addEventListener('submit', (event) => {
  event.preventDefault();
  const text = els.input.value;
  els.input.value = '';
  autoResize();
  sendMessage(text);
});

/* --- Upload wiring: file input + drag-and-drop ---------------------------- */

els.fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) uploadFile(file);
  event.target.value = '';
});

['dragenter', 'dragover'].forEach((type) =>
  els.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    els.dropzone.classList.add('dragover');
  })
);
['dragleave', 'dragend', 'drop'].forEach((type) =>
  els.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove('dragover');
  })
);
els.dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) uploadFile(file);
});
/* Keyboard access: the dropzone is a <label> for the hidden file input, so
 * Enter/Space should open the picker. */
els.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    els.fileInput.click();
  }
});

/* --- Session controls ----------------------------------------------------- */

els.clearDocsBtn.addEventListener('click', clearAllDocs);

els.newChatBtn.addEventListener('click', newChat);

/* Empties the conversation you have open. Deliberately NOT a delete: the chat
 * stays in the sidebar and keeps its language mode. Removing one entirely is the
 * ✕ on its row. */
els.clearChatBtn.addEventListener('click', async () => {
  if (!confirm('Clear the messages in this chat? Your other conversations are not affected.')) return;

  // Otherwise playback continues against buttons that are about to be detached.
  stopSpeaking();
  history = [];
  showWelcome('Chat cleared.',
    'This conversation is empty but still saved. Your documents, language setting and other conversations are untouched.');
  setStatus('');
  closeDrawer();

  if (conversationSaved) {
    try {
      const response = await fetch(
        `/chat/conversations/${encodeURIComponent(conversationId)}/messages?user_id=${encodeURIComponent(USER_ID)}`,
        { method: 'DELETE' });
      if (!response.ok && response.status !== 404) throw new Error(`Clear failed (${response.status})`);
      refreshConversations();
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }
});

document.addEventListener('click', (event) => {
  if (event.target.classList.contains('example-btn')) {
    els.input.value = event.target.textContent.trim();
    autoResize();
    els.input.focus();
  }
});

/* --- Startup -------------------------------------------------------------- */

/* The UI is only ever meant to be reached through the FastAPI process that also
 * serves the API — every call below is a root-absolute path to that same origin.
 * Opened straight off disk (file://) those all resolve to the filesystem root and
 * fail silently, which looks like "the app is broken" rather than "you opened the
 * wrong URL". Say so explicitly instead of booting into a dead UI. */
if (location.protocol === 'file:') {
  setStatus('Opened as a local file — start the server and use http://localhost:8000 instead.', 'error');
} else {
  refreshDocs();
  refreshCacheStat();
  loadLanguageOptions();
  refreshConversations();

  fetch(`/chat/profile?user_id=${encodeURIComponent(USER_ID)}`)
    .then((r) => r.json())
    .then((profile) => {
      if (profile.returning && profile.preferred_languages.length) {
        const pair = profile.preferred_languages.join(' + ');
        badgeVal(els.langBadge).textContent = pair;
        els.langBadge.classList.add('active');
        updateLangPair(pair);
      }
    })
    .catch(() => {});
}
