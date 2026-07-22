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
  cacheStat: document.getElementById('cacheStat'),

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
      body: JSON.stringify({ message: text, history: history.slice(0, -1), user_id: USER_ID }),
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

          metaChips = [{ text: `${payload.detection.label} · ${payload.detection.register}` }];
          if (payload.detection.code_mixed) metaChips.push({ text: 'code-mixed' });
          if (payload.detection.method === 'llm') metaChips.push({ text: 'LLM-classified' });
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

els.clearChatBtn.addEventListener('click', () => {
  // Otherwise playback continues against buttons that are about to be detached.
  stopSpeaking();
  history = [];
  els.messages.innerHTML =
    '<div class="welcome"><div class="welcome-glyph" aria-hidden="true">◈</div>' +
    '<h2>Conversation cleared.</h2>' +
    '<p>Your uploaded documents and remembered language preference are still active.</p></div>';
  setStatus('');
  closeDrawer();
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
