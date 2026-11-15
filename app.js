/* Semantic Sonar — daily word-guessing via MiniLM word-embedding similarity.
 *
 * Similarity engine:
 *  - Instant path: bundled lexicon of ~540 MiniLM-L6-v2 vectors (embeddings.json).
 *  - Extended path: the same model (int8 ONNX) runs fully in-browser via the
 *    vendored @huggingface/transformers runtime, so out-of-lexicon words get
 *    real embeddings too. Both paths share one vector space.
 */
import { TARGETS } from './scripts/words.mjs';

const params = new URLSearchParams(location.search);
const FORCED_TARGET = params.get('target')?.toLowerCase() || null;

const $ = (id) => document.getElementById(id);
const els = {
  status: $('model-status'), latest: $('latest'), latestWord: $('latest-word'),
  latestScore: $('latest-score'), latestBar: $('latest-bar'), latestHint: $('latest-hint'),
  history: $('history'), empty: $('empty'), form: $('guess-form'),
  input: $('guess-input'), btn: $('guess-btn'), toast: $('toast'),
  modal: $('win-modal'), winWord: $('win-word'), winCount: $('win-count'),
  winPlural: $('win-plural'), winGrid: $('win-grid'), shareBtn: $('share-btn'),
};

/* ---------- daily target ---------- */
function hashSeed(str) {
  let h = 2166136261;
  for (const ch of str) { h ^= ch.codePointAt(0); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
const today = new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD, local
const seed = params.get('seed') || today;
const TARGET = FORCED_TARGET || TARGETS[hashSeed(seed) % TARGETS.length];
const storeKey = `semantic-sonar:v1:${seed}:${TARGET}`;

/* ---------- warmth math ---------- */
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// MiniLM cosine sims: unrelated ≈ 0.1–0.3, related ≈ 0.4–0.65, near-synonym ≈ 0.85+
function warmthFromSim(sim) {
  if (sim >= 0.9999) return 100;
  const t = Math.min(1, Math.max(0, (sim - 0.15) / 0.85));
  return Math.round(100 * Math.pow(t, 1.2));
}

const emojiFor = (w) => w >= 100 ? '🟩' : w >= 85 ? '🟥' : w >= 55 ? '🟧' : w >= 25 ? '🟨' : '🟦';
const hintFor = (w) =>
  w >= 100 ? 'Direct hit!' : w >= 85 ? 'Scorching — right on top of it!' :
  w >= 70 ? 'Hot — very close conceptually.' : w >= 55 ? 'Warm — circling the idea.' :
  w >= 40 ? 'Lukewarm — some relation.' : w >= 25 ? 'Cool — faint echo.' : 'Ice cold — different universe.';

// Background temperature keyframes: [warmth, hue, sat%, light%]
const BG_STOPS = [
  [0, 232, 52, 10], [25, 252, 52, 13], [50, 285, 55, 17],
  [70, 318, 62, 19], [85, 348, 68, 22], [100, 26, 82, 25],
];
function setBackgroundTemp(warmth) {
  let a = BG_STOPS[0], b = BG_STOPS[BG_STOPS.length - 1];
  for (let i = 0; i < BG_STOPS.length - 1; i++) {
    if (warmth >= BG_STOPS[i][0] && warmth <= BG_STOPS[i + 1][0]) { a = BG_STOPS[i]; b = BG_STOPS[i + 1]; break; }
  }
  const t = a === b ? 0 : (warmth - a[0]) / (b[0] - a[0]);
  const root = document.documentElement.style;
  root.setProperty('--bg-h', (a[1] + (b[1] - a[1]) * t).toFixed(1));
  root.setProperty('--bg-s', (a[2] + (b[2] - a[2]) * t).toFixed(1));
  root.setProperty('--bg-l', (a[3] + (b[3] - a[3]) * t).toFixed(1));
}

/* ---------- haptics ---------- */
function pulse(warmth) {
  if (!('vibrate' in navigator)) return;
  if (warmth >= 100) navigator.vibrate([70, 50, 70, 50, 260]);       // celebration
  else if (warmth >= 85) navigator.vibrate([55, 35, 55, 35, 140]);   // intense bursts
  else if (warmth >= 70) navigator.vibrate([45, 45, 90]);
  else if (warmth >= 55) navigator.vibrate([35, 50, 60]);
  else if (warmth >= 40) navigator.vibrate(55);
  else if (warmth >= 25) navigator.vibrate(30);
  else navigator.vibrate(15);                                        // faint tap
}

/* ---------- embedding engine ---------- */
let lexicon = null;       // { word: number[] } from embeddings.json
let targetVec = null;
let liveEmbedder = null;  // lazily loaded in-browser model
let liveState = 'idle';   // idle | loading | ready | failed

async function loadLexicon() {
  const res = await fetch('embeddings.json');
  if (!res.ok) throw new Error('lexicon fetch failed');
  lexicon = (await res.json()).words;
}

function lookupVec(word) {
  if (lexicon[word]) return { vec: lexicon[word], oov: false };
  // light stemming so plurals/inflections reuse the lexicon
  for (const stem of [word.replace(/ies$/, 'y'), word.replace(/(es|s)$/, ''),
                      word.replace(/(ing|ed)$/, '')]) {
    if (stem.length > 2 && lexicon[stem]) return { vec: lexicon[stem], oov: false };
  }
  return null;
}

async function loadLiveModel() {
  if (liveState !== 'idle') return;
  liveState = 'loading';
  try {
    const { pipeline, env } = await import('./vendor/transformers.web.js');
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = new URL('models/', location.href).pathname;
    env.backends.onnx.wasm.wasmPaths = {
      mjs: './vendor/ort-wasm-simd-threaded.jsep.mjs',
      wasm: './vendor/ort/ort-wasm-simd-threaded.jsep.wasm',
    };
    liveEmbedder = await pipeline('feature-extraction', 'all-MiniLM-L6-v2', { dtype: 'q8' });
    liveState = 'ready';
    setStatus('ready', 'Full AI lexicon online — any English word works');
  } catch (err) {
    console.warn('live model unavailable:', err);
    liveState = 'failed';
    setStatus('warn', `Offline lexicon (${Object.keys(lexicon).length} words)`);
  }
}

async function embedWord(word) {
  const hit = lookupVec(word);
  if (hit) return hit;
  if (liveState === 'idle') loadLiveModel();
  if (liveState === 'ready') {
    const out = await liveEmbedder(word, { pooling: 'mean', normalize: true });
    return { vec: Array.from(out.data), oov: true };
  }
  return null;
}

/* ---------- state ---------- */
let guesses = []; // [{ w, s, oov }] in submission order
let won = false;

function saveState() {
  try { localStorage.setItem(storeKey, JSON.stringify({ guesses, won })); } catch {}
}
function loadState() {
  if (params.has('reset')) return;
  try {
    const s = JSON.parse(localStorage.getItem(storeKey));
    if (s && Array.isArray(s.guesses)) { guesses = s.guesses; won = !!s.won; }
  } catch {}
}

/* ---------- UI ---------- */
let toastTimer;
function toast(msg, ms = 2400) {
  els.toast.textContent = msg;
  els.toast.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.add('hidden'), ms);
}

function setStatus(kind, msg) {
  els.status.textContent = msg;
  els.status.className = `status status-${kind}`;
}

function renderLatest(g) {
  els.latest.classList.remove('hidden');
  els.latestWord.textContent = g.w;
  els.latestScore.textContent = `${g.s}%`;
  els.latestHint.textContent = `${emojiFor(g.s)} ${hintFor(g.s)}`;
  els.latestBar.style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    els.latestBar.style.width = `${g.s}%`;
  }));
}

function renderHistory(freshWord) {
  els.empty.classList.toggle('hidden', guesses.length > 0);
  const sorted = [...guesses].sort((a, b) => b.s - a.s || guesses.indexOf(a) - guesses.indexOf(b));
  els.history.replaceChildren(...sorted.map((g, i) => {
    const li = document.createElement('li');
    if (g.w === freshWord) li.classList.add('fresh');
    if (g.s === 100) li.classList.add('winner');
    li.innerHTML = `<span class="rank">${i + 1}</span>
      <span class="guess-word"></span>
      <span class="bar"><span class="bar-fill" style="width:${g.s}%"></span></span>
      <span class="pct">${g.s}%</span>`;
    li.querySelector('.guess-word').textContent = g.w;
    if (g.oov) {
      const tag = document.createElement('span');
      tag.className = 'oov';
      tag.textContent = 'AI';
      tag.title = 'Embedded live by the on-device model';
      li.querySelector('.guess-word').appendChild(tag);
    }
    return li;
  }));
}

function renderWin() {
  els.winWord.textContent = TARGET;
  els.winCount.textContent = guesses.length;
  els.winPlural.textContent = guesses.length === 1 ? '' : 'es';
  const emojis = guesses.map((g) => emojiFor(g.s));
  const rows = [];
  for (let i = 0; i < emojis.length; i += 8) rows.push(emojis.slice(i, i + 8).join(''));
  els.winGrid.textContent = rows.join('\n');
  els.modal.classList.remove('hidden');
  els.input.disabled = true;
  els.btn.disabled = true;
}

function shareText() {
  const emojis = guesses.map((g) => emojiFor(g.s));
  const rows = [];
  for (let i = 0; i < emojis.length; i += 8) rows.push(emojis.slice(i, i + 8).join(''));
  return `Semantic Sonar 📡 ${seed}\nSolved in ${guesses.length} guess${guesses.length === 1 ? '' : 'es'}\n${rows.join('\n')}`;
}

els.shareBtn.addEventListener('click', async () => {
  const text = shareText();
  try {
    await navigator.clipboard.writeText(text);
    els.shareBtn.textContent = '✅ Copied!';
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    els.shareBtn.textContent = ok ? '✅ Copied!' : '⬇️ Select & copy manually';
    if (!ok) { els.winGrid.textContent += `\n\n${text}`; }
  }
  setTimeout(() => { els.shareBtn.textContent = '📋 Share result'; }, 2200);
});

/* ---------- guessing ---------- */
let busy = false;

els.form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (busy || won) return;
  const word = els.input.value.trim().toLowerCase();
  if (!/^[a-z][a-z'-]{1,29}$/.test(word)) {
    toast('Enter a single English word (letters only)');
    return;
  }
  if (guesses.some((g) => g.w === word)) {
    toast(`You already pinged “${word}”`);
    els.input.select();
    return;
  }

  busy = true;
  els.btn.disabled = true;
  els.btn.textContent = '…';
  try {
    const hit = await embedWord(word);
    if (!hit) {
      toast(liveState === 'loading'
        ? 'AI warming up — try a common word for now'
        : `“${word}” isn’t in the offline lexicon`);
      return;
    }
    const warmth = word === TARGET ? 100 : warmthFromSim(dot(hit.vec, targetVec));
    const g = { w: word, s: warmth, oov: hit.oov };
    guesses.push(g);
    saveState();
    renderLatest(g);
    renderHistory(word);
    setBackgroundTemp(warmth);
    pulse(warmth);
    els.input.value = '';
    if (warmth === 100) { won = true; saveState(); setTimeout(renderWin, 650); }
  } finally {
    busy = false;
    els.btn.textContent = 'Ping';
    els.btn.disabled = won;
    if (!won) els.input.focus();
  }
});

// Pre-warm the on-device model shortly after the first interaction,
// so out-of-lexicon words are ready by the time the player needs them.
let prewarm = () => {
  prewarm = () => {};
  setTimeout(loadLiveModel, 1200);
};
els.input.addEventListener('input', prewarm, { once: true });

/* ---------- boot ---------- */
(async function init() {
  try {
    await loadLexicon();
  } catch {
    setStatus('warn', 'Could not load lexicon — check your connection');
    return;
  }
  if (!lexicon[TARGET]) {
    setStatus('warn', 'Daily word missing from lexicon');
    return;
  }
  targetVec = lexicon[TARGET];
  loadState();

  setStatus('ready', `Lexicon ready — ${Object.keys(lexicon).length} words loaded`);
  els.input.disabled = false;
  els.btn.disabled = false;

  if (guesses.length) {
    const sorted = [...guesses].sort((a, b) => b.s - a.s);
    renderLatest(sorted[0]);
    renderHistory(null);
    setBackgroundTemp(sorted[0].s);
    if (won) renderWin();
  } else {
    setBackgroundTemp(0);
  }
})();
