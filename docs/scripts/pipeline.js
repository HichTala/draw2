// Config
// Hosted ONNX artifacts. cardnames_onnx.json is the ViT-index-keyed variant
// built by export_models.py; the card_id-keyed one lives on the model repo.
const BUCKET = "https://huggingface.co/HichTala/draw2/resolve/main/onnx";
const NAMES_URL           = `${BUCKET}/cardnames_onnx.json?download=true`;
const YOLO_URL            = `${BUCKET}/ygo_yolo.onnx?download=true`;
const VIT_FP32_URL        = `${BUCKET}/vit_fp32.onnx?download=true`;
const VIT_FP16_URL        = `${BUCKET}/vit_fp16.onnx?download=true`;
const VIT_YUGISCAN_URL    = `${BUCKET}/vit_yugiscan_int8.onnx?download=true`;
const YUGISCAN_LABELS_URL = `${BUCKET}/card_labels_yugiscan.json?download=true`;
const CROP_SIZE  = 224;
const CONF_THRESH = 0.20;
const VIT_TOPK   = 3;

// ViT normalization
const MEAN = [0.5, 0.5, 0.5];
const STD  = [0.5, 0.5, 0.5];

// State
let yoloSession = null;
let vitSession  = null;
let cardnames   = {};
let cardId2Names = {}; // card_id -> {EN,FR,JA,...}, for Small's plain-string labels
let modelsReady = false;
let currentStream = null;
let cancelRequested = false;
let currentPrecision = null; // "fp32", "fp16", or "yugiscan"

// Debug Console
let dbgErrorCount = 0;

function dbgAppend(level, ...args) {
    const logEl = document.getElementById("dbg-log");
    if (!logEl) return;
    const text = args.map(a =>
        typeof a === "object" ? JSON.stringify(a, null, 2) : String(a)
    ).join(" ");
    const ts = new Date().toLocaleTimeString("fr-FR", { hour12: false });
    const el = document.createElement("div");
    el.className = `dbg-entry ${level}`;
    const tsSpan = document.createElement("span");
    tsSpan.className = "dbg-ts";
    tsSpan.textContent = ts;
    const msgNode = document.createTextNode(" " + text);
    el.appendChild(tsSpan);
    el.appendChild(msgNode);
    logEl.appendChild(el);
    logEl.scrollTop = logEl.scrollHeight;
    if (level === "err") {
        dbgErrorCount++;
        const toggle = document.getElementById("dbg-toggle");
        const label  = document.getElementById("dbg-label");
        const body   = document.getElementById("dbg-body");
        if (toggle) toggle.classList.add("has-error");
        if (label)  label.textContent = `Debug log (${dbgErrorCount} error${dbgErrorCount > 1 ? "s" : ""})`;
        if (body)   body.hidden = false; // auto-open on first error
    }
}

function dbgProgress(id, label, current, total, width = 20) {
    const logEl = document.getElementById("dbg-log");
    if (!logEl) return;
    const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
    const filled = Math.round((pct / 100) * width);
    const bar = "#".repeat(filled) + "-".repeat(width - filled);
    const text = `${label} [${bar}] ${pct}% (${current}/${total})`;
    const elId = `dbg-progress-${id}`;
    let el = document.getElementById(elId);
    if (!el) {
        el = document.createElement("div");
        el.id = elId;
        el.className = "dbg-entry log";
        const tsSpan = document.createElement("span");
        tsSpan.className = "dbg-ts";
        tsSpan.textContent = new Date().toLocaleTimeString("fr-FR", { hour12: false });
        el.appendChild(tsSpan);
        el.appendChild(document.createTextNode(""));
        logEl.appendChild(el);
    }
    el.lastChild.textContent = " " + text;
    logEl.scrollTop = logEl.scrollHeight;
}

function dbgProgressDone(id) {
    const el = document.getElementById(`dbg-progress-${id}`);
    if (el) el.removeAttribute("id");
}

// Proxy console
const _log  = console.log.bind(console);
const _warn = console.warn.bind(console);
const _err  = console.error.bind(console);
console.log   = (...a) => { _log(...a);  dbgAppend("log",  ...a); };
console.warn  = (...a) => { 
    _warn(...a); 
    const str = a.map(String).join(" ").toLowerCase();
    if (str.includes("onnx") || str.includes("ort") || str.includes("webgpu") || str.includes("wasm")) return;
    dbgAppend("warn", ...a); 
};
console.error = (...a) => { _err(...a);  dbgAppend("err",  ...a); };

window.addEventListener("unhandledrejection", e => {
    console.error("Unhandled:", String(e.reason));
});

// Verbosity (low = milestones only, high = per-step timings)
let verbosity = localStorage.getItem("draw2_verbosity") || "low";
function isVerbose() { return verbosity === "high"; }
const dbg = (...a) => { if (isVerbose()) console.log(...a); };
const status = (...a) => console.log(...a);

function injectVerbosityToggle() {
    const logEl = $("dbg-log");
    const header = logEl && logEl.previousElementSibling;
    if (!header || $("verbosity-toggle")) return;

    const wrap = document.createElement("div");
    wrap.id = "verbosity-toggle";
    wrap.style.cssText = "display:flex;align-items:center;gap:6px;font-family:monospace;font-size:10px;margin-left:12px;";
    wrap.innerHTML = `
        <span style="opacity:.5;">verbosity</span>
        <button id="verbosity-low"  type="button" style="padding:2px 8px;border-radius:999px;border:1px solid rgba(160,160,160,.35);cursor:pointer;background:transparent;color:inherit;">low</button>
        <button id="verbosity-high" type="button" style="padding:2px 8px;border-radius:999px;border:1px solid rgba(160,160,160,.35);cursor:pointer;background:transparent;color:inherit;">high</button>
    `;
    header.appendChild(wrap);

    const lowBtn = $("verbosity-low"), highBtn = $("verbosity-high");
    const refresh = () => {
        if (lowBtn)  { lowBtn.style.background  = verbosity === "low"  ? "#10b981" : "transparent"; lowBtn.style.color  = verbosity === "low"  ? "#000" : "inherit"; }
        if (highBtn) { highBtn.style.background = verbosity === "high" ? "#10b981" : "transparent"; highBtn.style.color = verbosity === "high" ? "#000" : "inherit"; }
    };
    lowBtn?.addEventListener("click", () => { verbosity = "low"; localStorage.setItem("draw2_verbosity", "low"); refresh(); });
    highBtn?.addEventListener("click", () => { verbosity = "high"; localStorage.setItem("draw2_verbosity", "high"); refresh(); });
    refresh();
}

// Load Panel
const $ = id => document.getElementById(id);

const T = (key, vars) => (window.t ? window.t(key, vars) : key);

// Small's labels are plain strings; cross-reference cardId2Names for those.
function localizedName(entry) {
    const lang = (window.getLang?.() || "en").toUpperCase();
    return entry[lang] || entry.EN;
}
function cardNameFor(entry, index) {
    if (typeof entry === "string") {
        const cardId = entry.match(/-(\d+)$/)?.[1];
        const localized = cardId && cardId2Names[cardId];
        if (localized) return localizedName(localized);
        return entry.replace(/-\d+$/, "").replace(/-/g, " ");
    }
    if (!entry) return String(index);
    return localizedName(entry) || String(index);
}

// Hotlinking ygoprodeck.com is prohibited and gets the site IP-blacklisted.
const ART_BUCKET = "https://huggingface.co/buckets/HichTala/ygoprodeck-images/resolve/ygoprodeck";

function cardArtUrl(index) {
    const entry = cardnames[String(index)];
    const label = typeof entry === "string" ? entry : entry?.label;
    const id = label?.match(/-(\d+)$/)?.[1];
    if (!label || !id) return null;
    return `${ART_BUCKET}/${encodeURIComponent(label)}/${id}.jpg`;
}

// The bucket is slow and sends no Cache-Control, so each artwork is fetched
// once into a blob and served from memory afterwards.
const artCache = new Map();

// Neutral card silhouette for the handful of ids the bucket is missing.
const ART_PLACEHOLDER = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 421 614">
        <rect x="16" y="16" width="389" height="582" rx="20" fill="rgba(255,255,255,.85)"/>
        <g fill="none" stroke="rgba(113,113,122,.38)" stroke-width="5">
            <rect x="16" y="16" width="389" height="582" rx="20"/>
            <rect x="52" y="120" width="317" height="317" rx="6"/>
        </g>
        <g fill="rgba(113,113,122,.22)">
            <rect x="52" y="62" width="242" height="34" rx="6"/>
            <rect x="52" y="470" width="317" height="13" rx="6"/>
            <rect x="52" y="500" width="268" height="13" rx="6"/>
            <rect x="52" y="530" width="196" height="13" rx="6"/>
        </g>
        <text x="210" y="280" text-anchor="middle" dominant-baseline="central"
              font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="168"
              font-weight="700" fill="rgba(113,113,122,.42)">?</text>
    </svg>`);

function artSrcFor(index) {
    if (artCache.has(index)) return artCache.get(index);
    const url = cardArtUrl(index);
    if (!url) return null;
    const pending = fetch(url, { referrerPolicy: "no-referrer" })
        .then(r => r.ok ? r.blob() : Promise.reject(r.status))
        .then(b => URL.createObjectURL(b))
        .catch(() => null);
    artCache.set(index, pending);
    return pending;
}

function prefetchArtwork(predictions) {
    hideHoverCard();
    for (const p of artCache.values()) {
        Promise.resolve(p).then(src => { if (src) URL.revokeObjectURL(src); });
    }
    artCache.clear();
    for (const preds of predictions || []) {
        const i = preds?.[0]?.i;
        if (i != null) artSrcFor(i);
    }
}

function setLoadStatus(text, pct, hint = "") {
    const bar = $("lp-bar");
    if (bar) bar.style.width = pct + "%";

    const label = $("btn-load-label");
    if (label) label.textContent = hint || `${text} - ${Math.round(pct)}%`;

    const s = $("lp-status"), p = $("lp-pct"), h = $("lp-hint");
    if (s) s.textContent = text;
    if (p) p.textContent = Math.round(pct) + "%";
    if (h) h.textContent = hint;
}

// YOLO, then the ViT, then the card DB
const DL_STEPS = 3;

const MODEL_CACHE = "draw2-models-v1";
// Max is deliberately absent: 386 MB is too much to park on someone's disk
// without asking. Each entry is keyed by URL, so switching models keeps both.
const CACHEABLE = new Set([YOLO_URL, VIT_YUGISCAN_URL, VIT_FP16_URL, NAMES_URL, YUGISCAN_LABELS_URL]);
const cacheAvailable = () => typeof caches !== "undefined";


async function cachedResponse(url, signal) {
    if (!CACHEABLE.has(url) || !cacheAvailable()) return null;
    try {
        const cache = await caches.open(MODEL_CACHE);
        const hit = await cache.match(url);
        if (!hit) return null;
        const known = hit.headers.get("x-draw2-etag");
        if (known) {
            // The URLs point at a branch, so the bytes behind them can change.
            // HuggingFace exposes ETag through CORS, so one HEAD settles it.
            const head = await fetch(url, { method: "HEAD", signal }).catch(() => null);
            const current = head?.headers.get("ETag");
            if (current && current !== known) { await cache.delete(url); return null; }
        }
        // Never hand an empty buffer to ORT: it fails deep inside the runtime
        // with "No graph was found in the protobuf", which says nothing about
        // the cache being at fault. Drop the entry and refetch instead.
        const expected = Number(hit.headers.get("x-draw2-size") || 0);
        const buf = await hit.arrayBuffer();
        if (!buf.byteLength || (expected && buf.byteLength !== expected)) {
            dbg(`  cached ${url.split("/").pop().split("?")[0]} is truncated, refetching`);
            await cache.delete(url);
            return null;
        }
        return buf;
    } catch { return null; }
}

// Writes are not awaited inline, but they still have to be waited on before the
// cache can be measured: a 193 MB put lands seconds after the load finishes.
const cacheWrites = [];
const flushCacheWrites = () => Promise.allSettled(cacheWrites.splice(0));

function cacheStore(url, bytes, etag) {
    if (!CACHEABLE.has(url) || !cacheAvailable()) return;

    // Built synchronously, before the buffer goes anywhere else. ORT runs in a
    // worker (wasm.proxy) and transfers the ArrayBuffer into it, which detaches
    // it: constructing the Response later captured zero bytes and poisoned the
    // cache with an empty model.
    let response;
    try {
        response = new Response(bytes, {
            headers: { "x-draw2-size": String(bytes.length), ...(etag ? { "x-draw2-etag": etag } : {}) }
        });
    } catch { return; }

    // The write itself is not awaited: 193 MB to disk must not hold up
    // compilation, and a failure (quota, private window) must not break loading.
    cacheWrites.push(caches.open(MODEL_CACHE)
        .then(c => c.put(url, response))
        .then(() => dbg(`  cached ${url.split("/").pop().split("?")[0]}`))
        .catch(() => {}));
}

// Summed from the headers rather than the bodies: reading 333 MB back just to
// measure it would defeat the point of caching it.
async function cachedBytes() {
    if (!cacheAvailable()) return 0;
    try {
        const cache = await caches.open(MODEL_CACHE);
        let total = 0;
        for (const k of await cache.keys()) {
            const r = await cache.match(k);
            total += Number(r?.headers.get("x-draw2-size") || 0);
        }
        return total;
    } catch { return 0; }
}

// Once the engine is up the load button has nothing left to load, so it becomes
// the way to get the cached weights back off the disk.
let clearCacheMode = false;

async function refreshLoadButton() {
    const btn = $("btn-load"), label = $("btn-load-label");
    if (!btn || !label || !modelsReady) return;
    const bytes = await cachedBytes();
    clearCacheMode = bytes > 0;
    btn.disabled = !clearCacheMode;
    // The progress bar sits above the button background, so it has to clear
    // for the red to show at all.
    btn.classList.toggle("btn-danger", clearCacheMode);
    if (clearCacheMode) $("lp-bar")?.style.setProperty("width", "0%");
    label.textContent = clearCacheMode
        ? T("runtime.clear_cache", { mb: (bytes / 1e6).toFixed(0) })
        : T("runtime.engine_ready");
}

async function clearModelCache() {
    try { await caches.delete(MODEL_CACHE); } catch {}
    status(T("runtime.cache_cleared"));
    await refreshLoadButton();
}

async function fetchWithProgress(url, label, fromPct, toPct, signal, step = 0) {
    const hit = await cachedResponse(url, signal);
    if (hit) {
        dbg(`  ${url.split("/").pop().split("?")[0]} served from cache`);
        setLoadStatus(label, toPct, T("runtime.dl_cached", { step, steps: DL_STEPS }));
        return hit;
    }
    const res = await fetch(url, { signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}  ${url}`);
    const total = parseInt(res.headers.get("Content-Length") || "0", 10);
    const reader = res.body.getReader();
    let received = 0;
    const chunks = [];

    let lastUi = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (total > 0) {
            const now = performance.now();
            // Repainting on every chunk is thousands of layouts on a 386 MB file
            if (now - lastUi > 250 || received === total) {
                lastUi = now;
                const frac = received / total;
                // Percentage is the whole load, not this file: it matches the
                // bar, and the file counter says where in the load we are.
                const pct = fromPct + frac * (toPct - fromPct);
                setLoadStatus(label, pct, T("runtime.dl_progress", {
                    pct: Math.round(pct),
                    step, steps: DL_STEPS,
                    done: (received / 1e6).toFixed(1),
                    total: (total / 1e6).toFixed(1)
                }));
            }
        }
    }
    const all = new Uint8Array(received);
    let pos = 0;
    for (const c of chunks) { all.set(c, pos); pos += c.length; }
    cacheStore(url, all, res.headers.get("ETag"));
    return all.buffer;
}

let loadAbortController = null;

async function init() {
    const btn = $("btn-load");
    const label = $("btn-load-label");
    const precision = document.querySelector('input[name="model"]:checked')?.value || "fp32";
    
    let vitUrl = VIT_YUGISCAN_URL;
    let namesUrl = YUGISCAN_LABELS_URL;
    if (precision === "fp16") {
        vitUrl = VIT_FP16_URL;
        namesUrl = NAMES_URL;
    } else if (precision === "fp32") {
        vitUrl = VIT_FP32_URL;
        namesUrl = NAMES_URL;
    }

    loadAbortController = new AbortController();
    const signal = loadAbortController.signal;

    if (btn && !label) btn.textContent = T("runtime.loading");
    $("load-progress")?.removeAttribute("hidden");

    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
        ort.env.logLevel = "error";
        ort.env.wasm.proxy = true; // Run WASM in a dedicated Web Worker

        setLoadStatus(T("runtime.dl_yolo"), 0, "");
        const yoloBuf = await fetchWithProgress(
            YOLO_URL, T("runtime.dl_yolo"), 0, 30, signal, 1
        );
        setLoadStatus(T("runtime.compiling_yolo"), 30, "");
        yoloSession = await ort.InferenceSession.create(yoloBuf, {
            executionProviders: ["webgpu", "wasm"],
            logSeverityLevel: 3
        });

        const vitSize = precision === "fp32" ? "386 MB" : precision === "fp16" ? "193 MB" : "98 MB";
        setLoadStatus(T("runtime.dl_vit", { size: vitSize }), 32, "");
        const vitBuf = await fetchWithProgress(
            vitUrl, T("runtime.dl_vit", { size: vitSize }), 32, 92, signal, 2
        );
        setLoadStatus(T("runtime.compiling_vit"), 92, "");
        
        // Quantized INT8 graphs run on WASM to avoid WebGPU INT8 quantization issues
        const vitProviders = (precision === "fp32" || precision === "fp16") ? ["webgpu", "wasm"] : ["wasm"];
        vitSession = await ort.InferenceSession.create(vitBuf, {
            executionProviders: vitProviders,
            logSeverityLevel: 3
        });

        const labelsBuf = await fetchWithProgress(namesUrl, "Downloading card DB", 92, 100, signal, 3);
        cardnames = JSON.parse(new TextDecoder().decode(labelsBuf));

        if (precision === "yugiscan" && Object.keys(cardId2Names).length === 0) {
            const namesBuf = await fetch(NAMES_URL, { signal }).then(r => r.arrayBuffer());
            const localized = JSON.parse(new TextDecoder().decode(namesBuf));
            for (const entry of Object.values(localized)) {
                if (entry?.card_id) cardId2Names[entry.card_id] = entry;
            }
        }

        setLoadStatus(T("runtime.warming_up"), 98, "");
        // WebGPU compiles shader pipelines on first real use, not at session
        // creation. Paying that cost here (dummy zero tensors) keeps it off
        // the first live-detection frames.
        await yoloSession.run({ [yoloSession.inputNames[0]]: new ort.Tensor("float32", new Float32Array(3 * YOLO_SIZE * YOLO_SIZE), [1, 3, YOLO_SIZE, YOLO_SIZE]) });
        await vitSession.run({ [vitSession.inputNames[0]]: new ort.Tensor("float32", new Float32Array(3 * CROP_SIZE * CROP_SIZE), [1, 3, CROP_SIZE, CROP_SIZE]) });

        setLoadStatus(T("runtime.ready"), 100, "");
        await new Promise(r => setTimeout(r, 600));

        if (btn) btn.disabled = true;
        if (label) label.textContent = T("runtime.engine_ready");
        else if (btn) btn.textContent = T("runtime.engine_ready");
        $("lp-bar")?.style.setProperty("width", "100%");
        $("load-progress")?.setAttribute("hidden", "");

        currentPrecision = precision;
        modelsReady = true;
        enableInputs();
        flushCacheWrites().then(refreshLoadButton);

        const precisionLabel = { yugiscan: "Small", fp16: "Medium", fp32: "Max" }[precision] || precision;
        status(T("runtime.model_downloaded", { model: precisionLabel }));

    } catch (err) {
        $("lp-bar")?.style.setProperty("width", "0%");
        if (err.name === "AbortError") {
            if (btn) btn.disabled = false;
            if (label) label.textContent = T("demo.btn_download");
            return;
        }
        if (label) label.textContent = T("runtime.retry");
        if (btn) btn.disabled = false;
        console.error(err);
    } finally {
        loadAbortController = null;
    }
}

// Mirror of enableInputs. Selecting another model does not load it, so the
// inputs have to go back behind the lock: otherwise a prediction runs on the
// session still in memory while the UI shows a different model selected.
function disableInputs() {
    $("dz-browse").disabled = true;
    $("btn-webcam").disabled = true;
    const dz = $("dropzone");
    dz.setAttribute("data-disabled", "");
    dz.removeAttribute("tabindex");
    $("dropzone-lock")?.removeAttribute("hidden");
    document.querySelectorAll(".sample-btn").forEach(b => b.disabled = true);
    const btnLive = $("btn-live");
    if (btnLive) btnLive.disabled = true;
}

function enableInputs() {
    $("dz-browse").disabled = false;
    $("btn-webcam").disabled = false;
    const dz = $("dropzone");
    dz.removeAttribute("data-disabled");
    dz.setAttribute("tabindex", "0");
    $("dropzone-lock")?.setAttribute("hidden", "");
    document.querySelectorAll(".sample-btn").forEach(b => b.disabled = false);

    // Small (WASM) stutters too badly for live mode; Medium/Max run on WebGPU.
    const btnLive = $("btn-live");
    if (btnLive) {
        const liveOk = currentPrecision === "fp32" || currentPrecision === "fp16";
        btnLive.disabled = !liveOk;
        btnLive.title = liveOk ? "" : "Live detection needs the Medium or Max model for smooth results";
    }
}

function setRunLabel(text) {
    const l = $("running-label");
    if (l) l.textContent = text;
}

function createSteps(labels) {
    const stepsEl = $("inference-steps");
    if (!stepsEl) return [];
    stepsEl.innerHTML = "";
    return labels.map(text => {
        const el = document.createElement("div");
        el.className = "inference-step";
        el.innerHTML = `<span class="step-icon"></span><span>${text}</span>`;
        stepsEl.appendChild(el);
        return el;
    });
}

function stepDone(el) {
    if (!el) return;
    el.className = "inference-step done";
    el.querySelector(".step-icon").textContent = "";
}
function stepActive(el) {
    if (!el) return;
    el.className = "inference-step active";
    el.querySelector(".step-icon").textContent = "";
}

function solveHomography(src, dst) {
    const A = [];
    for (let i = 0; i < 4; i++) {
        const sx = src[i].x, sy = src[i].y;
        const dx = dst[i].x, dy = dst[i].y;
        A.push([-sx, -sy, -1, 0, 0, 0, dx*sx, dx*sy, dx]);
        A.push([0, 0, 0, -sx, -sy, -1, dy*sx, dy*sy, dy]);
    }
    return solveH8x8(A);
}

function solveH8x8(A) {
    const B = A.map(row => row.slice(0, 8));
    const b = A.map(row => -row[8]);
    const h = gaussSolve(B, b);
    if (!h) return null;
    return [[h[0],h[1],h[2]],[h[3],h[4],h[5]],[h[6],h[7],1.0]];
}

function gaussSolve(A, b) {
    const n = 8;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
        let maxRow = col, maxVal = Math.abs(M[col][col]);
        for (let row = col+1; row < n; row++) {
            if (Math.abs(M[row][col]) > maxVal) { maxVal = Math.abs(M[row][col]); maxRow = row; }
        }
        [M[col], M[maxRow]] = [M[maxRow], M[col]];
        if (Math.abs(M[col][col]) < 1e-12) return null;
        const inv = 1 / M[col][col];
        for (let j = col; j <= n; j++) M[col][j] *= inv;
        for (let row = 0; row < n; row++) {
            if (row === col) continue;
            const f = M[row][col];
            for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
        }
    }
    return M.map(row => row[n]);
}

function applyH(Hinv, dx, dy) {
    const [a,b,c] = Hinv[0], [d,e,f] = Hinv[1], [g,h,i] = Hinv[2];
    const w = g*dx + h*dy + i;
    return { x: (a*dx + b*dy + c) / w, y: (d*dx + e*dy + f) / w };
}

function invertH(H) {
    const [[a,b,c],[d,e,f],[g,h,i]] = H;
    const det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
    if (Math.abs(det) < 1e-12) return null;
    const v = 1/det;
    return [
        [(e*i-f*h)*v, (c*h-b*i)*v, (b*f-c*e)*v],
        [(f*g-d*i)*v, (a*i-c*g)*v, (c*d-a*f)*v],
        [(d*h-e*g)*v, (b*g-a*h)*v, (a*e-b*d)*v]
    ];
}

function warpPerspective(srcImageData, srcPts, outW, outH) {
    const dstPts = [
        {x:0,   y:0  }, {x:outW, y:0  },
        {x:outW, y:outH}, {x:0,   y:outH},
    ];
    const H = solveHomography(srcPts, dstPts);
    if (!H) return null;
    const Hinv = invertH(H);
    if (!Hinv) return null;

    const srcW = srcImageData.width, srcH = srcImageData.height;
    const src = srcImageData.data;
    const out = new Uint8ClampedArray(outW * outH * 4);

    for (let dy = 0; dy < outH; dy++) {
        for (let dx = 0; dx < outW; dx++) {
            const { x: sx, y: sy } = applyH(Hinv, dx+.5, dy+.5);
            const x0 = Math.floor(sx), y0 = Math.floor(sy);
            const x1 = x0+1, y1 = y0+1;
            const fx = sx-x0, fy = sy-y0;
            const oi = (dy*outW + dx) * 4;
            if (x0 < 0 || y0 < 0 || x1 >= srcW || y1 >= srcH) {
                out[oi+3] = 255; continue;
            }
            const i00=(y0*srcW+x0)*4, i10=(y0*srcW+x1)*4;
            const i01=(y1*srcW+x0)*4, i11=(y1*srcW+x1)*4;
            for (let c = 0; c < 3; c++) {
                out[oi+c] = Math.round(
                    src[i00+c]*(1-fx)*(1-fy) + src[i10+c]*fx*(1-fy) +
                    src[i01+c]*(1-fx)*fy     + src[i11+c]*fx*fy
                );
            }
            out[oi+3] = 255;
        }
    }
    return new ImageData(out, outW, outH);
}

const YOLO_SIZE = 640;

function preprocessYOLO(imageData) {
    const { width: srcW, height: srcH } = imageData;
    const scale = Math.min(YOLO_SIZE/srcW, YOLO_SIZE/srcH);
    const newW = Math.round(srcW*scale), newH = Math.round(srcH*scale);
    const padX = Math.floor((YOLO_SIZE-newW)/2);
    const padY = Math.floor((YOLO_SIZE-newH)/2);

    const offscreen = new OffscreenCanvas(YOLO_SIZE, YOLO_SIZE);
    const ctx = offscreen.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);

    const small = new OffscreenCanvas(newW, newH);
    const sCtx = small.getContext("2d");
    const bmp = imageDataToBitmap(imageData);
    sCtx.drawImage(bmp, 0, 0, newW, newH);
    ctx.drawImage(small, padX, padY);
    bmp.close();

    const ld = ctx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE).data;
    const tensor = new Float32Array(3 * YOLO_SIZE * YOLO_SIZE);
    const area = YOLO_SIZE * YOLO_SIZE;
    for (let i = 0; i < area; i++) {
        tensor[i]        = ld[i*4]   / 255;
        tensor[area+i]   = ld[i*4+1] / 255;
        tensor[area*2+i] = ld[i*4+2] / 255;
    }
    return { tensor, scale, padX, padY };
}

function imageDataToBitmap(imageData) {
    const c = new OffscreenCanvas(imageData.width, imageData.height);
    c.getContext("2d").putImageData(imageData, 0, 0);
    return c.transferToImageBitmap();
}

function parseYOLOOutput(output, scale, padX, padY) {
    const data  = output.data;
    const shape = output.dims;
    const featsFirst = shape[1] < shape[2];
    const rows = featsFirst ? shape[2] : shape[1];
    const cols = featsFirst ? shape[1] : shape[2];
    const detections = [];

    for (let i = 0; i < rows; i++) {
        let xc, yc, w, h, angle, conf;
        if (featsFirst) {
            xc=data[0*rows+i]; yc=data[1*rows+i]; w=data[2*rows+i];
            h=data[3*rows+i]; conf=data[4*rows+i]; angle=data[5*rows+i];
        } else {
            xc=data[i*cols+0]; yc=data[i*cols+1]; w=data[i*cols+2];
            h=data[i*cols+3]; conf=data[i*cols+4]; angle=data[i*cols+5];
        }
        if (conf < CONF_THRESH) continue;
        const x  = (xc - padX) / scale;
        const y  = (yc - padY) / scale;
        let bw = w / scale, bh = h / scale, ang = angle;
        // Force the box onto the card's portrait convention. The crop is warped
        // into a square, so a landscape box maps the long axis onto x and the
        // card comes out both sideways and squeezed: rotating it upright after
        // the fact leaves the aspect inverted against what the ViT was trained
        // on, which caps confidence no matter which rotation wins.
        if (bw > bh) { const t = bw; bw = bh; bh = t; ang += Math.PI / 2; }
        const pts = xywhrToCorners(x, y, bw, bh, ang);
        detections.push({ pts, conf, w: bw, h: bh });
    }
    return nmsOBB(detections);
}

function xywhrToCorners(cx, cy, w, h, angle) {
    const cos=Math.cos(angle), sin=Math.sin(angle);
    const hw=w/2, hh=h/2;
    return [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]].map(([dx,dy]) => ({
        x: cx + dx*cos - dy*sin,
        y: cy + dx*sin + dy*cos
    }));
}

function nmsOBB(dets, thresh=0.5) {
    dets.sort((a,b) => b.conf - a.conf);
    const keep=[], sup=new Set();
    for (let i=0; i<dets.length; i++) {
        if (sup.has(i)) continue;
        keep.push(dets[i]);
        for (let j=i+1; j<dets.length; j++) {
            if (iouApprox(dets[i].pts, dets[j].pts) > thresh) sup.add(j);
        }
    }
    return keep;
}

function iouApprox(a, b) {
    const ax=[Math.min(...a.map(p=>p.x)),Math.max(...a.map(p=>p.x))];
    const ay=[Math.min(...a.map(p=>p.y)),Math.max(...a.map(p=>p.y))];
    const bx=[Math.min(...b.map(p=>p.x)),Math.max(...b.map(p=>p.x))];
    const by=[Math.min(...b.map(p=>p.y)),Math.max(...b.map(p=>p.y))];
    const ix=Math.max(0,Math.min(ax[1],bx[1])-Math.max(ax[0],bx[0]));
    const iy=Math.max(0,Math.min(ay[1],by[1])-Math.max(ay[0],by[0]));
    const inter=ix*iy;
    const aA=(ax[1]-ax[0])*(ay[1]-ay[0]);
    const bA=(bx[1]-bx[0])*(by[1]-by[0]);
    return inter / (aA + bA - inter + 1e-8);
}

function preprocessViT(imageData) {
    const src=imageData.data, area=CROP_SIZE*CROP_SIZE;
    const tensor=new Float32Array(3*area);
    for (let i=0; i<area; i++) {
        tensor[i]        = (src[i*4]   /255 - MEAN[0]) / STD[0];
        tensor[area+i]   = (src[i*4+1] /255 - MEAN[1]) / STD[1];
        tensor[area*2+i] = (src[i*4+2] /255 - MEAN[2]) / STD[2];
    }
    return tensor;
}

function softmax(arr) {
    const max=Math.max(...arr);
    const ex=arr.map(x=>Math.exp(x-max));
    const s=ex.reduce((a,b)=>a+b,0);
    return ex.map(x=>x/s);
}

function topK(logits, k) {
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > maxVal) maxVal = logits[i];
    
    let probs;
    if (maxVal <= 1.0) {
        probs = Array.from(logits);
    } else {
        probs = softmax(Array.from(logits));
    }
    
    return probs.map((p,i)=>({i,p})).sort((a,b)=>b.p-a.p).slice(0,k);
}

// Boxes are normalised to portrait upstream, so the card fills the crop either
// upright or flipped: only the 180 ambiguity is left, decided on which end
// carries the bright text box.
function correctRotation(imageData) {
    const w = CROP_SIZE, h = CROP_SIZE, data = imageData.data;
    const margin = Math.round(w * 0.15);
    let topL = 0, botL = 0;

    for (let y = 0; y < margin; y++) {
        for (let x = margin; x < w - margin; x++) {
            const i1 = (y * w + x) * 4;
            topL += 0.299 * data[i1] + 0.587 * data[i1+1] + 0.114 * data[i1+2];
            const i2 = ((h - 1 - y) * w + x) * 4;
            botL += 0.299 * data[i2] + 0.587 * data[i2+1] + 0.114 * data[i2+2];
        }
    }
    return topL > botL ? 180 : 0;
}

// Rotation fallback: try the flip if top-1 confidence is suspiciously low
const ROTATION_FALLBACK_CONFIDENCE = 0.15;
const ALL_ROTATIONS = [0, 180];
// A genuinely flipped card wins by an order of magnitude; a marginal gain on
// two near-zero scores is noise, and acting on it swaps one wrong label for
// another while flipping the crop shown to the user.
const ROTATION_FLIP_MARGIN = 1.25;

async function classifyCrop(imageData) {
    const vitTensor = preprocessViT(imageData);
    const vitInput  = new ort.Tensor("float32", vitTensor, [1,3,CROP_SIZE,CROP_SIZE]);
    const vitOut    = await vitSession.run({ [vitSession.inputNames[0]]: vitInput });
    const logits    = vitOut[vitSession.outputNames[0]].data;
    const top = topK(logits, VIT_TOPK);
    return { top, confidence: top[0]?.p ?? 0 };
}

async function classifyWithRotationFallback(crop) {
    const guess = correctRotation(crop);
    let bestRot = guess;
    let bestCorrected = rotateImageData(crop, guess);
    let best = await classifyCrop(bestCorrected);

    if (best.confidence < ROTATION_FALLBACK_CONFIDENCE) {
        dbg(`  low confidence (${(best.confidence*100).toFixed(1)}%) at rot=${guess}, trying other rotations`);
        // Measured against the first prediction, not against a moving best, so
        // the bar is "beats the detected orientation" however many are tried.
        const baseline = best.confidence;
        for (const rot of ALL_ROTATIONS) {
            if (rot === guess) continue;
            const corrected = rotateImageData(crop, rot);
            const result = await classifyCrop(corrected);
            const gain = baseline > 0 ? result.confidence / baseline : Infinity;
            dbg(`    rot=${rot}: ${(result.confidence*100).toFixed(1)}% (${gain.toFixed(2)}x)`);
            if (result.confidence > baseline * ROTATION_FLIP_MARGIN && result.confidence > best.confidence) {
                bestRot = rot; bestCorrected = corrected; best = result;
            } else {
                dbg(`    rot=${rot} rejected: needs >${ROTATION_FLIP_MARGIN}x to beat rot=${guess}`);
            }
        }
    }
    return { rotation: bestRot, corrected: bestCorrected, top: best.top };
}

function rotateImageData(imageData, degrees) {
    if (degrees===0) return imageData;
    const w=imageData.width, h=imageData.height;
    const c=new OffscreenCanvas(w,h);
    const ctx=c.getContext("2d");
    const bmp=imageDataToBitmap(imageData);
    ctx.translate(w/2,h/2);
    ctx.rotate(degrees*Math.PI/180);
    ctx.drawImage(bmp,-w/2,-h/2);
    bmp.close();
    return ctx.getImageData(0,0,w,h);
}

// Drawing & UI
function drawDetections(ctx, refWidth, detections, predictions, highlight = -1) {
    detections.forEach(({pts},idx) => {
        const dimmed = highlight >= 0 && idx !== highlight;
        ctx.globalAlpha = dimmed ? 0.25 : 1;
        ctx.beginPath();
        ctx.moveTo(pts[0].x,pts[0].y);
        for (let i=1;i<4;i++) ctx.lineTo(pts[i].x,pts[i].y);
        ctx.closePath();
        ctx.strokeStyle = idx === highlight ? "#10b981" : "#c8a95e";
        ctx.lineWidth=Math.max(2,refWidth/400) * (idx === highlight ? 2 : 1);
        ctx.stroke();

        if (predictions[idx]?.[0]) {
            const name=predictions[idx][0].name||"";
            const topX=Math.min(...pts.map(p=>p.x));
            const topY=Math.min(...pts.map(p=>p.y));
            const fs=Math.max(12,refWidth/60);
            ctx.font=`bold ${fs}px Inter,sans-serif`;
            const tw=ctx.measureText(name).width;
            ctx.fillStyle="rgba(0,0,0,.65)";
            ctx.fillRect(topX-2,topY-fs-4,tw+8,fs+6);
            ctx.fillStyle = idx === highlight ? "#10b981" : "#c8a95e";
            ctx.fillText(name,topX+2,topY-4);
        }
    });
    ctx.globalAlpha = 1;
}

function drawOverlay(canvas, srcImage, detections, predictions) {
    canvas.width=srcImage.width; canvas.height=srcImage.height;
    const ctx=canvas.getContext("2d");
    const bmp=imageDataToBitmap(srcImage);
    ctx.drawImage(bmp,0,0);
    bmp.close();
    drawDetections(ctx, srcImage.width, detections, predictions);
}

// Letterboxes the rendered result to match the result card's actual content
// box aspect ratio (not just its aspect-video CSS ratio: padding shifts the
// two slightly apart) with black bars, so a square/portrait source image
// never leaves uneven empty gutters around the canvas content, which would
// throw off the BorderTrail loading animation.
// Cached: reading it forces a layout pass, once per painted frame otherwise.
let resultAspect = 0;

function getResultAspect() {
    if (resultAspect) return resultAspect;
    const wrap = document.getElementById("canvas-wrap");
    if (wrap) {
        if (!wrap.dataset.aspectWatched) {
            wrap.dataset.aspectWatched = "1";
            new ResizeObserver(() => { resultAspect = 0; }).observe(wrap);
        }
        const cs = getComputedStyle(wrap);
        const w = wrap.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
        const h = wrap.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
        if (w > 0 && h > 0) return (resultAspect = w / h);
    }
    return 16 / 9;
}
// An ImageData needs a bitmap; a <video> draws straight and must not be closed.
function overlaySource(src) {
    if (src instanceof ImageData) {
        return { w: src.width, h: src.height, img: imageDataToBitmap(src), owned: true };
    }
    return { w: src.videoWidth || src.width, h: src.videoHeight || src.height, img: src, owned: false };
}

function drawOverlayLetterboxed(canvas, srcImage, detections, predictions, highlight = -1, aspectOverride = 0) {
    const src = overlaySource(srcImage);
    const targetAspect = aspectOverride || getResultAspect();
    const srcAspect = src.w / src.h;
    const canvasW = srcAspect > targetAspect ? src.w : Math.round(src.h * targetAspect);
    const canvasH = srcAspect > targetAspect ? Math.round(src.w / targetAspect) : src.h;
    // Assigning either one reallocates the buffer and resets the context.
    if (canvas.width !== canvasW || canvas.height !== canvasH) {
        canvas.width = canvasW;
        canvas.height = canvasH;
    }

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvasW, canvasH);

    const offsetX = Math.round((canvasW - src.w) / 2);
    const offsetY = Math.round((canvasH - src.h) / 2);
    const bmp = src.img;
    ctx.drawImage(bmp, offsetX, offsetY);

    ctx.save();
    ctx.translate(offsetX, offsetY);

    // Grey the other cards inside their own quad, so the highlight reads on the
    // image itself rather than on the outline alone. Clipped redraw of the same
    // bitmap: ctx.filter is ignored where unsupported, which just means no
    // desaturation instead of a broken frame.
    if (highlight >= 0) {
        for (let i = 0; i < detections.length; i++) {
            if (i === highlight) continue;
            const pts = detections[i].pts;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(pts[0].x, pts[0].y);
            for (let k = 1; k < 4; k++) ctx.lineTo(pts[k].x, pts[k].y);
            ctx.closePath();
            ctx.clip();
            ctx.filter = "grayscale(1)";
            ctx.drawImage(bmp, 0, 0);
            ctx.restore();
        }
    }

    drawDetections(ctx, src.w, detections, predictions, highlight);
    ctx.restore();
    if (src.owned) bmp.close();
}

// Set by the still-image path only. Hovering a result card redraws the output
// with that detection picked out; on an animated run the canvas shows a
// different frame, so the mapping would be wrong.
let hoverSource = null;

const HOVER_LINGER_MS = 250;
let hoverTimer = null;

function applyHover(idx) {
    if (hoverSource) {
        const { canvas, imageData, detections, predictions, aspect } = hoverSource;
        drawOverlayLetterboxed(canvas, imageData, detections, predictions, idx, aspect || 0);
    }
    const grid = $("cards-grid");
    if (!grid) return;
    for (const el of grid.children) {
        const off = idx >= 0 && Number(el.dataset.det) !== idx;
        el.style.filter  = off ? "grayscale(0.85)" : "";
        el.style.opacity = off ? "0.5" : "";
    }
}

function highlightDetection(idx) {
    clearTimeout(hoverTimer);
    if (idx >= 0) { applyHover(idx); return; }
    // Moving from one card to the next fires leave before enter, so clearing
    // at once makes the whole panel flash between every pair of cards.
    hoverTimer = setTimeout(() => applyHover(-1), HOVER_LINGER_MS);
}

const HOVER_CARD_W = 176;
let hoverCardEl = null, hoverCardPos = null, hoverCardTarget = null;
let hoverCardRaf = 0, canvasHoverIdx = -1;

function pointInQuad(px, py, pts) {
    let sign = 0;
    for (let i = 0; i < 4; i++) {
        const a = pts[i], b = pts[(i + 1) % 4];
        const cross = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
        if (!cross) continue;
        const s = cross > 0 ? 1 : -1;
        if (!sign) sign = s;
        else if (s !== sign) return false;
    }
    return true;
}

// Two nested letterboxes: buffer in element, then image in buffer.
function detectionAtPoint(clientX, clientY) {
    if (!hoverSource) return -1;
    const { canvas, imageData, detections } = hoverSource;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return -1;
    const srcW = imageData.videoWidth || imageData.width;
    const srcH = imageData.videoHeight || imageData.height;
    const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
    const x = (clientX - rect.left - (rect.width - canvas.width * scale) / 2) / scale
            - Math.round((canvas.width - srcW) / 2);
    const y = (clientY - rect.top - (rect.height - canvas.height * scale) / 2) / scale
            - Math.round((canvas.height - srcH) / 2);
    for (let i = 0; i < detections.length; i++) {
        if (pointInQuad(x, y, detections[i].pts)) return i;
    }
    return -1;
}

function ensureHoverCard() {
    if (hoverCardEl) return hoverCardEl;
    hoverCardEl = document.createElement("img");
    hoverCardEl.alt = "";
    hoverCardEl.style.cssText = "position:fixed;left:0;top:0;z-index:110;pointer-events:none;"
        + `width:${HOVER_CARD_W}px;border-radius:10px;opacity:0;`
        + "box-shadow:0 18px 40px rgba(0,0,0,.45);transition:opacity .16s ease;will-change:transform";
    document.body.appendChild(hoverCardEl);
    return hoverCardEl;
}

function moveHoverCard(clientX, clientY) {
    const h = Math.round(HOVER_CARD_W * CARD_ART_H / CARD_ART_W);
    const pad = 30;
    let x = clientX + pad;
    if (x + HOVER_CARD_W > window.innerWidth - 8) x = clientX - pad - HOVER_CARD_W;
    const y = Math.max(8, Math.min(window.innerHeight - h - 8, clientY - h / 2));
    hoverCardTarget = { x, y };
    if (!hoverCardPos) hoverCardPos = { x, y };
}

// Lag and tilt with the cursor, so the card floats instead of being pinned.
function hoverCardFrame() {
    const dx = hoverCardTarget.x - hoverCardPos.x;
    const dy = hoverCardTarget.y - hoverCardPos.y;
    hoverCardPos.x += dx * 0.16;
    hoverCardPos.y += dy * 0.16;
    const yaw   = Math.max(-14, Math.min(14, dx * 0.4));
    const pitch = Math.max(-10, Math.min(10, dy * 0.3));
    hoverCardEl.style.transform = `translate3d(${hoverCardPos.x}px, ${hoverCardPos.y}px, 0) `
        + `perspective(700px) rotateY(${yaw}deg) rotateX(${pitch}deg)`;
    hoverCardRaf = requestAnimationFrame(hoverCardFrame);
}

function showHoverCard(idx, clientX, clientY) {
    const top = hoverSource?.predictions?.[idx]?.[0];
    const src = top ? artSrcFor(top.i) : null;
    if (!src) { hideHoverCard(); return; }
    const el = ensureHoverCard();
    const wasHidden = el.style.opacity !== "1";
    el.dataset.want = String(top.i);
    Promise.resolve(src).then(u => {
        if (el.dataset.want !== String(top.i)) return;
        el.src = u || ART_PLACEHOLDER;
        el.style.opacity = "1";
    });
    moveHoverCard(clientX, clientY);
    if (wasHidden) hoverCardPos = { ...hoverCardTarget };
    if (!hoverCardRaf) hoverCardRaf = requestAnimationFrame(hoverCardFrame);
}

function hideHoverCard() {
    if (!hoverCardEl) return;
    hoverCardEl.style.opacity = "0";
    hoverCardEl.dataset.want = "";
    cancelAnimationFrame(hoverCardRaf);
    hoverCardRaf = 0;
}

// Plays a clip on the canvas, so the displayed frame is known and hit-testable.
let clipPlayer = null;

function stopClipPlayer() {
    clipPlayer?.stop();
    clipPlayer = null;
}

function startClipPlayer(canvas, source, keyResults, nearestKey) {
    stopClipPlayer();
    const isArray = Array.isArray(source);
    // Decoded once, not rebuilt on every painted frame.
    const bitmaps = isArray ? source.map(imageDataToBitmap) : null;
    const state = { idx: 0, paused: false, bitmaps };

    let target = canvas, fitNative = false;
    // Fullscreen retargets the player, so the clip keeps playing.
    state.setCanvas = (c, native = false) => { target = c; fitNative = native; };

    const paint = () => {
        const r = keyResults.get(nearestKey(state.idx)) || { detections: [], allPredictions: [] };
        const frame = isArray ? bitmaps[Math.min(state.idx, bitmaps.length - 1)] : source;
        const w = frame.videoWidth || frame.width, h = frame.videoHeight || frame.height;
        const aspect = fitNative ? w / h : 0;
        hoverSource = { canvas: target, imageData: frame, detections: r.detections, predictions: r.allPredictions, aspect };
        drawOverlayLetterboxed(target, frame, r.detections, r.allPredictions, canvasHoverIdx, aspect);
    };

    if (isArray) {
        let raf = 0, last = performance.now(), acc = 0;
        const step = 1000 / EXTRACT_FPS;
        const loop = now => {
            acc += now - last;
            last = now;
            while (acc >= step) {
                acc -= step;
                if (!state.paused) state.idx = (state.idx + 1) % bitmaps.length;
            }
            paint();
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        state.stop = () => { cancelAnimationFrame(raf); bitmaps.forEach(b => b.close()); };
        state.setPaused = p => { state.paused = p; };
    } else {
        source.loop = true;
        source.muted = true;
        source.play().catch(() => {});
        let cancelled = false;
        const rvfc = typeof source.requestVideoFrameCallback === "function";
        const onFrame = (now, meta) => {
            if (cancelled) return;
            state.idx = Math.round((meta ? meta.mediaTime : source.currentTime) * EXTRACT_FPS);
            paint();
            schedule();
        };
        const schedule = () => {
            if (rvfc) source.requestVideoFrameCallback(onFrame);
            else requestAnimationFrame(() => onFrame(0, null));
        };
        schedule();
        state.stop = () => { cancelled = true; source.pause(); };
        state.setPaused = p => { p ? source.pause() : source.play().catch(() => {}); };
    }

    clipPlayer = state;
    return state;
}

function setupCanvasHover() {
    for (const id of ["canvas-out", "fullscreen-viewer-canvas"]) bindCanvasHover($(id));
}

function bindCanvasHover(canvas) {
    if (!canvas) return;
    canvas.addEventListener("mousemove", e => {
        const idx = detectionAtPoint(e.clientX, e.clientY);
        if (idx !== canvasHoverIdx) {
            canvasHoverIdx = idx;
            highlightDetection(idx);
            clipPlayer?.setPaused(idx >= 0);
            if (idx >= 0) showHoverCard(idx, e.clientX, e.clientY);
            else hideHoverCard();
        } else if (idx >= 0) {
            moveHoverCard(e.clientX, e.clientY);
        }
    });
    canvas.addEventListener("mouseleave", () => {
        canvasHoverIdx = -1;
        highlightDetection(-1);
        clipPlayer?.setPaused(false);
        hideHoverCard();
    });
}

//  RESULT CARDS 
function renderResultCards(grid, croppedImages, predictions) {
    grid.innerHTML="";
    const hint = $("compare-hint");
    if (hint) {
        hint.hidden = croppedImages.every(c => !c);
        if (localStorage.getItem("draw2_compare_seen")) hint.classList.remove("animate-pulse");
    }
    croppedImages.forEach((cropData,idx) => {
        if (!cropData) return;
        const top=predictions[idx]?.[0];
        const name=top?.name||top?.label||"Unknown";
        const score=top ? (top.p*100).toFixed(1)+"%" : "";

        const item=document.createElement("div");
        item.className="flex bg-white border border-zinc-200 rounded overflow-hidden shadow-sm hover:shadow-md group";
        item.style.transition = "box-shadow .2s, filter .15s, opacity .15s";
        item.dataset.det = idx;

        const cc=document.createElement("canvas");
        cc.width=CROP_SIZE; cc.height=CROP_SIZE; cc.className="w-24 h-24 object-contain bg-zinc-900 shrink-0 border-r border-zinc-200";
        cc.getContext("2d").putImageData(cropData,0,0);

        const meta=document.createElement("div"); meta.className="p-3 flex-1 min-w-0 flex flex-col justify-center";
        const nameEl=document.createElement("div"); nameEl.className="font-display font-bold text-zinc-900 truncate mb-1";
        nameEl.textContent=name; nameEl.title=name;
        const scoreEl=document.createElement("div"); scoreEl.className="font-mono text-emerald-600 text-sm font-semibold";
        scoreEl.textContent=score;

        meta.appendChild(nameEl); meta.appendChild(scoreEl);
        item.appendChild(cc); item.appendChild(meta);

        // A name alone is unverifiable for anyone who does not know the cards by
        // heart, so the whole card opens a side-by-side against the real artwork.
        const preds = predictions[idx] || [];
        item.classList.add("cursor-zoom-in", "text-left", "w-full");
        item.setAttribute("role", "button");
        item.tabIndex = 0;
        item.title = T("demo.compare_open_title");
        const open = () => {
            localStorage.setItem("draw2_compare_seen", "1");
            $("compare-hint")?.classList.remove("animate-pulse");
            openCompare(cropData, preds);
        };
        item.addEventListener("click", open);
        item.addEventListener("mouseenter", () => highlightDetection(idx));
        item.addEventListener("mouseleave", () => highlightDetection(-1));
        item.addEventListener("keydown", e => {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
        });

        const badge=document.createElement("div");
        badge.className="self-center pr-3 text-zinc-300 group-hover:text-emerald-600 transition-colors shrink-0";
        badge.innerHTML='<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-4.35-4.35M11 19a8 8 0 100-16 8 8 0 000 16zM11 8v6M8 11h6"/></svg>';
        item.appendChild(badge);

        grid.appendChild(item);
    });

    prefetchArtwork(predictions);
}

//  COMPARE VIEWER 
// Shows the detected crop against the official artwork of a candidate, so a
// prediction can be judged on the image rather than on a name the user may
// not recognise. Clicking a runner-up swaps the right pane.
// Native size of a YGOPRODeck card image, so both panes share one shape
const CARD_ART_W = 421, CARD_ART_H = 614;
const ALT_RELEVANCE_RATIO = 0.1; // a runner-up must reach 10% of the top score

function openCompare(cropData, predictions) {
    const viewer = $("compare-viewer");
    if (!viewer || !predictions?.length) return;

    // The crop is a card warped into a square, so showing it as-is puts a
    // flattened card next to a correctly proportioned one. Undo the squeeze by
    // redrawing it at the artwork's own proportions.
    const cropCanvas = $("cmp-crop");
    if (cropCanvas && cropData) {
        cropCanvas.width = CARD_ART_W;
        cropCanvas.height = CARD_ART_H;
        const bmp = imageDataToBitmap(cropData);
        cropCanvas.getContext("2d").drawImage(bmp, 0, 0, CARD_ART_W, CARD_ART_H);
        bmp.close();
    }

    const art     = $("cmp-art");
    const missing = $("cmp-art-missing");
    const nameEl  = $("cmp-name");
    const scoreEl = $("cmp-score");

    function show(pred) {
        if (nameEl)  nameEl.textContent  = pred.name;
        if (scoreEl) scoreEl.textContent = (pred.p * 100).toFixed(1) + "%";
        const src = artSrcFor(pred.i);
        if (art) art.dataset.want = String(pred.i);
        Promise.resolve(src).then(u => {
            if (art && art.dataset.want !== String(pred.i)) return;
            if (art) {
                art.hidden = false;
                art.src = u || ART_PLACEHOLDER;
                art.alt = pred.name;
            }
            if (missing) missing.hidden = !!u;
        });
    }

    // Runners-up are only worth showing when the model is actually hesitating.
    // Against a confident top-1 they sit near zero and are pure noise, so the
    // cut is relative to the top score rather than absolute.
    const alts     = $("cmp-alts");
    const altsWrap = $("cmp-alts-wrap");
    const floor    = predictions[0].p * ALT_RELEVANCE_RATIO;
    const shown    = [predictions[0], ...predictions.slice(1).filter(pr => pr.p >= floor)];
    if (alts) {
        alts.innerHTML = "";
        shown.forEach(pred => {
            const b = document.createElement("button");
            b.className = "btn flex items-center gap-2 pr-2.5 rounded border border-zinc-200 dark:border-white/10 hover:border-emerald-500 overflow-hidden bg-white dark:bg-white/5";
            const thumbSrc = artSrcFor(pred.i);
            if (thumbSrc) {
                const th = document.createElement("img");
                th.alt = ""; th.loading = "lazy";
                Promise.resolve(thumbSrc).then(u => { th.src = u || ART_PLACEHOLDER; });
                th.className = "w-8 h-11 object-cover shrink-0";
                th.addEventListener("error", () => th.remove());
                b.appendChild(th);
            }
            const txt = document.createElement("span");
            txt.className = "text-xs text-left max-w-[11rem] truncate text-zinc-700 dark:text-white/80";
            txt.textContent = pred.name;
            const pct = document.createElement("span");
            pct.className = "font-mono text-xs text-emerald-600 shrink-0";
            pct.textContent = (pred.p * 100).toFixed(1) + "%";
            b.appendChild(txt); b.appendChild(pct);
            b.addEventListener("click", () => show(pred));
            alts.appendChild(b);
        });
    }
    if (altsWrap) altsWrap.hidden = shown.length < 2;

    show(predictions[0]);
    viewer.hidden = false;
}

function setupCompareViewer() {
    const viewer = $("compare-viewer");
    if (!viewer) return;
    const close = () => { viewer.hidden = true; const a = $("cmp-art"); if (a) a.src = ""; };
    $("btn-compare-close")?.addEventListener("click", close);
    viewer.addEventListener("click", e => { if (e.target === viewer) close(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape" && !viewer.hidden) close(); });
}

async function detectAndClassify(imageData, { showSteps = true, onDetections, onCard } = {}) {
    const stepsEl = showSteps ? $("inference-steps") : null;
    if (stepsEl) stepsEl.innerHTML = "";

    if (cancelRequested) throw new Error("Cancelled by user");

    setRunLabel("Detecting cards......");
    const steps = showSteps ? createSteps(["YOLO - object detection", "Warping card crops"]) : [];
    if (showSteps) { stepActive(steps[0]); await nextFrame(); }

    dbg(`Image: ${imageData.width}${imageData.height}`);
    if (cancelRequested) throw new Error("Cancelled by user");

    const { tensor: yoloTensor, scale, padX, padY } = preprocessYOLO(imageData);
    dbg(`YOLO preprocess done  scale=${scale.toFixed(3)} padX=${padX} padY=${padY}`);

    const yoloInput = new ort.Tensor("float32", yoloTensor, [1,3,YOLO_SIZE,YOLO_SIZE]);
    dbg("Running YOLO inference......");
    const yoloOut   = await yoloSession.run({ [yoloSession.inputNames[0]]: yoloInput });
    const rawOut    = yoloOut[yoloSession.outputNames[0]];
    dbg(`YOLO output shape: [${rawOut.dims.join(", ")}]`);

    const detections = parseYOLOOutput(rawOut, scale, padX, padY);
    dbg(`Detections after NMS: ${detections.length} (conf${CONF_THRESH})`);
    onDetections?.(detections);

    if (showSteps) stepDone(steps[0]);
    if (cancelRequested) throw new Error("Cancelled by user");

    if (detections.length === 0) {
        return { detections: [], allPredictions: [], croppedImages: [] };
    }

    setRunLabel(`Classifying card${detections.length>1?"s":""}...`);
    if (showSteps) stepActive(steps[1]);
    const vitSteps = detections.map((_, i) => {
        if (!showSteps) return null;
        const el = document.createElement("div");
        el.className = "inference-step";
        el.innerHTML = `<span class="step-icon"></span><span>ViT - card ${i+1}/${detections.length}</span>`;
        if (stepsEl) stepsEl.appendChild(el);
        return el;
    });
    if (showSteps) stepDone(steps[1]);

    const croppedImages = [];
    const allPredictions = [];

    for (let idx = 0; idx < detections.length; idx++) {
        if (cancelRequested) {
            dbg("Pipeline cancelled during ViT loop.");
            throw new Error("Cancelled by user");
        }
        const det = detections[idx];
        if (showSteps) stepActive(vitSteps[idx]);
        await nextFrame(); // yield before each card's ViT call so a busy scene doesn't freeze rendering

        dbg(`Card ${idx+1}: warping perspective`);
        const t0 = performance.now();
        const crop = warpPerspective(imageData, det.pts, CROP_SIZE, CROP_SIZE);
        dbg(`  warpPerspective done in ${(performance.now()-t0).toFixed(0)}ms`);

        if (!crop) {
            console.warn(`  Card ${idx+1}: warpPerspective returned null (degenerate homography?)`);
            if (showSteps) stepDone(vitSteps[idx]);
            croppedImages.push(null);
            allPredictions.push([]);
            continue;
        }

        dbg(`  Running ViT......`);
        const t1 = performance.now();
        const { rotation, corrected, top } = await classifyWithRotationFallback(crop);
        dbg(`  rotation correction: ${rotation}`);
        dbg(`  ViT done in ${(performance.now()-t1).toFixed(0)}ms`);
        croppedImages.push(corrected);

        const topPreds  = top.map(({i,p}) => ({ name: cardNameFor(cardnames[String(i)], i), p, i }));
        dbg(`  Top-1: "${topPreds[0]?.name}" (${(topPreds[0]?.p*100).toFixed(1)}%)`);
        onCard?.(idx, detections.length, topPreds);
        allPredictions.push(topPreds);
        if (showSteps) stepDone(vitSteps[idx]);
    }

    return { detections, allPredictions, croppedImages };
}

// Main Pipeline
async function runPipeline(imageData) {
    if (cancelRequested) return;
    stopClipPlayer();

    const dl = $("gif-dl"); if (dl) dl.remove();
    if ($("canvas-out")) $("canvas-out").style.display = "";

    const resultsEl  = $("results");
    const grid       = $("cards-grid");
    const canvasOut  = $("canvas-out");
    const canvasWrap = $("canvas-wrap");
    const runRow     = $("running-row");

    cancelRequested = false;
    const pipelineT0 = performance.now();

    if (runRow) runRow.hidden = false;
    if (resultsEl) resultsEl.hidden = true;
    if (canvasWrap) canvasWrap.hidden = false;

    // Letterbox the preview up front, before inference runs: the BorderTrail
    // loading effect hugs the card edge, so the shown frame must already be
    // padded to the card's aspect ratio, not just the final result.
    if (canvasOut) drawOverlayLetterboxed(canvasOut, imageData, [], []);

    // Pause decorative background during inference to preserve GPU budget
    window.__bgAnim?.pause();
    if ($("canvas-trail")) $("canvas-trail").hidden = false;
    try {
        const { detections, allPredictions, croppedImages } = await detectAndClassify(imageData, {
            onDetections: dets => status(T("log.found_cards", { count: dets.length })),
            onCard: (idx, total, topPreds) => status(T("log.card_progress", { idx: idx+1, total, name: topPreds[0]?.name, pct: ((topPreds[0]?.p ?? 0)*100).toFixed(1) }))
        });

        if (detections.length === 0) {
            if (runRow) runRow.hidden = true;
            if (canvasOut) drawOverlayLetterboxed(canvasOut, imageData, [], []);
            if (grid) grid.innerHTML = `<p style="grid-column:1/-1;color:var(--muted);font-size:.875rem;">No cards found above confidence threshold (${CONF_THRESH*100}%). Try a clearer image.</p>`;
            if (resultsEl) resultsEl.hidden = false;
            return;
        }

        dbg("Rendering overlay");
        if (canvasOut) drawOverlayLetterboxed(canvasOut, imageData, detections, allPredictions);
        hoverSource = canvasOut ? { canvas: canvasOut, imageData, detections, predictions: allPredictions } : null;
        if (grid) renderResultCards(grid, croppedImages, allPredictions);
        if (runRow) runRow.hidden = true;
        if (resultsEl) resultsEl.hidden = false;
        status(T("log.done_in", { ms: (performance.now()-pipelineT0).toFixed(0) }));

    } catch (err) {
        if (err.message === "Cancelled by user") {
            dbg("Execution aborted.");
            if (runRow) runRow.hidden = true;
            if (canvasWrap) canvasWrap.hidden = true;
            resetUI();
            return;
        }
        console.error("Pipeline error:", err.message, err.stack);
        const runRow = $("running-row");
        const grid = $("cards-grid");
        const resultsEl = $("results");
        if (runRow) runRow.hidden = true;
        setRunLabel("");
        if (grid) grid.innerHTML = `<p style="grid-column:1/-1;color:var(--danger);font-size:.875rem;">Error: ${err.message}</p>`;
        if (resultsEl) resultsEl.hidden = false;
    } finally {
        window.__bgAnim?.resume();
        if ($("canvas-trail")) $("canvas-trail").hidden = true;
    }
}

function nextFrame() {
    return new Promise(r => requestAnimationFrame(r));
}

// Image Input
function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            const c = document.createElement("canvas");
            c.width=img.naturalWidth; c.height=img.naturalHeight;
            c.getContext("2d").drawImage(img,0,0);
            URL.revokeObjectURL(url);
            resolve(c.getContext("2d").getImageData(0,0,c.width,c.height));
        };
        img.onerror = reject;
        img.src = url;
    });
}

function hideDropzoneOnLoad() {
    const dz = $("dropzone");
    if (dz && !("keepVisible" in dz.dataset)) dz.hidden = true;
}

function resetUI() {
    stopClipPlayer();
    hoverSource = null;
    if ($("canvas-wrap")) $("canvas-wrap").hidden = true;
    if ($("results")) $("results").hidden = true;
    if ($("running-row")) $("running-row").hidden = true;
    if ($("dropzone")) $("dropzone").hidden = false;
    if ($("webcam-row")) $("webcam-row").style.display = "";
}

function setupLoadButton() {
    const btnCancel = $("btn-cancel");
    if (btnCancel) btnCancel.addEventListener("click", () => { cancelRequested = true; });

    const btn = $("btn-load");
    if (btn) btn.addEventListener("click", () => {
        if (loadAbortController) loadAbortController.abort();
        else if (clearCacheMode) clearModelCache();
        else init();
    });

    document.querySelectorAll('input[name="model"]').forEach(radio => {
        radio.addEventListener("change", () => {
            if (!modelsReady || !btn) return;
            if (radio.value !== currentPrecision) {
                clearCacheMode = false;
                btn.disabled = false;
                btn.classList.remove("btn-danger");
                $("btn-load-label").textContent = T("demo.btn_download");
                $("lp-bar")?.style.setProperty("width", "0%");
                // The session in memory is still the previous model.
                liveActive = false;
                stopWebcam();
                disableInputs();
            } else {
                // Back on the model that is actually loaded, so it is usable again.
                enableInputs();
                refreshLoadButton();
            }
        });
    });
}

function setupDropzone() {
    const dz      = $("dropzone");
    const input   = $("file-input");
    const browse  = $("dz-browse");

    if (!dz || !input) return;

    const guard = () => { if (!modelsReady) return false; return true; };

    if (browse) browse.addEventListener("click", e => { e.stopPropagation(); if(guard()) input.click(); });
    dz.addEventListener("click", () => { if(guard()) input.click(); });
    dz.addEventListener("keydown", e => { if((e.key==="Enter"||e.key===" ")&&guard()) input.click(); });

    dz.addEventListener("dragover", e => {
        e.preventDefault();
        if (modelsReady) dz.classList.add("drag-over");
    });
    dz.addEventListener("dragleave", e => {
        e.preventDefault();
        dz.classList.remove("border-emerald-500", "bg-emerald-50");
    });
    dz.addEventListener("drop", e => {
        e.preventDefault();
        dz.classList.remove("border-emerald-500", "bg-emerald-50");
        if(guard() && e.dataTransfer.files[0]) handleImage(e.dataTransfer.files[0]);
    });

    document.addEventListener("paste", (e) => {
        if (!modelsReady) return;
        if (cancelRequested) return;
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf("image") !== -1) {
                const file = items[i].getAsFile();
                if (file) {
                    e.preventDefault();
                    handleImage(file);
                    return;
                }
            }
        }
    });

    input.addEventListener("change", () => { if(input.files[0]) handleImage(input.files[0]); });
}

function setupSampleButtons() {
    // Sits inside #dropzone; don't also trigger its file picker on click.
    $("source-video-link")?.addEventListener("click", e => e.stopPropagation());

    document.querySelectorAll(".sample-btn").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation(); // buttons sit inside #dropzone; don't also trigger its file picker
            if (!modelsReady || btn.disabled) return;
            const url  = btn.dataset.sample;
            const type = btn.dataset.sampleType || "";
            const name = url.split("/").pop();
            try {
                status(T("log.loading_sample", { name }));
                const res = await fetch(url);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const buf  = await res.arrayBuffer();
                const file = new File([buf], name, { type });
                hideDropzoneOnLoad();
                if ($("webcam-row")) $("webcam-row").style.display = "none";
                await handleImage(file);
            } catch (err) {
                console.error("Failed to load sample:", err.message);
                alert(`Couldn't load sample file: ${err.message}`);
            }
        });
    });
}

async function handleImage(file) {
    hideDropzoneOnLoad();
    if ($("webcam-row")) $("webcam-row").style.display = "none";
    if (file.type === 'image/gif' || file.type.startsWith('video/')) {
        await processAnimated(file);
    } else {
        const imageData = await loadImageFromFile(file);
        await runPipeline(imageData);
    }
}

// Live webcam detection: YOLO runs every tick, ViT classification refreshes periodically
let liveActive = false;
const LIVE_CLASSIFY_INTERVAL_MS = 700;

function centroidOf(pts) {
    const x = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const y = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    return { x, y };
}

async function liveLoop(video) {
    const liveCanvas  = $("webcam-live-canvas");
    const latencyEl   = $("webcam-live-latency");
    const scratch = document.createElement("canvas");
    const sctx = scratch.getContext("2d", { willReadFrequently: true });

    let heldDetections  = [];
    let heldPredictions = [];
    let lastClassifyAt  = 0;

    while (liveActive) {
        const t0 = performance.now();
        if (!video.videoWidth) { await nextFrame(); continue; } // stream not ready yet

        scratch.width = video.videoWidth;
        scratch.height = video.videoHeight;
        sctx.drawImage(video, 0, 0);
        const imageData = sctx.getImageData(0, 0, scratch.width, scratch.height);

        try {
            // -- YOLO: every tick, cheap --
            const { tensor: yoloTensor, scale, padX, padY } = preprocessYOLO(imageData);
            const yoloInput = new ort.Tensor("float32", yoloTensor, [1,3,YOLO_SIZE,YOLO_SIZE]);
            const yoloOut   = await yoloSession.run({ [yoloSession.inputNames[0]]: yoloInput });
            const detections = parseYOLOOutput(yoloOut[yoloSession.outputNames[0]], scale, padX, padY);
            if (!liveActive) break;

            const countChanged  = detections.length !== heldDetections.length;
            const dueToRefresh  = performance.now() - lastClassifyAt > LIVE_CLASSIFY_INTERVAL_MS;
            let predictions;

            if (detections.length === 0) {
                predictions = [];
                heldDetections = []; heldPredictions = [];
            } else if (countChanged || dueToRefresh) {
                // -- ViT: only here, the expensive part --
                predictions = [];
                for (const det of detections) {
                    if (!liveActive) break;
                    const crop = warpPerspective(imageData, det.pts, CROP_SIZE, CROP_SIZE);
                    if (!crop) { predictions.push([]); continue; }
                    const { top } = await classifyWithRotationFallback(crop);
                    predictions.push(top.map(({i, p}) => ({ name: cardNameFor(cardnames[String(i)], i), p, i })));
                    await nextFrame(); // yield between cards so a busy scene doesn't freeze the tab
                }
                if (!liveActive) break;
                heldDetections  = detections;
                heldPredictions = predictions;
                lastClassifyAt  = performance.now();
            } else {
                // Re-pair previous labels to new boxes by nearest centroid
                predictions = detections.map(det => {
                    const c = centroidOf(det.pts);
                    let best = 0, bestDist = Infinity;
                    heldDetections.forEach((hd, i) => {
                        const hc = centroidOf(hd.pts);
                        const d = Math.hypot(c.x - hc.x, c.y - hc.y);
                        if (d < bestDist) { bestDist = d; best = i; }
                    });
                    return heldPredictions[best] || [];
                });
            }

            if (liveCanvas) drawOverlay(liveCanvas, imageData, detections, predictions);
            if (latencyEl) latencyEl.textContent = `${(performance.now() - t0).toFixed(0)}ms`;
        } catch (err) {
            console.error("Live detection error:", err.message);
            break;
        }
        await nextFrame();
    }
}

function setupWebcam() {
    const btnStart = $("btn-webcam");
    const btnSnap  = $("btn-snap");
    const btnStop  = $("btn-stop-webcam");
    const btnLive  = $("btn-live");
    const video    = $("webcam-video");
    const wrap     = $("webcam-wrap");
    const liveCanvas  = $("webcam-live-canvas");
    const liveBadge   = $("webcam-live-badge");

    if (!btnStart || !video) return;

    function stopLive() {
        liveActive = false;
        window.__bgAnim?.resume();
        if (btnLive) btnLive.textContent = T("runtime.live_detect");
        if (liveCanvas) liveCanvas.hidden = true;
        if (liveBadge) liveBadge.hidden = true;
    }

    btnStart.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!modelsReady) return;
        try {
            currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", aspectRatio: { ideal: 16/9 } } });
            video.srcObject = currentStream;
            hideDropzoneOnLoad();
            const webcamRow = btnStart.closest(".webcam-row");
            if (webcamRow) webcamRow.style.display = "none";
            if ($("canvas-wrap")) $("canvas-wrap").hidden = true;
            if ($("results")) $("results").hidden = true;
            $("gif-dl")?.remove();
            if (wrap) wrap.hidden = false;
        } catch { alert("Webcam access denied or unavailable."); }
    });

    if (btnLive) btnLive.addEventListener("click", () => {
        if (!modelsReady) return;
        if (liveActive) {
            stopLive();
            return;
        }
        if (currentPrecision !== "fp32" && currentPrecision !== "fp16") {
            alert("Live detection needs the Medium or Max model for smooth results. Switch precision and reload the engine to use it.");
            return;
        }
        liveActive = true;
        window.__bgAnim?.pause();
        btnLive.textContent = T("runtime.stop_detecting");
        if (liveCanvas) liveCanvas.hidden = false;
        if (liveBadge) liveBadge.hidden = false;
        liveLoop(video);
    });

    if (btnSnap) btnSnap.addEventListener("click", () => {
        stopLive();
        const c = document.createElement("canvas");
        c.width=video.videoWidth; c.height=video.videoHeight;
        c.getContext("2d").drawImage(video,0,0);
        stopWebcam();
        if (wrap) wrap.hidden = true;
        runPipeline(c.getContext("2d").getImageData(0,0,c.width,c.height));
    });

    if (btnStop) btnStop.addEventListener("click", () => { stopLive(); stopWebcam(); if (wrap) wrap.hidden=true; resetUI(); });
}

function stopWebcam() {
    currentStream?.getTracks().forEach(t => t.stop());
    currentStream = null;
}

// Keeps the fullscreen bubble glued to the image's own corner, not the
// letterboxed canvas box: object-contain can leave empty gutter around the
// image, so the button offset is derived from the actual rendered rect.
function setupFullscreenButtonPosition() {
    const canvasOut = $("canvas-out");
    const btn = $("btn-fullscreen");
    if (!canvasOut || !btn) return;

    function reposition() {
        const boxW = canvasOut.clientWidth, boxH = canvasOut.clientHeight;
        if (!boxW || !boxH || !canvasOut.width || !canvasOut.height) return;
        const scale = Math.min(boxW / canvasOut.width, boxH / canvasOut.height);
        const offsetX = (boxW - canvasOut.width * scale) / 2;
        const offsetY = (boxH - canvasOut.height * scale) / 2;
        const margin = 14;
        btn.style.top = (offsetY + margin) + "px";
        btn.style.right = (offsetX + margin) + "px";
    }
    new ResizeObserver(reposition).observe(canvasOut);
}

// Fullscreen Viewer
function setupFullscreenViewer() {
    const btn      = $("btn-fullscreen");
    const closeBtn = $("btn-fullscreen-close");
    const viewer   = $("fullscreen-viewer");
    const img      = $("fullscreen-viewer-img");
    if (!btn || !viewer || !img) return;

    const fsCanvas = $("fullscreen-viewer-canvas");

    function open() {
        const canvasOut = $("canvas-out");
        // Tailwind's preflight makes canvas/img display:block, which beats the
        // UA rule for [hidden], so these have to be switched on style.
        if (clipPlayer && fsCanvas) {
            img.style.display = "none";
            fsCanvas.style.display = "";
            clipPlayer.setCanvas(fsCanvas, true);
        } else if (fsCanvas && hoverSource && hoverSource.canvas === canvasOut) {
            // A still keeps its overlay and stays hoverable, enlarged.
            const src = hoverSource.imageData;
            img.style.display = "none";
            fsCanvas.style.display = "";
            hoverSource.canvas = fsCanvas;
            hoverSource.aspect = (src.videoWidth || src.width) / (src.videoHeight || src.height);
            applyHover(canvasHoverIdx);
        } else if (canvasOut && canvasOut.width) {
            if (fsCanvas) fsCanvas.style.display = "none";
            img.style.display = "";
            img.src = canvasOut.toDataURL();
        } else {
            return;
        }
        viewer.hidden = false;
    }
    function close() {
        viewer.hidden = true;
        img.src = "";
        if (fsCanvas) fsCanvas.style.display = "none";
        img.style.display = "";
        const canvasOut = $("canvas-out");
        clipPlayer?.setCanvas(canvasOut, false);
        if (!clipPlayer && hoverSource && hoverSource.canvas === fsCanvas) {
            hoverSource.canvas = canvasOut;
            hoverSource.aspect = 0;
            applyHover(-1);
        }
    }

    btn.addEventListener("click", open);
    closeBtn?.addEventListener("click", close);
    viewer.addEventListener("click", e => { if (e.target === viewer) close(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape" && !viewer.hidden) close(); });
}

// Boot
document.addEventListener("DOMContentLoaded", () => {
    setupLoadButton();
    setupDropzone();
    setupWebcam();
    setupSampleButtons();
    setupFullscreenViewer();
    setupFullscreenButtonPosition();
    setupCompareViewer();
    setupCanvasHover();
    injectVerbosityToggle();
});

// Video/GIF processing parameters
const MAX_PREDICTION_SIDE = 1920;
const GIF_OUTPUT_SIDE = 1280;
const EXTRACT_FPS = 15;
const INFERENCE_FPS = 5;

// Under ~0.8s on screen is capture noise. INFERENCE_FPS key frames = 1s.
const MIN_CARD_FRAMES = 4;

function confirmedCards(keyResults) {
    const count = new Map(), names = new Map();
    for (const { allPredictions } of keyResults.values()) {
        const seen = new Set();
        for (const preds of allPredictions) {
            const top = preds?.[0];
            if (top?.i == null) continue;
            seen.add(top.i);
            if (!names.has(top.i)) names.set(top.i, top.name);
        }
        for (const i of seen) count.set(i, (count.get(i) || 0) + 1);
    }
    const floor = keyResults.size < MIN_CARD_FRAMES ? 1 : MIN_CARD_FRAMES;
    const ok = new Set();
    for (const [i, n] of count) if (n >= floor) ok.add(i);
    dbg(`Card frame counts (${keyResults.size} key frames, floor ${floor}): `
        + [...count].sort((a, b) => b[1] - a[1])
            .map(([i, n]) => `${names.get(i)}=${n}${ok.has(i) ? "" : " DROPPED"}`).join(", "));
    return ok;
}

function quadCenter(pts) {
    let x = 0, y = 0;
    for (const p of pts) { x += p.x; y += p.y; }
    return { x: x / 4, y: y / 4 };
}

// Predictions of the closest box sitting at the same spot in an adjacent key
// frame, or null: a stable box with a bad label, as opposed to a ghost.
function neighbourPreds(det, around) {
    const c = quadCenter(det.pts);
    const tol = Math.hypot(det.pts[0].x - det.pts[2].x, det.pts[0].y - det.pts[2].y) * 0.4;
    let best = null, bestDist = Infinity;
    for (const frame of around) {
        frame.detections.forEach((o, k) => {
            const oc = quadCenter(o.pts);
            const d = Math.hypot(oc.x - c.x, oc.y - c.y);
            if (d <= tol && d < bestDist) { bestDist = d; best = frame.allPredictions[k] || []; }
        });
    }
    return best;
}

function denoiseKeyResults(keyResults, keyIndices, confirmed) {
    const order = keyIndices.filter(i => keyResults.has(i));
    const raw = new Map(order.map(i => [i, keyResults.get(i)]));
    let dropped = 0, relabelled = 0, unlabelled = 0;
    order.forEach((idx, n) => {
        const { detections, allPredictions } = keyResults.get(idx);
        const around = [order[n - 1], order[n + 1]].filter(v => v != null).map(v => raw.get(v));
        const dets = [], preds = [];
        detections.forEach((det, j) => {
            const list = allPredictions[j] || [];
            if (confirmed.has(list[0]?.i)) { dets.push(det); preds.push(list); return; }
            const alt = list.find(p => confirmed.has(p.i));
            if (alt) { dets.push(det); preds.push([alt, ...list.filter(p => p !== alt)]); relabelled++; return; }
            const nb = neighbourPreds(det, around);
            if (nb) {
                const carried = nb.find(p => confirmed.has(p.i));
                dets.push(det);
                preds.push(carried ? [carried] : []);
                if (carried) relabelled++; else unlabelled++;
                return;
            }
            dropped++;
        });
        keyResults.set(idx, { detections: dets, allPredictions: preds });
    });
    dbg(`Denoise: ${dropped} ghost box(es) removed, ${relabelled} relabelled, ${unlabelled} left unlabelled`);
}

async function processAnimated(file) {
    const dl = $("gif-dl"); if (dl) dl.remove();
    if ($("results")) $("results").hidden = true; // don't carry over the last run's cards
    stopClipPlayer();
    hoverSource = null; // the canvas will be showing frames, not the hovered image
    if ($("canvas-out")) $("canvas-out").style.display = "";
    if ($("canvas-wrap")) $("canvas-wrap").hidden = false;
    const runRow = $("running-row");
    if (runRow) runRow.hidden = false;
    cancelRequested = false;

    function cancelCleanup() {
        dbg("Execution aborted.");
        if (runRow) runRow.hidden = true;
        if ($("canvas-wrap")) $("canvas-wrap").hidden = true;
        resetUI();
    }

    // Declared outside the try below: the export runs after that block's finally.
    let playbackVideo = null, canvasOut = null, gifW = 0, gifH = 0, clipCount = 0;
    const frames = [];
    const keyIndices = [];
    const keyResults = new Map();

    function nearestKey(i) {
        let best = keyIndices[0], bestDist = Infinity;
        for (const k of keyIndices) {
            const d = Math.abs(k - i);
            if (d < bestDist) { bestDist = d; best = k; }
        }
        return best;
    }

    window.__bgAnim?.pause();
    if ($("canvas-trail")) $("canvas-trail").hidden = false;
    try {

    if (!window.GIF) {
        setRunLabel("Loading GIF encoder...");
        status(T("log.loading_gif_encoder"));
        await new Promise(r => {
            const s = document.createElement("script");
            s.src = "https://cdn.jsdelivr.net/npm/gif.js@0.2.0/dist/gif.js";
            s.onload = r;
            document.head.appendChild(s);
        });
    }

    setRunLabel("Extracting frames...");

    // Extraction takes seconds; show frame 0 as soon as it exists so the
    // capsule is filled at the right size immediately, like a still image.
    const previewTarget = $("canvas-out");
    function previewFirstFrame() {
        if (frames.length !== 1 || !previewTarget) return;
        drawOverlayLetterboxed(previewTarget, frames[0], [], []);
    }

    if (file.type.startsWith("video/")) {
        const video = document.createElement("video");
        playbackVideo = video;
        video.src = URL.createObjectURL(file);
        video.muted = true;
        await new Promise((r, reject) => { video.onloadeddata = r; video.onerror = reject; });
        if (video.duration > 15) {
            alert("Please use a short clip (under 15s) for this in-browser demo.");
            if (runRow) runRow.hidden = true;
            if ($("canvas-wrap")) $("canvas-wrap").hidden = true;
            resetUI();
            return;
        }

        const canvas = document.createElement("canvas");
        const scale = Math.min(1.0, MAX_PREDICTION_SIDE / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        const ctx = canvas.getContext("2d", { willReadFrequently: true });

        const totalFrames = Math.floor(video.duration * EXTRACT_FPS);
        status(T("log.extract_video", { dur: video.duration.toFixed(1), fps: EXTRACT_FPS, w: canvas.width, h: canvas.height }));
        for (let i = 0; i < totalFrames; i++) {
            video.currentTime = i / EXTRACT_FPS;
            await new Promise(r => { video.onseeked = r; });
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            frames.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
            previewFirstFrame();
            setRunLabel(`Extracting frames (${i+1}/${totalFrames})...`);
            dbgProgress("extract", "Extracting frames", i + 1, totalFrames);
        }
        dbgProgressDone("extract");
        status(T("log.extraction_done", { count: frames.length }));
    } else if (file.type === "image/gif") {
        if (!window.ImageDecoder) {
            alert("Your browser doesn't support GIF frame extraction (use Chrome or Edge).");
            if (runRow) runRow.hidden = true;
            if ($("canvas-wrap")) $("canvas-wrap").hidden = true;
            resetUI();
            return;
        }
        const decoder = new ImageDecoder({ type: "image/gif", data: file.stream() });
        await decoder.tracks.ready;
        // tracks.ready only means the track metadata arrived: frameCount keeps
        // growing while the stream is parsed, so reading it here saw whatever
        // had landed so far (3 of 30 on the sample clip).
        await decoder.completed;
        const track = decoder.tracks.selectedTrack;
        const frameCount = Math.min(track.frameCount, 30);
        status(T("log.extract_gif", { count: frameCount }));
        for (let i = 0; i < frameCount; i++) {
            const result = await decoder.decode({ frameIndex: i });
            const vf = result.image;
            const canvas = document.createElement("canvas");
            const scale = Math.min(1.0, MAX_PREDICTION_SIDE / Math.max(vf.displayWidth, vf.displayHeight));
            canvas.width = Math.round(vf.displayWidth * scale);
            canvas.height = Math.round(vf.displayHeight * scale);
            const ctx = canvas.getContext("2d", { willReadFrequently: true });
            ctx.drawImage(vf, 0, 0, canvas.width, canvas.height);
            frames.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
            vf.close();
            previewFirstFrame();
            setRunLabel(`Extracting frames (${i+1}/${frameCount})...`);
            dbgProgress("extract", "Extracting frames", i + 1, frameCount);
        }
        dbgProgressDone("extract");
        status(T("log.extraction_done", { count: frames.length }));
    }

    if (frames.length === 0) { if (runRow) runRow.hidden = true; if ($("canvas-wrap")) $("canvas-wrap").hidden = true; resetUI(); return; }

    // Inference on keyframes (stride = EXTRACT_FPS / INFERENCE_FPS)
    const stride = Math.max(1, Math.round(EXTRACT_FPS / INFERENCE_FPS));
    for (let i = 0; i < frames.length; i += stride) keyIndices.push(i);
    if (keyIndices[keyIndices.length - 1] !== frames.length - 1) keyIndices.push(frames.length - 1);

    // Every card seen over the run, each with its best-scoring crop.
    const bestByCard = new Map();
    status(T("log.running_detection", { count: keyIndices.length }));
    for (let k = 0; k < keyIndices.length; k++) {
        if (cancelRequested) break;
        const idx = keyIndices[k];
        setRunLabel(`Analyzing key frame ${k+1}/${keyIndices.length}...`);
        const { detections, allPredictions, croppedImages } = await detectAndClassify(frames[idx], { showSteps: false });
        keyResults.set(idx, { detections, allPredictions });
        allPredictions.forEach((list, j) => {
            const top = list?.[0];
            if (!top || !croppedImages[j]) return;
            const prev = bestByCard.get(top.i);
            if (!prev || top.p > prev.preds[0].p) bestByCard.set(top.i, { crop: croppedImages[j], preds: list });
        });
        dbgProgress("infer", "Running detection", k + 1, keyIndices.length);
        await nextFrame();
    }

    const confirmed = confirmedCards(keyResults);
    denoiseKeyResults(keyResults, keyIndices, confirmed);
    const seenCards = [...bestByCard]
        .filter(([i]) => confirmed.has(i))
        .sort((a, b) => b[1].preds[0].p - a[1].preds[0].p);
    if (seenCards.length) {
        const grid = $("cards-grid");
        if (grid) renderResultCards(grid, seenCards.map(e => e[1].crop), seenCards.map(e => e[1].preds));
        if ($("results")) $("results").hidden = false;
    }
    dbgProgressDone("infer");
    if (cancelRequested) { cancelCleanup(); return; }
    status(T("log.keyframe_done"));

    // Encoded at the aspect the clip is previewed at.
    const boxAspect = getResultAspect();
    const srcAspect = frames[0].width / frames[0].height;
    const boxW = srcAspect > boxAspect ? frames[0].width  : Math.round(frames[0].height * boxAspect);
    const boxH = srcAspect > boxAspect ? Math.round(frames[0].width / boxAspect) : frames[0].height;
    const outScale = Math.min(1.0, GIF_OUTPUT_SIDE / Math.max(boxW, boxH));
    gifW = Math.round(boxW * outScale);
    gifH = Math.round(boxH * outScale);
    clipCount = frames.length;
    canvasOut = $("canvas-out");

    } finally {
        window.__bgAnim?.resume();
        if ($("canvas-trail")) $("canvas-trail").hidden = true;
    }

    if (!canvasOut) return;
    document.getElementById("gif-dl")?.remove();
    canvasOut.style.display = "";
    const player = startClipPlayer(canvasOut, playbackVideo || frames, keyResults, nearestKey);
    frames.length = 0;

    setRunLabel("Engine Ready");
    if (runRow) runRow.hidden = true;

    const DL_CLASS = "absolute bottom-4 right-4 px-4 py-2 rounded-lg font-bold text-xs shadow-lg z-50 text-white";
    const btn = document.createElement("button");
    btn.id = "gif-dl";
    btn.className = DL_CLASS + " bg-emerald-500 hover:bg-emerald-400";
    btn.textContent = T("runtime.export_gif");
    btn.addEventListener("click", () => exportClipGif(btn), { once: true });
    canvasOut.parentNode.style.position = "relative";
    canvasOut.parentNode.appendChild(btn);

    function seekTo(video, t) {
        return new Promise(r => { video.onseeked = () => r(); video.currentTime = t; });
    }

    // Nothing is encoded until asked: the encode is the longest step.
    async function exportClipGif(button) {
        button.disabled = true;
        button.className = DL_CLASS + " bg-zinc-500 opacity-70 cursor-default";
        const label = pct => { button.textContent = T("runtime.encoding_gif") + " " + pct + "%"; };
        label(0);
        player.setPaused(true);

        const workerBlob = new Blob(
            [`importScripts("https://cdn.jsdelivr.net/npm/gif.js@0.2.0/dist/gif.worker.js");`],
            { type: "application/javascript" });
        const encoder = new GIF({
            workers: 2,
            quality: 10,
            workerScript: URL.createObjectURL(workerBlob),
            width: gifW,
            height: gifH
        });

        const out = document.createElement("canvas");
        out.width = gifW;
        out.height = gifH;
        const outCtx = out.getContext("2d");
        const scratch = document.createElement("canvas");

        status(T("log.rendering_frames", { count: clipCount }));
        for (let i = 0; i < clipCount; i++) {
            if (cancelRequested) {
                player.setPaused(false);
                button.disabled = false;
                button.className = DL_CLASS + " bg-emerald-500 hover:bg-emerald-400";
                button.textContent = T("runtime.export_gif");
                button.addEventListener("click", () => exportClipGif(button), { once: true });
                return;
            }
            const r = keyResults.get(nearestKey(i)) || { detections: [], allPredictions: [] };
            let frame = playbackVideo;
            if (player.bitmaps) frame = player.bitmaps[i];
            else await seekTo(playbackVideo, i / EXTRACT_FPS);
            drawOverlayLetterboxed(scratch, frame, r.detections, r.allPredictions);
            outCtx.fillStyle = "#000";
            outCtx.fillRect(0, 0, gifW, gifH);
            outCtx.drawImage(scratch, 0, 0, gifW, gifH);
            encoder.addFrame(out, { delay: Math.round(1000 / EXTRACT_FPS), copy: true });
            label(Math.round(((i + 1) / clipCount) * 50));
            dbgProgress("render", "Rendering frames", i + 1, clipCount);
            if (i % 5 === 0) await nextFrame();
        }
        dbgProgressDone("render");
        status(T("log.encoding_gif"));

        encoder.on("progress", p => label(50 + Math.round(p * 50)));
        encoder.on("finished", blob => {
            status(T("log.gif_ready", { size: (blob.size / 1024).toFixed(0) }));
            player.setPaused(false);
            if (!button.isConnected) return;
            const dl = document.createElement("a");
            dl.id = "gif-dl";
            dl.href = URL.createObjectURL(blob);
            dl.download = "draw2_prediction.gif";
            dl.className = DL_CLASS + " bg-emerald-500 hover:bg-emerald-400";
            dl.textContent = T("runtime.download_gif");
            button.replaceWith(dl);
        });
        encoder.render();
    }
}

