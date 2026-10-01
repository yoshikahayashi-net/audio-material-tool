let worker = null;
let outputs = { all:"", finish:"", shot:"", unknown:"", raw:"" };
let currentTab = "all";
let lastGroups = {finish:[], shot:[], unknown:[]};
let lastRaw = "";
let lastSegments = [];
let processingStartedAt = 0;

const $ = (id) => document.getElementById(id);
const fileInput = $("audioFile");
const runBtn = $("runBtn");
const statusEl = $("status");
const bar = $("bar");
const results = $("results");
const out = $("out");
let transcriptionTimer = null;

function stopTranscriptionProgress() {
  clearInterval(transcriptionTimer);
  transcriptionTimer = null;
}

function setChunkProgress(done, total, state) {
  const pct = total ? 30 + (done / total) * 62 : 30;
  setProgress(Math.min(92, pct));
  if (state === "running") {
    setStatus("③ 文字起こし中… " + done + " / " + total + " 区間完了｜" + Math.min(done + 1, total) + "区間目を処理中");
  } else {
    setStatus("③ 文字起こし中… " + done + " / " + total + " 区間完了");
  }
}


function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = "status" + (kind ? " " + kind : "");
}
function setProgress(v) {
  bar.classList.remove("processing");
  bar.style.width = Math.max(0, Math.min(100, v)) + "%";
}
function setProcessing(active) {
  if (active) {
    bar.style.width = "45%";
    bar.classList.add("processing");
  } else {
    bar.classList.remove("processing");
  }
}

function fmtBytes(n) {
  return n < 1024*1024 ? (n/1024).toFixed(0) + " KB" : (n/1024/1024).toFixed(1) + " MB";
}
function escapeHtml(s) {
  return String(s).replace(/[&<>'"]/g, function(c) {
    return {"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c];
  });
}

fileInput.addEventListener("change", function() {
  const f = fileInput.files && fileInput.files[0];
  runBtn.disabled = !f;
  if (f) {
    $("fileInfo").textContent = f.name + " ／ " + fmtBytes(f.size);
    setStatus("準備OK。資料化を押してください。");
  } else {
    $("fileInfo").textContent = "まだ音声ファイルが選択されていません。";
    setStatus("待機中です。");
  }
});

function getWorker() {
  if (worker) return worker;
  worker = new Worker("./audio-worker.js?v=20261001-17", { type: "module" });
  return worker;
}

function transcribeWithWorker(audio) {
  return new Promise(function(resolve, reject) {
    const w = getWorker();
    const segments = [];
    let settled = false;

    function cleanup() {
      w.removeEventListener("message", onMessage);
      w.removeEventListener("error", onError);
    }

    function onError(event) {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(event && event.message ? event.message : "音声認識Workerでエラーが発生しました。"));
    }

    function onMessage(event) {
      const data = event.data || {};

      if (data.type === "model-progress") {
        setProgress(Math.min(28, data.progress * 0.28));
        setStatus("① モデルを準備中… " + Math.round(data.progress) + "%");
        return;
      }

      if (data.type === "model-ready") {
        setProgress(28);
        setStatus("① モデル準備完了（" + (data.device === "webgpu" ? "GPU" : "CPU") + "）");
        return;
      }

      if (data.type === "model-fallback") {
        setStatus(data.message || "CPUモードに切り替えます…");
        return;
      }

      if (data.type === "status") {
        setStatus(data.message || "処理中…");
        return;
      }

      if (data.type === "batch-start") {
        const nextStart = Math.min(data.done + 1, data.total);
        const nextEnd = Math.min(data.batchEnd, data.total);
        setProgress(Math.min(92, 30 + (data.done / data.total) * 62));
        setStatus("③ 文字起こし中… " + data.done + " / " + data.total + " 区間完了｜" +
          nextStart + "〜" + nextEnd + "区間を同時処理中");
        return;
      }

      if (data.type === "batch-fallback") {
        setStatus(data.message || "1区間ずつ処理しています…");
        return;
      }

      if (data.type === "batch-done") {
        setChunkProgress(data.done, data.total, "done");
        return;
      }

      if (data.type === "complete") {
        settled = true;
        cleanup();
        resolve(data.segments || []);
        return;
      }

      if (data.type === "error") {
        settled = true;
        cleanup();
        reject(new Error(data.message || "音声認識に失敗しました。"));
      }
    }

    w.addEventListener("message", onMessage);
    w.addEventListener("error", onError);

    try {
      // Transfer the underlying audio buffer so the main thread does not
      // keep copying a large Float32Array while the worker runs inference.
      const transferable = audio.mono.buffer;
      w.postMessage({
        type: "transcribe",
        audio: transferable,
        sampleRate: 16000,
        duration: audio.duration
      }, [transferable]);
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

async function decodeTo16k(file) {
  setStatus("② 音声を読み込んでいます…");
  const raw = await file.arrayBuffer();
  const sourceCtx = new AudioContext();
  const audio = await sourceCtx.decodeAudioData(raw);
  await sourceCtx.close();

  const targetRate = 16000;
  const targetLength = Math.max(1, Math.ceil(audio.duration * targetRate));
  const offline = new OfflineAudioContext(1, targetLength, targetRate);
  const buffer = offline.createBuffer(audio.numberOfChannels, audio.length, audio.sampleRate);

  for (let c = 0; c < audio.numberOfChannels; c++) {
    buffer.copyToChannel(audio.getChannelData(c), c);
  }

  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start(0);

  const rendered = await offline.startRendering();
  return {
    mono: rendered.getChannelData(0),
    duration: audio.duration
  };
}

function mergeChunkText(previous, current) {
  const a = normalize(previous);
  const b = normalize(current);
  if (!a) return b;
  if (!b) return a;

  // Find the longest exact overlap between the end of the previous chunk
  // and the beginning of the current chunk.
  const max = Math.min(80, a.length, b.length);
  for (let n = max; n >= 8; n--) {
    if (a.slice(-n) === b.slice(0, n)) {
      return a + b.slice(n);
    }
  }
  return a + " " + b;
}

function normalize(s) {
  return s.replace(/\s+/g, " ").replace(/\s*([、。！？])\s*/g, "$1").trim();
}
function splitSentences(text) {
  const t = normalize(text);
  const parts = t.split(/(?<=[。！？])\s*|\n+/).map(function(x){return x.trim();}).filter(Boolean);
  return parts.length ? parts : [t];
}
function parseKeywords(raw) {
  return raw.split(/[、,，\n]/).map(function(x){return x.trim();}).filter(Boolean).sort(function(a,b){return b.length-a.length;});
}
function formatTime(sec) {
  const total = Math.max(0, Math.floor(sec));
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return m + ":" + s;
}
function formatSegment(seg) {
  return "[" + formatTime(seg.start) + "] " + normalize(seg.text || "");
}
function classifySegments(segments, finishKeys, shotKeys) {
  let current = "unknown";
  const groups = {finish:[], shot:[], unknown:[]};

  for (const seg of segments) {
    const text = seg.text || "";
    const f = finishKeys.some(function(k){return text.includes(k);});
    const s = shotKeys.some(function(k){return text.includes(k);});

    if (f && !s) current = "finish";
    else if (s && !f) current = "shot";
    else if (f && s) {
      const fk = finishKeys.find(function(k){return text.includes(k);});
      const sk = shotKeys.find(function(k){return text.includes(k);});
      current = text.indexOf(fk) <= text.indexOf(sk) ? "finish" : "shot";
    }

    groups[current].push(seg);
  }
  return groups;
}
function joinLines(list, prefix) {
  return list.length
    ? list.map(function(x){return prefix + (typeof x === "string" ? x : formatSegment(x));}).join("\n")
    : prefix + "内容なし";
}
function buildText(groups, raw) {
  return "【仕上】\n" + joinLines(groups.finish, "・") +
    "\n\n【ショット】\n" + joinLines(groups.shot, "・") +
    "\n\n【未分類】\n" + joinLines(groups.unknown, "・") +
    "\n\n【文字起こし全文】\n" + raw;
}
function buildMarkdown(groups, raw) {
  return "# 音声資料化\n\n## 仕上\n" + joinLines(groups.finish, "- ") +
    "\n\n## ショット\n" + joinLines(groups.shot, "- ") +
    "\n\n## 未分類\n" + joinLines(groups.unknown, "- ") +
    "\n\n## 文字起こし全文\n\n" + raw;
}
function buildDoc(text) {
  const html = text.split("\n").map(function(x){return x ? "<p>" + escapeHtml(x) + "</p>" : "<br>";}).join("");
  return "<!doctype html><html><head><meta charset='utf-8'><title>音声資料</title></head><body>" + html + "</body></html>";
}
function download(name, content, type) {
  const blob = new Blob([content], {type:type});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function(){URL.revokeObjectURL(a.href);}, 1000);
}
function render() {
  out.textContent = outputs[currentTab] || "";
}

document.querySelectorAll(".tab").forEach(function(btn) {
  btn.addEventListener("click", function() {
    document.querySelectorAll(".tab").forEach(function(x){x.classList.remove("active");});
    btn.classList.add("active");
    currentTab = btn.dataset.tab;
    render();
  });
});

$("txtBtn").addEventListener("click", function(){download("audio_material.txt", outputs.all, "text/plain;charset=utf-8");});
$("mdBtn").addEventListener("click", function(){download("audio_material.md", buildMarkdown(lastGroups,lastRaw), "text/markdown;charset=utf-8");});
$("docBtn").addEventListener("click", function(){download("audio_material.doc", buildDoc(buildText(lastGroups,lastRaw)), "application/msword");});

runBtn.addEventListener("click", async function() {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  runBtn.disabled = true;
  results.style.display = "none";
  setProgress(0);
  processingStartedAt = performance.now();
  try {
    const audio = await decodeTo16k(file);
    const duration = audio.duration;
    const sampleRate = 16000;
    const chunkSeconds = 30;
    const overlapSeconds = 2;
    const totalChunks = Math.max(
      1,
      Math.ceil(Math.max(0, audio.mono.length - overlapSeconds * sampleRate) /
        ((chunkSeconds - overlapSeconds) * sampleRate))
    );

    setChunkProgress(0, totalChunks);
    setProcessing(true);

    const segments = await transcribeWithWorker(audio);

    let rawText = "";
    for (const seg of segments) {
      rawText = mergeChunkText(rawText, seg && seg.text ? seg.text : "");
    }

    const raw = normalize(rawText || "");
    const chars = raw.replace(/\s/g, "");
    const repeated = chars.length >= 80
      ? Math.max(...Array.from(new Set(chars)).map(function(ch){ return chars.split(ch).length - 1; })) / chars.length
      : 0;
    if (!raw || repeated > 0.65) {
      throw new Error("文字起こし結果が不自然です。音声を正しく認識できていない可能性があります。");
    }
    lastRaw = raw;
    lastSegments = segments;
    const groups = classifySegments(
      segments,
      parseKeywords($("kwFinish").value),
      parseKeywords($("kwShot").value)
    );
    lastGroups = groups;

    const totalSeconds = Math.max(0, Math.round((performance.now() - processingStartedAt) / 1000));
    const elapsedLabel = Math.floor(totalSeconds / 60) + "分 " + String(totalSeconds % 60).padStart(2, "0") + "秒";

    const timedRaw = segments.length
      ? segments.map(formatSegment).join("\n")
      : raw;

    outputs.raw = timedRaw;
    outputs.finish = groups.finish.length ? groups.finish.map(formatSegment).join("\n") : "（該当内容なし）";
    outputs.shot = groups.shot.length ? groups.shot.map(formatSegment).join("\n") : "（該当内容なし）";
    outputs.unknown = groups.unknown.length ? groups.unknown.map(formatSegment).join("\n") : "（該当内容なし）";
    outputs.all = buildText(groups, raw) +
      "\n\n【処理時間】\n" + elapsedLabel;
    render();
    results.style.display = "block";
    setProgress(100);
    setStatus("⑤ 完了。音声 " + totalChunks + "区間を資料化しました。処理時間 " + elapsedLabel + " ／ 仕上 " + groups.finish.length + "件 ／ ショット " + groups.shot.length + "件 ／ 未分類 " + groups.unknown.length + "件", "ok");
  } catch (err) {
        setProcessing(false);
    console.error(err);
    setStatus("処理に失敗しました：" + (err && err.message ? err.message : err), "error");
    setProgress(0);
  } finally {
        setProcessing(false);
    runBtn.disabled = false;
  }
});
