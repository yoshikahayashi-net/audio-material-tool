import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

const MODEL = "onnx-community/whisper-large-v3-turbo";
let transcriber = null;

function send(type, payload) {
  self.postMessage(Object.assign({ type }, payload || {}));
}

function normalize(s) {
  return String(s || "").replace(/\s+/g, " ").replace(/\s*([、。！？])\s*/g, "$1").trim();
}

async function getTranscriber() {
  if (transcriber) return transcriber;

  const webgpu = !!(self.navigator && self.navigator.gpu);
  const make = async (device) => {
    return await pipeline("automatic-speech-recognition", MODEL, {
      device,
      dtype: device === "webgpu" ? "q4f16" : "q8",
      progress_callback: function(p) {
        if (!p) return;
        if (p.status === "progress" && typeof p.progress === "number") {
          send("model-progress", { progress: p.progress, device });
        } else if (p.status === "ready") {
          send("model-ready", { device });
        }
      }
    });
  };

  try {
    transcriber = await make(webgpu ? "webgpu" : "wasm");
    return transcriber;
  } catch (e) {
    if (webgpu) {
      send("model-fallback", { message: "GPUで起動できなかったため、CPUモードに切り替えます…" });
      transcriber = await make("wasm");
      return transcriber;
    }
    throw e;
  }
}

self.onmessage = async function(event) {
  const data = event.data || {};
  if (data.type !== "transcribe") return;

  try {
    send("status", { message: "① 音声認識モデルを準備しています。初回だけ時間がかかります。" });
    const pipe = await getTranscriber();

    const sampleRate = data.sampleRate || 16000;
    const audio = new Float32Array(data.audio);
    const chunkSeconds = 30;
    const overlapSeconds = 2;
    const chunkSamples = chunkSeconds * sampleRate;
    const overlapSamples = overlapSeconds * sampleRate;
    const stepSamples = chunkSamples - overlapSamples;
    const totalChunks = Math.max(
      1,
      Math.ceil(Math.max(0, audio.length - overlapSamples) / stepSamples)
    );

    const segments = [];
    const BATCH_SIZE = 1;

    for (let batchStart = 0; batchStart < totalChunks; batchStart += BATCH_SIZE) {
      const inputs = [];
      const meta = [];
      const batchEnd = Math.min(totalChunks, batchStart + BATCH_SIZE);

      for (let i = batchStart; i < batchEnd; i++) {
        const startSample = i * stepSamples;
        const endSample = Math.min(audio.length, startSample + chunkSamples);
        inputs.push(audio.slice(startSample, endSample));
        meta.push({
          index: i,
          start: startSample / sampleRate,
          end: endSample / sampleRate
        });
      }

      send("batch-start", {
        done: batchStart,
        total: totalChunks,
        batchEnd
      });

      let parts;
      try {
        parts = await pipe(inputs, {
          return_timestamps: false,
          language: "japanese",
          task: "transcribe"
        });
      } catch (batchError) {
        send("batch-fallback", {
          message: "GPUの同時処理に対応できないため、1区間ずつ処理します。"
        });
        parts = [];
        for (const input of inputs) {
          parts.push(await pipe(input, {
            return_timestamps: false,
            language: "japanese",
            task: "transcribe"
          }));
        }
      }

      const results = Array.isArray(parts) ? parts : [parts];

      for (let j = 0; j < meta.length; j++) {
        const text = normalize(results[j] && results[j].text ? results[j].text : "");
        if (text) {
          segments.push({
            start: meta[j].start,
            end: meta[j].end,
            text
          });
        }
      }

      send("batch-done", {
        done: batchEnd,
        total: totalChunks
      });
    }

    send("complete", { segments });
  } catch (error) {
    let detail = "";
    if (error && error.stack) detail = error.stack;
    else if (typeof error === "number") detail = "数値エラーコード: " + error;
    else if (error && error.message) detail = error.message;
    else detail = String(error);
    send("error", {
      message: "音声認識モデルの処理に失敗しました。 " + detail
    });
  }
};
