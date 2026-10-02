import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

const MODEL = "onnx-community/kotoba-whisper-v2.2-ONNX";
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
      dtype: device === "webgpu" ? { encoder_model: "fp16", decoder_model_merged: "q4f16" } : "q8",
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
    const useWebGPU = !!(self.navigator && self.navigator.gpu);
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
    let batchSize = useWebGPU ? 2 : 1;

    // 日本語音声での言語判定を毎区間やり直さないよう固定する。
    const inferenceOptions = {
      return_timestamps: false,
      language: "ja",
      task: "transcribe"
    };

    // 無音に近い区間は推論そのものをスキップする。
    // 閾値はかなり保守的にして、通常の小さな声を落としにくくする。
    function isSilence(buffer) {
      let sumSq = 0;
      let peak = 0;
      const stride = Math.max(1, Math.floor(buffer.length / 4096));
      let count = 0;

      for (let i = 0; i < buffer.length; i += stride) {
        const v = buffer[i];
        const a = Math.abs(v);
        if (a > peak) peak = a;
        sumSq += v * v;
        count++;
      }

      const rms = Math.sqrt(sumSq / Math.max(1, count));
      return rms < 0.001 && peak < 0.01;
    }

    for (let batchStart = 0; batchStart < totalChunks; batchStart += batchSize) {
      const inputs = [];
      const meta = [];
      const batchEnd = Math.min(totalChunks, batchStart + batchSize);
      let skipped = 0;

      for (let i = batchStart; i < batchEnd; i++) {
        const startSample = i * stepSamples;
        const endSample = Math.min(audio.length, startSample + chunkSamples);
        const chunk = audio.slice(startSample, endSample);

        if (isSilence(chunk)) {
          skipped++;
          continue;
        }

        inputs.push(chunk);
        meta.push({
          index: i,
          start: startSample / sampleRate,
          end: endSample / sampleRate
        });
      }

      send("batch-start", {
        done: batchStart,
        total: totalChunks,
        batchEnd,
        batchSize,
        skipped,
        actualCount: inputs.length
      });

      let parts = [];
      if (inputs.length > 0) {
        try {
          parts = await pipe(inputs, inferenceOptions);
        } catch (batchError) {
          // 一度でもバッチ実行に失敗した環境では、それ以降のバッチも
          // 毎回例外を起こしてからフォールバックする必要はない。
          if (batchSize > 1) {
            batchSize = 1;
            send("batch-fallback", {
              message: "GPUの同時処理に対応できなかったため、以降は1区間ずつ処理します。"
            });
          }
          parts = [];
          for (const input of inputs) {
            parts.push(await pipe(input, inferenceOptions));
          }
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
        total: totalChunks,
        skipped,
        batchSize,
        actualCount: inputs.length
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
