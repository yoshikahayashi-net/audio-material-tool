from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from faster_whisper import BatchedInferencePipeline, WhisperModel

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("FW_PORT", "7860"))
MODEL_NAME = os.environ.get("FW_MODEL", "turbo")

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

_model = None
_batched_model = None
_device = None
_compute_type = None

def load_model():
    global _model, _batched_model, _device, _compute_type
    if _model is not None:
        return _model, _batched_model, _device, _compute_type

    # Prefer GPU. If the local CUDA/cuDNN stack is unavailable,
    # fall back to a bounded-thread CPU mode so the desktop stays usable.
    try:
        _model = WhisperModel(
            MODEL_NAME,
            device="cuda",
            compute_type="float16",
            cpu_threads=4,
        )
        _device = "GPU"
        _compute_type = "float16"
    except Exception as gpu_error:
        print(f"[faster-whisper] GPU unavailable: {gpu_error}")
        _model = WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1,
        )
        _device = "CPU"
        _compute_type = "int8"

    _batched_model = BatchedInferencePipeline(model=_model)
    return _model, _batched_model, _device, _compute_type


def normalize(text: str) -> str:
    text = " ".join((text or "").split())
    for p in "、。！？":
        text = text.replace(f" {p}", p)
    return text.strip()


def parse_keywords(raw: str) -> list[str]:
    values = []
    for chunk in (raw or "").replace("，", ",").replace("、", ",").split(","):
        item = chunk.strip()
        if item:
            values.append(item)
    return sorted(set(values), key=len, reverse=True)


def classify_segments(segments, finish_keys, shot_keys):
    current = "unknown"
    groups = {"finish": [], "shot": [], "unknown": []}

    for seg in segments:
        text = seg["text"]
        f = next((k for k in finish_keys if k in text), None)
        s = next((k for k in shot_keys if k in text), None)

        if f and not s:
            current = "finish"
        elif s and not f:
            current = "shot"
        elif f and s:
            current = "finish" if text.index(f) <= text.index(s) else "shot"

        groups[current].append(seg)

    return groups


def format_segment(seg):
    return f"[{int(seg['start']//60):02d}:{int(seg['start']%60):02d}] {seg['text']}"


def build_output(groups, raw, elapsed, audio_duration, device, compute_type, batch_size):
    speed = audio_duration / elapsed if elapsed > 0 else 0.0
    all_lines = [
        "【仕上】",
        *(f"・{format_segment(x)}" for x in groups["finish"]),
        "【ショット】",
        *(f"・{format_segment(x)}" for x in groups["shot"]),
        "【未分類】",
        *(f"・{format_segment(x)}" for x in groups["unknown"]),
        "【文字起こし全文】",
        raw,
        "",
        "【処理情報】",
        f"エンジン：faster-whisper / {MODEL_NAME}",
        f"実行：{device} / {compute_type}",
        f"Batch：{batch_size}",
        f"音声長：{audio_duration:.1f}秒",
        f"文字起こし：{elapsed:.1f}秒",
        f"速度：約{speed:.2f}倍速",
    ]
    return "\n".join(all_lines), speed


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "model": MODEL_NAME,
        "loaded": _model is not None,
        "device": _device,
        "compute_type": _compute_type,
    })


@app.post("/api/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"ok": False, "error": "音声ファイルがありません。"}), 400

    try:
        batch_size = max(1, min(8, int(request.form.get("batch_size", "4"))))
    except ValueError:
        batch_size = 4

    finish_keys = parse_keywords(request.form.get("finish_keywords", "仕上,仕上げ,仕上工程"))
    shot_keys = parse_keywords(request.form.get("shot_keywords", "ショット,ショット工程"))

    suffix = Path(audio.filename).suffix or ".bin"
    temp_path = Path(tempfile.gettempdir()) / f"fw_{uuid.uuid4().hex}{suffix}"

    try:
        audio.save(temp_path)
        _, batched_model, device, compute_type = load_model()

        # Keep CPU mode lightweight. Batching is reserved for GPU runs.
        if device == "CPU":
            batch_size = 1

        started = time.perf_counter()
        segments, info = batched_model.transcribe(
            str(temp_path),
            batch_size=batch_size,
            language="ja",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        materialized = []
        for seg in segments:
            text = normalize(seg.text)
            if text:
                materialized.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                })

        elapsed = time.perf_counter() - started
        raw = normalize(" ".join(x["text"] for x in materialized))
        groups = classify_segments(materialized, finish_keys, shot_keys)
        output, speed = build_output(
            groups,
            raw,
            elapsed,
            float(info.duration),
            device,
            compute_type,
            batch_size,
        )

        return jsonify({
            "ok": True,
            "model": MODEL_NAME,
            "device": device,
            "compute_type": compute_type,
            "language": info.language,
            "language_probability": float(info.language_probability or 0),
            "audio_duration": float(info.duration),
            "elapsed": elapsed,
            "speed": speed,
            "batch_size": batch_size,
            "segments": materialized,
            "groups": groups,
            "raw": raw,
            "output": output,
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }), 500
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("")
    print("=== faster-whisper comparison tool ===")
    print(f"Open: http://{HOST}:{PORT}/")
    print("The browser UI stays separate from the transcription worker.")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
