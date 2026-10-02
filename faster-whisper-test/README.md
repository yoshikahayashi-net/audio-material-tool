# faster-whisper 比較版

本番の GitHub Pages 版とは分離したローカル比較環境です。

## 使い方

Windows では `start.bat` をダブルクリック。

初回だけ Python 仮想環境と必要パッケージを作成し、その後ローカルサーバーを起動します。

ブラウザ：
`http://127.0.0.1:7860/`

## エンジン

- faster-whisper 1.2.1
- Whisper turbo
- 日本語固定（`ja`）
- beam size 5
- VAD有効
- BatchedInferencePipeline
- Batch 4を標準値
- GPUが使えなければCPU int8へ自動フォールバック
- CPUでは4スレッドに制限して、PC全体の負荷を抑える

公式の faster-whisper はCTranslate2ベースで、同一精度条件でOpenAI Whisperより最大4倍高速、8-bit量子化やバッチ推論にも対応しています。
GPU利用には現在のfaster-whisper/CTranslate2側のCUDA 12 + cuBLAS + cuDNN 9要件があります。

この比較版では、本番版のモデルやコードを変更しません。
