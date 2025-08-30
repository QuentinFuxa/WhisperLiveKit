<h1 align="center">WhisperLiveKit</h1>

<p align="center">
<img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/demo.png" alt="WhisperLiveKit Demo" width="730">
</p>

<p align="center"><b>話者識別機能付き、リアルタイム、完全ローカルな音声テキスト変換</b></p>

<p align="center">
<a href="https://pypi.org/project/whisperlivekit/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/whisperlivekit?color=g"></a>
<a href="https://pepy.tech/project/whisperlivekit"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/whisperlivekit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=installations"></a>
<a href="https://pypi.org/project/whisperlivekit/"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.9--3.13-dark_green"></a>
<a href="https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT/Dual Licensed-dark_green"></a>
</p>

すぐに使えるバックエンド+サーバーとシンプルなフロントエンドで、リアルタイムの音声文字起こしをブラウザに直接提供します。✨

#### 主要な研究による技術：

- [SimulStreaming](https://github.com/ufal/SimulStreaming) (SOTA 2025) - AlignAttポリシーによる超低遅延文字起こし
- [WhisperStreaming](https://github.com/ufal/whisper_streaming) (SOTA 2023) - LocalAgreementポリシーによる低遅延文字起こし
- [Streaming Sortformer](https://arxiv.org/abs/2507.18446) (SOTA 2025) - 高度なリアルタイム話者ダイアライゼーション
- [Diart](https://github.com/juanmc2005/diart) (SOTA 2021) - リアルタイム話者ダイアライゼーション
- [Silero VAD](https://github.com/snakers4/silero-vad) (2024) - エンタープライズグレードの音声区間検出

> **なぜ各音声バッチで単純なWhisperモデルを実行しないのか？** Whisperは完全な発話向けに設計されており、リアルタイムのチャンク向けではありません。小さなセグメントを処理するとコンテキストが失われ、単語が音節の途中で途切れ、質の悪い文字起こしになります。WhisperLiveKitは、インテリジェントなバッファリングとインクリメンタルな処理のために、最先端の同時音声研究を利用しています。

### アーキテクチャ

<img alt="Architecture" src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/architecture.png" />

*バックエンドは複数の同時ユーザーをサポートします。音声が検出されない場合、音声区間検出がオーバーヘッドを削減します。*

### インストールとクイックスタート

```bash
pip install whisperlivekit
```

>  **FFmpegが必要です** WhisperLiveKitを使用する前にインストールする必要があります。
>
> | OS | インストール方法 |
> |-----------|-------------|
>  | Ubuntu/Debian | `sudo apt install ffmpeg` |
> | MacOS | `brew install ffmpeg` |
> | Windows | https://ffmpeg.org/download.html から.exeをダウンロードし、PATHに追加 |

#### クイックスタート
1. **文字起こしサーバーを起動します:**
   ```bash
   whisperlivekit-server --model base --language en
   ```

2. **ブラウザを開き** `http://localhost:8000` にアクセスします。話し始めると、あなたの言葉がリアルタイムで表示されます！


> - 利用可能なすべての言語のリストについては、[tokenizer.py](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/simul_whisper/whisper/tokenizer.py) を参照してください。
> - HTTPSの要件については、**パラメータ**セクションのSSL設定オプションを参照してください。

#### オプションの依存関係

| オプション | `pip install` |
|-----------|-------------|
| **Sortformerによる話者ダイアライゼーション** | `git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]` |
| Diartによる話者ダイアライゼーション | `diart` |
| オリジナルのWhisperバックエンド | `whisper` |
| タイムスタンプ改善バックエンド | `whisper-timestamped` |
| Apple Silicon最適化バックエンド | `mlx-whisper` |
| OpenAI APIバックエンド | `openai` |

それらの使用方法については、以下の**パラメータと設定**を参照してください。

### 使用例

**コマンドラインインターフェース**: 様々なオプションで文字起こしサーバーを起動します:

```bash
# デフォルト(small)より良いモデルを使用
whisperlivekit-server --model large-v3

# ダイアライゼーションと言語を指定した高度な設定
whisperlivekit-server --host 0.0.0.0 --port 8000 --model medium --diarization --language fr
```

**Python API連携**: 関数やクラスの使用方法のより完全な例については、[basic_server](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/basic_server.py) を確認してください。

```python
from whisperlivekit import TranscriptionEngine, AudioProcessor, parse_args
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import asyncio

transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(model="medium", diarization=True, lan="en")
    yield

app = FastAPI(lifespan=lifespan)

async def handle_websocket_results(websocket: WebSocket, results_generator):
    async for response in results_generator:
        await websocket.send_json(response)
    await websocket.send_json({"type": "ready_to_stop"})

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine

    # 接続ごとに新しいAudioProcessorを作成し、共有エンジンを渡す
    audio_processor = AudioProcessor(transcription_engine=transcription_engine)
    results_generator = await audio_processor.create_tasks()
    results_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))
    await websocket.accept()
    while True:
        message = await websocket.receive_bytes()
        await audio_processor.process_audio(message)
```

**フロントエンド実装**: パッケージにはHTML/JavaScript実装が[ここ](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/web/live_transcription.html)に含まれています。`from whisperlivekit import get_web_interface_html` & `page = get_web_interface_html()` を使ってインポートすることもできます。


## パラメータと設定

重要なパラメータのリストを変更できます。しかし、何を*変更すべき*でしょうか？
- `--model` サイズ。リストと推奨事項は[こちら](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/available_models.md)
- `--language`。リストは[こちら](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/simul_whisper/whisper/tokenizer.py)。`auto`を使用すると、モデルは自動的に言語を検出しようとしますが、英語に偏る傾向があります。
- `--backend`？ `simulstreaming`が正しく動作しない場合や、デュアルライセンス要件を避けたい場合は`--backend faster-whisper`に切り替えることができます。
- `--warmup-file`、もしあれば
- `--host`, `--port`, `--ssl-certfile`, `--ssl-keyfile`、サーバーをセットアップする場合
- `--diarization`、使用したい場合。

残りは推奨しません。しかし、以下があなたのオプションです。

| パラメータ | 説明 | デフォルト |
|-----------|-------------|---------|
| `--model` | Whisperモデルのサイズ。 | `small` |
| `--language` | ソース言語コードまたは`auto` | `auto` |
| `--task` | `transcribe`または`translate` | `transcribe` |
| `--backend` | 処理バックエンド | `simulstreaming` |
| `--min-chunk-size` | 最小音声チャンクサイズ（秒） | `1.0` |
| `--no-vac` | 音声アクティビティコントローラーを無効化 | `False` |
| `--no-vad` | 音声区間検出を無効化 | `False` |
| `--warmup-file` | モデルのウォームアップ用音声ファイルパス | `jfk.wav` |
| `--host` | サーバーホストアドレス | `localhost` |
| `--port` | サーバーポート | `8000` |
| `--ssl-certfile` | SSL証明書ファイルへのパス（HTTPSサポート用） | `None` |
| `--ssl-keyfile` | SSL秘密鍵ファイルへのパス（HTTPSサポート用） | `None` |


| WhisperStreamingバックエンドオプション | 説明 | デフォルト |
|-----------|-------------|---------|
| `--confidence-validation` | 高速な検証のために信頼スコアを使用 | `False` |
| `--buffer_trimming` | バッファトリミング戦略（`sentence`または`segment`） | `segment` |


| SimulStreamingバックエンドオプション | 説明 | デフォルト |
|-----------|-------------|---------|
| `--frame-threshold` | AlignAttフレームしきい値（低いほど速く、高いほど正確） | `25` |
| `--beams` | ビームサーチのビーム数（1 = 貪欲デコーディング） | `1` |
| `--decoder` | デコーダタイプを強制（`beam`または`greedy`） | `auto` |
| `--audio-max-len` | 最大音声バッファ長（秒） | `30.0` |
| `--audio-min-len` | 処理する最小音声長（秒） | `0.0` |
| `--cif-ckpt-path` | 単語境界検出用CIFモデルへのパス | `None` |
| `--never-fire` | 未完了の単語を決して切り捨てない | `False` |
| `--init-prompt` | モデルの初期プロンプト | `None` |
| `--static-init-prompt` | スクロールしない静的プロンプト | `None` |
| `--max-context-tokens` | 最大コンテキストトークン数 | `None` |
| `--model-path` | .ptモデルファイルへの直接パス。見つからない場合はダウンロード | `./base.pt` |
| `--preloaded-model-count` | オプション。メモリにプリロードするモデルの数（予想される同時ユーザー数まで設定） | `1` |

| ダイアライゼーションオプション | 説明 | デフォルト |
|-----------|-------------|---------|
| `--diarization` | 話者識別を有効化 | `False` |
| `--diarization-backend` | `diart`または`sortformer` | `sortformer` |
| `--segmentation-model` | DiartセグメンテーションモデルのHugging FaceモデルID。[利用可能なモデル](https://github.com/juanmc2005/diart/tree/main?tab=readme-ov-file#pre-trained-models) | `pyannote/segmentation-3.0` |
| `--embedding-model` | Diart埋め込みモデルのHugging FaceモデルID。[利用可能なモデル](https://github.com/juanmc2005/diart/tree/main?tab=readme-ov-file#pre-trained-models) | `speechbrain/spkrec-ecapa-voxceleb` |


> Diartを使用したダイアライゼーションには、pyannote.audioモデルへのアクセスが必要です：
> 1. `pyannote/segmentation`モデルの[ユーザー条件に同意](https://huggingface.co/pyannote/segmentation)
> 2. `pyannote/segmentation-3.0`モデルの[ユーザー条件に同意](https://huggingface.co/pyannote/segmentation-3.0)
> 3. `pyannote/embedding`モデルの[ユーザー条件に同意](https://huggingface.co/pyannote/embedding)
>4. HuggingFaceでログイン: `huggingface-cli login`

### 🚀 デプロイガイド

WhisperLiveKitを本番環境にデプロイするには：

1. **サーバーセットアップ**: 本番用ASGIサーバーをインストールし、複数のワーカーで起動します
   ```bash
   pip install uvicorn gunicorn
   gunicorn -k uvicorn.workers.UvicornWorker -w 4 your_app:app
   ```

2. **フロントエンド**: カスタマイズした`html`のバージョンをホストし、WebSocket接続が正しくポイントするようにします

3. **Nginx設定** (本番環境で推奨):
    ```nginx
   server {
       listen 80;
       server_name your-domain.com;
        location / {
            proxy_pass http://localhost:8000;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
    }}
    ```

4. **HTTPSサポート**: 安全なデプロイメントのために、WebSocket URLで "ws://" の代わりに "wss://" を使用します

## 🐋 Docker

GPUまたはCPUサポート付きでDockerを使用してアプリケーションを簡単にデプロイします。

### 前提条件
- Dockerがシステムにインストールされていること
- GPUサポートの場合: NVIDIA Dockerランタイムがインストールされていること

### クイックスタート

**GPUアクセラレーション付き (推奨):**
```bash
docker build -t wlk .
docker run --gpus all -p 8000:8000 --name wlk wlk
```

**CPUのみ:**
```bash
docker build -f Dockerfile.cpu -t wlk .
docker run -p 8000:8000 --name wlk wlk
```

### 高度な使用法

**カスタム設定:**
```bash
# カスタムモデルと言語の例
docker run --gpus all -p 8000:8000 --name wlk wlk --model large-v3 --language fr
```

### メモリ要件
- **大規模モデル**: Dockerランタイムに十分なメモリが割り当てられていることを確認してください


#### カスタマイズ

- `--build-arg` オプション:
  - `EXTRAS="whisper-timestamped"` - イメージのインストールにエクストラを追加します（スペースなし）。必要なコンテナオプションを設定することを忘れないでください！
  - `HF_PRECACHE_DIR="./.cache/"` - 初回起動を高速化するためにモデルキャッシュをプリロードします
  - `HF_TKN_FILE="./token"` - ゲート付きモデルをダウンロードするためにHugging Face Hubアクセストークンを追加します

## 🔮 ユースケース
会議の文字起こしのためにリアルタイムで議論をキャプチャする、聴覚障害のあるユーザーがアクセシビリティツールを通じて会話を追うのを助ける、コンテンツ作成のためにポッドキャストやビデオを自動的に文字起こしする、カスタマーサービスのために話者識別付きでサポートコールを文字起こしする...
