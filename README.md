# Whisper Streaming with FastAPI and WebSocket Integration

This project extends the [Whisper Streaming](https://github.com/ufal/whisper_streaming) implementation by incorporating few extras. The enhancements include:

1. **FastAPI Server with WebSocket Endpoint**: Enables real-time speech-to-text transcription directly from the browser.

2. **Buffering Indication**: Improves streaming display by showing the current processing status, providing users with immediate feedback.

3. **Javascript Client implementation**: Functionnal and minimalist MediaRecorder implementation that can be copied on your client side

4. **MLX Whisper backend**: Integrates the alternative backend option MLX Whisper, optimized for efficient speech recognition on Apple silicon.

![Demo Screenshot](src/demo.png)


## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/QuentinFuxa/whisper_streaming_web
   cd whisper_streaming_web
   ```


### How to Launch the Server

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```
2. Install a whisper backend among:

 ```
whisper
whisper-timestamped
faster-whisper (faster backend on NVIDIA GPU)
mlx-whisper (faster backend on Apple Silicon)

and torch if you want to use VAC (Voice Activity Controller)
```


3. **Run the FastAPI Server**:

    ```bash
    python whisper_fastapi_online_server.py --host 0.0.0.0 --port 8000
    ```

    - `--host` and `--port` let you specify the server’s IP/port.  

4. **Open the Provided HTML**:

    - By default, the server root endpoint `/` serves a simple `live_transcription.html` page.  
    - Open your browser at `http://localhost:8000` (or replace `localhost` and `8000` with whatever you specified).  
    - The page uses vanilla JavaScript and the WebSocket API to capture your microphone and stream audio to the server in real time.

### How the Live Interface Works

- Once you **allow microphone access**, the page records small chunks of audio using the **MediaRecorder** API in **webm/opus** format.  
- These chunks are sent over a **WebSocket** to the FastAPI endpoint at `/ws`.  
- The Python server decodes `.webm` chunks on the fly using **FFmpeg** and streams them into the **whisper streaming** implementation for transcription.  
- **Partial transcription** appears as soon as enough audio is processed. The “unvalidated” text is shown in **lighter or grey color** (i.e., an ‘aperçu’) to indicate it’s still buffered partial output. Once Whisper finalizes that segment, it’s displayed in normal text.  
- You can watch the transcription update in near real time, ideal for demos, prototyping, or quick debugging.

### Deploying to a Remote Server

If you want to **deploy** this setup:

1. **Host the FastAPI app** behind a production-grade HTTP(S) server (like **Uvicorn + Nginx** or Docker).  
2. The **HTML/JS page** can be served by the same FastAPI app or a separate static host.  
3. Users open the page in **Chrome/Firefox** (any modern browser that supports MediaRecorder + WebSocket).  

No additional front-end libraries or frameworks are required. The WebSocket logic in `live_transcription.html` is minimal enough to adapt for your own custom UI or embed in other pages.

## Acknowledgments

This project builds upon the foundational work of the Whisper Streaming project. We extend our gratitude to the original authors for their contributions.

