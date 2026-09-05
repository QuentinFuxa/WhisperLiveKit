# Browser tab capture

The extension captures tab audio and sends it to a WhisperLiveKit server. It shares the browser UI maintained in `whisperlivekit/web/`.

## Run

1. Start WLK, for example `wlk --model base --language en --pcm-input`.
2. From the repository root, run `python scripts/sync_extension.py`. This copies the page, styles, scripts, icons, PCM worklet, and recorder worker into the extension directory.
3. Enable developer mode at `chrome://extensions` and load `chrome-extension/` as an unpacked extension.
4. Open the extension on the tab to capture and set its WebSocket URL to `ws://localhost:8000/asr` (or your server's URL).

The UI tries tab capture first. If that fails, it requests microphone input and reports the selected source. Generated frontend files are ignored by Git; edit the originals in `whisperlivekit/web/` and rerun the sync script after changes.
