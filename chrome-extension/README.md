## WhisperLiveKit Chrome Extension v0.1.1
Capture the audio of your current tab, transcribe diarize and translate it using WhisperliveKit, in Chrome and other Chromium-based browsers.

> Currently, only the tab audio is captured; your microphone audio is not recorded.

<img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/chrome-extension/demo-extension.png" alt="WhisperLiveKit Demo" width="730">

## Running this extension
1. Run `python scripts/sync_extension.py` to copy frontend files to the `chrome-extension` directory.
2. Load the `chrome-extension` directory in Chrome as an unpacked extension.


## Devs:
- Impossible to capture audio from tabs if extension is a pannel, unfortunately: 
- https://issues.chromium.org/issues/40926394
- https://groups.google.com/a/chromium.org/g/chromium-extensions/c/DET2SXCFnDg
- https://issues.chromium.org/issues/40916430

- To capture microphone in an extension, there are tricks: https://github.com/justinmann/sidepanel-audio-issue , https://medium.com/@lynchee.owo/how-to-enable-microphone-access-in-chrome-extensions-by-code-924295170080 (comments)
