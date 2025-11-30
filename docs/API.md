# WhisperLiveKit WebSocket API Documentation

WLK provides real-time speech transcription, speaker diarization, and translation through a WebSocket API. The server sends updates as audio is processed, allowing clients to display live transcription results with minimal latency.

---

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Main web interface with visual styling |
| `/text` | Simple text-based interface for easy copy/paste (debug/development) |
| `/asr` | WebSocket endpoint for audio streaming |

---

## Message Format

### Transcript Update (Server → Client)

```typescript
{
  "type": "transcript_update",
  "status": "active_transcription" | "no_audio_detected",
  "segments": [
    {
      "id": number,
      "speaker": number,
      "text": string,
      "start_speaker": string,    // HH:MM:SS format
      "start": string,            // HH:MM:SS format  
      "end": string,              // HH:MM:SS format
      "language": string | null,
      "translation": string,
      "buffer": {
        "transcription": string,
        "diarization": string,
        "translation": string
      }
    }
  ],
  "metadata": {
    "remaining_time_transcription": float,
    "remaining_time_diarization": float
  }
}
```

### Other Message Types

#### Config Message (sent on connection)
```json
{
  "type": "config",
  "useAudioWorklet": true
}
```
- `useAudioWorklet`: If `true`, client should use AudioWorklet for PCM streaming. If `false`, use MediaRecorder for WebM.

#### Ready to Stop Message (sent after processing complete)
```json
{
  "type": "ready_to_stop"
}
```
Indicates all audio has been processed and the client can safely close the connection.

---

## Field Descriptions

### Segment Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `number` | Unique identifier for this segment. |
| `speaker` | `number` | Speaker ID (1, 2, 3...). Special value `-2` indicates silence. |
| `text` | `string` | Validated transcription text. |
| `start_speaker` | `string` | Timestamp (HH:MM:SS) when this speaker segment began. |
| `start` | `string` | Timestamp (HH:MM:SS) of the first word. |
| `end` | `string` | Timestamp (HH:MM:SS) of the last word. |
| `language` | `string \| null` | ISO language code (e.g., "en", "fr"). `null` until detected. |
| `translation` | `string` | Validated translation text. |
| `buffer` | `Object` | Per-segment temporary buffers (see below). |

### Buffer Object (Per-Segment)

Buffers are **ephemeral**. They should be displayed to the user but are overwritten on each update. Only the **last non-silent segment** contains buffer content.

| Field | Type | Description |
|-------|------|-------------|
| `transcription` | `string` | Text pending validation (waiting for more context). |
| `diarization` | `string` | Text pending speaker assignment (diarization hasn't caught up). |
| `translation` | `string` | Translation pending validation. |

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `remaining_time_transcription` | `float` | Seconds of audio waiting for transcription. |
| `remaining_time_diarization` | `float` | Seconds of audio waiting for diarization. |

### Status Values

| Status | Description |
|--------|-------------|
| `active_transcription` | Normal operation, transcription is active. |
| `no_audio_detected` | No audio/speech has been detected yet. |

---

## Behavior Notes

### Silence Handling

- **Short silences (< 2 seconds)** are filtered out and not displayed.
- Only significant pauses appear as silence segments with `speaker: -2`.
- Consecutive same-speaker segments are merged even across short silences.

### Update Frequency

- **Active transcription**: ~20 updates/second (every 50ms)
- **During silence**: ~2 updates/second (every 500ms) to reduce bandwidth

### Token-by-Token Validation (Diarization Mode)

When diarization is enabled, text is validated **token-by-token** as soon as diarization covers each token, rather than waiting for punctuation. This provides:
- Faster text validation
- More responsive speaker attribution
- Buffer only contains tokens that diarization hasn't processed yet

---

## Example Messages

### Normal Transcription

```json
{
  "type": "transcript_update",
  "status": "active_transcription",
  "segments": [
    {
      "id": 1,
      "speaker": 1,
      "text": "Hello, how are you today?",
      "start_speaker": "0:00:02",
      "start": "0:00:02",
      "end": "0:00:05",
      "language": "en",
      "translation": "",
      "buffer": {
        "transcription": " I'm doing",
        "diarization": "",
        "translation": ""
      }
    }
  ],
  "metadata": {
    "remaining_time_transcription": 0.5,
    "remaining_time_diarization": 0
  }
}
```

### With Diarization Buffer

```json
{
  "type": "transcript_update",
  "status": "active_transcription",
  "segments": [
    {
      "id": 1,
      "speaker": 1,
      "text": "The meeting starts at nine.",
      "start_speaker": "0:00:03",
      "start": "0:00:03",
      "end": "0:00:06",
      "language": "en",
      "translation": "",
      "buffer": {
        "transcription": "",
        "diarization": " Let me check my calendar",
        "translation": ""
      }
    }
  ],
  "metadata": {
    "remaining_time_transcription": 0.3,
    "remaining_time_diarization": 2.1
  }
}
```

### Silence Segment

```json
{
  "id": 5,
  "speaker": -2,
  "text": "",
  "start_speaker": "0:00:10",
  "start": "0:00:10",
  "end": "0:00:15",
  "language": null,
  "translation": "",
  "buffer": {
    "transcription": "",
    "diarization": "",
    "translation": ""
  }
}
```

---

## Text Transcript Endpoint (`/text`)

The `/text` endpoint provides a simple, monospace text interface designed for:
- Easy copy/paste of transcripts
- Debugging and development
- Integration testing

Output uses text markers instead of HTML styling:

```
[METADATA transcription_lag=0.5s diarization_lag=1.2s]

[SPEAKER 1] 0:00:03 - 0:00:11 [LANG: en]
Hello world, how are you doing today?[DIAR_BUFFER] I'm doing fine[/DIAR_BUFFER]

[SILENCE 0:00:15 - 0:00:18]

[SPEAKER 2] 0:00:18 - 0:00:22 [LANG: en]
That's great to hear!
[TRANSLATION]C'est super à entendre![/TRANSLATION]
```

### Markers

| Marker | Description |
|--------|-------------|
| `[SPEAKER N]` | Speaker label with ID |
| `[SILENCE start - end]` | Silence segment |
| `[LANG: xx]` | Detected language code |
| `[DIAR_BUFFER]...[/DIAR_BUFFER]` | Text pending speaker assignment |
| `[TRANS_BUFFER]...[/TRANS_BUFFER]` | Text pending validation |
| `[TRANSLATION]...[/TRANSLATION]` | Translation content |
| `[METADATA ...]` | Lag/timing information |

