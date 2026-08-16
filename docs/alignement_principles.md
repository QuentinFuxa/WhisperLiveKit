# Alignment between transcription and diarization

WhisperLiveKit aligns timestamped ASR tokens with timestamped diarization
spans. Punctuation is not required to detect a speaker turn.

For every committed ASR token covered by the diarization timeline:

1. Measure its overlap with the available speaker spans.
2. Select the speaker with the largest overlap.
3. Start a new output line when that speaker differs from the previous token.

The output uses 1-based speaker IDs, while Sortformer uses 0-based IDs
internally. A token that crosses a speaker boundary stays intact and is
assigned to the side with the larger overlap. A zero-duration token is
assigned to the speaker span that contains its timestamp. A standalone
punctuation token closes the preceding speaker line, including when its
timestamp falls exactly on the next speaker boundary.

```text
ASR tokens:       [Hello ][there ][Goodbye ][now]
Diarization:      [ speaker 1    ][ speaker 2     ]
Output line 1:    [Hello there   ]
Output line 2:                    [Goodbye now    ]
```

## Diarization lag

Transcription can run ahead of diarization. Tokens that start at or after the
latest diarization timestamp remain in `buffer_diarization` instead of being
assigned speculatively.

```text
ASR tokens:       [first ][second]
Diarization:      [ speaker 1 ]
Lines:            [first ]
Buffer:                   [second]
```

Each refresh rebuilds the lines from the committed token timestamps. When
diarization catches up, buffered text moves once into the resolved speaker
line without losing text or word timestamps.

## Translation spans

A validated translation may cover source tokens that diarization splits into
several speaker lines. WhisperLiveKit attaches that translation exactly once,
to the speech line with the largest temporal overlap. A point translation on
an internal boundary belongs to the following line. If no line overlaps the
translation, the nearest speech line is used; equal distances select the
preceding line by stable transcript order.

The association is rebuilt on every refresh, so the same translated text is
not appended twice when the client requests another snapshot.

## Silence and model limits

Explicit silence is always a separate line with `speaker: -2`. It is never
folded into a speech line or held in the diarization buffer.

The alignment selects one speaker per ASR token. It does not change the
number of speakers supported by the configured diarization model, and it does
not represent simultaneous speakers within one token.

For Sortformer, `--sortformer-max-speakers N` retains the first N model
channels in speaker-arrival order. It is a declared upper bound, not an
estimate. If the recording actually contains more than N speakers, later
speakers may be assigned to one of the retained labels. The setting does not
change transcription text, word timestamps, or the one-speaker-per-token
alignment described above.
