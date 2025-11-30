# Alignment Principles

This document explains how transcription tokens are aligned with diarization (speaker identification) segments.

---

## Token-by-Token Validation

When diarization is enabled, text is validated **token-by-token** rather than waiting for sentence boundaries. As soon as diarization covers a token's time range, that token is validated and assigned to the appropriate speaker.

### How It Works

1. **Transcription produces tokens** with timestamps (start, end)
2. **Diarization produces speaker segments** with timestamps
3. **For each token**: Check if diarization has caught up to that token's time
   - If yes → Find speaker with maximum overlap, validate token
   - If no → Keep token in "pending" (becomes diarization buffer)

```
Timeline:        0s -------- 5s -------- 10s -------- 15s
                 |           |            |            |
Transcription:   [Hello, how are you doing today?]
                 |_______|___|____|_____|_____|_____|
                   tok1  tok2 tok3 tok4  tok5  tok6

Diarization:     [SPEAKER 1        ][SPEAKER 2        ]
                 |__________________|__________________|
                 0s               8s                  15s

At time t when diarization covers up to 8s:
- Tokens 1-4 (0s-7s) → Validated as SPEAKER 1
- Tokens 5-6 (7s-10s) → In buffer (diarization hasn't caught up)
```

---

## Silence Handling

- **Short silences (< 2 seconds)**: Filtered out, not displayed
- **Significant silences (≥ 2 seconds)**: Displayed as silence segments with `speaker: -2`
- **Same speaker across gaps**: Segments are merged even if separated by short silences

```
Before filtering:
[SPK1 0:00-0:03] [SILENCE 0:03-0:04] [SPK1 0:04-0:08]

After filtering (silence < 2s):
[SPK1 0:00-0:08]  ← Merged into single segment
```

---

## Buffer Types

| Buffer | Contains | Displayed When |
|--------|----------|----------------|
| `transcription` | Text awaiting validation (more context needed) | Always on last segment |
| `diarization` | Text awaiting speaker assignment | When diarization lags behind transcription |
| `translation` | Translation awaiting validation | When translation is enabled |

---

## Legacy: Punctuation-Based Splitting

The previous approach split segments at punctuation marks and aligned with diarization at those boundaries. This is now replaced by token-by-token validation for faster, more responsive results.

### Historical Examples (for reference)

Example of punctuation-based alignment:

```text
punctuations_segments : __#_______.__________________!____
diarization_segments:
SPK1                    __#____________
SPK2                      #            ___________________
-->
ALIGNED SPK1            __#_______.
ALIGNED SPK2              #        __________________!____
```

With token-by-token validation, the alignment happens continuously rather than at punctuation boundaries.
