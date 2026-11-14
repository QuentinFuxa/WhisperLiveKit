# Partial review: seven of sixteen reports

These are intermediate observations, not a completed comparison. The driver is paused on battery power. AC/battery conditions changed during the series, so the raw latency differences below do not establish gains under matched power conditions. See [the method](METHODS.md), [the retained summaries](summary.md) and the compressed source reports in `reports/`.

## Whisper small MLX

All 270 short-clip sessions complete without pipeline errors or invalid line timestamps. WER is 11.6645% in English and 17.7778% in French; Chinese CER is 25.3077%. Each language has identical edit counts across the three passes.

The ten-minute Chinese stream completes and preserves its accumulated output, but its hypothesis contains 31 consecutive occurrences of `的學者和衛星` starting at character 1473. The same repetition occurred in the discarded pacing trial. This is an observed long-stream quality failure, not a successful streaming validation. The report contains no token-level emission trace that would justify assigning this text a precise audio offset. Full output mode has unlimited retention here, so the default diff-mode retention limit does not explain the missing/repeated content.

## Nemotron, English

All 90 sessions complete without the previous MLX stream-ownership failure or invalid line timestamps. WER is 19.2661% in all three passes, 7.6016 percentage points worse than Whisper small. EOF p95 improves by 82.6–85.2% per pass, while measured RSS and MLX allocator peaks are higher. This fails the agreed quality threshold against Whisper small.

The gap includes spoken versus written numbers (`fleurs_en_1850`, Wi-Fi standard names), but is not solely formatting: `fleurs_en_1975` changes “Vatican City's” to “American city” and drops part of the population statement; `fleurs_en_1866` substitutes several words in the riding-boots sentence. Keep the original normalization and retained hypotheses; no post-hoc score adjustment is used to obtain a pass.

## Nemotron, French

All 90 sessions complete with valid line timestamp ordering. WER is 14.9708% in all three passes, 2.8070 points better than Whisper small. Pooled EOF p95 is 0.1134 s, compared with 0.6996 s for Whisper; per-pass gains are 82.9–84.4%. The numerical gate passes against Whisper for this language. MLX allocator peak is about 64.8% higher, while RSS varies between passes. This language-specific result does not cancel the English regression or establish long-stream reliability.


## Nemotron, Chinese

All 90 sessions complete with valid line timestamp ordering. Pooled CER is 22.6154%, compared with 25.3077% for Whisper small. Pooled EOF p95 is 0.0515 s and first-visible p95 is 4.1435 s. The ten-minute Nemotron stream has not run, so its long-stream behavior remains unvalidated. Neither Qwen implementation has completed this matrix yet.


## Separate check: Whisper MLX direct English translation

The adapter fix in [#446](https://github.com/QuentinFuxa/WhisperLiveKit/pull/446) forwards the previously ignored translation task. Its factory/adapter scenario and all CI checks pass, but real-audio validation does not. These checks use the same pinned Whisper small model and the French/Chinese FLEURS recording with sentence ID 1682; they are separate from the ASR ranking above.

The Chinese streaming output starts in English, then repeats `插` extensively. The French stream starts with repeated “Moses” and contains several mistranslations. Calling `mlx_whisper.transcribe` directly on the full recordings also produces an unrelated English loop for Chinese and mistranslations for French, with or without word timestamps. This rules out word-timestamp conversion as the sole cause; it does not isolate every contribution from the model, runtime and streaming prompts. The fix remains in draft pending a usable configuration and further diagnosis.

The retained `whisper-direct-{fr,zh}.json.gz` files contain the WLK transcription/translation pair, runtime, source commit, reference and audio hash. The `whisper-direct-upstream-{fr,zh}.json.gz` files retain both full-recording calls (`task="translate"`, source language specified, `word_timestamps=False` then `True`). The Chinese validation status is an assertion failure on the output language, not a reported pipeline exception; the French output was reviewed manually and is not accepted as a quality pass.
