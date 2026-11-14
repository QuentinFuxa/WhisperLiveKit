"""Nemotron RNNT streaming using mlx-audio's mel frontend and encoder caches.

WLK owns endpointing. The small RNNT loop keeps decoder state between pushes;
mlx-audio's batch generator does not expose that state across calls. Token times
are encoder emission times, not forced word alignments.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

# MLX graphs retain stream IDs owned by the thread that created them.
# Keep model loading and every decode on that same thread across sessions.
_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wlk-nemotron-mlx")
_DEFAULT_LANGUAGE_TAGS = {
    "de": "de-DE", "en": "en-US", "es": "es-ES", "fr": "fr-FR",
    "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR", "pt": "pt-PT", "zh": "zh-CN",
}


def _normalize_language(language, supported_keys):
    if not language or language.casefold() == "auto":
        return None
    keys = {key.casefold(): key for key in supported_keys}
    normalized = language.replace('_', '-').casefold()
    if normalized in keys:
        return keys[normalized]
    primary = normalized.split('-')[0]
    default = _DEFAULT_LANGUAGE_TAGS.get(primary, '').casefold()
    if default in keys:
        return keys[default]
    matches = [value for key, value in keys.items() if key.split('-')[0] == primary]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"ASR language {language!r} is not supported; available: {', '.join(keys.values())}")


class NemotronMLXASR:
    sep = ""
    backend_choice = "nemotron-mlx-asr"

    def __init__(self, logfile=sys.stderr, **kwargs):
        self.model_id = kwargs.get('nemotron_mlx_asr_model', 'mlx-community/nemotron-3.5-asr-streaming-0.6b')
        self.att_context = list(kwargs.get('nemotron_mlx_asr_att_context', [56, 6]))
        if len(self.att_context) != 2 or any(value < 0 for value in self.att_context):
            raise ValueError('Nemotron attention context must contain two non-negative integers')
        _MLX_EXECUTOR.submit(self._load_model, kwargs.get("lan")).result()

    def _load_model(self, language):
        import mlx.core as mx
        from mlx_audio.stt import load

        self.model = load(self.model_id, strict=True)
        self.original_language = _normalize_language(language, self.model.prompt_dictionary)
        self.model.generate(mx.zeros((8000,), dtype=mx.float32),
                            language=self.original_language, att_context_size=self.att_context)


class NemotronMLXOnlineProcessor:
    SAMPLING_RATE = 16_000

    def __init__(self, asr, logfile=None):
        self.asr = asr
        self.model = asr.model
        self.sep = ""
        language = (asr._session_language if getattr(asr, '_override_language', False)
                    else asr.original_language)
        self.language = _normalize_language(language, self.model.prompt_dictionary)
        self._frame_sec = (self.model.encoder_config.subsampling_factor
                           * self.model.preprocessor_config.hop_length / self.SAMPLING_RATE)
        self._pending = []
        self._encoder = None
        self._mel = None
        self._audio_end = 0.0
        self._utterance_start = 0.0
        self._frame_offset = 0
        self._last_token = self.model.blank_id
        self._decoder_hidden = None
        self._detected_language = self.language
        self._last_text_ends_with_space = False

    def insert_audio_chunk(self, audio, audio_stream_end_time):
        audio = np.asarray(audio, dtype=np.float32)
        if not len(audio):
            return
        if self._encoder is None and not self._pending:
            self._utterance_start = max(0.0, audio_stream_end_time - len(audio) / self.SAMPLING_RATE)
        self._pending.append(audio.copy())
        self._audio_end = audio_stream_end_time

    def _process(self, *, final=False):
        return _MLX_EXECUTOR.submit(self._process_on_model_thread, final=final).result()

    def _process_on_model_thread(self, *, final=False):
        import mlx.core as mx
        from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
        from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState

        if not self._pending and self._encoder is None:
            return [], self._audio_end
        if self._encoder is None:
            self._encoder = ConformerStreamingState(self.model.encoder, att_context_size=self.asr.att_context)
            self._mel = StreamingLogMelSpectrogram(self.model.preprocessor_config)
        audio = np.concatenate(self._pending) if self._pending else np.empty(0, dtype=np.float32)
        self._pending.clear()
        mel = self._mel.push(mx.array(audio), final=final)
        tokens = []
        for encoded in self._encoder.push(mel, final=final):
            prompted = self.model.apply_prompt(encoded, self.language)
            tokens.extend(self._decode_chunk(prompted))
        hidden = self._decoder_hidden or ()
        self._encoder.materialize(*hidden)
        tokens = self._collapse_space_runs(tokens)
        if final:
            # The model closes an utterance with a standalone '▁' token and opens the
            # next one with a space-prefixed first word. Both encode the same
            # separator, so joining the committed pieces reads 'test.  We'. Keep the
            # next utterance's space and drop this trailing one.
            while tokens and not tokens[-1].text.strip():
                tokens.pop()
            if tokens:
                tokens[-1].text = tokens[-1].text.rstrip()
                # This line no longer ends in a space; the next utterance's
                # leading space supplies the separator.
                self._last_text_ends_with_space = False
            # A chunk that emits nothing (trailing silence) leaves the flag as the
            # previous chunk left it: that text is still the last one committed.
            self._encoder = self._mel = None
            self._frame_offset = 0
            self._last_token = self.model.blank_id
            self._decoder_hidden = None
            self._detected_language = self.language
        return tokens, self._audio_end

    def _collapse_space_runs(self, tokens):
        # The decoder emits a standalone space token and then a space-prefixed word
        # whenever a word starts after a token boundary -- at an utterance start, but
        # also mid-utterance and across decode batches. Joined verbatim that reads
        # 'the.  We'. Collapse each run to exactly one space: the space separates the
        # two words, so dropping all of it would glue them into 'the.We'.
        kept = []
        for token in tokens:
            text = token.text
            if self._last_text_ends_with_space and text.startswith(' '):
                text = text.lstrip(' ')
                token.text = text
            if not text:
                continue
            self._last_text_ends_with_space = text.endswith(' ')
            kept.append(token)
        return kept

    def _decode_chunk(self, prompted):
        # The greedy recurrence follows mlx-audio's Model._decode_prompted_chunks;
        # only its state lifetime and WLK token conversion differ.
        import mlx.core as mx
        from mlx_audio.stt.models.nemotron_asr import tokenizer

        model = self.model
        frame, symbols = 0, 0
        tokens = []
        while frame < prompted.shape[1]:
            feature = prompted[:, frame:frame+1]
            current = (mx.array([[self._last_token]], dtype=mx.int32)
                       if self._last_token != model.blank_id else None)
            decoded, (h, c) = model.decoder(current, self._decoder_hidden)
            predicted = int(mx.argmax(model.joint(feature, decoded.astype(feature.dtype))))
            if predicted != model.blank_id:
                self._last_token = predicted
                self._decoder_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
                detected = tokenizer.detected_language([predicted], model.vocabulary)
                if detected:
                    self._detected_language = detected
                if not tokenizer.is_special_token(predicted, model.vocabulary):
                    text = tokenizer.decode([predicted], model.vocabulary)
                    start = min(self._audio_end, self._utterance_start + (self._frame_offset + frame) * self._frame_sec)
                    if text:
                        tokens.append(ASRToken(start=start, end=min(start+self._frame_sec, self._audio_end),
                                               text=text,
                                               detected_language=self._detected_language))
                symbols += 1
                if model.max_symbols is not None and symbols >= model.max_symbols:
                    frame += 1
                    symbols = 0
            else:
                frame += 1
                symbols = 0
        self._frame_offset += prompted.shape[1]
        return tokens

    def process_iter(self, is_last=False):
        return self._process(final=is_last)

    def finish(self):
        return self._process(final=True)

    def start_silence(self):
        return self.finish()

    def new_speaker(self, *args, **kwargs):
        return self.finish()

    def end_silence(self, silence_duration, offset):
        self._audio_end += silence_duration

    def get_buffer(self):
        # RNNT symbols are committed as they are emitted; there is no mutable tail.
        return Transcript(start=self._audio_end, end=self._audio_end, text='')
