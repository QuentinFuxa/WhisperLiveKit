from __future__ import annotations

import re

import numpy as np
import torch

from .realtime_config import RealtimeAudioConfig


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    config: RealtimeAudioConfig,
) -> torch.Tensor:
    if sample_rate != config.sample_rate:
        import librosa

        audio = librosa.resample(
            audio.astype(np.float32, copy=False),
            orig_sr=sample_rate,
            target_sr=config.sample_rate,
        )
        sample_rate = config.sample_rate

    import librosa

    hop_length = int(round(sample_rate * config.mel_hop_ms / 1000.0))
    n_fft = 400
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32, copy=False),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=config.n_mels,
        fmin=20,
        fmax=sample_rate / 2,
        power=2.0,
        center=False,
    )
    mel = np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)
    if mel.shape[0] < config.frames_per_decoder_step:
        pad = config.frames_per_decoder_step - mel.shape[0]
        mel = np.pad(mel, ((0, pad), (0, 0)), mode="constant")
    return torch.from_numpy(mel)


def clean_decoded_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def decode_realtime_token_ids(
    tokenizer,
    token_ids: list[int],
    *,
    wait_token_id: int,
    word_start_token_id: int,
) -> str:
    filtered = [
        token_id
        for token_id in token_ids
        if token_id not in {wait_token_id, word_start_token_id, -100}
    ]
    if not filtered:
        return ""
    return clean_decoded_text(tokenizer.decode(filtered, skip_special_tokens=True))
