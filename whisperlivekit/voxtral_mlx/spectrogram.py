"""
Mel spectrogram computation for Voxtral Realtime.

Provides both a full-audio function and an incremental streaming variant
that maintains overlap state between calls.  The DFT is computed via
matrix multiplication in MLX — no external FFT dependency required.
"""

import math

import mlx.core as mx
import numpy as np

# Audio / mel constants matching the Voxtral Realtime model expectations.
SAMPLE_RATE = 16_000
WINDOW_SIZE = 400        # n_fft
HOP = 160
MEL_BANDS = 128
MEL_MAX = 1.5            # global log-mel normalisation ceiling
# Each output audio token spans: hop * conv_stride(2) * downsample_factor(4)
SAMPLES_PER_TOKEN = HOP * 2 * 4  # = 1280 samples = 80 ms

# Padding tokens used by the model prompt structure.
LEFT_PAD_TOKENS = 32
RIGHT_PAD_TOKENS = 17


# ---------------------------------------------------------------------------
# Slaney mel filterbank
# ---------------------------------------------------------------------------

def _build_slaney_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = WINDOW_SIZE,
    n_mels: int = MEL_BANDS,
    lo_hz: float = 0.0,
    hi_hz: float = 8000.0,
) -> np.ndarray:
    """Compute a Slaney-normalised triangular mel filterbank.

    Returns an array of shape ``[n_mels, n_fft//2 + 1]``.
    """

    def _hz2mel(f):
        threshold = 1000.0
        base_mel = 15.0
        log_coeff = 27.0 / np.log(6.4)
        mel = 3.0 * f / 200.0
        if isinstance(f, np.ndarray):
            above = f >= threshold
            mel[above] = base_mel + np.log(f[above] / threshold) * log_coeff
        elif f >= threshold:
            mel = base_mel + np.log(f / threshold) * log_coeff
        return mel

    def _mel2hz(m):
        threshold = 1000.0
        base_mel = 15.0
        log_coeff = np.log(6.4) / 27.0
        hz = 200.0 * m / 3.0
        above = m >= base_mel
        hz[above] = threshold * np.exp(log_coeff * (m[above] - base_mel))
        return hz

    n_bins = n_fft // 2 + 1
    fft_hz = np.linspace(0, sr / 2, n_bins)
    mel_lo, mel_hi = _hz2mel(lo_hz), _hz2mel(hi_hz)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = _mel2hz(mel_pts)
    diffs = np.diff(hz_pts)

    slopes = np.expand_dims(hz_pts, 0) - np.expand_dims(fft_hz, 1)
    rising = -slopes[:, :-2] / diffs[:-1]
    falling = slopes[:, 2:] / diffs[1:]
    fb = np.maximum(0.0, np.minimum(rising, falling))

    # Slaney area normalisation
    widths = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= np.expand_dims(widths, 0)
    return fb.T.astype(np.float32)


_CACHED_FILTERS: mx.array | None = None


def _mel_filters() -> mx.array:
    global _CACHED_FILTERS
    if _CACHED_FILTERS is None:
        _CACHED_FILTERS = mx.array(_build_slaney_filterbank())
    return _CACHED_FILTERS


# ---------------------------------------------------------------------------
# DFT helpers
# ---------------------------------------------------------------------------

def _hann_window() -> mx.array:
    return mx.array(np.hanning(WINDOW_SIZE + 1)[:-1].astype(np.float32))


def _dft_matrices():
    """Pre-compute the real / imaginary DFT basis matrices."""
    n_bins = WINDOW_SIZE // 2 + 1
    k = mx.arange(n_bins, dtype=mx.float32)[:, None]
    n = mx.arange(WINDOW_SIZE, dtype=mx.float32)[None, :]
    phase = -2.0 * math.pi * (k @ n) / WINDOW_SIZE
    return mx.cos(phase), mx.sin(phase)


def _stft_frames(audio: mx.array, window: mx.array) -> mx.array:
    """Frame *audio* using the Hann window and compute power spectrogram."""
    n_bins = WINDOW_SIZE // 2 + 1
    n_frames = 1 + (audio.shape[0] - WINDOW_SIZE) // HOP
    if n_frames <= 0:
        return mx.zeros((0, n_bins))

    offsets = (mx.arange(n_frames) * HOP)[:, None]
    indices = offsets + mx.arange(WINDOW_SIZE)[None, :]
    windowed = audio[indices] * window[None, :]

    dft_re, dft_im = _dft_matrices()
    real_part = windowed @ dft_re.T
    imag_part = windowed @ dft_im.T
    return real_part ** 2 + imag_part ** 2


def _apply_mel_and_log(power: mx.array) -> mx.array:
    """Convert a power spectrogram to log-mel and normalise."""
    mel = power @ _mel_filters().T
    log_mel = mx.log10(mx.maximum(mel, 1e-10))
    log_mel = mx.maximum(log_mel, MEL_MAX - 8.0)
    return (log_mel + 4.0) / 4.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_mel(audio: np.ndarray) -> mx.array:
    """Compute log-mel spectrogram for a complete audio signal.

    Args:
        audio: 1-D float32 numpy array at ``SAMPLE_RATE``.

    Returns:
        ``[MEL_BANDS, T]`` MLX array.
    """
    x = mx.array(audio)
    pad = WINDOW_SIZE // 2
    x = mx.pad(x, [(pad, pad)])
    window = _hann_window()

    power = _stft_frames(x, window)
    # Drop last frame to match reference STFT behaviour
    power = power[:-1]
    return _apply_mel_and_log(power).T


def compute_mel_streaming(
    chunk: np.ndarray,
    overlap: np.ndarray | None,
) -> tuple[mx.array, np.ndarray]:
    """Incrementally compute log-mel for a new audio chunk.

    Args:
        chunk: New audio samples (float32 numpy).
        overlap: The last ``WINDOW_SIZE - HOP`` = 240 samples from the
            previous call, or *None* on the first call (uses zero-padding).

    Returns:
        ``(mel, new_overlap)`` where *mel* is ``[MEL_BANDS, N]`` and
        *new_overlap* is the 240-sample tail for the next call.
    """
    tail_len = WINDOW_SIZE - HOP  # 240

    if overlap is not None:
        combined = np.concatenate([overlap, chunk])
    else:
        combined = np.concatenate([np.zeros(WINDOW_SIZE // 2, dtype=np.float32), chunk])

    new_overlap = combined[-tail_len:].copy()

    x = mx.array(combined)
    window = _hann_window()
    power = _stft_frames(x, window)

    if power.shape[0] == 0:
        return mx.zeros((MEL_BANDS, 0)), new_overlap

    return _apply_mel_and_log(power).T, new_overlap


def pad_audio(
    audio: np.ndarray,
    n_left: int = LEFT_PAD_TOKENS,
    n_right: int = RIGHT_PAD_TOKENS,
) -> np.ndarray:
    """Pad audio with silence for batch (non-streaming) inference."""
    left = n_left * SAMPLES_PER_TOKEN
    align = (SAMPLES_PER_TOKEN - (len(audio) % SAMPLES_PER_TOKEN)) % SAMPLES_PER_TOKEN
    right = align + n_right * SAMPLES_PER_TOKEN
    return np.pad(audio, (left, right))
