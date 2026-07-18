# Canary-1b-v2 Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `canary` ASR backend that runs NVIDIA `nvidia/canary-1b-v2` on the existing LocalAgreement streaming policy, with AmberNet-based detect-once-then-lock auto language detection.

**Architecture:** A new `whisperlivekit/canary_backend.py` holds `CanaryASR` (shared NeMo `EncDecMultiTaskModel` implementing the LocalAgreement `transcribe`/`ts_words`/`segments_end_ts` contract), `CanaryLID` (shared `EncDecSpeakerLabelModel` / `langid_ambernet` language identifier), and `CanarySessionASR` (a per-session `SessionASRProxy` subclass that resolves the source language, detecting once via LID for `auto` sessions then locking it). `core.py` instantiates and routes `canary` before the `backend_policy` branch; the whole VAD→ASR→output pipeline and `OnlineASRProcessor` are reused unchanged.

**Tech Stack:** Python, NeMo (`nemo_toolkit[asr]`, lazy-imported), PyTorch/CUDA, existing WhisperLiveKit LocalAgreement pipeline, pytest.

**Spec:** `docs/superpowers/specs/2026-07-18-canary-1b-v2-backend-design.md`

---

## File Structure

- **Create** `whisperlivekit/canary_backend.py` — `CanaryASR`, `CanaryLID`, `CanarySessionASR`, and the pure helpers `map_voxlingua_to_canary()`, `canary_words_to_tokens()`, `canary_segment_end_ts()`. NeMo/torch imported lazily inside methods so the module imports without NeMo installed.
- **Modify** `whisperlivekit/config.py` — add `canary_*` dataclass fields.
- **Modify** `whisperlivekit/parse_args.py` — add `"canary"` to `--backend` choices and a `--canary-*` argument group.
- **Modify** `whisperlivekit/core.py` — `_do_init()` branch to build `CanaryASR` (+ attach `CanaryLID`); `online_factory()` routing to wrap in `CanarySessionASR` + `OnlineASRProcessor`.
- **Modify** `pyproject.toml` — add a `canary` optional-dependency extra.
- **Create** `tests/test_canary_backend.py` — pure-function unit tests (no model) + availability-gated integration tests.

---

## Task 1: Config fields and CLI arguments

**Files:**
- Modify: `whisperlivekit/config.py` (after the qwen3 streaming block, before `def __post_init__`, around line 150)
- Modify: `whisperlivekit/parse_args.py` (`--backend` choices at line 209; new group after the SimulStreaming group)
- Test: `tests/test_canary_backend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_canary_backend.py` with:

```python
def test_parse_args_accepts_canary_options(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "whisperlivekit-server",
            "--backend", "canary",
            "--canary-model", "nvidia/canary-1b-v2",
            "--canary-default-lang", "de",
            "--canary-lid-model", "langid_ambernet",
            "--canary-lid-min-sec", "3.0",
            "--canary-lid-min-conf", "0.6",
        ],
    )
    cfg = parse_args()
    assert cfg.backend == "canary"
    assert cfg.canary_model == "nvidia/canary-1b-v2"
    assert cfg.canary_default_lang == "de"
    assert cfg.canary_lid_model == "langid_ambernet"
    assert cfg.canary_lid_min_sec == 3.0
    assert cfg.canary_lid_min_conf == 0.6


def test_canary_config_defaults():
    from whisperlivekit.config import WhisperLiveKitConfig

    cfg = WhisperLiveKitConfig.from_kwargs(backend="canary")
    assert cfg.canary_model == "nvidia/canary-1b-v2"
    assert cfg.canary_default_lang == "en"
    assert cfg.canary_lid_model == "langid_ambernet"
    assert cfg.canary_lid_min_sec == 2.0
    assert cfg.canary_lid_min_conf == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canary_backend.py::test_canary_config_defaults tests/test_canary_backend.py::test_parse_args_accepts_canary_options -v`
Expected: FAIL — `AttributeError`/`TypeError` (unknown `canary_*` fields) and `argparse` error on `--canary-model`.

- [ ] **Step 3: Add config fields**

In `whisperlivekit/config.py`, immediately after the `qwen3_streaming_block_frames: int = 192` line and before `def __post_init__(self):`, add:

```python
    # Canary backend (NeMo EncDecMultiTaskModel on LocalAgreement)
    canary_model: str = "nvidia/canary-1b-v2"
    canary_default_lang: str = "en"
    canary_lid_model: str = "langid_ambernet"
    canary_lid_min_sec: float = 2.0
    canary_lid_min_conf: float = 0.5
```

- [ ] **Step 4: Add the `canary` backend choice**

In `whisperlivekit/parse_args.py`, edit the `--backend` `choices` list (line ~209) to include `"canary"`:

```python
        choices=["auto", "mlx-whisper", "faster-whisper", "whisper", "openai-api", "voxtral", "voxtral-mlx", "qwen3-vllm", "qwen3-vllm-metal", "qwen3-streaming", "canary"],
```

- [ ] **Step 5: Add the `--canary-*` argument group**

In `whisperlivekit/parse_args.py`, after the SimulStreaming argument group (search for `simulstreaming_group = parser.add_argument_group`) and before the return, add:

```python
    # Canary backend arguments
    canary_group = parser.add_argument_group(
        "Canary backend arguments (only used with --backend canary)"
    )
    canary_group.add_argument(
        "--canary-model",
        type=str,
        default="nvidia/canary-1b-v2",
        dest="canary_model",
        help="Canary model: HuggingFace/NGC id or local .nemo path. Default nvidia/canary-1b-v2.",
    )
    canary_group.add_argument(
        "--canary-default-lang",
        type=str,
        default="en",
        dest="canary_default_lang",
        help="Fallback source language used while auto-detecting (and if detection stays low-confidence).",
    )
    canary_group.add_argument(
        "--canary-lid-model",
        type=str,
        default="langid_ambernet",
        dest="canary_lid_model",
        help="NeMo spoken-language-ID model used for auto detection (EncDecSpeakerLabelModel).",
    )
    canary_group.add_argument(
        "--canary-lid-min-sec",
        type=float,
        default=2.0,
        dest="canary_lid_min_sec",
        help="Minimum seconds of buffered audio before language detection runs.",
    )
    canary_group.add_argument(
        "--canary-lid-min-conf",
        type=float,
        default=0.5,
        dest="canary_lid_min_conf",
        help="Minimum LID confidence (0-1) required to lock the detected language.",
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_canary_backend.py::test_canary_config_defaults tests/test_canary_backend.py::test_parse_args_accepts_canary_options -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add whisperlivekit/config.py whisperlivekit/parse_args.py tests/test_canary_backend.py
git commit -m "feat(canary): add canary backend config fields and CLI args"
```

---

## Task 2: Pure helpers — word/segment timestamp mapping

**Files:**
- Create: `whisperlivekit/canary_backend.py`
- Test: `tests/test_canary_backend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_canary_backend.py`:

```python
def test_canary_words_to_tokens_maps_word_timestamps():
    from whisperlivekit.canary_backend import canary_words_to_tokens

    word_stamps = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.4, "end": 0.9},
    ]
    tokens = canary_words_to_tokens(word_stamps)
    assert [t.text for t in tokens] == ["hello", "world"]
    assert tokens[0].start == 0.0 and tokens[0].end == 0.4
    assert tokens[1].start == 0.4 and tokens[1].end == 0.9


def test_canary_words_to_tokens_handles_missing_stamps():
    from whisperlivekit.canary_backend import canary_words_to_tokens

    assert canary_words_to_tokens(None) == []
    assert canary_words_to_tokens([]) == []


def test_canary_segment_end_ts():
    from whisperlivekit.canary_backend import canary_segment_end_ts

    seg_stamps = [
        {"segment": "hello world.", "start": 0.0, "end": 0.9},
        {"segment": "bye.", "start": 1.0, "end": 1.5},
    ]
    assert canary_segment_end_ts(seg_stamps) == [0.9, 1.5]
    assert canary_segment_end_ts(None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canary_backend.py -k "words_to_tokens or segment_end_ts" -v`
Expected: FAIL — `ModuleNotFoundError: whisperlivekit.canary_backend`.

- [ ] **Step 3: Create the module with the pure helpers**

Create `whisperlivekit/canary_backend.py`:

```python
"""NVIDIA Canary-1b-v2 backend for WhisperLiveKit (LocalAgreement policy).

Contains:
  - CanaryASR: shared NeMo EncDecMultiTaskModel implementing the LocalAgreement
    transcribe()/ts_words()/segments_end_ts() contract.
  - CanaryLID: shared NeMo language-ID model (langid_ambernet).
  - CanarySessionASR: per-session proxy that resolves the source language,
    auto-detecting once via CanaryLID for ``auto`` sessions then locking it.

NeMo and torch are imported lazily inside methods so this module imports fine
on machines without ``nemo_toolkit`` installed (routing/unit tests, non-canary
deployments).
"""

import logging
from typing import List, Optional, Tuple

from whisperlivekit.session_asr_proxy import SessionASRProxy
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


# Canary-1b-v2's 25 supported source-language codes.
CANARY_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "uk",
}

# VoxLingua107 (AmberNet) codes that differ from Canary's expected code.
# VoxLingua107 is mostly ISO 639-1 already, so the map is small; extend after
# validating against the actual langid_ambernet label set.
_VOXLINGUA_TO_CANARY = {
    # placeholder for known mismatches, e.g. "gr": "el"
}


def map_voxlingua_to_canary(code: str) -> Optional[str]:
    """Map an AmberNet/VoxLingua107 language code to Canary's source_lang set.

    Returns the mapped code if Canary supports it, else None.
    """
    if not code:
        return None
    code = _VOXLINGUA_TO_CANARY.get(code, code)
    return code if code in CANARY_LANGS else None


def canary_words_to_tokens(word_stamps) -> List[ASRToken]:
    """Convert Canary ``timestamp['word']`` entries to ASRToken objects."""
    if not word_stamps:
        return []
    tokens: List[ASRToken] = []
    for w in word_stamps:
        text = w.get("word")
        if not text:
            continue
        tokens.append(ASRToken(w["start"], w["end"], text))
    return tokens


def canary_segment_end_ts(segment_stamps) -> List[float]:
    """Extract segment end times from Canary ``timestamp['segment']`` entries."""
    if not segment_stamps:
        return []
    return [s["end"] for s in segment_stamps]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_canary_backend.py -k "words_to_tokens or segment_end_ts" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add whisperlivekit/canary_backend.py tests/test_canary_backend.py
git commit -m "feat(canary): pure word/segment timestamp mapping helpers"
```

---

## Task 3: Language-code mapping helper

**Files:**
- Modify: `whisperlivekit/canary_backend.py` (helper already added in Task 2)
- Test: `tests/test_canary_backend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_canary_backend.py`:

```python
def test_map_voxlingua_to_canary_supported():
    from whisperlivekit.canary_backend import map_voxlingua_to_canary

    assert map_voxlingua_to_canary("en") == "en"
    assert map_voxlingua_to_canary("de") == "de"
    assert map_voxlingua_to_canary("uk") == "uk"


def test_map_voxlingua_to_canary_unsupported_returns_none():
    from whisperlivekit.canary_backend import map_voxlingua_to_canary

    assert map_voxlingua_to_canary("zh") is None   # Chinese not in Canary's 25
    assert map_voxlingua_to_canary("") is None
    assert map_voxlingua_to_canary(None) is None
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_canary_backend.py -k "map_voxlingua" -v`
Expected: PASS (2 passed) — the helper was added in Task 2; these lock its contract.

- [ ] **Step 3: Commit**

```bash
git add tests/test_canary_backend.py
git commit -m "test(canary): lock voxlingua->canary code mapping contract"
```

---

## Task 4: `CanarySessionASR` auto-detect wrapper

**Files:**
- Modify: `whisperlivekit/canary_backend.py`
- Test: `tests/test_canary_backend.py`

This task tests control-flow (language resolution / detect-once-then-lock) with
tiny deterministic stubs — not the ASR pipeline. Real end-to-end coverage is
Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_canary_backend.py`:

```python
import numpy as np


class _RecordingCanaryASR:
    """Minimal stand-in for CanaryASR that records the source language used."""
    sep = " "

    def __init__(self):
        self.original_language = None
        self.calls = []

    def transcribe(self, audio, init_prompt=""):
        self.calls.append(self.original_language)
        return f"decoded:{self.original_language}"


class _StubLID:
    def __init__(self, code="de", conf=0.9):
        self._code, self._conf = code, conf
        self.n_calls = 0

    def detect(self, audio):
        self.n_calls += 1
        return self._code, self._conf


def _audio(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_explicit_language_bypasses_lid():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID()
    session = CanarySessionASR(asr, "fr", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(5))
    assert asr.calls == ["fr"]
    assert lid.n_calls == 0


def test_auto_uses_default_until_enough_audio():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID(code="de", conf=0.9)
    session = CanarySessionASR(asr, "auto", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(1.0))          # below lid_min_sec -> default, no LID
    assert asr.calls == ["en"]
    assert lid.n_calls == 0


def test_auto_detects_once_then_locks():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID(code="de", conf=0.9)
    session = CanarySessionASR(asr, "auto", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(3.0))          # detects -> "de"
    session.transcribe(_audio(4.0))          # locked -> "de", no second detect
    assert asr.calls == ["de", "de"]
    assert lid.n_calls == 1


def test_auto_low_confidence_stays_on_default():
    from whisperlivekit.canary_backend import CanarySessionASR

    asr = _RecordingCanaryASR()
    lid = _StubLID(code="de", conf=0.2)       # below lid_min_conf
    session = CanarySessionASR(asr, "auto", lid=lid, default_lang="en",
                               lid_min_sec=2.0, lid_min_conf=0.5)
    session.transcribe(_audio(3.0))
    assert asr.calls == ["en"]                # not locked, retried later
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_canary_backend.py -k "CanarySession or explicit_language or auto_" -v`
Expected: FAIL — `ImportError: cannot import name 'CanarySessionASR'`.

- [ ] **Step 3: Implement `CanarySessionASR`**

Append to `whisperlivekit/canary_backend.py`:

```python
class CanarySessionASR(SessionASRProxy):
    """Per-session Canary proxy with auto language detection.

    For explicit-language sessions this behaves like SessionASRProxy, forcing
    ``source_lang`` to the chosen code. For ``auto`` sessions it uses
    ``default_lang`` until ``lid_min_sec`` of audio is buffered, then runs the
    shared LID once; on a confident result it locks that language for the rest
    of the session.
    """

    SAMPLING_RATE = 16000

    def __init__(self, asr, language, lid=None, default_lang="en",
                 lid_min_sec=2.0, lid_min_conf=0.5):
        super().__init__(asr, language)
        is_auto = (language is None) or (language == "auto")
        object.__setattr__(self, "_is_auto", is_auto)
        object.__setattr__(self, "_lid", lid)
        object.__setattr__(self, "_default_lang", default_lang)
        object.__setattr__(self, "_lid_min_sec", lid_min_sec)
        object.__setattr__(self, "_lid_min_conf", lid_min_conf)
        object.__setattr__(self, "_detected_lang", None)

    def _resolve_language(self, audio) -> str:
        # Explicit language: SessionASRProxy stored it as _session_language.
        if not self._is_auto:
            return self._session_language
        if self._detected_lang is not None:
            return self._detected_lang
        if self._lid is not None and len(audio) >= self._lid_min_sec * self.SAMPLING_RATE:
            try:
                code, conf = self._lid.detect(audio)
            except Exception as e:  # noqa: BLE001
                logger.warning("Canary LID failed: %s", e)
                code, conf = None, 0.0
            if code is not None and conf >= self._lid_min_conf:
                object.__setattr__(self, "_detected_lang", code)
                logger.info("Canary auto-detected language: %s (conf=%.2f)", code, conf)
                return code
        return self._default_lang

    def transcribe(self, audio, init_prompt=""):
        with self._lock:
            lang = self._resolve_language(audio)
            saved = self._asr.original_language
            self._asr.original_language = lang
            try:
                return self._asr.transcribe(audio, init_prompt=init_prompt)
            finally:
                self._asr.original_language = saved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_canary_backend.py -k "CanarySession or explicit_language or auto_" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add whisperlivekit/canary_backend.py tests/test_canary_backend.py
git commit -m "feat(canary): CanarySessionASR detect-once-then-lock auto language"
```

---

## Task 5: `CanaryASR` and `CanaryLID` model classes

**Files:**
- Modify: `whisperlivekit/canary_backend.py`
- Test: `tests/test_canary_backend.py` (availability-gated integration test)

- [ ] **Step 1: Write the failing (gated) integration test**

Append to `tests/test_canary_backend.py`:

```python
import importlib.util

import pytest

_NEMO_AVAILABLE = importlib.util.find_spec("nemo") is not None
requires_nemo = pytest.mark.skipif(not _NEMO_AVAILABLE, reason="NeMo not installed")


@requires_nemo
def test_canary_asr_transcribes_with_word_timestamps():
    from whisperlivekit.canary_backend import CanaryASR
    from whisperlivekit.test_data import load_test_audio  # 16kHz mono float32 + text

    asr = CanaryASR(
        lan="en",
        canary_model="nvidia/canary-1b-v2",
        buffer_trimming="segment",
        buffer_trimming_sec=15.0,
        confidence_validation=False,
    )
    audio, _reference = load_test_audio()
    res = asr.transcribe(audio, source_lang="en")
    tokens = asr.ts_words(res)
    assert tokens, "expected at least one word token"
    assert all(t.end >= t.start for t in tokens)
    ends = asr.segments_end_ts(res)
    assert ends and ends[-1] > 0
```

Note: adapt `load_test_audio` to whatever `tests/` or `whisperlivekit/test_data.py`
exposes (see `test_pipeline.py` for the existing audio-loading helper); the point
is a real 16 kHz numpy clip.

- [ ] **Step 2: Run test to verify it fails or skips**

Run: `pytest tests/test_canary_backend.py::test_canary_asr_transcribes_with_word_timestamps -v`
Expected (no NeMo): SKIPPED. Expected (NeMo present, class missing): FAIL — `ImportError: cannot import name 'CanaryASR'`.

- [ ] **Step 3: Implement `CanaryASR`**

Append to `whisperlivekit/canary_backend.py`:

```python
class CanaryASR:
    """Shared Canary model holder implementing the LocalAgreement contract."""

    sep = " "
    SAMPLING_RATE = 16000

    def __init__(self, lan="auto", canary_model="nvidia/canary-1b-v2",
                 buffer_trimming="segment", buffer_trimming_sec=15.0,
                 confidence_validation=False, canary_default_lang="en",
                 logfile=None, **_unused):
        import time

        self.original_language = None if lan == "auto" else lan
        self.canary_default_lang = canary_default_lang
        self.backend_choice = "canary"
        self.confidence_validation = confidence_validation
        self.tokenizer = None  # segment trimming needs no sentence tokenizer
        self.buffer_trimming = buffer_trimming
        self.buffer_trimming_sec = buffer_trimming_sec
        self.transcribe_kargs = {}
        self.lid_model = None  # attached by core.py when auto detection is enabled

        from nemo.collections.asr.models import ASRModel

        t = time.time()
        logger.info("Loading Canary model '%s' via NeMo...", canary_model)
        if canary_model.endswith(".nemo"):
            self.model = ASRModel.restore_from(canary_model)
        else:
            self.model = ASRModel.from_pretrained(model_name=canary_model)
        self.model.eval()
        logger.info("Canary model loaded in %.2fs", time.time() - t)

    def transcribe(self, audio, init_prompt="", source_lang=None):
        """Run Canary on a 16kHz mono float32 numpy window. Returns hyp[0]."""
        import numpy as np

        lang = source_lang or self.original_language or self.canary_default_lang
        audio = np.asarray(audio, dtype=np.float32)
        outputs = self.model.transcribe(
            [audio],
            source_lang=lang,
            target_lang=lang,
            timestamps=True,
            batch_size=1,
            verbose=False,
        )
        return outputs[0]

    def _word_stamps(self, res):
        ts = getattr(res, "timestamp", None) or {}
        return ts.get("word")

    def _segment_stamps(self, res):
        ts = getattr(res, "timestamp", None) or {}
        return ts.get("segment")

    def ts_words(self, res) -> List[ASRToken]:
        return canary_words_to_tokens(self._word_stamps(res))

    def segments_end_ts(self, res) -> List[float]:
        return canary_segment_end_ts(self._segment_stamps(res))

    def use_vad(self):
        logger.warning("VAD is handled upstream (Silero); CanaryASR.use_vad() is a no-op.")
```

- [ ] **Step 4: Implement `CanaryLID`**

Append to `whisperlivekit/canary_backend.py`:

```python
class CanaryLID:
    """Shared spoken-language-ID model (NeMo langid_ambernet / AmberNet)."""

    SAMPLING_RATE = 16000

    def __init__(self, lid_model="langid_ambernet", logfile=None, **_unused):
        import time

        import nemo.collections.asr as nemo_asr

        t = time.time()
        logger.info("Loading Canary LID model '%s' via NeMo...", lid_model)
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=lid_model
        )
        self.model.eval()
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            self.device = "cpu"
        logger.info("Canary LID model loaded in %.2fs", time.time() - t)

    def detect(self, audio) -> Tuple[Optional[str], float]:
        """Return (canary_lang_code_or_None, confidence) for a 16kHz clip."""
        import numpy as np
        import torch

        arr = np.asarray(audio, dtype=np.float32)
        sig = torch.tensor(arr).unsqueeze(0).to(self.device)
        sig_len = torch.tensor([sig.shape[1]]).to(self.device)
        with torch.no_grad():
            logits, _ = self.model.forward(input_signal=sig, input_signal_length=sig_len)
            probs = logits.softmax(dim=-1)
            conf, idx = probs.max(dim=-1)
        raw_code = self.model.cfg.labels[int(idx.item())]
        return map_voxlingua_to_canary(raw_code), float(conf.item())
```

- [ ] **Step 5: Run the gated test**

Run: `pytest tests/test_canary_backend.py::test_canary_asr_transcribes_with_word_timestamps -v`
Expected (no NeMo): SKIPPED. Expected (NeMo + weights): PASS.

- [ ] **Step 6: Verify the module still imports without NeMo**

Run: `python -c "import whisperlivekit.canary_backend; print('ok')"`
Expected: `ok` (no NeMo import at module load — it is inside `__init__`/`detect`).

- [ ] **Step 7: Commit**

```bash
git add whisperlivekit/canary_backend.py tests/test_canary_backend.py
git commit -m "feat(canary): CanaryASR and CanaryLID NeMo model classes"
```

---

## Task 6: Wire `canary` into `core.py` (build + routing)

**Files:**
- Modify: `whisperlivekit/core.py` (`_do_init()` around line 196–205; `online_factory()` around line 307–332)
- Test: `tests/test_canary_backend.py`

- [ ] **Step 1: Write the failing routing test (no NeMo needed)**

Append to `tests/test_canary_backend.py`:

```python
from argparse import Namespace


class _FakeCanaryASR:
    """Stands in for a loaded CanaryASR so online_factory needs no NeMo."""
    sep = " "

    def __init__(self):
        self.original_language = None
        self.lid_model = _StubLID()
        self.canary_default_lang = "en"
        self.confidence_validation = False
        self.tokenizer = None
        self.buffer_trimming = "segment"
        self.buffer_trimming_sec = 15.0

    def transcribe(self, audio, init_prompt=""):
        return "x"


def test_online_factory_routes_canary_to_localagreement():
    from whisperlivekit.canary_backend import CanarySessionASR
    from whisperlivekit.core import online_factory
    from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor

    args = Namespace(
        backend="canary", backend_policy="simulstreaming", lan="auto",
        canary_default_lang="en", canary_lid_min_sec=2.0, canary_lid_min_conf=0.5,
    )
    asr = _FakeCanaryASR()
    processor = online_factory(args, asr, language="auto")
    assert isinstance(processor, OnlineASRProcessor)
    assert isinstance(processor.asr, CanarySessionASR)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canary_backend.py::test_online_factory_routes_canary_to_localagreement -v`
Expected: FAIL — canary falls through to the SimulStreaming branch (`backend_policy == "simulstreaming"`), so `processor` is a `SimulStreamingOnlineProcessor`, not `OnlineASRProcessor` wrapping `CanarySessionASR`.

- [ ] **Step 3: Add the `_do_init()` build branch**

In `whisperlivekit/core.py`, inside `_do_init()`, add this branch alongside the other explicit-backend branches — after the `elif config.backend == "voxtral":` block and before `elif config.backend_policy == "simulstreaming":`:

```python
            elif config.backend == "canary":
                from whisperlivekit.canary_backend import CanaryASR, CanaryLID
                self.tokenizer = None
                self.asr = CanaryASR(
                    lan=config.lan,
                    canary_model=config.canary_model,
                    canary_default_lang=config.canary_default_lang,
                    buffer_trimming=config.buffer_trimming,
                    buffer_trimming_sec=config.buffer_trimming_sec,
                    confidence_validation=config.confidence_validation,
                )
                # Load the LID model so any session may request auto-detection.
                self.asr.lid_model = CanaryLID(lid_model=config.canary_lid_model)
                logger.info("Using LocalAgreement policy with Canary backend")
```

- [ ] **Step 4: Add the `online_factory()` routing branch**

In `whisperlivekit/core.py`, inside `online_factory()`, add this **before** the existing `if language is not None:` proxy-wrap block near the top of the function:

```python
    backend = getattr(args, 'backend', None)
    if backend == "canary":
        from whisperlivekit.canary_backend import CanarySessionASR
        effective = language if language is not None else getattr(args, 'lan', 'auto')
        wrapped = CanarySessionASR(
            asr,
            effective,
            lid=getattr(asr, 'lid_model', None),
            default_lang=getattr(args, 'canary_default_lang', 'en'),
            lid_min_sec=getattr(args, 'canary_lid_min_sec', 2.0),
            lid_min_conf=getattr(args, 'canary_lid_min_conf', 0.5),
        )
        return OnlineASRProcessor(wrapped)
```

Note: `OnlineASRProcessor` is already imported at the top of `core.py` (line 7). The existing `backend = getattr(args, 'backend', None)` line later in the function becomes redundant; leave the later per-backend `if` checks as-is (this early return handles canary first).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_canary_backend.py::test_online_factory_routes_canary_to_localagreement -v`
Expected: PASS.

- [ ] **Step 6: Run the full canary test file**

Run: `pytest tests/test_canary_backend.py -v`
Expected: PASS (integration test SKIPPED if NeMo absent; all others pass).

- [ ] **Step 7: Commit**

```bash
git add whisperlivekit/core.py tests/test_canary_backend.py
git commit -m "feat(canary): route canary backend to LocalAgreement in core"
```

---

## Task 7: Optional dependency extra, end-to-end test, and docs

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]` around line 38)
- Modify: `README.md` (backend list / usage section)
- Test: `tests/test_canary_backend.py` (gated end-to-end via `TestHarness`)

- [ ] **Step 1: Add the `canary` optional extra**

In `pyproject.toml` under `[project.optional-dependencies]`, add:

```toml
canary = ["nemo_toolkit[asr]>=2.5.0"]
```

(If 2.5 is not yet released at implementation time, pin the known-good main-branch
install per the spec's NeMo-version risk note and document it in the README.)

- [ ] **Step 2: Write the failing (gated) end-to-end test**

Append to `tests/test_canary_backend.py`:

```python
@requires_nemo
def test_canary_end_to_end_via_testharness():
    import asyncio

    from whisperlivekit import TestHarness

    async def _run():
        async with TestHarness(backend="canary", lan="en") as h:
            await h.feed("audio.wav", speed=1.0)   # replace with a real fixture path
            await h.drain(3.0)
            result = await h.finish()
            assert result.text.strip(), "expected non-empty transcription"

    asyncio.run(_run())
```

Note: use whatever real audio fixture the existing suite uses (see `test_pipeline.py`
/ `whisperlivekit/test_data.py`). `TestHarness` calls `TranscriptionEngine.reset()`
internally to switch backends — confirm against the harness API before finalizing.

- [ ] **Step 3: Run the end-to-end test**

Run: `pytest tests/test_canary_backend.py::test_canary_end_to_end_via_testharness -v`
Expected (no NeMo): SKIPPED. Expected (NeMo + CUDA + weights): PASS.

- [ ] **Step 4: Document the backend in the README**

Add `canary` to the backend list/table in `README.md` with a short usage note:

```markdown
- **canary** — NVIDIA Canary-1b-v2 (NeMo, CUDA). 25 European languages, native
  word timestamps, LocalAgreement streaming, auto language detection via AmberNet.
  Install: `pip install -e ".[canary]"`. Run: `wlk --backend canary --language auto`.
```

- [ ] **Step 5: Run the full suite for regressions**

Run: `pytest tests/ -v`
Expected: PASS (no regressions; canary integration tests SKIPPED without NeMo).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md tests/test_canary_backend.py
git commit -m "feat(canary): optional extra, end-to-end test, and docs"
```

---

## Self-Review Notes

- **Spec coverage:** backend + LocalAgreement routing (Tasks 5–6), native word/segment timestamps (Tasks 2, 5), explicit per-session language via proxy (Task 4), AmberNet detect-once-then-lock auto (Tasks 3–5), config/CLI surface (Task 1), optional dependency + lazy import (Tasks 2, 7), testing incl. gated E2E (Tasks 5, 7). Non-goals (AST translation, `canary-streaming`, re-detection) intentionally excluded.
- **Naming consistency:** helpers `canary_words_to_tokens` / `canary_segment_end_ts` / `map_voxlingua_to_canary`, classes `CanaryASR` / `CanaryLID` / `CanarySessionASR`, attribute `asr.lid_model`, config `canary_model` / `canary_default_lang` / `canary_lid_model` / `canary_lid_min_sec` / `canary_lid_min_conf` — used identically across tasks.
- **Known risks to validate during implementation** (from spec): exact NeMo timestamp-object shape (`res.timestamp['word']` vs a Hypothesis attribute), NeMo version for timestamps, `langid_ambernet` label codes vs `_VOXLINGUA_TO_CANARY` map, whether Canary exposes any prompt-conditioning (currently `init_prompt` accepted and ignored), and the real audio-fixture / `TestHarness.reset` API used by the existing suite.
```
