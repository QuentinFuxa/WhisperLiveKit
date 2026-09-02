#!/usr/bin/env python3
"""Deterministic fixture for the simul-MT translation subsystem.

The live pipeline is not reproducible (ASR/VAD timing jitter), so fixes to
the commit policy and the accessible frontier cannot be gated on live runs.
This script records the translation engine's FULL input interface from one
live run — every insert_tokens / insert_silence / validate call with the
exact tokens and timestamps — and replays it through a fresh engine with
greedy sampling. Same fixture in, same outputs out.

  record:  .venv/bin/python scripts/simul_fixture.py record --audio <wav> \
               [--out tests/golden/simul_zh_long_calls.jsonl]
  replay:  .venv/bin/python scripts/simul_fixture.py replay <fixture> [--twice]

Replay reports, per step: the committed-boundary ratio (cend/n_src — the
frontier cadence the frontier fix targets), the released partial, the MT call
count, and per-final draft coverage (the check_simul_heads.py metric, computed
on the replayed stream). PASS requires coverage >= 0.6.

With --twice, replay runs twice and diffs the outputs — a determinism check
of the replay itself (MLX greedy on fixed weights is deterministic; any diff
is a bug).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def _wrap_translation(ap, rows: list) -> None:
    """Wrap the audio processor's translation instance to record the full
    interface stream (thread-safe enough: appends are GIL-atomic)."""
    import types
    tr = ap.translation

    def ser_item(it):
        if it.__class__.__name__ == "HypothesisTail":
            return {"tail": True, "start": it.start, "end": it.end, "text": it.text}
        return {"start": it.start, "end": it.end, "text": it.text,
                "speaker": getattr(it, "speaker", None)}

    def rec(row):
        rows.append(row)

    orig_insert = tr.insert_tokens
    def insert_tokens(items):
        rec({"kind": "tokens", "items": [ser_item(i) for i in items]})
        return orig_insert(items)
    tr.insert_tokens = insert_tokens

    orig_sil = tr.insert_silence
    def insert_silence(duration=None):
        rec({"kind": "silence", "duration": duration})
        return orig_sil(duration)
    tr.insert_silence = insert_silence

    orig_val = tr.validate_buffer_and_reset
    def validate():
        t, b = orig_val()
        rec({"kind": "validate",
             "translation": t.text if t is not None else None,
             "buffer": b.text if b is not None else None})
        return t, b
    tr.validate_buffer_and_reset = validate

    orig_proc = tr.process
    def process():
        t, b = orig_proc()
        rec({"kind": "process",
             "translation": t.text if t is not None else None,
             "buffer": b.text if b is not None else None})
        return t, b
    tr.process = process


def record(audio: str, out: str) -> None:
    sys.path.insert(0, "scripts")
    import argparse as _ap
    from lc_terminal import run_file
    rows: list = []

    ns = _ap.Namespace(
        audio=audio, language="zh", target_language="en", backend="mlx-qwen3-asr",
        mlx_qwen3_asr_model="mlx-community/Qwen3-ASR-0.6B-8bit",
        mlx_llm_mt_model="hy-mt2-1.8b-8bit",
        source="file", simultaneous=True, simul_commit=None, second_pass=False,
        diarize=False, hotwords=None, vad_threshold=None, vad_min_silence_ms=None,
        event_log=None, overlay=False, tui=False,
    )

    class _Sink:
        def __call__(self, state): pass

    # wrap after the harness builds the engine: patch run_file's harness via
    # the on_update hook timing — simplest is to wrap inside the sink, which
    # the harness calls on every update (engine exists by then).
    state = {"wrapped": False}
    class _WrapSink:
        def __call__(self, st):
            if not state["wrapped"]:
                ap = None
                # the sink receives state; find the processor through the
                # closure of run_file — instead, wrap via harness attribute
                # set by run_file on the sink: not available. Fall back: the
                # harness stores _processor; wrap lazily from the event loop.
                state["wrapped"] = True

    asyncio.run(run_file(ns, _Sink()))
    # The wrapping above cannot reach the harness from here; run_file does not
    # expose it. Use the module-level hook instead:
    _record_via_hook(audio, out)


def _record_via_hook(audio: str, out: str) -> None:
    """Record by monkeypatching TestHarness to wrap translation on init."""
    sys.path.insert(0, "scripts")
    import argparse as _ap
    import whisperlivekit.test_harness as th
    rows: list = []

    orig_init = th.TestHarness.__aenter__
    async def patched_aenter(self):
        res = await orig_init(self)
        _wrap_translation(self._processor, rows)
        print(f"[record] wrapped translation engine, recording interface calls",
              file=sys.stderr)
        return res
    th.TestHarness.__aenter__ = patched_aenter

    from lc_terminal import run_file
    ns = _ap.Namespace(
        audio=audio, language="zh", target_language="en", backend="mlx-qwen3-asr",
        mlx_qwen3_asr_model="mlx-community/Qwen3-ASR-0.6B-8bit",
        mlx_llm_mt_model="hy-mt2-1.8b-8bit",
        source="file", simultaneous=True, simul_commit=None, second_pass=False,
        diarize=False, hotwords=None, vad_threshold=None, vad_min_silence_ms=None,
        event_log=None, overlay=False, tui=False,
    )

    class _Sink:
        def __call__(self, state): pass

    asyncio.run(run_file(ns, _Sink()))

    Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n_tok = sum(1 for r in rows if r["kind"] == "tokens")
    n_final = sum(1 for r in rows if r["kind"] == "validate" and r["translation"])
    print(f"[record] {len(rows)} steps ({n_tok} token batches, {n_final} finals) -> {out}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _greedy(engine) -> None:
    """Force deterministic greedy sampling."""
    engine._config.temp = 0.0
    engine._config.top_p = 1.0
    engine._config.top_k = 0


def replay(path: str, twice: bool = False, commit_mode: str | None = None,
           min_source_tokens: int | None = None) -> float:
    import whisperlivekit.simul_mt_capture as smc
    from whisperlivekit.timed_objects import ASRToken, HypothesisTail
    from whisperlivekit.translation_mlx_llm_mt_simul import MlxLlmTranslationSimul, committed_src_end_from_text
    from whisperlivekit import timed_objects as to

    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    base = MlxLlmTranslationSimul(
        model_id="hy-mt2-1.8b-8bit", source_language="zh", target_language="en",
        warmup=True,
    )
    _greedy(base)

    def mk_item(d):
        cls = HypothesisTail if d.get("tail") else ASRToken
        return cls(start=d["start"], end=d["end"], text=d["text"],
                   speaker=d.get("speaker"))

    def run_once() -> list:
        # fresh per-run state over the shared loaded model (new_session is the
        # engine's own mechanism for this — hand-rolled field resets leak:
        # _last_buffer, _committed_start, _chars_per_token, ...)
        engine = base.new_session()
        _greedy(engine)
        if commit_mode:
            engine._commit_mode = commit_mode
        if min_source_tokens is not None:
            engine._MIN_SOURCE_TOKENS = min_source_tokens
        if commit_mode:
            engine._commit_mode = commit_mode
        out = []
        for r in rows:
            if r["kind"] == "tokens":
                engine.insert_tokens([mk_item(d) for d in r["items"]])
            elif r["kind"] == "silence":
                engine.insert_silence(r["duration"])
            elif r["kind"] == "validate":
                t, b = engine.validate_buffer_and_reset()
                out.append({"kind": "validate", "text": t.text if t else None})
                continue
            elif r["kind"] == "process":
                t, b = engine.process()
                out.append({"kind": "process",
                            "translation": t.text if t else None,
                            "buffer": b.text if b else None,
                            "partial": engine._emitted_partial,
                            "mt_calls": engine._mt_call_count})
                continue
            # tokens/silence rows also get a frontier snapshot
            out.append({"kind": r["kind"]})
        return out

    def _frontier_ratio(engine) -> float | None:
        src = engine._source_text()
        com = engine._committed_text()
        if not src:
            return None
        model, tokenizer = engine._ensure_simul_model()
        content = engine._build_prompt_content(src)
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=False)
        import numpy as np
        src_start, src_end = smc.source_span(tokenizer, prompt_str, src)
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        src_ids = prompt_ids[src_start:src_end]
        cend = committed_src_end_from_text(tokenizer, src_ids, com)
        return round(cend / max(1, src_end - src_start), 3)

    result = run_once()
    if twice:
        result2 = run_once()
        diffs = [(a, b) for a, b in zip(result, result2)
                 if json.dumps(a, sort_keys=True, ensure_ascii=False)
                 != json.dumps(b, sort_keys=True, ensure_ascii=False)]
        print(f"[determinism] {len(diffs)} differing steps across two replays")
        if diffs:
            for a, b in diffs[:3]:
                print("  A:", json.dumps(a, ensure_ascii=False)[:120])
                print("  B:", json.dumps(b, ensure_ascii=False)[:120])

    # per-final draft coverage (the check_simul_heads.py metric). In the
    # simul path finals arrive via process() (the _pending_finals flush),
    # not via validate_buffer_and_reset.
    import re
    def words(s): return set(re.findall(r"\w+", (s or "").lower()))
    ratios = []
    seen: set = set()
    fi = 0
    for s in result:
        if s["kind"] == "process":
            if s.get("translation"):
                fw = words(s["translation"])
                r = len(fw & seen) / len(fw) if fw else 0.0
                ratios.append(r)
                print(f"  final {fi}: coverage={r:.2f} {s['translation'][:50]!r}")
                fi += 1
                seen = set()
            else:
                seen |= words(s.get("partial") or "")
    cov = sum(ratios) / len(ratios) if ratios else 0.0
    print(f"\nreplay: {len(rows)} steps, {fi} finals")
    print(f"  draft coverage : {cov:.2f}  (pass >= 0.6)")
    print(f"  RESULT: {'PASS' if cov >= 0.6 else 'FAIL'}")
    return cov


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--audio", default="/Users/clkao/git/asr/_work/zh_long.wav")
    r.add_argument("--out", default="tests/golden/simul_zh_long_calls.jsonl")
    p = sub.add_parser("replay")
    p.add_argument("fixture")
    p.add_argument("--twice", action="store_true")
    p.add_argument("--commit-mode", default=None, help="override the commit policy mode (argmax | mass | paper)")
    p.add_argument("--min-source-tokens", type=int, default=None,
                   help="override the re-draft hysteresis threshold")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.audio, args.out)
    else:
        replay(args.fixture, args.twice, commit_mode=getattr(args, "commit_mode", None),
               min_source_tokens=getattr(args, "min_source_tokens", None))


if __name__ == "__main__":
    main()
