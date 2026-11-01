#!/usr/bin/env python3
"""Plot the retained M5 screening summary, without modifying historical figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BACKENDS = {
    "whisper-small-la": ("Whisper small / LA", "#222222", "o"),
    "qwen-metal": ("Qwen3 / Metal", "#888888", "s"),
    "qwen-native": ("Qwen3 / native MLX", "#2867b2", "D"),
    "nemotron-native": ("Nemotron / native MLX", "#bd6026", "^"),
}
LANGUAGES = {"en": "English · WER", "fr": "French · WER", "zh": "Chinese · CER"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    required = {f"{backend}-{language}" for backend in BACKENDS for language in LANGUAGES}
    if missing := required - data["reports"].keys():
        parser.error("Missing reports: " + ", ".join(sorted(missing)))
    destination = args.output.resolve()
    if destination.suffix != ".png" or destination.exists():
        parser.error("Choose a new .png file; existing figures are never overwritten")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    omitted = []
    for column, (language, title) in enumerate(LANGUAGES.items()):
        for row, (field, axis_label) in enumerate([
            ("first_visible_time_s_p95", "First visible text p95 (s)"),
            ("finalization_time_s_p95", "EOF finalization p95 (s)"),
        ]):
            ax = axes[row, column]
            for backend, (label, color, marker) in BACKENDS.items():
                key = f"{backend}-{language}"
                report = data["reports"][key]
                record = report["summary"]
                complete = not report["readiness_issues"]
                if not complete or record["quality"] is None or record[field] is None:
                    omitted.append(key)
                    continue
                ax.scatter(record[field], record["quality"] * 100, c=color, marker=marker, s=65)
            ax.set(xlabel=axis_label, ylabel="Error rate (%)", xlim=(0, None), ylim=(0, None))
            ax.grid(alpha=.2)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(title)
    handles = [Line2D([], [], marker=marker, color=color, linestyle="", markersize=7, label=label)
               for label, color, marker in BACKENDS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, .07), frameon=False)
    fig.suptitle("M5 · paced streaming · 30 FLEURS clips per language × 3 passes", fontsize=14)
    note = "Pooled clip p95 and reference-weighted errors. First visible text may be provisional. Per-pass values and limits: METHODS.md / summary.json."
    if omitted:
        note += "\nIncomplete measurements omitted: " + ", ".join(sorted(set(omitted)))
    fig.text(.5, .025, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
