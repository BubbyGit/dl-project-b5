import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

from flipbench.reference import PAPER, paper_value
from flipbench.stats import compare_paired, describe
from flipbench.sweep import read_jsonl

FLIP_ORDER = ("none", "random", "alternating")
CONTRASTS = (("alternating", "random"), ("random", "none"), ("alternating", "none"))
METRICS = (("val_acc", False, "Accuracy without TTA"), ("tta_val_acc", True, "Accuracy with TTA"))


def fmt(value, digits=3):
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def fmt_p(value):
    if value is None or math.isnan(value):
        return "—"
    return f"{value:.1e}" if value < 1e-3 else f"{value:.3f}"


def group_runs(runs):
    grouped = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        grouped[(float(run["epochs"]), int(run["translate"]))][run["flip"]][int(run["seed"])] = run
    return grouped


def cell_tables(epochs, translate, by_flip):
    lines = []
    for metric, tta, label in METRICS:
        lines += [
            f"### {label}",
            "",
            "| flip | n | mean, % | std, pp | 95% CI, pp | paper, % | ours − paper, pp |",
            "|---|---|---|---|---|---|---|",
        ]
        for flip in FLIP_ORDER:
            if flip not in by_flip:
                continue
            summary = describe(100 * run[metric] for run in by_flip[flip].values())
            reference = paper_value(epochs, tta, flip) if translate == 2 else None
            delta = None if reference is None else summary.mean - reference
            lines.append(
                f"| {flip} | {summary.n} | {fmt(summary.mean)} | {fmt(summary.std)} | ±{fmt(summary.ci95)} | {fmt(reference, 2)} | {fmt(delta)} |"
            )
        lines += [
            "",
            "| contrast | pairs | mean diff, pp | 95% CI, pp | p (paired t) | p (Welch) | corr |",
            "|---|---|---|---|---|---|---|",
        ]
        for a, b in CONTRASTS:
            if a not in by_flip or b not in by_flip:
                continue
            comparison = compare_paired(
                {seed: 100 * run[metric] for seed, run in by_flip[a].items()},
                {seed: 100 * run[metric] for seed, run in by_flip[b].items()},
            )
            lines.append(
                f"| {a} − {b} | {comparison.n} | {fmt(comparison.diff)} | [{fmt(comparison.ci_low)}, {fmt(comparison.ci_high)}] "
                f"| {fmt_p(comparison.p_paired)} | {fmt_p(comparison.p_welch)} | {fmt(comparison.correlation, 2)} |"
            )
        lines.append("")
    times = [run["wall_time_s"] for runs in by_flip.values() for run in runs.values()]
    lines += [f"mean wall time per run, s: {fmt(sum(times) / len(times), 1)} (n={len(times)})", ""]
    return lines


def render(runs, title):
    lines = [f"# {title}", "", f"reference: {PAPER}", ""]
    for (epochs, translate), by_flip in sorted(group_runs(runs).items()):
        lines += [f"## epochs = {epochs:g}, translate = {translate}", ""]
        lines += cell_tables(epochs, translate, by_flip)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize runs.jsonl: per-mode accuracy and seed-paired contrasts.")
    parser.add_argument("results", type=Path, help="experiment directory or runs.jsonl")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    path = args.results / "runs.jsonl" if args.results.is_dir() else args.results
    runs = read_jsonl(path)
    if not runs:
        raise SystemExit(f"no runs in {path}")
    text = render(runs, title=runs[0].get("experiment", path.parent.name))
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
