import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from flipbench import envinfo
from flipbench.reference import TABLE6
from flipbench.stats import describe, minimum_detectable_effect, required_n
from flipbench.sweep import read_jsonl

REPO_ROOT = envinfo.REPO_ROOT
CORE_EPOCHS = (10.0, 20.0)
CORE_FLIPS = ("none", "random", "alternating")
RHOS = (0.25, 0.5)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def render(runs, sessions, original, seed_counts):
    timing = defaultdict(list)
    for run in runs:
        timing[(float(run["epochs"]), run["flip"])].append(run["wall_time_s"])
    timing = {cell: describe(times) for cell, times in timing.items()}

    epochs = np.array([run["epochs"] for run in runs], dtype=float)
    walls = np.array([run["wall_time_s"] for run in runs], dtype=float)
    slope, intercept = np.polyfit(epochs, walls, 1)
    overhead = float(np.mean([s["data_load_s"] + s["warmup_s"] for s in sessions])) if sessions else 0.0
    gpu = (sessions[0]["environment"]["gpu"] or {}).get("name") if sessions else None

    def run_seconds(e, flip):
        cell = timing.get((float(e), flip))
        return cell.mean if cell else intercept + slope * e

    per_seed = sum(run_seconds(e, flip) for e in CORE_EPOCHS for flip in CORE_FLIPS)
    order = {flip: i for i, flip in enumerate(CORE_FLIPS)}

    lines = [f"GPU: {gpu}", f"pilot runs: {len(runs)}", f"sessions: {len(sessions)}", "", "wall time per run, s"]
    lines += table(
        ["epochs", "flip", "n", "mean", "std"],
        [[f"{e:g}", flip, s.n, f"{s.mean:.2f}", f"{s.std:.2f}"] for (e, flip), s in sorted(timing.items(), key=lambda item: (item[0][0], order.get(item[0][1], 99)))],
    )
    lines += [
        "",
        f"linear fit, s: {intercept:.2f} + {slope:.3f} * epochs",
        f"session overhead (data load + warm-up), s: {overhead:.1f}",
        "",
        f"table6 core: {len(CORE_FLIPS)} flip modes x {len(CORE_EPOCHS)} epoch settings, s per seed: {per_seed:.1f}",
    ]
    lines += table(
        ["seeds per config", "runs", "minutes", "hours"],
        [[n, 6 * n, f"{(n * per_seed + overhead) / 60:.1f}", f"{(n * per_seed + overhead) / 3600:.2f}"] for n in seed_counts],
    )
    lines += ["", "per configuration, minutes"]
    lines += table(
        ["epochs", "flip"] + [f"n={n}" for n in seed_counts],
        [[f"{e:g}", flip] + [f"{n * run_seconds(e, flip) / 60:.1f}" for n in seed_counts] for e in CORE_EPOCHS for flip in CORE_FLIPS],
    )

    if original:
        sigma = {False: 100 * original["summary"]["val_acc"]["std"], True: 100 * original["summary"]["tta_val_acc"]["std"]}
        lines += [
            "",
            f"sigma from original run (n={original['summary']['completed_runs']}, 9.9 epochs, alternating), pp: "
            f"without TTA {sigma[False]:.3f}, with TTA {sigma[True]:.3f}",
            "required seeds per mode, two-sided t-test, alpha 0.05, power 0.8, delta from paper Table 6",
        ]
        rows = []
        for tta in (False, True):
            for e in CORE_EPOCHS:
                row = TABLE6[(int(e), tta)]
                for a, b in (("alternating", "random"), ("random", "none")):
                    delta = row[a] - row[b]
                    n_unpaired = required_n(delta, sigma[tta])
                    rows.append(
                        [f"{e:g}", "yes" if tta else "no", f"{a} - {b}", f"{delta:.3f}", n_unpaired]
                        + [required_n(delta, sigma[tta], paired=True, rho=rho) for rho in RHOS]
                        + [f"{(n_unpaired * per_seed + overhead) / 3600:.2f}"]
                    )
        lines += table(
            ["epochs", "TTA", "contrast", "delta, pp", "n unpaired"] + [f"n paired rho={rho}" for rho in RHOS] + ["core hours at n unpaired"],
            rows,
        )
        lines += ["", "minimum detectable effect, pp, unpaired, alpha 0.05, power 0.8"]
        lines += table(
            ["seeds per mode", "without TTA", "with TTA"],
            [[n, f"{minimum_detectable_effect(sigma[False], n):.3f}", f"{minimum_detectable_effect(sigma[True], n):.3f}"] for n in seed_counts],
        )

    per_seed_translate = 9 * run_seconds(10.0, "random")
    lines += ["", "translate {0, 2, 4} x 3 flip modes x 10 epochs, assuming the cost of translate=2"]
    lines += table(
        ["seeds per config", "runs", "hours"],
        [[n, 9 * n, f"{(n * per_seed_translate + overhead) / 3600:.2f}"] for n in seed_counts],
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Estimate GPU time and required number of seeds from pilot runs.")
    parser.add_argument("--pilot", type=Path, default=REPO_ROOT / "results" / "checkpoint1" / "pilot")
    parser.add_argument("--original", type=Path, default=REPO_ROOT / "results" / "checkpoint1" / "original" / "run.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 10, 20, 50, 100])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    runs = read_jsonl(args.pilot / "runs.jsonl")
    if not runs:
        raise SystemExit(f"no pilot runs in {args.pilot}")
    sessions = read_jsonl(args.pilot / "sessions.jsonl")
    original = json.loads(args.original.read_text(encoding="utf-8")) if args.original.exists() else None
    text = render(runs, sessions, original, args.seeds)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
