import argparse
import json
import statistics
from pathlib import Path


def parse_stdout(text):
    header = None
    runs = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        row = dict(zip(header, cells))
        if row.get("run"):
            runs.append({"run": int(row["run"]), "epochs": []})
        if not runs:
            continue
        current = runs[-1]
        if row["epoch"] == "eval":
            current["tta_val_acc"] = float(row["tta_val_acc"])
            current["total_time_seconds"] = float(row["total_time_seconds"])
        else:
            epoch = {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
                "total_time_seconds": float(row["total_time_seconds"]),
            }
            current["epochs"].append(epoch)
            current["val_acc"] = epoch["val_acc"]
    return [run for run in runs if "tta_val_acc" in run]


def describe(values):
    values = list(values)
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def summarize(runs):
    times = [run["total_time_seconds"] for run in runs]
    return {
        "completed_runs": len(runs),
        "val_acc": describe(run["val_acc"] for run in runs),
        "tta_val_acc": describe(run["tta_val_acc"] for run in runs),
        "total_time_seconds": describe(times),
        "total_time_seconds_first_run": times[0] if times else None,
        "total_time_seconds_excluding_first_run": describe(times[1:]),
    }


def main():
    parser = argparse.ArgumentParser(description="Parse the stdout table printed by original/airbench94.py.")
    parser.add_argument("stdout", type=Path)
    args = parser.parse_args()
    runs = parse_stdout(args.stdout.read_text(encoding="utf-8"))
    print(json.dumps(summarize(runs), indent=2))


if __name__ == "__main__":
    main()
