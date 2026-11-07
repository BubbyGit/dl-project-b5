import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from flipbench import envinfo
from flipbench.config import load_experiment
from flipbench.data import load_cifar10
from flipbench.train import train, warmup

REPO_ROOT = envinfo.REPO_ROOT


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_key(record):
    return (int(record["seed"]), float(record["epochs"]), str(record["flip"]), int(record["translate"]))


def utc_now():
    return datetime.now(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description="Run every (seed, epochs, translate, flip) combination of an experiment config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="output directory, defaults to results/<name>")
    parser.add_argument("--seed-count", type=int, help="use only the first N seeds of the config")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "cifar10"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    if args.seed_count:
        experiment = experiment.limit_seeds(args.seed_count)
    out_dir = (args.out or REPO_ROOT / "results" / experiment.name).resolve()
    runs_path = out_dir / "runs.jsonl"

    done = {run_key(record) for record in read_jsonl(runs_path)}
    planned = list(experiment.runs())
    pending = [spec for spec in planned if spec.key() not in done]
    print(f"{experiment.name}: {len(experiment.cells)} cells x {len(experiment.seeds)} seeds = {len(planned)} runs, {len(planned) - len(pending)} done, {len(pending)} pending")
    if args.dry_run or not pending:
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    out_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    session_id = started.strftime("%Y%m%dT%H%M%SZ")

    t0 = time.perf_counter()
    data = load_cifar10(args.data_root)
    torch.cuda.synchronize()
    data_load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    warmup(data)
    warmup_s = time.perf_counter() - t0

    append_jsonl(
        out_dir / "sessions.jsonl",
        {
            "session": session_id,
            "started_utc": started.isoformat(timespec="seconds"),
            "config_path": Path(args.config).as_posix(),
            "config": experiment.as_dict(),
            "pending_runs": len(pending),
            "data_load_s": data_load_s,
            "warmup_s": warmup_s,
            "environment": envinfo.collect(),
        },
    )

    elapsed = 0.0
    for index, spec in enumerate(pending, 1):
        result = train(data, spec)
        result.update(experiment=experiment.name, session=session_id, finished_utc=utc_now().isoformat(timespec="seconds"))
        append_jsonl(runs_path, result)
        elapsed += result["wall_time_s"]
        eta = timedelta(seconds=round(elapsed / index * (len(pending) - index)))
        print(
            f"[{index}/{len(pending)}] seed={spec.seed} epochs={spec.epochs:g} translate={spec.translate} flip={spec.flip:<11} "
            f"val_acc={result['val_acc']:.4f} tta_val_acc={result['tta_val_acc']:.4f} "
            f"time={result['wall_time_s']:.1f}s eta={eta}",
            flush=True,
        )


if __name__ == "__main__":
    main()
