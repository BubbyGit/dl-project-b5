import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from flipbench import envinfo
from flipbench.original_log import parse_stdout, summarize

REPO_ROOT = envinfo.REPO_ROOT
SCRIPT = Path("original") / "airbench94.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream(cmd, log_path):
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def archive_log_pt(stdout_text, out_dir):
    printed = [line.strip() for line in stdout_text.splitlines() if line.strip().endswith("log.pt")]
    if not printed:
        return None
    source = Path(printed[-1])
    if not source.exists():
        return None
    target = out_dir / "log.pt"
    shutil.copy2(source, target)
    return {"source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source), "archived": target.name}


def main():
    parser = argparse.ArgumentParser(description="Run original/airbench94.py without modifications and archive its output.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "checkpoint1" / "original")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = envinfo.snapshot(out_dir / "environment")
    data_cached = (REPO_ROOT / "cifar10" / "train.pt").exists() and (REPO_ROOT / "cifar10" / "test.pt").exists()

    cmd = [sys.executable, "-u", SCRIPT.as_posix()]
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    returncode = stream(cmd, out_dir / "stdout.txt")
    wall_clock = time.perf_counter() - t0
    finished = datetime.now(timezone.utc)

    stdout_text = (out_dir / "stdout.txt").read_text(encoding="utf-8")
    runs = parse_stdout(stdout_text)
    record = {
        "command": f"python -u {SCRIPT.as_posix()}",
        "script_sha256": sha256(REPO_ROOT / SCRIPT),
        "returncode": returncode,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": finished.isoformat(timespec="seconds"),
        "wall_clock_seconds": round(wall_clock, 2),
        "dataset_cached_before_run": data_cached,
        "log_pt": archive_log_pt(stdout_text, out_dir),
        "environment": env,
        "summary": summarize(runs),
        "runs": runs,
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("returncode", "wall_clock_seconds", "summary")}, indent=2))
    return returncode


if __name__ == "__main__":
    sys.exit(main())
