import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]

GPU_QUERY = "name,driver_version,compute_cap,memory.total,power.limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max"


def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True, cwd=REPO_ROOT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def sha256_of(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


TRAINING_CODE = ("airbench.py", "data.py", "train.py")


def code_fingerprint():
    return {
        "training_code_sha256": sha256_of([REPO_ROOT / "flipbench" / name for name in TRAINING_CODE]),
        "original_sha256": hashlib.sha256((REPO_ROOT / "original" / "airbench94.py").read_bytes()).hexdigest(),
    }


def gpu_state():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "memory_gib": round(props.total_memory / 2**30, 2),
        "multiprocessors": props.multi_processor_count,
        "driver": run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }


def collect():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_state(),
        "code": code_fingerprint(),
    }


def nvidia_smi_report():
    summary = run_command(["nvidia-smi"])
    query = run_command(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv"])
    if summary is None:
        return None
    lines = summary.splitlines()
    cut = next((i for i, line in enumerate(lines) if "Processes:" in line), len(lines) + 1)
    table = "\n".join(lines[: max(cut - 1, 0)]).rstrip()
    return table + "\n\n" + (query or "") + "\n"


def snapshot(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = collect()
    (out_dir / "environment.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    freeze = run_command([sys.executable, "-m", "pip", "freeze"])
    if freeze is not None:
        (out_dir / "pip-freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    report = nvidia_smi_report()
    if report is not None:
        (out_dir / "nvidia-smi.txt").write_text(report, encoding="utf-8")
    return info


def main():
    parser = argparse.ArgumentParser(description="Print or save a description of the software and hardware environment.")
    parser.add_argument("--out", type=Path, help="directory for environment.json, pip-freeze.txt and nvidia-smi.txt")
    args = parser.parse_args()
    info = snapshot(args.out) if args.out else collect()
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
