import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return "cuda"


@pytest.fixture(scope="session")
def original():
    spec = importlib.util.spec_from_file_location("airbench94_original", REPO_ROOT / "original" / "airbench94.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_dir(tmp_path):
    images = torch.randint(0, 256, (4096, 32, 32, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    labels = torch.randint(0, 10, (4096,), generator=torch.Generator().manual_seed(1))
    for name in ("train.pt", "test.pt"):
        torch.save({"images": images, "labels": labels, "classes": [str(i) for i in range(10)]}, tmp_path / name)
    return tmp_path
