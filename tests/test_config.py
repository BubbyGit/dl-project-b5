from pathlib import Path

import pytest

from flipbench.config import load_experiment
from flipbench.data import FlipMode

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_config_loads(path):
    experiment = load_experiment(path)
    assert experiment.name == path.stem
    assert experiment.seeds
    assert len(list(experiment.runs())) == len(experiment.cells) * len(experiment.seeds)


def test_table6_covers_the_reproduction_core():
    experiment = load_experiment(CONFIGS / "table6.yaml")
    assert set(experiment.flips) == set(FlipMode)
    assert experiment.epochs == (10.0, 20.0)
    assert experiment.translates == (2,)


def test_pilot_seeds_are_disjoint_from_confirmatory_seeds():
    pilot = set(load_experiment(CONFIGS / "pilot.yaml").seeds)
    for name in ("table6", "translate"):
        assert pilot.isdisjoint(load_experiment(CONFIGS / f"{name}.yaml").seeds)


def test_runs_are_ordered_by_seed():
    seeds = [spec.seed for spec in load_experiment(CONFIGS / "table6.yaml").runs()]
    assert seeds == sorted(seeds)


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\nflip: random\nepochs: 10\nseeds: [0]\nlr: 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_experiment(path)
