from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import yaml

from flipbench.data import FlipMode
from flipbench.train import RunSpec

KEYS = {"name", "flip", "epochs", "translate", "seeds"}


def as_tuple(value):
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def parse_seeds(raw):
    if isinstance(raw, dict):
        unknown = set(raw) - {"start", "count"}
        if unknown:
            raise ValueError(f"unknown seed keys: {sorted(unknown)}")
        return tuple(range(int(raw["start"]), int(raw["start"]) + int(raw["count"])))
    return tuple(int(seed) for seed in as_tuple(raw))


@dataclass(frozen=True)
class Experiment:
    name: str
    flips: tuple[FlipMode, ...]
    epochs: tuple[float, ...]
    translates: tuple[int, ...]
    seeds: tuple[int, ...]

    @property
    def cells(self):
        return list(product(self.epochs, self.translates, self.flips))

    def runs(self):
        for seed in self.seeds:
            for epochs, translate, flip in self.cells:
                yield RunSpec(seed=seed, epochs=epochs, flip=flip, translate=translate)

    def limit_seeds(self, count):
        if count > len(self.seeds):
            raise ValueError(f"{self.name} defines only {len(self.seeds)} seeds")
        return replace(self, seeds=self.seeds[:count])

    def as_dict(self):
        return {
            "name": self.name,
            "flip": [str(flip) for flip in self.flips],
            "epochs": list(self.epochs),
            "translate": list(self.translates),
            "seeds": list(self.seeds),
        }


def load_experiment(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    seeds = parse_seeds(raw["seeds"])
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate seeds")
    return Experiment(
        name=raw["name"],
        flips=tuple(FlipMode(flip) for flip in as_tuple(raw["flip"])),
        epochs=tuple(float(epochs) for epochs in as_tuple(raw["epochs"])),
        translates=tuple(int(translate) for translate in as_tuple(raw.get("translate", 2))),
        seeds=seeds,
    )
