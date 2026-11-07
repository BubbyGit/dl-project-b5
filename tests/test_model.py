from math import ceil

import pytest
import torch

from flipbench.airbench import HYP, make_net
from flipbench.data import Cifar10, FlipMode, load_split
from flipbench.train import RunSpec, train


def test_hyperparameters_match_original(original):
    ours = {section: dict(values) for section, values in HYP.items()}
    theirs = {section: dict(values) for section, values in original.hyp.items()}
    assert ours["aug"].pop("flip") == "alternating"
    assert theirs["aug"].pop("flip") is True
    assert ours == theirs


def test_network_initialization_matches_original(original, cuda):
    torch.manual_seed(0)
    theirs = original.make_net().state_dict()
    torch.manual_seed(0)
    ours = make_net(HYP["net"]).state_dict()
    assert ours.keys() == theirs.keys()
    for key in ours:
        assert ours[key].dtype == theirs[key].dtype
        assert torch.equal(ours[key], theirs[key]), key


@pytest.mark.parametrize("flip", list(FlipMode))
def test_training_run_completes(cache_dir, cuda, flip):
    train_images, train_labels = load_split(cache_dir, True, cuda)
    test_images, test_labels = load_split(cache_dir, False, cuda)
    data = Cifar10(train_images, train_labels, test_images[:2000], test_labels[:2000])
    result = train(data, RunSpec(seed=0, epochs=2, flip=flip, translate=2))
    assert result["flip"] == str(flip)
    assert result["steps"] == ceil(len(train_images) // 1024 * 2)
    assert len(result["history"]) == 2
    assert 0.0 <= result["val_acc"] <= 1.0
    assert 0.0 <= result["tta_val_acc"] <= 1.0
    assert result["wall_time_s"] >= result["train_time_s"] > 0
