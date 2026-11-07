import pytest
import torch

from flipbench.data import CifarLoader, FlipMode, RngStreams, load_split

N_EPOCHS = 6


@pytest.fixture
def split(cache_dir, cuda):
    return load_split(cache_dir, True, cuda)


def make_loader(split, flip, seed=0, translate=0, shuffle=False):
    images, labels = split
    rng = RngStreams.from_seed(seed, images.device)
    return CifarLoader(images, labels, batch_size=1024, train=True, flip=flip, translate=translate, rng=rng, shuffle=shuffle, drop_last=False)


def epochs_of(loader, count):
    return [[(x.clone(), y.clone()) for x, y in loader] for _ in range(count)]


def orientations(loader, count):
    base = loader.normalize(loader.images)
    result = []
    for batches in epochs_of(loader, count):
        images = torch.cat([x for x, _ in batches])
        flipped = (images == base.flip(-1)).flatten(1).all(1)
        straight = (images == base).flatten(1).all(1)
        assert torch.all(flipped ^ straight)
        result.append(flipped)
    return torch.stack(result)


def assert_same_batches(left, right):
    assert len(left) == len(right)
    for epoch_left, epoch_right in zip(left, right):
        assert len(epoch_left) == len(epoch_right)
        for (x1, y1), (x2, y2) in zip(epoch_left, epoch_right):
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)


def test_none_never_flips(split):
    assert not orientations(make_loader(split, FlipMode.NONE), N_EPOCHS).any()


def test_alternating_flips_every_image_every_epoch(split):
    o = orientations(make_loader(split, FlipMode.ALTERNATING), N_EPOCHS)
    assert torch.all(o[1:] != o[:-1])
    assert abs(o[0].float().mean().item() - 0.5) < 0.05


def test_random_flips_independently_every_epoch(split):
    o = orientations(make_loader(split, FlipMode.RANDOM), N_EPOCHS).float()
    assert torch.all((o.mean(1) - 0.5).abs() < 0.05)
    agreement = (o[1:] == o[:-1]).float().mean(1)
    assert torch.all((agreement - 0.5).abs() < 0.05)


def test_random_and_alternating_share_the_first_epoch(split):
    random = orientations(make_loader(split, FlipMode.RANDOM), 2)
    alternating = orientations(make_loader(split, FlipMode.ALTERNATING), 2)
    assert torch.equal(random[0], alternating[0])
    assert not torch.equal(random[1], alternating[1])


def test_order_and_translation_do_not_depend_on_flip_mode(split):
    runs = {flip: epochs_of(make_loader(split, flip, translate=2, shuffle=True), 3) for flip in FlipMode}
    labels = {flip: [torch.cat([y for _, y in epoch]) for epoch in epochs] for flip, epochs in runs.items()}
    for flip in FlipMode:
        for a, b in zip(labels[flip], labels[FlipMode.NONE]):
            assert torch.equal(a, b)
    first_random = torch.cat([x for x, _ in runs[FlipMode.RANDOM][0]])
    first_alternating = torch.cat([x for x, _ in runs[FlipMode.ALTERNATING][0]])
    assert torch.equal(first_random, first_alternating)


def test_same_seed_reproduces_batches(split):
    for flip in FlipMode:
        first = epochs_of(make_loader(split, flip, seed=7, translate=2, shuffle=True), 3)
        second = epochs_of(make_loader(split, flip, seed=7, translate=2, shuffle=True), 3)
        assert_same_batches(first, second)


def test_different_seeds_give_different_batches(split):
    a = epochs_of(make_loader(split, FlipMode.RANDOM, seed=1, translate=2, shuffle=True), 1)
    b = epochs_of(make_loader(split, FlipMode.RANDOM, seed=2, translate=2, shuffle=True), 1)
    assert not torch.equal(a[0][0][0], b[0][0][0])


@pytest.mark.parametrize(
    ("aug", "flip", "translate"),
    [
        ({"flip": True, "translate": 2}, FlipMode.ALTERNATING, 2),
        ({"flip": True, "translate": 0}, FlipMode.ALTERNATING, 0),
        ({"flip": False, "translate": 2}, FlipMode.NONE, 2),
        ({"flip": False, "translate": 0}, FlipMode.NONE, 0),
    ],
)
def test_matches_original_loader(original, cache_dir, cuda, aug, flip, translate):
    torch.manual_seed(2024)
    theirs = original.CifarLoader(str(cache_dir), train=True, batch_size=1024, aug=aug)
    expected = epochs_of(theirs, 4)

    images, labels = load_split(cache_dir, True, cuda)
    torch.manual_seed(2024)
    ours = CifarLoader(images, labels, batch_size=1024, train=True, flip=flip, translate=translate)
    assert_same_batches(epochs_of(ours, 4), expected)


def test_test_loader_matches_original(original, cache_dir, cuda):
    theirs = original.CifarLoader(str(cache_dir), train=False, batch_size=2000)
    images, labels = load_split(cache_dir, False, cuda)
    ours = CifarLoader(images, labels, batch_size=2000, train=False)
    assert torch.equal(ours.normalize(ours.images), theirs.normalize(theirs.images))
    assert_same_batches(epochs_of(ours, 1), epochs_of(theirs, 1))
