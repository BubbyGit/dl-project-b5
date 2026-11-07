import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from enum import StrEnum
from math import ceil

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


class FlipMode(StrEnum):
    NONE = "none"
    RANDOM = "random"
    ALTERNATING = "alternating"


def derive_seed(seed, stream):
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class RngStreams:
    order: torch.Generator | None = None
    translate: torch.Generator | None = None
    flip: torch.Generator | None = None

    @classmethod
    def from_seed(cls, seed, device):
        def generator(stream):
            gen = torch.Generator(device=device)
            gen.manual_seed(derive_seed(seed, stream))
            return gen

        return cls(order=generator("order"), translate=generator("translate"), flip=generator("flip"))


@dataclass(frozen=True)
class Cifar10:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def ensure_cached(root, train):
    path = os.path.join(root, "train.pt" if train else "test.pt")
    if not os.path.exists(path):
        dset = torchvision.datasets.CIFAR10(root, download=True, train=train)
        images = torch.tensor(dset.data)
        labels = torch.tensor(dset.targets)
        torch.save({"images": images, "labels": labels, "classes": dset.classes}, path)
    return path


def load_split(root, train, device):
    data = torch.load(ensure_cached(root, train), map_location=torch.device(device))
    images = (data["images"].half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
    return images, data["labels"]


def load_cifar10(root="cifar10", device="cuda"):
    train_images, train_labels = load_split(root, True, device)
    test_images, test_labels = load_split(root, False, device)
    return Cifar10(train_images, train_labels, test_images, test_labels)


def batch_flip_lr(inputs, generator=None):
    flip_mask = (torch.rand(len(inputs), device=inputs.device, generator=generator) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size, generator=None):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device, generator=generator)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r + sy:r + sy + crop_size, r + sx:r + sx + crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype)
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s:r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s:r + s + crop_size]
    return images_out


class CifarLoader:
    def __init__(self, images, labels, batch_size, train, flip=FlipMode.NONE, translate=0, rng=None, drop_last=None, shuffle=None):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.flip = FlipMode(flip)
        self.translate = translate
        self.rng = rng or RngStreams()
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0

    def __len__(self):
        return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        flipping = self.flip is not FlipMode.NONE

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if flipping:
                images = self.proc_images["flip"] = batch_flip_lr(images, self.rng.flip)
            if self.translate > 0:
                self.proc_images["pad"] = F.pad(images, (self.translate,) * 4, "reflect")

        if self.translate > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2], self.rng.translate)
        elif flipping:
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.flip is FlipMode.ALTERNATING and self.epoch % 2 == 1:
            images = images.flip(-1)
        elif self.flip is FlipMode.RANDOM and self.epoch > 0:
            images = batch_flip_lr(images, self.rng.flip)

        self.epoch += 1

        if self.shuffle:
            indices = torch.randperm(len(images), device=images.device, generator=self.rng.order)
        else:
            indices = torch.arange(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield images[idxs], self.labels[idxs]


def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-10, convert it to the airbench tensor cache and report timings.")
    parser.add_argument("--root", default="cifar10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    t0 = time.perf_counter()
    for train in (True, False):
        ensure_cached(args.root, train)
    t1 = time.perf_counter()
    data = load_cifar10(args.root, args.device)
    if data.train_images.is_cuda:
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"download_and_convert_seconds={t1 - t0:.2f}")
    print(f"load_to_{args.device}_seconds={t2 - t1:.2f}")
    print(f"train={tuple(data.train_images.shape)} test={tuple(data.test_images.shape)}")


if __name__ == "__main__":
    main()
