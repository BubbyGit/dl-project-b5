import time
from dataclasses import dataclass
from math import ceil

import torch
from torch import nn

from flipbench.airbench import LookaheadState, default_hyp, evaluate, init_whitening_conv, make_net
from flipbench.data import CifarLoader, FlipMode, RngStreams, derive_seed

torch.backends.cudnn.benchmark = True


@dataclass(frozen=True)
class RunSpec:
    seed: int
    epochs: float
    flip: FlipMode
    translate: int = 2

    def key(self):
        return (self.seed, float(self.epochs), str(self.flip), int(self.translate))


class CudaTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.seconds = 0.0

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        torch.cuda.synchronize()
        self.seconds += 1e-3 * self.start.elapsed_time(self.end)


def train(data, spec, hyp=None):
    hyp = hyp or default_hyp()
    opt = hyp["opt"]
    wall_start = time.perf_counter()

    batch_size = opt["batch_size"]
    epochs = spec.epochs
    momentum = opt["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = opt["lr"] / kilostep_scale
    wd = opt["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * opt["bias_scaler"]

    device = data.train_images.device
    rng = RngStreams.from_seed(spec.seed, device)
    torch.manual_seed(derive_seed(spec.seed, "init"))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=opt["label_smoothing"], reduction="none")
    test_loader = CifarLoader(data.test_images, data.test_labels, batch_size=2000, train=False)
    train_loader = CifarLoader(
        data.train_images,
        data.train_labels,
        batch_size=batch_size,
        train=True,
        flip=spec.flip,
        translate=spec.translate,
        rng=rng,
    )
    total_train_steps = ceil(len(train_loader) * epochs)

    model = make_net(hyp["net"])
    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if "norm" not in k and p.requires_grad]
    param_configs = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other_params, lr=lr, weight_decay=wd / lr),
    ]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.23)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        frac = (step - warmup_steps) / warmdown_steps
        return 1.0 * (1 - frac) + 0.07 * frac

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps + 1) / total_train_steps) ** 3
    lookahead_state = LookaheadState(model)

    train_timer = CudaTimer()
    with train_timer:
        init_whitening_conv(model[0], train_loader.normalize(train_loader.images[:5000]))

    history = []
    for epoch in range(ceil(epochs)):
        model[0].bias.requires_grad = epoch < opt["whiten_bias_epochs"]

        with train_timer:
            model.train()
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                current_steps += 1

                if current_steps % 5 == 0:
                    lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

                if current_steps >= total_train_steps:
                    lookahead_state.update(model, decay=1.0)
                    break

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_time_s": train_timer.seconds,
            }
        )

    tta_timer = CudaTimer()
    with tta_timer:
        tta_val_acc = evaluate(model, test_loader, tta_level=hyp["net"]["tta_level"])

    return {
        "seed": spec.seed,
        "epochs": float(epochs),
        "flip": str(spec.flip),
        "translate": int(spec.translate),
        "tta_level": hyp["net"]["tta_level"],
        "steps": current_steps,
        "val_acc": val_acc,
        "tta_val_acc": tta_val_acc,
        "train_time_s": train_timer.seconds,
        "tta_time_s": tta_timer.seconds,
        "wall_time_s": time.perf_counter() - wall_start,
        "history": history,
    }


def warmup(data, hyp=None):
    return train(data, RunSpec(seed=-1, epochs=1, flip=FlipMode.ALTERNATING, translate=2), hyp)
