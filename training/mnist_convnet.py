#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.home>
#
# Distributed under terms of the MIT license.

from enum import Enum
from pathlib import Path
from typing import Callable
from tinygrad import Tensor, TinyJit, nn
from tinygrad.device import Device
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from export_model import export_model

import math


class SamplingMod(Enum):
    BILINEAR = 0
    NEAREST = 1

def geometric_transform(X: Tensor, angle_deg: Tensor, scale: Tensor, shift_x: Tensor, shift_y: Tensor, sampling: SamplingMod) -> Tensor:
    B, C, H, W = X.shape

    angle = angle_deg * math.pi / 180.0
    cos_a, sin_a = Tensor.cos(angle), Tensor.sin(angle)
    R11, R12, T13 = cos_a * scale, -sin_a * scale, shift_x
    R21, R22, T23 = sin_a * scale,  cos_a * scale, shift_y
    row1 = Tensor.cat(R11.reshape(B, 1), R12.reshape(B, 1), T13.reshape(B, 1), dim=1).reshape(B, 1, 3)
    row2 = Tensor.cat(R21.reshape(B, 1), R22.reshape(B, 1), T23.reshape(B, 1), dim=1).reshape(B, 1, 3)
    row3 = Tensor([[0.0, 0.0, 1.0]]).expand(B, 1, 3)
    affine_matrix = Tensor.cat(row1, row2, row3, dim=1)

    x_idx, y_idx = Tensor.arange(W).float(), Tensor.arange(H).float()
    grid_y, grid_x = y_idx.reshape(-1, 1).expand(H, W), x_idx.reshape(1, -1).expand(H, W)
    coords = Tensor.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords_homo = Tensor.cat(coords, Tensor.ones(H * W, 1), dim=1).reshape(1, H * W, 3).expand(B, H * W, 3)
    transformed_coords = coords_homo.matmul(affine_matrix.permute(0, 2, 1))

    match sampling:
        case SamplingMod.NEAREST:
            x_idx = transformed_coords[:, :, 0].round().clip(0, W - 1).int()
            y_idx = transformed_coords[:, :, 1].round().clip(0, H - 1).int()
            return X.reshape(B, C * H * W).gather(1, y_idx * W + x_idx).reshape(B, C, H, W)
        case SamplingMod.BILINEAR:
            x_prime, y_prime = transformed_coords[:, :, 0],  transformed_coords[:, :, 1]
            x0, y0 = x_prime.floor().int(), y_prime.floor().int()
            dx, dy = x_prime - x0.float(), y_prime - y0.float()

            x1, y1 = x0 + 1, y0 + 1
            x0, y0 = x0.clip(0, W - 1), y0.clip(0, H - 1)
            x1, y1 = x1.clip(0, W - 1), y1.clip(0, H - 1)

            w00 = (1.0 - dx) * (1.0 - dy)
            w10 = dx * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w11 = dx * dy

            X_flat = X.reshape(B, C * H * W)
            v00 = X_flat.gather(1, y0 * W + x0)
            v10 = X_flat.gather(1, y0 * W + x1)
            v01 = X_flat.gather(1, y1 * W + x0)
            v11 = X_flat.gather(1, y1 * W + x1)

            return ((w00 * v00) + (w10 * v10) + (w01 * v01) + (w11 * v11)).reshape(B, C, H, W)

def normalize(X: Tensor) -> Tensor:
    return X * 2 / 255 - 1


class Model:
    def __init__(self, activation="silu"):
        act = Tensor.silu if activation.lower() == "silu" else Tensor.relu
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5), act,
            nn.Conv2d(32, 32, 5), act,
            nn.BatchNorm(32), Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3), act,
            nn.Conv2d(64, 64, 3), act,
            nn.BatchNorm(64), Tensor.max_pool2d,
            lambda x: x.flatten(1), nn.Linear(576, 10),
        ]

    def __call__(self, x:Tensor) -> Tensor:
        return x.sequential(self.layers)


def cosine_lr(step:int, total:int, lr_min:float, lr_max:float) -> float:
    t = min(step, total)
    return lr_min + 0.5*(lr_max-lr_min)*(1.0 + math.cos(math.pi * t / total))

if __name__ == "__main__":
    B = int(getenv("BATCH", 256))
    OPT = getenv("OPT", "ADAMW").upper()             
    MOMENTUM = float(getenv("MOMENTUM", 0.9))
    WD = float(getenv("WD", 0.0))
    STEPS = int(getenv("STEPS", 3000))

    LR_MAX = float(getenv("LR_MAX", 1e-3 if OPT=="ADAMW" else 0.1))
    LR_MIN = float(getenv("LR_MIN", LR_MAX/50))

    ANGLE = float(getenv("ANGLE", 15))
    SCALE = float(getenv("SCALE", 0.05))
    SHIFT = float(getenv("SHIFT", 0.05))
    SAMPLING = SamplingMod(int(getenv("SAMPLING", SamplingMod.BILINEAR.value)))

    LOG_EVERY = int(getenv("LOG_EVERY", 100))
    VAL_CHECK_EVERY = int(getenv("VAL_CHECK_EVERY", 100))
    PATIENCE = int(getenv("PATIENCE", 1200))

    ACT = getenv("ACT", "silu").lower()

    model_name = Path(__file__).name.split('.')[0]
    dir_name = Path(__file__).parent / model_name
    dir_name.mkdir(exist_ok=True)

    log_path = dir_name / f"{model_name}_trainlog.csv"
    with open(log_path, "w") as f:
        f.write("step,lr,train_loss,train_acc,val_acc,val_loss\n")

    X_train_full, Y_train_full, X_test, Y_test = mnist()
    VAL_SIZE = int(getenv("VAL_SIZE", 10000))
    X_train, Y_train = X_train_full[:-VAL_SIZE], Y_train_full[:-VAL_SIZE]
    X_val, Y_val     = X_train_full[-VAL_SIZE:], Y_train_full[-VAL_SIZE:]

    model = Model(activation=ACT)

    params = nn.state.get_parameters(model)
    if OPT == "SGD":
        opt = nn.optim.SGD(params, lr=LR_MAX, momentum=MOMENTUM, weight_decay=WD)
    else:
        opt = nn.optim.AdamW(params, lr=LR_MAX, weight_decay=WD)

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        samples = Tensor.randint(B, high=int(X_train.shape[0]))
        angle_deg = (Tensor.rand(B) * 2 * ANGLE - ANGLE)
        scale = 1.0 + (Tensor.rand(B) * 2 * SCALE - SCALE)
        shift_x = (Tensor.rand(B) * 2 * SHIFT - SHIFT)
        shift_y = (Tensor.rand(B) * 2 * SHIFT - SHIFT)

        opt.zero_grad()
        inp = normalize(geometric_transform(X_train[samples], angle_deg, scale, shift_x, shift_y, SAMPLING))
        logits = model(inp)
        loss = logits.sparse_categorical_crossentropy(Y_train[samples]).backward()
        train_acc = (logits.argmax(axis=1) == Y_train[samples]).mean() * 100
        return loss.realize(*opt.schedule_step()), train_acc.realize()

    @TinyJit
    def get_val_acc() -> Tensor:
        return (model(normalize(X_val)).argmax(axis=1) == Y_val).mean() * 100

    @TinyJit
    def get_val_loss() -> Tensor:
        logits = model(normalize(X_val))
        return logits.sparse_categorical_crossentropy(Y_val).mean()


    best_val, best_step, steps_since_best = 0.0, 0, 0
    val_acc, val_loss = float('nan'), float('nan')

    for i in (t := trange(STEPS)):
        lr_now = cosine_lr(i, STEPS, LR_MIN, LR_MAX)
        opt.lr = lr_now

        loss, tr_acc = train_step()
        tr_acc = tr_acc.item()

        if (i % VAL_CHECK_EVERY) == (VAL_CHECK_EVERY - 1):
            val_acc = get_val_acc().item()
            val_loss = get_val_loss().item()
            if val_acc > best_val:
                best_val, best_step, steps_since_best = val_acc, i, 0
                state_dict = get_state_dict(model)
                safe_save(state_dict, dir_name / f"{model_name}.safetensors")
            else:
                steps_since_best += VAL_CHECK_EVERY

        with open(log_path, "a") as f:
            f.write(f"{i},{lr_now},{loss.item():.6f},{tr_acc:.4f},"
                f"{'' if math.isnan(val_acc) else f'{val_acc:.4f}'},"
                f"{'' if math.isnan(val_loss) else f'{val_loss:.6f}'}\n")

        if (i % LOG_EVERY) == (LOG_EVERY - 1):
            t.set_description(f"lr: {lr_now:2.2e}  loss: {loss.item():2.2f}  best_val: {best_val:2.2f}%")

        if steps_since_best >= PATIENCE:
            break

    state_dict = safe_load(dir_name / f"{model_name}.safetensors")
    load_state_dict(model, state_dict)
    test_acc = (model(normalize(X_test)).argmax(axis=1) == Y_test).mean().item() * 100
    print(f"Best validation at step {best_step}, Test Accuracy: {test_acc:.2f}%")

    Device.DEFAULT = "WEBGPU"
    model = Model(activation=ACT)
    load_state_dict(model, state_dict)
    input = Tensor.randn(1, 1, 28, 28)
    prg, *_, state = export_model(model, Device.DEFAULT.lower(), input, model_name=model_name)
    safe_save(state, dir_name / f"{model_name}.webgpu.safetensors")
    with open(dir_name / f"{model_name}.js", "w") as text_file:
        text_file.write(prg)

