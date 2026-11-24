"""Transforms: standard training aug, underwater colour-cast aug, and probe transforms."""
import random

import torch
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class UnderwaterCast:
    """Random blue/green tint, haze and brightness change (tensor in [0,1], CxHxW).

    Simulates different water colour, depth and turbidity so the model cannot rely on
    the absolute colour of one dataset.
    """

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        gain = torch.tensor([random.uniform(0.6, 1.0), random.uniform(0.8, 1.1),
                             random.uniform(0.9, 1.2)]).view(3, 1, 1)
        x = x * gain
        t = random.uniform(0.0, 0.35)
        haze = torch.tensor([random.uniform(0.1, 0.4), random.uniform(0.4, 0.7),
                             random.uniform(0.5, 0.8)]).view(3, 1, 1)
        x = (1 - t) * x + t * haze
        x = x * random.uniform(0.7, 1.2)
        return x.clamp(0, 1)


class Grayscale3:
    """Remove colour, keep structure."""

    def __call__(self, x):
        return x.mean(0, keepdim=True).repeat(3, 1, 1)


class PatchShuffle:
    """Shuffle a grid of patches: destroys shape/structure, keeps colour statistics."""

    def __init__(self, grid=4):
        self.grid = grid

    def __call__(self, x):
        _, H, W = x.shape
        g = self.grid
        h, w = H // g, W // g
        patches = [x[:, i * h:(i + 1) * h, j * w:(j + 1) * w] for i in range(g) for j in range(g)]
        random.shuffle(patches)
        out = x.clone()
        k = 0
        for i in range(g):
            for j in range(g):
                out[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = patches[k]
                k += 1
        return out


def build_transforms(train, size=224, cast=False, probe=None):
    """probe: None | 'grayscale' | 'patch_shuffle' (evaluation-time stress tests)."""
    if train:
        ops = [T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ToTensor()]
        if cast:
            ops.append(UnderwaterCast())
    else:
        ops = [T.Resize((size, size)), T.ToTensor()]
        if probe == "grayscale":
            ops.append(Grayscale3())
        elif probe == "patch_shuffle":
            ops.append(PatchShuffle())
    ops.append(T.Normalize(MEAN, STD))
    return T.Compose(ops)
