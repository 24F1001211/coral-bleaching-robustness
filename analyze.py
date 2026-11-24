"""Shortcut analysis for a trained run.

  python analyze.py --run runs/effb0_ce --task probes       # grayscale / patch-shuffle stress tests
  python analyze.py --run runs/effb0_ce --task colour       # colour-histogram-only baseline
  python analyze.py --run runs/effb0_ce --task gradcam      # attention overlays

Reading the results:
  * accuracy collapses in grayscale        -> model relies on colour
  * accuracy survives patch-shuffle        -> model uses colour statistics, not coral structure
  * colour-only baseline ~ CNN accuracy    -> the "detection" is mostly colour
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression

from augment import MEAN, STD, build_transforms
from metrics import summarize
from models import build_model, to_probs
from train import make_loader, predict


def load_run(run, device):
    ckpt = torch.load(Path(run) / "model.pt", map_location=device)
    K = len(ckpt["classes"])
    model = build_model(ckpt["arch"], K, ckpt["loss"], pretrained=False)
    model.load_state_dict(ckpt["state"])
    model.to(device).eval()
    items = json.loads((Path(run) / "items.json").read_text())
    return model, ckpt, items


def task_probes(model, ckpt, items, device, run, workers):
    K = len(ckpt["classes"])
    table = {}
    for name, ev in items["eval"].items():
        table[name] = {}
        for probe in [None, "grayscale", "patch_shuffle"]:
            loader = make_loader(ev, ckpt["size"], 32, workers, probe=probe)
            P, Y = predict(model, loader, device, ckpt["loss"])
            table[name][probe or "normal"] = summarize(P, Y, K)
    print(f"{'dataset':38s} {'normal':>8s} {'grayscale':>10s} {'shuffle':>8s}   (macro-F1)")
    for name, r in table.items():
        print(f"{name:38s} {r['normal']['macro_f1']:8.3f} {r['grayscale']['macro_f1']:10.3f} "
              f"{r['patch_shuffle']['macro_f1']:8.3f}")
    (Path(run) / "probes.json").write_text(json.dumps(table, indent=2))


def colour_features(src, bins=16):
    """src: image path or PIL image."""
    img = src if isinstance(src, Image.Image) else Image.open(src)
    a = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    hist = [np.histogram(a[..., c], bins=bins, range=(0, 1))[0] / (64 * 64) for c in range(3)]
    return np.concatenate(hist + [a.mean((0, 1)), a.std((0, 1))])


def task_colour(ckpt, items, run):
    K = len(ckpt["classes"])
    Xtr = np.stack([colour_features(p) for p, _ in items["train"]])
    ytr = np.array([l for _, l in items["train"]])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    table = {}
    for name, ev in items["eval"].items():
        X = np.stack([colour_features(p) for p, _ in ev])
        y = np.array([l for _, l in ev])
        probs = np.zeros((len(y), K))
        probs[:, clf.classes_] = clf.predict_proba(X)
        table[name] = summarize(probs, y, K)
        print(f"[colour-only] {name}: acc={table[name]['accuracy']:.3f} f1={table[name]['macro_f1']:.3f}")
    (Path(run) / "colour_baseline.json").write_text(json.dumps(table, indent=2))


def target_layer(model, arch):
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.blocks[-1]
    if arch.startswith("convnext"):
        return model.stages[-1]
    raise ValueError(f"Grad-CAM layer not defined for {arch}; add it in target_layer()")


def gradcam(model, layer, x, loss):
    store = {}

    def fwd_hook(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    model.zero_grad()
    probs = to_probs(model(x), loss)
    cls = probs.argmax(1)
    probs.gather(1, cls[:, None]).sum().backward()
    h.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mn, mx = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
    cam = (cam - mn) / (mx - mn + 1e-8)
    return cam.detach().cpu().numpy(), cls.cpu().numpy(), probs.detach().cpu().numpy()


def task_gradcam(model, ckpt, items, device, run, per_set):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer = target_layer(model, ckpt["arch"])
    tf = build_transforms(False, ckpt["size"])
    names = ckpt["classes"]
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    sets = list(items["eval"].items())
    fig, axes = plt.subplots(len(sets), per_set, figsize=(3 * per_set, 3.2 * len(sets)), squeeze=False)
    rng = random.Random(0)
    for r, (name, ev) in enumerate(sets):
        for c, (path, label) in enumerate(rng.sample(ev, min(per_set, len(ev)))):
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            cam, cls, probs = gradcam(model, layer, x, ckpt["loss"])
            img = (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[r][c]
            ax.imshow(img)
            ax.imshow(cam[0], cmap="jet", alpha=0.45)
            ax.set_title(f"pred {names[cls[0]]} ({probs[0, cls[0]]:.2f})\ntrue {names[label]}", fontsize=8)
            ax.axis("off")
        axes[r][0].set_ylabel(name, fontsize=8)
        axes[r][0].text(-0.05, 0.5, name, transform=axes[r][0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    plt.tight_layout()
    out = Path(run) / "gradcam.png"
    plt.savefig(out, dpi=130)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", choices=["probes", "colour", "gradcam"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--per-set", type=int, default=6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt, items = load_run(args.run, device)
    if args.task == "probes":
        task_probes(model, ckpt, items, device, args.run, args.workers)
    elif args.task == "colour":
        task_colour(ckpt, items, args.run)
    else:
        task_gradcam(model, ckpt, items, device, args.run, args.per_set)


if __name__ == "__main__":
    main()
