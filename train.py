"""Train on source dataset(s), then evaluate in-domain and on unseen target dataset(s).

Example:
  python train.py --data-root data --source roboflow --targets kaggle,kohtao \
      --arch efficientnet_b0 --loss ce --epochs 15 --out runs/effb0_ce
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from augment import build_transforms
from data import CoralDataset, class_counts, class_names, dedupe, scan_dataset, split_items
from metrics import summarize
from models import build_model, compute_loss, to_probs


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device, loss):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(to_probs(model(x.to(device)).float(), loss).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def make_loader(items, size, bs, workers, train=False, cast=False, sampler=None, probe=None):
    ds = CoralDataset(items, build_transforms(train, size, cast, probe))
    return DataLoader(ds, batch_size=bs, shuffle=train and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--source", required=True, help="comma-separated dataset folder names to train on")
    ap.add_argument("--targets", default="", help="comma-separated unseen datasets for cross-dataset test")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--loss", choices=["ce", "coral"], default="ce")
    ap.add_argument("--scheme", choices=["3class", "binary"], default="3class")
    ap.add_argument("--cast", action="store_true", help="underwater colour-cast augmentation")
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="runs/run")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = class_names(args.scheme)
    K = len(names)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = [s for s in args.source.split(",") if s]
    targets = [t for t in args.targets.split(",") if t]

    # --- data: split each dataset on its own, so every source keeps its own test portion
    splits, train_items, val_items = {}, [], []
    for n in dict.fromkeys(sources + targets):
        items = dedupe(scan_dataset(Path(args.data_root) / n, args.scheme))
        splits[n] = split_items(items, args.seed)
        print(f"[data] {n}: {len(items)} images, class counts {class_counts(items, K)}")
    for n in sources:
        train_items += splits[n][0]
        val_items += splits[n][1]

    eval_sets = {}
    for n in sources:
        eval_sets[f"{n} (in-domain test)"] = splits[n][2]
    for n in targets:
        # a target that is also a source is only evaluated on its held-out test split
        eval_sets[f"{n} (cross-dataset)"] = splits[n][2] if n in sources else sum(splits[n], [])

    sampler = None
    if args.balanced_sampler:
        cnt = class_counts(train_items, K)
        w = [1.0 / max(cnt[l], 1) for _, l in train_items]
        sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = make_loader(train_items, args.size, args.bs, args.workers, True, args.cast, sampler)
    val_loader = make_loader(val_items, args.size, args.bs, args.workers)

    # --- model
    model = build_model(args.arch, K, args.loss, args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_f1, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(x)
            loss = compute_loss(logits.float(), y, args.loss, K)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(y)
        sched.step()
        P, Y = predict(model, val_loader, device, args.loss)
        m = summarize(P, Y, K)
        print(f"epoch {ep:02d} loss {run / len(train_items):.4f} val_acc {m['accuracy']:.3f} val_f1 {m['macro_f1']:.3f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "arch": args.arch, "loss": args.loss, "scheme": args.scheme,
                "size": args.size, "classes": names}, out / "model.pt")

    # --- evaluation: in-domain vs cross-dataset (the headline comparison)
    results = {}
    for name, items in eval_sets.items():
        P, Y = predict(model, make_loader(items, args.size, args.bs, args.workers), device, args.loss)
        results[name] = summarize(P, Y, K)
        r = results[name]
        print(f"[eval] {name}: n={r['n']} acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} "
              f"mae={r['mae']:.3f} ece={r['ece']:.3f}")

    (out / "results.json").write_text(json.dumps({"args": vars(args), "best_val_f1": best_f1,
                                                  "eval": results}, indent=2))
    (out / "items.json").write_text(json.dumps({"train": train_items, "eval": eval_sets}))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
