"""Download real coral datasets into data/<name>/<class>/ (the layout train.py reads).

  python download_datasets.py --list
  python download_datasets.py noaa                 # healthy vs bleached patches (Hawaii)
  python download_datasets.py noaa coralscapes     # + whole-reef photos (Red Sea), labels from masks
  ROBOFLOW_API_KEY=... python download_datasets.py roboflow_reef roboflow_health

Datasets
  noaa             NOAA PIFSC ESD, Hugging Face De129472/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset
                   (10,419 patches 224x224, CORAL / CORAL_BL, Hawaii 2014-2019). No key needed.
  coralscapes      Hugging Face EPFL-ECEO/coralscapes (2,075 reef photos, expert segmentation masks,
                   Apache-2.0). Healthy/bleached labels are DERIVED here from the masks (see below).
  roboflow_reef    Roboflow Universe "coral-reef-image-classification" (bleached/dead/healthy).
  roboflow_health  Roboflow Universe "coral-health-image-classifier-bleached-or-healthy" (CC BY 4.0).
                   Roboflow needs a free account + API key and the project forked into your workspace.

Coralscapes label rule (mask ids from the dataset card):
  alive coral   : 6, 17, 22, 25, 28, 31, 34, 36
  bleached coral: 4, 16, 19, 33
  dead coral    : 3, 20, 23, 32, 37   (dropped unless --keep-dead, so it matches NOAA's two classes)
  A patch/image gets a label only if coral covers >= --min-coral of it and one coral type makes up
  >= --purity of the coral pixels; mixed ones are skipped. Check "mask values seen" in the output.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

HF_NOAA = "De129472/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset"
HF_CORALSCAPES = "EPFL-ECEO/coralscapes"
CS_ALIVE = [6, 17, 22, 25, 28, 31, 34, 36]
CS_BLEACHED = [4, 16, 19, 33]
CS_DEAD = [3, 20, 23, 32, 37]

ROBOFLOW = {
    "roboflow_reef": ("project-idpmi", "coral-reef-image-classification"),
    "roboflow_health": ("techwiz-wbywj", "coral-health-image-classifier-bleached-or-healthy"),
}
ALL = ["noaa", "coralscapes"] + list(ROBOFLOW)


# ------------------------------------------------------------------ summary
def summarize_folder(name, root):
    """Use the project's own loader so we see exactly what train.py will see."""
    from data import scan_dataset
    try:
        items = scan_dataset(Path(root) / name, "3class")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] PROBLEM: {e}")
        return False
    counts = {}
    for _, l in items:
        counts[l] = counts.get(l, 0) + 1
    names = ["healthy", "bleached", "dead"]
    print(f"[{name}] ready: {len(items)} images, " + ", ".join(f"{names[k]}={v}" for k, v in sorted(counts.items())))
    return True


# ------------------------------------------------------------------ NOAA
def get_noaa(root, **_):
    out = Path(root) / "noaa"
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_NOAA, repo_type="dataset", local_dir=str(out))
    imgs = [p for p in out.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not imgs:  # files came as parquet: decode them into class folders
        print("[noaa] no image files found, decoding parquet via `datasets`")
        from datasets import load_dataset
        ds = load_dataset(HF_NOAA)
        for split in ds:
            names = ds[split].features["label"].names
            for i, ex in enumerate(ds[split]):
                d = out / split / names[ex["label"]]
                d.mkdir(parents=True, exist_ok=True)
                ex["image"].convert("RGB").save(d / f"{i}.png")
    return summarize_folder("noaa", root)


# ------------------------------------------------------------------ Coralscapes
def _label_from_counts(a, b, d, total, min_coral, purity, keep_dead):
    coral = a + b + d
    if coral < min_coral * total or coral == 0:
        return None
    if a / coral >= purity:
        return "healthy"
    if b / coral >= purity:
        return "bleached"
    if keep_dead and d / coral >= purity:
        return "dead"
    return None


def _mask_array(label_img, offset):
    m = np.asarray(label_img)
    if m.ndim == 3:
        raise ValueError("Coralscapes mask is RGB; expected a class-index mask. "
                         "Open one mask and tell the maintainer / adjust _mask_array().")
    return m.astype(np.int32) + offset


def convert_coralscapes(rows, out_dir, mode="patch", tile=256, min_coral=0.5, purity=0.8, keep_dead=False,
                        max_per_class_per_image=3, max_per_class=2500, offset=0, seed=0):
    """rows: iterable of (tag, PIL image, PIL mask). Writes <out_dir>/<class>/<tag>__rXcY.jpg."""
    rng = random.Random(seed)
    out = Path(out_dir)
    classes = ["healthy", "bleached"] + (["dead"] if keep_dead else [])
    for c in classes:
        (out / c).mkdir(parents=True, exist_ok=True)
    saved = {c: 0 for c in classes}
    seen_values = set()
    for tag, img, mask in rows:
        m = _mask_array(mask, offset)
        if len(seen_values) < 60:
            seen_values |= set(np.unique(m).tolist())
        img = img.convert("RGB")
        cand = {c: [] for c in classes}
        if mode == "image":
            a = np.isin(m, CS_ALIVE).sum()
            b = np.isin(m, CS_BLEACHED).sum()
            d = np.isin(m, CS_DEAD).sum()
            lab = _label_from_counts(a, b, d, m.size, min_coral=0.1, purity=purity, keep_dead=keep_dead)
            if lab:
                cand[lab].append((None, None))
        else:
            H, W = m.shape
            for r in range(0, H - tile + 1, tile):
                for c in range(0, W - tile + 1, tile):
                    t = m[r:r + tile, c:c + tile]
                    a = np.isin(t, CS_ALIVE).sum()
                    b = np.isin(t, CS_BLEACHED).sum()
                    d = np.isin(t, CS_DEAD).sum()
                    lab = _label_from_counts(a, b, d, t.size, min_coral, purity, keep_dead)
                    if lab:
                        cand[lab].append((r, c))
        for lab, spots in cand.items():
            rng.shuffle(spots)
            for r, c in spots[:max_per_class_per_image]:
                if saved[lab] >= max_per_class:
                    break
                if r is None:
                    im = img.copy()
                    im.thumbnail((768, 768))
                    name = f"{tag}.jpg"
                else:
                    im = img.crop((c, r, c + tile, r + tile))
                    name = f"{tag}__r{r}c{c}.jpg"
                im.save(out / lab / name, quality=92)
                saved[lab] += 1
        if all(v >= max_per_class for v in saved.values()):
            break
    print(f"[coralscapes] mask values seen (first images): {sorted(seen_values)[:45]}")
    print(f"[coralscapes] saved {saved}")
    if saved["healthy"] == 0 or saved["bleached"] == 0:
        print("[coralscapes] WARNING: a class is empty. Mask ids may be shifted: try --coralscapes-offset 1 "
              "(or -1), and compare 'mask values seen' with the id table in this file's header.")
    return saved


def _coralscapes_rows(max_images):
    from datasets import load_dataset
    n = 0
    for split in ("train", "validation", "test"):
        for i, ex in enumerate(load_dataset(HF_CORALSCAPES, split=split, streaming=True)):
            if max_images and n >= max_images:
                return
            yield f"{split}{i:04d}", ex["image"], ex["label"]
            n += 1


def get_coralscapes(root, max_images=800, mode="patch", keep_dead=False, offset=0, min_coral=0.5,
                    purity=0.8, seed=0, **_):
    out = Path(root) / f"coralscapes_{mode}"
    convert_coralscapes(_coralscapes_rows(max_images), out, mode=mode, keep_dead=keep_dead, offset=offset,
                        min_coral=min_coral, purity=purity, seed=seed)
    return summarize_folder(out.name, root)


# ------------------------------------------------------------------ Roboflow
def get_roboflow(name, root, version=None, rf_workspace=None, **_):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print(f"[{name}] skipped: set ROBOFLOW_API_KEY (free account) and fork the project first. "
              "Or download the 'Folder Structure' zip by hand into data/" + name + "/")
        return False
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    orig_ws, project = ROBOFLOW[name]
    out = Path(root) / name
    for ws in (rf_workspace, None, orig_ws):  # your fork first, then the original
        try:
            ws_obj = rf.workspace(ws) if ws else rf.workspace()
            proj = ws_obj.project(project)
            v = version
            if v is None:
                v = max(int(str(getattr(x, "version", x)).split("/")[-1]) for x in proj.versions())
            proj.version(int(v)).download("folder", location=str(out))
            return summarize_folder(name, root)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] workspace {ws or '(default)'} failed: {type(e).__name__}: {str(e)[:150]}")
    print(f"[{name}] could not download. Fork the project on roboflow.com, then download 'Folder Structure' manually.")
    return False


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"any of: {', '.join(ALL)}, or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--coralscapes-mode", choices=["patch", "image"], default="patch")
    ap.add_argument("--max-images", type=int, default=800, help="Coralscapes photos to read (0 = all)")
    ap.add_argument("--keep-dead", action="store_true")
    ap.add_argument("--coralscapes-offset", type=int, default=0, help="add to mask values if ids are shifted")
    ap.add_argument("--min-coral", type=float, default=0.5)
    ap.add_argument("--purity", type=float, default=0.8)
    ap.add_argument("--roboflow-version", type=int, default=None)
    ap.add_argument("--roboflow-workspace", default=None, help="your workspace slug if the fork is not the default")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.list or not a.names:
        print(__doc__)
        return
    names = ALL if "all" in a.names else a.names
    for n in names:
        if n not in ALL:
            raise SystemExit(f"unknown dataset '{n}'. Choose from: {', '.join(ALL)}")
    for n in names:
        print(f"\n=== {n} ===")
        if n == "noaa":
            get_noaa(a.data_root)
        elif n == "coralscapes":
            get_coralscapes(a.data_root, a.max_images, a.coralscapes_mode, a.keep_dead, a.coralscapes_offset,
                            a.min_coral, a.purity, a.seed)
        else:
            get_roboflow(n, a.data_root, a.roboflow_version, a.roboflow_workspace)


if __name__ == "__main__":
    main()
