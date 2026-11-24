"""Dataset utilities.

Every dataset is a folder tree where the *parent folder of each image* is its class,
e.g. data/roboflow/train/bleached_coral/img1.jpg. Roboflow/Kaggle exports with
train/valid/test sub-folders work as they are: all splits are merged and re-split here
with our own seed so that results are reproducible.
"""
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder-name aliases -> canonical class. Add yours here if a dataset uses other names.
ALIASES = {
    "healthy": "healthy", "healthy_coral": "healthy", "healthy coral": "healthy",
    "unbleached": "healthy", "normal": "healthy",
    "bleached": "bleached", "bleached_coral": "bleached", "bleached coral": "bleached",
    "bleaching": "bleached",
    "dead": "dead", "dead_coral": "dead", "dead coral": "dead",
    # NOAA PIFSC dataset folder names
    "coral": "healthy", "coral_bl": "bleached",
    # other Roboflow and Kaggle naming variants
    "healthy_corals": "healthy", "healthy corals": "healthy", "not_bleached": "healthy",
    "bleached_corals": "bleached", "bleached corals": "bleached", "dead corals": "dead", "dead_corals": "dead",
}

# Ordinal order: healthy < bleached < dead
SCHEMES = {
    "3class": {"healthy": 0, "bleached": 1, "dead": 2},
    "binary": {"healthy": 0, "bleached": 1, "dead": 1},
}


def class_names(scheme):
    return ["healthy", "bleached", "dead"] if scheme == "3class" else ["healthy", "unhealthy"]


def scan_dataset(root, scheme="3class"):
    """Return [(path, label)] for every image whose parent folder is a known class."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    items, skipped = [], Counter()
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        canon = ALIASES.get(p.parent.name.lower().strip())
        if canon is None:
            skipped[p.parent.name] += 1
            continue
        items.append((str(p), SCHEMES[scheme][canon]))
    if skipped:
        print(f"[data] {root.name}: skipped folders {dict(skipped)} (not in ALIASES)")
    if not items:
        raise RuntimeError(f"No usable images found in {root}. Check folder names vs ALIASES in data.py")
    return items


def dedupe(items):
    """Drop byte-identical duplicates (exports with augmented copies leak across splits otherwise)."""
    seen, out = set(), []
    for path, label in items:
        h = hashlib.md5(Path(path).read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append((path, label))
    if len(out) != len(items):
        print(f"[data] removed {len(items) - len(out)} exact duplicates")
    return out


def source_group(path):
    """Source-image id, so crops/augmented copies of one photo never straddle train and test.

    Recognised: '<id>__<anything>.jpg' (patches made by download_datasets.py) and
    Roboflow exports '<id>.rf.<hash>.jpg' (augmented copies). Returns None otherwise.
    """
    name = Path(path).name
    if ".rf." in name:
        return name.split(".rf.")[0]
    stem = Path(path).stem
    if "__" in stem:
        return stem.split("__")[0]
    return None


def _split_grouped(items, seed, fracs):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[source_group(it[0]) or it[0]].append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(items)
    train, val, test = [], [], []
    for k in keys:
        if len(train) < fracs[0] * n:
            train += groups[k]
        elif len(val) < fracs[1] * n:
            val += groups[k]
        else:
            test += groups[k]
    return train, val, test


def split_items(items, seed=0, fracs=(0.7, 0.15, 0.15)):
    """Train/val/test split. Group-aware when files carry a source-image id, else stratified."""
    if any(source_group(p) for p, _ in items):
        return _split_grouped(items, seed, fracs)
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for it in items:
        by_class[it[1]].append(it)
    train, val, test = [], [], []
    for _, group in sorted(by_class.items()):
        rng.shuffle(group)
        n = len(group)
        a = int(n * fracs[0])
        b = a + int(n * fracs[1])
        train += group[:a]
        val += group[a:b]
        test += group[b:]
    return train, val, test


def class_counts(items, n_classes):
    c = Counter(l for _, l in items)
    return [c.get(i, 0) for i in range(n_classes)]


class CoralDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
