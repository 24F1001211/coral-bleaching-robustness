"""Generate two tiny synthetic datasets to smoke-test the pipeline (NOT for results)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "dummy")
rng = np.random.default_rng(0)
# base colour per class, and a different water tint per dataset
base = {"healthy": (120, 90, 50), "bleached_coral": (225, 225, 215), "dead": (110, 110, 105)}
tint = {"dsA": (1.0, 1.0, 1.0), "dsB": (0.7, 1.0, 1.25)}

for ds, t in tint.items():
    for cls, col in base.items():
        d = root / ds / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(40):
            img = np.clip(np.array(col) * np.array(t) + rng.normal(0, 25, (96, 96, 3)), 0, 255)
            Image.fromarray(img.astype(np.uint8)).save(d / f"{i}.jpg")
print(f"wrote dummy data to {root}/")
