"""Single-image inference for the live app: prediction, Grad-CAM, stress tests, colour-only baseline."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from analyze import colour_features, gradcam, target_layer
from augment import MEAN, STD, build_transforms
from models import build_model, to_probs

_MEAN = torch.tensor(MEAN).view(3, 1, 1)
_STD = torch.tensor(STD).view(3, 1, 1)


class Detector:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self.run_dir / "model.pt", map_location=self.device)
        self.classes = ckpt["classes"]
        self.loss = ckpt["loss"]
        self.size = ckpt["size"]
        self.arch = ckpt["arch"]
        self.model = build_model(self.arch, len(self.classes), self.loss, pretrained=False)
        self.model.load_state_dict(ckpt["state"])
        self.model.to(self.device).eval()
        try:
            self.layer = target_layer(self.model, self.arch)
        except ValueError:
            self.layer = None  # Grad-CAM not defined for this architecture
        self._colour = None

    def _prep(self, img, probe=None):
        return build_transforms(False, self.size, probe=probe)(img.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def probs(self, img, probe=None):
        """probe: None | 'grayscale' | 'patch_shuffle'. Returns class probabilities."""
        return to_probs(self.model(self._prep(img, probe)), self.loss)[0].cpu().numpy()

    def cam_overlay(self, img):
        """Grad-CAM heat map blended over the (resized) image, as a uint8 array. None if unsupported."""
        if self.layer is None:
            return None
        x = self._prep(img)
        cam, _, _ = gradcam(self.model, self.layer, x, self.loss)
        base = (x[0].cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        heat = matplotlib.colormaps["jet"](cam[0])[..., :3]
        return (np.clip(0.55 * base + 0.45 * heat, 0, 1) * 255).astype(np.uint8)

    def colour_probs(self, img):
        """Colour-histogram-only logistic regression, fitted lazily on this run's training images."""
        if self._colour is None:
            items = json.loads((self.run_dir / "items.json").read_text())["train"]
            valid_items = [it for it in items if Path(it[0]).exists()]
            if not valid_items:
                return np.full(len(self.classes), 1.0 / len(self.classes))
            idx = np.random.default_rng(0).permutation(len(valid_items))[:1500]
            X = np.stack([colour_features(valid_items[i][0]) for i in idx])
            y = np.array([valid_items[i][1] for i in idx])
            self._colour = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
        p = np.zeros(len(self.classes))
        p[self._colour.classes_] = self._colour.predict_proba(colour_features(img)[None])[0]
        return p
