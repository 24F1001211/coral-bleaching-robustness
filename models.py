"""Models and losses: standard cross-entropy or ordinal (CORAL-style) head."""
import timm
import torch
import torch.nn.functional as F


def build_model(arch, n_classes, loss="ce", pretrained=True):
    n_out = n_classes - 1 if loss == "coral" else n_classes
    return timm.create_model(arch, pretrained=pretrained, num_classes=n_out)


def compute_loss(logits, y, loss, n_classes):
    if loss == "ce":
        return F.cross_entropy(logits, y)
    # CORAL: K-1 binary tasks "is y > k?"
    ks = torch.arange(n_classes - 1, device=y.device).unsqueeze(0)
    levels = (y.unsqueeze(1) > ks).float()
    return F.binary_cross_entropy_with_logits(logits, levels)


def to_probs(logits, loss):
    """Class probabilities for both head types (used for accuracy, ECE and Grad-CAM)."""
    if loss == "ce":
        return logits.softmax(1)
    c = torch.cummin(torch.sigmoid(logits), dim=1).values  # P(y > k), forced monotone
    probs = torch.cat([1 - c[:, :1], c[:, :-1] - c[:, 1:], c[:, -1:]], dim=1).clamp_min(0)
    return probs / probs.sum(1, keepdim=True).clamp_min(1e-8)
