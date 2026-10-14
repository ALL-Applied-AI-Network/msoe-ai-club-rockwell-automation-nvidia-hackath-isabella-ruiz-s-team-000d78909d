"""Grad-CAM for the defect score: which pixels pushed the crop toward 'defect'.

Use it to check the model looks at the target cap, not glare, the fixture or a
neighbouring vial.
"""
import numpy as np
import torch
from PIL import Image

from labels import DEFECT_IDX
from model import eval_transform


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        layer = model.layer4
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _inp, out):
        self.acts = out.detach()

    def _save_grads(self, _m, _gin, gout):
        self.grads = gout[0].detach()

    def __call__(self, x):
        """x: (1, 3, H, W). Returns a (h, w) map in [0, 1]."""
        self.model.zero_grad()
        probs = torch.softmax(self.model(x), dim=1)
        target = probs[0, DEFECT_IDX].sum().clamp_min(1e-8).log()
        target.backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(dim=1))[0]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def overlay(path, cam, size, alpha=0.45):
    """The crop at model input size with the map drawn in red on top."""
    with Image.open(path) as im:
        base = im.convert("RGB").resize((size, size))
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    h = np.asarray(heat, dtype=np.float32) / 255.0
    color = np.zeros((size, size, 3), dtype=np.float32)
    color[..., 0] = 255.0
    color[..., 1] = 255.0 * (1 - h) * 0.4
    a = (alpha * h)[..., None]
    out = np.asarray(base, dtype=np.float32) * (1 - a) + color * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def save_cams(model, paths, size, device, out_paths):
    model.eval()
    cam = GradCAM(model)
    tf = eval_transform(size)
    for path, dest in zip(paths, out_paths):
        with Image.open(path) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        x.requires_grad_(True)
        overlay(path, cam(x), size).save(dest)
