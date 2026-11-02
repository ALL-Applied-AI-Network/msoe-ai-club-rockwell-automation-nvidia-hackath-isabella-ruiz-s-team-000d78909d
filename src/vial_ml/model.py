"""PyTorch pieces shared by train.py and evaluate.py."""
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from labels import CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
        # Real camera crops are softer than clean renders (lens, motion, JPEG).
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.2))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CropDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), label


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("classes") != CLASSES:
        raise SystemExit(f"Checkpoint classes {ckpt.get('classes')} don't match {CLASSES}")
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict(model, items, size, device, batch_size=64, workers=2):
    """Softmax probabilities, shape (len(items), 5), in item order."""
    loader = DataLoader(CropDataset(items, eval_transform(size)), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(out).numpy()
