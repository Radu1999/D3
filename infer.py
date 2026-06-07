"""
Inference script for D³.

Unlabeled mode (default):
    python infer.py --checkpoint ckpt/classifier.pth --image_dir /path/to/images

Eval mode (labeled real/fake directory):
    python infer.py --checkpoint ckpt/classifier.pth --image_dir /path/to/dataset --eval

In eval mode, image_dir must contain real/ and fake/ subdirectories.
Outputs F1, accuracy, ROC-AUC, precision, and recall.

Scores near 1.0 = fake, near 0.0 = real.
"""

import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)
from torch.utils.data import Dataset, DataLoader
from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer
from models import get_model

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".PNG", ".JPG", ".JPEG"}


def load_model(checkpoint, arch, head_type, fix_backbone):
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

    if arch.startswith("CLIP:"):
        clip_model = arch.split("CLIP:")[1]  # e.g. "ViT-L/14"
        model = CLIPModelShuffleAttentionPenultimateLayer(
            clip_model, shuffle_times=1, original_times=1, patch_size=[14]
        )
    else:
        # Build a fake opt for get_model
        class _Opt:
            pass
        opt = _Opt()
        opt.arch = arch
        opt.head_type = head_type
        model = get_model(opt)

    if fix_backbone:
        if head_type in ("attention", "crossattention"):
            model.attention_head.load_state_dict(state_dict)
        elif head_type == "mlp":
            model.mlp.load_state_dict(state_dict)
        elif head_type == "transformer":
            model.transformer_block.load_state_dict(state_dict["transformer"])
            model.fc.load_state_dict(state_dict["fc"])
        else:  # fc
            model.fc.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    return model


def collect_images(image_dir):
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1] in EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def collect_labeled_images(image_dir):
    """Collect images from real/ and fake/ subdirs, return (paths, labels)."""
    real_dir = os.path.join(image_dir, "real")
    fake_dir = os.path.join(image_dir, "fake")
    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise ValueError(
            f"--eval mode requires {image_dir} to contain real/ and fake/ subdirectories"
        )
    paths, labels = [], []
    for p in sorted(collect_images(real_dir)):
        paths.append(p)
        labels.append(0)
    for p in sorted(collect_images(fake_dir)):
        paths.append(p)
        labels.append(1)
    return paths, labels


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels  # None for unlabeled
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx] if self.labels is not None else -1
        return img, label, path


def run_inference(model, paths, batch_size, threshold, transform, labels=None, num_workers=4):
    """Run inference using a DataLoader for parallel image loading."""
    dataset = ImageDataset(paths, labels, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    results = []
    valid_labels = [] if labels is not None else None
    model.eval()
    with torch.no_grad():
        for imgs, batch_labels, batch_paths in tqdm(loader, unit="batch"):
            imgs = imgs.cuda(non_blocking=True)
            out = model(imgs)
            if out.shape[-1] == 2:
                out = out[:, 0]
            scores = out.sigmoid().flatten().cpu().numpy()
            for path, score, lbl in zip(batch_paths, scores, batch_labels.tolist()):
                results.append((path, float(score), "fake" if score > threshold else "real"))
                if valid_labels is not None:
                    valid_labels.append(lbl)
    return results, valid_labels


def print_eval_metrics(results, labels, threshold):
    scores = np.array([r[1] for r in results])
    y_true = np.array(labels)
    y_pred = (scores > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, scores)

    n_real = int((y_true == 0).sum())
    n_fake = int((y_true == 1).sum())
    n_correct = int((y_pred == y_true).sum())

    print(f"\n{'='*50}")
    print(f"  Eval results  (threshold={threshold})")
    print(f"{'='*50}")
    print(f"  Total:     {len(results)}  (real={n_real}, fake={n_fake})")
    print(f"  Correct:   {n_correct}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1:        {f1*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["real", "fake"]))


def main():
    parser = argparse.ArgumentParser(description="D³ inference / eval")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image_dir", required=True,
                        help="Image directory. In --eval mode must contain real/ and fake/ subdirs.")
    parser.add_argument("--eval", action="store_true",
                        help="Eval mode: image_dir has real/ and fake/ subdirs; compute metrics.")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--arch", default="CLIP:ViT-L/14", help="Backbone (e.g. CLIP:ViT-L/14, res50)")
    parser.add_argument("--head_type", default="attention",
                        choices=["attention", "crossattention", "fc", "mlp", "transformer"],
                        help="Classifier head type (must match training config)")
    parser.add_argument("--fix_backbone", action="store_true", default=True,
                        help="Was backbone frozen during training? (default: True)")
    parser.add_argument("--no_fix_backbone", dest="fix_backbone", action="store_false",
                        help="Full model checkpoint (backbone was not frozen)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold: score > threshold => fake")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes for parallel image loading")
    opt = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    print(f"Loading model from {opt.checkpoint} ...")
    model = load_model(opt.checkpoint, opt.arch, opt.head_type, opt.fix_backbone)
    model.cuda()
    print("Model loaded.")

    if opt.eval:
        paths, labels = collect_labeled_images(opt.image_dir)
    else:
        paths = collect_images(opt.image_dir)
        labels = None

    if not paths:
        print(f"No images found in {opt.image_dir}")
        return
    print(f"Found {len(paths)} images. Running inference...")

    results, valid_labels = run_inference(
        model, paths, opt.batch_size, opt.threshold, transform,
        labels=labels, num_workers=opt.num_workers,
    )

    with open(opt.output, "w", newline="") as f:
        writer = csv.writer(f)
        if opt.eval:
            writer.writerow(["path", "score", "prediction", "ground_truth"])
            for (path, score, pred), gt in zip(results, valid_labels):
                writer.writerow([path, score, pred, "fake" if gt == 1 else "real"])
        else:
            writer.writerow(["path", "score", "prediction"])
            writer.writerows(results)

    scores = np.array([r[1] for r in results])
    n_fake = sum(1 for r in results if r[2] == "fake")
    print(f"\nDone. Results saved to {opt.output}")
    print(f"  Total: {len(results)}  |  Fake: {n_fake}  |  Real: {len(results) - n_fake}")
    print(f"  Score mean: {scores.mean():.3f}  |  std: {scores.std():.3f}")

    if opt.eval:
        print_eval_metrics(results, valid_labels, opt.threshold)


if __name__ == "__main__":
    main()
