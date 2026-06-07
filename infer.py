"""
Unlabeled inference script for D³.

Usage:
    python infer.py --checkpoint ckpt/classifier.pth --image_dir /path/to/images

Scores near 1.0 = fake, near 0.0 = real.
"""

import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
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


def run_inference(model, paths, batch_size, threshold, transform):
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            imgs = []
            valid_paths = []
            for p in batch_paths:
                try:
                    imgs.append(transform(Image.open(p).convert("RGB")))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Warning: skipping {p} ({e})")
            if not imgs:
                continue
            tensor = torch.stack(imgs).cuda()
            out = model(tensor)
            if out.shape[-1] == 2:
                out = out[:, 0]
            scores = out.sigmoid().flatten().cpu().numpy()
            for path, score in zip(valid_paths, scores):
                results.append((path, float(score), "fake" if score > threshold else "real"))
            if (i // batch_size) % 10 == 0:
                print(f"  processed {min(i + batch_size, len(paths))}/{len(paths)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="D³ unlabeled inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image_dir", required=True, help="Directory of images to classify")
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

    paths = collect_images(opt.image_dir)
    if not paths:
        print(f"No images found in {opt.image_dir}")
        return
    print(f"Found {len(paths)} images. Running inference...")

    results = run_inference(model, paths, opt.batch_size, opt.threshold, transform)

    with open(opt.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "score", "prediction"])
        writer.writerows(results)

    scores = np.array([r[1] for r in results])
    n_fake = sum(1 for r in results if r[2] == "fake")
    print(f"\nDone. Results saved to {opt.output}")
    print(f"  Total: {len(results)}  |  Fake: {n_fake}  |  Real: {len(results) - n_fake}")
    print(f"  Score mean: {scores.mean():.3f}  |  std: {scores.std():.3f}")


if __name__ == "__main__":
    main()
