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
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


def run_inference(model, paths, batch_size, threshold, transform,
                  labels=None, num_workers=4, collect_embeddings=False):
    """Run inference using a DataLoader for parallel image loading.

    When collect_embeddings=True and the model has an attention_head, a forward
    hook captures the pre-fc embedding (input to attention_head.fc) for every
    sample.  This adds no extra GPU passes.
    """
    dataset = ImageDataset(paths, labels, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    emb_batches = []
    hook_handle = None
    if collect_embeddings and hasattr(model, "attention_head"):
        def _capture_pre_fc(module, inp, out):
            # inp[0]: (B, input_dim * n_tokens) — the flattened vector before fc
            emb_batches.append(inp[0].detach().cpu().float().numpy())
        hook_handle = model.attention_head.fc.register_forward_hook(_capture_pre_fc)

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

    if hook_handle is not None:
        hook_handle.remove()

    embeddings = np.concatenate(emb_batches, axis=0) if emb_batches else None
    return results, valid_labels, embeddings


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


def cluster_threshold_analysis(embeddings, scores, n_pca=50, random_state=418):
    """K-means (k=2) on pre-fc embeddings → optimal threshold via Youden's J.

    Returns
    -------
    optimal_threshold : float
        Score cutpoint that maximises Youden's J against cluster pseudo-labels.
    cluster_labels : np.ndarray  shape (N,)
        Per-sample cluster assignment, aligned so 1 = higher-score cluster (fake).
    pca_2d : np.ndarray  shape (N, 2)
        First two PCA components for scatter plotting.
    youden_curve : (thresholds, j_scores)
        Full sweep so callers can plot it.
    """
    n_samples, n_features = embeddings.shape
    n_components = min(n_pca, n_samples, n_features)
    print(f"\n[Cluster] PCA {n_features}→{n_components} dims on {n_samples} samples...")
    pca_full = PCA(n_components=n_components, random_state=random_state)
    emb_pca = pca_full.fit_transform(embeddings)
    var_explained = pca_full.explained_variance_ratio_.sum()
    print(f"[Cluster] Variance explained by {n_components} PCs: {var_explained*100:.1f}%")

    pca_2d_obj = PCA(n_components=2, random_state=random_state)
    pca_2d = pca_2d_obj.fit_transform(embeddings)

    print("[Cluster] Running KMeans (k=2)...")
    km = KMeans(n_clusters=2, n_init=20, random_state=random_state)
    raw_labels = km.fit_predict(emb_pca)

    # Align: cluster with higher mean score → label 1 (fake)
    mean0 = scores[raw_labels == 0].mean()
    mean1 = scores[raw_labels == 1].mean()
    cluster_labels = raw_labels if mean1 >= mean0 else 1 - raw_labels

    # Sweep thresholds, maximise Youden's J = sensitivity + specificity − 1
    thresholds = np.linspace(0.01, 0.99, 990)
    j_scores = np.array([
        recall_score(cluster_labels, (scores > t).astype(int), zero_division=0)
        + recall_score(1 - cluster_labels, (scores <= t).astype(int), zero_division=0)
        - 1
        for t in thresholds
    ])
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])

    n_fake_cluster = int(cluster_labels.sum())
    n_real_cluster = len(cluster_labels) - n_fake_cluster
    print(f"[Cluster] Cluster sizes — real: {n_real_cluster}, fake: {n_fake_cluster}")
    print(f"[Cluster] Optimal threshold (max Youden's J={j_scores[best_idx]:.4f}): {optimal_threshold:.4f}")

    return optimal_threshold, cluster_labels, pca_2d, (thresholds, j_scores)


def plot_cluster_analysis(results, scores, cluster_labels, pca_2d,
                          threshold, optimal_threshold, youden_curve,
                          labels=None, output_path="cluster_analysis.png"):
    """Three-panel figure: PCA scatter | score distributions | Youden's J curve."""
    thresholds, j_scores = youden_curve
    y_true = np.array(labels) if labels is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: 2D PCA scatter coloured by cluster ---
    ax = axes[0]
    colors = np.where(cluster_labels == 1, "#E05C5C", "#4C8EDA")
    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, s=8, alpha=0.5, linewidths=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C8EDA", label="Cluster: real"),
                        Patch(color="#E05C5C", label="Cluster: fake")], fontsize=9)
    ax.set_title("Pre-fc Embeddings (PCA 2D)", fontsize=12)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    # --- Panel 2: score histograms with both thresholds ---
    ax = axes[1]
    bins = np.linspace(0, 1, 51)
    if y_true is not None:
        ax.hist(scores[y_true == 0], bins=bins, alpha=0.55, color="#4C8EDA", label="Real (GT)")
        ax.hist(scores[y_true == 1], bins=bins, alpha=0.55, color="#E05C5C", label="Fake (GT)")
    else:
        real_mask = cluster_labels == 0
        ax.hist(scores[real_mask],  bins=bins, alpha=0.55, color="#4C8EDA", label="Cluster: real")
        ax.hist(scores[~real_mask], bins=bins, alpha=0.55, color="#E05C5C", label="Cluster: fake")
    ax.axvline(threshold, color="black",  linestyle="--", linewidth=1.5,
               label=f"Current threshold = {threshold}")
    ax.axvline(optimal_threshold, color="#FF8C00", linestyle="-", linewidth=2,
               label=f"Cluster-optimal = {optimal_threshold:.4f}")
    ax.set_xlabel("Score  (0 = real, 1 = fake)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Score Distribution", fontsize=12)
    ax.legend(fontsize=9)

    # --- Panel 3: Youden's J vs threshold ---
    ax = axes[2]
    ax.plot(thresholds, j_scores, color="#7B68EE", linewidth=1.5)
    ax.axvline(optimal_threshold, color="#FF8C00", linestyle="-", linewidth=2,
               label=f"Optimal = {optimal_threshold:.4f}  (J={j_scores.max():.4f})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Current = {threshold}")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Youden's J", fontsize=11)
    ax.set_title("Threshold Sweep (Youden's J)", fontsize=12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Cluster analysis plot saved to {output_path}")


def plot_score_distribution(results, threshold, labels=None, output_path="score_distribution.png"):
    scores = np.array([r[1] for r in results])

    fig, ax = plt.subplots(figsize=(9, 5))

    if labels is not None:
        y_true = np.array(labels)
        real_scores = scores[y_true == 0]
        fake_scores = scores[y_true == 1]
        bins = np.linspace(0, 1, 51)
        ax.hist(real_scores, bins=bins, alpha=0.65, color="#4C8EDA", label=f"Real  (n={len(real_scores)})")
        ax.hist(fake_scores, bins=bins, alpha=0.65, color="#E05C5C", label=f"Fake  (n={len(fake_scores)})")
    else:
        bins = np.linspace(0, 1, 51)
        ax.hist(scores, bins=bins, color="#7B68EE", alpha=0.8, label=f"All images  (n={len(scores)})")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")
    ax.set_xlabel("Score  (0 = real, 1 = fake)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("D³ Score Distribution", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Score distribution saved to {output_path}")


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
    parser.add_argument("--output_json", default="predictions.json", help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes for parallel image loading")
    parser.add_argument("--plot", action="store_true",
                        help="Save a score-distribution histogram to --plot_output")
    parser.add_argument("--plot_output", default="score_distribution.png",
                        help="Path for the score distribution plot (requires --plot)")
    parser.add_argument("--cluster", action="store_true",
                        help="K-means clustering on pre-fc embeddings to find optimal threshold")
    parser.add_argument("--cluster_pca", type=int, default=50,
                        help="PCA components before KMeans (default: 50)")
    parser.add_argument("--subset", type=int, default=0,
                        help="Randomly sample N images before inference (0 = use all)")
    parser.add_argument("--cluster_output", default="cluster_analysis.png",
                        help="Path for the cluster analysis plot (requires --cluster)")
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

    if opt.subset and opt.subset < len(paths):
        rng = np.random.default_rng(418)
        idx = rng.choice(len(paths), size=opt.subset, replace=False)
        idx.sort()
        paths = [paths[i] for i in idx]
        labels = [labels[i] for i in idx] if labels is not None else None
        print(f"Subset: using {opt.subset} randomly sampled images.")

    print(f"Found {len(paths)} images. Running inference...")

    results, valid_labels, embeddings = run_inference(
        model, paths, opt.batch_size, opt.threshold, transform,
        labels=labels, num_workers=opt.num_workers,
        collect_embeddings=opt.cluster,
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

    json_list = [
        {"id": os.path.basename(path), "pred_label": 1 if pred == "fake" else 0}
        for path, score, pred in results
    ]
    with open(opt.output_json, "w") as f:
        json.dump(json_list, f, indent=2)

    scores = np.array([r[1] for r in results])
    n_fake = sum(1 for r in results if r[2] == "fake")

    if opt.cluster and embeddings is not None:
        optimal_threshold, cluster_labels, pca_2d, youden_curve = cluster_threshold_analysis(
            embeddings, scores, n_pca=opt.cluster_pca,
        )
        plot_cluster_analysis(
            results, scores, cluster_labels, pca_2d,
            threshold=opt.threshold,
            optimal_threshold=optimal_threshold,
            youden_curve=youden_curve,
            labels=valid_labels if opt.eval else None,
            output_path=opt.cluster_output,
        )
    elif opt.cluster:
        print("[Cluster] Skipped: model has no attention_head (unsupported arch).")

    if opt.plot:
        plot_score_distribution(
            results, opt.threshold,
            labels=valid_labels if opt.eval else None,
            output_path=opt.plot_output,
        )

    print(f"\nDone. Results saved to {opt.output} and {opt.output_json}")
    print(f"  Total: {len(results)}  |  Fake: {n_fake}  |  Real: {len(results) - n_fake}")
    print(f"  Score mean: {scores.mean():.3f}  |  std: {scores.std():.3f}")

    if opt.eval:
        print_eval_metrics(results, valid_labels, opt.threshold)


if __name__ == "__main__":
    main()
