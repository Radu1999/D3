"""
D³ saliency visualization — three methods:

  gradcam   — GradCAM on the input activations of ViT-L/14's last block.
              Patch positions in the last block's OUTPUT have zero gradient
              (ln_post only consumes the CLS token), so we hook the block's
              INPUT where all 256 patch positions still have nonzero grad.

  occlusion — Replace each 14×14 patch with the dataset mean and record the
              score drop. Matches ViT-L/14's native patch grid exactly.

  ig        — Integrated Gradients w.r.t. input pixels from a zero baseline
              (= mean image in normalised space). Runs `steps` interpolations.

Usage (single image):
  python explain.py --checkpoint ckpt/classifier.pth --image path/to/img.jpg
  python explain.py --checkpoint ckpt/classifier.pth --image img.jpg --method gradcam occlusion
  python explain.py --checkpoint ckpt/classifier.pth --image img.jpg --method ig --ig_steps 50
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer

# ── ViT-L/14 constants ───────────────────────────────────────────
GRID = 16           # 224 / 14 = 16 patches per side
N_PATCHES = GRID * GRID   # 256 patch tokens

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]
_MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD_T  = torch.tensor(STD).view(1, 3, 1, 1)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── model ────────────────────────────────────────────────────────

def load_model(checkpoint: str) -> CLIPModelShuffleAttentionPenultimateLayer:
    """Load D³ with the default ViT-L/14 config (matches training defaults)."""
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[14]
    )
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.attention_head.load_state_dict(sd)
    model.eval().cuda()
    return model


# ── image helpers ────────────────────────────────────────────────

def load_image(path: str):
    """Return (cuda tensor [1,3,224,224], PIL 224×224) from any image path."""
    pil = Image.open(path).convert("RGB").resize((224, 224), Image.LANCZOS)
    tensor = _transform(pil).unsqueeze(0).cuda()
    return tensor, pil


def _to_numpy_img(tensor):
    """Normalised [1,3,224,224] → float32 numpy [224,224,3] in [0,1]."""
    t = tensor.cpu().float() * _STD_T + _MEAN_T
    return t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def _upsample_map(patch_map: np.ndarray, size: int = 224) -> np.ndarray:
    """Upscale a small heatmap to `size`×`size` with bilinear interpolation."""
    return np.array(
        Image.fromarray((patch_map * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR
        )
    ) / 255.0


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


# ── forward that carries gradients through the original CLIP branch ──

def _forward_grad(model, img_tensor):
    """
    Mirrors CLIPModelShuffleAttentionPenultimateLayer.forward but keeps the
    computation graph alive through the original branch so we can backprop.

    Shuffled branch is still detached (no_grad) — consistent with training.

    Returns the raw logit (pre-sigmoid) with a grad_fn.
    """
    clip = model.model

    # Shuffled branch — no grad, exactly as in training
    with torch.no_grad():
        shuffled = model.shuffle_patches(img_tensor.detach(), model.patch_size[0])
        clip.encode_image(shuffled)
        shuffled_feat = model.features.clone().float().detach()

    # Original branch — build computation graph so backprop can flow back
    # img_tensor is fp32; CLIP expects fp16 (convert_weights made it fp16)
    clip.encode_image(img_tensor.half())
    # model.features is set by the ln_post forward hook; clone() preserves grad_fn
    orig_feat = model.features.float()

    feats = torch.stack([shuffled_feat, orig_feat], dim=-2)  # [1, 2, 1024]
    return model.attention_head(feats)   # [1, 1] logit


# ── GradCAM ──────────────────────────────────────────────────────

def compute_gradcam(model, img_tensor) -> np.ndarray | None:
    """
    GradCAM on the INPUT activations of the last ViT-L/14 ResidualAttentionBlock.

    Why the input, not the output?
      ln_post only reads x[:,0,:] (the CLS token), so the patch-position outputs
      of the last block have ∂score/∂output = 0 and GradCAM would be all zeros.
      The block's INPUT is the output of layer 22.  There, every patch position
      has a nonzero gradient because the CLS token's output in layer 23 is
      computed via self-attention over all 257 input tokens.

    Returns a [224,224] float32 heatmap in [0,1], or None on failure.
    """
    vit = model.model.visual
    last_block = list(vit.transformer.resblocks.children())[-1]

    _inp = [None]

    def pre_hook(module, inp):
        # Detach so this becomes a leaf tensor with requires_grad=True.
        # Leaf tensors always retain their gradient — no retain_grad() needed.
        # Detaching here also means the graph below block-23 is cut, which is
        # fine: GradCAM only needs grad w.r.t. the last block's input activations.
        x = inp[0].detach().requires_grad_(True)
        _inp[0] = x
        return (x,) + inp[1:]

    fh = last_block.register_forward_pre_hook(pre_hook)
    try:
        model.zero_grad()
        with torch.enable_grad():
            img_g = img_tensor.float().detach().requires_grad_(True)
            score = _forward_grad(model, img_g).sigmoid()
            score.backward()

        if _inp[0] is None or _inp[0].grad is None:
            print("GradCAM: gradient did not reach the target layer.")
            return None

        # Patch tokens: indices 1:257 (index 0 = CLS), squeeze batch dim
        act  = _inp[0][1:, 0, :].float().detach()   # [256, 1024]
        grad = _inp[0].grad[1:, 0, :].float()        # [256, 1024]

        # Pool gradients over patch positions → per-channel importance weight
        weights = grad.mean(dim=0)                   # [1024]
        cam = F.relu((act * weights).sum(dim=-1))    # [256]

        heatmap = _normalise(cam.reshape(GRID, GRID).cpu().numpy())
        return _upsample_map(heatmap)

    finally:
        fh.remove()


# ── Occlusion ────────────────────────────────────────────────────

def compute_occlusion(
    model, img_tensor,
    patch_size: int = 14,
    stride: int | None = None,
    batch_size: int = 32,
    smooth_sigma: float = 1.5,
):
    """
    Sliding-window occlusion: replace each (patch_size × patch_size) window with
    the dataset mean (= 0 in normalised space) and accumulate the score drop at
    pixel level.  Overlapping windows are averaged, giving a full-resolution map.

    Improvements over the naive approach:
      • stride < patch_size  — sub-patch resolution via overlapping windows
      • batched inference     — all masks run in mini-batches for speed
      • pixel-level accumulation — no upsampling artefacts
      • Gaussian smoothing   — reduces grid-pattern noise before normalisation

    scores > 0  →  occluding this region lowered the fake score  →  supports fake
    scores < 0  →  occluding this region raised the fake score   →  suppresses fake

    Returns (heatmap [224,224] in [0,1], raw_pixel_scores [224,224]).
    """
    H, W = 224, 224
    if stride is None:
        stride = patch_size // 2   # 2× resolution by default

    with torch.no_grad():
        baseline = model(img_tensor).sigmoid().item()

    # Enumerate all sliding-window positions
    positions = [
        (r0, r0 + patch_size, c0, c0 + patch_size)
        for r0 in range(0, H - patch_size + 1, stride)
        for c0 in range(0, W - patch_size + 1, stride)
    ]

    scores = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_pos = positions[start: start + batch_size]
            n = len(batch_pos)
            batch = img_tensor.expand(n, -1, -1, -1).clone()
            for k, (r0, r1, c0, c1) in enumerate(batch_pos):
                batch[k, :, r0:r1, c0:c1] = 0.0   # occlude with dataset mean
            s = model(batch).sigmoid().squeeze(-1).cpu().numpy()   # [n]
            scores.extend((baseline - s).tolist())

    # Accumulate per-pixel score drops (overlapping windows → average)
    pixel_scores = np.zeros((H, W), dtype=np.float32)
    pixel_counts = np.zeros((H, W), dtype=np.float32)
    for (r0, r1, c0, c1), sc in zip(positions, scores):
        pixel_scores[r0:r1, c0:c1] += sc
        pixel_counts[r0:r1, c0:c1] += 1.0
    pixel_scores /= np.maximum(pixel_counts, 1.0)

    # Gaussian smoothing to reduce grid-pattern noise
    if smooth_sigma > 0:
        pixel_scores = gaussian_filter(pixel_scores, sigma=smooth_sigma)

    # Symmetric normalisation: 0-drop → 0.5 (neutral), pos → >0.5, neg → <0.5
    abs_max = max(abs(pixel_scores.min()), abs(pixel_scores.max())) + 1e-8
    heatmap = _normalise((pixel_scores + abs_max) / (2 * abs_max))
    return heatmap, pixel_scores


# ── Attention Rollout ────────────────────────────────────────────

def compute_attention_rollout(model, img_tensor) -> np.ndarray | None:
    """
    Attention Rollout over CLIP ViT-L/14.  Hooks each block's forward to
    extract Q,K projections and compute per-layer attention matrices, then
    multiplies them with identity residuals to roll out CLS→patch importance.

    Returns a [224,224] float32 heatmap in [0,1], or None on failure.
    """
    vit = model.model.visual
    blocks = list(vit.transformer.resblocks.children())

    stored = []   # one list per block, each holding a single [L,L] tensor

    def make_hook(idx):
        def hook(module, inp, out):
            x = inp[0].detach().float()   # [L, 1, D]  (LND convention)
            attn_mod = module.attn
            L, B, D = x.shape
            H = attn_mod.num_heads
            head_dim = D // H

            w = attn_mod.in_proj_weight.float()
            b = attn_mod.in_proj_bias.float() if attn_mod.in_proj_bias is not None else None
            xf = x.reshape(L * B, D)
            qkv = xf @ w.T
            if b is not None:
                qkv = qkv + b
            q, k, _ = qkv.chunk(3, dim=-1)
            q = q.reshape(L, B, H, head_dim).permute(1, 2, 0, 3)
            k = k.reshape(L, B, H, head_dim).permute(1, 2, 0, 3)
            A = (q @ k.transpose(-2, -1) * (head_dim ** -0.5)).softmax(dim=-1)
            stored[idx] = A.mean(dim=1)[0].cpu()   # [L, L]
        return hook

    stored = [None] * len(blocks)
    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]

    try:
        with torch.no_grad():
            model(img_tensor)
    finally:
        for h in handles:
            h.remove()

    if any(s is None for s in stored):
        print("Attention rollout: some layers did not fire.")
        return None

    # Rollout: R = (A_L + I)/2 @ … @ (A_1 + I)/2
    # Adding identity models the residual connection; /2 re-normalises rows.
    L = stored[0].shape[0]
    rollout = torch.eye(L)
    for A in stored:
        A_res = (A + torch.eye(L)) / 2.0
        # re-normalise rows so they sum to 1
        A_res = A_res / A_res.sum(dim=-1, keepdim=True)
        rollout = A_res @ rollout

    # CLS token row: how much each token flows into CLS
    cls_row = rollout[0, 1:]   # [256] — drop CLS→CLS entry
    heatmap = _normalise(cls_row.reshape(GRID, GRID).numpy())
    return _upsample_map(heatmap)


# ── Integrated Gradients ─────────────────────────────────────────

def compute_integrated_gradients(
    model, img_tensor, steps: int = 50
) -> np.ndarray | None:
    """
    IG w.r.t. input pixels.  Baseline = all-zero normalised image (= dataset mean).
    Gradient flows only through the original CLIP branch (shuffled is detached),
    which is consistent with training — the shuffled branch uses a random patch
    permutation and is not deterministic w.r.t. the input pixels.

    Returns a [224,224] float32 heatmap in [0,1], or None on failure.
    """
    base  = torch.zeros_like(img_tensor)   # 0 in normalised space = mean image
    delta = (img_tensor - base).float()    # [1, 3, 224, 224]

    grads = []
    model.zero_grad()

    with torch.enable_grad():
        for alpha in torch.linspace(0.0, 1.0, steps):
            # Interpolated image between baseline and actual image
            interp = (base + alpha.item() * delta).cuda()
            interp.requires_grad_(True)

            score = _forward_grad(model, interp).sigmoid()
            score.backward()

            if interp.grad is not None:
                grads.append(interp.grad.detach().cpu())

            model.zero_grad()

    if not grads:
        print("IG: no gradients collected.")
        return None

    # Riemann approximation of the integral
    avg_grads = torch.stack(grads).mean(dim=0)   # [1, 3, 224, 224]
    ig = (delta.cpu() * avg_grads)               # element-wise product

    # Sum absolute attributions over colour channels → spatial importance map
    # Using abs() prevents positive/negative channel contributions from cancelling
    heatmap = _normalise(ig.squeeze(0).abs().sum(dim=0).numpy())
    return heatmap


# ── visualisation ────────────────────────────────────────────────

def _overlay(img_np, heatmap, cmap="jet", alpha=0.45):
    """Blend a [0,1] heatmap over an RGB image, both [224,224,(3)]."""
    colored = plt.get_cmap(cmap)(heatmap)[:, :, :3]
    return (alpha * img_np + (1 - alpha) * colored).clip(0, 1)


def save_visualization(img_pil, results, score, out_path):
    """
    img_pil  : PIL 224×224
    results  : list of (title_str, heatmap_or_None)
    score    : float, model output after sigmoid
    out_path : where to write the PNG
    """
    n_cols = len(results) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.2))

    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    label  = "FAKE" if score > 0.5 else "REAL"
    color  = "red" if score > 0.5 else "green"

    axes[0].imshow(img_pil)
    axes[0].set_title(f"Input image\nscore {score:.3f}  [{label}]",
                      fontsize=10, color=color, fontweight="bold")
    axes[0].axis("off")

    cmaps = {"GradCAM": "hot", "Occlusion": "RdBu_r", "Integrated Gradients": "hot",
             "Attention Rollout": "hot"}
    for ax, (title, heatmap) in zip(axes[1:], results):
        if heatmap is None:
            ax.text(0.5, 0.5, "failed", ha="center", va="center", fontsize=12)
            ax.axis("off")
        else:
            key = next((k for k in cmaps if k in title), "hot")
            ax.imshow(_overlay(img_np, heatmap, cmap=cmaps[key]))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ── entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="D³ saliency maps for a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to attention_head .pth checkpoint")
    parser.add_argument("--image", required=True,
                        help="Path to a single image file")
    parser.add_argument("--method", nargs="+",
                        choices=["gradcam", "occlusion", "ig", "attn", "all"],
                        default=["all"],
                        help="Which methods to run")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <image>_explain.png)")
    parser.add_argument("--ig_steps", type=int, default=50,
                        help="Number of interpolation steps for IG")
    parser.add_argument("--occlusion_patch", type=int, default=14,
                        help="Occlusion tile size in pixels (14 = ViT patch grid)")
    parser.add_argument("--occlusion_stride", type=int, default=None,
                        help="Stride for sliding-window occlusion (default: patch//2)")
    parser.add_argument("--occlusion_batch", type=int, default=32,
                        help="Mini-batch size for occlusion forward passes")
    parser.add_argument("--occlusion_sigma", type=float, default=1.5,
                        help="Gaussian smoothing sigma applied to occlusion map (0 = off)")
    opt = parser.parse_args()

    methods = set(opt.method)
    if "all" in methods:
        methods = {"gradcam", "occlusion", "ig", "attn"}

    out_path = opt.output or (os.path.splitext(opt.image)[0] + "_explain.png")

    print(f"Loading model …")
    model = load_model(opt.checkpoint)

    print(f"Image: {opt.image}")
    img_tensor, img_pil = load_image(opt.image)

    with torch.no_grad():
        score = model(img_tensor).sigmoid().item()
    label = "FAKE" if score > 0.5 else "REAL"
    print(f"Score: {score:.4f}  [{label}]")

    results = []

    if "gradcam" in methods:
        print("GradCAM …")
        h = compute_gradcam(model, img_tensor)
        results.append(("GradCAM\n(last ViT block input)", h))

    if "occlusion" in methods:
        p = opt.occlusion_patch
        s = opt.occlusion_stride or p // 2
        n_windows = len(range(0, 224 - p + 1, s)) ** 2
        print(f"Occlusion (patch={p}px, stride={s}px, {n_windows} windows, batch={opt.occlusion_batch}) …")
        h, raw = compute_occlusion(
            model, img_tensor,
            patch_size=p,
            stride=s,
            batch_size=opt.occlusion_batch,
            smooth_sigma=opt.occlusion_sigma,
        )
        results.append((f"Occlusion\n(patch={p}px, stride={s}px)", h))
        np.save(os.path.splitext(out_path)[0] + "_occlusion_raw.npy", raw)

    if "ig" in methods:
        print(f"Integrated Gradients (steps={opt.ig_steps}) …")
        h = compute_integrated_gradients(model, img_tensor, steps=opt.ig_steps)
        results.append((f"Integrated Gradients\n(steps={opt.ig_steps})", h))

    if "attn" in methods:
        print("Attention Rollout …")
        h = compute_attention_rollout(model, img_tensor)
        results.append(("Attention Rollout\n(all ViT layers)", h))

    save_visualization(img_pil, results, score, out_path)


if __name__ == "__main__":
    main()
