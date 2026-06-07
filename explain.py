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

def compute_gradcam(model, img_tensor, block_idx: int = -8) -> np.ndarray | None:
    """
    GradCAM on the output of an intermediate ViT-L/14 ResidualAttentionBlock.

    The key trick: inject a leaf tensor at ln_pre (just before the transformer)
    so that every subsequent intermediate tensor — including the target block's
    output — has requires_grad=True and can call retain_grad().  This means the
    gradient flows naturally through ALL self-attention layers (including the
    target block's own attention), with no graph-cutting at the target layer.

    Gradient at the target block output = ∂score/∂act, computed by backprop
    through the full transformer stack (blocks target → 23 → ln_post → head).

    Why intermediate block (default -8 = layer 16/24)?
      ln_post reads only x[:,0,:] (CLS token). By layer 23 the CLS is a
      fully-global summary; attention to individual patches is near-uniform and
      the gradient is diffuse. Layer 16 still has local-spatial selectivity
      while carrying enough semantics for the deepfake cues.

    Returns a [224,224] float32 heatmap in [0,1], or None on failure.
    """
    vit = model.model.visual
    blocks = list(vit.transformer.resblocks.children())
    target_block = blocks[block_idx]

    _act = [None]
    _leaf = [None]

    def ln_pre_hook(module, inp, out):
        # Replace ln_pre output with a leaf tensor so that every downstream
        # intermediate tensor inherits requires_grad=True.  The permute+transformer
        # after this point are all differentiable, so retain_grad() will work
        # anywhere inside the transformer.
        leaf = out.detach().requires_grad_(True)
        _leaf[0] = leaf
        return leaf

    def target_hook(module, inp, out):
        # This hook fires for both the shuffled branch (torch.no_grad → requires_grad=False)
        # and the original branch (enable_grad → requires_grad=True from the ln_pre leaf).
        # Only retain grad and record on the original branch pass.
        if out.requires_grad:
            out.retain_grad()
            _act[0] = out

    h_pre    = vit.ln_pre.register_forward_hook(ln_pre_hook)
    h_target = target_block.register_forward_hook(target_hook)

    try:
        model.zero_grad()
        with torch.enable_grad():
            img_g = img_tensor.float().detach().requires_grad_(True)
            score = _forward_grad(model, img_g).sigmoid()
            score.backward()

        if _act[0] is None or _act[0].grad is None:
            print("GradCAM: gradient did not reach the target layer.")
            return None

        # Patch tokens: indices 1:257  (index 0 = CLS),  squeeze batch dim
        act  = _act[0][1:, 0, :].float().detach()   # [256, 1024]
        grad = _act[0].grad[1:, 0, :].float()        # [256, 1024]

        # Per-channel importance weights (global-average-pooled over patch positions)
        weights = grad.mean(dim=0)                    # [1024]
        cam = F.relu((act * weights).sum(dim=-1))     # [256]

        heatmap = _normalise(cam.reshape(GRID, GRID).cpu().numpy())
        return _upsample_map(heatmap)

    finally:
        h_pre.remove()
        h_target.remove()


# ── SAM-based semantic occlusion ─────────────────────────────────

def compute_sam_occlusion(
    model,
    img_tensor,
    img_pil,
    sam_checkpoint: str,
    sam_model_type: str = "vit_h",
    points_per_side: int = 32,
    top_k_vis: int = 10,
    out_prefix: str = "",
):
    """
    Semantic occlusion using Segment Anything Model (SAM).

    Instead of occluding fixed 14×14 grid tiles, SAM first segments the image
    into semantically meaningful regions.  Each region is then replaced with the
    dataset mean (0 in normalised space) and the change in fake score is recorded.

      drop[segment] = baseline_score − score_with_segment_occluded
        > 0  →  this region supports the fake detection
        < 0  →  this region suppresses the fake detection

    Two outputs:
      1. A continuous heatmap [224,224] where pixel value = accumulated score
         drop of all segments that cover that pixel.  Used in the main figure.
      2. A ranked segment figure (saved as <out_prefix>_sam_segments.png) showing
         the top_k_vis most influential segments colour-coded by their drop.

    Requires:
      pip install segment-anything
      # download a SAM checkpoint, e.g. vit_h from Meta's model zoo
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("SAM not installed — run: pip install segment-anything")
        return None

    print(f"  Loading SAM ({sam_model_type}) …")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.cuda().eval()

    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,   # drop tiny fragments
    )

    img_np = np.array(img_pil)   # [224, 224, 3]  uint8
    masks  = mask_gen.generate(img_np)
    print(f"  SAM found {len(masks)} segments")

    if not masks:
        return None

    with torch.no_grad():
        baseline = model(img_tensor).sigmoid().item()

    # Score drop per segment
    drops = []
    with torch.no_grad():
        for m in masks:
            seg = m["segmentation"]          # [224, 224] bool
            masked = img_tensor.clone()
            masked[:, :, seg] = 0.0          # occlude with dataset mean
            score = model(masked).sigmoid().item()
            drops.append(baseline - score)

    # ── continuous heatmap ────────────────────────────────────────
    heatmap = np.zeros((224, 224), dtype=np.float32)
    for m, d in zip(masks, drops):
        heatmap[m["segmentation"]] += d

    # Centre around 0, normalise to [0,1] using symmetric scale
    abs_max = max(abs(heatmap.min()), abs(heatmap.max())) + 1e-8
    heatmap_norm = (heatmap + abs_max) / (2 * abs_max)

    # ── ranked segment figure ─────────────────────────────────────
    if out_prefix:
        _save_sam_segments(img_np, masks, drops, baseline, top_k_vis, out_prefix)

    return heatmap_norm


def _save_sam_segments(img_np, masks, drops, baseline, top_k, out_prefix):
    """Save a figure showing the top_k segments ranked by |score drop|."""
    from scipy.ndimage import binary_dilation

    ranked = sorted(zip(drops, masks), key=lambda x: abs(x[0]), reverse=True)
    ranked = ranked[:top_k]

    # Normalise raw drops for consistent colour intensity across shown segments
    pp_vals = [abs(d) for d, m in ranked]
    pp_max  = max(pp_vals) + 1e-8

    cols = min(3, top_k)
    rows = (top_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.5 * rows + 1.0),
                             facecolor="#111")
    axes = np.array(axes).reshape(-1)

    img_f = img_np.astype(np.float32) / 255.0

    for rank, (ax, (drop, m)) in enumerate(zip(axes, ranked)):
        seg  = m["segmentation"]          # [H, W] bool
        n_px = max(seg.sum(), 1)
        pp   = abs(drop) / n_px

        # ── background: desaturate + dim ──────────────────────────
        grey = img_f.mean(axis=2, keepdims=True)
        bg   = img_f * 0.15 + grey * 0.15   # very dark, slightly grey

        # ── segment: full colour + semi-transparent fill ──────────
        intensity = abs(drop) / pp_max       # 0→1 relative to strongest segment
        if drop > 0:                        # supports fake → warm red/orange
            fill_color = np.array([1.0, 0.25 * (1 - intensity), 0.0])
        else:                               # suppresses fake → cool blue
            fill_color = np.array([0.0, 0.45, 1.0])

        fill = np.zeros_like(img_f)
        fill[seg] = fill_color

        vis = bg.copy()
        vis[seg] = img_f[seg] * 0.55 + fill[seg] * 0.45   # segment: image + tint

        # ── border: dilated mask outline ──────────────────────────
        border = binary_dilation(seg, iterations=2) & ~seg
        border_color = np.array([1.0, 0.9, 0.0]) if drop > 0 else np.array([0.4, 0.9, 1.0])
        vis[border] = border_color

        ax.imshow(np.clip(vis, 0, 1), interpolation="lanczos")

        sign      = "▼ fake" if drop > 0 else "▲ fake"
        txt_color = "#ff9955" if drop > 0 else "#55bbff"
        ax.set_title(
            f"#{rank+1}  {sign}\nΔ = {drop:+.3f}   |   {n_px} px",
            fontsize=11, color=txt_color, pad=6,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(ranked):]:
        ax.set_facecolor("#111")
        ax.axis("off")

    plt.suptitle(
        f"Top-{top_k} SAM segments  ·  ranked by |Δ score|  ·  base score {baseline:.3f}",
        fontsize=14, color="white", y=1.01,
    )
    plt.tight_layout(pad=2.0)
    plt.tight_layout()
    seg_path = out_prefix + "_sam_segments.png"
    plt.savefig(seg_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved segment ranking → {seg_path}")


# ── Occlusion-guided SAM occlusion ───────────────────────────────

def compute_guided_sam_occlusion(
    model,
    img_tensor,
    img_pil,
    sam_checkpoint: str,
    sam_model_type: str = "vit_h",
    occlusion_patch: int = 14,
    importance_percentile: int = 75,
    overlap_thresh: float = 0.25,
    points_per_side: int = 32,
    top_k_vis: int = 10,
    out_prefix: str = "",
):
    """
    Two-stage coarse-to-fine occlusion:

    Stage 1 — Grid occlusion (fast, coarse):
      Occlude each (occlusion_patch × occlusion_patch) tile and record score drops.
      Threshold at `importance_percentile` to build a binary "hot region" mask.

    Stage 2 — SAM inside hot regions (precise, semantic):
      Run SAM on the full image to get object-level segments.
      Keep only segments where at least `overlap_thresh` fraction of their pixels
      fall inside the hot region mask.  This discards background blobs like the
      roof or seat that dominated raw-Δ ranking purely due to their size.

    Stage 3 — Re-occlude filtered segments:
      For each kept segment, replace its pixels with the dataset mean and record
      the exact score drop.  Segments are ranked and visualised as before.

    Returns the continuous heatmap [224,224] for the main figure, and saves a
    separate ranked-segment figure at <out_prefix>_sam_guided_segments.png.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("SAM not installed — run: pip install segment-anything")
        return None

    # ── Stage 1: coarse grid occlusion ───────────────────────────
    print("  Stage 1: coarse grid occlusion …")
    _, raw = compute_occlusion(model, img_tensor, patch_size=occlusion_patch)
    # raw: [n_h, n_w]  score drops per tile

    # Upsample to pixel level
    raw_up = np.repeat(np.repeat(raw, occlusion_patch, axis=0), occlusion_patch, axis=1)
    # Pad/crop to exactly 224×224 (in case 224 % patch_size != 0)
    raw_up = raw_up[:224, :224]
    pad_h  = max(0, 224 - raw_up.shape[0])
    pad_w  = max(0, 224 - raw_up.shape[1])
    if pad_h or pad_w:
        raw_up = np.pad(raw_up, ((0, pad_h), (0, pad_w)), mode="edge")

    threshold      = np.percentile(raw_up, importance_percentile)
    important_mask = raw_up >= threshold   # [224, 224] bool

    hot_frac = important_mask.mean()
    print(f"  Hot region covers {hot_frac*100:.1f}% of pixels "
          f"(threshold Δ ≥ {threshold:.4f})")

    # ── Stage 2: SAM + filter by overlap ─────────────────────────
    print(f"  Stage 2: SAM ({sam_model_type}) …")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.cuda().eval()

    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    img_np = np.array(img_pil)
    masks  = mask_gen.generate(img_np)
    print(f"  SAM found {len(masks)} total segments")

    filtered = [
        m for m in masks
        if (m["segmentation"] & important_mask).sum() / max(m["segmentation"].sum(), 1)
           >= overlap_thresh
    ]
    print(f"  {len(filtered)} segments overlap hot region "
          f"(overlap_thresh={overlap_thresh:.0%})")

    if not filtered:
        print("  No segments passed overlap filter — lowering threshold to 0.1")
        filtered = [
            m for m in masks
            if (m["segmentation"] & important_mask).sum() / max(m["segmentation"].sum(), 1)
               >= 0.1
        ]

    if not filtered:
        return None

    # ── Stage 3: re-occlude filtered segments ────────────────────
    print(f"  Stage 3: re-occluding {len(filtered)} segments …")
    with torch.no_grad():
        baseline = model(img_tensor).sigmoid().item()

    drops = []
    with torch.no_grad():
        for m in filtered:
            seg    = m["segmentation"]
            masked = img_tensor.clone()
            masked[:, :, seg] = 0.0
            drops.append(baseline - model(masked).sigmoid().item())

    # ── heatmap ───────────────────────────────────────────────────
    heatmap = np.zeros((224, 224), dtype=np.float32)
    for m, d in zip(filtered, drops):
        heatmap[m["segmentation"]] += d
    abs_max      = max(abs(heatmap.min()), abs(heatmap.max())) + 1e-8
    heatmap_norm = (heatmap + abs_max) / (2 * abs_max)

    if out_prefix:
        _save_sam_segments(
            img_np, filtered, drops, baseline, top_k_vis,
            out_prefix + "_guided",
        )

    return heatmap_norm


# ── Occlusion ────────────────────────────────────────────────────

def compute_occlusion(model, img_tensor, patch_size: int = 14):
    """
    Replace each (patch_size × patch_size) tile with the dataset mean (= 0 in
    normalised space) and record the change in fake score.

    scores[i,j] = baseline_score − occluded_score
      > 0  →  occluding this patch lowered the fake score  →  region supports fake detection
      < 0  →  occluding this patch raised the fake score   →  region suppresses fake detection

    Returns (heatmap [224,224] in [0,1], raw_scores [n_h, n_w]).
    Default patch_size=14 aligns with ViT-L/14's native patch grid.
    """
    H, W = 224, 224
    n_h, n_w = H // patch_size, W // patch_size

    with torch.no_grad():
        baseline = model(img_tensor).sigmoid().item()

    # Dataset mean in normalised space = 0
    occluder_val = 0.0
    raw = np.zeros((n_h, n_w), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_h):
            for j in range(n_w):
                masked = img_tensor.clone()
                r0, r1 = i * patch_size, (i + 1) * patch_size
                c0, c1 = j * patch_size, (j + 1) * patch_size
                masked[:, :, r0:r1, c0:c1] = occluder_val
                raw[i, j] = baseline - model(masked).sigmoid().item()

    # Shift so 0-drop maps to 0.5 (neutral grey), above = important, below = suppressive
    abs_max = max(abs(raw.min()), abs(raw.max())) + 1e-8
    heatmap = _upsample_map((raw + abs_max) / (2 * abs_max))
    return heatmap, raw


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
             "Attention Rollout": "hot", "SAM Occlusion": "RdBu_r",
             "SAM Guided Occlusion": "RdBu_r"}
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
                        choices=["gradcam", "occlusion", "ig", "attn", "sam", "sam_guided", "all"],
                        default=["all"],
                        help="Which methods to run")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <image>_explain.png)")
    parser.add_argument("--ig_steps", type=int, default=50,
                        help="Number of interpolation steps for IG")
    parser.add_argument("--occlusion_patch", type=int, default=14,
                        help="Occlusion tile size in pixels (14 = ViT patch grid)")
    parser.add_argument("--gradcam_layer", type=int, default=-8,
                        help="ViT block index for GradCAM (negative = from end; "
                             "default -8 = layer 16/24, good spatial selectivity)")
    parser.add_argument("--sam_checkpoint", default=None,
                        help="Path to SAM checkpoint (.pth). Required for --method sam")
    parser.add_argument("--sam_model_type", default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model variant (vit_h is most accurate, vit_b fastest)")
    parser.add_argument("--sam_top_k", type=int, default=10,
                        help="Number of top segments to show in the segment ranking figure")
    parser.add_argument("--sam_importance_pct", type=int, default=75,
                        help="Percentile threshold for grid-occlusion hot region (sam_guided only)")
    parser.add_argument("--sam_overlap", type=float, default=0.25,
                        help="Min fraction of a SAM segment inside hot region to keep it (sam_guided only)")
    opt = parser.parse_args()

    methods = set(opt.method)
    if "all" in methods:
        methods = {"gradcam", "occlusion", "ig", "attn"}

    out_path = opt.output or (os.path.splitext(opt.image)[0] + "_explain.png")

    print("Loading model …")
    model = load_model(opt.checkpoint)

    print(f"Image: {opt.image}")
    img_tensor, img_pil = load_image(opt.image)

    with torch.no_grad():
        score = model(img_tensor).sigmoid().item()
    label = "FAKE" if score > 0.5 else "REAL"
    print(f"Score: {score:.4f}  [{label}]")

    results = []

    if "gradcam" in methods:
        layer = opt.gradcam_layer
        n_blocks = len(list(model.model.visual.transformer.resblocks.children()))
        abs_idx = layer if layer >= 0 else n_blocks + layer
        print(f"GradCAM (block {abs_idx}/{n_blocks-1}) …")
        h = compute_gradcam(model, img_tensor, block_idx=layer)
        results.append((f"GradCAM\n(ViT block {abs_idx}/{n_blocks-1} output)", h))

    if "occlusion" in methods:
        p = opt.occlusion_patch
        print(f"Occlusion (patch={p}px, {224//p}×{224//p} grid) …")
        h, raw = compute_occlusion(model, img_tensor, patch_size=p)
        results.append((f"Occlusion\n(patch={p}px)", h))
        # Optionally save raw score matrix alongside the image
        np.save(os.path.splitext(out_path)[0] + "_occlusion_raw.npy", raw)

    if "ig" in methods:
        print(f"Integrated Gradients (steps={opt.ig_steps}) …")
        h = compute_integrated_gradients(model, img_tensor, steps=opt.ig_steps)
        results.append((f"Integrated Gradients\n(steps={opt.ig_steps})", h))

    if "sam_guided" in methods:
        if not opt.sam_checkpoint:
            print("Skipping SAM guided occlusion — pass --sam_checkpoint <path>")
        else:
            print("SAM guided occlusion (coarse→fine) …")
            out_prefix = os.path.splitext(out_path)[0]
            h = compute_guided_sam_occlusion(
                model, img_tensor, img_pil,
                sam_checkpoint=opt.sam_checkpoint,
                sam_model_type=opt.sam_model_type,
                occlusion_patch=opt.occlusion_patch,
                importance_percentile=opt.sam_importance_pct,
                overlap_thresh=opt.sam_overlap,
                top_k_vis=opt.sam_top_k,
                out_prefix=out_prefix,
            )
            results.append(("SAM Guided Occlusion\n(coarse→fine)", h))

    if "attn" in methods:
        print("Attention Rollout …")
        h = compute_attention_rollout(model, img_tensor)
        results.append(("Attention Rollout\n(all ViT layers)", h))

    if "sam" in methods:
        if not opt.sam_checkpoint:
            print("Skipping SAM occlusion — pass --sam_checkpoint <path>")
        else:
            print("SAM semantic occlusion …")
            out_prefix = os.path.splitext(out_path)[0]
            h = compute_sam_occlusion(
                model, img_tensor, img_pil,
                sam_checkpoint=opt.sam_checkpoint,
                sam_model_type=opt.sam_model_type,
                top_k_vis=opt.sam_top_k,
                out_prefix=out_prefix,
            )
            results.append(("SAM Occlusion\n(semantic segments)", h))

    save_visualization(img_pil, results, score, out_path)


if __name__ == "__main__":
    main()
