"""L4 — Gemma 4 vision support: image encode + text↔image embed splice.

L4 is the first kernel layer allowed to depend on external package code. This
module owns everything image-specific so L1 (engine) and L3 (scheduler) only
gain a thin, modality-agnostic ``input_embeddings`` passthrough.

Gemma 4 12B (``gemma4_unified``) has no separate SigLIP tower. Its vision side
is a small ``vision_embedder`` (patchify -> patch_dense 6912->3840 -> norms +
positional embedding) followed by ``embed_vision`` (an RMSNorm-then-Linear
projection into the 3840-dim text space). Those weights live in the
assistant-pair bundle under ``vision/vision.safetensors`` and are NOT loaded by
the text+MTP path, so we load them here on demand.

The preprocessing (resize/patchify/normalize) is re-implemented from mlx-vlm's
``processing_gemma4_unified`` / ``processing_gemma4`` algorithm using only
numpy/PIL/mlx — we deliberately do NOT import transformers (its processor chain
collides with the installed mlx-vlm↔transformers versions). The forward
(``VisionEmbedder`` + ``embed_vision``) mirrors mlx-vlm's modules exactly so the
same weights reproduce the same image features (verified allclose).

Scale note: the mlx-vlm unified model scales the TEXT embeddings by
``embed_scale`` before scattering image features (image features enter
un-scaled). The mlx-lm ``gemma4_text`` trunk we run instead multiplies WHATEVER
``input_embeddings`` it receives by ``embed_scale`` uniformly. So we build the
prefill embeds pre-scale: text = raw ``embed_tokens(ids)`` and image features
divided by ``embed_scale``; after the trunk's uniform multiply both land at the
correct magnitude. See ``build_prefill_embeds``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# --- Gemma 4 unified vision constants (from vision_config.json / processor) ---
IMAGE_TOKEN_ID = 258880      # <|image|>
BOI_TOKEN_ID = 255999        # <|image>  (start of image)
EOI_TOKEN_ID = 258882        # <image|>  (end of image)


@dataclass
class VisionParams:
    model_patch_size: int = 48       # patch_size(16) * pooling_kernel_size(3)
    patch_size: int = 16
    pooling_kernel_size: int = 3
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120
    num_soft_tokens: int = 280       # max soft tokens per image
    output_proj_dims: int = 3840
    rms_norm_eps: float = 1e-6
    text_hidden_size: int = 3840
    do_convert_rgb: bool = True
    do_resize: bool = True
    do_rescale: bool = True
    rescale_factor: float = 1.0 / 255.0
    do_normalize: bool = False
    image_mean: tuple = (0.0, 0.0, 0.0)
    image_std: tuple = (1.0, 1.0, 1.0)

    @property
    def patch_dim(self) -> int:
        return self.model_patch_size * self.model_patch_size * 3

    @property
    def max_patches(self) -> int:
        return self.num_soft_tokens * self.pooling_kernel_size ** 2


# ---------------------------------------------------------------------------
# Forward modules (mirrors mlx_vlm.models.gemma4_unified.VisionEmbedder and
# mlx_vlm.models.gemma4.MultimodalEmbedder so the bundle weights reproduce the
# same image features numerically).
# ---------------------------------------------------------------------------
class _RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale (embed_vision pre-projection norm)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class VisionEmbedder(nn.Module):
    def __init__(self, p: VisionParams):
        super().__init__()
        self.patch_dim = p.patch_dim
        self.patch_ln1 = nn.LayerNorm(p.patch_dim)
        self.patch_dense = nn.Linear(p.patch_dim, p.mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(p.mm_embed_dim)
        self.pos_embedding = mx.zeros((p.mm_posemb_size, 2, p.mm_embed_dim))
        self.pos_norm = nn.LayerNorm(p.mm_embed_dim)

    def __call__(self, pixel_values: mx.array,
                 image_position_ids: Optional[mx.array] = None) -> mx.array:
        if pixel_values.ndim == 4 and pixel_values.shape[-1] == self.patch_dim:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0], -1, self.patch_dim
            )
        h = self.patch_ln1(pixel_values)
        h = self.patch_dense(h)
        h = self.patch_ln2(h)

        if image_position_ids is not None:
            clamped = mx.maximum(image_position_ids, 0).astype(mx.int32)
            valid = (image_position_ids != -1).astype(h.dtype)
            x_pos = self.pos_embedding[clamped[..., 0], 0]
            y_pos = self.pos_embedding[clamped[..., 1], 1]
            h = h + (
                x_pos * mx.expand_dims(valid[..., 0], -1)
                + y_pos * mx.expand_dims(valid[..., 1], -1)
            )
        return self.pos_norm(h)


class MultimodalEmbedder(nn.Module):
    """Projects vision soft tokens into the text embedding space."""

    def __init__(self, embedding_dim: int, text_hidden_size: int, eps: float):
        super().__init__()
        self.embedding_projection = nn.Linear(
            embedding_dim, text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = _RMSNormNoScale(eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


def _compact_prefix_rows(features: mx.array, valid_mask: mx.array) -> mx.array:
    """Drop padded soft-token rows (position_ids == -1), flatten to [N, D]."""
    rows = []
    for batch_idx, row in enumerate(valid_mask.tolist()):
        length = sum(bool(v) for v in row)
        if length:
            rows.append(features[batch_idx, :length])
    if not rows:
        return features.reshape(-1, features.shape[-1])[:0]
    return mx.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Preprocessing (numpy/PIL only — no transformers). Mirrors
# processing_gemma4.Gemma4ImageProcessor + processing_gemma4_unified.
# ---------------------------------------------------------------------------
def _load_pil(image):
    from PIL import Image
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, (str, os.PathLike)):
        s = str(image)
        if s.startswith(("http://", "https://")):
            import urllib.request
            from io import BytesIO
            with urllib.request.urlopen(s) as r:
                return Image.open(BytesIO(r.read()))
        return Image.open(s)
    raise TypeError(f"unsupported image input: {type(image)}")


def _aspect_ratio_preserving_resize(img_hw3: np.ndarray, p: VisionParams) -> np.ndarray:
    """Largest resize (channels-last uint8/float) that fits max_patches and is
    divisible by pooling_kernel_size*patch_size. Mirrors mlx-vlm exactly."""
    from PIL import Image
    height, width = img_hw3.shape[0], img_hw3.shape[1]
    target_px = p.max_patches * (p.patch_size ** 2)
    factor = math.sqrt(target_px / (height * width))
    side_mult = p.pooling_kernel_size * p.patch_size

    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width = int(math.floor(factor * width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("Attempting to resize to a 0 x 0 image.")

    max_side_length = (p.max_patches // p.pooling_kernel_size ** 2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height == height and target_width == width:
        return img_hw3

    arr = img_hw3
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil = pil.resize((target_width, target_height), resample=Image.BICUBIC)
    return np.array(pil)


def _convert_image_to_model_patches(image_c_h_w: np.ndarray, model_patch_size: int):
    channels, height, width = image_c_h_w.shape
    patch_height = height // model_patch_size
    patch_width = width // model_patch_size
    patches = image_c_h_w.reshape(
        channels, patch_height, model_patch_size, patch_width, model_patch_size
    )
    patches = patches.transpose(1, 3, 2, 4, 0)
    patches = patches.reshape(
        patch_height * patch_width, model_patch_size * model_patch_size * channels
    )
    grid = np.meshgrid(
        np.arange(patch_width, dtype=np.int64),
        np.arange(patch_height, dtype=np.int64),
        indexing="xy",
    )
    positions = np.stack(grid, axis=-1).reshape(-1, 2)
    return patches.astype(np.float32), positions


def _pad_patches(patches: np.ndarray, positions: np.ndarray, target_length: int):
    current = patches.shape[0]
    if current > target_length:
        return patches[:target_length], positions[:target_length]
    pad = target_length - current
    if pad == 0:
        return patches, positions
    patches = np.pad(patches, ((0, pad), (0, 0)), mode="constant", constant_values=0)
    positions = np.pad(positions, ((0, pad), (0, 0)), mode="constant", constant_values=-1)
    return patches, positions


def preprocess_image(image, p: VisionParams):
    """One image -> (pixel_values [1, num_soft_tokens, patch_dim],
    image_position_ids [1, num_soft_tokens, 2], num_real_patches).

    Channels-last numpy pipeline (like mlx-vlm with input_data_format=LAST)."""
    pil = _load_pil(image)
    if p.do_convert_rgb and pil.mode != "RGB":
        pil = pil.convert("RGB")
    image = np.array(pil)  # H, W, 3 (channels-last)

    if p.do_resize:
        image = _aspect_ratio_preserving_resize(image, p)

    if p.do_rescale:
        image = image.astype(np.float32) * p.rescale_factor

    if p.do_normalize:
        mean = np.array(p.image_mean, dtype=np.float32)
        std = np.array(p.image_std, dtype=np.float32)
        image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))  # channels-first
    patches, positions = _convert_image_to_model_patches(image, p.model_patch_size)
    num_real = int(patches.shape[0])
    patches, positions = _pad_patches(patches, positions, p.num_soft_tokens)
    return (
        np.stack([patches]),
        np.stack([positions]),
        num_real,
    )


# ---------------------------------------------------------------------------
# Encoder: load bundle vision weights and produce image features.
# ---------------------------------------------------------------------------
def _bundle_vision_dir(model_path: str) -> Optional[str]:
    """Given an assistant-pair bundle root (or its target/ subdir), find the
    sibling vision/ directory holding vision.safetensors."""
    for cand in (model_path, os.path.dirname(model_path.rstrip("/"))):
        vdir = os.path.join(cand, "vision")
        if os.path.isfile(os.path.join(vdir, "vision.safetensors")):
            return vdir
    return None


def _load_vision_params(vision_dir: str) -> VisionParams:
    p = VisionParams()
    cfg_path = os.path.join(vision_dir, "vision_config.json")
    try:
        cfg = json.load(open(cfg_path))
    except (OSError, json.JSONDecodeError):
        return p
    vc = cfg.get("vision_config", {}) or {}
    p.model_patch_size = vc.get("model_patch_size", p.model_patch_size)
    p.patch_size = vc.get("patch_size", p.patch_size)
    p.pooling_kernel_size = vc.get("pooling_kernel_size", p.pooling_kernel_size)
    p.mm_embed_dim = vc.get("mm_embed_dim", p.mm_embed_dim)
    p.mm_posemb_size = vc.get("mm_posemb_size", p.mm_posemb_size)
    p.num_soft_tokens = vc.get("num_soft_tokens", p.num_soft_tokens)
    p.output_proj_dims = vc.get("output_proj_dims", p.output_proj_dims)
    p.rms_norm_eps = vc.get("rms_norm_eps", p.rms_norm_eps)
    return p


def _load_preprocessor_overrides(model_path: str, p: VisionParams) -> None:
    """Apply processor_config.json image_processor settings if present next to
    the model (matches mlx-vlm reading preprocessor/processor config)."""
    for cand in (model_path, os.path.dirname(model_path.rstrip("/"))):
        for name in ("processor_config.json", "preprocessor_config.json"):
            path = os.path.join(cand, name)
            if not os.path.isfile(path):
                continue
            try:
                cfg = json.load(open(path))
            except (OSError, json.JSONDecodeError):
                continue
            ip = cfg.get("image_processor", cfg) or {}
            if "do_convert_rgb" in ip:
                p.do_convert_rgb = ip["do_convert_rgb"]
            if "do_resize" in ip:
                p.do_resize = ip["do_resize"]
            if "do_rescale" in ip:
                p.do_rescale = ip["do_rescale"]
            if "rescale_factor" in ip:
                p.rescale_factor = ip["rescale_factor"]
            if "do_normalize" in ip:
                p.do_normalize = ip["do_normalize"]
            if ip.get("image_mean") is not None:
                p.image_mean = tuple(ip["image_mean"])
            if ip.get("image_std") is not None:
                p.image_std = tuple(ip["image_std"])
            if "max_soft_tokens" in ip:
                p.num_soft_tokens = ip["max_soft_tokens"]
            if "patch_size" in ip:
                p.patch_size = ip["patch_size"]
            if "pooling_kernel_size" in ip:
                p.pooling_kernel_size = ip["pooling_kernel_size"]
            if "model_patch_size" in ip:
                p.model_patch_size = ip["model_patch_size"]
            return


class VisionEncoder:
    """Loads the bundle's Gemma 4 vision weights and encodes images into text
    embedding space. Instantiated lazily by the hub/harness only when an image
    request arrives, so the text-only path pays nothing."""

    def __init__(self, model_path: str):
        vision_dir = _bundle_vision_dir(model_path)
        if vision_dir is None:
            raise FileNotFoundError(
                f"no vision/vision.safetensors found for bundle at {model_path!r}"
            )
        self.vision_dir = vision_dir
        self.params = _load_vision_params(vision_dir)
        _load_preprocessor_overrides(model_path, self.params)
        self._build_and_load()

    def _build_and_load(self) -> None:
        p = self.params
        self.vision_embedder = VisionEmbedder(p)
        self.embed_vision = MultimodalEmbedder(
            p.output_proj_dims, p.text_hidden_size, p.rms_norm_eps
        )
        weights = mx.load(os.path.join(self.vision_dir, "vision.safetensors"))
        # Keys are prefixed "model." in the bundle safetensors.
        ve, ev = {}, {}
        for k, v in weights.items():
            if k.startswith("model.vision_embedder."):
                ve[k[len("model.vision_embedder."):]] = v
            elif k.startswith("model.embed_vision."):
                ev[k[len("model.embed_vision."):]] = v
        self.vision_embedder.load_weights(list(ve.items()))
        self.embed_vision.load_weights(list(ev.items()))
        self.vision_embedder.eval()
        self.embed_vision.eval()
        mx.eval(self.vision_embedder.parameters(), self.embed_vision.parameters())

    def encode(self, image) -> tuple[mx.array, int]:
        """Preprocess + encode one image. Returns (features [N, H], N) where N
        is the real (unpadded) soft-token count and features are in text space."""
        pixel_values, position_ids, num_real = preprocess_image(image, self.params)
        pv = mx.array(pixel_values)
        pid = mx.array(position_ids)
        return self.image_features(pv, pid), num_real

    def image_features(self, pixel_values: mx.array,
                       image_position_ids: Optional[mx.array] = None) -> mx.array:
        """VisionEmbedder -> embed_vision projection, then drop padded rows.
        Mirrors gemma4_unified.Model.get_image_features."""
        embedded = self.vision_embedder(pixel_values, image_position_ids)
        projected = self.embed_vision(embedded)
        if image_position_ids is None:
            return projected.reshape(-1, projected.shape[-1])
        padding_mask = mx.all(image_position_ids == -1, axis=-1)
        return _compact_prefix_rows(projected, ~padding_mask)


# ---------------------------------------------------------------------------
# Text/image token stream + embed splice.
# ---------------------------------------------------------------------------
def expand_image_placeholders(token_ids: list[int], counts: list[int]) -> list[int]:
    """Replace each single ``<|image|>`` (IMAGE_TOKEN_ID) placeholder emitted by
    the chat template with ``BOI + IMAGE_TOKEN*n + EOI`` for the matching image,
    where n is that image's real soft-token count. Returns the expanded ids.

    The chat template emits one image_token per image; the real prompt needs n
    placeholders bracketed by begin/end-of-image markers (mlx-vlm does the same
    expansion in its processor)."""
    out: list[int] = []
    it = iter(counts)
    for tid in token_ids:
        if tid == IMAGE_TOKEN_ID:
            try:
                n = next(it)
            except StopIteration:
                raise ValueError(
                    "more <|image|> placeholders than provided images"
                )
            out.append(BOI_TOKEN_ID)
            out.extend([IMAGE_TOKEN_ID] * n)
            out.append(EOI_TOKEN_ID)
        else:
            out.append(tid)
    remaining = list(it)
    if remaining:
        raise ValueError("more images than <|image|> placeholders in prompt")
    return out


def build_prefill_embeds(model, token_ids: list[int],
                         image_features: list[mx.array]) -> mx.array:
    """Build the pre-scale prefill embeds [1, L, H] for an image request.

    - text positions: raw embed_tokens(id) (NO embed_scale; the trunk applies it)
    - image_token positions: image features / embed_scale (so after the trunk's
      uniform *embed_scale multiply they match mlx-vlm's un-scaled scatter)

    ``image_features`` is a list of [n_i, H] arrays, one per image, in prompt
    order. The count of IMAGE_TOKEN_ID positions must equal sum(n_i)."""
    lm = model.language_model
    embed_scale = lm.model.embed_scale
    ids = mx.array([token_ids], dtype=mx.int32)
    text_embeds = lm.model.embed_tokens(ids)          # [1, L, H], un-scaled

    if not image_features:
        return text_embeds

    feats = mx.concatenate(image_features, axis=0)     # [sum n_i, H]
    feats = feats.astype(text_embeds.dtype) / embed_scale

    mask = np.array(token_ids) == IMAGE_TOKEN_ID
    n_img = int(mask.sum())
    if n_img != int(feats.shape[0]):
        raise ValueError(
            f"image placeholder count {n_img} != image feature rows "
            f"{int(feats.shape[0])}"
        )
    idx = np.nonzero(mask)[0]
    embeds = text_embeds
    # Scatter each image feature row into its placeholder position.
    embeds[0, mx.array(idx)] = feats
    return embeds
