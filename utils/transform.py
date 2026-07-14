import cv2
import numpy as np
import torch

_INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

# (mean, std) per RGB channel, applied after scaling pixels to [0, 1].
# Shared by both transform_image() and normalize_image() so that a given
# normalize_type string produces identical results regardless of which
# function processes the image.
_NORMALIZE_PRESETS = {
    "01": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),                # leave in [0, 1]
    "-1_1": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),               # [0,1] -> [-1,1]
    "arcface": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),            # alias of "-1_1", standard for insightface/arcface backbones
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}

# Extra alias so old callers using "neg1_1" keep working.
_NORMALIZE_PRESETS["neg1_1"] = _NORMALIZE_PRESETS["-1_1"]


def _resolve_mean_std(normalize_type, device=None, dtype=torch.float32):
    if normalize_type not in _NORMALIZE_PRESETS:
        raise ValueError(
            f"Unknown normalize_type '{normalize_type}'. Choose from: {sorted(_NORMALIZE_PRESETS)}"
        )
    mean, std = _NORMALIZE_PRESETS[normalize_type]
    mean_t = torch.tensor(mean, dtype=dtype, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=dtype, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def transform_image(image_size, normalize_type="imagenet", interpolation_type="bilinear"):
    """
    Returns a callable transform(img) -> torch.Tensor of shape (3, H, W).

    `img` is expected to be a BGR uint8 numpy array, as returned directly by
    cv2.imread/cv2.imdecode — the BGR->RGB conversion happens inside here,
    so callers should NOT pre-convert before passing the image in.
    """
    if interpolation_type not in _INTERPOLATION_MAP:
        raise ValueError(
            f"Unknown interpolation_type '{interpolation_type}'. Choose from: {list(_INTERPOLATION_MAP)}"
        )
    # Validate early, same error behavior as before.
    if normalize_type not in _NORMALIZE_PRESETS:
        raise ValueError(
            f"Unknown normalize_type '{normalize_type}'. Choose from: {sorted(_NORMALIZE_PRESETS)}"
        )

    cv2_interp = _INTERPOLATION_MAP[interpolation_type]
    mean_t, std_t = _resolve_mean_std(normalize_type)
    mean = mean_t.view(3, 1, 1)  # transform() works on a single image (3,H,W), not a batch
    std = std_t.view(3, 1, 1)

    target_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

    def transform(img):
        if img is None:
            raise ValueError("transform_image received a None image (failed to load?)")

        if img.ndim == 2:  # grayscale safety net
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # cv2.resize wants (width, height)
        if (img.shape[1], img.shape[0]) != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2_interp)

        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # HWC -> CHW
        tensor = (tensor - mean) / std
        return tensor

    return transform


def normalize_image(imgs: torch.Tensor, image_size: int, normalize_type: str) -> torch.Tensor:
    """
    Normalize a batch of images that are already scaled to [0, 1].

    Args:
        imgs: Tensor of shape (N, C, H, W), values in [0, 1].
        image_size: Expected spatial size (H == W == image_size). Used only
            for a sanity check here, since resizing should already have
            happened upstream (e.g. via warpAffine or a Resize transform).
        normalize_type: Any key supported by transform_image's presets:
            "01", "-1_1" (alias "arcface"/"neg1_1"), "imagenet", "clip".

    Returns:
        Normalized tensor, same shape as input.
    """
    if imgs.dim() != 4:
        raise ValueError(f"Expected imgs of shape (N, C, H, W), got {imgs.shape}")

    if imgs.shape[-1] != image_size or imgs.shape[-2] != image_size:
        raise ValueError(
            f"Expected spatial size {image_size}x{image_size}, "
            f"got {imgs.shape[-2]}x{imgs.shape[-1]}"
        )

    mean, std = _resolve_mean_std(normalize_type, device=imgs.device, dtype=imgs.dtype)
    return (imgs - mean) / std