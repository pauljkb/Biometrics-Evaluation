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

# (mean, std) per RGB channel, applied after scaling pixels to [0, 1]
_NORMALIZE_PRESETS = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "arcface": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # maps [0,1] -> [-1,1], standard for insightface/arcface backbones
    "01": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),        # scale to [0,1] only, no further normalization
}


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
    if normalize_type not in _NORMALIZE_PRESETS:
        raise ValueError(
            f"Unknown normalize_type '{normalize_type}'. Choose from: {list(_NORMALIZE_PRESETS)}"
        )

    cv2_interp = _INTERPOLATION_MAP[interpolation_type]
    mean, std = _NORMALIZE_PRESETS[normalize_type]
    mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

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