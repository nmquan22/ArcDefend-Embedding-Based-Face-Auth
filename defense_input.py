import cv2
import numpy as np
from typing import Literal, Dict, Any, Tuple, Callable
from PIL import Image
import io

DefenseMode = Literal["off", "gaussian", "median", "jpeg"]

# --------- Core ops ---------
def _to_bgr(img: np.ndarray) -> np.ndarray:
    # accept RGB/HWC uint8 or float[0..1]; convert to uint8 BGR for OpenCV
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.shape[2] == 3:  # assume RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img  

def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def gaussian_blur(img_rgb: np.ndarray, ksize: int = 3, sigma: float = 0.8) -> np.ndarray:
    k = max(3, ksize | 1)  # odd
    bgr = _to_bgr(img_rgb)
    out = cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma, borderType=cv2.BORDER_REFLECT101)
    return _to_rgb(out)

def median_filter(img_rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = max(3, ksize | 1)
    bgr = _to_bgr(img_rgb)
    out = cv2.medianBlur(bgr, k)
    return _to_rgb(out)

def jpeg_compress(img_rgb: np.ndarray, quality: int = 60) -> np.ndarray:
    # Use PIL for consistent JPEG quality handling
    quality = int(np.clip(quality, 10, 95))
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    pil2 = Image.open(buf).convert("RGB")
    return np.array(pil2)

# --------- Router & config ---------
class InputDefense:
    def __init__(self, mode: DefenseMode = "off", params: Dict[str, Any] = None):
        self.mode: DefenseMode = mode
        self.params = params or {}

    def set(self, mode: DefenseMode, **params):
        self.mode = mode
        self.params.update(params)

    def apply(self, img_rgb: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return img_rgb
        if self.mode == "gaussian":
            return gaussian_blur(
                img_rgb,
                ksize=int(self.params.get("ksize", 3)),
                sigma=float(self.params.get("sigma", 0.8)),
            )
        if self.mode == "median":
            return median_filter(
                img_rgb,
                ksize=int(self.params.get("ksize", 3)),
            )
        if self.mode == "jpeg":
            return jpeg_compress(
                img_rgb,
                quality=int(self.params.get("quality", 60)),
            )
        return img_rgb  # fallback

# --------- Convenience helpers ---------
def defend_frame_before_mtcnn(frame_rgb: np.ndarray, defense: InputDefense) -> np.ndarray:
    """Apply on full frame before face detection/align."""
    return defense.apply(frame_rgb)

def defend_crop_before_embed(face_crop_rgb: np.ndarray, defense: InputDefense) -> np.ndarray:
    """Apply on aligned face crop before embedding."""
    return defense.apply(face_crop_rgb)

# --------- Evaluation: cosine drift & verification rate ---------
def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def evaluate_defense_on_pairs(
    imgs_a: np.ndarray,  # shape [N,H,W,3], RGB uint8 or float
    imgs_b: np.ndarray,  # matched pairs or impostor pairs
    embed_fn: Callable[[np.ndarray], np.ndarray],  # returns (D,) for single image
    defense: InputDefense,
    threshold: float = 0.8,
) -> Dict[str, float]:
    """
    Quick test: compute cosine before/after defense on A images (or both).
    For simplicity we defend A only; toggle as needed.
    """
    assert imgs_a.shape == imgs_b.shape
    N = imgs_a.shape[0]
    cos_before, cos_after = [], []
    tp_before = tp_after = 0

    for i in range(N):
        a0 = imgs_a[i]
        b0 = imgs_b[i]

        # Embeddings before
        ea = embed_fn(a0)
        eb = embed_fn(b0)
        c0 = cosine_sim(ea, eb)
        cos_before.append(c0)
        if c0 >= threshold:
            tp_before += 1

        # Embeddings after (defend A only)
        a1 = defense.apply(a0)
        ea1 = embed_fn(a1)
        c1 = cosine_sim(ea1, eb)
        cos_after.append(c1)
        if c1 >= threshold:
            tp_after += 1

    return {
        "mean_cos_before": float(np.mean(cos_before)),
        "mean_cos_after": float(np.mean(cos_after)),
        "pass_rate_before@thr": tp_before / N,
        "pass_rate_after@thr": tp_after / N,
    }
