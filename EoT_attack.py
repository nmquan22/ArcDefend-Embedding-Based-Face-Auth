# --- thêm vào script của bạn (drop-in) ---
import torch
import torch.nn.functional as F
import random
import math
import numpy as np
from PIL import Image
import cv2
import os, sys
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from typing import Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 160
EPSILON = 16.0/255.0
PGD_STEPS = 60
PGD_ALPHA = 1.0/255.0
PGD_RESTARTS = 3
USE_MOMENTUM = True
MOMENTUM_MU = 1.0
SAVE_PREFIX = "resultAdv/adv_eot_pgd"
SAVE_PATH = f"{SAVE_PREFIX}_best.png"
MODE = "pgd_eot"   # "fgsm_eot" or "pgd_eot"

cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

def to_pil(tensor: torch.Tensor) -> Image.Image:
    # tensor (1,C,H,W) or (C,H,W) in [0,1]
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().permute(1,2,0).numpy()
    arr = (arr * 255.0).clip(0,255).astype('uint8')
    return Image.fromarray(arr)

def tensor_to_bgr_image(tensor: torch.Tensor) -> np.ndarray:
    pil = to_pil(tensor)
    arr = np.array(pil)[:,:,::-1]  # RGB->BGR
    return arr

def save_diff_heatmap(orig_tensor: torch.Tensor, adv_tensor: torch.Tensor, out_path: str):
    o = to_pil(orig_tensor).convert('L')
    a = to_pil(adv_tensor).convert('L')
    diff = np.abs(np.array(a).astype(np.int16) - np.array(o).astype(np.int16)).astype(np.uint8)
    diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imwrite(out_path, diff)

# Facenet InceptionResnet expects inputs in range [-1, 1]
def normalize_for_facenet(x: torch.Tensor) -> torch.Tensor:
    # x expected in [0,1], returns in [-1,1]
    return (x - 0.5) * 2.0

# MTCNN wrapper: returns tensor (1, c, IMG_SIZE, IMG_SIZE) in [0,1]
def crop_face_tensor(mtcnn, pil_image: Image.Image, device=DEVICE):
    face = mtcnn(pil_image)  
    if face is None:
        return None
    if face.ndimension() == 3:
        face = face.unsqueeze(0)
    return face.to(device)

def extract_embedding(resnet, face_tensor: torch.Tensor):
    # face_tensor: (B,C,H,W) in [0,1] -> convert to [-1,1] then pass
    #x = normalize_for_facenet(face_tensor)
    x = face_tensor.to(device=DEVICE).float()
    with torch.no_grad():
        emb = resnet(x)
    return emb  # (B, D), torch.tensor on device


# --- EoT: tạo các biến đổi ngẫu nhiên trực tiếp trên biến "x" có requires_grad ---
# Input/Output: x dạng (1,C,H,W) float [0,1]
def random_affine_batch_on(x, tx=0.03, ty=0.03, scale_range=(0.97, 1.03),
                           rot_deg=3.0, bright=0.05, samples=8):
    """
    Trả về list các tensor đã biến đổi, MỖI tensor vẫn là hàm của x (giữ đồ thị autograd).
    """
    b, c, h, w = x.shape
    outs = []
    for _ in range(samples):
        angle = (torch.rand(())*2 - 1) * rot_deg
        scale = torch.empty(()).uniform_(scale_range[0], scale_range[1])
        trans_x = (torch.rand(())*2 - 1) * tx
        trans_y = (torch.rand(())*2 - 1) * ty

        ca = torch.cos(torch.deg2rad(angle))
        sa = torch.sin(torch.deg2rad(angle))
        theta = torch.stack([
            torch.stack([ scale*ca, -scale*sa, trans_x ]),
            torch.stack([ scale*sa,  scale*ca, trans_y ])
        ], dim=0).unsqueeze(0).to(x.dtype).to(x.device)  # (1,2,3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_t = F.grid_sample(x, grid, padding_mode='border', align_corners=False)

        # brightness jitter (vẫn khả vi)
        delta = (torch.rand(())*2 - 1) * bright
        x_t = (x_t + delta).clamp(0.0, 1.0)
        outs.append(x_t)
    return outs


# --- FGSM EoT (targeted): trung bình gradient qua các biến đổi ---
def fgsm_targeted_eot(model, crop_tensor, target_emb, epsilon=8/255., samples=12, device='cpu'):
    """
    model: InceptionResnetV1 (eval)
    crop_tensor: (1,C,H,W) [0,1]
    target_emb: (1,D)
    """
    model.eval().to(device)
    x_orig = crop_tensor.to(device).detach()
    x = x_orig.clone().detach().requires_grad_(True)

    total_grad = torch.zeros_like(x)
    loss_acc = 0.0

    # EoT: áp biến đổi lên "x" (không phải x_orig) để còn gradient về x
    trans_list = random_affine_batch_on(x, samples=samples)
    for t in trans_list:
        t_n = normalize_for_facenet(t)              # chuẩn hoá cho FaceNet
        emb = model(t_n)                            # (1,D)
        loss = 1.0 - F.cosine_similarity(emb, target_emb, dim=1).mean()
        # lấy grad theo x qua t (t phụ thuộc x)
        g = torch.autograd.grad(loss, x, retain_graph=True)[0]
        total_grad = total_grad + g
        loss_acc += float(loss.detach().cpu().item())

    avg_grad = total_grad / float(samples)

    # Bước FGSM (targeted => giảm loss => đi NGƯỢC dấu grad)
    adv = (x_orig - epsilon * torch.sign(avg_grad)).clamp(0.0, 1.0).detach()

    with torch.no_grad():
        sim_before = float(F.cosine_similarity(model(normalize_for_facenet(x_orig)),
                                               target_emb, dim=1).item())
        sim_after  = float(F.cosine_similarity(model(normalize_for_facenet(adv)),
                                               target_emb, dim=1).item())
    stats = {'sim_before': sim_before,
             'sim_after':  sim_after,
             'loss_avg':   loss_acc / float(samples),
             'epsilon':    float(epsilon)}
    return adv, stats


# --- PGD EoT (targeted) với momentum ---
def pgd_targeted_eot(model, crop_tensor, target_emb, eps=16/255., steps=40, alpha=1/255.,
                     samples=8, device='cpu', momentum=0.9):
    """
    Iterative PGD với EoT sampling mỗi bước; targeted: minimize (1 - cos).
    """
    model.eval().to(device)
    x_orig = crop_tensor.to(device).detach()
    adv = x_orig.clone().detach()
    g_m = torch.zeros_like(adv)

    for i in range(steps):
        adv = adv.detach().requires_grad_(True)

        total_grad = torch.zeros_like(adv)
        loss_acc = 0.0

        # EoT trên chính "adv"
        trans_list = random_affine_batch_on(adv, samples=samples)
        for t in trans_list:
            t_n = normalize_for_facenet(t)
            emb = model(t_n)
            loss = 1.0 - F.cosine_similarity(emb, target_emb, dim=1).mean()
            g = torch.autograd.grad(loss, adv, retain_graph=True)[0]
            total_grad = total_grad + g
            loss_acc += float(loss.detach().cpu().item())

        avg_grad = total_grad / float(samples)

        # Momentum (chuẩn hoá để ổn định)
        denom = avg_grad.abs().mean().clamp(min=1e-12)
        g_m = momentum * g_m + avg_grad / denom

        # Targeted: giảm loss => trừ theo sign(gradient)
        adv = adv.detach() - alpha * torch.sign(g_m)

        # Project vào L_inf-ball và clamp
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
        adv = adv.clamp(0.0, 1.0).detach()

        if (i+1) % 10 == 0:
            with torch.no_grad():
                sim_cur = float(F.cosine_similarity(model(normalize_for_facenet(adv)),
                                                    target_emb, dim=1).item())
            print(f"[PGD-EoT] iter {i+1}/{steps} sim={sim_cur:.4f} avg_loss={loss_acc/samples:.4f}")

    with torch.no_grad():
        sim_before = float(F.cosine_similarity(model(normalize_for_facenet(x_orig)),
                                               target_emb, dim=1).item())
        sim_after  = float(F.cosine_similarity(model(normalize_for_facenet(adv)),
                                               target_emb, dim=1).item())
    stats = {'sim_before': sim_before, 'sim_after': sim_after,
             'eps': float(eps), 'steps': steps, 'alpha': float(alpha), 'samples': samples}
    return adv, stats

# --- helper: paste adv crop into full image at bbox (simple resize and replace) ---
def paste_adv_to_fullframe(adv_crop, pil_img_full, mtcnn):
    """
    adv_crop: tensor 1xC x H x W in [0,1] (torch)
    pil_img_full: PIL.Image RGB (original full-frame)
    mtcnn: MTCNN instance used to obtain bbox for original image
    Returns: PIL.Image RGB with adv_crop pasted into first detected face bbox (or None if no bbox)
    """
    # detect boxes using mtcnn (returns boxes in pixels)
    boxes, probs = mtcnn.detect(pil_img_full)
    if boxes is None or len(boxes) == 0:
        return None
    # choose the highest-prob box
    best_idx = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[best_idx].astype(int)
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    # resize adv_crop to (h,w)
    adv_np = (adv_crop.clamp(0,1).cpu().numpy()[0].transpose(1,2,0) * 255.0).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np).resize((w, h), resample=Image.BILINEAR)
    full = pil_img_full.copy()
    full.paste(adv_pil, (x1, y1, x2, y2))
    return full

mtcnn = MTCNN(image_size=IMG_SIZE, margin=0, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

pathA = "D:\\ArcDefend\\Image\\man1.jpg"
pathB = "D:\\ArcDefend\\Image\\man2.jpg"

if not os.path.exists(pathA) or not os.path.exists(pathB):
    print("Please put imageA and imageB in the paths or change paths in the script.")
    sys.exit(1)
# ----------------------------------------------------------------
# ---- Integration snippet: replace your mode-choice block with this
# assume: mtcnn, resnet defined; pathA, pathB defined; MODE, EPSILON, PGD_STEPS, PGD_ALPHA, etc.
# ----------------------------------------------------------------

# load images and crops like you already do
imgA = Image.open(pathA).convert("RGB")
imgB = Image.open(pathB).convert("RGB")

faceA = crop_face_tensor(mtcnn, imgA, device=DEVICE)  # 1xC x H x W float [0,1]
faceB = crop_face_tensor(mtcnn, imgB, device=DEVICE)
if faceA is None or faceB is None:
    print("Face not detected in one of the images.")
    sys.exit(1)

with torch.no_grad():
    embB = extract_embedding(resnet, faceB.to(DEVICE))  # shape (1,D)
    embA_orig = extract_embedding(resnet, faceA.to(DEVICE))
    sim_before = float(F.cosine_similarity(embA_orig, embB, dim=1).item())
print("sim_before (direct crop):", sim_before)

mode = MODE.lower()

if mode == "fgsm_eot":
    adv_tensor, stats = fgsm_targeted_eot(resnet, faceA, embB, epsilon=EPSILON, samples=12, device=DEVICE)
    print("FGSM-EoT stats:", stats)
elif mode == "pgd_eot":
    adv_tensor, stats = pgd_targeted_eot(resnet, faceA, embB, eps=EPSILON, steps=PGD_STEPS, alpha=PGD_ALPHA, samples=8, device=DEVICE, momentum=MOMENTUM_MU)
    print("PGD-EoT stats:", stats)

# Evaluate on direct adv crop
with torch.no_grad():
    embA_adv = extract_embedding(resnet, adv_tensor.to(DEVICE))
    sim_after = float(F.cosine_similarity(embA_adv, embB, dim=1).item())
print("sim_after (direct adv crop):", sim_after)

# Paste adv into full-frame and run full pipeline (MTCNN->align->resnet) to measure real sim on sandbox
full_adv = paste_adv_to_fullframe(adv_tensor.cpu(), imgA, mtcnn)
if full_adv is not None:
    # run MTCNN detect+align on pasted image to get new aligned crop
    face_adv_from_frame = crop_face_tensor(mtcnn, full_adv, device=DEVICE)
    if face_adv_from_frame is not None:
        with torch.no_grad():
            emb_adv_frame = extract_embedding(resnet, face_adv_from_frame)
            sim_frame = float(F.cosine_similarity(emb_adv_frame, embB, dim=1).item())
        print("sim_after (pasted -> pipeline):", sim_frame)
    else:
        print("MTCNN failed to detect face on pasted adv full-frame.")
else:
    print("No bbox found to paste adv into full-frame.")

# save adv crop as before
cv2.imwrite(SAVE_PATH, tensor_to_bgr_image(adv_tensor))
print("Saved adversarial image:", SAVE_PATH)
