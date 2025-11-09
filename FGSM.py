import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import sys
from typing import Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 160
EPSILON = 16.0/255.0
PGD_STEPS = 60
PGD_ALPHA = 1.0/255.0
PGD_RESTARTS = 3
USE_MOMENTUM = True
MOMENTUM_MU = 1.0
ADAM_ITERS = 200
ADAM_LR = 0.05
SAVE_PREFIX = "resultAdv/adv_out_norm"
SAVE_PATH = f"{SAVE_PREFIX}_best.png"
MODE = "adam"   # "fgsm", "pgd", or "adam"

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


def fgsm_targeted(resnet, adv_tensor, embB, epsilon=EPSILON, device=DEVICE) -> Tuple[torch.Tensor, dict]:
    resnet.eval()
    adv = adv_tensor.clone().detach().to(device)
    adv.requires_grad_(True)

    x_in = normalize_for_facenet(adv)
    embA_adv = resnet(x_in)
    loss = (1.0 - cosine(embA_adv, embB)).mean()  # minimize => increase cosine

    adv.grad = None
    resnet.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = adv.grad.data.sign()
    # gradient descent step to reduce loss -> increase cosine
    adv_pert = adv - epsilon * grad_sign
    adv_pert = torch.clamp(adv_pert, 0.0, 1.0).detach()

    with torch.no_grad():
        embA_adv2 = extract_embedding(resnet, adv_pert)
        sim_after = float(cosine(embA_adv2, embB).item())
        loss_val = float((1.0 - cosine(embA_adv2, embB)).mean().item())

    stats = {"sim_after": sim_after, "loss": loss_val, "epsilon": epsilon}
    return adv_pert.detach().cpu(), stats

def pgd_with_restarts(resnet, orig_adv, embB, eps=EPSILON, steps=PGD_STEPS, alpha=PGD_ALPHA,
                      restarts=PGD_RESTARTS, device=DEVICE, use_momentum=USE_MOMENTUM, mu=MOMENTUM_MU,
                      save_prefix: str = SAVE_PREFIX) -> Tuple[torch.Tensor, dict]:
    resnet.eval()
    best_sim = -1.0
    best_adv = None
    best_trace = None
    orig = orig_adv.clone().detach().to(device)

    for r in range(restarts):
        # random start inside L_inf ball
        adv = orig + torch.empty_like(orig).uniform_(-eps, eps)
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        if use_momentum:
            g_m = torch.zeros_like(adv)

        sims = []
        for it in range(steps):
            adv.requires_grad_(True)
            x_in = normalize_for_facenet(adv)
            embA_adv = resnet(x_in)
            loss = (1.0 - cosine(embA_adv, embB)).mean()

            # zero grads
            if adv.grad is not None:
                adv.grad.zero_()
            resnet.zero_grad(set_to_none=True)
            loss.backward()
            grad = adv.grad.data

            if use_momentum:
                # normalize grad by its mean abs to stabilize
                g_m = mu * g_m + grad / (grad.abs().mean() + 1e-12)
                step = alpha * g_m.sign()
            else:
                step = alpha * grad.sign()

            # gradient descent (reduce loss)
            adv = (adv.detach() - step).clamp(0.0, 1.0).detach()
            adv = torch.max(torch.min(adv, orig + eps), orig - eps)
            adv = torch.clamp(adv, 0.0, 1.0).detach()

            with torch.no_grad():
                emb_tmp = extract_embedding(resnet, adv)
                sim_tmp = float(cosine(emb_tmp, embB).item())
            sims.append(sim_tmp)

            # if save_prefix and r == 0 and (it % max(1, steps//5) == 0 or it == steps-1):
            #     cv2.imwrite(f"{save_prefix}_r{r}_it{it}.png", tensor_to_bgr_image(adv.detach().cpu()))

        final_sim = sims[-1]
        if final_sim > best_sim:
            best_sim = final_sim
            best_adv = adv.clone().detach()
            best_trace = sims

    # save best adv + heatmap
    if save_prefix and best_adv is not None:
        cv2.imwrite(f"{save_prefix}_best.png", tensor_to_bgr_image(best_adv.cpu()))
        try:
            save_diff_heatmap(orig.cpu(), best_adv.cpu(), f"{save_prefix}_heatmap.png")
        except Exception:
            pass

    stats = {
        "best_sim": best_sim,
        "sims_trace": best_trace,
        "epsilon": eps,
        "steps": steps,
        "alpha": alpha,
        "restarts": restarts,
        "use_momentum": use_momentum
    }
    return best_adv.cpu(), stats

def optimize_pixels_adam(resnet, orig_adv, embB, eps=EPSILON, iters=ADAM_ITERS, lr=ADAM_LR,
                         save_prefix: str = SAVE_PREFIX) -> Tuple[torch.Tensor, dict]:
    
    resnet.eval()
    orig = orig_adv.clone().detach().to(DEVICE)
    adv = orig.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([adv], lr=lr)
    best_sim = -1.0
    best_adv = None
    sims = []

    for i in range(iters):
        opt.zero_grad()
        x_in = normalize_for_facenet(adv)
        embA = resnet(x_in)
        loss = (1.0 - cosine(embA, embB)).mean()
        loss.backward()
        opt.step()
        # project to L_inf ball and clamp
        adv.data = torch.max(torch.min(adv.data, orig + eps), orig - eps)
        adv.data.clamp_(0.0, 1.0)

        with torch.no_grad():
            sim = float(cosine(extract_embedding(resnet, adv), embB).item())
        sims.append(sim)
        if sim > best_sim:
            best_sim = sim
            best_adv = adv.clone().detach()

        # if save_prefix and (i % max(1, iters//5) == 0 or i == iters-1):
        #     cv2.imwrite(f"{save_prefix}_adam_it{i}.png", tensor_to_bgr_image(best_adv.cpu()))

    if save_prefix and best_adv is not None:
        cv2.imwrite(f"{save_prefix}_adam_best.png", tensor_to_bgr_image(best_adv.cpu()))
        save_diff_heatmap(orig.cpu(), best_adv.cpu(), f"{save_prefix}_adam_heatmap.png")

    return best_adv.cpu(), {"best_sim": best_sim, "sims_trace": sims, "eps": eps, "iters": iters, "lr": lr}


if __name__ == "__main__":
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=0, device=DEVICE)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    pathA = "D:\\ArcDefend\\Image\\man1.jpg"
    pathB = "D:\\ArcDefend\\Image\\man2.jpg"

    if not os.path.exists(pathA) or not os.path.exists(pathB):
        print("Please put imageA and imageB in the paths or change paths in the script.")
        sys.exit(1)

    imgA = Image.open(pathA).convert("RGB")
    imgB = Image.open(pathB).convert("RGB")

    # crop faces
    faceA = crop_face_tensor(mtcnn, imgA, device=DEVICE)
    faceB = crop_face_tensor(mtcnn, imgB, device=DEVICE)
    if faceA is None or faceB is None:
        print("Face not detected in one of the images.")
        sys.exit(1)

    # compute embeddings for target
    with torch.no_grad():
        embB = extract_embedding(resnet, faceB)  # (1, D)
        embA_orig = extract_embedding(resnet, faceA)
        sim_before = float(cosine(embA_orig, embB).item())

    print("sim_before:", sim_before)

    # Choose mode
    mode = MODE.lower()
    if mode == "fgsm":
        adv_tensor, stats = fgsm_targeted(resnet, faceA, embB, epsilon=EPSILON, device=DEVICE)
        print("FGSM stats:", stats)
    elif mode == "pgd":
        adv_tensor, stats = pgd_with_restarts(resnet, faceA, embB,
                                              eps=EPSILON, steps=PGD_STEPS, alpha=PGD_ALPHA,
                                              restarts=PGD_RESTARTS, device=DEVICE,
                                              use_momentum=USE_MOMENTUM, mu=MOMENTUM_MU,
                                              save_prefix=SAVE_PREFIX)
        print("PGD stats:", stats)
    elif mode == "adam":
        adv_tensor, stats = optimize_pixels_adam(resnet, faceA, embB,
                                                eps=EPSILON, iters=ADAM_ITERS, lr=ADAM_LR,
                                                save_prefix=SAVE_PREFIX)
        print("ADAM stats:", stats)
    else:
        print("Unknown mode. Choose 'fgsm', 'pgd', or 'adam'.")
        sys.exit(1)

    with torch.no_grad():
        embA_adv = extract_embedding(resnet, adv_tensor.to(DEVICE))
        sim_after = float(cosine(embA_adv, embB).item())
        print("sim_after: ", sim_after)
        print(f"epsilon (L_inf): {EPSILON} mode: {mode}")

    cv2.imwrite(SAVE_PATH, tensor_to_bgr_image(adv_tensor))
    print("Saved adversarial image:", SAVE_PATH)
   