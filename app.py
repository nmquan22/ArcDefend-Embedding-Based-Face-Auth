import io, json, sqlite3, time
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch, numpy as np, cv2
import uvicorn
from fastapi import Query


DB_PATH = "arcdefend.db"
THRESHOLD = 0.80
EMBEDDING_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT,
        emb_id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding TEXT,
        metadata TEXT,
        created_at REAL
    )
    """)
    conn.commit()
    conn.close()

def image_bytes_to_rgb(data: bytes):
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_face_tensor_from_bytes(data: bytes):
    img_rgb = image_bytes_to_rgb(data)
    face = mtcnn(img_rgb)
    if face is None:
        return None
    if face.ndimension() == 3:
        return face.unsqueeze(0).to(device)
    return face.to(device)

def extract_embedding_from_tensor(face_tensor):
    x = face_tensor.to(device).float()      
    with torch.no_grad():
        emb = resnet(x)[0].cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb



def save_user_embedding(user_id: str, emb, metadata: dict = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO users (user_id, embedding, metadata, created_at) VALUES (?,?,?,?)",
              (user_id, json.dumps(emb.tolist()), json.dumps(metadata or {}), time.time()))
    conn.commit()
    conn.close()

def load_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, embedding FROM users")
    rows = c.fetchall()
    conn.close()
    data = {}
    for user_id, emb_json in rows:
        emb = np.array(json.loads(emb_json), dtype=float)
        data.setdefault(user_id, []).append(emb)
    return data

def cosine_similarity(a,b):
    a = a / (np.linalg.norm(a)+1e-10)
    b = b / (np.linalg.norm(b)+1e-10)
    return float(np.dot(a,b))

# --- Defense transforms & helpers ---------------------------------------
import math
from io import BytesIO

def _rgb_to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def _bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_gaussian_blur(img_rgb: np.ndarray, ksize: int = 3, sigma: float = 0.0):
    k = max(3, int(ksize) | 1)  # k phải lẻ
    bgr = _rgb_to_bgr(img_rgb)
    out = cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma)
    return _bgr_to_rgb(out)

def apply_median_filter(img_rgb: np.ndarray, ksize: int = 3):
    k = max(3, int(ksize) | 1)
    bgr = _rgb_to_bgr(img_rgb)
    out = cv2.medianBlur(bgr, k)
    return _bgr_to_rgb(out)

def apply_jpeg_compression(img_rgb: np.ndarray, quality: int = 70):
    quality = int(np.clip(quality, 10, 100))
    bgr = _rgb_to_bgr(img_rgb)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img_rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return _bgr_to_rgb(dec)

def sanitize_image_rgb(img_rgb: np.ndarray,
                       mode: str = "gaussian",
                       gaussian_ksize: int = 3,
                       gaussian_sigma: float = 0.0,
                       median_ksize: int = 3,
                       jpeg_quality: int = 70):
    mode = (mode or "gaussian").lower()
    if mode == "gaussian":
        return apply_gaussian_blur(img_rgb, gaussian_ksize, gaussian_sigma)
    elif mode == "median":
        return apply_median_filter(img_rgb, median_ksize)
    elif mode == "jpeg":
        return apply_jpeg_compression(img_rgb, jpeg_quality)
    elif mode == "chain":   # ví dụ: blur -> jpeg
        x = apply_gaussian_blur(img_rgb, gaussian_ksize, gaussian_sigma)
        x = apply_jpeg_compression(x, jpeg_quality)
        return x
    else:
        return img_rgb

def image_bytes_to_rgb_numpy(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_face_tensor_from_rgb(img_rgb: np.ndarray):
    face = mtcnn(img_rgb)
    if face is None:
        return None
    if face.ndimension() == 3:
        return face.unsqueeze(0).to(device)
    return face.to(device)

def cosine_scores_against_db(emb: np.ndarray, db: dict):
    best_user, best_score = None, -1.0
    for uid, embs in db.items():
        for u_emb in embs:
            s = cosine_similarity(emb, u_emb)
            if s > best_score:
                best_score = s
                best_user = uid
    return best_user, best_score

# --- Expectation-over-Transformation (EoT) + variance gate --------------
def eot_defense_login(
    data_bytes: bytes,
    threshold: float,
    db: dict,
    mode: str,
    K: int = 5,
    gaussian_ksize: int = 3,
    gaussian_sigma_min: float = 0.0,
    gaussian_sigma_max: float = 1.0,
    median_ksize: int = 3,
    jpeg_q_min: int = 50,
    jpeg_q_max: int = 85,
    variance_gate: float = 0.0025,   # nếu var quá cao -> reject
    aggregate: str = "mean",         # "mean" hoặc "median" hoặc "max"
):
    # 1) decode
    img_rgb = image_bytes_to_rgb_numpy(data_bytes)

    scores = []
    best_overall_user = None
    best_overall_score = -1.0

    for _ in range(max(1, int(K))):
        if mode == "gaussian":
            sig = float(np.random.uniform(gaussian_sigma_min, gaussian_sigma_max))
            img_san = sanitize_image_rgb(img_rgb, "gaussian", gaussian_ksize, sig)
        elif mode == "median":
            img_san = sanitize_image_rgb(img_rgb, "median", median_ksize=median_ksize)
        elif mode == "jpeg":
            q = int(np.random.uniform(jpeg_q_min, jpeg_q_max))
            img_san = sanitize_image_rgb(img_rgb, "jpeg", jpeg_quality=q)
        elif mode == "chain":
            sig = float(np.random.uniform(gaussian_sigma_min, gaussian_sigma_max))
            q = int(np.random.uniform(jpeg_q_min, jpeg_q_max))
            img_tmp = sanitize_image_rgb(img_rgb, "gaussian", gaussian_ksize, sig)
            img_san = sanitize_image_rgb(img_tmp, "jpeg", jpeg_quality=q)
        else:
            img_san = img_rgb  # no-op

        face_tensor = get_face_tensor_from_rgb(img_san)
        if face_tensor is None:
            continue
        emb = extract_embedding_from_tensor(face_tensor)

        uid, sc = cosine_scores_against_db(emb, db)
        scores.append(sc)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_user = uid

    if not scores:
        return {"accepted": False, "reason": "no_face_or_no_samples"}

    scores = np.array(scores, dtype=np.float32)
    var = float(np.var(scores))
    if aggregate == "median":
        agg = float(np.median(scores))
    elif aggregate == "max":
        agg = float(np.max(scores))
    else:
        agg = float(np.mean(scores))

    accepted = (var <= variance_gate) and (agg >= threshold)

    return {
        "accepted": bool(accepted),
        "best_user": best_overall_user,
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "score_min": float(np.min(scores)),
        "score_var": var,
        "score_list": [float(x) for x in scores.tolist()],
        "threshold": float(threshold),
        "mode": mode,
        "K": int(K),
        "variance_gate": float(variance_gate),
        "aggregate": aggregate
    }



app = FastAPI(title="ArcDefend Demo API")
ensure_db()

@app.post("/enroll")
async def enroll(user_id: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    face_tensor = get_face_tensor_from_bytes(data)
    if face_tensor is None:
        raise HTTPException(400, "No face detected")
    emb = extract_embedding_from_tensor(face_tensor)
    save_user_embedding(user_id, emb, metadata={"filename": file.filename})
    return {"status":"ok", "user_id": user_id}

@app.post("/login")
async def login(file: UploadFile = File(...), threshold: float = THRESHOLD):
    data = await file.read()
    face_tensor = get_face_tensor_from_bytes(data)
    if face_tensor is None:
        raise HTTPException(400, "No face detected")
    emb = extract_embedding_from_tensor(face_tensor)
    db = load_all_embeddings()
    if not db:
        raise HTTPException(404, "No enrolled users")
    best_user, best_score = None, -1.0
    for uid, embs in db.items():
        # compare to each embedding for user and take max
        for u_emb in embs:
            s = cosine_similarity(emb, u_emb)
            if s > best_score:
                best_score = s; best_user = uid
    accepted = best_score >= threshold
    return {"accepted": bool(accepted), "best_user": best_user, "score": best_score, "threshold": threshold}

@app.get("/users")
def list_users():
    db = load_all_embeddings()
    return {"n_users": len(db), "users": {u: len(db[u]) for u in db}}

# helper to dump embeddings to JSON for eval
@app.get("/export_embeddings")
def export_embeddings():
    db = load_all_embeddings()
    out = {}
    for u, embs in db.items():
        out[u] = [e.tolist() for e in embs]
    return out


from fastapi import Query

@app.post("/login_defense")
async def login_defense(
    file: UploadFile = File(...),
    threshold: float = Query(THRESHOLD),
    mode: str = Query("gaussian"),   # ["gaussian","median","jpeg","chain","none"]
    K: int = Query(5, ge=1, le=20),
    # gaussian params
    gaussian_ksize: int = Query(3, ge=3, le=11),
    gaussian_sigma_min: float = Query(0.0, ge=0.0, le=3.0),
    gaussian_sigma_max: float = Query(1.0, ge=0.0, le=3.0),
    # median params
    median_ksize: int = Query(3, ge=3, le=11),
    # jpeg params
    jpeg_q_min: int = Query(50, ge=10, le=100),
    jpeg_q_max: int = Query(85, ge=10, le=100),
    # variance gate & aggregator
    variance_gate: float = Query(0.0025, ge=0.0, le=0.1),
    aggregate: str = Query("mean"),  # ["mean","median","max"]
):
    data = await file.read()
    db = load_all_embeddings()
    if not db:
        raise HTTPException(404, "No enrolled users")

    # guard tham số
    if gaussian_sigma_max < gaussian_sigma_min:
        gaussian_sigma_max = gaussian_sigma_min
    if jpeg_q_max < jpeg_q_min:
        jpeg_q_max = jpeg_q_min

    res = eot_defense_login(
        data_bytes=data,
        threshold=threshold,
        db=db,
        mode=mode,
        K=K,
        gaussian_ksize=gaussian_ksize,
        gaussian_sigma_min=gaussian_sigma_min,
        gaussian_sigma_max=gaussian_sigma_max,
        median_ksize=median_ksize,
        jpeg_q_min=jpeg_q_min,
        jpeg_q_max=jpeg_q_max,
        variance_gate=variance_gate,
        aggregate=aggregate,
    )
    return res



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
