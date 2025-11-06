import io, json, sqlite3, time
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch, numpy as np, cv2
import uvicorn

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
    with torch.no_grad():
        emb = resnet(face_tensor)  # (1,512)
    emb = emb.detach().cpu().numpy().reshape(-1)
    emb = emb / (np.linalg.norm(emb)+1e-10)
    return emb.astype(float)

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
