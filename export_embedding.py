# export_embeddings.py
import os, json, sqlite3, time
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch, cv2, numpy as np
from tqdm import tqdm

DB_PATH = "arcdefend.db"
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

def save_user_embedding(user_id, emb, meta):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO users (user_id, embedding, metadata, created_at) VALUES (?,?,?,?)",
              (user_id, json.dumps(emb.tolist()), json.dumps(meta), time.time()))
    conn.commit()
    conn.close()

def img_to_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_folder(data_dir="data"):
    ensure_db()
    for user in os.listdir(data_dir):
        up = os.path.join(data_dir, user)
        if not os.path.isdir(up):
            continue
        for fn in tqdm(os.listdir(up), desc=user):
            if not fn.lower().endswith((".jpg",".png",".jpeg")):
                continue
            p = os.path.join(up, fn)
            img = img_to_rgb(p)
            face = mtcnn(img)
            if face is None:
                print("No face:", p)
                continue
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy().reshape(-1)
                emb = emb / (np.linalg.norm(emb)+1e-10)
            save_user_embedding(user, emb, {"source": p})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="data/<user>/*.jpg")
    args = parser.parse_args()
    process_folder(args.data)
