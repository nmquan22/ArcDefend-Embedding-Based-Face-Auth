# eval_threshold.py
import sqlite3, json, itertools, random, numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse

EMBEDDING_SIZE = 512

def load_embeddings(db="arcdefend.db"):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("SELECT user_id, embedding FROM users")
    rows = c.fetchall()
    conn.close()
    d={}
    for u,e in rows:
        emb = np.array(json.loads(e), dtype=float)
        d.setdefault(u,[]).append(emb)
    return d

def build_pairs(d, max_impostor=20000, max_genuine_per_user=1000, seed=42):
    random.seed(seed)
    users = list(d.keys())
    genuine=[]
    impostor=[]
    for u in users:
        embs=d[u]
        if len(embs)<2: continue
        pairs=list(itertools.combinations(range(len(embs)),2))
        random.shuffle(pairs)
        for i,j in pairs[:max_genuine_per_user]:
            genuine.append((embs[i], embs[j]))
    attempts=0
    while len(impostor) < max_impostor and attempts < max_impostor*5:
        u1,u2=random.sample(users,2)
        impostor.append((random.choice(d[u1]), random.choice(d[u2])))
        attempts+=1
    return genuine, impostor

def cos(a,b):
    a=a/ (np.linalg.norm(a)+1e-10)
    b=b/ (np.linalg.norm(b)+1e-10)
    return float(np.dot(a,b))

def compute_scores(pairs):
    return [cos(a,b) for a,b in pairs]

def find_eer(fpr,tpr,thresholds):
    fnr = 1 - tpr
    idx = (np.abs(fpr - fnr)).argmin()
    return max(fpr[idx], fnr[idx]), thresholds[idx]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--db", default="arcdefend.db")
    parser.add_argument("--max_impostor", type=int, default=10000)
    parser.add_argument("--max_genuine", type=int, default=500)
    parser.add_argument("--target_fpr", type=float, default=None)
    parser.add_argument("--out", default="roc.png")
    args=parser.parse_args()

    d = load_embeddings(args.db)
    print("Users:", len(d))
    genuine, impostor = build_pairs(d, args.max_impostor, args.max_genuine)
    y_true = np.array([1]*len(genuine) + [0]*len(impostor))
    scores = np.array(compute_scores(genuine) + compute_scores(impostor))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    auc = metrics.auc(fpr,tpr)
    eer, eer_th = find_eer(fpr,tpr,thresholds)
    print("AUC:", auc, "EER:", eer, "EER_threshold:", eer_th)

    if args.target_fpr is not None:
        idxs = np.where(fpr <= args.target_fpr)[0]
        if len(idxs)==0:
            chosen_idx = fpr.argmin()
        else:
            chosen_idx = idxs[np.argmax(tpr[idxs])]
        chosen_th = thresholds[chosen_idx]
    else:
        chosen_th = eer_th

    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    plt.grid(True)
    plt.savefig(args.out)
    print("Saved roc to", args.out)
    print("Chosen threshold:", chosen_th)

if __name__ == "__main__":
    main()
