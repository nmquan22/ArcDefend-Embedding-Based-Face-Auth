# 🛡️ ArcDefend  
### A sandbox for embedding-based face authentication, attacks & defenses

ArcDefend is a modular, research-oriented project that demonstrates how to build, evaluate, attack, and defend an embedding-based face authentication system.  
It is designed for security labs, red-team/blue-team exercises, and educational purposes.

---

# 📌 1. Overview

ArcDefend implements the **full face recognition pipeline** used in most modern authentication systems:

```
Image → Face Detector → Face Alignment → Feature Extraction (Embedding) → Cosine Similarity → Threshold Decision
```

We provide API endpoints (FastAPI), a simple UI (Streamlit), a reproducible evaluation pipeline (ROC/EER), and extensions for attacks & defenses.

---

# 🚀 2. Features

✅ Face enrollment (one image or multiple images per user)  
✅ Face login using cosine similarity over 512-D embeddings  
✅ MTCNN detection + alignment  
✅ FaceNet (VGGFace2) baseline model for embeddings  
✅ SQLite database for storing user embeddings  
✅ Streamlit UI for Enroll/Login  
✅ Export embeddings for evaluation  
✅ Evaluation script (ROC, AUC, EER, threshold selection)  
✅ Extensible tasks for model comparison, attacks, defenses  

---

# 📂 3. Directory Structure

```
arc-defend/
│
├── app.py                 
├── streamlit_app.py       
├── export_embeddings.py   
├── eval_threshold.py      
├── compare_models.py      
├── arcdefend.db           
├── requirements.txt
└── README.md
```

---

# 🧬 4. System Architecture (Pipeline)

### Authentication Flow

1. Image Input — user uploads a photo.  
2. Face Detection & Alignment (MTCNN).  
3. Feature Extraction (FaceNet, 512-D embedding).  
4. Cosine Similarity Matching.  
5. Threshold Decision (Accept/Reject).

---

# 🛠️ 5. Installation

```
pip install -r requirements.txt
```

Run API:

```
uvicorn app:app --host 0.0.0.0 --port=8000
```

Run UI:

```
streamlit run streamlit_app.py
```

---

# 🧪 6. Evaluation (ROC / EER / Threshold)

Use:

```
python eval_threshold.py --db arcdefend.db --max_impostor 10000 --max_genuine 500 --target_fpr 0.001 --out roc.png
```

Outputs:  
- ROC curve  
- AUC  
- EER  
- Optimal threshold  

---

# 🧩 7. Project Tasks (for 4 members)

## Task 1 — Core Face Authentication System  
Owner: Person A  
- FastAPI (enroll/login)  
- Streamlit UI  
- Detection + alignment  
- Embeddings + cosine  
- SQLite storage  

## Task 2 — Evaluation  
Owner: Person B  
- Generate genuine/impostor pairs  
- ROC, AUC, EER  
- Threshold tuning  

## Task 3 — Model Comparison  
Owner: Person C  
- Compare FaceNet / ArcFace / CLIP  
- ROC for each model  
- Analysis  

## Task 4 — Attack & Defense  
Owner: Person D  
- Printed photo attack  
- Screen replay  
- Simple morphing  
- Liveness (blink/motion)  
- Logging, anomaly detection  

---

# 🔒 8. Security Notes

- Educational use only.  
- Do not test on real users without consent.  
- Do not attack production systems.  

---

# 🎯 9. Future Work

- JWT login tokens  
- Webcam support  
- MediaPipe liveness  
- Encrypted embeddings  
- Benchmarking models  

---

# 📜 License
MIT License
