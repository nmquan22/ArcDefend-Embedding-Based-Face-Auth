# ArcDefend
A sandbox framework for studying embedding-based face authentication, model evaluation, attack simulation, and defensive strategies. ArcDefend is designed for security education, red-team/blue-team exercises, and reproducible research.

---

## 1. Overview

ArcDefend implements the complete face recognition pipeline used in modern authentication systems:

Image → Face Detection → Alignment → Embedding Extraction → Cosine Similarity → Threshold Decision

The project provides:
- FastAPI backend for enrollment, authentication, and embedding export  
- Streamlit interface for interactive demonstration  
- A reproducible evaluation toolkit (ROC, AUC, EER, threshold calibration)  
- Extensions for model comparison, adversarial attacks, and simple defenses  

---

## 2. Features

- Face enrollment (single or multiple samples per identity)  
- Authentication using cosine similarity over 512‑D embeddings  
- MTCNN-based detection and alignment  
- FaceNet (VGGFace2) as the baseline embedding model  
- SQLite database for persistent storage  
- Streamlit user interface (Enroll/Login)  
- Embedding export for external experiments  
- Evaluation scripts (ROC, AUC, EER, threshold selection)  
- Modular design for integrating alternative models, attacks, and defenses  

---

## 3. Directory Structure

```
arcdefend/
│
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit interface
├── export_embeddings.py    # Export embeddings for evaluation
├── eval_threshold.py        # ROC/EER analysis
├── compare_models.py       # Model comparison pipeline
├── arcdefend.db            # SQLite database
├── requirements.txt
└── README.md
```

---

## 4. System Architecture

### Authentication Pipeline

1. Input image  
2. Face detection and landmark extraction (MTCNN)  
3. Geometric alignment  
4. Embedding extraction (FaceNet or alternative model)  
5. Cosine similarity comparison  
6. Threshold-based accept/reject decision  

---

## 5. Installation

Install dependencies:
```
pip install -r requirements.txt
```

Run the API:
```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Run the UI:
```
streamlit run streamlit_app.py
```

---

## 6. Evaluation (ROC / AUC / EER)

Using notebook model_comparision.ipynb
---

## 7. Project Task Division (4 Members)

### Task 1 — Core Authentication System  
- FastAPI endpoints  
- Streamlit UI  
- Detection and alignment pipeline  
- Embedding extraction  
- SQLite storage  

### Task 2 — Evaluation  
- Genuine/impostor pair generation  
- ROC/AUC computation  
- Threshold selection  

### Task 3 — Model Comparison  
- Benchmark FaceNet, ArcFace, CLIP  
- ROC plotting  
- Performance analysis  

### Task 4 — Attack and Defense  
- Printed-photo attack  
- Screen replay  
- Basic morphing  
- Liveness cues (blink/motion)  
- Score anomaly detection  

---

## 8. Security Considerations

- Intended strictly for educational and research use  
- Do not test on real users without explicit consent  
- Do not apply adversarial methods to production systems  

---

## 9. Future Work

- JWT-based authenticated sessions  
- Real-time webcam support  
- Advanced liveness detection  
- Encrypted or cancelable embeddings  
- Benchmarking additional embedding models  

---

## License
MIT License
