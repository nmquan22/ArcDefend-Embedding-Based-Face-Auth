import os
import io
import requests
import streamlit as st
from PIL import Image

# Page config: professional, wide layout
st.set_page_config(page_title="ArcDefend UI", page_icon=None, layout="wide")

# --- Header ---
st.title("ArcDefend — Face Authentication Sandbox")
st.caption("Embedding-based face authentication • Enroll → Login → Evaluate")

# --- Sidebar config ---
st.sidebar.header("Configuration")
default_api = os.getenv("ARCDEFEND_API", "http://127.0.0.1:8000")
api_base = st.sidebar.text_input("API Base URL", value=default_api, help="Example: http://127.0.0.1:8000")
st.sidebar.markdown("---")
st.sidebar.write("Quick start:")
st.sidebar.write("1) Start server: `uvicorn app:app --host 0.0.0.0 --port 8000`")
st.sidebar.write("2) Use the Enroll / Login tabs")

# --- Helper functions ---
def post_file(endpoint: str, files: dict, data: dict = None, params: dict = None):
    url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.post(url, files=files, data=data, params=params, timeout=30)
        content_type = resp.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            return resp.status_code, resp.json()
        return resp.status_code, {"text": resp.text}
    except requests.RequestException as e:
        return 0, {"error": str(e)}

def get_json(endpoint: str, params: dict = None):
    url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        content_type = resp.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            return resp.status_code, resp.json()
        return resp.status_code, {"text": resp.text}
    except requests.RequestException as e:
        return 0, {"error": str(e)}

# --- Layout tabs ---
tab_enroll, tab_login, tab_users = st.tabs(["Enroll", "Login", "Users"])

# --- Enroll Tab ---
with tab_enroll:
    st.subheader("Enroll user")
    user_id = st.text_input("User ID", placeholder="e.g. alice")
    img = st.file_uploader("Upload face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])
    with col1:
        preview = st.checkbox("Preview image", value=True)
    with col2:
        submit = st.button("Enroll", disabled=not (user_id and img))

    if img and preview:
        try:
            # Reset pointer and show
            img.seek(0)
            st.image(Image.open(img), caption="Uploaded image", use_column_width=True)
        except Exception:
            st.warning("Unable to render preview image (but upload may still work).")

    if submit and user_id and img:
        img.seek(0)
        files = {"file": (img.name, img.read(), img.type or "image/jpeg")}
        data = {"user_id": user_id}
        code, res = post_file("/enroll", files=files, data=data)
        if code == 200:
            st.success(f"Enroll succeeded: {res}")
        elif code == 400:
            st.error("No face detected — please choose a clear face image.")
        elif code == 0:
            st.error(f"Network error: {res.get('error')}")
        else:
            st.error(f"Error ({code}): {res}")

# --- Login Tab ---
with tab_login:
    st.subheader("Login with face")

    # === Left: image & preview ===
    login_img = st.file_uploader("Upload image for login", type=["jpg", "jpeg", "png"], key="login_file")
    login_preview = st.checkbox("Preview login image", True, key="login_prev")
    if login_img and login_preview:
        try:
            login_img.seek(0)
            st.image(Image.open(login_img), caption="Login image", use_column_width=True)
        except Exception:
            st.warning("Unable to render preview image (but upload may still work).")

    # === Right: thresholds & defense options ===
    st.markdown("---")
    colA, colB = st.columns([1,1])

    with colA:
        th = st.slider(
            "Cosine threshold",
            min_value=0.50, max_value=0.99, value=0.80, step=0.01,
            help="If cosine ≥ threshold → accept (sau khi defense)."
        )
        enable_defense = st.checkbox("Enable defense (call /login_defense)", value=True)

    with colB:
        if enable_defense:
            mode = st.selectbox("Defense mode", ["gaussian", "median", "jpeg", "chain", "none"], index=0)
            K = st.slider("EoT samples K", min_value=1, max_value=20, value=5, step=1,
                          help="K lần làm sạch + đánh giá, tính trung bình/median/max và kiểm tra variance.")
            aggregate = st.selectbox("Aggregate scores", ["mean","median","max"], index=0)
            variance_gate = st.number_input("Variance gate", min_value=0.0, max_value=0.1, value=0.0025, step=0.0005)

            # params theo mode
            if mode in ["gaussian", "chain"]:
                gaussian_ksize = st.selectbox("Gaussian ksize (odd)", [3,5,7,9,11], index=0)
                gaussian_sigma_min = st.number_input("Gaussian σ min", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
                gaussian_sigma_max = st.number_input("Gaussian σ max", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
            else:
                gaussian_ksize, gaussian_sigma_min, gaussian_sigma_max = 3, 0.0, 1.0

            if mode in ["median"]:
                median_ksize = st.selectbox("Median ksize (odd)", [3,5,7,9,11], index=0)
            else:
                median_ksize = 3

            if mode in ["jpeg", "chain"]:
                jpeg_q_min = st.slider("JPEG quality min", min_value=10, max_value=100, value=50, step=1)
                jpeg_q_max = st.slider("JPEG quality max", min_value=10, max_value=100, value=85, step=1)
            else:
                jpeg_q_min, jpeg_q_max = 50, 85
        else:
            mode = "none"; K = 1; aggregate = "mean"; variance_gate = 0.0025
            gaussian_ksize = 3; gaussian_sigma_min = 0.0; gaussian_sigma_max = 1.0
            median_ksize = 3; jpeg_q_min = 50; jpeg_q_max = 85

    btn_login = st.button("Login", disabled=not login_img)

    if btn_login and login_img:
        login_img.seek(0)
        files = {"file": (login_img.name, login_img.read(), login_img.type or "image/jpeg")}

        if enable_defense:
            params = {
                "threshold": th,
                "mode": mode,
                "K": K,
                "gaussian_ksize": gaussian_ksize,
                "gaussian_sigma_min": gaussian_sigma_min,
                "gaussian_sigma_max": gaussian_sigma_max,
                "median_ksize": median_ksize,
                "jpeg_q_min": jpeg_q_min,
                "jpeg_q_max": jpeg_q_max,
                "variance_gate": variance_gate,
                "aggregate": aggregate,
            }
            code, res = post_file("/login_defense", files=files, params=params)
        else:
            # dùng /login cũ
            params = {"threshold": th}
            code, res = post_file("/login", files=files, params=params)

        if code == 200 and isinstance(res, dict):
            st.markdown("---")
            if enable_defense:
                # hiển thị các số liệu defense
                score_mean = res.get("score_mean", 0.0)
                score_var  = res.get("score_var", 0.0)
                st.metric("Cosine (aggregate)", f"{score_mean:.4f}")
                st.metric("Variance", f"{score_var:.6f}")
                st.json({k: v for k, v in res.items()
                         if k in ["accepted","best_user","score_mean","score_median","score_max","score_min","score_var","threshold","mode","K","variance_gate","aggregate"]})
                accepted = bool(res.get("accepted", False))
                best_user = res.get("best_user")
                prog_val = min(max((score_mean - 0.5) / 0.49, 0.0), 1.0)
                st.progress(prog_val)
                if accepted:
                    st.success(f"[DEFENSE] Accepted — matched user: {best_user}")
                else:
                    st.error("[DEFENSE] Rejected")
            else:
                # logic cũ
                accepted = res.get("accepted", False)
                best_user = res.get("best_user")
                score = float(res.get("score", 0.0))
                threshold = float(res.get("threshold", th))
                st.metric("Cosine score", f"{score:.4f}")
                st.progress(min(max((score - 0.5) / 0.49, 0.0), 1.0))
                if accepted:
                    st.success(f"Accepted — matched user: {best_user} (score={score:.4f} ≥ threshold={threshold:.2f})")
                else:
                    st.error(f"Rejected — best match: {best_user} (score={score:.4f} < threshold={threshold:.2f})")

        elif code == 0:
            st.error(f"Network error: {res.get('error')}")
        else:
            st.error(f"Error ({code}): {res}")

# --- Users Tab ---
with tab_users:
    st.subheader("Enrolled users")
    refresh_col1, refresh_col2 = st.columns([1, 3])
    with refresh_col1:
        btn_refresh = st.button("Refresh list")
    with refresh_col2:
        st.write("Use the Export Embeddings endpoint to download embeddings for ROC/EER evaluation.")

    # Fetch users on load and when refresh clicked
    if 'users_cache' not in st.session_state or btn_refresh:
        code, res = get_json("/users")
        st.session_state['users_cache'] = (code, res)

    code, res = st.session_state['users_cache']
    if code == 200 and isinstance(res, dict):
        # show a compact table-like json
        try:
            # Try to pretty print users list if it's a list
            if isinstance(res.get("users"), list):
                st.table(res.get("users"))
            else:
                st.json(res)
        except Exception:
            st.json(res)
    elif code == 0:
        st.error(f"Network error: {res.get('error')}")
    else:
        st.error(f"Error ({code}): {res}")

    st.markdown("---")
    st.subheader("Export embeddings")

    if st.button("Download embeddings"):
        code, res = get_json("/export_embeddings")

        if code == 200 and isinstance(res, dict):
            # Convert JSON to bytes for downloading
            import json
            raw_bytes = json.dumps(res, indent=2).encode("utf-8")

            st.download_button(
                label="Click to download JSON",
                data=raw_bytes,
                file_name="embeddings_export.json",
                mime="application/json"
            )

            # Also display a short preview (first 2 users)
            try:
                users = list(res.items())[:2]  # preview only first two items
                st.json(dict(users))
            except Exception:
                st.json(res)

        elif code == 0:
            st.error(f"Network error: {res.get('error')}")
        else:
            st.error(f"Error ({code}): {res}")

