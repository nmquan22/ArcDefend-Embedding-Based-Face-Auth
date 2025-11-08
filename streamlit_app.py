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
    login_img = st.file_uploader("Upload image for login", type=["jpg", "jpeg", "png"], key="login_file")
    th = st.slider(
        "Cosine threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01,
        help="Set threshold according to ROC/EER evaluation. If cosine ≥ threshold → accept."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        login_preview = st.checkbox("Preview login image", True, key="login_prev")
    with c2:
        btn_login = st.button("Login", disabled=not login_img)

    if login_img and login_preview:
        try:
            login_img.seek(0)
            st.image(Image.open(login_img), caption="Login image", use_column_width=True)
        except Exception:
            st.warning("Unable to render preview image (but upload may still work).")

    if btn_login and login_img:
        login_img.seek(0)
        files = {"file": (login_img.name, login_img.read(), login_img.type or "image/jpeg")}
        params = {"threshold": th}
        code, res = post_file("/login", files=files, params=params)
        if code == 200 and isinstance(res, dict):
            accepted = res.get("accepted", False)
            best_user = res.get("best_user")
            score = float(res.get("score", 0.0))
            threshold = float(res.get("threshold", th))

            st.markdown("---")
            st.metric("Cosine score", f"{score:.4f}")
            # progress: normalized between 0.5 and threshold/0.99 range
            prog_val = min(max((score - 0.5) / 0.49, 0.0), 1.0)
            st.progress(prog_val)

            if accepted:
                st.success(f"Accepted — matched user: {best_user} (score={score:.4f} ≥ threshold={threshold:.2f})")
            else:
                st.error(f"Rejected — best match: {best_user} (score={score:.4f} < threshold={threshold:.2f})")
        elif code == 400:
            st.error("Bad request — please check the uploaded image.")
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

