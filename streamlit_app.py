import os
import io
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="ArcDefend UI", page_icon="🛡️", layout="centered")

st.title("🛡️ ArcDefend — Face Authentication Sandbox")
st.caption("Embedding-based face auth • Enroll → Login → Evaluate")

# --- Sidebar config ---
st.sidebar.header("⚙️ Configuration")
default_api = os.getenv("ARCDEFEND_API", "http://127.0.0.1:8000")
api_base = st.sidebar.text_input("API Base URL", value=default_api, help="Ví dụ: http://127.0.0.1:8000")
st.sidebar.markdown("---")
st.sidebar.write("Hướng dẫn nhanh:")
st.sidebar.write("1) Mở server: `uvicorn app:app --host 0.0.0.0 --port 8000`")
st.sidebar.write("2) Dùng tab Enroll / Login bên dưới")


def post_file(endpoint: str, files: dict, data: dict = None, params: dict = None):
    url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.post(url, files=files, data=data, params=params, timeout=30)
        if resp.headers.get("content-type","").startswith("application/json"):
            return resp.status_code, resp.json()
        return resp.status_code, {"text": resp.text}
    except requests.RequestException as e:
        return 0, {"error": str(e)}

def get_json(endpoint: str, params: dict = None):
    url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.headers.get("content-type","").startswith("application/json"):
            return resp.status_code, resp.json()
        return resp.status_code, {"text": resp.text}
    except requests.RequestException as e:
        return 0, {"error": str(e)}

tab_enroll, tab_login, tab_users = st.tabs(["📝 Enroll", "🔐 Login", "👥 Users"])

# --- Enroll Tab ---
with tab_enroll:
    st.subheader("📝 Enroll người dùng")
    user_id = st.text_input("User ID", placeholder="ví dụ: alice")
    img = st.file_uploader("Upload ảnh khuôn mặt (JPG/PNG)", type=["jpg","jpeg","png"])

    col1, col2 = st.columns(2)
    with col1:
        preview = st.checkbox("Xem trước ảnh", True)
    with col2:
        submit = st.button("Enroll", disabled=not (user_id and img))

    if img and preview:
        try:
            st.image(Image.open(img), caption="Ảnh upload", use_column_width=True)
        except Exception:
            st.warning("Không hiển thị được ảnh (nhưng vẫn có thể enroll).")

    if submit and user_id and img:
        img.seek(0)
        files = {"file": (img.name, img.read(), img.type or "image/jpeg")}
        data = {"user_id": user_id}
        code, res = post_file("/enroll", files=files, data=data)
        if code == 200:
            st.success(f"Enroll thành công: {res}")
        elif code == 400:
            st.error("No face detected — hãy chọn ảnh thấy rõ khuôn mặt.")
        else:
            st.error(f"Lỗi ({code}): {res}")

# --- Login Tab ---
with tab_login:
    st.subheader("🔐 Đăng nhập bằng khuôn mặt")
    login_img = st.file_uploader("Upload ảnh để login", type=["jpg","jpeg","png"], key="login_file")
    th = st.slider("Threshold (cosine)", min_value=0.50, max_value=0.99, value=0.80, step=0.01,
                   help="Chọn theo ROC/Phase 2. Cosine ≥ threshold → accept.")

    c1, c2 = st.columns(2)
    with c1:
        login_preview = st.checkbox("Xem trước ảnh login", True, key="login_prev")
    with c2:
        btn_login = st.button("Login", disabled=not login_img)

    if login_img and login_preview:
        try:
            st.image(Image.open(login_img), caption="Ảnh login", use_column_width=True)
        except Exception:
            st.warning("Không hiển thị được ảnh (nhưng vẫn có thể login).")

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

            # Kết quả
            st.markdown("---")
            st.metric("Cosine score", f"{score:.4f}")
            st.progress(min(max((score - 0.5)/0.49, 0), 1.0))  # progress bar tương đối

            if accepted:
                st.success(f"✅ Accepted: khớp với **{best_user}** (score={score:.4f} ≥ threshold={threshold:.2f})")
            else:
                st.error(f"❌ Rejected: best match **{best_user}** (score={score:.4f} < threshold={threshold:.2f})")
        else:
            st.error(f"Lỗi ({code}): {res}")

    st.caption("Gợi ý: Sau Phase 2 (ROC), hãy đặt threshold tương ứng FPR mục tiêu (VD: 0.001).")

# --- Users Tab ---
with tab_users:
    st.subheader("👥 Danh sách Users đã enroll")
    btn_refresh = st.button("Refresh")
    if btn_refresh or True:
        code, res = get_json("/users")
        if code == 200 and isinstance(res, dict):
            st.json(res)
        else:
            st.error(f"Lỗi ({code}): {res}")

    st.markdown("---")
    st.caption("Bạn có thể dùng `/export_embeddings` để lấy toàn bộ embeddings → phục vụ Phase 2 (ROC).")
    if st.button("Export embeddings (xem nhanh)"):
        code, res = get_json("/export_embeddings")
        if code == 200 and isinstance(res, dict):
            st.json(res if len(str(res)) < 8000 else {"info":"Kết quả dài; xem bằng curl/HTTP client tốt hơn."})
        else:
            st.error(f"Lỗi ({code}): {res}")
