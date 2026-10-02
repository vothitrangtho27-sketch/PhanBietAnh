import time
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

import ui
from features import extract_features_rf
from predictors import CNNPredictor, SklearnPredictor

BASE = Path(__file__).parent / "models"

# Khi B, C, D nộp file: thả vào thư mục models/ đúng tên bên dưới là chạy.
MODEL_REGISTRY = {
    "CNN": {"file": "cnn_model.keras",              # hoặc .h5 / .pt / .pth
            "abbr": "CNN", "desc": "Mạng nơ-ron tích chập",
            "build": lambda p: CNNPredictor(p)},
    "SVM": {"file": "svm.joblib",
            "abbr": "SVM", "desc": "Support Vector Machine",
            "build": lambda p: SklearnPredictor(p, extract_features_rf)},
    "Logistic Regression": {"file": "logistic_regression.joblib",
                            "abbr": "LR", "desc": "Hồi quy logistic",
                            "build": lambda p: SklearnPredictor(p, extract_features_rf)},
    "Random Forest": {"file": "random_forest.joblib",
                      "abbr": "RF", "desc": "Rừng ngẫu nhiên",
                      "build": lambda p: SklearnPredictor(p, extract_features_rf)},
}
NAMES = list(MODEL_REGISTRY)
MODE_ONE, MODE_ALL = "Một mô hình", "Cả 4 mô hình"


def find_model_file(name):
    stem = Path(MODEL_REGISTRY[name]["file"]).stem
    for f in sorted(BASE.glob(stem + ".*")):
        return f
    return None


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model(name):
    path = find_model_file(name)
    if path is None:
        return None, f"Chưa có file mô hình trong `models/` (cần `{MODEL_REGISTRY[name]['file']}`)."
    try:
        return MODEL_REGISTRY[name]["build"](path), None
    except Exception as e:
        return None, f"Không tải được mô hình: {e}"


def run_model(name, img):
    model, err = load_model(name)
    if err:
        return {"error": err}
    try:
        t0 = time.time()
        label, p_ai = model.predict(img)
        return {"label": label, "p_ai": p_ai, "ms": (time.time() - t0) * 1000}
    except Exception as e:
        return {"error": f"Lỗi khi dự đoán: {e}"}


def item(name, res=None):
    m = MODEL_REGISTRY[name]
    if res is not None and "error" not in res:
        state = "result"
    elif res is not None:
        state = "missing" if find_model_file(name) is None else "error"
    else:
        state = "ready" if find_model_file(name) else "missing"
    return {"name": name, "abbr": m["abbr"], "desc": m["desc"], "state": state, "res": res}


# ───────────────────────── Trang ─────────────────────────
st.set_page_config(page_title="Soi ảnh · Thật hay AI", page_icon="🔍",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.FONT_IMPORT, unsafe_allow_html=True)
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown(ui.masthead() + ui.headline(), unsafe_allow_html=True)

left, right = st.columns([1, 1.12], gap="large")

with left:
    up = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp", "bmp"],
                          label_visibility="collapsed",
                          help="Hỗ trợ JPG, PNG, WEBP, BMP")

img = None
if up is not None:
    try:
        img = Image.open(up)
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        with left:
            st.error("File không phải ảnh hợp lệ hoặc bị lỗi. Hãy thử một ảnh khác.")
        img = None

ran = False
with right:
    mode = st.segmented_control("Chế độ", [MODE_ALL, MODE_ONE], default=MODE_ALL,
                                label_visibility="collapsed") or MODE_ALL
    if mode == MODE_ONE:
        chosen = st.selectbox(
            "Mô hình", NAMES, index=NAMES.index("Random Forest"),
            format_func=lambda n: n if find_model_file(n) else f"{n} (chưa có file)")
        selected = [chosen]
    else:
        selected = NAMES

    go = st.button("Dự đoán bằng cả 4 mô hình" if len(selected) > 1 else f"Dự đoán bằng {selected[0]}",
                   type="primary", width="stretch", disabled=img is None)

    if img is None:
        st.markdown(ui.rows_panel([item(n) for n in selected], title="Mô hình sẽ chạy", with_head=False),
                    unsafe_allow_html=True)
        st.caption("Tải một ảnh lên ở bên trái để bắt đầu.")
    elif not go:
        st.markdown(ui.rows_panel([item(n) for n in selected], title="Mô hình sẽ chạy", with_head=False),
                    unsafe_allow_html=True)
    else:
        with st.spinner("Đang phân tích ảnh..."):
            results = {n: run_model(n, img) for n in selected}
        ok = {n: r for n, r in results.items() if "error" not in r}
        if not ok:
            st.warning("Chưa có mô hình nào sẵn sàng để dự đoán. "
                       "Hãy thả file mô hình vào thư mục `models/`.")
            st.markdown(ui.rows_panel([item(n, r) for n, r in results.items()],
                                      title="Mô hình sẽ chạy", with_head=False),
                        unsafe_allow_html=True)
        else:
            ran = True
            if len(selected) == 1:
                n, r = next(iter(ok.items()))
                conf = r["p_ai"] if r["label"] == 1 else 1 - r["p_ai"]
                st.markdown(ui.verdict(r["label"] == 1, r["p_ai"],
                                       f"Mô hình <b>{n}</b> · độ tin cậy <b>{conf:.0%}</b> · "
                                       f"xác suất AI <b>{r['p_ai']:.0%}</b>"),
                            unsafe_allow_html=True)
            else:
                n_ai = sum(r["label"] == 1 for r in ok.values())
                n_real = len(ok) - n_ai
                avg = sum(r["p_ai"] for r in ok.values()) / len(ok)
                is_ai = n_ai > n_real or (n_ai == n_real and avg >= 0.5)
                tie = " Hòa phiếu nên xét theo xác suất trung bình." if n_ai == n_real else ""
                skipped = len(selected) - len(ok)
                extra = f" ({skipped} mô hình chưa có file)" if skipped else ""
                st.markdown(ui.verdict(
                    is_ai, avg,
                    f"<b>{n_ai}/{len(ok)}</b> mô hình cho là ảnh AI, <b>{n_real}/{len(ok)}</b> cho là ảnh thật{extra}. "
                    f"Xác suất AI trung bình <b>{avg:.0%}</b>.{tie}"), unsafe_allow_html=True)
            st.markdown(ui.rows_panel([item(n, r) for n, r in results.items()]),
                        unsafe_allow_html=True)

if img is not None:
    with left:
        st.markdown(ui.image_frame(img, up.name, scanning=ran), unsafe_allow_html=True)
# để chạy : python -m streamlit run app.py