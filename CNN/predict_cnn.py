"""
Dùng trong Streamlit:
    import streamlit as st
    from predict_cnn import load_cnn_model, predict_image

    @st.cache_resource    # chỉ nạp mô hình 1 lần
    def get_model():
        return load_cnn_model("cnn_output/cnn_model.keras")

    file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "webp"])
    if file:
        result = predict_image(get_model(), file)   
        st.write(result["label_vi"], f'({result["confidence"]:.1%})')

"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LABEL_VI = {"real": "Ảnh thật", "fake": "Ảnh do AI tạo"}


def load_cnn_model(model_path="cnn_output/cnn_model.keras"):
    from tensorflow import keras
    return keras.models.load_model(model_path)


def _to_pil(image):
    """Nhận đường dẫn / PIL.Image / bytes / file-like (st.file_uploader) -> PIL RGB."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    else:  
        img = Image.open(image)
    return img.convert("RGB")  


def predict_image(model, image, threshold=0.5):
    """
    Trả về dict:
        label        : "real" hoặc "fake"
        label_vi     : "Ảnh thật" / "Ảnh do AI tạo"
        prob_real    : xác suất ảnh là ảnh thật (0-1)
        prob_fake    : xác suất ảnh do AI tạo (0-1)
        confidence   : độ tin cậy của nhãn được chọn (0.5-1)
    """
    size = model.input_shape[1]  
    img = _to_pil(image).resize((size, size), Image.LANCZOS)
    x = np.asarray(img, dtype="float32")[None, ...]  

    prob_real = float(model.predict(x, verbose=0)[0, 0])
    label = "real" if prob_real >= threshold else "fake"
    return {
        "label": label,
        "label_vi": LABEL_VI[label],
        "prob_real": prob_real,
        "prob_fake": 1.0 - prob_real,
        "confidence": prob_real if label == "real" else 1.0 - prob_real,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Cách dùng: python predict_cnn.py <ảnh> [đường_dẫn_model]")
    model_path = sys.argv[2] if len(sys.argv) > 2 else "cnn_output/cnn_model.keras"
    r = predict_image(load_cnn_model(model_path), sys.argv[1])
    print(f"{r['label_vi']}  |  P(thật) = {r['prob_real']:.4f}  |  P(AI) = {r['prob_fake']:.4f}")
