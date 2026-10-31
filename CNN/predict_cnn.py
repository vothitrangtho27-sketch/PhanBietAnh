import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image

LABEL_VI = {"real": "Ảnh thật", "fake": "Ảnh do AI tạo"}

def load_cnn_model(model_path="cnn_output_v2/cnn_model.keras"):
    from tensorflow import keras
    return keras.models.load_model(model_path)

def _to_pil(image):
    if isinstance(image, Image.Image): img = image
    elif isinstance(image, (str, Path)): img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)): img = Image.open(io.BytesIO(image))
    else: img = Image.open(image)
    return img.convert("RGB")  

def predict_image(model, image, threshold=0.5):
    size = model.input_shape[1]  
    img = _to_pil(image).resize((size, size), Image.LANCZOS)
    x = np.asarray(img, dtype="float32")[None, ...]  
    prob_real = float(model.predict(x, verbose=0)[0, 0])
    label = "real" if prob_real >= threshold else "fake"
    return {"label": label, "label_vi": LABEL_VI[label], "prob_real": prob_real, "prob_fake": 1.0 - prob_real, "confidence": prob_real if label == "real" else 1.0 - prob_real}

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit("Cách dùng: python predict_cnn.py <đường_dẫn_ảnh>")
    model_path = "cnn_output_v2/cnn_model.keras"
    r = predict_image(load_cnn_model(model_path), sys.argv[1])
    print(f"Kết quả: {r['label_vi']} | P(Thật) = {r['prob_real']:.4f} | P(AI) = {r['prob_fake']:.4f}")