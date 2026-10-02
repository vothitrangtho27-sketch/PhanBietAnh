"""Mỗi mô hình có hàm predict(img) -> (nhãn, xác suất AI).
Quy ước: nhãn 0 = Real, 1 = AI-Generated.
"""

from pathlib import Path
import importlib

import joblib
import numpy as np
from PIL import Image


CLASS_NAMES = ["Real", "AI-Generated"]


class SklearnPredictor:
    def __init__(self, path, feature_fn):
        obj = joblib.load(path)

        if isinstance(obj, dict):
            self.model = obj["model"]
            self.scaler = obj.get("scaler")
            self.pca = obj.get("pca")
        else:
            self.model = obj
            self.scaler = None
            self.pca = None

        self.feature_fn = feature_fn

    def predict(self, img: Image.Image):
        x = self.feature_fn(img).reshape(1, -1)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        if self.pca is not None:
            x = self.pca.transform(x)

        if hasattr(self.model, "predict_proba"):
            p_ai = float(
                self.model.predict_proba(x)[0, 1]
            )
        elif hasattr(self.model, "decision_function"):
            d = float(
                self.model.decision_function(x)[0]
            )

            d = np.clip(d, -50, 50)

            p_ai = 1.0 / (
                1.0 + np.exp(-d)
            )
        else:
            prediction = int(
                self.model.predict(x)[0]
            )

            p_ai = float(prediction)

        label = int(p_ai >= 0.5)

        return label, p_ai


class CNNPredictor:
    IMG_SIZE = 224
    # Model CNN của bạn đã có lớp Rescaling(1/255) bên trong nên ta không chia 255 ở đây nữa
    NORMALIZE = "none" 

    def __init__(self, path):
        self.path = Path(path)
        suffix = self.path.suffix.lower()

        if suffix in {".keras", ".h5"}:
            self.kind = "keras"
            self._load_keras()
        elif suffix in {".pt", ".pth"}:
            self.kind = "torch"
            self._load_torch()
        else:
            raise ValueError(f"Định dạng CNN không được hỗ trợ: {suffix}")

    def _load_keras(self):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.path))
        except ImportError:
            raise ImportError("Chưa cài TensorFlow. Không thể chạy mô hình CNN .keras")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi load model Keras: {e}")

    def _load_torch(self):
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            raise ImportError("Chưa cài PyTorch. Không thể chạy mô hình CNN .pt/.pth")

        self.torch = torch
        try:
            self.model = torch.jit.load(str(self.path), map_location="cpu")
        except Exception:
            self.model = torch.load(str(self.path), map_location="cpu", weights_only=False)
        self.model.eval()

    def _prep(self, img):
        image = img.convert("RGB").resize((self.IMG_SIZE, self.IMG_SIZE))
        array = np.asarray(image, dtype=np.float32)
        
        if self.NORMALIZE == "rescale":
            array /= 255.0
        elif self.NORMALIZE == "imagenet":
            array = array / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            array = (array - mean) / std
            
        return array.astype(np.float32)

    def predict(self, img):
        array = self._prep(img)

        if self.kind == "keras":
            output = self.model.predict(array[None], verbose=0).ravel()
        else:
            tensor = self.torch.from_numpy(array.transpose(2, 0, 1)[None])
            with self.torch.no_grad():
                output = self.model(tensor)
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()
            else:
                output = np.asarray(output)
            output = output.ravel()

        if output.size == 1:
            prob_real = float(output[0])
            # QUAN TRỌNG: Model CNN trả về xác suất ẢNH THẬT (prob_real)
            # App cần xác suất ẢNH AI (p_ai) => p_ai = 1 - prob_real
            p_ai = 1.0 - prob_real
        else:
            # Trường hợp model trả về 2 lớp (softmax)
            output = output.astype(np.float64)
            output = output - np.max(output)
            exp_output = np.exp(output)
            probabilities = exp_output / exp_output.sum()
            p_ai = float(probabilities[1])

        label = int(p_ai >= 0.5)
        return label, p_ai