# -*- coding: utf-8 -*-
from pathlib import Path
import json
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           ConfusionMatrixDisplay, classification_report)

# Tự động lấy đường dẫn thư mục gốc (lùi ra ngoài 1 cấp từ thư mục SVM_LogReg)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "A_TienXuLyAnh" / "Dataset_Final2"
OUT_DIR = Path(__file__).resolve().parent / "outputs_v2"
CACHE_DIR = OUT_DIR / "cache"

IMG_SIZE = 224
SEED = 42
CLASS_NAMES = ["real", "fake"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FFT_BINS = 32

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def extract_features(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    f_hog = hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), block_norm="L2-Hys")
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    f_lbp = (np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size)
    hsv = np.asarray(img.convert("HSV"))
    f_col = np.concatenate([np.histogram(hsv[..., c], bins=16, range=(0, 256), density=True)[0] for c in range(3)])
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray / 255.0))))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - IMG_SIZE // 2, xx - IMG_SIZE // 2)
    rbin = np.minimum((r / (IMG_SIZE / 2) * FFT_BINS).astype(int), FFT_BINS - 1)
    f_fft = (np.bincount(rbin.ravel(), mag.ravel(), FFT_BINS) / np.maximum(np.bincount(rbin.ravel(), minlength=FFT_BINS), 1))
    return np.concatenate([f_hog, f_lbp, f_col, f_fft]).astype(np.float32)

def feature_groups():
    n_hog = len(hog(np.zeros((IMG_SIZE, IMG_SIZE)), orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2)))
    return {"HOG": n_hog, "LBP": 10, "HSV hist": 48, "FFT": FFT_BINS}

def _safe_feat(path):
    try:
        with Image.open(path) as im:
            return extract_features(im)
    except Exception:
        return None

def load_split(split_dir):
    split_dir = Path(split_dir)
    cache = CACHE_DIR / f"{split_dir.name}_features.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"[cache] {split_dir.name}: {len(d['y'])} ảnh")
        return d["X"], d["y"]
    paths, labels = [], []
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = 0 if "real" in cls_dir.name.lower() else 1
        files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in EXTS]
        paths += files
        labels += [label] * len(files)
    if not paths: raise FileNotFoundError(f"Không tìm thấy ảnh trong: {split_dir}")
    feats = Parallel(n_jobs=-1)(delayed(_safe_feat)(p) for p in tqdm(paths, desc=f"Trích xuất {split_dir.name}"))
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep])
    y = np.array([labels[i] for i in keep])
    np.savez_compressed(cache, X=X, y=y)
    return X, y

def evaluate(model, X, y, name, model_name):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
               "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}
    print(f"\n{'=' * 70}\n{model_name.upper()} - {name.upper()}\n{'=' * 70}")
    print({k: round(v, 4) for k, v in metrics.items()})
    print("\n" + classification_report(y, pred, target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(y, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(values_format="d")
    plt.title(f"{model_name} - Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{name.lower()}.png", dpi=150)
    plt.show()
    metrics["confusion_matrix"] = cm.tolist()
    return metrics

if __name__ == "__main__":
    print("Dataset:", DATA_DIR)
    X_train, y_train = load_split(DATA_DIR / "train")
    X_val, y_val = load_split(DATA_DIR / "validation")
    X_test, y_test = load_split(DATA_DIR / "test")

    # Dùng LinearSVC kết hợp CalibratedClassifierCV(ensemble=False) để chạy cực nhanh và hỗ trợ predict_proba không bị lỗi thời
    svm_model = Pipeline([
        ("scaler", StandardScaler()), 
        ("svm", CalibratedClassifierCV(estimator=LinearSVC(C=1.0, random_state=SEED, dual=False, max_iter=2000), method="sigmoid", ensemble=False))
    ])
    print("Đang huấn luyện SVM (Linear + Calibration nhanh)...")
    svm_model.fit(X_train, y_train)
    svm_val = evaluate(svm_model, X_val, y_val, "validation", "SVM")
    svm_test = evaluate(svm_model, X_test, y_test, "test", "SVM")

    lr_model = Pipeline([
        ("scaler", StandardScaler()), 
        ("logistic_regression", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    ])
    print("Đang huấn luyện Logistic Regression...")
    lr_model.fit(X_train, y_train)
    lr_val = evaluate(lr_model, X_val, y_val, "validation", "Logistic Regression")
    lr_test = evaluate(lr_model, X_test, y_test, "test", "Logistic Regression")

    joblib.dump({"model": svm_model, "img_size": IMG_SIZE, "classes": CLASS_NAMES, "feature_groups": feature_groups()}, OUT_DIR / "svm.joblib", compress=3)
    joblib.dump({"model": lr_model, "img_size": IMG_SIZE, "classes": CLASS_NAMES, "feature_groups": feature_groups()}, OUT_DIR / "logistic_regression.joblib", compress=3)
    
    results = {
        "dataset": str(DATA_DIR), "feature_groups": feature_groups(), "feature_count": int(X_train.shape[1]),
        "class_names": CLASS_NAMES, "label_mapping": {"real": 0, "fake": 1},
        "SVM": {"parameters": {"kernel": "linear", "C": 1.0, "calibration": "sigmoid"}, "validation": svm_val, "test": svm_test},
        "Logistic_Regression": {"parameters": {"C": 1.0, "max_iter": 2000}, "validation": lr_val, "test": lr_test}
    }
    with open(OUT_DIR / "svm_logreg_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nĐã lưu kết quả thành công!")