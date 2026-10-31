# -*- coding: utf-8 -*-
from pathlib import Path
import json, time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

# Tự động lấy đường dẫn thư mục gốc (lùi ra ngoài 1 cấp từ thư mục RandomForest)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "A_TienXuLyAnh" / "Dataset_Final2"
OUT_DIR  = Path(__file__).resolve().parent / "outputs_v2"

IMG_SIZE    = 224
SEED        = 42
CLASS_NAMES = ["real", "fake"]
EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FFT_BINS    = 32

PARAM_GRID = {
    "n_estimators": [200, 400], "max_depth": [None, 30],
    "max_features": ["sqrt", "log2"], "min_samples_leaf": [1, 2],
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "cache").mkdir(exist_ok=True)

def extract_features(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    f_hog = hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), block_norm="L2-Hys")
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    f_lbp = np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size
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
        with Image.open(path) as im: return extract_features(im)
    except Exception: return None

def load_split(split_dir):
    split_dir = Path(split_dir)
    cache = OUT_DIR / "cache" / f"{split_dir.name}_features.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"[cache] {split_dir.name}: {len(d['y'])} anh")
        return d["X"], d["y"]
    paths, labels = [], []
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = 0 if "real" in cls_dir.name.lower() else 1
        files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in EXTS]
        paths += files
        labels += [label] * len(files)
    if not paths: raise FileNotFoundError(f"Khong tim thay anh trong {split_dir}")
    feats = Parallel(n_jobs=-1)(delayed(_safe_feat)(p) for p in tqdm(paths, desc=f"Trich xuat {split_dir.name}"))
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep])
    y = np.array([labels[i] for i in keep])
    np.savez_compressed(cache, X=X, y=y)
    return X, y

if __name__ == "__main__":
    X_train, y_train = load_split(DATA_DIR / "train")
    X_val,   y_val   = load_split(DATA_DIR / "validation")
    X_test,  y_test  = load_split(DATA_DIR / "test")

    best_model, best_f1, best_params, grid_log = None, -1, None, []
    grid = list(ParameterGrid(PARAM_GRID))

    for i, p in enumerate(grid, 1):
        t = time.time()
        rf = RandomForestClassifier(**p, bootstrap=True, oob_score=True, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
        f1 = f1_score(y_val, rf.predict(X_val))
        grid_log.append({**p, "oob_acc": rf.oob_score_, "val_f1": f1})
        print(f"[{i}/{len(grid)}] {p} | OOB={rf.oob_score_:.4f} | val F1={f1:.4f} ({time.time() - t:.0f}s)")
        if f1 > best_f1: best_model, best_f1, best_params = rf, f1, p

    def evaluate(model, X, y, name):
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
             "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}
        print(f"\n {name.upper()} ")
        print({k: round(v, 4) for k, v in m.items()})
        print(classification_report(y, pred, target_names=CLASS_NAMES, digits=4))
        cm = confusion_matrix(y, pred)
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
        plt.title(f"Random Forest - Confusion Matrix ({name})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.show()
        m["confusion_matrix"] = cm.tolist()
        return m

    res_val  = evaluate(best_model, X_val,  y_val,  "validation")
    res_test = evaluate(best_model, X_test, y_test, "test")

    imp, start, group_imp = best_model.feature_importances_, 0, {}
    for g, n in feature_groups().items():
        group_imp[g] = float(imp[start:start + n].sum())
        start += n

    plt.figure(figsize=(6, 4))
    plt.bar(group_imp.keys(), group_imp.values(), color="#4C78A8")
    plt.ylabel("Tong Gini importance")
    plt.title("Random Forest - Feature importance theo nhom")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.show()

    joblib.dump({"model": best_model, "img_size": IMG_SIZE, "classes": CLASS_NAMES}, OUT_DIR / "random_forest.joblib", compress=3)
    json.dump({"best_params": best_params, "oob_accuracy": best_model.oob_score_, "val": res_val, "test": res_test,
               "feature_importance_by_group": group_imp, "grid_log": grid_log},
              open(OUT_DIR / "rf_results.json", "w"), indent=2)
    print("Da luu vao:", OUT_DIR.resolve())