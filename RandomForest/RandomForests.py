# -*- coding: utf-8 -*-
"""

RANDOM FOREST BASELINE 

Lưu ý : sửa DATA_DIR và OUT_DIR ; 

Trong SPYDER: gõ vào IPython Console
        %pip install scikit-learn scikit-image pillow joblib matplotlib tqdm
    Nếu thấy dòng "Requirement already satisfied": đã cài rồi.
    Nếu báo ModuleNotFoundError: vào menu Consoles - Restart kernel.
    Ctrl+A rồi Run


TRONG VISUAL STUDIO CODE:
    Cài extension "Python" của Microsoft: bấm Ctrl+Shift+X, gõ "Python", bấm Install.
    Chọn Python của Anaconda: bấm Ctrl+Shift+P, gõ "Python: Select Interpreter",
       chọn dòng có chữ "anaconda3".
    Run từng cell

Ý tưởng chung:
    Ảnh -> trích đặc trưng (vector số) -> Random Forest học -> dự đoán Real/AI


"""

#%%  1. CẤU HÌNH 
from pathlib import Path


DATA_DIR = Path(r"D:\A_TienXuLyAnh\Dataset_Final")    

OUT_DIR  = Path(r"D:\RandomForest\outputs")           

IMG_SIZE    = 224                        
SEED        = 42                          
CLASS_NAMES = ["Real", "AI-Generated"]     
EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"} 
FFT_BINS    = 32                         


PARAM_GRID = {
    "n_estimators":     [200, 400],      
    "max_depth":        [None, 30],     
    "max_features":     ["sqrt", "log2"], 
    "min_samples_leaf": [1, 2],          
}

OUT_DIR.mkdir(parents=True, exist_ok=True)         
(OUT_DIR / "cache").mkdir(exist_ok=True)           


#%% 2. THƯ VIỆN
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


#%% 3. TRÍCH XUẤT ĐẶC TRƯNG 
def extract_features(img):
    """
    Nhận 1 ảnh PIL, trả về 1 vector số gồm 4 nhóm đặc trưng:
      a) HOG      : hình dạng, đường biên cục bộ
      b) LBP      : kết cấu bề mặt (texture) 
      c) HSV hist : phân bố màu sắc, độ bão hòa, độ sáng
      d) FFT      : phổ tần số 
    """
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))      
    gray = np.asarray(img.convert("L"), dtype=np.uint8)        

    #  a) HOG: chia ảnh thành ô 32x32, đo hướng của cạnh trong từng ô 
    f_hog = hog(gray, orientations=9, pixels_per_cell=(32, 32),
                cells_per_block=(2, 2), block_norm="L2-Hys")

    # b) LBP (uniform, P=8, R=1): so sánh mỗi điểm ảnh với 8 điểm xung quanh 
    # Sau đó đếm tần suất 10 loại mẫu -> histogram 10 số (chia cho tổng để chuẩn hóa)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    f_lbp = np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size

    # c) Histogram màu HSV: mỗi kênh (H, S, V) chia 16 khoảng -> 48 số 
    hsv = np.asarray(img.convert("HSV"))
    f_col = np.concatenate([
        np.histogram(hsv[..., c], bins=16, range=(0, 256), density=True)[0]
        for c in range(3)])

    # d) Phổ tần số FFT 
    # Biến đổi Fourier 2D -> lấy độ lớn (log) -> tính trung bình theo từng vòng tròn
    # tính từ tâm ra ngoài (32 vòng) -> vector 32 số (tần số thấp -> cao)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray / 255.0))))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - IMG_SIZE // 2, xx - IMG_SIZE // 2)      
    rbin = np.minimum((r / (IMG_SIZE / 2) * FFT_BINS).astype(int), FFT_BINS - 1)
    f_fft = (np.bincount(rbin.ravel(), mag.ravel(), FFT_BINS) /
             np.maximum(np.bincount(rbin.ravel(), minlength=FFT_BINS), 1))


    return np.concatenate([f_hog, f_lbp, f_col, f_fft]).astype(np.float32)


def feature_groups():
    """Trả về số lượng đặc trưng của từng nhóm."""
    n_hog = len(hog(np.zeros((IMG_SIZE, IMG_SIZE)), orientations=9,
                    pixels_per_cell=(32, 32), cells_per_block=(2, 2)))
    return {"HOG": n_hog, "LBP": 10, "HSV hist": 48, "FFT": FFT_BINS}


def _safe_feat(path):
    """Trích đặc trưng 1 ảnh"""
    try:
        with Image.open(path) as im:
            return extract_features(im)
    except Exception:
        return None


def load_split(split_dir):
    """
    Đọc tập (train / validation / test), trả về:
        X : ma trận đặc trưng  (số ảnh * số đặc trưng)
        y : nhãn               (0 = Real, 1 = AI)
    """
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
    if not paths:
        raise FileNotFoundError(f"Khong tim thay anh trong {split_dir}")

    # Trích đặc trưng song song trên tất cả lõi CPU (n_jobs=-1)
    feats = Parallel(n_jobs=-1)(delayed(_safe_feat)(p)
                                for p in tqdm(paths, desc=f"Trich xuat {split_dir.name}"))
    keep = [i for i, f in enumerate(feats) if f is not None]     # giữ lại ảnh đọc được
    print(f"{split_dir.name}: bo qua {len(paths) - len(keep)} anh loi/khong mo duoc")
    X = np.stack([feats[i] for i in keep])
    y = np.array([labels[i] for i in keep])
    np.savez_compressed(cache, X=X, y=y)                       
    return X, y



#  CHẠY CODE
if __name__ == "__main__":
 

    val_dir = DATA_DIR / "validation" 
    X_train, y_train = load_split(DATA_DIR / "train")
    X_val,   y_val   = load_split(val_dir)
    X_test,  y_test  = load_split(DATA_DIR / "test")

    print("train:", X_train.shape, "| val:", X_val.shape, "| test:", X_test.shape)
    # Kiểm tra cân bằng lớp
    print("So anh Real/AI  train:", np.bincount(y_train),
          "| val:", np.bincount(y_val), "| test:", np.bincount(y_test))


    #  TÌM SIÊU THAM SỐ TỐT NHẤT 
    # Quy tắc chuẩn: học trên TRAIN, chọn tham số bằng VALIDATION, TEST chỉ dùng 1 lần cuối.
    best_model, best_f1, best_params, grid_log = None, -1, None, []
    grid = list(ParameterGrid(PARAM_GRID))                        

    for i, p in enumerate(grid, 1):
        t = time.time()
        rf = RandomForestClassifier(
            **p,                     
            bootstrap=True,         
            oob_score=True,          
            n_jobs=-1,              
            random_state=SEED      
        ).fit(X_train, y_train)     
        f1 = f1_score(y_val, rf.predict(X_val))                     
        grid_log.append({**p, "oob_acc": rf.oob_score_, "val_f1": f1})
        print(f"[{i}/{len(grid)}] {p} | OOB={rf.oob_score_:.4f} | val F1={f1:.4f} "
              f"({time.time() - t:.0f}s)")
        if f1 > best_f1:                                             
            best_model, best_f1, best_params = rf, f1, p

    print("\n Tham so tot nhat:", best_params, "| val F1 =", round(best_f1, 4))


    # ĐÁNH GIÁ 
    def evaluate(model, X, y, name):
        """Tính các chỉ số, in báo cáo, vẽ và lưu ma trận nhầm lẫn."""
        pred  = model.predict(X)                     
        proba = model.predict_proba(X)[:, 1]         
        m = {"accuracy":  accuracy_score(y, pred),   
             "precision": precision_score(y, pred), 
             "recall":    recall_score(y, pred),     
             "f1":        f1_score(y, pred),         
             "roc_auc":   roc_auc_score(y, proba)}   
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


    #  MỨC ĐÓNG GÓP CỦA TỪNG NHÓM ĐẶC TRƯNG 

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
    print(group_imp)


    # LƯU MODEL + KẾT QUẢ 
    # File .joblib dùng cho giao diện Streamlit; file .json chứa toàn bộ số liệu cho báo cáo
    joblib.dump({"model": best_model, "img_size": IMG_SIZE, "classes": CLASS_NAMES},
                OUT_DIR / "random_forest.joblib", compress=3)
    json.dump({"best_params": best_params, "oob_accuracy": best_model.oob_score_,
               "val": res_val, "test": res_test,
               "feature_importance_by_group": group_imp, "grid_log": grid_log},
              open(OUT_DIR / "rf_results.json", "w"), indent=2)
    print("Da luu vao:", OUT_DIR.resolve())
# %%
