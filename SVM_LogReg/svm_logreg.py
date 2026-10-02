# -*- coding: utf-8 -*-
"""
PHẦN C - SVM + LOGISTIC REGRESSION BASELINE

Ý tưởng:
Ảnh
  ↓
HOG + LBP + HSV + FFT
  ↓
Vector đặc trưng
  ↓
 ┌─────────────────────┐
 │                     │
SVM              Logistic Regression
 │                     │
 └──────────┬──────────┘
            ↓
     Real / AI-Generated
"""

# ============================================================
# 1. CẤU HÌNH
# ============================================================

from pathlib import Path


# Thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parent.parent


# Dataset sau tiền xử lý của phần A
DATA_DIR = (
    ROOT_DIR
    / "A_TienXuLyAnh"
    / "Dataset_Final"
)


# Thư mục output của phần C
OUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
)


# Thư mục cache
CACHE_DIR = OUT_DIR / "cache"


# Kích thước ảnh
IMG_SIZE = 224


# Random seed
SEED = 42


# Nhãn
# 0 = Real
# 1 = AI-Generated
CLASS_NAMES = [
    "Real",
    "AI-Generated"
]


# Định dạng ảnh
EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# Số bin FFT
FFT_BINS = 32


# Tạo thư mục
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. THƯ VIỆN
# ============================================================

import json
import time

import numpy as np
import joblib
import matplotlib.pyplot as plt

from PIL import Image

from joblib import Parallel, delayed
from tqdm import tqdm

from skimage.feature import (
    hog,
    local_binary_pattern
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG
# ============================================================

def extract_features(img):
    """
    Trích 4 nhóm đặc trưng:

    1. HOG       : hình dạng / đường biên
    2. LBP       : kết cấu / texture
    3. HSV hist  : màu sắc
    4. FFT       : phổ tần số
    """

    # --------------------------------------------------------
    # Resize ảnh
    # --------------------------------------------------------

    img = img.convert(
        "RGB"
    ).resize(
        (IMG_SIZE, IMG_SIZE)
    )


    # Chuyển sang grayscale
    gray = np.asarray(
        img.convert("L"),
        dtype=np.uint8
    )


    # ========================================================
    # A. HOG
    # ========================================================

    f_hog = hog(
        gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


    # ========================================================
    # B. LBP
    # ========================================================

    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )


    f_lbp = (
        np.bincount(
            lbp.astype(int).ravel(),
            minlength=10
        )
        / lbp.size
    )


    # ========================================================
    # C. HSV HISTOGRAM
    # ========================================================

    hsv = np.asarray(
        img.convert("HSV")
    )


    f_col = np.concatenate([
        np.histogram(
            hsv[..., c],
            bins=16,
            range=(0, 256),
            density=True
        )[0]
        for c in range(3)
    ])


    # ========================================================
    # D. FFT
    # ========================================================

    mag = np.log1p(
        np.abs(
            np.fft.fftshift(
                np.fft.fft2(
                    gray / 255.0
                )
            )
        )
    )


    yy, xx = np.indices(
        mag.shape
    )


    r = np.hypot(
        yy - IMG_SIZE // 2,
        xx - IMG_SIZE // 2
    )


    rbin = np.minimum(
        (
            r
            / (IMG_SIZE / 2)
            * FFT_BINS
        ).astype(int),
        FFT_BINS - 1
    )


    f_fft = (
        np.bincount(
            rbin.ravel(),
            mag.ravel(),
            FFT_BINS
        )
        /
        np.maximum(
            np.bincount(
                rbin.ravel(),
                minlength=FFT_BINS
            ),
            1
        )
    )


    # ========================================================
    # GHÉP TẤT CẢ ĐẶC TRƯNG
    # ========================================================

    return np.concatenate([
        f_hog,
        f_lbp,
        f_col,
        f_fft
    ]).astype(
        np.float32
    )


# ============================================================
# 4. THỐNG KÊ ĐẶC TRƯNG
# ============================================================

def feature_groups():
    """
    Trả về số lượng đặc trưng của từng nhóm.
    """

    n_hog = len(
        hog(
            np.zeros(
                (IMG_SIZE, IMG_SIZE)
            ),
            orientations=9,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2)
        )
    )


    return {
        "HOG": n_hog,
        "LBP": 10,
        "HSV hist": 48,
        "FFT": FFT_BINS
    }


# ============================================================
# 5. ĐỌC VÀ TRÍCH ĐẶC TRƯNG 1 ẢNH
# ============================================================

def _safe_feat(path):

    try:

        with Image.open(path) as im:

            return extract_features(im)

    except Exception:

        return None


# ============================================================
# 6. LOAD TRAIN / VALIDATION / TEST
# ============================================================

def load_split(split_dir):

    split_dir = Path(
        split_dir
    )


    # --------------------------------------------------------
    # File cache
    # --------------------------------------------------------

    cache = (
        CACHE_DIR
        / f"{split_dir.name}_features.npz"
    )


    # --------------------------------------------------------
    # Nếu cache tồn tại
    # --------------------------------------------------------

    if cache.exists():

        d = np.load(
            cache
        )

        X = d["X"]
        y = d["y"]


        # Kiểm tra cache có dữ liệu hay không
        if len(y) > 0:

            print(
                f"[cache] "
                f"{split_dir.name}: "
                f"{len(y)} ảnh"
            )

            return X, y


        # Cache rỗng
        print(
            f"[cache] "
            f"{split_dir.name} "
            f"đang rỗng -> tạo lại"
        )

        cache.unlink()


    # --------------------------------------------------------
    # Tìm ảnh
    # --------------------------------------------------------

    paths = []
    labels = []


    for cls_dir in sorted(
        p
        for p in split_dir.iterdir()
        if p.is_dir()
    ):

        # Real = 0
        # Fake = 1

        if "real" in cls_dir.name.lower():

            label = 0

        else:

            label = 1


        files = [
            f
            for f in cls_dir.rglob("*")
            if f.suffix.lower() in EXTS
        ]


        paths += files

        labels += [
            label
        ] * len(files)


    # Không tìm thấy ảnh
    if not paths:

        raise FileNotFoundError(
            f"Không tìm thấy ảnh trong: "
            f"{split_dir}"
        )


    # --------------------------------------------------------
    # Trích xuất đặc trưng song song
    # --------------------------------------------------------

    feats = Parallel(
        n_jobs=-1
    )(
        delayed(_safe_feat)(p)
        for p in tqdm(
            paths,
            desc=f"Trích xuất {split_dir.name}"
        )
    )


    # --------------------------------------------------------
    # Bỏ ảnh lỗi
    # --------------------------------------------------------

    keep = [
        i
        for i, f in enumerate(feats)
        if f is not None
    ]


    print(
        f"{split_dir.name}: "
        f"bỏ qua "
        f"{len(paths) - len(keep)} "
        f"ảnh lỗi/không mở được"
    )


    # --------------------------------------------------------
    # Tạo X và y
    # --------------------------------------------------------

    X = np.stack([
        feats[i]
        for i in keep
    ])


    y = np.array([
        labels[i]
        for i in keep
    ])


    # --------------------------------------------------------
    # Lưu cache
    # --------------------------------------------------------

    np.savez_compressed(
        cache,
        X=X,
        y=y
    )


    return X, y


# ============================================================
# 7. ĐÁNH GIÁ MODEL
# ============================================================

def evaluate(
    model,
    X,
    y,
    name,
    model_name
):

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    pred = model.predict(
        X
    )


    # --------------------------------------------------------
    # Probability AI
    # --------------------------------------------------------

    proba = model.predict_proba(
        X
    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        "accuracy":
            accuracy_score(
                y,
                pred
            ),

        "precision":
            precision_score(
                y,
                pred
            ),

        "recall":
            recall_score(
                y,
                pred
            ),

        "f1":
            f1_score(
                y,
                pred
            ),

        "roc_auc":
            roc_auc_score(
                y,
                proba
            )
    }


    print()
    print("=" * 70)

    print(
        f"{model_name.upper()} - "
        f"{name.upper()}"
    )

    print("=" * 70)


    print(
        {
            k: round(v, 4)
            for k, v in metrics.items()
        }
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()

    print(
        classification_report(
            y,
            pred,
            target_names=CLASS_NAMES,
            digits=4
        )
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        pred
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )


    disp.plot(
        values_format="d"
    )


    plt.title(
        f"{model_name} - "
        f"Confusion Matrix ({name})"
    )


    plt.tight_layout()


    filename = (
        "confusion_matrix_"
        f"{model_name.lower().replace(' ', '_')}_"
        f"{name.lower()}.png"
    )


    plt.savefig(
        OUT_DIR / filename,
        dpi=150
    )


    plt.show()


    # --------------------------------------------------------
    # Thêm confusion matrix vào kết quả
    # --------------------------------------------------------

    metrics[
        "confusion_matrix"
    ] = cm.tolist()


    return metrics


# ============================================================
# 8. CHƯƠNG TRÌNH CHÍNH
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # KIỂM TRA DATASET
    # ========================================================

    print("=" * 70)

    print(
        "KIỂM TRA DATASET"
    )

    print("=" * 70)


    print(
        "Dataset:",
        DATA_DIR
    )


    # --------------------------------------------------------
    # Kiểm tra các thư mục
    # --------------------------------------------------------

    required_dirs = [

        DATA_DIR / "train",

        DATA_DIR / "validation",

        DATA_DIR / "test"
    ]


    for d in required_dirs:

        if not d.exists():

            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {d}"
            )


    print(
        "Dataset hợp lệ!"
    )


    # ========================================================
    # TIÊU ĐỀ PHẦN C
    # ========================================================

    print()

    print("=" * 70)

    print(
        "PHẦN C - SVM + LOGISTIC REGRESSION"
    )

    print("=" * 70)


    # ========================================================
    # 9. LOAD TRAIN
    # ========================================================

    print()

    print(
        "[1] LOAD TRAIN"
    )


    X_train, y_train = load_split(
        DATA_DIR / "train"
    )


    # ========================================================
    # 10. LOAD VALIDATION
    # ========================================================

    print()

    print(
        "[2] LOAD VALIDATION"
    )


    X_val, y_val = load_split(
        DATA_DIR / "validation"
    )


    # ========================================================
    # 11. LOAD TEST
    # ========================================================

    print()

    print(
        "[3] LOAD TEST"
    )


    X_test, y_test = load_split(
        DATA_DIR / "test"
    )


    # ========================================================
    # 12. THÔNG TIN DATASET
    # ========================================================

    print()

    print("=" * 70)

    print(
        "THÔNG TIN DỮ LIỆU"
    )

    print("=" * 70)


    print(
        "Train:",
        X_train.shape
    )


    print(
        "Validation:",
        X_val.shape
    )


    print(
        "Test:",
        X_test.shape
    )


    print()

    print(
        "Train Real/AI:",
        np.bincount(y_train)
    )


    print(
        "Validation Real/AI:",
        np.bincount(y_val)
    )


    print(
        "Test Real/AI:",
        np.bincount(y_test)
    )


    print()

    print(
        "Tổng số đặc trưng:",
        X_train.shape[1]
    )


    print(
        "Các nhóm đặc trưng:",
        feature_groups()
    )


    # ========================================================
    # 13. TRAINING SVM
    # ========================================================

    print()

    print("=" * 70)

    print(
        "TRAINING SVM"
    )

    print("=" * 70)


    start = time.time()


    # --------------------------------------------------------
    # SVM + Calibration
    #
    # Không dùng:
    # SVC(probability=True)
    #
    # Vì sklearn mới đã deprecated probability=True.
    #
    # CalibratedClassifierCV giúp SVM có predict_proba().
    # --------------------------------------------------------

    svm_model = Pipeline([

        (
            "scaler",

            StandardScaler()
        ),

        (
            "svm",

            CalibratedClassifierCV(

                estimator=SVC(

                    kernel="rbf",

                    C=10,

                    gamma="scale",

                    random_state=SEED
                ),

                method="sigmoid",

                cv=3,

                ensemble=False
            )
        )
    ])


    print(
        "Đang huấn luyện SVM..."
    )


    svm_model.fit(
        X_train,
        y_train
    )


    print()

    print(
        "SVM hoàn thành!"
    )


    print(
        "Thời gian:",
        round(
            time.time() - start,
            1
        ),
        "giây"
    )


    # ========================================================
    # 14. ĐÁNH GIÁ SVM
    # ========================================================

    svm_val = evaluate(

        svm_model,

        X_val,

        y_val,

        "validation",

        "SVM"
    )


    svm_test = evaluate(

        svm_model,

        X_test,

        y_test,

        "test",

        "SVM"
    )


    # ========================================================
    # 15. TRAINING LOGISTIC REGRESSION
    # ========================================================

    print()

    print("=" * 70)

    print(
        "TRAINING LOGISTIC REGRESSION"
    )

    print("=" * 70)


    start = time.time()


    lr_model = Pipeline([

        (
            "scaler",

            StandardScaler()
        ),

        (
            "logistic_regression",

            LogisticRegression(

                C=1.0,

                max_iter=2000,

                random_state=SEED
            )
        )
    ])


    print(
        "Đang huấn luyện Logistic Regression..."
    )


    lr_model.fit(
        X_train,
        y_train
    )


    print()

    print(
        "Logistic Regression hoàn thành!"
    )


    print(
        "Thời gian:",
        round(
            time.time() - start,
            1
        ),
        "giây"
    )


    # ========================================================
    # 16. ĐÁNH GIÁ LOGISTIC REGRESSION
    # ========================================================

    lr_val = evaluate(

        lr_model,

        X_val,

        y_val,

        "validation",

        "Logistic Regression"
    )


    lr_test = evaluate(

        lr_model,

        X_test,

        y_test,

        "test",

        "Logistic Regression"
    )


    # ========================================================
    # 17. LƯU MODEL SVM
    # ========================================================

    print()

    print("=" * 70)

    print(
        "LƯU MODEL"
    )

    print("=" * 70)


    joblib.dump(

        {
            "model":
                svm_model,

            "img_size":
                IMG_SIZE,

            "classes":
                CLASS_NAMES,

            "feature_groups":
                feature_groups()
        },

        OUT_DIR / "svm.joblib",

        compress=3
    )


    print(
        "Đã lưu:",
        OUT_DIR / "svm.joblib"
    )


    # ========================================================
    # 18. LƯU MODEL LOGISTIC REGRESSION
    # ========================================================

    joblib.dump(

        {
            "model":
                lr_model,

            "img_size":
                IMG_SIZE,

            "classes":
                CLASS_NAMES,

            "feature_groups":
                feature_groups()
        },

        OUT_DIR / "logistic_regression.joblib",

        compress=3
    )


    print(
        "Đã lưu:",
        OUT_DIR / "logistic_regression.joblib"
    )


    # ========================================================
    # 19. LƯU KẾT QUẢ JSON
    # ========================================================

    results = {

        "dataset":
            str(DATA_DIR),

        "feature_groups":
            feature_groups(),

        "feature_count":
            int(X_train.shape[1]),

        "class_names":
            CLASS_NAMES,

        "label_mapping": {

            "Real": 0,

            "AI-Generated": 1
        },


        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        "SVM": {

            "parameters": {

                "kernel":
                    "rbf",

                "C":
                    10,

                "gamma":
                    "scale",

                "calibration":
                    "sigmoid",

                "cv":
                    3
            },

            "validation":
                svm_val,

            "test":
                svm_test
        },


        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        "Logistic_Regression": {

            "parameters": {

                "C":
                    1.0,

                "max_iter":
                    2000
            },

            "validation":
                lr_val,

            "test":
                lr_test
        }
    }


    with open(

        OUT_DIR
        / "svm_logreg_results.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            results,

            f,

            indent=2,

            ensure_ascii=False
        )


    print()

    print(
        "Đã lưu:",
        OUT_DIR / "svm_logreg_results.json"
    )


    # ========================================================
    # 20. TỔNG KẾT
    # ========================================================

    print()

    print("=" * 70)

    print(
        "KẾT QUẢ CUỐI CÙNG - PHẦN C"
    )

    print("=" * 70)


    print()

    print(
        "SVM:"
    )


    print(
        f"Accuracy  = "
        f"{svm_test['accuracy']:.4f}"
    )

    print(
        f"Precision = "
        f"{svm_test['precision']:.4f}"
    )

    print(
        f"Recall    = "
        f"{svm_test['recall']:.4f}"
    )

    print(
        f"F1        = "
        f"{svm_test['f1']:.4f}"
    )

    print(
        f"ROC-AUC   = "
        f"{svm_test['roc_auc']:.4f}"
    )


    print()

    print(
        "Logistic Regression:"
    )


    print(
        f"Accuracy  = "
        f"{lr_test['accuracy']:.4f}"
    )

    print(
        f"Precision = "
        f"{lr_test['precision']:.4f}"
    )

    print(
        f"Recall    = "
        f"{lr_test['recall']:.4f}"
    )

    print(
        f"F1        = "
        f"{lr_test['f1']:.4f}"
    )

    print(
        f"ROC-AUC   = "
        f"{lr_test['roc_auc']:.4f}"
    )


    # ========================================================
    # 21. FILE OUTPUT
    # ========================================================

    print()

    print("=" * 70)

    print(
        "CÁC FILE ĐÃ TẠO"
    )

    print("=" * 70)


    print(
        "1.",
        OUT_DIR / "svm.joblib"
    )


    print(
        "2.",
        OUT_DIR / "logistic_regression.joblib"
    )


    print(
        "3.",
        OUT_DIR / "svm_logreg_results.json"
    )


    print(
        "4.",
        OUT_DIR / "confusion_matrix_svm_validation.png"
    )


    print(
        "5.",
        OUT_DIR / "confusion_matrix_svm_test.png"
    )


    print(
        "6.",
        OUT_DIR
        / "confusion_matrix_logistic_regression_validation.png"
    )


    print(
        "7.",
        OUT_DIR
        / "confusion_matrix_logistic_regression_test.png"
    )


    print()

    print(
        "HOÀN THÀNH PHẦN C!"
    )