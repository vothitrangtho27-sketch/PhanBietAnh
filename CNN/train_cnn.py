"""
Quy ước nhãn: fake = 0 (ảnh AI tạo), real = 1 (ảnh thật)
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras import layers

# index 0 = fake, index 1 = real
CLASS_NAMES = ["fake", "real"]

def parse_args():
    p = argparse.ArgumentParser(description="Huấn luyện CNN phân loại ảnh thật / ảnh AI")
    p.add_argument("--data_dir", type=str, default="A_TienXuLyAnh/Dataset_Final",
                   help="Thư mục chứa train/ validation/ test/")
    p.add_argument("--output_dir", type=str, default="cnn_output",
                   help="Nơi lưu mô hình, biểu đồ, số liệu")
    p.add_argument("--img_size", type=int, default=224, help="Kích thước ảnh vuông (ảnh của bạn A đã là 224)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30, help="Số epoch tối đa (EarlyStopping có thể dừng sớm hơn)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate ban đầu")
    p.add_argument("--patience", type=int, default=6, help="Số epoch chờ trước khi EarlyStopping dừng")
    p.add_argument("--base_filters", type=int, default=32,
                   help="Số filter của khối đầu (32 -> 32/64/128/256). Không có GPU: dùng 16 để chạy nhanh ~4 lần")
    p.add_argument("--no_augment", action="store_true", help="Tắt tăng cường dữ liệu (data augmentation)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_split_dir(data_dir: Path, names):
    """Tìm thư mục theo nhiều tên có thể (vd: validation hoặc val)."""
    for n in names:
        if (data_dir / n).is_dir():
            return data_dir / n
    raise FileNotFoundError(f"Không tìm thấy thư mục {names} trong {data_dir}")


def load_split(split_dir: Path, img_size, batch_size, shuffle, seed):
    """
    Đọc 1 tập (train/val/test) từ thư mục. Keras tự gán nhãn theo tên thư mục con.
    Trả về (dataset, danh_sách_đường_dẫn_file).
    Lưu ý: ảnh KHÔNG chia cho 255 ở đây - việc chuẩn hóa nằm trong mô hình
    (lớp Rescaling) để lúc dự đoán trong Streamlit chỉ cần đưa ảnh thô vào.
    """
    ds = keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="binary",          # nhãn 0/1 -> dùng với sigmoid
        class_names=CLASS_NAMES,      # ép thứ tự: fake=0, real=1
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    file_paths = list(ds.file_paths)  
    ds = ds.prefetch(tf.data.AUTOTUNE)  
    return ds, file_paths


def conv_block(x, filters):
    """
    1 khối tích chập: [Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool -> Dropout nhẹ
    - Conv: trích xuất đặc trưng (cạnh, kết cấu, vân ảnh...)
    - BatchNorm: ổn định + tăng tốc huấn luyện
    - MaxPool: giảm kích thước ảnh 1/2, giữ đặc trưng nổi bật
    - Dropout: giảm overfitting
    """
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)
    return x


def build_cnn(img_size=224, augment=True, base_filters=32):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs

    if augment:
        x = layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = layers.RandomRotation(0.03, name="aug_rotate")(x)
        x = layers.RandomContrast(0.1, name="aug_contrast")(x)

    x = layers.Rescaling(1.0 / 255, name="rescale")(x)  

    for mult in (1, 2, 4, 8):
        x = conv_block(x, base_filters * mult)

    x = layers.GlobalAveragePooling2D(name="gap")(x)     
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_real")(x)  

    return keras.Model(inputs, outputs, name="cnn_real_vs_ai")


def train(model, train_ds, val_ds, args, out_dir: Path):
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(out_dir / "cnn_model.keras", monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(out_dir / "history.csv"),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, shuffle=False, verbose=1)  
    return history

def plot_history(history, out_dir: Path):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, h["loss"], label="Train")
    ax[0].plot(epochs, h["val_loss"], label="Validation")
    ax[0].set(title="Loss theo epoch", xlabel="Epoch", ylabel="Loss")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(epochs, h["accuracy"], label="Train")
    ax[1].plot(epochs, h["val_accuracy"], label="Validation")
    ax[1].set(title="Accuracy theo epoch", xlabel="Epoch", ylabel="Accuracy")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Fake (AI)", "Real (thật)"], yticklabels=["Fake (AI)", "Real (thật)"],
           xlabel="Dự đoán", ylabel="Thực tế", title="Ma trận nhầm lẫn (tập Test)")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc(y_true, prob_real, auc, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, prob_real)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"CNN (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ngẫu nhiên")
    ax.set(title="Đường cong ROC (tập Test)", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def plot_errors(file_paths, y_true, prob_real, out_dir: Path, max_show=12):
    """Lưới các ảnh bị dự đoán sai - hữu ích để phân tích lỗi trong báo cáo."""
    pred = (prob_real >= 0.5).astype(int)
    wrong = np.where(pred != y_true)[0]
    if len(wrong) == 0:
        return
    # ưu tiên những ảnh mô hình "tự tin nhưng sai" nhất
    conf = np.abs(prob_real[wrong] - 0.5)
    wrong = wrong[np.argsort(-conf)][:max_show]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        ax.imshow(Image.open(file_paths[idx]).convert("RGB"))
        ax.set_title(f"Thực tế: {CLASS_NAMES[int(y_true[idx])]}\n"
                     f"Dự đoán: {CLASS_NAMES[int(pred[idx])]} (P_real={prob_real[idx]:.2f})",
                     fontsize=8)
    fig.suptitle("Các ảnh bị dự đoán sai (tự tin nhất)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "misclassified_examples.png", dpi=130)
    plt.close(fig)


def evaluate(model, test_ds, test_paths, out_dir: Path):
    y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    prob_real = model.predict(test_ds, verbose=1).ravel()
    y_pred = (prob_real >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, prob_real)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, digits=4, zero_division=0)
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán) [fake, real]:\n", cm)
    print(f"ROC-AUC: {auc:.4f}")

    plot_confusion(cm, out_dir)
    plot_roc(y_true, prob_real, auc, out_dir)
    plot_errors(test_paths, y_true, prob_real, out_dir)

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "roc_auc": float(auc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASS_NAMES},
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": cm.tolist()},
    }

def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed) 

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPU: {[g.name for g in gpus] if gpus else 'KHÔNG CÓ (chạy CPU - sẽ chậm)'}")

    print("\n[1/4] Nạp dữ liệu...")
    train_ds, _ = load_split(find_split_dir(data_dir, ["train"]), args.img_size, args.batch_size, True, args.seed)
    val_ds, _ = load_split(find_split_dir(data_dir, ["validation", "val"]), args.img_size, args.batch_size, False, args.seed)
    test_ds, test_paths = load_split(find_split_dir(data_dir, ["test"]), args.img_size, args.batch_size, False, args.seed)

    print("\n[2/4] Xây dựng mô hình CNN...")
    model = build_cnn(args.img_size, augment=not args.no_augment, base_filters=args.base_filters)
    model.summary()
    with open(out_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    print("\n[3/4] Huấn luyện...")
    t0 = time.time()
    history = train(model, train_ds, val_ds, args, out_dir)
    train_minutes = (time.time() - t0) / 60
    plot_history(history, out_dir)

    print("\n[4/4] Đánh giá trên tập Test...")
    best_model = keras.models.load_model(out_dir / "cnn_model.keras")
    results = evaluate(best_model, test_ds, test_paths, out_dir)
    results.update({
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "train_minutes": round(train_minutes, 1),
        "params": int(best_model.count_params()),
        "config": vars(args),
        "label_mapping": {"fake": 0, "real": 1, "note": "đầu ra sigmoid = xác suất ảnh THẬT"},
    })
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== HOÀN TẤT ===  Accuracy test = {results['accuracy']:.4f} | AUC = {results['roc_auc']:.4f}")
    print(f"Mô hình + biểu đồ + số liệu đã lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
