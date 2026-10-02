"""Trích đặc trưng cho các mô hình cổ điển (RF, SVM, LR).
Copy y nguyên hàm extract_features trong RandomForests.py của bạn D
để vector đặc trưng khớp 100% với lúc train."""
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = 224
FFT_BINS = 32


def extract_features_rf(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    f_hog = hog(gray, orientations=9, pixels_per_cell=(32, 32),
                cells_per_block=(2, 2), block_norm="L2-Hys")
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    f_lbp = np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size
    hsv = np.asarray(img.convert("HSV"))
    f_col = np.concatenate([
        np.histogram(hsv[..., c], bins=16, range=(0, 256), density=True)[0]
        for c in range(3)])
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray / 255.0))))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - IMG_SIZE // 2, xx - IMG_SIZE // 2)
    rbin = np.minimum((r / (IMG_SIZE / 2) * FFT_BINS).astype(int), FFT_BINS - 1)
    f_fft = (np.bincount(rbin.ravel(), mag.ravel(), FFT_BINS) /
             np.maximum(np.bincount(rbin.ravel(), minlength=FFT_BINS), 1))
    return np.concatenate([f_hog, f_lbp, f_col, f_fft]).astype(np.float32)
