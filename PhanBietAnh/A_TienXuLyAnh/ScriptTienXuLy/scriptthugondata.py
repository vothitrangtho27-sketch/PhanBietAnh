import argparse
import random
import shutil
from pathlib import Path
 
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def get_image_files(folder: Path):
    """Lấy danh sách tất cả file ảnh hợp lệ trong 1 thư mục."""
    if not folder.exists():
        return []
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ]
 
 
def sample_and_copy(src_folder: Path, dst_folder: Path, n_target: int, seed: int, tag: str):
    """
    Lấy ngẫu nhiên n_target ảnh trong src_folder, copy sang dst_folder.
    """
    images = get_image_files(src_folder)
    total = len(images)
 
    if total == 0:
        print(f"  [!] Không tìm thấy ảnh nào trong: {src_folder}")
        return 0
 
    if n_target > total:
        print(f"  [!] Cảnh báo: {tag} yêu cầu {n_target} ảnh nhưng thư mục gốc "
              f"chỉ có {total} ảnh -> lấy tối đa {total} ảnh.")
        n_target = total
 
    rng = random.Random(seed)
    sampled = rng.sample(images, n_target)
 
    dst_folder.mkdir(parents=True, exist_ok=True)
    for img_path in sampled:
        shutil.copy2(img_path, dst_folder / img_path.name)
 
    print(f"  - {tag}: {total} ảnh gốc -> lấy {n_target} ảnh")
    return n_target
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Lọc ngẫu nhiên số lượng ảnh cụ thể theo cấu trúc train/test x real/fake."
    )
    parser.add_argument("--src", required=True,
                         help="Đường dẫn thư mục dataset gốc (chứa train/ và test/)")
    parser.add_argument("--dst", required=True,
                         help="Đường dẫn thư mục sẽ chứa dataset đã thu gọn")
 
    # Số lượng ảnh muốn lấy cho từng nhóm (mặc định theo yêu cầu)
    parser.add_argument("--train_real", type=int, default=8000)
    parser.add_argument("--train_fake", type=int, default=8000)
    parser.add_argument("--test_real", type=int, default=2000)
    parser.add_argument("--test_fake", type=int, default=2000)
 
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed cho random, để kết quả lặp lại được")
    args = parser.parse_args()
 
    src_root = Path(args.src)
    dst_root = Path(args.dst)
 
    if not src_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset gốc: {src_root}")
 
    # Danh sách (split, class, số lượng muốn lấy)
    plan = [
        ("train", "real", args.train_real),
        ("train", "fake", args.train_fake),
        ("test", "real", args.test_real),
        ("test", "fake", args.test_fake),
    ]
 
    print(f"Seed: {args.seed}\n")
 
    total_sampled = 0
    for split, cls, n_target in plan:
        src_folder = src_root / split / cls
        dst_folder = dst_root / split / cls
        tag = f"{split}/{cls}"
        total_sampled += sample_and_copy(src_folder, dst_folder, n_target, args.seed, tag)
 
    print(f"\nHoàn tất! Tổng số ảnh đã lấy: {total_sampled}")
    print(f"Dataset thu gọn được lưu tại: {dst_root.resolve()}")
 
    # In tóm tắt số lượng theo train/test để tiện kiểm tra
    n_train = args.train_real + args.train_fake
    n_test = args.test_real + args.test_fake
    print(f"\nTóm tắt:")
    print(f"  Train: {args.train_fake} Fake + {args.train_real} Real = {n_train}")
    print(f"  Test : {args.test_fake} Fake + {args.test_real} Real = {n_test}")
    print(f"  Tổng cộng: {n_train + n_test} ảnh")
 
 
if __name__ == "__main__":
    main()