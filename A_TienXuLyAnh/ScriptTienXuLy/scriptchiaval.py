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
 
 
def split_class_folder(
    train_class_folder: Path,
    val_class_folder: Path,
    ratio: float,
    seed: int,
    use_copy: bool,
    tag: str,
):
    """
    Từ train_class_folder, lấy ngẫu nhiên `ratio` số ảnh, chuyển (hoặc copy)
    sang val_class_folder.
    """
    images = get_image_files(train_class_folder)
    total = len(images)
 
    if total == 0:
        print(f"  [!] Không tìm thấy ảnh nào trong: {train_class_folder}")
        return 0, 0
 
    n_val = round(total * ratio)
 
    rng = random.Random(seed)
    val_images = rng.sample(images, n_val)
 
    val_class_folder.mkdir(parents=True, exist_ok=True)
 
    for img_path in val_images:
        dst_path = val_class_folder / img_path.name
        if use_copy:
            shutil.copy2(img_path, dst_path)
        else:
            shutil.move(str(img_path), str(dst_path))
 
    n_train_left = total - n_val
    action = "copy" if use_copy else "move"
    print(f"  - {tag}: {total} ảnh gốc -> {action} {n_val} ảnh sang validation "
          f"(còn lại {n_train_left} ảnh trong train)")
    return n_val, n_train_left
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Tách 1 phần ảnh từ train sang validation, giữ tỉ lệ giữa các lớp."
    )
    parser.add_argument("--src", required=True,
                         help="Thư mục gốc dataset (chứa thư mục train/ bên trong)")
    parser.add_argument("--ratio", type=float, default=0.15,
                         help="Tỉ lệ ảnh tách sang validation (mặc định 0.15 = 15%%)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed cho random, để kết quả lặp lại được")
    parser.add_argument("--copy", action="store_true",
                         help="Copy thay vì move (giữ nguyên ảnh trong train)")
    args = parser.parse_args()
 
    src_root = Path(args.src)
    train_root = src_root / "train"
    val_root = src_root / "validation"
 
    if not train_root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục train: {train_root}\n"
            "Script này giả định cấu trúc: <src>/train/real, <src>/train/fake"
        )
 
    # Tự động phát hiện các lớp (thư mục con) trong train/
    class_folders = [f for f in train_root.iterdir() if f.is_dir()]
 
    if not class_folders:
        raise RuntimeError(f"Không tìm thấy thư mục lớp nào bên trong: {train_root}")
 
    print(f"Tìm thấy {len(class_folders)} lớp trong train: "
          f"{[f.name for f in class_folders]}")
    print(f"Tỉ lệ tách sang validation: {args.ratio*100:.0f}%")
    print(f"Chế độ: {'COPY (giữ nguyên train)' if args.copy else 'MOVE (train sẽ giảm)'}")
    print(f"Seed: {args.seed}\n")
 
    total_val = 0
    total_train_left = 0
 
    for class_folder in class_folders:
        val_class_folder = val_root / class_folder.name
        n_val, n_train_left = split_class_folder(
            class_folder, val_class_folder, args.ratio, args.seed,
            args.copy, class_folder.name
        )
        total_val += n_val
        total_train_left += n_train_left
 
    print(f"\nHoàn tất!")
    print(f"  Validation: {total_val} ảnh -> lưu tại {val_root.resolve()}")
    print(f"  Train còn lại: {total_train_left} ảnh -> tại {train_root.resolve()}")
 
 
if __name__ == "__main__":
    main()