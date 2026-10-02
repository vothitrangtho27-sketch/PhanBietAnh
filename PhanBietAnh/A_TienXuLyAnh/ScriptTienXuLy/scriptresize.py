import argparse
from pathlib import Path
from PIL import Image
 
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".tif"}
 
 
def find_all_images(root: Path):
    """Duyệt đệ quy toàn bộ thư mục con, tìm tất cả file ảnh hợp lệ."""
    return [
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ]
 
 
def resize_and_convert(
    img_path: Path,
    src_root: Path,
    dst_root: Path,
    size: tuple,
    quality: int,
    keep_aspect: bool,
):
    """
    Mở 1 ảnh, resize, chuyển sang RGB, lưu ra dst_root (giữ nguyên
    cấu trúc thư mục con so với src_root), đổi đuôi thành .jpg.
    """
    try:
        # Xác định đường dẫn tương ứng bên thư mục đích (giữ cấu trúc con)
        rel_path = img_path.relative_to(src_root)
        dst_path = (dst_root / rel_path).with_suffix(".jpg")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
 
        with Image.open(img_path) as img:
            # Chuyển về RGB để đảm bảo lưu .jpg không lỗi
            # (ảnh PNG có kênh alpha, ảnh grayscale "L", ảnh CMYK, v.v.)
            if img.mode != "RGB":
                img = img.convert("RGB")
 
            if keep_aspect:
                # Giữ tỉ lệ ảnh gốc, resize sao cho vừa khít trong size,
                # phần dư sẽ được đệm (pad) màu đen cho đủ kích thước.
                img.thumbnail(size, Image.LANCZOS)
                canvas = Image.new("RGB", size, (0, 0, 0))
                offset = (
                    (size[0] - img.width) // 2,
                    (size[1] - img.height) // 2,
                )
                canvas.paste(img, offset)
                img = canvas
            else:
                # Resize thẳng về đúng kích thước (có thể làm méo ảnh
                # nếu tỉ lệ khung hình gốc khác tỉ lệ đích).
                img = img.resize(size, Image.LANCZOS)
 
            img.save(dst_path, "JPEG", quality=quality)
        return True, None
    except Exception as e:
        return False, str(e)
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Resize và chuyển toàn bộ ảnh trong dataset thành .jpg."
    )
    parser.add_argument("--src", required=True, help="Thư mục dataset gốc")
    parser.add_argument("--dst", required=True, help="Thư mục lưu dataset sau khi xử lý")
    parser.add_argument("--size", type=int, default=None,
                         help="Kích thước vuông (ví dụ --size 224 => 224x224)")
    parser.add_argument("--width", type=int, default=224,
                         help="Chiều rộng đích (dùng khi không truyền --size)")
    parser.add_argument("--height", type=int, default=224,
                         help="Chiều cao đích (dùng khi không truyền --size)")
    parser.add_argument("--quality", type=int, default=95,
                         help="Chất lượng nén JPEG (1-100, mặc định 95)")
    parser.add_argument("--keep_aspect", action="store_true",
                         help="Giữ tỉ lệ khung hình gốc, đệm viền đen thay vì làm méo ảnh")
    args = parser.parse_args()
 
    src_root = Path(args.src)
    dst_root = Path(args.dst)
 
    if not src_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset gốc: {src_root}")
 
    if args.size is not None:
        target_size = (args.size, args.size)
    else:
        target_size = (args.width, args.height)
 
    print(f"Kích thước đích: {target_size[0]}x{target_size[1]}")
    print(f"Giữ tỉ lệ khung hình gốc (pad viền đen): {'Có' if args.keep_aspect else 'Không'}")
    print(f"Chất lượng JPEG: {args.quality}\n")
 
    images = find_all_images(src_root)
    total = len(images)
    print(f"Tìm thấy {total} ảnh cần xử lý...\n")
 
    success_count = 0
    fail_list = []
 
    for i, img_path in enumerate(images, 1):
        ok, err = resize_and_convert(
            img_path, src_root, dst_root, target_size, args.quality, args.keep_aspect
        )
        if ok:
            success_count += 1
        else:
            fail_list.append((img_path, err))
 
        # In tiến trình mỗi 500 ảnh để không bị spam màn hình
        if i % 500 == 0 or i == total:
            print(f"  Đã xử lý {i}/{total} ảnh...")
 
    print(f"\nHoàn tất! Thành công: {success_count}/{total}")
 
    if fail_list:
        print(f"  Có {len(fail_list)} ảnh bị lỗi:")
        for path, err in fail_list[:10]:
            print(f"   - {path}: {err}")
        if len(fail_list) > 10:
            print(f"   ... và {len(fail_list) - 10} lỗi khác.")
 
    print(f"\nDataset đã xử lý được lưu tại: {dst_root.resolve()}")
 
 
if __name__ == "__main__":
    main()