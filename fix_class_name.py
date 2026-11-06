
from pathlib import Path

ROOT = Path("dataset/fruits-360-100x100")

old_dir = ROOT / "Test" / "Blackberry 4"
new_dir = ROOT / "Test" / "BlackBerry 4"

if not old_dir.exists():
    print(f"[ERROR] Không tìm thấy thư mục: {old_dir}")
elif new_dir.exists():
    print(f"[ERROR] Thư mục đích đã tồn tại: {new_dir}")
else:
    old_dir.rename(new_dir)
    print(f"[OK] Đã đổi tên:\n{old_dir}\n-> {new_dir}")
