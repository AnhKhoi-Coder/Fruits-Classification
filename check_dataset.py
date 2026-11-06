
from pathlib import Path
from collections import Counter

from PIL import Image
from torchvision.datasets import ImageFolder

DATASET_DIR = Path("dataset/fruits-360-100x100")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_split(split_name):
    split_dir = DATASET_DIR / split_name

    print(f"\n{'=' * 55}")
    print(f"DATA SPLIT: {split_name}")
    print(f"Path: {split_dir.resolve()}")

    if not split_dir.is_dir():
        print("[ERROR] Không tìm thấy thư mục:", split_dir)
        return None

    dataset = ImageFolder(str(split_dir))

    print("Number of images:", len(dataset))
    print("Number of classes:", len(dataset.classes))
    print("First 10 classes:", dataset.classes[:10])

    # Đếm số ảnh theo từng lớp
    counts = Counter(label for _, label in dataset.samples)

    print("\nSố ảnh của 10 lớp đầu:")
    for class_name in dataset.classes[:10]:
        label = dataset.class_to_idx[class_name]
        print(f"  {class_name}: {counts[label]} ảnh")

    # Đọc thử một ảnh
    image_path, label = dataset.samples[0]

    with Image.open(image_path) as image:
        image.verify()

    print("\nSample image:", image_path)
    print("Sample label:", label)
    print("Sample class:", dataset.classes[label])
    print("[OK] ImageFolder đọc được dữ liệu.")

    return dataset


def main():
    print("Dataset root:", DATASET_DIR.resolve())

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Không tìm thấy dataset. Hãy kiểm tra DATASET_DIR."
        )

    train = inspect_split("Training")
    test = inspect_split("Test")

    if train is None or test is None:
        return

    print("\n" + "=" * 55)
    print("SO SÁNH NHÃN TRAIN VÀ TEST")

    if train.classes == test.classes:
        print("[OK] Danh sách lớp và thứ tự nhãn trùng nhau.")
    else:
        print("[ERROR] Danh sách lớp hoặc thứ tự nhãn không trùng.")
        print("Train:", train.classes)
        print("Test :", test.classes)

    print("\nHOÀN THÀNH KIỂM TRA DATASET.")


if __name__ == "__main__":
    main()