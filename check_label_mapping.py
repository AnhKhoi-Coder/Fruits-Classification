
from pathlib import Path
from torchvision.datasets import ImageFolder

ROOT = Path("dataset/fruits-360-100x100")

train = ImageFolder(str(ROOT / "Training"))
test = ImageFolder(str(ROOT / "Test"))

print("Số lớp Training:", len(train.classes))
print("Số lớp Test:", len(test.classes))

train_set = set(train.classes)
test_set = set(test.classes)

print("\nChỉ có trong Training:")
print(sorted(train_set - test_set))

print("\nChỉ có trong Test:")
print(sorted(test_set - train_set))

print("\nCác vị trí có thứ tự khác nhau:")
count = 0

for i, (train_name, test_name) in enumerate(
    zip(train.classes, test.classes)
):
    if train_name != test_name:
        print(
            f"Index {i}: "
            f"Training='{train_name}' | Test='{test_name}'"
        )
        count += 1

        if count >= 20:
            print("Dừng hiển thị sau 20 vị trí.")
            break

if count == 0 and train.classes == test.classes:
    print("OK: Danh sách lớp và thứ tự nhãn trùng khớp.")
elif train_set == test_set:
    print(
        "\nCùng tập tên lớp nhưng khác thứ tự. "
        "Cần dùng chung class_to_idx."
    )
else:
    print(
        "\nHai tập có tên lớp khác nhau. "
        "Cần xác minh các lớp bị thiếu hoặc thừa."
    )
