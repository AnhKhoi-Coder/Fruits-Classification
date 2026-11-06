
from pathlib import Path
from collections import defaultdict
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "dataset" / "fruits-360-100x100"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_RATIO = 0.2
RANDOM_SEED = 42


# Chuẩn hóa theo thống kê ImageNet.
# Quan trọng: cả CNN Baseline và FasterNet phải dùng cùng pipeline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# 2. TIỀN XỬ LÝ ẢNH
# ============================================================

def get_transforms(image_size=IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return train_transform, eval_transform


# ============================================================
# 3. CHIA TRAIN / VALIDATION THEO TỪNG LỚP
# ============================================================

def stratified_split(targets, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """
    Chia chỉ số dữ liệu theo từng lớp để mỗi lớp đều có
    mẫu train và validation nếu lớp có ít nhất 2 ảnh.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio phải nằm giữa 0 và 1.")

    class_indices = defaultdict(list)

    for index, label in enumerate(targets):
        class_indices[label].append(index)

    rng = random.Random(seed)

    train_indices = []
    val_indices = []

    for label in sorted(class_indices):
        indices = class_indices[label].copy()
        rng.shuffle(indices)

        # Với lớp chỉ có một ảnh, giữ ảnh đó trong train.
        if len(indices) < 2:
            train_indices.extend(indices)
            continue

        val_count = max(1, round(len(indices) * val_ratio))
        val_count = min(val_count, len(indices) - 1)

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    # Xáo trộn thứ tự các mẫu sau khi chia.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


# ============================================================
# 4. XÂY DỰNG DATALOADER
# ============================================================

def build_dataloaders(
    data_root=DEFAULT_DATA_ROOT,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    val_ratio=VAL_RATIO,
    seed=RANDOM_SEED,
    num_workers=0,
):
    data_root = Path(data_root)

    train_dir = data_root / "Training"
    test_dir = data_root / "Test"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục Training: {train_dir}"
        )

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục Test: {test_dir}"
        )

    train_transform, eval_transform = get_transforms(image_size)

    # Hai dataset Training có cùng thứ tự mẫu:
    # một bản dùng augmentation, một bản không augmentation.
    train_aug_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_transform,
    )

    train_eval_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=eval_transform,
    )

    # Kiểm tra ánh xạ lớp giữa các tập.
    if train_aug_dataset.class_to_idx != train_eval_dataset.class_to_idx:
        raise ValueError(
            "Ánh xạ lớp giữa hai bản Training không trùng khớp."
        )

    if train_aug_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "Ánh xạ lớp Training và Test không trùng khớp."
        )

    class_names = train_aug_dataset.classes

    train_indices, val_indices = stratified_split(
        targets=train_aug_dataset.targets,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_dataset = Subset(
        train_aug_dataset,
        train_indices,
    )

    val_dataset = Subset(
        train_eval_dataset,
        val_indices,
    )

    # Cố định seed cho thứ tự shuffle của tập train.
    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names
