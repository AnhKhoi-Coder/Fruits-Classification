
from pathlib import Path
import sys
import csv
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_baseline import CNNBaseline
from utils.dataset import build_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp=False,
    epoch=1,
    total_epochs=1,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

            if is_train and (step % 100 == 0 or step == len(loader)):
                print(
                    f"\rEpoch {epoch}/{total_epochs} "
                    f"| Batch {step}/{len(loader)} "
                    f"| Loss hiện tại: {loss.item():.4f}",
                    end="",
                    flush=True,
                )

    if is_train:
        print()

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện CNN Baseline trên Fruits-360"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    print("=" * 65)
    print("HUẤN LUYỆN CNN BASELINE")
    print("=" * 65)
    print("Device:", device)

    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))
        print("Mixed precision (AMP): bật")
    else:
        print("Mixed precision (AMP): tắt")

    # 1. Đọc dữ liệu
    train_loader, val_loader, _, class_names = build_dataloaders(
        image_size=224,
        batch_size=args.batch_size,
        val_ratio=0.2,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print("Số lớp:", len(class_names))
    print("Số mẫu train:", len(train_loader.dataset))
    print("Số mẫu validation:", len(val_loader.dataset))
    print("Số batch train mỗi epoch:", len(train_loader))

    # 2. Khởi tạo mô hình và các thành phần huấn luyện
    model = CNNBaseline(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "cnn_baseline_best.pth"
    history_path = results_dir / "cnn_baseline_history.csv"

    best_val_acc = -1.0
    epochs_without_improvement = 0

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
        ])

        # 3. Huấn luyện
        for epoch in range(1, args.epochs + 1):
            print(f"\n{'=' * 65}")
            print(f"EPOCH {epoch}/{args.epochs}")
            print("=" * 65)

            train_loss, train_acc = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                epoch=epoch,
                total_epochs=args.epochs,
            )

            val_loss, val_acc = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                epoch=epoch,
                total_epochs=args.epochs,
            )

            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc * 100:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc * 100:.2f}%")
            print(f"Learning Rate: {current_lr:.6g}")

            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{current_lr:.8g}",
            ])
            f.flush()

            # 4. Lưu checkpoint tốt nhất theo validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_accuracy": best_val_acc,
                        "class_names": class_names,
                        "image_size": 224,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )

                print(
                    f"[SAVE] Checkpoint tốt nhất: {checkpoint_path}"
                )
            else:
                epochs_without_improvement += 1
                print(
                    "Validation chưa cải thiện: "
                    f"{epochs_without_improvement}/{args.patience}"
                )

            # 5. Early stopping
            if epochs_without_improvement >= args.patience:
                print(
                    "\nEarly stopping: validation accuracy "
                    "không cải thiện đủ lâu."
                )
                break

    print("\n" + "=" * 65)
    print("HOÀN TẤT HUẤN LUYỆN")
    print("=" * 65)
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print("Lưu ý: tập Test chưa được sử dụng trong bước huấn luyện.")


if __name__ == "__main__":
    main()
