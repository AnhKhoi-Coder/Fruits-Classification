

# ============================================================
# 1. DUONG DAN VA IMPORT
# ============================================================

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
FASTERNET_ROOT = PROJECT_ROOT / "FasterNet"

# Buoc 1: Import DataLoader cua du an truoc.
# Chua them FasterNet vao sys.path de tranh xung dot package utils.
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import build_dataloaders

# Buoc 2: Them repo FasterNet va dang ky kien truc.
sys.path.insert(0, str(FASTERNET_ROOT))

import models as fasternet_registry
from timm.models import create_model



# ============================================================
# 2. CO DINH RANDOM SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. HUAN LUYEN HOAC DANH GIA MOT EPOCH
# ============================================================

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

    if is_train:
        model.train()
    else:
        model.eval()

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
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

            if is_train and (
                step % 100 == 0 or step == len(loader)
            ):
                print(
                    f"\rEpoch {epoch}/{total_epochs} "
                    f"| Batch {step}/{len(loader)} "
                    f"| Loss: {loss.item():.4f}",
                    end="",
                    flush=True,
                )

    if is_train:
        print()

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# 4. CHUONG TRINH CHINH
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Huan luyen FasterNet-T1 tren Fruits-360"
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    print("=" * 65)
    print("HUAN LUYEN FASTERNET-T1")
    print("=" * 65)
    print("Device:", device)

    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))
        print("Mixed precision (AMP): bat")
    else:
        print("Mixed precision (AMP): tat")

    # --------------------------------------------------------
    # BUOC 1: DOC DU LIEU
    # --------------------------------------------------------

    train_loader, val_loader, _, class_names = build_dataloaders(
        image_size=224,
        batch_size=args.batch_size,
        val_ratio=0.2,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    num_classes = len(class_names)

    print("So lop:", num_classes)
    print("So mau train:", len(train_loader.dataset))
    print("So mau validation:", len(val_loader.dataset))
    print("So batch train moi epoch:", len(train_loader))

    # --------------------------------------------------------
    # BUOC 2: KHOI TAO FASTERNET-T1
    # --------------------------------------------------------

    model = create_model(
        "fasternet",
        pretrained=False,
        num_classes=num_classes,
        mlp_ratio=2,
        embed_dim=64,
        depths=[1, 2, 8, 2],
        feature_dim=1280,
        patch_size=4,
        patch_stride=4,
        patch_size2=2,
        patch_stride2=2,
        layer_scale_init_value=0,
        drop_path_rate=0.02,
        norm_layer="BN",
        act_layer="GELU",
        n_div=4,
        pconv_fw_type="split_cat",
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # BUOC 3: LOSS, OPTIMIZER, SCHEDULER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BUOC 4: NOI LUU CHECKPOINT VA LICH SU
    # --------------------------------------------------------

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / "fasternet_t1_best.pth"
    )
    history_path = (
        results_dir / "fasternet_t1_history.csv"
    )

    best_val_acc = -1.0
    epochs_without_improvement = 0

    # --------------------------------------------------------
    # BUOC 5: VONG LAP HUAN LUYEN
    # --------------------------------------------------------

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
        ])

        for epoch in range(1, args.epochs + 1):
            print("\n" + "=" * 65)
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

            # Luu checkpoint neu validation accuracy tot hon.
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
                        "model_name": "fasternet_t1",
                    },
                    checkpoint_path,
                )

                print(f"[SAVE] Checkpoint: {checkpoint_path}")
            else:
                epochs_without_improvement += 1

                print(
                    "Validation chua cai thien: "
                    f"{epochs_without_improvement}/{args.patience}"
                )

            if epochs_without_improvement >= args.patience:
                print("\nEarly stopping.")
                break

    print("\n" + "=" * 65)
    print("HOAN TAT HUAN LUYEN FASTERNET-T1")
    print("=" * 65)
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print("Tap Test chua duoc su dung trong qua trinh huan luyen.")


if __name__ == "__main__":
    main()
