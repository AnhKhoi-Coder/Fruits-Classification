
from pathlib import Path
import sys
import csv

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. DUONG DAN VA IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTERNET_ROOT = PROJECT_ROOT / "FasterNet"

# Import DataLoader cua du an truoc de tranh xung dot utils.
sys.path.insert(0, str(PROJECT_ROOT))
from utils.dataset import build_dataloaders

# Dang ky kien truc FasterNet vao timm.
sys.path.insert(0, str(FASTERNET_ROOT))
import models as fasternet_registry
from timm.models import create_model


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    checkpoint_path = (
        PROJECT_ROOT / "checkpoints" / "fasternet_t1_best.pth"
    )
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {checkpoint_path}"
        )

    print("=" * 65)
    print("DANH GIA FASTERNET-T1 TREN TAP TEST")
    print("=" * 65)
    print("Device:", device)

    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # BUOC 1: DOC DU LIEU
    # --------------------------------------------------------

    _, _, test_loader, class_names = build_dataloaders(
        image_size=224,
        batch_size=32,
        val_ratio=0.2,
        seed=42,
        num_workers=0,
    )

    num_classes = len(class_names)
    print("So lop:", num_classes)
    print("So mau test:", len(test_loader.dataset))

    # --------------------------------------------------------
    # BUOC 2: KHOI TAO MO HINH
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

    # Checkpoint do chinh chuong trinh huan luyen tao ra.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best validation accuracy:",
        f"{checkpoint['best_val_accuracy'] * 100:.2f}%",
    )

    # --------------------------------------------------------
    # BUOC 3: DU DOAN TREN TAP TEST
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0

    # Cross entropy dung de tinh test loss.
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, (images, labels) in enumerate(
            test_loader, start=1
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            if step % 200 == 0 or step == len(test_loader):
                print(
                    f"\rTest batch: {step}/{len(test_loader)}",
                    end="",
                    flush=True,
                )

    print()

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    test_loss = total_loss / total_samples
    accuracy = accuracy_score(y_true, y_pred)

    # Macro: cac lop co trong so ngang nhau.
    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    # Weighted: trong so theo so mau cua tung lop.
    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print("\n" + "=" * 65)
    print("KET QUA TEST")
    print("=" * 65)
    print(f"Test Loss:        {test_loss:.6f}")
    print(f"Accuracy:         {accuracy * 100:.4f}%")
    print(f"Macro Precision:  {macro_p:.6f}")
    print(f"Macro Recall:     {macro_r:.6f}")
    print(f"Macro F1:         {macro_f1:.6f}")
    print(f"Weighted Precision: {weighted_p:.6f}")
    print(f"Weighted Recall:    {weighted_r:.6f}")
    print(f"Weighted F1:        {weighted_f1:.6f}")

    # --------------------------------------------------------
    # BUOC 4: LUU CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_path = (
        results_dir / "fasternet_t1_classification_report.csv"
    )

    with report_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1-score",
            "support",
        ])

        for name, values in report.items():
            if isinstance(values, dict):
                writer.writerow([
                    name,
                    values.get("precision", ""),
                    values.get("recall", ""),
                    values.get("f1-score", ""),
                    values.get("support", ""),
                ])

    # --------------------------------------------------------
    # BUOC 5: LUU CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    cm_path = results_dir / "fasternet_t1_confusion_matrix.csv"

    with cm_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted"] + class_names)

        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name] + row.tolist())

    print("\nDa luu classification report:", report_path)
    print("Da luu confusion matrix:", cm_path)
    print("Hoan tat danh gia.")


if __name__ == "__main__":
    main()
