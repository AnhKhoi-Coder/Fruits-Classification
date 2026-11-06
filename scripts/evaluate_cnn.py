
from pathlib import Path
import sys
import csv

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_baseline import CNNBaseline
from utils.dataset import build_dataloaders


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        PROJECT_ROOT / "checkpoints" / "cnn_baseline_best.pth"
    )
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    # Tạo DataLoader. Chỉ sử dụng test_loader trong đánh giá.
    _, _, test_loader, class_names = build_dataloaders(
        image_size=224,
        batch_size=32,
        val_ratio=0.2,
        seed=42,
        num_workers=0,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Kiểm tra checkpoint tương ứng với đúng danh sách lớp.
    if checkpoint["class_names"] != class_names:
        raise ValueError(
            "Danh sách lớp trong checkpoint không khớp dataset."
        )

    model = CNNBaseline(
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"

    print("=" * 60)
    print("ĐÁNH GIÁ CNN BASELINE TRÊN TẬP TEST")
    print("=" * 60)
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best validation accuracy:",
        f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
    )
    print("Số ảnh test:", len(test_loader.dataset))

    with torch.inference_mode():
        for step, (images, labels) in enumerate(test_loader, start=1):
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

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            if step % 100 == 0 or step == len(test_loader):
                print(
                    f"\rBatch {step}/{len(test_loader)}",
                    end="",
                    flush=True,
                )

    print()

    accuracy = accuracy_score(all_labels, all_predictions)

    # Macro: mỗi lớp có trọng số ngang nhau.
    # Weighted: tính trọng số theo số mẫu của mỗi lớp.
    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(len(class_names))),
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(len(class_names))),
            average="weighted",
            zero_division=0,
        )
    )

    average_loss = total_loss / total_samples

    print("\n--- KẾT QUẢ TEST ---")
    print(f"Test Loss: {average_loss:.6f}")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Macro Precision: {precision_macro:.6f}")
    print(f"Macro Recall: {recall_macro:.6f}")
    print(f"Macro F1-score: {f1_macro:.6f}")
    print(f"Weighted Precision: {precision_weighted:.6f}")
    print(f"Weighted Recall: {recall_weighted:.6f}")
    print(f"Weighted F1-score: {f1_weighted:.6f}")

    # Lưu báo cáo theo từng lớp.
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_path = results_dir / "cnn_baseline_classification_report.csv"

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

        for class_name in class_names:
            row = report[class_name]
            writer.writerow([
                class_name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            ])

    # Lưu confusion matrix dạng CSV để phân tích sau.
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    cm_path = results_dir / "cnn_baseline_confusion_matrix.csv"

    with cm_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted"] + class_names)

        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name] + row.tolist())

    print("\nĐã lưu báo cáo:", report_path)
    print("Đã lưu confusion matrix:", cm_path)
    print("[OK] Đánh giá tập Test hoàn tất.")


if __name__ == "__main__":
    main()
