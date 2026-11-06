
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

CNN_CSV = RESULTS_DIR / "cnn_baseline_history.csv"
FASTERNET_CSV = RESULTS_DIR / "fasternet_t1_history.csv"

# Tạo thư mục lưu 4 biểu đồ
OUTPUT_DIR = RESULTS_DIR / "training_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU CSV
# =========================================================

def load_history(csv_path, model_name):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file CSV: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\n{model_name} thiếu các cột: {missing}"
            f"\nCác cột hiện có: {list(df.columns)}"
        )

    df = df[required_columns].copy()

    # Chuyển dữ liệu thành kiểu số
    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Loại bỏ dòng dữ liệu không hợp lệ
    df = df.dropna()

    # Sắp xếp theo Epoch
    df = df.sort_values("epoch")

    if df.empty:
        raise ValueError(
            f"{model_name}: CSV không có dữ liệu hợp lệ."
        )

    # Kiểm tra số Epoch
    print(f"\n========== {model_name} ==========")
    print("Số Epoch:", len(df))
    print("Epoch đầu:", df["epoch"].min())
    print("Epoch cuối:", df["epoch"].max())

    if len(df) != 50:
        print(
            f"[CẢNH BÁO] Có {len(df)} Epoch, "
            "không phải 50 Epoch."
        )

    # Chuyển Accuracy về phần trăm nếu dữ liệu ở dạng 0-1
    for col in ["train_accuracy", "val_accuracy"]:

        if df[col].max() <= 1:
            df[col] = df[col] * 100

    return df


# Đọc dữ liệu hai mô hình
cnn = load_history(
    CNN_CSV,
    "CNN Baseline"
)

fasternet = load_history(
    FASTERNET_CSV,
    "FasterNet-T1"
)


# =========================================================
# 3. THIẾT LẬP GIAO DIỆN CHUNG
# =========================================================

# Màu sắc
TRAIN_COLOR = "#244F6A"
VAL_COLOR = "#218C87"

# Kích thước biểu đồ
FIG_SIZE = (10, 6)

# Font chữ
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


# =========================================================
# 4. HÀM VẼ BIỂU ĐỒ ACCURACY
# =========================================================

def plot_accuracy(df, model_name, file_name):

    fig, ax = plt.subplots(
        figsize=FIG_SIZE
    )

    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("white")

    epochs = df["epoch"]

    # Training Accuracy
    ax.plot(
        epochs,
        df["train_accuracy"],
        color=TRAIN_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Training Accuracy"
    )

    # Validation Accuracy
    ax.plot(
        epochs,
        df["val_accuracy"],
        color=VAL_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Validation Accuracy"
    )

    # Tiêu đề và nhãn trục
    ax.set_title(
        f"{model_name} - ACCURACY THEO EPOCH",
        fontsize=14,
        fontweight="bold",
        color=VAL_COLOR,
        pad=15
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=12
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=12
    )

    # Giới hạn trục
    ax.set_ylim(0, 105)

    # Chia trục Epoch
    max_epoch = int(df["epoch"].max())

    ax.set_xticks(
        range(0, max_epoch + 1, 5)
    )

    # Lưới
    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    # Chú thích
    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True
    )

    # Loại bỏ viền phía trên và bên phải
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Lưu ảnh
    output_path = OUTPUT_DIR / file_name

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print("Đã lưu:", output_path)


# =========================================================
# 5. HÀM VẼ BIỂU ĐỒ LOSS
# =========================================================

def plot_loss(df, model_name, file_name):

    fig, ax = plt.subplots(
        figsize=FIG_SIZE
    )

    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("white")

    epochs = df["epoch"]

    # Training Loss
    ax.plot(
        epochs,
        df["train_loss"],
        color=TRAIN_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Training Loss"
    )

    # Validation Loss
    ax.plot(
        epochs,
        df["val_loss"],
        color=VAL_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Validation Loss"
    )

    # Tiêu đề và nhãn trục
    ax.set_title(
        f"{model_name} - LOSS THEO EPOCH",
        fontsize=14,
        fontweight="bold",
        color=VAL_COLOR,
        pad=15
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=12
    )

    ax.set_ylabel(
        "Cross-entropy Loss",
        fontsize=12
    )

    # Chia trục Epoch
    max_epoch = int(df["epoch"].max())

    ax.set_xticks(
        range(0, max_epoch + 1, 5)
    )

    # Lưới
    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    # Chú thích
    ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=True
    )

    # Loại bỏ viền phía trên và bên phải
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Lưu ảnh
    output_path = OUTPUT_DIR / file_name

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print("Đã lưu:", output_path)


# =========================================================
# 6. TẠO 4 BIỂU ĐỒ RIÊNG BIỆT
# =========================================================

print("\n========================================")
print("BẮT ĐẦU VẼ 4 BIỂU ĐỒ TRAINING HISTORY")
print("========================================")

# Ảnh 1: CNN Accuracy
plot_accuracy(
    cnn,
    "CNN Baseline",
    "01_CNN_Accuracy.png"
)

# Ảnh 2: CNN Loss
plot_loss(
    cnn,
    "CNN Baseline",
    "02_CNN_Loss.png"
)

# Ảnh 3: FasterNet Accuracy
plot_accuracy(
    fasternet,
    "FasterNet-T1",
    "03_FasterNet_Accuracy.png"
)

# Ảnh 4: FasterNet Loss
plot_loss(
    fasternet,
    "FasterNet-T1",
    "04_FasterNet_Loss.png"
)


# =========================================================
# 7. THÔNG BÁO HOÀN THÀNH
# =========================================================

print("\n========================================")
print("ĐÃ TẠO THÀNH CÔNG 4 BIỂU ĐỒ!")
print("========================================")

print("Thư mục lưu ảnh:", OUTPUT_DIR)

print("\nDanh sách ảnh:")
print("1. 01_CNN_Accuracy.png")
print("2. 02_CNN_Loss.png")
print("3. 03_FasterNet_Accuracy.png")
print("4. 04_FasterNet_Loss.png")

print("\nĐộ phân giải: 300 DPI")
