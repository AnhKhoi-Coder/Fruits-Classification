from pathlib import Path
import sys
import csv
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTERNET_ROOT = PROJECT_ROOT / "FasterNet"

# Ưu tiên import CNN từ dự án hiện tại
sys.path.insert(0, str(PROJECT_ROOT))
from models.cnn_baseline import CNNBaseline

# Nạp registry chính thức của FasterNet
sys.path.insert(0, str(FASTERNET_ROOT))

import importlib.util

fasternet_init = FASTERNET_ROOT / "models" / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "fasternet_official_models",
    fasternet_init,
    submodule_search_locations=[str(FASTERNET_ROOT / "models")],
)

fasternet_registry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fasternet_registry
spec.loader.exec_module(fasternet_registry)

from timm.models import create_model

print("FasterNet registry:", fasternet_init)
print("fasternet registered:", "fasternet" in __import__(
    "timm.models", fromlist=["list_models"]
).list_models())

CNN_CKPT = PROJECT_ROOT / "checkpoints" / "cnn_baseline_best.pth"
FASTERNET_CKPT = PROJECT_ROOT / "checkpoints" / "fasternet_t1_best.pth"
OUTPUT_CSV = PROJECT_ROOT / "results" / "inference_benchmark.csv"
IMAGE_SIZE = 224
NUM_CLASSES = 274
BATCH_SIZES = [1, 8, 32, 64]
WARMUP_STEPS = 30
TIMED_STEPS = 100


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    return torch.load(path, map_location=device, weights_only=True)


def make_fasternet():
    return create_model(
        "fasternet", pretrained=False, num_classes=NUM_CLASSES,
        mlp_ratio=2, embed_dim=64, depths=[1, 2, 8, 2],
        feature_dim=1280, patch_size=4, patch_stride=4,
        patch_size2=2, patch_stride2=2, layer_scale_init_value=0,
        drop_path_rate=0.02, norm_layer="BN", act_layer="GELU",
        n_div=4, pconv_fw_type="split_cat",
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_one(model, model_name, batch_size, device, use_amp):
    model.eval()
    x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            for _ in range(TIMED_STEPS):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    model(x)
        end_event.record()
        torch.cuda.synchronize(device)
        total_ms = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(TIMED_STEPS):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    model(x)
        total_ms = (time.perf_counter() - start) * 1000

    batch_ms = total_ms / TIMED_STEPS
    return {
        "model": model_name,
        "device": str(device),
        "precision": "FP16 autocast" if use_amp else "FP32",
        "batch_size": batch_size,
        "avg_batch_latency_ms": round(batch_ms, 4),
        "avg_latency_per_image_ms": round(batch_ms / batch_size, 4),
        "throughput_images_per_sec": round(batch_size * TIMED_STEPS / (total_ms / 1000), 2),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    cnn_ckpt = load_checkpoint(CNN_CKPT, device)
    cnn = CNNBaseline(num_classes=NUM_CLASSES).to(device)
    cnn.load_state_dict(cnn_ckpt["model_state_dict"])
    cnn.eval()

    faster_ckpt = load_checkpoint(FASTERNET_CKPT, device)
    faster = make_fasternet().to(device)
    faster.load_state_dict(faster_ckpt.get("model_state_dict", faster_ckpt))
    faster.eval()

    models = [("CNN Baseline", cnn), ("FasterNet-T1", faster)]
    print("=" * 68)
    print("BENCHMARK INFERENCE: CNN BASELINE vs FASTERNET-T1")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")
    print(f"Input: [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}] | Precision: {'FP16 autocast' if use_amp else 'FP32'}")
    print(f"Warm-up: {WARMUP_STEPS} | Timed iterations: {TIMED_STEPS}")
    print("\n--- Model size ---")
    for name, model in models:
        params = count_parameters(model)
        print(f"{name}: {params:,} parameters; FP32 weights approximately {params * 4 / (1024**2):.2f} MiB")

    rows = []
    for batch_size in BATCH_SIZES:
        print(f"\n--- Batch size = {batch_size} ---")
        for name, model in models:
            result = benchmark_one(model, name, batch_size, device, use_amp)
            rows.append(result)
            print(f"{name:14s} | batch latency {result['avg_batch_latency_ms']:.4f} ms"
                  f" | per image {result['avg_latency_per_image_ms']:.4f} ms"
                  f" | throughput {result['throughput_images_per_sec']:.2f} images/s")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nĐã lưu CSV: {OUTPUT_CSV}")
    print("Lưu ý: đo model inference trên tensor giả lập đã nằm trên thiết bị; không tính thời gian đọc ảnh/DataLoader.")


if __name__ == "__main__":
    main()