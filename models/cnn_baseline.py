
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Khối tích chập: Conv2d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    CNN Baseline cho bài toán phân loại ảnh Fruits-360.
    Input : [B, 3, 224, 224]
    Output: [B, num_classes] (logits, chưa qua Softmax)
    """

    def __init__(self, num_classes=274):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),       # 224 -> 112
            ConvBlock(32, 64),      # 112 -> 56
            ConvBlock(64, 128),     # 56 -> 28
            ConvBlock(128, 256),    # 28 -> 14
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNNBaseline(num_classes=274)

    # Kiểm tra trên GPU nếu có.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    num_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 50)
    print("KIỂM TRA CNN BASELINE")
    print("=" * 50)
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Output contains only finite values:",
          torch.isfinite(output).all().item())

    assert output.shape == (2, 274)
    assert torch.isfinite(output).all()

    print("[OK] CNN Baseline hoạt động bình thường.")
