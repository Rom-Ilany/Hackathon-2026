"""
Small ResNet, implemented manually from scratch.

No pretrained weights, no external datasets, no downloads.
Only torch / torch.nn are used (no torchvision.models).

The class ModelArchitecture is what predict.py imports.
It must:
    - take input of shape  [batch_size, 3, 224, 224]
    - return logits  of shape  [batch_size, num_classes]
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock.

    Two 3x3 convolutions with BatchNorm + ReLU, plus a residual
    connection that adds the input back at the end.

    If the input and the second conv's output do NOT have the same
    shape (because of a stride change or a channel change), the
    shortcut path applies a 1x1 conv (+ BN) to project the input
    to the right shape. Otherwise the shortcut is just the identity.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # First conv: may downsample (stride 2) and may change channels.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv: keeps channels and spatial size.
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut: 1x1 projection only when the residual addition would
        # otherwise have mismatched shapes.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # residual addition
        out = self.relu(out)
        return out


def _make_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
) -> nn.Sequential:
    """
    Build one stage made of `num_blocks` BasicBlocks.

    Only the first block in the stage uses the given stride (so it
    is the one that downsamples). The remaining blocks keep stride 1.
    """
    blocks = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(num_blocks - 1):
        blocks.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*blocks)


class ModelArchitecture(nn.Module):
    """
    Small ResNet for the 20-class hackathon task.

    Spatial size trace for a 224x224 input:

        Stem  (Conv 7x7, stride 2)       :  224 -> 112    channels: 3   -> 64
        Stage 1 (2 BasicBlocks, stride 1):  112 -> 112    channels: 64  -> 64
        Stage 2 (2 BasicBlocks, stride 2):  112 -> 56     channels: 64  -> 128
        Stage 3 (2 BasicBlocks, stride 2):  56  -> 28     channels: 128 -> 256
        AdaptiveAvgPool2d((1, 1))        :  28  -> 1
        Flatten + Linear(256 -> 20)      :  -> 20 logits

    Roughly ~2.8M parameters, which is much smaller than full ResNet18 (~11.7M).
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Stem: standard ResNet-style. Lowers resolution by 2 (224 -> 112)
        # and lifts channels from 3 to 64.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Three stages of residual blocks. Each new stage doubles the
        # channel count and halves the spatial size (except stage 1).
        self.stage1 = _make_stage(in_channels=64,  out_channels=64,  num_blocks=2, stride=1)
        self.stage2 = _make_stage(in_channels=64,  out_channels=128, num_blocks=2, stride=2)
        self.stage3 = _make_stage(in_channels=128, out_channels=256, num_blocks=2, stride=2)

        # Global average pool + linear classifier.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Sensible from-scratch init (He for conv/linear, 1/0 for BN)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # [batch_size, 256]
        logits = self.fc(x)      # [batch_size, num_classes]
        return logits


if __name__ == "__main__":
    # Quick shape sanity check.
    model = ModelArchitecture(num_classes=20)
    model.eval()

    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}")
    print(f"params : {n_params:,}")

    assert y.shape == (4, 20), f"expected (4, 20), got {tuple(y.shape)}"
    print("[OK] forward pass returns shape [4, 20]")
