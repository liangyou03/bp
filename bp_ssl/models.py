import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# ResNet-1D building blocks
# -----------------------------
class BasicBlock1D(nn.Module):
    """
    ResNet basic block for 1D signals.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    ResNet for 1D time-series.
    Input:  (B, C, L)
    Output: (B, feature_dim)
    """

    def __init__(
        self,
        block,
        layers,
        input_channels: int = 1,
        feature_dim: int = 256,
        stem_width: int = 64,
        kernel_size_stem: int = 15,
        stride_stem: int = 2,
        use_maxpool: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_planes = stem_width
        pad = kernel_size_stem // 2

        # Stem
        stem = [
            nn.Conv1d(
                input_channels,
                stem_width,
                kernel_size=kernel_size_stem,
                stride=stride_stem,
                padding=pad,
                bias=False,
            ),
            nn.BatchNorm1d(stem_width),
            nn.ReLU(inplace=True),
        ]
        if use_maxpool:
            stem.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem)

        # Stages (ResNet-18: [2,2,2,2])
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, feature_dim)

        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int):
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        # Kaiming init for conv, zeros for BN bias, ones for BN weight
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)  # (B,512)
        x = self.dropout(x)
        x = self.fc(x)                # (B,feature_dim)
        return x


def ResNet18_1D(input_channels: int = 1, feature_dim: int = 256, dropout: float = 0.0) -> ResNet1D:
    """
    Factory function: ResNet-18 1D encoder
    """
    return ResNet1D(
        block=BasicBlock1D,
        layers=[2, 2, 2, 2],
        input_channels=input_channels,
        feature_dim=feature_dim,
        dropout=dropout,
    )


# -----------------------------
# SimCLR head
# -----------------------------
class MLPProjector(nn.Module):
    """
    Standard SimCLR projector: 2-layer MLP
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    """
    Wrap an encoder with a projector.
    forward(x) -> (h, z)
      h: encoder embedding (B, feature_dim)
      z: projected embedding (B, projection_dim)
    """
    def __init__(self, encoder: nn.Module, feature_dim: int = 256, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = MLPProjector(feature_dim, projection_dim, hidden_dim=feature_dim)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
