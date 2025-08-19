import torch
import torch.nn as nn
import torch.nn.functional as F

# Inferred from UI (available to import in main.py)
CIN = 1
H = 28
W = 28

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(max(0.0, min(1.0, p)))

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        if keep <= 0.0:
            return torch.zeros_like(x)
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        noise = x.new_empty(shape).bernoulli_(keep) / keep
        return x * noise

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        # apply LayerNorm over channels in channels-last order
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class GeneratedBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.layer_0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.layer_1 = LayerNorm2d(1)
        self.layer_2 = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer_3 = nn.GELU()
        self.layer_4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer_5 = DropPath(p=0.1)
        self.layer_6 = nn.Identity()  # TODO: Residual Add handling in forward

    def forward(self, x):
        ys = []
        x = self.layer_0(x)
        ys.append(x)
        x = self.layer_1(x)
        ys.append(x)
        x = self.layer_2(x)
        ys.append(x)
        x = self.layer_3(x)
        ys.append(x)
        x = self.layer_4(x)
        ys.append(x)
        x = self.layer_5(x)
        ys.append(x)
        x = x + ys[5]  # Residual add
        ys.append(x)
        return x

class GeneratedModel(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.m_0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.m_1 = LayerNorm2d(1)
        self.m_2 = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_3 = nn.GELU()
        self.m_4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_5 = DropPath(p=0.1)
        self.m_6 = nn.Identity()
        self.m_7 = nn.AdaptiveAvgPool2d(1)
        self.m_8 = nn.Linear(64, 10)

    def forward(self, x):
        ys = []
        x = self.m_0(x)
        ys.append(x)
        x = self.m_1(x)
        ys.append(x)
        x = self.m_2(x)
        ys.append(x)
        x = self.m_3(x)
        ys.append(x)
        x = self.m_4(x)
        ys.append(x)
        x = self.m_5(x)
        ys.append(x)
        x = x + ys[5]  # Residual add
        ys.append(x)
        x = self.m_7(x)
        ys.append(x)
        if x.dim() > 2:
            x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.m_8(x)
        ys.append(x)
        return x