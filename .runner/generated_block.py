import torch
import torch.nn as nn
import torch.nn.functional as F

# Inferred from UI (available to import in main.py)
CIN = 3
H = 32
W = 32

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

class MHSA2D(nn.Module):
    def __init__(self, c, heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=int(heads), dropout=float(attn_drop), batch_first=True)
        self.proj = nn.Linear(c, c)
        self.proj_drop = nn.Dropout(p=float(proj_drop))

    def forward(self, x):
        # x: (N,C,H,W) -> (N,HW,C)
        N, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(N, H*W, C)
        out, _ = self.attn(x_seq, x_seq, x_seq)
        out = self.proj_drop(self.proj(out))
        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return out

class GeneratedBlock(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

    def forward(self, x):
        ys = []
        return x

class GeneratedModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.m_0 = nn.Conv2d(3, 384, kernel_size=16, stride=16, padding=0, bias=False)
        self.m_1 = LayerNorm2d(384)
        self.m_2 = MHSA2D(384, heads=8, attn_drop=0, proj_drop=0)
        self.m_3 = nn.Identity()
        self.m_4 = LayerNorm2d(384)
        self.m_5 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_6 = nn.GELU()
        self.m_7 = nn.Dropout(p=0.5)
        self.m_8 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_9 = nn.Identity()
        self.m_10 = nn.Identity()
        self.m_11 = LayerNorm2d(384)
        self.m_12 = MHSA2D(384, heads=8, attn_drop=0, proj_drop=0)
        self.m_13 = nn.Identity()
        self.m_14 = LayerNorm2d(384)
        self.m_15 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_16 = nn.GELU()
        self.m_17 = nn.Dropout(p=0.5)
        self.m_18 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_19 = nn.Identity()
        self.m_20 = nn.Identity()
        self.m_21 = LayerNorm2d(384)
        self.m_22 = MHSA2D(384, heads=8, attn_drop=0, proj_drop=0)
        self.m_23 = nn.Identity()
        self.m_24 = LayerNorm2d(384)
        self.m_25 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_26 = nn.GELU()
        self.m_27 = nn.Dropout(p=0.5)
        self.m_28 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_29 = nn.Identity()
        self.m_30 = nn.Identity()
        self.m_31 = LayerNorm2d(384)
        self.m_32 = MHSA2D(384, heads=8, attn_drop=0, proj_drop=0)
        self.m_33 = nn.Identity()
        self.m_34 = LayerNorm2d(384)
        self.m_35 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_36 = nn.GELU()
        self.m_37 = nn.Dropout(p=0.5)
        self.m_38 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_39 = nn.Identity()
        self.m_40 = nn.Identity()
        self.m_41 = LayerNorm2d(384)
        self.m_42 = MHSA2D(384, heads=8, attn_drop=0, proj_drop=0)
        self.m_43 = nn.Identity()
        self.m_44 = LayerNorm2d(384)
        self.m_45 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_46 = nn.GELU()
        self.m_47 = nn.Dropout(p=0.5)
        self.m_48 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_49 = nn.Identity()
        self.m_50 = nn.Identity()
        self.m_51 = LayerNorm2d(384)
        self.m_52 = MHSA2D(384, heads=8, attn_drop=0, proj_drop=0)
        self.m_53 = nn.Identity()
        self.m_54 = LayerNorm2d(384)
        self.m_55 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_56 = nn.GELU()
        self.m_57 = nn.Dropout(p=0.5)
        self.m_58 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_59 = nn.Identity()
        self.m_60 = nn.Identity()
        self.m_61 = nn.AdaptiveAvgPool2d(1)
        self.m_62 = nn.Linear(384, 10)

    def forward(self, x):
        ys = []
        x = self.m_0(x)
        ys.append(x)
        x = self.m_1(x)
        ys.append(x)
        x = self.m_2(x)
        ys.append(x)
        res = ys[2]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_4(x)
        ys.append(x)
        x = self.m_5(x)
        ys.append(x)
        x = self.m_6(x)
        ys.append(x)
        x = self.m_7(x)
        ys.append(x)
        x = self.m_8(x)
        ys.append(x)
        res = ys[8]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        res = ys[0]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_11(x)
        ys.append(x)
        x = self.m_12(x)
        ys.append(x)
        res = ys[12]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_14(x)
        ys.append(x)
        x = self.m_15(x)
        ys.append(x)
        x = self.m_16(x)
        ys.append(x)
        x = self.m_17(x)
        ys.append(x)
        x = self.m_18(x)
        ys.append(x)
        res = ys[18]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        res = ys[10]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_21(x)
        ys.append(x)
        x = self.m_22(x)
        ys.append(x)
        res = ys[22]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_24(x)
        ys.append(x)
        x = self.m_25(x)
        ys.append(x)
        x = self.m_26(x)
        ys.append(x)
        x = self.m_27(x)
        ys.append(x)
        x = self.m_28(x)
        ys.append(x)
        res = ys[28]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        res = ys[20]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_31(x)
        ys.append(x)
        x = self.m_32(x)
        ys.append(x)
        res = ys[32]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_34(x)
        ys.append(x)
        x = self.m_35(x)
        ys.append(x)
        x = self.m_36(x)
        ys.append(x)
        x = self.m_37(x)
        ys.append(x)
        x = self.m_38(x)
        ys.append(x)
        res = ys[38]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        res = ys[30]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_41(x)
        ys.append(x)
        x = self.m_42(x)
        ys.append(x)
        res = ys[42]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_44(x)
        ys.append(x)
        x = self.m_45(x)
        ys.append(x)
        x = self.m_46(x)
        ys.append(x)
        x = self.m_47(x)
        ys.append(x)
        x = self.m_48(x)
        ys.append(x)
        res = ys[48]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        res = ys[40]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_51(x)
        ys.append(x)
        x = self.m_52(x)
        ys.append(x)
        res = ys[52]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_54(x)
        ys.append(x)
        x = self.m_55(x)
        ys.append(x)
        x = self.m_56(x)
        ys.append(x)
        x = self.m_57(x)
        ys.append(x)
        x = self.m_58(x)
        ys.append(x)
        res = ys[58]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        res = ys[50]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_61(x)
        ys.append(x)
        if x.dim() > 2:
            x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.m_62(x)
        ys.append(x)
        return x