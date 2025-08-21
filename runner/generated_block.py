import torch
import torch.nn as nn
import torch.nn.functional as F

# Inferred from UI (available to import in main.py)
CIN = 3
H = 32
W = 32

class GeneratedBlock(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

    def forward(self, x):
        ys = []
        return x

class GeneratedModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.m_0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_1 = nn.BatchNorm2d(64)
        self.m_2 = nn.ReLU(inplace=True)
        self.m_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_4 = nn.BatchNorm2d(64)
        self.m_5 = nn.ReLU(inplace=True)
        self.m_6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_7 = nn.BatchNorm2d(64)
        self.m_8 = nn.Identity()
        self.m_9 = nn.ReLU(inplace=True)
        self.m_10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_11 = nn.BatchNorm2d(64)
        self.m_12 = nn.ReLU(inplace=True)
        self.m_13 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_14 = nn.BatchNorm2d(64)
        self.m_15 = nn.Identity()
        self.m_16 = nn.ReLU(inplace=True)
        self.m_17 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=1, dilation=1, bias=False)
        self.m_18 = nn.BatchNorm2d(128)
        self.m_19 = nn.ReLU(inplace=True)
        self.m_20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_21 = nn.BatchNorm2d(128)
        self.m_22 = nn.Identity()
        self.m_proj_22 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128)
        )
        self.m_23 = nn.ReLU(inplace=True)
        self.m_24 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_25 = nn.BatchNorm2d(128)
        self.m_26 = nn.ReLU(inplace=True)
        self.m_27 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_28 = nn.BatchNorm2d(128)
        self.m_29 = nn.Identity()
        self.m_30 = nn.ReLU(inplace=True)
        self.m_31 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, groups=1, dilation=1, bias=False)
        self.m_32 = nn.BatchNorm2d(256)
        self.m_33 = nn.ReLU(inplace=True)
        self.m_34 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_35 = nn.BatchNorm2d(256)
        self.m_36 = nn.Identity()
        self.m_proj_36 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256)
        )
        self.m_37 = nn.ReLU(inplace=True)
        self.m_38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_39 = nn.BatchNorm2d(256)
        self.m_40 = nn.ReLU(inplace=True)
        self.m_41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_42 = nn.BatchNorm2d(256)
        self.m_43 = nn.Identity()
        self.m_44 = nn.ReLU(inplace=True)
        self.m_45 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, groups=1, dilation=1, bias=False)
        self.m_46 = nn.BatchNorm2d(512)
        self.m_47 = nn.ReLU(inplace=True)
        self.m_48 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_49 = nn.BatchNorm2d(512)
        self.m_50 = nn.Identity()
        self.m_proj_50 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512)
        )
        self.m_51 = nn.ReLU(inplace=True)
        self.m_52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_53 = nn.BatchNorm2d(512)
        self.m_54 = nn.ReLU(inplace=True)
        self.m_55 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.m_56 = nn.BatchNorm2d(512)
        self.m_57 = nn.Identity()
        self.m_58 = nn.ReLU(inplace=True)
        self.m_59 = nn.AdaptiveAvgPool2d(1)
        self.m_60 = nn.Linear(512, 10)

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
        x = self.m_6(x)
        ys.append(x)
        x = self.m_7(x)
        ys.append(x)
        res = ys[2]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_9(x)
        ys.append(x)
        x = self.m_10(x)
        ys.append(x)
        x = self.m_11(x)
        ys.append(x)
        x = self.m_12(x)
        ys.append(x)
        x = self.m_13(x)
        ys.append(x)
        x = self.m_14(x)
        ys.append(x)
        res = ys[9]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_16(x)
        ys.append(x)
        x = self.m_17(x)
        ys.append(x)
        x = self.m_18(x)
        ys.append(x)
        x = self.m_19(x)
        ys.append(x)
        x = self.m_20(x)
        ys.append(x)
        x = self.m_21(x)
        ys.append(x)
        res = ys[16]
        res = self.m_proj_22(res)
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_23(x)
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
        res = ys[23]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_30(x)
        ys.append(x)
        x = self.m_31(x)
        ys.append(x)
        x = self.m_32(x)
        ys.append(x)
        x = self.m_33(x)
        ys.append(x)
        x = self.m_34(x)
        ys.append(x)
        x = self.m_35(x)
        ys.append(x)
        res = ys[30]
        res = self.m_proj_36(res)
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_37(x)
        ys.append(x)
        x = self.m_38(x)
        ys.append(x)
        x = self.m_39(x)
        ys.append(x)
        x = self.m_40(x)
        ys.append(x)
        x = self.m_41(x)
        ys.append(x)
        x = self.m_42(x)
        ys.append(x)
        res = ys[37]
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
        x = self.m_49(x)
        ys.append(x)
        res = ys[44]
        res = self.m_proj_50(res)
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_51(x)
        ys.append(x)
        x = self.m_52(x)
        ys.append(x)
        x = self.m_53(x)
        ys.append(x)
        x = self.m_54(x)
        ys.append(x)
        x = self.m_55(x)
        ys.append(x)
        x = self.m_56(x)
        ys.append(x)
        res = ys[51]
        if res.shape[2:] != x.shape[2:]:
            res = F.adaptive_avg_pool2d(res, x.shape[2:])
        x = x + res  # Residual add
        ys.append(x)
        x = self.m_58(x)
        ys.append(x)
        x = self.m_59(x)
        ys.append(x)
        if x.dim() > 2:
            x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.m_60(x)
        ys.append(x)
        return x