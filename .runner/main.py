from runner.generated_block import GeneratedBlock, CIN, H, W
import torch

model = GeneratedBlock(in_channels=CIN)
x = torch.randn(1, CIN, H, W)
model.eval()
with torch.no_grad():
    y = model(x)

print('Output shape:', tuple(y.shape))