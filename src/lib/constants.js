// Shared constants and catalogs

export const CAT_COLORS = {
  Convolution: { chip: "bg-blue-100 text-blue-700", ring: "ring-blue-200" },
  Normalization: { chip: "bg-violet-100 text-violet-700", ring: "ring-violet-200" },
  Activation: { chip: "bg-amber-100 text-amber-800", ring: "ring-amber-200" },
  Pooling: { chip: "bg-teal-100 text-teal-800", ring: "ring-teal-200" },
  Attention: { chip: "bg-rose-100 text-rose-700", ring: "ring-rose-200" },
  Residual: { chip: "bg-slate-200 text-slate-800", ring: "ring-slate-300" },
  Regularization: { chip: "bg-emerald-100 text-emerald-800", ring: "ring-emerald-200" },
  Linear: { chip: "bg-fuchsia-100 text-fuchsia-800", ring: "ring-fuchsia-200" },
  Meta: { chip: "bg-neutral-200 text-neutral-800", ring: "ring-neutral-300" },
};

export const LAYERS = [
  { id: "conv", name: "Conv2d", category: "Convolution", role: "Learnable local features", op: "k×k, stride s, groups g", defaults: { k: 3, s: 1, g: 1, outC: 64 }, dimReq: "N,C,H,W · flexible; updates H,W by stride" },
  { id: "pwconv", name: "Pointwise Conv (1×1)", category: "Convolution", role: "Channel mixing", op: "1×1", defaults: { k: 1, s: 1, g: 1, outC: 64 }, dimReq: "N,C,H,W · preserves H,W" },
  { id: "dwconv", name: "Depthwise Conv", category: "Convolution", role: "Per-channel spatial conv", op: "k×k, groups=inC", defaults: { k: 3, s: 1 }, dimReq: "N,C,H,W · groups=inC (per-channel)" },
  { id: "grpconv", name: "Grouped Conv", category: "Convolution", role: "Split channels into groups", op: "k×k, groups>1", defaults: { k: 3, s: 1, g: 32, outC: 256 }, dimReq: "N,C,H,W · C divisible by groups" },
  { id: "dilconv", name: "Dilated Conv", category: "Convolution", role: "Expand receptive field", op: "k×k, dilation d", defaults: { k: 3, s: 1, d: 2, outC: 64 }, dimReq: "N,C,H,W · dilation-aware padding" },
  { id: "deform", name: "Deformable Conv v2", category: "Convolution", role: "Learnable offsets", op: "DCNv2", defaults: { k: 3, s: 1, outC: 64 }, dimReq: "N,C,H,W · conv-compatible" },
  { id: "bn", name: "BatchNorm2d", category: "Normalization", role: "Normalize activations", op: "(x−μ)/σ·γ+β", dimReq: "N,C,H,W · per-channel" },
  { id: "gn", name: "GroupNorm", category: "Normalization", role: "Batch-size agnostic", op: "Group-wise norm", defaults: { groups: 32 }, dimReq: "N,C,H,W · C divisible by groups" },
  { id: "ln", name: "LayerNorm", category: "Normalization", role: "Per-sample norm", op: "Across channels", dimReq: "N,C,H,W or features · per-sample" },
  { id: "relu", name: "ReLU", category: "Activation", role: "Non-linearity", op: "max(0,x)", dimReq: "Any tensor · same shape out" },
  { id: "gelu", name: "GELU", category: "Activation", role: "Smooth activation", op: "x·Φ(x)", dimReq: "Any tensor · same shape out" },
  { id: "silu", name: "SiLU (Swish)", category: "Activation", role: "Smooth gated", op: "x·sigmoid(x)", dimReq: "Any tensor · same shape out" },
  { id: "hswish", name: "Hardswish", category: "Activation", role: "Efficient swish", op: "x·ReLU6(x+3)/6", dimReq: "Any tensor · same shape out" },
  { id: "prelu", name: "PReLU", category: "Activation", role: "Learnable negative slope", op: "ax for x<0", dimReq: "Any tensor · same shape out" },
  { id: "maxpool", name: "MaxPool2d", category: "Pooling", role: "Downsample by max", op: "k×k s", dimReq: "N,C,H,W · reduces H,W" },
  { id: "avgpool", name: "AvgPool2d", category: "Pooling", role: "Downsample by avg", op: "k×k s", dimReq: "N,C,H,W · reduces H,W" },
  { id: "gap", name: "Global Avg Pool", category: "Pooling", role: "H×W → 1×1", op: "AdaptiveAvgPool2d(1)", dimReq: "N,C,H,W · outputs 1×1" },
  { id: "se", name: "Squeeze-and-Excitation", category: "Regularization", role: "Channel attention", op: "GAP→MLP→sigmoid→scale", defaults: { r: 16 }, dimReq: "N,C,H,W · channel-wise" },
  { id: "eca", name: "ECA", category: "Regularization", role: "Efficient channel attention", op: "1D conv over channels", dimReq: "N,C,H,W · channel-wise" },
  { id: "cbam", name: "CBAM", category: "Regularization", role: "Channel+Spatial attention", op: "SE + spatial mask", dimReq: "N,C,H,W · channel+spatial" },
  { id: "mhsa", name: "MHSA (2D)", category: "Attention", role: "Self-attention", op: "softmax(QK^T/√d)·V", dimReq: "N,C,H,W · O((H·W)^2)" },
  { id: "winattn", name: "Windowed Attention", category: "Attention", role: "Local attention", op: "Swin-style windows", dimReq: "N,C,H,W · windowable H,W" },
  { id: "add", name: "Residual Add", category: "Residual", role: "Skip connection", op: "x + F(x)", defaults: { from: null }, dimReq: "Requires same (C,H,W) as source" },
  { id: "concat", name: "Concatenate", category: "Residual", role: "Channel concat", op: "[x, F(x)]", dimReq: "Match (H,W) across inputs" },
  { id: "linear", name: "Linear", category: "Linear", role: "Projection", op: "out = xW + b", defaults: { outF: 1000 }, dimReq: "Features · flattens (C·H·W)" },
  { id: "dropout", name: "Dropout", category: "Regularization", role: "Random feature drop", op: "p", dimReq: "Any tensor" },
  { id: "droppath", name: "Stochastic Depth", category: "Regularization", role: "Randomly drop residual branch", op: "prob p", dimReq: "Residual branch only" },
];

export const PRESETS = [
  { id: "resnet_basic", name: "ResNet BasicBlock", family: "ResNet-18/34", composition: ["conv","bn","relu","conv","bn","add","relu"], strengths: ["Simple","Stable"], drawbacks: ["Less params-efficient when very deep"], goodSlots: ["stage"] },
  { id: "resnet_bottleneck", name: "ResNet Bottleneck", family: "ResNet-50/101/152", composition: ["pwconv","bn","relu","conv","bn","relu","pwconv","bn","add","relu"], strengths: ["Efficient at scale"], drawbacks: ["1×1 bandwidth","Memory"], goodSlots: ["stage"] },
  { id: "resnext", name: "ResNeXt Bottleneck (cardinality)", family: "ResNeXt", composition: ["pwconv","bn","relu","grpconv","bn","relu","pwconv","bn","add","relu"], strengths: ["Higher accuracy at similar cost"], drawbacks: ["Group-conv perf variance"], goodSlots: ["stage"] },
  { id: "preact_bottleneck", name: "Pre-activation Bottleneck", family: "ResNet v2", composition: ["bn","relu","pwconv","bn","relu","conv","bn","relu","pwconv","add"], strengths: ["Smoother optimization"], drawbacks: ["Layout-only change"], goodSlots: ["stage"] },
  { id: "mbv1", name: "MobileNetV1 DW-Separable", family: "MobileNetV1", composition: ["dwconv","bn","relu","pwconv","bn","relu"], strengths: ["Very efficient"], drawbacks: ["Depthwise kernel perf sensitivity"], goodSlots: ["stage"] },
  { id: "mbv2", name: "MobileNetV2 Inverted Residual", family: "MobileNetV2", composition: ["pwconv","bn","relu","dwconv","bn","relu","pwconv","bn","add"], strengths: ["Edge efficiency"], drawbacks: ["Linear bottleneck sensitivity"], goodSlots: ["stage"] },
  { id: "mbv3", name: "MobileNetV3 Block (+SE, h-swish)", family: "MobileNetV3", composition: ["pwconv","bn","relu","dwconv","bn","se","hswish","pwconv","bn","add"], strengths: ["Strong mobile accuracy"], drawbacks: ["Extra complexity"], goodSlots: ["stage"] },
  { id: "efficient_mbconv", name: "EfficientNet MBConv + SE (+SiLU)", family: "EfficientNet", composition: ["pwconv","bn","silu","dwconv","bn","se","silu","pwconv","bn","add"], strengths: ["Accuracy/efficiency balance"], drawbacks: ["Training sensitivity"], goodSlots: ["stage"] },
  { id: "convnext", name: "ConvNeXt Block", family: "ConvNeXt", composition: ["dwconv","ln","pwconv","gelu","pwconv","droppath","add"], strengths: ["Modern accuracy","Simple"], drawbacks: ["DW perf varies"], goodSlots: ["stage"] },
  { id: "densenet_dense", name: "DenseNet Dense Block (k growth)", family: "DenseNet", composition: ["bn","relu","conv","concat"], strengths: ["Feature reuse"], drawbacks: ["Memory footprint"], goodSlots: ["stage"] },
  { id: "inception_a", name: "Inception-v3 Module (A)", family: "InceptionV3", composition: ["conv","conv","conv","concat","bn","relu"], strengths: ["Multi-scale"], drawbacks: ["Complex wiring"], goodSlots: ["stage"] },
  { id: "repvgg", name: "RepVGG Block (train-time branches)", family: "RepVGG", composition: ["conv","bn","relu","conv","bn","relu","add"], strengths: ["Re-parameterizable to 3×3"], drawbacks: ["Reparam step"], goodSlots: ["stage"] },
  { id: "ghost", name: "GhostNet Ghost Bottleneck", family: "GhostNet", composition: ["pwconv","relu","dwconv","pwconv","add"], strengths: ["Cheap feature maps"], drawbacks: ["Approximation artifacts"], goodSlots: ["stage"] },
  { id: "squeezenet_fire", name: "SqueezeNet Fire Module", family: "SqueezeNet", composition: ["pwconv","relu","conv","conv","concat"], strengths: ["Few params"], drawbacks: ["Lower accuracy than modern nets"], goodSlots: ["stage"] },
];

export const MODEL_PRESETS = [
  {
    id: 'resnet18_toy',
    name: 'ResNet-18 (toy)',
    family: 'ResNet',
    description: 'Stem + stacks of BasicBlocks, GAP + Linear',
    plan: [
      { type: 'layer', id: 'conv', cfg: { outC: 64, k: 7, s: 2 } },
      { type: 'layer', id: 'bn' },
      { type: 'layer', id: 'relu' },
      { type: 'layer', id: 'maxpool', cfg: { k: 3, s: 2 } },
      { type: 'blockRef', preset: 'resnet_basic', repeat: 2, outC: 64 },
      { type: 'blockRef', preset: 'resnet_basic', repeat: 2, outC: 128 },
      { type: 'blockRef', preset: 'resnet_basic', repeat: 2, outC: 256 },
      { type: 'blockRef', preset: 'resnet_basic', repeat: 2, outC: 512 },
      { type: 'layer', id: 'gap' },
      { type: 'layer', id: 'linear', cfg: { outF: 1000 } },
    ],
  },
  {
    id: 'mobilenetv2_stack',
    name: 'MobileNetV2 Stack',
    family: 'MobileNetV2',
    description: 'Light stem + MBConv blocks, GAP + Linear',
    plan: [
      { type: 'layer', id: 'conv', cfg: { outC: 32, k: 3, s: 2 } },
      { type: 'layer', id: 'bn' },
      { type: 'layer', id: 'relu' },
      { type: 'blockRef', preset: 'mbv2', repeat: 3, outC: 32 },
      { type: 'blockRef', preset: 'mbv2', repeat: 4, outC: 64 },
      { type: 'blockRef', preset: 'mbv2', repeat: 3, outC: 96 },
      { type: 'blockRef', preset: 'mbv2', repeat: 3, outC: 160 },
      { type: 'layer', id: 'gap' },
      { type: 'layer', id: 'linear', cfg: { outF: 1000 } },
    ],
  },
  {
    id: 'convnext_stack',
    name: 'ConvNeXt Stack',
    family: 'ConvNeXt',
    description: 'Stacks of ConvNeXt blocks, GAP + Linear',
    plan: [
      { type: 'layer', id: 'conv', cfg: { outC: 96, k: 4, s: 4 } },
      { type: 'layer', id: 'bn' },
      { type: 'layer', id: 'relu' },
      { type: 'blockRef', preset: 'convnext', repeat: 3, outC: 96 },
      { type: 'blockRef', preset: 'convnext', repeat: 3, outC: 192 },
      { type: 'blockRef', preset: 'convnext', repeat: 9, outC: 384 },
      { type: 'blockRef', preset: 'convnext', repeat: 3, outC: 768 },
      { type: 'layer', id: 'gap' },
      { type: 'layer', id: 'linear', cfg: { outF: 1000 } },
    ],
  },
];

export const SYNERGIES = [
  { need:["conv","bn"], why:"BatchNorm after Conv stabilizes stats; enables larger LR.", tag:"stability" },
  { need:["dwconv","pwconv"], why:"Depthwise + pointwise forms an efficient separable conv.", tag:"efficiency" },
  { need:["bn","relu"], why:"BN followed by ReLU is a robust pairing for CNNs.", tag:"stability" },
  { need:["se"], why:"SE adds channel recalibration; often improves accuracy with minor cost.", tag:"accuracy" },
  { need:["droppath","add"], why:"Stochastic depth regularizes residual paths in deep stacks.", tag:"regularization" },
  { need:["mhsa","gelu"], why:"GELU pairs well with attention due to smoother gradients.", tag:"stability" },
];

export const HP_PRESETS = [
  { id:"resnet_modern", name:"ResNet (modern)", details:{ optimizer:"SGD", lr:0.1, momentum:0.9, weightDecay:1e-4, scheduler:"cosine", warmup:5, epochs:200, labelSmoothing:0.1, mixup:0.2, cutmix:0.2, ema:false } },
  { id:"convnext_recipe", name:"ConvNeXt (AdamW)", details:{ optimizer:"AdamW", lr:0.001, weightDecay:0.05, scheduler:"cosine", warmup:20, epochs:300, stochasticDepth:0.1, autoAugment:true, ema:true } },
  { id:"mobile_recipe", name:"MobileNetV3/EfficientNet", details:{ optimizer:"AdamW", lr:0.0015, weightDecay:0.05, scheduler:"cosine_warm_restarts", T0:10, Tmult:2, warmup:5, epochs:350, labelSmoothing:0.1, mixup:0.2, cutmix:0.2, ema:true } },
];

export const DATASETS = [
  { id:'CIFAR10', name:'CIFAR-10', H:32, W:32, C:3, classes:10, desc: '60k 32×32 color images (50k train/10k test), 10 classes. Natural object categories.' },
  { id:'CIFAR100', name:'CIFAR-100', H:32, W:32, C:3, classes:100, desc: '60k 32×32 color images (50k train/10k test), 100 fine-grained classes.' },
  { id:'MNIST', name:'MNIST', H:28, W:28, C:1, classes:10, desc: '70k 28×28 grayscale digit images (60k train/10k test), 10 classes (0–9).' },
  { id:'FashionMNIST', name:'Fashion-MNIST', H:28, W:28, C:1, classes:10, desc: '70k 28×28 grayscale fashion images (60k train/10k test), 10 apparel classes.' },
  { id:'STL10', name:'STL10', H:96, W:96, C:3, classes:10, desc: '13k labeled 96×96 color images (5k train/8k test), 10 classes; extra 100k unlabeled.' },
];
