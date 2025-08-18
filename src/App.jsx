import React, { useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { CodeEditor } from "@/components/ui/code-editor";
import { X, Plus, ArrowUp, ArrowDown, Info, Wand2, Wrench, Layers as LayersIcon, Blocks, Settings2, Download, Link2, AlertTriangle, CheckCircle, Library, Boxes, Box } from "lucide-react";

/**
 * Blocks & Builder — Library
 *
 * Tabs:
 *  - Blocks (presets): famous block architectures from well-known backbones
 *  - Build Block: construct a custom block from layers, see shapes & rough params/FLOPs, and synergy tips
 *  - Hyperparameters: optimizers, schedulers (incl. Warm Restarts), epoch scheduling, regularization/aug recipes
 *
 * Color-coded categories; real names only.
 */

// ---------------- Colors per category ----------------
const CAT_COLORS = {
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
const ColorChip = ({ category }) => <span className={`px-2 py-0.5 rounded-md text-[11px] ${CAT_COLORS[category]?.chip || CAT_COLORS.Meta.chip}`}>{category}</span>;

// ---------------- Layer library (expanded) ----------------
const LAYERS = [
  // Convolutions
  { id: "conv", name: "Conv2d", category: "Convolution", role: "Learnable local features", op: "k×k, stride s, groups g", defaults: { k: 3, s: 1, g: 1, outC: 64 } },
  { id: "pwconv", name: "Pointwise Conv (1×1)", category: "Convolution", role: "Channel mixing", op: "1×1", defaults: { k: 1, s: 1, g: 1, outC: 64 } },
  { id: "dwconv", name: "Depthwise Conv", category: "Convolution", role: "Per-channel spatial conv", op: "k×k, groups=inC", defaults: { k: 3, s: 1 } },
  { id: "grpconv", name: "Grouped Conv", category: "Convolution", role: "Split channels into groups", op: "k×k, groups>1", defaults: { k: 3, s: 1, g: 32, outC: 256 } },
  { id: "dilconv", name: "Dilated Conv", category: "Convolution", role: "Expand receptive field", op: "k×k, dilation d", defaults: { k: 3, s: 1, d: 2, outC: 64 } },
  { id: "deform", name: "Deformable Conv v2", category: "Convolution", role: "Learnable offsets", op: "DCNv2", defaults: { k: 3, s: 1, outC: 64 } },

  // Norms
  { id: "bn", name: "BatchNorm2d", category: "Normalization", role: "Normalize activations", op: "(x−μ)/σ·γ+β" },
  { id: "gn", name: "GroupNorm", category: "Normalization", role: "Batch-size agnostic", op: "Group-wise norm", defaults: { groups: 32 } },
  { id: "ln", name: "LayerNorm", category: "Normalization", role: "Per-sample norm", op: "Across channels" },

  // Activations
  { id: "relu", name: "ReLU", category: "Activation", role: "Non-linearity", op: "max(0,x)" },
  { id: "gelu", name: "GELU", category: "Activation", role: "Smooth activation", op: "x·Φ(x)" },
  { id: "silu", name: "SiLU (Swish)", category: "Activation", role: "Smooth gated", op: "x·sigmoid(x)" },
  { id: "hswish", name: "Hardswish", category: "Activation", role: "Efficient swish", op: "x·ReLU6(x+3)/6" },
  { id: "prelu", name: "PReLU", category: "Activation", role: "Learnable negative slope", op: "ax for x<0" },

  // Pooling / reductions
  { id: "maxpool", name: "MaxPool2d", category: "Pooling", role: "Downsample by max", op: "k×k s" },
  { id: "avgpool", name: "AvgPool2d", category: "Pooling", role: "Downsample by avg", op: "k×k s" },
  { id: "gap", name: "Global Avg Pool", category: "Pooling", role: "H×W → 1×1", op: "AdaptiveAvgPool2d(1)" },

  // Attention & channel/spatial
  { id: "se", name: "Squeeze-and-Excitation", category: "Regularization", role: "Channel attention", op: "GAP→MLP→sigmoid→scale", defaults: { r: 16 } },
  { id: "eca", name: "ECA", category: "Regularization", role: "Efficient channel attention", op: "1D conv over channels" },
  { id: "cbam", name: "CBAM", category: "Regularization", role: "Channel+Spatial attention", op: "SE + spatial mask" },
  { id: "mhsa", name: "MHSA (2D)", category: "Attention", role: "Self-attention", op: "softmax(QK^T/√d)·V" },
  { id: "winattn", name: "Windowed Attention", category: "Attention", role: "Local attention", op: "Swin-style windows" },

  // Structure
  { id: "add", name: "Residual Add", category: "Residual", role: "Skip connection", op: "x + F(x)", defaults: { from: null } },
  { id: "concat", name: "Concatenate", category: "Residual", role: "Channel concat", op: "[x, F(x)]" },

  // Linear head
  { id: "linear", name: "Linear", category: "Linear", role: "Projection", op: "out = xW + b", defaults: { outF: 1000 } },

  // Regularization
  { id: "dropout", name: "Dropout", category: "Regularization", role: "Random feature drop", op: "p" },
  { id: "droppath", name: "Stochastic Depth", category: "Regularization", role: "Randomly drop residual branch", op: "prob p" },
];

// ---------------- Famous preset blocks (Blocks tab) ----------------
const PRESETS = [
  {
    id: "resnet_basic",
    name: "ResNet BasicBlock",
    family: "ResNet-18/34",
    composition: ["conv","bn","relu","conv","bn","add","relu"],
    strengths: ["Simple","Stable"],
    drawbacks: ["Less params-efficient when very deep"],
    goodSlots: ["stage"],
  },
  {
    id: "resnet_bottleneck",
    name: "ResNet Bottleneck",
    family: "ResNet-50/101/152",
    composition: ["pwconv","bn","relu","conv","bn","relu","pwconv","bn","add","relu"],
    strengths: ["Efficient at scale"],
    drawbacks: ["1×1 bandwidth","Memory"],
    goodSlots: ["stage"],
  },
  {
    id: "resnext",
    name: "ResNeXt Bottleneck (cardinality)",
    family: "ResNeXt",
    composition: ["pwconv","bn","relu","grpconv","bn","relu","pwconv","bn","add","relu"],
    strengths: ["Higher accuracy at similar cost"],
    drawbacks: ["Group-conv perf variance"],
    goodSlots: ["stage"],
  },
  {
    id: "preact_bottleneck",
    name: "Pre-activation Bottleneck",
    family: "ResNet v2",
    composition: ["bn","relu","pwconv","bn","relu","conv","bn","relu","pwconv","add"],
    strengths: ["Smoother optimization"],
    drawbacks: ["Layout-only change"],
    goodSlots: ["stage"],
  },
  {
    id: "mbv1",
    name: "MobileNetV1 DW-Separable",
    family: "MobileNetV1",
    composition: ["dwconv","bn","relu","pwconv","bn","relu"],
    strengths: ["Very efficient"],
    drawbacks: ["Depthwise kernel perf sensitivity"],
    goodSlots: ["stage"],
  },
  {
    id: "mbv2",
    name: "MobileNetV2 Inverted Residual",
    family: "MobileNetV2",
    composition: ["pwconv","bn","relu","dwconv","bn","relu","pwconv","bn","add"],
    strengths: ["Edge efficiency"],
    drawbacks: ["Linear bottleneck sensitivity"],
    goodSlots: ["stage"],
  },
  {
    id: "mbv3",
    name: "MobileNetV3 Block (+SE, h-swish)",
    family: "MobileNetV3",
    composition: ["pwconv","bn","relu","dwconv","bn","se","hswish","pwconv","bn","add"],
    strengths: ["Strong mobile accuracy"],
    drawbacks: ["Extra complexity"],
    goodSlots: ["stage"],
  },
  {
    id: "efficient_mbconv",
    name: "EfficientNet MBConv + SE (+SiLU)",
    family: "EfficientNet",
    composition: ["pwconv","bn","silu","dwconv","bn","se","silu","pwconv","bn","add"],
    strengths: ["Accuracy/efficiency balance"],
    drawbacks: ["Training sensitivity"],
    goodSlots: ["stage"],
  },
  {
    id: "convnext",
    name: "ConvNeXt Block",
    family: "ConvNeXt",
    composition: ["dwconv","ln","pwconv","gelu","pwconv","droppath","add"],
    strengths: ["Modern accuracy","Simple"],
    drawbacks: ["DW perf varies"],
    goodSlots: ["stage"],
  },
  {
    id: "densenet_dense",
    name: "DenseNet Dense Block (k growth)",
    family: "DenseNet",
    composition: ["bn","relu","conv","concat"],
    strengths: ["Feature reuse"],
    drawbacks: ["Memory footprint"],
    goodSlots: ["stage"],
  },
  {
    id: "inception_a",
    name: "Inception-v3 Module (A)",
    family: "InceptionV3",
    composition: ["conv","conv","conv","concat","bn","relu"],
    strengths: ["Multi-scale"],
    drawbacks: ["Complex wiring"],
    goodSlots: ["stage"],
  },
  {
    id: "repvgg",
    name: "RepVGG Block (train-time branches)",
    family: "RepVGG",
    composition: ["conv","bn","relu","conv","bn","relu","add"],
    strengths: ["Re-parameterizable to 3×3"],
    drawbacks: ["Reparam step"],
    goodSlots: ["stage"],
  },
  {
    id: "ghost",
    name: "GhostNet Ghost Bottleneck",
    family: "GhostNet",
    composition: ["pwconv","relu","dwconv","pwconv","add"],
    strengths: ["Cheap feature maps"],
    drawbacks: ["Approximation artifacts"],
    goodSlots: ["stage"],
  },
  {
    id: "squeezenet_fire",
    name: "SqueezeNet Fire Module",
    family: "SqueezeNet",
    composition: ["pwconv","relu","conv","conv","concat"],
    strengths: ["Few params"],
    drawbacks: ["Lower accuracy than modern nets"],
    goodSlots: ["stage"],
  },
];

// ---------------- FLOPs/Params estimators (rough) ----------------
const mm = (a,b)=>a*b; const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function convParams(inC,outC,k,groups=1){ return (k*k*inC/groups)*outC; }
function convFLOPs(H,W,inC,outC,k,groups=1){ return H*W*outC*(k*k*inC/groups); }
function dwParams(inC,k){ return inC*k*k; }
function dwFLOPs(H,W,inC,k){ return H*W*inC*k*k; }
function pwParams(inC,outC){ return inC*outC; }
function pwFLOPs(H,W,inC,outC){ return H*W*inC*outC; }
function seParams(C,r=16){ const hidden=Math.max(1,Math.floor(C/r)); return C*hidden + hidden*C; }

// ---------------- Synergy rules (used in builder tips) ----------------
const SYNERGIES = [
  { need:["conv","bn"], why:"BatchNorm after Conv stabilizes stats; enables larger LR.", tag:"stability" },
  { need:["dwconv","pwconv"], why:"Depthwise + pointwise forms an efficient separable conv.", tag:"efficiency" },
  { need:["bn","relu"], why:"BN followed by ReLU is a robust pairing for CNNs.", tag:"stability" },
  { need:["se"], why:"SE adds channel recalibration; often improves accuracy with minor cost.", tag:"accuracy" },
  { need:["droppath","add"], why:"Stochastic depth regularizes residual paths in deep stacks.", tag:"regularization" },
  { need:["mhsa","gelu"], why:"GELU pairs well with attention due to smoother gradients.", tag:"stability" },
];

// ---------------- Hyperparameter recipes ----------------
const HP_PRESETS = [
  { id:"resnet_modern", name:"ResNet (modern)", details:{ optimizer:"SGD", lr:0.1, momentum:0.9, weightDecay:1e-4, scheduler:"cosine", warmup:5, epochs:200, labelSmoothing:0.1, mixup:0.2, cutmix:0.2, ema:false } },
  { id:"convnext_recipe", name:"ConvNeXt (AdamW)", details:{ optimizer:"AdamW", lr:0.001, weightDecay:0.05, scheduler:"cosine", warmup:20, epochs:300, stochasticDepth:0.1, autoAugment:true, ema:true } },
  { id:"mobile_recipe", name:"MobileNetV3/EfficientNet", details:{ optimizer:"AdamW", lr:0.0015, weightDecay:0.05, scheduler:"cosine_warm_restarts", T0:10, Tmult:2, warmup:5, epochs:350, labelSmoothing:0.1, mixup:0.2, cutmix:0.2, ema:true } },
];

// ---------------- Main App ----------------
export default function BlocksBuilderLibrary(){
  const [tab,setTab]=useState("blocks");
  const [code,setCode]=useState("");
  const [mainCode, setMainCode] = useState(
    `from runner.generated_block import GeneratedBlock, CIN, H, W
import torch

model = GeneratedBlock(in_channels=CIN)
x = torch.randn(1, CIN, H, W)
model.eval()
with torch.no_grad():
    y = model(x)

print('Output shape:', tuple(y.shape))`
  );

  // Build Block state
  const [H,setH]=useState(56); const [W,setW]=useState(56); const [Cin,setCin]=useState(64);
  const [block,setBlock]=useState([]); // [{id, cfg}]
  const [selectedIdx,setSelectedIdx]=useState(-1);
  const selected = selectedIdx>=0 ? block[selectedIdx] : null;

  // Saved Blocks library (persisted)
  const [savedBlocks, setSavedBlocks] = useState(()=>{
    try {
      const raw = localStorage.getItem('savedBlocksV1');
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  });
  React.useEffect(()=>{
    try { localStorage.setItem('savedBlocksV1', JSON.stringify(savedBlocks)); } catch {}
  }, [savedBlocks]);

  // Build Model state: sequence of elements: { type: 'layer', id, cfg } | { type: 'block', name, steps: [{id,cfg}, ...] }
  const [model, setModel] = useState([]);
  const [modelSelIdx, setModelSelIdx] = useState(-1);
  const modelSelected = modelSelIdx>=0 ? model[modelSelIdx] : null;

  // Derived: run shape/param/flops propagation + dimension checks
  const stats = useMemo(()=>{
    let h=H,w=W,c=Cin; let params=0, flops=0; const steps=[]; const synergiesHit=new Set();
    const shapesBefore=[]; const shapesAfter=[]; const issues=[];
    block.forEach((step,i)=>{
      const l = LAYERS.find(x=>x.id===step.id);
      let info=""; let outC=c; let inBefore={C:c,H:h,W:w};
      shapesBefore[i]=inBefore;
      if(l.id==="conv"||l.id==="grpconv"||l.id==="dilconv"||l.id==="deform"){
        const k=step.cfg.k||3; const p=Math.floor(k/2); const s=step.cfg.s||1; const g=step.cfg.g||1; const o=step.cfg.outC||c;
        const pcount = convParams(c,o,k, g); const f = convFLOPs(h,w,c,o,k,g);
        params+=pcount; flops+=f; outC=o; h=Math.floor((h + 2*p - k)/s + 1); w=Math.floor((w + 2*p - k)/s + 1); info=`Conv ${k}×${k} s${s} g${g} → C=${o}`;
      } else if(l.id==="dwconv"){
        const k=step.cfg.k||3; const p=Math.floor(k/2); const s=step.cfg.s||1; const pcount = dwParams(c,k); const f = dwFLOPs(h,w,c,k);
        params+=pcount; flops+=f; outC=c; h=Math.floor((h + 2*p - k)/s + 1); w=Math.floor((w + 2*p - k)/s + 1); info=`DW ${k}×${k} s${s}`;
      } else if(l.id==="pwconv"){
        const o=step.cfg.outC||c; const pcount=pwParams(c,o); const f=pwFLOPs(h,w,c,o); params+=pcount; flops+=f; outC=o; info=`PW 1×1 → C=${o}`;
      } else if(l.id==="bn"||l.id==="gn"||l.id==="ln"){
        info=l.name;
      } else if(l.id==="relu"||l.id==="gelu"||l.id==="silu"||l.id==="hswish"||l.id==="prelu"){
        info=l.name;
      } else if(l.id==="maxpool"||l.id==="avgpool"){
        const k=step.cfg?.k||2; const s=step.cfg?.s||2; h=Math.floor((h - k)/s + 1); w=Math.floor((w - k)/s + 1); info=`${l.name} ${k}×${k} s${s}`;
      } else if(l.id==="gap"){
        h=1; w=1; info="GlobalAvgPool";
      } else if(l.id==="se"){
        const add=seParams(c, step.cfg?.r||16); params+=add; info=`SE r=${step.cfg?.r||16}`;
      } else if(l.id==="mhsa"||l.id==="winattn"){
        info=l.id==="mhsa"?"MHSA ~O(N^2)":"Windowed Attention";
      } else if(l.id==="droppath"||l.id==="dropout"){
        info=l.name;
      } else if(l.id==="add"){
        const from = step.cfg?.from;
        if(typeof from!=="number" || from<0 || from>=i){
          issues.push({ type:"add_invalid_from", step:i, msg:"Residual Add needs a valid previous step index." });
          info = "Residual Add (select source)";
        } else {
          const ref = shapesAfter[from];
          const now = {C:c,H:h,W:w};
          const match = ref && ref.C===now.C && ref.H===now.H && ref.W===now.W;
          if(!match){
            issues.push({ type:"add_mismatch", step:i, from, ref, now, msg:`Shape mismatch: from #${from} (${ref?`${ref.C}×${ref.H}×${ref.W}`:"?"}) vs current ${now.C}×${now.H}×${now.W}` });
          }
          const refName = block[from] ? (LAYERS.find(v=>v.id===block[from].id)?.name || block[from].id) : "?";
          info = `Residual Add ← #${from} ${refName}${!match?" (mismatch)":""}`;
        }
      } else if(l.id==="concat"){
        info="Concat (channels++)"; outC = c * 2;
      } else if(l.id==="linear"){
        const o=step.cfg?.outF||1000; params += c*o; info=`Linear ${c}→${o}`; outC=o;
      }
      SYNERGIES.forEach(s=>{ if(s.need.every(t=>[l.id, ...(i>0?[block[i-1].id]:[])].includes(t))) synergiesHit.add(s.tag); });
      steps.push({ i, name:l.name, info, shape:`(${outC}, ${h}, ${w})` });
      c=outC; shapesAfter[i]={C:c,H:h,W:w};
    });
    return { steps, outC:c, H:h, W:w, params, flops, tags:[...synergiesHit], issues, inShapes: shapesBefore, outShapes: shapesAfter };
  },[block,H,W,Cin]);

  // Support: simulate stats for an arbitrary steps array
  function simulateStatsForSteps(stepsArr, H0, W0, C0){
    let h=H0,w=W0,c=C0; let params=0, flops=0; const steps=[]; const synergiesHit=new Set();
    const shapesBefore=[]; const shapesAfter=[]; const issues=[];
    stepsArr.forEach((step,i)=>{
      const l = LAYERS.find(x=>x.id===step.id);
      let info=""; let outC=c; let inBefore={C:c,H:h,W:w};
      shapesBefore[i]=inBefore;
      if(l.id==="conv"||l.id==="grpconv"||l.id==="dilconv"||l.id==="deform"){
        const k=step.cfg.k||3; const p=Math.floor(k/2); const s=step.cfg.s||1; const g=step.cfg.g||1; const o=step.cfg.outC||c;
        const pcount = convParams(c,o,k, g); const f = convFLOPs(h,w,c,o,k,g);
        params+=pcount; flops+=f; outC=o; h=Math.floor((h + 2*p - k)/s + 1); w=Math.floor((w + 2*p - k)/s + 1); info=`Conv ${k}×${k} s${s} g${g} → C=${o}`;
      } else if(l.id==="dwconv"){
        const k=step.cfg.k||3; const p=Math.floor(k/2); const s=step.cfg.s||1; const pcount = dwParams(c,k); const f = dwFLOPs(h,w,c,k);
        params+=pcount; flops+=f; outC=c; h=Math.floor((h + 2*p - k)/s + 1); w=Math.floor((w + 2*p - k)/s + 1); info=`DW ${k}×${k} s${s}`;
      } else if(l.id==="pwconv"){
        const o=step.cfg.outC||c; const pcount=pwParams(c,o); const f=pwFLOPs(h,w,c,o); params+=pcount; flops+=f; outC=o; info=`PW 1×1 → C=${o}`;
      } else if(l.id==="bn"||l.id==="gn"||l.id==="ln"){
        info=l.name;
      } else if(l.id==="relu"||l.id==="gelu"||l.id==="silu"||l.id==="hswish"||l.id==="prelu"){
        info=l.name;
      } else if(l.id==="maxpool"||l.id==="avgpool"){
        const k=step.cfg?.k||2; const s=step.cfg?.s||2; h=Math.floor((h - k)/s + 1); w=Math.floor((w - k)/s + 1); info=`${l.name} ${k}×${k} s${s}`;
      } else if(l.id==="gap"){
        h=1; w=1; info="GlobalAvgPool";
      } else if(l.id==="se"){
        const add=seParams(c, step.cfg?.r||16); params+=add; info=`SE r=${step.cfg?.r||16}`;
      } else if(l.id==="mhsa"||l.id==="winattn"){
        info=l.id==="mhsa"?"MHSA ~O(N^2)":"Windowed Attention";
      } else if(l.id==="droppath"||l.id==="dropout"){
        info=l.name;
      } else if(l.id==="add"){
        const from = step.cfg?.from;
        if(typeof from!=="number" || from<0 || from>=i){
          issues.push({ type:"add_invalid_from", step:i, msg:"Residual Add needs a valid previous step index." });
          info = "Residual Add (select source)";
        } else {
          const ref = shapesAfter[from];
          const now = {C:c,H:h,W:w};
          const match = ref && ref.C===now.C && ref.H===now.H && ref.W===now.W;
          if(!match){
            issues.push({ type:"add_mismatch", step:i, from, ref, now, msg:`Shape mismatch: from #${from} (${ref?`${ref.C}×${ref.H}×${ref.W}`:"?"}) vs current ${now.C}×${now.H}×${now.W}` });
          }
          const refName = stepsArr[from] ? (LAYERS.find(v=>v.id===stepsArr[from].id)?.name || stepsArr[from].id) : "?";
          info = `Residual Add ← #${from} ${refName}${!match?" (mismatch)":""}`;
        }
      } else if(l.id==="concat"){
        info="Concat (channels++)"; outC = c * 2;
      } else if(l.id==="linear"){
        const o=step.cfg?.outF||1000; params += c*o; info=`Linear ${c}→${o}`; outC=o;
      }
      SYNERGIES.forEach(s=>{ if(s.need.every(t=>[l.id, ...(i>0?[stepsArr[i-1].id]:[])].includes(t))) synergiesHit.add(s.tag); });
      steps.push({ i, name:l.name, info, shape:`(${outC}, ${h}, ${w})` });
      c=outC; shapesAfter[i]={C:c,H:h,W:w};
    });
    return { steps, outC:c, H:h, W:w, params, flops, tags:[...synergiesHit], issues, inShapes: shapesBefore, outShapes: shapesAfter };
  }

  // Preset -> Build Block import (auto-wire residual Add)
  const importPreset = (p)=>{
    const built = p.composition.map(id=>({ id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }));
    const wired = autowireResiduals(built, H, W, Cin);
    setBlock(wired); setSelectedIdx(-1); setTab("build");
  };

  // Add preset block into Model
  const addPresetToModel = (p)=>{
    const built = p.composition.map(id=>({ id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }));
    const wired = autowireResiduals(built, H, W, Cin);
    setModel(prev=>{
      const el = { type:'block', name: p.name, steps: wired };
      if (modelSelIdx>=0){ const arr=[...prev]; arr.splice(modelSelIdx+1,0, el); return arr; }
      return [...prev, el];
    });
    setTab('model');
  };

  // Simulate shapes to wire residual connections
  function autowireResiduals(steps, H0, W0, C0){
    let h=H0, w=W0, c=C0;
    const shapesAfter=[]; const shapesBefore=[];
    steps.forEach((s,i)=>{
      const l = LAYERS.find(x=>x.id===s.id);
      shapesBefore[i] = {C:c,H:h,W:w};
      if(l.id==="conv"||l.id==="grpconv"||l.id==="dilconv"||l.id==="deform"){
        const k=s.cfg.k||3; const p=Math.floor(k/2); const st=s.cfg.s||1; const g=s.cfg.g||1; const o=s.cfg.outC||c; c=o; h=Math.floor((h + 2*p - k)/st + 1); w=Math.floor((w + 2*p - k)/st + 1);
      } else if(l.id==="dwconv"){
        const k=s.cfg.k||3; const p=Math.floor(k/2); const st=s.cfg.s||1; h=Math.floor((h + 2*p - k)/st + 1); w=Math.floor((w + 2*p - k)/st + 1);
      } else if(l.id==="pwconv"){
        const o=s.cfg.outC||c; c=o;
      } else if(l.id==="maxpool"||l.id==="avgpool"){
        const k=s.cfg?.k||2; const st=s.cfg?.s||2; h=Math.floor((h - k)/st + 1); w=Math.floor((w - k)/st + 1);
      } else if(l.id==="gap"){
        h=1; w=1;
      } else if(l.id==="concat"){
        c = c * 2;
      } else if(l.id==="linear"){
        // keep c as features for next ops
        c = s.cfg?.outF || c;
      }
      shapesAfter[i] = {C:c,H:h,W:w};
    });
    // assign from index where shapes match pre-add shape
    let curC=C0, curH=H0, curW=W0;
    return steps.map((s,i)=>{
      const l = LAYERS.find(x=>x.id===s.id);
      if(l.id==="add"){
        // find the nearest j<i where shapesAfter[j] matches current (pre-add) shape
        const pre = {C:curC,H:curH,W:curW};
        let from = null;
        for(let j=i-1;j>=0;j--){
          const ref = shapesAfter[j];
          if(ref && ref.C===pre.C && ref.H===pre.H && ref.W===pre.W){ from = j; break; }
        }
        if(from!==null){
          return { ...s, cfg: { ...(s.cfg||{}), from } };
        }
      }
      // advance current shape (same rules as above)
      if(l.id==="conv"||l.id==="grpconv"||l.id==="dilconv"||l.id==="deform"){
        const k=s.cfg.k||3; const p=Math.floor(k/2); const st=s.cfg.s||1; const o=s.cfg.outC||curC; curC=o; curH=Math.floor((curH + 2*p - k)/st + 1); curW=Math.floor((curW + 2*p - k)/st + 1);
      } else if(l.id==="dwconv"){
        const k=s.cfg.k||3; const p=Math.floor(k/2); const st=s.cfg.s||1; curH=Math.floor((curH + 2*p - k)/st + 1); curW=Math.floor((curW + 2*p - k)/st + 1);
      } else if(l.id==="pwconv"){
        const o=s.cfg.outC||curC; curC=o;
      } else if(l.id==="maxpool"||l.id==="avgpool"){
        const k=s.cfg?.k||2; const st=s.cfg?.s||2; curH=Math.floor((curH - k)/st + 1); curW=Math.floor((curW - k)/st + 1);
      } else if(l.id==="gap"){
        curH=1; curW=1;
      } else if(l.id==="concat"){
        curC = curC * 2;
      } else if(l.id==="linear"){
        curC = s.cfg?.outF || curC;
      }
      return s;
    });
  }

  // Controls
  const addLayer = (id)=> setBlock(prev=>[...prev,{ id, cfg:{ ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }]);
  const removeIdx = (i)=> setBlock(prev=>prev.filter((_,j)=>j!==i));
  const moveIdx = (i,dir)=> setBlock(prev=>{ const arr=[...prev]; const j=i+dir; if(j<0||j>=arr.length) return arr; const t=arr[i]; arr[i]=arr[j]; arr[j]=t; return arr; });

  // Save current Block into library
  const [saveName, setSaveName] = useState("");
  const saveCurrentBlock = ()=>{
    const name = saveName?.trim() || `Custom Block #${savedBlocks.length+1}`;
    if(block.length===0) return;
    const steps = block.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } }));
    setSavedBlocks(prev=>[...prev, { id: `${Date.now()}_${Math.random().toString(36).slice(2,7)}`, name, steps }]);
    setSaveName("");
  };

  // Model controls
  const addModelLayer = (id)=> setModel(prev=>{
    const el = { type:'layer', id, cfg:{ ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } };
    if (modelSelIdx>=0){ const arr=[...prev]; arr.splice(modelSelIdx+1,0, el); return arr; }
    return [...prev, el];
  });
  const addSavedBlockToModel = (blk)=> setModel(prev=>{
    const el = { type:'block', name: blk.name, steps: blk.steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } })) };
    if (modelSelIdx>=0){ const arr=[...prev]; arr.splice(modelSelIdx+1,0, el); return arr; }
    return [...prev, el];
  });
  const removeModelIdx = (i)=> setModel(prev=>prev.filter((_,j)=>j!==i));
  const moveModelIdx = (i,dir)=> setModel(prev=>{ const arr=[...prev]; const j=i+dir; if(j<0||j>=arr.length) return arr; const t=arr[i]; arr[i]=arr[j]; arr[j]=t; return arr; });
  const expandBlockAt = (i)=> setModel(prev=>{
    const arr=[...prev]; const el=arr[i]; if(!el || el.type!=='block') return arr;
    // splice block steps as individual layer elements
    const layers = el.steps.map(s=>({ type:'layer', id:s.id, cfg:{ ...(s.cfg||{}) } }));
    arr.splice(i,1, ...layers);
    return arr;
  });

  // Flatten model into steps and adjust residual 'from' to global indices
  function flattenModelWithFromAdjust(modelEls){
    const out=[]; let base=0;
    for(const el of modelEls){
      if(el.type==='layer'){
        const s = { id: el.id, cfg: { ...(el.cfg||{}) } };
        out.push(s); base += 1;
      } else if(el.type==='block'){
        el.steps.forEach((step)=>{
          const s = { id: step.id, cfg: { ...(step.cfg||{}) } };
          if(s.id==='add' && typeof s.cfg.from==='number'){
            s.cfg = { ...s.cfg, from: (base + s.cfg.from) };
          }
          out.push(s);
        });
        base += el.steps.length;
      }
    }
    return out;
  }

  // Stats for Model (flattened)
  const modelStats = useMemo(()=>{
    const steps = flattenModelWithFromAdjust(model);
    return simulateStatsForSteps(steps, H, W, Cin);
  }, [model, H, W, Cin]);

  // helper: flattened index for a given model index
  const flattenedIndexForModelIdx = (idx)=>{
    let base=0;
    for(let i=0;i<idx;i++){
      const el=model[i];
      base += (el.type==='layer') ? 1 : (el.steps?.length||0);
    }
    return base;
  };

  // Hyperparameters tab
  const [hp,setHp]=useState({ optimizer:"SGD", lr:0.1, momentum:0.9, weightDecay:1e-4, scheduler:"cosine", warmup:5, epochs:200, labelSmoothing:0.1, mixup:0.0, cutmix:0.0, stochasticDepth:0.0, ema:false, cosineRestarts:false, T0:10, Tmult:2 });
  const applyPreset = (p)=>{ setHp(prev=>({ ...prev, ...p.details, cosineRestarts: p.details.scheduler==="cosine_warm_restarts" })); };

  const exportJSON = ()=>{
    const payload = { 
      block: block.map(s=>({ id:s.id, cfg:s.cfg })),
      model: model.map(el=> el.type==='layer' ? ({ type:'layer', id:el.id, cfg:el.cfg }) : ({ type:'block', name:el.name, steps: el.steps })),
      input: { H,W,Cin }, stats, modelStats, hyperparams: hp };
    const blob = new Blob([JSON.stringify(payload,null,2)], { type:"application/json" });
    const url = URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='block_builder_export.json'; a.click(); URL.revokeObjectURL(url);
  };

  // keep code preview in sync when switching to code tab or when block changes
  React.useEffect(()=>{
    if(tab==='code'){
      setCode(generateTorchAll(block, model, H, W, Cin));
    }
  }, [tab, block, model, H, W, Cin]);

  return (
    <div className="w-full h-screen bg-neutral-50 flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b bg-white flex items-center gap-3 sticky top-0 z-10">
        <LayersIcon className="w-5 h-5"/>
        <div className="text-lg font-semibold">Blocks & Builder — Library</div>
        <Badge variant="secondary">Presets • Build • Hyperparameters</Badge>
        <div className="ml-auto"><Button variant="outline" onClick={exportJSON}><Download className="w-4 h-4 mr-2"/>Export</Button></div>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-3 p-3">
        <div className="col-span-3 min-w-[280px] flex flex-col gap-3">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Info className="w-4 h-4"/>Input / Builder Settings</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid grid-cols-3 gap-2">
                <div><div className="text-xs">H</div><Input type="number" value={H} onChange={e=>setH(parseInt(e.target.value||"56",10))}/></div>
                <div><div className="text-xs">W</div><Input type="number" value={W} onChange={e=>setW(parseInt(e.target.value||"56",10))}/></div>
                <div><div className="text-xs">Channels</div><Input type="number" value={Cin} onChange={e=>setCin(parseInt(e.target.value||"64",10))}/></div>
              </div>
              <div className="text-xs text-neutral-600">These sizes drive rough params/FLOPs shown in Build tab.</div>
            </CardContent>
          </Card>

          <Palette addLayer={addLayer} />
        </div>

        <div className="col-span-9">
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList>
              <TabsTrigger value="blocks"><Blocks className="w-4 h-4 mr-1"/>Blocks</TabsTrigger>
              <TabsTrigger value="build"><Wrench className="w-4 h-4 mr-1"/>Build Block</TabsTrigger>
              <TabsTrigger value="model"><Boxes className="w-4 h-4 mr-1"/>Build Model</TabsTrigger>
              <TabsTrigger value="hparams"><Settings2 className="w-4 h-4 mr-1"/>Hyperparameters</TabsTrigger>
              <TabsTrigger value="code"><Wrench className="w-4 h-4 mr-1"/>PyTorch</TabsTrigger>
            </TabsList>

            {/* Blocks tab: famous presets */}
            <TabsContent value="blocks">
              <Card>
                <CardHeader><CardTitle>Preset Blocks from Famous Architectures</CardTitle></CardHeader>
                <CardContent className="grid grid-cols-2 gap-3">
                  {PRESETS.map(p=> (
                    <div key={p.id} className="border border-neutral-200 rounded-2xl p-3 bg-white/90 backdrop-blur-sm shadow-sm hover:shadow-md transition">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{p.name}</div>
                          <div className="text-xs text-neutral-600">{p.family}</div>
                        </div>
                        <div className="flex gap-2">
                          <Button size="sm" variant="secondary" onClick={()=>importPreset(p)}><Wand2 className="w-4 h-4 mr-1"/>Build</Button>
                          <Button size="sm" variant="outline" onClick={()=>addPresetToModel(p)} title="Append this preset to the Model tab">To Model</Button>
                        </div>
                      </div>
                      <div className="text-xs mt-1"><b>Composition:</b> {p.composition.map(id=>LAYERS.find(l=>l.id===id)?.name||id).join(" → ")}</div>
                      <div className="text-[11px]"><b>Strengths:</b> {p.strengths.join(", ")}</div>
                      <div className="text-[11px]"><b>Drawbacks:</b> {p.drawbacks.join(", ")}</div>
                      <div className="text-[11px]"><b>Good slots:</b> {p.goodSlots.join(", ")}</div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Build tab: construct block */}
            <TabsContent value="build">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  <Card>
                    <CardHeader><CardTitle>Current Block</CardTitle></CardHeader>
                    <CardContent className="space-y-2">
                      {block.length===0 && <div className="text-neutral-600 text-sm">Add layers from the left palette or import a preset.</div>}
            {block.map((s,i)=>{
                        const l=LAYERS.find(x=>x.id===s.id);
                        const addIssue = stats.issues.find(it => it.step===i && (it.type==="add_invalid_from" || it.type==="add_mismatch"));
                        return (
              <div key={i} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${CAT_COLORS[l.category]?.ring||''}`}>
                            <div>
                              <div className="text-sm font-medium flex items-center gap-2">{l.name}<ColorChip category={l.category}/></div>
                              <div className="text-xs text-neutral-600">{renderStepSummary(l,s)}</div>
                              {l.id==="add" && (
                                <div className="text-[11px] mt-0.5 flex items-center gap-2">
                                  <Link2 className="w-3.5 h-3.5"/>
                                  {typeof s.cfg?.from === 'number' ? (
                                    <>
                                      <span>From: #{s.cfg.from} {block[s.cfg.from] ? (LAYERS.find(v=>v.id===block[s.cfg.from].id)?.name || block[s.cfg.from].id) : '?'}</span>
                                      {addIssue ? (
                                        <span className="inline-flex items-center gap-1 text-rose-700"><AlertTriangle className="w-3.5 h-3.5"/>Mismatch</span>
                                      ) : (
                                        <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle className="w-3.5 h-3.5"/>OK</span>
                                      )}
                                      <Button size="sm" variant="outline" onClick={()=>setSelectedIdx(s.cfg.from)}>Go</Button>
                                    </>
                                  ) : (
                                    <span className="text-neutral-500">Select a source in the inspector</span>
                                  )}
                                </div>
                              )}
                            </div>
                            <div className="flex items-center gap-1">
                              <Button size="icon" variant="ghost" onClick={()=>moveIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>moveIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>{ setSelectedIdx(i); }}><Wrench className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>removeIdx(i)}><X className="w-4 h-4"/></Button>
                            </div>
                          </div>
                        );
                      })}
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Synergy Tips</CardTitle></CardHeader>
                    <CardContent className="flex flex-wrap gap-2 text-xs">
                      {SYNERGIES.filter(s=>s.need.every(id=>block.some(b=>b.id===id))).map((s,i)=>(
                        <span key={i} className="px-2 py-1 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-200">{s.why}</span>
                      ))}
                      {SYNERGIES.filter(s=>!s.need.every(id=>block.some(b=>b.id===id)) && s.need.some(id=>block.some(b=>b.id===id))).map((s,i)=>(
                        <span key={i} className="px-2 py-1 rounded-md bg-neutral-50 text-neutral-700 border">Consider: {s.need.map(id=>LAYERS.find(l=>l.id===id)?.name||id).join(" + ")} — {s.why}</span>
                      ))}
                      {block.length===0 && <div className="text-neutral-600">Add layers to see synergy suggestions.</div>}
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Save Block</CardTitle></CardHeader>
                    <CardContent className="flex items-center gap-2 text-sm">
                      <Input placeholder="Name (optional)" value={saveName} onChange={(e)=>setSaveName(e.target.value)} />
                      <Button onClick={saveCurrentBlock} disabled={block.length===0}><Library className="w-4 h-4 mr-1"/>Save</Button>
                    </CardContent>
                  </Card>
                </div>

                <div className="col-span-2">
                  <Card>
                    <CardHeader><CardTitle>Inspector & Config</CardTitle></CardHeader>
                    <CardContent className="text-sm space-y-3">
                      {selected ? (
                        <LayerConfig selected={selected} selectedIdx={selectedIdx} block={block} stats={stats} onChange={(cfg)=>{ setBlock(prev=>{ const arr=[...prev]; arr[selectedIdx]={...arr[selectedIdx], cfg}; return arr; }); }} />
                      ) : (
                        <div className="text-neutral-600">Select a step (wrench) to edit its parameters.</div>
                      )}
                      <div className="border border-neutral-200 rounded-xl p-2 text-xs bg-white/90 backdrop-blur-sm shadow-sm">
                        <div className="font-medium mb-1">Rough Totals</div>
                        <div>Output shape: ({stats.outC}, {stats.H}, {stats.W})</div>
                        <div>Params: {(stats.params/1e6).toFixed(3)} M (rough)</div>
                        <div>FLOPs: {(stats.flops/1e9).toFixed(3)} GFLOPs @ {H}×{W}</div>
                        {stats.tags.length>0 && <div className="mt-1">Tags: {stats.tags.join(", ")}</div>}
                        <div className="mt-1">Dimensions: {stats.issues.length===0 ? <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle className="w-3.5 h-3.5"/>All good</span> : <span className="inline-flex items-center gap-1 text-rose-700"><AlertTriangle className="w-3.5 h-3.5"/>{stats.issues.length} issue(s)</span>}</div>
                      </div>

                      <Card className="mt-2">
                        <CardHeader><CardTitle>Dimension Checker</CardTitle></CardHeader>
                        <CardContent className="space-y-2 text-xs">
                          {stats.issues.length===0 && <div className="text-emerald-700 flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5"/>No dimensional conflicts detected.</div>}
                          {stats.issues.map((it,idx)=> (
                            <div key={idx} className="border border-neutral-200 rounded-md p-2 bg-white/90 backdrop-blur-sm shadow-sm">
                              <div className="font-medium flex items-center gap-2">
                                {it.type==="add_mismatch" || it.type==="add_invalid_from" ? <AlertTriangle className="w-3.5 h-3.5 text-rose-700"/> : null}
                                Step #{it.step}: {it.msg}
                              </div>
                              {it.type==="add_mismatch" && it.ref && (
                                <div className="mt-1 text-[11px] text-neutral-700">Suggestion: insert a <b>1×1 Conv</b> (projection) or adjust <b>stride</b>/padding so both paths yield the same (C,H,W).</div>
                              )}
                            </div>
                          ))}
                        </CardContent>
                      </Card>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* Build Model tab */}
            <TabsContent value="model">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  <Card>
                    <CardHeader><CardTitle>Current Model</CardTitle></CardHeader>
                    <CardContent className="space-y-2">
                      <div className="border border-dashed border-neutral-300 rounded-xl p-2 bg-white/60 text-xs text-neutral-700">Input • ({Cin}, {H}, {W}) — mandatory</div>
                      {model.length===0 && <div className="text-neutral-600 text-sm">Add saved blocks, presets, or layers from the right panel.</div>}
                      {model.map((el,i)=>{
                        if(el.type==='layer'){
                          const l=LAYERS.find(x=>x.id===el.id);
                          return (
                            <div key={`L${i}`} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${CAT_COLORS[l.category]?.ring||''}`}>
                              <div>
                                <div className="text-sm font-medium flex items-center gap-2">{l.name}<ColorChip category={l.category}/></div>
                                <div className="text-xs text-neutral-600">{renderStepSummary(l, el)}</div>
                              </div>
                              <div className="flex items-center gap-1">
                                <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                                <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                                <Button size="icon" variant="ghost" onClick={()=>{ setModelSelIdx(i); }}><Wrench className="w-4 h-4"/></Button>
                                <Button size="icon" variant="ghost" onClick={()=>removeModelIdx(i)}><X className="w-4 h-4"/></Button>
                              </div>
                            </div>
                          );
                        }
                        // block element
                        return (
                          <div key={`B${i}`} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition`}>
                            <div>
                              <div className="text-sm font-medium flex items-center gap-2"><Box className="w-4 h-4"/>{el.name}<span className="px-2 py-0.5 rounded-md text-[11px] bg-neutral-200 text-neutral-800">Block</span></div>
                              <div className="text-xs text-neutral-600">{el.steps.map(s=>LAYERS.find(x=>x.id===s.id)?.name||s.id).join(' → ')}</div>
                            </div>
                            <div className="flex items-center gap-1">
                              <Button size="sm" variant="outline" onClick={()=>expandBlockAt(i)} title="Expand this block into its layers so you can edit or insert between">Expand</Button>
                              <Button size="icon" variant="ghost" onClick={()=>{ setModelSelIdx(i); }}><Wrench className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>removeModelIdx(i)}><X className="w-4 h-4"/></Button>
                            </div>
                          </div>
                        );
                      })}
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Rough Totals</CardTitle></CardHeader>
                    <CardContent className="text-xs space-y-1">
                      <div>Output shape: ({modelStats.outC}, {modelStats.H}, {modelStats.W})</div>
                      <div>Params: {(modelStats.params/1e6).toFixed(3)} M (rough)</div>
                      <div>FLOPs: {(modelStats.flops/1e9).toFixed(3)} GFLOPs @ {H}×{W}</div>
                      <div>Issues: {modelStats.issues.length===0 ? <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle className="w-3.5 h-3.5"/>None</span> : <span className="inline-flex items-center gap-1 text-rose-700"><AlertTriangle className="w-3.5 h-3.5"/>{modelStats.issues.length}</span>}</div>
                    </CardContent>
                  </Card>
                </div>

                <div className="col-span-2">
                  <Card>
                    <CardHeader><CardTitle>Model Builder</CardTitle></CardHeader>
                    <CardContent className="space-y-3 text-sm">
                      <div className="text-xs text-neutral-600">Add blocks or layers. Input is implicit from the left settings.</div>
                      <div className="border rounded-xl p-2">
                        <div className="font-medium text-sm mb-1 flex items-center gap-2"><Library className="w-4 h-4"/>Saved Blocks</div>
                        {savedBlocks.length===0 ? (
                          <div className="text-xs text-neutral-600">Save blocks from the Build Block tab to appear here.</div>
                        ) : (
                          <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                            {savedBlocks.map(sb=> (
                              <div key={sb.id} className="border border-neutral-200 rounded-md p-2 bg-white/90 flex items-center justify-between">
                                <div>
                                  <div className="text-sm font-medium">{sb.name}</div>
                                  <div className="text-[11px] text-neutral-600 truncate">{sb.steps.map(s=>LAYERS.find(x=>x.id===s.id)?.name||s.id).join(' → ')}</div>
                                </div>
                                <Button size="sm" variant="outline" onClick={()=>addSavedBlockToModel(sb)}>Add</Button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      <div className="border rounded-xl p-2">
                        <div className="font-medium text-sm mb-1 flex items-center gap-2"><Blocks className="w-4 h-4"/>Preset Blocks</div>
                        <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                          {PRESETS.map(p=> (
                            <div key={p.id} className="border border-neutral-200 rounded-md p-2 bg-white/90 flex items-center justify-between">
                              <div>
                                <div className="text-sm font-medium">{p.name}</div>
                                <div className="text-[11px] text-neutral-600 truncate">{p.family}</div>
                              </div>
                              <Button size="sm" variant="outline" onClick={()=>addPresetToModel(p)}>Add</Button>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="border rounded-xl p-2">
                        <div className="font-medium text-sm mb-1 flex items-center gap-2"><LayersIcon className="w-4 h-4"/>Add Single Layer</div>
                        <div className="grid grid-cols-1 gap-2 max-h-[24vh] overflow-auto pr-1">
                          {LAYERS.map(l=> (
                            <div key={l.id} className="flex items-center justify-between border border-neutral-200 rounded-md p-2 bg-white/90">
                              <div className="text-sm flex items-center gap-2">{l.name}<ColorChip category={l.category}/></div>
                              <Button size="sm" variant="outline" onClick={()=>addModelLayer(l.id)}><Plus className="w-4 h-4 mr-1"/>Add</Button>
                            </div>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Inspector & Config</CardTitle></CardHeader>
                    <CardContent className="text-sm">
                      {modelSelected ? (
                        modelSelected.type==='layer' ? (
                          <LayerConfig selected={modelSelected} selectedIdx={flattenedIndexForModelIdx(modelSelIdx)} block={flattenModelWithFromAdjust(model).map(s=>s)} stats={modelStats} onChange={(cfg)=>{ setModel(prev=>{ const arr=[...prev]; arr[modelSelIdx]={...arr[modelSelIdx], cfg}; return arr; }); }} />
                        ) : (
                          <div className="text-xs text-neutral-700">
                            <div className="font-medium mb-1">Block: {modelSelected.name}</div>
                            <div>Steps: {modelSelected.steps.map(s=>LAYERS.find(x=>x.id===s.id)?.name||s.id).join(' → ')}</div>
                            <div className="mt-2">Blocks are treated as black boxes. Use "Expand" in the list to edit layers inline.</div>
                          </div>
                        )
                      ) : (
                        <div className="text-neutral-600 text-sm">Select a layer to edit, or expand a block to edit its layers inline.</div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* Hyperparameters tab */}
            <TabsContent value="hparams">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  <Card>
                    <CardHeader><CardTitle>Training Hyperparameters</CardTitle></CardHeader>
                    <CardContent className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <div className="text-xs">Optimizer</div>
                        <Select value={hp.optimizer} onValueChange={(v)=>setHp({...hp, optimizer:v})}>
                          <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="SGD">SGD (momentum)</SelectItem>
                            <SelectItem value="AdamW">AdamW</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <div className="text-xs">Learning rate</div>
                        <Input type="number" value={hp.lr} onChange={(e)=>setHp({...hp, lr:parseFloat(e.target.value||"0")})}/>
                      </div>
                      {hp.optimizer==="SGD" && (
                        <>
                          <div>
                            <div className="text-xs">Momentum</div>
                            <Input type="number" value={hp.momentum} onChange={(e)=>setHp({...hp, momentum:parseFloat(e.target.value||"0.9")})}/>
                          </div>
                        </>
                      )}
                      <div>
                        <div className="text-xs">Weight decay</div>
                        <Input type="number" value={hp.weightDecay} onChange={(e)=>setHp({...hp, weightDecay:parseFloat(e.target.value||"0")})}/>
                      </div>
                      <div>
                        <div className="text-xs">Scheduler</div>
                        <Select value={hp.scheduler} onValueChange={(v)=>setHp({...hp, scheduler:v, cosineRestarts: v==="cosine_warm_restarts" })}>
                          <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="cosine">Cosine Annealing</SelectItem>
                            <SelectItem value="cosine_warm_restarts">Cosine with Warm Restarts (SGDR)</SelectItem>
                            <SelectItem value="step">Step Decay</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      {hp.scheduler==="cosine_warm_restarts" && (
                        <>
                          <div>
                            <div className="text-xs">T_0</div>
                            <Input type="number" value={hp.T0} onChange={(e)=>setHp({...hp, T0:parseInt(e.target.value||"10",10)})}/>
                          </div>
                          <div>
                            <div className="text-xs">T_mult</div>
                            <Input type="number" value={hp.Tmult} onChange={(e)=>setHp({...hp, Tmult:parseInt(e.target.value||"2",10)})}/>
                          </div>
                        </>
                      )}
                      <div>
                        <div className="text-xs">Warmup (epochs)</div>
                        <Input type="number" value={hp.warmup} onChange={(e)=>setHp({...hp, warmup:parseInt(e.target.value||"0",10)})}/>
                      </div>
                      <div>
                        <div className="text-xs">Total epochs</div>
                        <Input type="number" value={hp.epochs} onChange={(e)=>setHp({...hp, epochs:parseInt(e.target.value||"0",10)})}/>
                      </div>
                      <div>
                        <div className="text-xs">Label smoothing</div>
                        <Input type="number" value={hp.labelSmoothing} onChange={(e)=>setHp({...hp, labelSmoothing:parseFloat(e.target.value||"0")})}/>
                      </div>
                      <div>
                        <div className="text-xs">Mixup</div>
                        <Input type="number" value={hp.mixup} onChange={(e)=>setHp({...hp, mixup:parseFloat(e.target.value||"0")})}/>
                      </div>
                      <div>
                        <div className="text-xs">CutMix</div>
                        <Input type="number" value={hp.cutmix} onChange={(e)=>setHp({...hp, cutmix:parseFloat(e.target.value||"0")})}/>
                      </div>
                      <div>
                        <div className="text-xs">Stochastic Depth</div>
                        <Slider value={[hp.stochasticDepth]} min={0} max={0.3} step={0.01} onValueChange={([v])=>setHp({...hp, stochasticDepth:v})}/>
                        <div className="text-xs mt-1">{hp.stochasticDepth.toFixed(2)}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch checked={hp.ema} onCheckedChange={(v)=>setHp({...hp, ema:v})}/> <span>EMA</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Presets</CardTitle></CardHeader>
                    <CardContent className="grid grid-cols-2 gap-3">
                      {HP_PRESETS.map(p=> (
                        <div key={p.id} className="border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm shadow-sm hover:shadow-md transition">
                          <div className="flex items-center justify-between">
                            <div className="font-medium text-sm">{p.name}</div>
                            <Button size="sm" variant="secondary" onClick={()=>applyPreset(p)}>Apply</Button>
                          </div>
                          <div className="text-[11px] text-neutral-600 mt-1">{Object.entries(p.details).map(([k,v])=>`${k}:${v}`).join(" · ")}</div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                </div>

                <div className="col-span-2">
                  <Card>
                    <CardHeader><CardTitle>Export Preview</CardTitle></CardHeader>
                    <CardContent>
                      <textarea className="w-full h-[52vh] text-xs font-mono p-2 border border-neutral-200 rounded-xl bg-white/90 shadow-inner" readOnly value={JSON.stringify({ block: block.map(s=>({id:s.id,cfg:s.cfg})), input:{H,W,Cin}, stats, hyperparams: hp }, null, 2)} />
                      <div className="text-xs text-neutral-500 mt-1">Copy or use Export to download JSON.</div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* PyTorch preview tab */}
            <TabsContent value="code">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  <Card>
                    <CardHeader><CardTitle>GeneratedBlock (auto-generated)</CardTitle></CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-2 mb-2">
                        <Button variant="secondary" onClick={()=>copyText(code)}>Copy</Button>
                        <Button variant="outline" onClick={()=>downloadText('generated_block.py', code)}>Download .py</Button>
                        <Button onClick={()=>saveGenerated(code)} variant="default">Save to runner/generated_block.py</Button>
                        <Button variant="secondary" onClick={()=>runPython(code, mainCode)} title="Runs in a uv-managed venv on CPU">Run (CPU)</Button>
                      </div>
                      <CodeEditor language="python" value={code} onChange={setCode} className="h-[30vh]"/>
                      <div className="text-xs text-neutral-500 mt-1">This file now emits CIN/H/W, a GeneratedBlock for the Build tab, and if a Model is defined, a GeneratedModel class that flattens blocks and layers.</div>
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>main.py (optional entrypoint)</CardTitle></CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-2 mb-2">
                        <Button variant="secondary" onClick={()=>copyText(mainCode)}>Copy</Button>
                        <Button variant="outline" onClick={()=>downloadText('main.py', mainCode)}>Download main.py</Button>
                        <Button onClick={()=>saveMain(mainCode)} variant="default">Save to .runner/main.py</Button>
                      </div>
                      <CodeEditor language="python" value={mainCode} onChange={setMainCode} className="h-[26vh]"/>
                      <div className="text-xs text-neutral-500 mt-1">Tip: explain or script your own workflow here. When present, the runner executes this script.</div>
                    </CardContent>
                  </Card>
                </div>

                <div className="col-span-2">
                  <Card>
                    <CardHeader><CardTitle>Run Output</CardTitle></CardHeader>
                    <CardContent className="text-xs space-y-2">
                      <AnsiLog url="/api/run-output" />
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Notes</CardTitle></CardHeader>
                    <CardContent className="text-xs space-y-2">
                      <div>• Residual Add uses intermediate outputs by index. Ensure shapes match.</div>
                      <div>• Concat and Deformable Conv are annotated as TODO in the code.</div>
                      <div>• Linear flattens spatial dims automatically (torch.flatten(x, 1)).</div>
                      <div>• SE is emitted as a small helper module when used.</div>
                      <div>• GeneratedModel adjusts residual indices within saved blocks automatically.</div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Legend */}
      <div className="px-4 py-2 border-t bg-white text-xs text-neutral-600 flex flex-wrap gap-2 items-center">
        <span className="font-medium mr-1">Categories:</span>
        {Object.entries(CAT_COLORS).map(([k,v])=> (<span key={k} className={`px-2 py-0.5 rounded-md ${v.chip}`}>{k}</span>))}
      </div>
    </div>
  );
}
// save generated code into runner/generated_block.py (for the runner)
function saveGenerated(text){
  return fetch('/api/save-generated', { method: 'PUT', body: text })
}

function saveMain(text){
  return fetch('/api/save-main', { method: 'PUT', body: text })
}

// kick off a background run via npm scripts (requires terminal)
async function runPython(text, main){
  try {
    if (typeof text === 'string' && text.length > 0) {
      await saveGenerated(text)
    }
    if (typeof main === 'string' && main.length > 0) {
      await saveMain(main)
    }
  } catch {}
  fetch('/api/run-python', { method: 'POST' }).catch(()=>{})
}

// Polls ANSI-colored output file and renders colorized lines
function AnsiLog({ url }){
  const [text,setText]=React.useState('')
  React.useEffect(()=>{
    let alive=true
    const tick=()=>{
      fetch(url, { cache: 'no-store' })
        .then(r=>r.text())
        .then(t=>{ if(alive) setText(t) })
        .catch(()=>{})
    }
    tick()
    const id = setInterval(tick, 1000)
    return ()=>{ alive=false; clearInterval(id) }
  },[url])
  return (
    <pre className="h-[40vh] overflow-auto rounded-xl border border-neutral-200 bg-black text-white p-2 text-xs">
      <code dangerouslySetInnerHTML={{ __html: ansiToHtml(text) }} />
    </pre>
  )
}

// minimal ANSI to HTML coloring
function ansiToHtml(s){
  // escape HTML first, then inject spans for ANSI codes
  const esc = (x)=> x
    .replaceAll(/&/g,'&amp;')
    .replaceAll(/</g,'&lt;')
    .replaceAll(/>/g,'&gt;')
  let t = esc(String(s ?? ''))
  t = t
    .replaceAll(/\x1b\[31m/g, '<span style="color:#f87171">')
    .replaceAll(/\x1b\[32m/g, '<span style="color:#34d399">')
    .replaceAll(/\x1b\[33m/g, '<span style="color:#fbbf24">')
    .replaceAll(/\x1b\[0m/g, '</span>')
  return t
}

// ---------------- Palette ----------------
function Palette({ addLayer }){
  const [cat,setCat]=useState("All");
  const [q,setQ]=useState("");
  const list = useMemo(()=> LAYERS.filter(l=> (cat==="All"||l.category===cat) && (q==="" || [l.name,l.role,l.op].join(" ").toLowerCase().includes(q.toLowerCase())) ), [cat,q]);
  return (
    <Card className="flex-1">
      <CardHeader><CardTitle>Layer Palette</CardTitle></CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="grid grid-cols-2 gap-2">
          <Select value={cat} onValueChange={setCat}>
            <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
            <SelectContent>
              {['All',...new Set(LAYERS.map(l=>l.category))].map(c=> <SelectItem key={c} value={c}>{c}</SelectItem>)}
            </SelectContent>
          </Select>
          <Input placeholder="Search layers" value={q} onChange={(e)=>setQ(e.target.value)}/>
        </div>
    <div className="max-h-[50vh] overflow-auto space-y-2 pr-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-neutral-100 [&::-webkit-scrollbar-thumb]:bg-neutral-300 [&::-webkit-scrollbar-thumb]:rounded">
          {list.map(l=> (
      <div key={l.id} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${CAT_COLORS[l.category]?.ring||''}`}>
              <div>
                <div className="font-medium text-sm flex items-center gap-2">{l.name}<ColorChip category={l.category}/></div>
                <div className="text-[11px] text-neutral-600">{l.role}</div>
              </div>
              <Button size="sm" variant="outline" onClick={()=>addLayer(l.id)}><Plus className="w-4 h-4 mr-1"/>Add</Button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------- Layer Config ----------------
function LayerConfig({ selected, onChange, selectedIdx, block, stats }){
  const l = LAYERS.find(x=>x.id===selected.id);
  const cfg = { ...(l.defaults||{}), ...(selected.cfg||{}) };
  const inS = stats?.inShapes?.[selectedIdx];
  const outS = stats?.outShapes?.[selectedIdx];
  const set = (k,v)=> onChange({ ...cfg, [k]:v });
  return (
    <div className="space-y-2">
      <div className="text-base font-semibold flex items-center gap-2">{l.name} <ColorChip category={l.category}/></div>
      <div className="border rounded-xl p-2 text-xs bg-white">
        <div className="font-medium mb-1">Dimensions</div>
        <div>Input: ({inS?.C ?? "?"}, {inS?.H ?? "?"}, {inS?.W ?? "?"})</div>
        <div>Output: ({outS?.C ?? "?"}, {outS?.H ?? "?"}, {outS?.W ?? "?"})</div>
      </div>
      {l.category==="Convolution" && l.id!=="dwconv" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Out channels</div><Input type="number" value={cfg.outC||64} onChange={e=>set('outC',parseInt(e.target.value||"64",10))}/></div>
          <div><div className="text-xs">Kernel</div><Input type="number" value={cfg.k||3} onChange={e=>set('k',parseInt(e.target.value||"3",10))}/></div>
          <div><div className="text-xs">Stride</div><Input type="number" value={cfg.s||1} onChange={e=>set('s',parseInt(e.target.value||"1",10))}/></div>
          {l.id==="grpconv" && <div><div className="text-xs">Groups</div><Input type="number" value={cfg.g||32} onChange={e=>set('g',parseInt(e.target.value||"32",10))}/></div>}
          {l.id==="dilconv" && <div><div className="text-xs">Dilation</div><Input type="number" value={cfg.d||2} onChange={e=>set('d',parseInt(e.target.value||"2",10))}/></div>}
        </div>
      )}
      {l.id==="dwconv" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Kernel</div><Input type="number" value={cfg.k||3} onChange={e=>set('k',parseInt(e.target.value||"3",10))}/></div>
          <div><div className="text-xs">Stride</div><Input type="number" value={cfg.s||1} onChange={e=>set('s',parseInt(e.target.value||"1",10))}/></div>
          <div className="col-span-2 text-[11px] text-neutral-600">Depthwise uses groups=inC and keeps channels.</div>
        </div>
      )}
      {l.id==="pwconv" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Out channels</div><Input type="number" value={cfg.outC||64} onChange={e=>set('outC',parseInt(e.target.value||"64",10))}/></div>
        </div>
      )}
      {l.id==="se" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Reduction r</div><Input type="number" value={cfg.r||16} onChange={e=>set('r',parseInt(e.target.value||"16",10))}/></div>
        </div>
      )}
      {(l.id==="maxpool"||l.id==="avgpool") ? (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Kernel</div><Input type="number" value={cfg.k||2} onChange={e=>set('k',parseInt(e.target.value||"2",10))}/></div>
          <div><div className="text-xs">Stride</div><Input type="number" value={cfg.s||2} onChange={e=>set('s',parseInt(e.target.value||"2",10))}/></div>
        </div>
      ) : null}
      {l.id==="linear" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Out features</div><Input type="number" value={cfg.outF||1000} onChange={e=>set('outF',parseInt(e.target.value||"1000",10))}/></div>
        </div>
      )}
      {l.id==="add" && (
        <div className="grid grid-cols-1 gap-2">
          <div>
            <div className="text-xs">Skip from step</div>
            <Select value={cfg.from!==null && cfg.from!==undefined ? String(cfg.from) : ""} onValueChange={(v)=>set('from', parseInt(v,10))}>
              <SelectTrigger className="w-full"><SelectValue placeholder={selectedIdx>0?"Choose a previous step":"No previous steps"}/></SelectTrigger>
              <SelectContent>
                {block.map((b,idx)=> idx<selectedIdx ? (
                  <SelectItem key={idx} value={String(idx)}>#{idx} {LAYERS.find(t=>t.id===b.id)?.name || b.id}</SelectItem>
                ) : null)}
              </SelectContent>
            </Select>
            <div className="text-[11px] text-neutral-600 mt-1">Tip: choose the layer at the start of your block (e.g., before Conv/BN stack) for identity addition. Mismatched shapes will be flagged below.</div>
          </div>
        </div>
      )}
      <div className="text-[11px] text-neutral-600">Tip: Use <b>Residual Add</b> around a stack to form a block.</div>
    </div>
  );
}

function renderStepSummary(l, s){
  if(l.id==="conv"||l.id==="grpconv"||l.id==="dilconv") return `${l.name} k${s.cfg.k||3} s${s.cfg.s||1} → C${s.cfg.outC||'?'}`;
  if(l.id==="dwconv") return `Depthwise k${s.cfg.k||3} s${s.cfg.s||1}`;
  if(l.id==="pwconv") return `Pointwise 1×1 → C${s.cfg.outC||'?'}`;
  if(l.id==="maxpool"||l.id==="avgpool") return `${l.name} k${s.cfg.k||2} s${s.cfg.s||2}`;
  if(l.id==="se") return `SE r=${s.cfg.r||16}`;
  if(l.id==="linear") return `Linear → ${s.cfg.outF||'?'}`;
  if(l.id==="add") return `Residual Add${typeof s.cfg?.from==='number' ? ` ← #${s.cfg.from}` : ''}`;
  return l.role;
}

// ---------- Utilities: copy/download ----------
function copyText(text){
  try { navigator.clipboard?.writeText(text); } catch(e) { /* no-op */ }
}
function downloadText(filename, text){
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ---------- PyTorch code generator ----------
function generateTorch(block, H, W, Cin){
  return generateTorchAll(block, [], H, W, Cin);
}

function generateTorchAll(block, modelEls, H, W, Cin){
  const lines = [];
  const emit = (s='')=>lines.push(s);
  const has = (id)=>block.some(s=>s.id===id);

  emit('import torch');
  emit('import torch.nn as nn');
  emit('import torch.nn.functional as F');
  emit('');
  emit('# Inferred from UI (available to import in main.py)');
  emit(`CIN = ${Cin}`);
  emit(`H = ${H}`);
  emit(`W = ${W}`);
  emit('');
  const modelStepsForSe = flattenModelFromCode(modelEls);
  const needsSE = has('se') || modelStepsForSe.some(s=>s.id==='se');
  if (needsSE){
    emit('class SqueezeExcite(nn.Module):');
    emit('    def __init__(self, c, r=16):');
    emit('        super().__init__()');
    emit('        self.pool = nn.AdaptiveAvgPool2d(1)');
    emit('        self.fc1 = nn.Conv2d(c, max(1, c // r), 1)');
    emit('        self.act = nn.SiLU()');
    emit('        self.fc2 = nn.Conv2d(max(1, c // r), c, 1)');
    emit('        self.gate = nn.Sigmoid()');
    emit('');
    emit('    def forward(self, x):');
    emit('        s = self.pool(x)');
    emit('        s = self.fc1(s)');
    emit('        s = self.act(s)');
    emit('        s = self.fc2(s)');
    emit('        s = self.gate(s)');
    emit('        return x * s');
    emit('');
  }

  emit('class GeneratedBlock(nn.Module):');
  emit('    def __init__(self, in_channels='+String(Cin)+'):');
  emit('        super().__init__()');
  let c = Cin;
  block.forEach((s, i)=>{
    const l = LAYERS.find(x=>x.id===s.id);
    const name = `layer_${i}`;
    const cfg = s.cfg || {};
    if(['conv','grpconv','dilconv'].includes(s.id)){
      const k = cfg.k || 3; const srt = cfg.s || 1; const g = cfg.g || (s.id==='grpconv'? (cfg.g||2) : 1); const d = s.id==='dilconv' ? (cfg.d||2) : 1; const o = cfg.outC || c;
      emit(`        self.${name} = nn.Conv2d(${c}, ${o}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${g}, dilation=${d}, bias=False)`);
      c = o;
    } else if(s.id==='pwconv'){
      const o = cfg.outC || c;
      emit(`        self.${name} = nn.Conv2d(${c}, ${o}, kernel_size=1, stride=1, padding=0, bias=False)`);
      c = o;
    } else if(s.id==='dwconv'){
      const k = cfg.k || 3; const srt = cfg.s || 1;
      emit(`        self.${name} = nn.Conv2d(${c}, ${c}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${c}, bias=False)`);
    } else if(s.id==='bn'){
      emit(`        self.${name} = nn.BatchNorm2d(${c})`);
    } else if(s.id==='gn'){
      const groups = cfg.groups || 32;
      emit(`        self.${name} = nn.GroupNorm(num_groups=${groups}, num_channels=${c})`);
    } else if(s.id==='ln'){
      emit(`        self.${name} = nn.LayerNorm(${c})  # NOTE: verify normalized shape`);
    } else if(s.id==='relu'){
      emit(`        self.${name} = nn.ReLU(inplace=True)`);
    } else if(s.id==='gelu'){
      emit(`        self.${name} = nn.GELU()`);
    } else if(s.id==='silu'){
      emit(`        self.${name} = nn.SiLU()`);
    } else if(s.id==='hswish'){
      emit(`        self.${name} = nn.Hardswish()`);
    } else if(s.id==='prelu'){
      emit(`        self.${name} = nn.PReLU(num_parameters=${c})`);
    } else if(s.id==='maxpool'){
      const k = cfg.k || 2; const srt = cfg.s || 2; emit(`        self.${name} = nn.MaxPool2d(kernel_size=${k}, stride=${srt})`);
    } else if(s.id==='avgpool'){
      const k = cfg.k || 2; const srt = cfg.s || 2; emit(`        self.${name} = nn.AvgPool2d(kernel_size=${k}, stride=${srt})`);
    } else if(s.id==='gap'){
      emit(`        self.${name} = nn.AdaptiveAvgPool2d(1)`);
    } else if(s.id==='se'){
      const r = cfg.r || 16; emit(`        self.${name} = SqueezeExcite(${c}, r=${r})`);
    } else if(s.id==='dropout'){
      const p = cfg.p ?? 0.5; emit(`        self.${name} = nn.Dropout(p=${p})`);
    } else if(s.id==='droppath'){
      emit(`        self.${name} = nn.Identity()  # TODO: Stochastic Depth not implemented`);
    } else if(s.id==='mhsa' || s.id==='winattn'){
      emit(`        self.${name} = nn.Identity()  # TODO: Attention placeholder`);
    } else if(s.id==='linear'){
      const o = cfg.outF || 1000; emit(`        self.${name} = nn.Linear(${c}, ${o})`); c = o;
    } else if(s.id==='add' || s.id==='concat' || s.id==='deform'){
      emit(`        self.${name} = nn.Identity()  # TODO: ${l.name} handling in forward`);
    } else {
      emit(`        self.${name} = nn.Identity()`);
    }
  });

  emit('');
  emit('    def forward(self, x):');
  emit('        ys = []');
  block.forEach((s, i)=>{
    const name = `layer_${i}`;
    if(s.id==='add'){
      const from = (s.cfg && typeof s.cfg.from==='number') ? s.cfg.from : null;
      if(from!==null){
        emit(`        x = x + ys[${from}]  # Residual add`);
      } else {
        emit(`        # TODO: set a valid source for residual add`);
      }
    } else if(s.id==='concat'){
      emit(`        # TODO: concat requires specifying sources; keeping x unchanged`);
    } else if(s.id==='linear'){
      emit(`        x = torch.flatten(x, 1)`);
      emit(`        x = self.${name}(x)`);
    } else {
      emit(`        x = self.${name}(x)`);
    }
    emit('        ys.append(x)');
  });
  emit('        return x');

  // If model is defined, also emit a GeneratedModel that flattens blocks/layers
  const modelSteps = flattenModelFromCode(modelEls);
  if (modelSteps.length > 0){
    emit('');
    emit('class GeneratedModel(nn.Module):');
    emit('    def __init__(self, in_channels='+String(Cin)+'):');
    emit('        super().__init__()');
    let mc = Cin;
    modelSteps.forEach((s, i)=>{
      const name = `m_${i}`;
      const cfg = s.cfg || {};
      if(['conv','grpconv','dilconv'].includes(s.id)){
        const k = cfg.k || 3; const srt = cfg.s || 1; const g = cfg.g || (s.id==='grpconv'? (cfg.g||2) : 1); const d = s.id==='dilconv' ? (cfg.d||2) : 1; const o = cfg.outC || mc;
        emit(`        self.${name} = nn.Conv2d(${mc}, ${o}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${g}, dilation=${d}, bias=False)`);
        mc = o;
      } else if(s.id==='pwconv'){
        const o = cfg.outC || mc;
        emit(`        self.${name} = nn.Conv2d(${mc}, ${o}, kernel_size=1, stride=1, padding=0, bias=False)`);
        mc = o;
      } else if(s.id==='dwconv'){
        const k = cfg.k || 3; const srt = cfg.s || 1;
        emit(`        self.${name} = nn.Conv2d(${mc}, ${mc}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${mc}, bias=False)`);
      } else if(s.id==='bn'){
        emit(`        self.${name} = nn.BatchNorm2d(${mc})`);
      } else if(s.id==='gn'){
        const groups = cfg.groups || 32;
        emit(`        self.${name} = nn.GroupNorm(num_groups=${groups}, num_channels=${mc})`);
      } else if(s.id==='ln'){
        emit(`        self.${name} = nn.LayerNorm(${mc})  # NOTE: verify normalized shape`);
      } else if(s.id==='relu'){
        emit(`        self.${name} = nn.ReLU(inplace=True)`);
      } else if(s.id==='gelu'){
        emit(`        self.${name} = nn.GELU()`);
      } else if(s.id==='silu'){
        emit(`        self.${name} = nn.SiLU()`);
      } else if(s.id==='hswish'){
        emit(`        self.${name} = nn.Hardswish()`);
      } else if(s.id==='prelu'){
        emit(`        self.${name} = nn.PReLU(num_parameters=${mc})`);
      } else if(s.id==='maxpool'){
        const k = cfg.k || 2; const srt = cfg.s || 2; emit(`        self.${name} = nn.MaxPool2d(kernel_size=${k}, stride=${srt})`);
      } else if(s.id==='avgpool'){
        const k = cfg.k || 2; const srt = cfg.s || 2; emit(`        self.${name} = nn.AvgPool2d(kernel_size=${k}, stride=${srt})`);
      } else if(s.id==='gap'){
        emit(`        self.${name} = nn.AdaptiveAvgPool2d(1)`);
      } else if(s.id==='se'){
        const r = cfg.r || 16; emit(`        self.${name} = SqueezeExcite(${mc}, r=${r})`);
      } else if(s.id==='dropout'){
        const p = cfg.p ?? 0.5; emit(`        self.${name} = nn.Dropout(p=${p})`);
      } else if(s.id==='droppath' || s.id==='mhsa' || s.id==='winattn' || s.id==='concat' || s.id==='deform'){
        emit(`        self.${name} = nn.Identity()  # TODO`);
      } else if(s.id==='linear'){
        const o = cfg.outF || 1000; emit(`        self.${name} = nn.Linear(${mc}, ${o})`); mc = o;
      } else if(s.id==='add'){
        emit(`        self.${name} = nn.Identity()`);
      } else {
        emit(`        self.${name} = nn.Identity()`);
      }
    });
    emit('');
    emit('    def forward(self, x):');
    emit('        ys = []');
    modelSteps.forEach((s, i)=>{
      const name = `m_${i}`;
      if(s.id==='add'){
        const from = (s.cfg && typeof s.cfg.from==='number') ? s.cfg.from : null;
        if(from!==null){
          emit(`        x = x + ys[${from}]  # Residual add`);
        } else {
          emit(`        # TODO: set a valid source for residual add`);
        }
      } else if(s.id==='concat'){
        emit(`        # TODO: concat requires specifying sources; keeping x unchanged`);
      } else if(s.id==='linear'){
        emit(`        x = torch.flatten(x, 1)`);
        emit(`        x = self.${name}(x)`);
      } else {
        emit(`        x = self.${name}(x)`);
      }
      emit('        ys.append(x)');
    });
    emit('        return x');
  }

  return lines.join('\n');
}

// helper used inside code generator
function flattenModelFromCode(modelEls){
  const out=[]; let base=0;
  for(const el of modelEls){
    if(el.type==='layer'){
      const s = { id: el.id, cfg: { ...(el.cfg||{}) } };
      out.push(s); base += 1;
    } else if(el.type==='block'){
      el.steps.forEach((step)=>{
        const s = { id: step.id, cfg: { ...(step.cfg||{}) } };
        if(s.id==='add' && typeof s.cfg.from==='number'){
          s.cfg = { ...s.cfg, from: (base + s.cfg.from) };
        }
        out.push(s);
      });
      base += el.steps.length;
    }
  }
  return out;
}
