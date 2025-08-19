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
import { X, ArrowUp, ArrowDown, Info, Wrench, Layers as LayersIcon, Blocks, Settings2, Download, Link2, AlertTriangle, CheckCircle, Library, Boxes, Box, PlayCircle, LineChart, HelpCircle, Copy, Save } from "lucide-react";
import { FaApple } from "react-icons/fa";
import { BsNvidia } from "react-icons/bs";
import { TbCpu } from "react-icons/tb";
import { CAT_COLORS, LAYERS, PRESETS, MODEL_PRESETS, SYNERGIES, HP_PRESETS, DATASETS } from "@/lib/constants";
import { copyText, downloadText } from "@/lib/utils";
import Palette from "@/components/builder/Palette";
// PresetBlocksPanel removed in favor of unified collapsible UI
import BlockLayersPreview from "@/components/builder/BlockLayersPreview";
import LayerToken from "@/components/builder/LayerToken";
import AnsiLog from "@/components/builder/AnsiLog";
import ColorChip from "@/components/builder/ColorChip";
import { MetricsViewer, MetricsSummary, ConfusionMatrix } from "@/components/builder/Metrics";
import TrainingProgress from "@/components/builder/TrainingProgress";
import CheckpointPicker from "@/components/builder/CheckpointPicker";

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

// Colors and ColorChip moved to shared modules

// Layer catalog moved to @/lib/constants

// Preset blocks moved to @/lib/constants

// Model presets moved to @/lib/constants

// ---------------- FLOPs/Params estimators (rough) ----------------
const mm = (a,b)=>a*b; const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function convParams(inC,outC,k,groups=1){ return (k*k*inC/groups)*outC; }
function convFLOPs(H,W,inC,outC,k,groups=1){ return H*W*outC*(k*k*inC/groups); }
function dwParams(inC,k){ return inC*k*k; }
function dwFLOPs(H,W,inC,k){ return H*W*inC*k*k; }
function pwParams(inC,outC){ return inC*outC; }
function pwFLOPs(H,W,inC,outC){ return H*W*inC*outC; }
function seParams(C,r=16){ const hidden=Math.max(1,Math.floor(C/r)); return C*hidden + hidden*C; }

// Synergies, HP presets, and datasets moved to @/lib/constants

// ---------------- Main App ----------------
export default function BlocksBuilderLibrary(){
  const [tab,setTab]=useState("build");
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
  const [resumeOpen, setResumeOpen] = useState(false);
  // Runtime device badge (shown on PyTorch tab when a run starts)
  const [deviceDetecting, setDeviceDetecting] = useState(false);
  const [deviceUsed, setDeviceUsed] = useState(null); // e.g., 'cuda' | 'mps' | 'cpu'
  const [runCounter, setRunCounter] = useState(0);

  // Build Block state
  const [H,setH]=useState(56); const [W,setW]=useState(56); const [Cin,setCin]=useState(64);
  const [inputMode, setInputMode] = useState('dataset'); // 'dataset' | 'custom'
  const [datasetId, setDatasetId] = useState('MNIST');
  const [datasetPct, setDatasetPct] = useState(100); // 1-100% of training set
  const dsInfo = DATASETS.find(d=>d.id===datasetId);
  React.useEffect(()=>{
    if(inputMode==='dataset' && dsInfo){ setH(dsInfo.H); setW(dsInfo.W); setCin(dsInfo.C); }
  }, [inputMode, datasetId]);
  const [block,setBlock]=useState([]); // [{id, cfg}]
  const [selectedIdx,setSelectedIdx] = useState(-1);
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
  // When editing a block from the Model in the Build Block tab, track its model index
  const [editingModelBlockIdx, setEditingModelBlockIdx] = useState(null);
  // Foldable panels in Build Model
  const [showSaved, setShowSaved] = useState(false);
  const [showPresetBlocks, setShowPresetBlocks] = useState(false);
  const [showPresetModels, setShowPresetModels] = useState(false);
  const [showPresetBlocksBuild, setShowPresetBlocksBuild] = useState(false);

  // Live-sync: when editing a model block in the builder, push changes back to the model
  React.useEffect(()=>{
    if(editingModelBlockIdx==null) return;
    setModel(prev=>{
      const i = editingModelBlockIdx;
      if(!Array.isArray(prev) || i<0 || i>=prev.length) return prev;
      const el = prev[i];
      if(!el || el.type !== 'block') return prev;
      const steps = block.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } }));
      const arr=[...prev];
      arr[i] = { ...el, steps };
      return arr;
    });
  }, [block, editingModelBlockIdx]);

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
  // Warn if applying Linear while spatial dims > 1 (will be globally pooled at codegen)
  if(h>1 || w>1){ issues.push({ type:"linear_spatial", step:i, msg:`Linear placed with spatial dims ${h}×${w}; will GAP before flatten in codegen.` }); }
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
  if(h>1 || w>1){ issues.push({ type:"linear_spatial", step:i, msg:`Linear placed with spatial dims ${h}×${w}; will GAP before flatten in codegen.` }); }
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
  const name = nextIndexedBlockName(p.name, prev);
  const el = { type:'block', name, steps: wired };
      if (modelSelIdx>=0){ const arr=[...prev]; arr.splice(modelSelIdx+1,0, el); return arr; }
  return [...prev, el];
    });
    setTab('model');
  };

  // Helper to offset residual 'from' indices by a base offset
  function offsetAddIndices(steps, offset){
    return steps.map(s=> (s.id==='add' && typeof s.cfg?.from==='number') ? ({ ...s, cfg: { ...s.cfg, from: s.cfg.from + offset } }) : s);
  }

  // Append a preset block to current Block builder
  const appendPresetToBlock = (p)=>{
    const built = p.composition.map(id=>({ id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }));
    const wired = autowireResiduals(built, H, W, Cin);
    const adjusted = offsetAddIndices(wired, block.length);
    setBlock(prev=>[...prev, ...adjusted]);
    setSelectedIdx(-1);
    setTab('build');
  };

  // Helper: build a block's steps from a preset id, overriding channels and optional first-layer stride
  function buildBlockStepsFromPreset(presetId, outC, strideFirst=1){
    const preset = PRESETS.find(pp=>pp.id===presetId);
    if(!preset) return [];
    let firstConvDone=false;
    const steps = preset.composition.map(id=>{
      const base = { id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } };
      if((id==='conv' || id==='pwconv') && !firstConvDone){
        firstConvDone = true;
        base.cfg.outC = outC;
        if(id==='conv') base.cfg.s = strideFirst;
      } else if(id==='pwconv'){
        // ensure final pointwise aligns to outC
        base.cfg.outC = outC;
      }
      return base;
    });
    return autowireResiduals(steps, H, W, Cin);
  }

  // Build a full model from a model preset plan
  function buildModelFromPreset(plan){
  const out = [];
  const countByName = new Map();
    plan.forEach(seg=>{
      if(seg.type==='layer'){
        out.push({ type:'layer', id: seg.id, cfg: { ...(seg.cfg||{}) } });
      } else if(seg.type==='blockRef'){
        const repeat = seg.repeat || 1;
        for(let i=0;i<repeat;i++){
          const steps = buildBlockStepsFromPreset(seg.preset, seg.outC, seg.downsample && i===0 ? 2 : 1);
      const base = PRESETS.find(p=>p.id===seg.preset)?.name || seg.preset;
      const idx = (countByName.get(base) || 0) + 1; countByName.set(base, idx);
      const name = `${base} #${idx}`;
          out.push({ type:'block', name, steps });
          // Residual outside the block: add an Add layer that auto-sources from the input to this block
          out.push({ type:'layer', id:'add', cfg: { autoFromOffset: steps.length } });
        }
      }
    });
    return out;
  }

  // Use/append a model preset
  const applyModelPreset = (mp)=>{
    const els = buildModelFromPreset(mp.plan);
    setModel(els);
    setModelSelIdx(-1);
    setTab('model');
  };
  const appendModelPreset = (mp)=>{
    const els = buildModelFromPreset(mp.plan);
    setModel(prev=>[...prev, ...els]);
    setTab('model');
  };

  // Simulate shapes to wire residual connections
  function autowireResiduals(steps, H0, W0, C0){
    // First pass: compute shapes after each step
    let h=H0, w=W0, c=C0;
    const shapesAfter = [];
    steps.forEach((s,i)=>{
      const l = LAYERS.find(x=>x.id===s.id) || {};
      if(l.id==="conv"||l.id==="grpconv"||l.id==="dilconv"||l.id==="deform"){
        const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; const o=s.cfg?.outC||c; c=o; h=Math.floor((h + 2*p - k)/st + 1); w=Math.floor((w + 2*p - k)/st + 1);
      } else if(l.id==="dwconv"){
        const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; h=Math.floor((h + 2*p - k)/st + 1); w=Math.floor((w + 2*p - k)/st + 1);
      } else if(l.id==="pwconv"){
        const o=s.cfg?.outC||c; c=o;
      } else if(l.id==="maxpool"||l.id==="avgpool"){
        const k=s.cfg?.k||2; const st=s.cfg?.s||2; h=Math.floor((h - k)/st + 1); w=Math.floor((w - k)/st + 1);
      } else if(l.id==="gap"){
        h=1; w=1;
      } else if(l.id==="concat"){
        c = c * 2;
      } else if(l.id==="linear"){
        c = s.cfg?.outF || c;
      }
      shapesAfter[i] = { C:c, H:h, W:w };
    });
    // Second pass: set/rescue 'from' for adds and advance running shape
    let curC=C0, curH=H0, curW=W0;
    return steps.map((s,i)=>{
      const l = LAYERS.find(x=>x.id===s.id) || {};
      if(l.id==="add"){
        const pre = { C:curC, H:curH, W:curW };
        let from = (typeof s.cfg?.from==='number') ? s.cfg.from : null;
        const isValid = (j)=> j!=null && j>=0 && j<i && shapesAfter[j] && shapesAfter[j].C===pre.C && shapesAfter[j].H===pre.H && shapesAfter[j].W===pre.W;
        if(!isValid(from)){
          from = null;
          for(let j=i-1;j>=0;j--){ if(isValid(j)){ from=j; break; } }
        }
        s = { ...s, cfg: { ...(s.cfg||{}), ...(from!=null ? { from } : {}) } };
      }
      // advance current shape
      if(l.id==="conv"||l.id==="grpconv"||l.id==="dilconv"||l.id==="deform"){
        const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; const o=s.cfg?.outC||curC; curC=o; curH=Math.floor((curH + 2*p - k)/st + 1); curW=Math.floor((curW + 2*p - k)/st + 1);
      } else if(l.id==="dwconv"){
        const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; curH=Math.floor((curH + 2*p - k)/st + 1); curW=Math.floor((curW + 2*p - k)/st + 1);
      } else if(l.id==="pwconv"){
        const o=s.cfg?.outC||curC; curC=o;
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
  const duplicateIdx = (i)=> setBlock(prev=>{ const arr=[...prev]; const src=arr[i]; if(!src) return arr; const dup={ id: src.id, cfg: { ...(src.cfg||{}) } }; arr.splice(i+1,0, dup); return arr; });

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
  const el = { type:'block', name: nextIndexedBlockName(blk.name, prev), steps: blk.steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } })) };
    if (modelSelIdx>=0){ const arr=[...prev]; arr.splice(modelSelIdx+1,0, el); return arr; }
    return [...prev, el];
  });
  const removeModelIdx = (i)=> setModel(prev=>prev.filter((_,j)=>j!==i));
  const moveModelIdx = (i,dir)=> setModel(prev=>{ const arr=[...prev]; const j=i+dir; if(j<0||j>=arr.length) return arr; const t=arr[i]; arr[i]=arr[j]; arr[j]=t; return arr; });
  const duplicateModelIdx = (i)=> setModel(prev=>{
    const arr=[...prev];
    const el=arr[i];
    if(!el) return arr;
    if(el.type==="layer"){
      const dup={ type:'layer', id: el.id, cfg: { ...(el.cfg||{}) } };
      arr.splice(i+1,0, dup);
      return arr;
    }
    if(el.type==="block"){
      // duplicate block with a new indexed name
      const steps = el.steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } }));
      const name = nextIndexedBlockName(el.name.replace(/\s+#\d+$/,'') || 'Block', arr);
      const dup={ type:'block', name, steps };
      arr.splice(i+1,0, dup);
      return arr;
    }
    return arr;
  });
  const expandBlockAt = (i)=> setModel(prev=>{
    const arr=[...prev]; const el=arr[i]; if(!el || el.type!=='block') return arr;
    // splice block steps as individual layer elements
    const layers = el.steps.map(s=>({ type:'layer', id:s.id, cfg:{ ...(s.cfg||{}) } }));
    arr.splice(i,1, ...layers);
    return arr;
  });

  // New: Edit a model block inside the Build Block tab (live-linked)
  const editBlockInBuilder = (i)=>{
    const el = model[i];
    if(!el || el.type!=='block') return;
    setBlock(el.steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } })));
    setSelectedIdx(-1);
    setEditingModelBlockIdx(i);
    setTab('build');
  };

  // Flatten model into steps and adjust residual 'from' to global indices
  function flattenModelWithFromAdjust(modelEls){
    const out=[]; let base=0;
    for(const el of modelEls){
      if(el.type==='layer'){
        const s = { id: el.id, cfg: { ...(el.cfg||{}) } };
        // If an absolute reference sneaks onto a layer (unlikely), normalize
        if(s.id==='add' && typeof s.cfg?.fromGlobal==='number'){
          s.cfg = { ...s.cfg, from: s.cfg.fromGlobal };
          delete s.cfg.fromGlobal;
        } else if(s.id==='add' && s.cfg?.autoFrom==='prev'){
          // Auto-connect to previous global output (identity path around prior stack)
          s.cfg = { ...s.cfg, from: Math.max(0, base - 1) };
          delete s.cfg.autoFrom;
        } else if(s.id==='add' && typeof s.cfg?.autoFromOffset==='number'){
          // Connect to the element located `offset` steps before the add, minus one to reach the input of that block
          const offset = Math.max(0, parseInt(s.cfg.autoFromOffset, 10) || 0);
          s.cfg = { ...s.cfg, from: Math.max(0, base - offset - 1) };
          delete s.cfg.autoFromOffset;
        }
        out.push(s); base += 1;
      } else if(el.type==='block'){
        el.steps.forEach((step)=>{
          const s = { id: step.id, cfg: { ...(step.cfg||{}) } };
          if(s.id==='add'){
            if(typeof s.cfg.fromGlobal==='number'){
              s.cfg = { ...s.cfg, from: s.cfg.fromGlobal };
              delete s.cfg.fromGlobal;
            } else if(typeof s.cfg.from==='number'){
              s.cfg = { ...s.cfg, from: (base + s.cfg.from) };
            }
          }
          out.push(s);
        });
        base += el.steps.length;
        // If the very next element is a model-level Add with autoFrom, it will be resolved on its own branch above
      }
    }
    return out;
  }

  // Utility: generate a unique block name with incremental index if duplicates exist
  function nextIndexedBlockName(baseName, currentModel){
    const existing = (currentModel||[]).filter(el=> el.type==='block' && typeof el.name==='string');
    const same = existing.filter(el=> el.name===baseName || el.name?.startsWith(baseName + ' #'));
    if(same.length===0) return baseName;
    const indices = same.map(el=>{
      const m = String(el.name||'').match(/#(\d+)$/); return m? parseInt(m[1],10) : 0;
    });
    const next = Math.max(0, ...indices) + 1;
    return `${baseName} #${next}`;
  }

  // Stats for Model (flattened)
  const modelStats = useMemo(()=>{
    const steps = flattenModelWithFromAdjust(model);
    return simulateStatsForSteps(steps, H, W, Cin);
  }, [model, H, W, Cin]);

  // Flattened steps for quick lookups (labels/graph)
  const flattenedSteps = useMemo(()=> flattenModelWithFromAdjust(model), [model]);

  // helper: model index for a flattened step index
  function modelIdxForFlattened(flatIdx){
    let base=0;
    for(let i=0;i<model.length;i++){
      const el=model[i];
      const span = (el.type==='layer') ? 1 : (el.steps?.length||0);
      if(flatIdx < base + span) return i;
      base += span;
    }
    return -1;
  }

  // helper: flattened index for a given model index
  const flattenedIndexForModelIdx = (idx)=>{
    let base=0;
    for(let i=0;i<idx;i++){
      const el=model[i];
      base += (el.type==='layer') ? 1 : (el.steps?.length||0);
    }
    return base;
  };

  // Build a flattened meta list with block names and layer names
  const flattenedModelMeta = useMemo(()=>{
    const meta=[]; let base=0;
    model.forEach((el, mi)=>{
      if(el.type==='layer'){
        const l = LAYERS.find(x=>x.id===el.id);
        meta.push({ g: base, modelIdx: mi, inBlockIdx: null, blockName: null, layerName: l?.name||el.id, id: el.id });
        base += 1;
      } else if(el.type==='block'){
        el.steps.forEach((s, bi)=>{
          const l = LAYERS.find(x=>x.id===s.id);
          meta.push({ g: base+bi, modelIdx: mi, inBlockIdx: bi, blockName: el.name, layerName: l?.name||s.id, id: s.id });
        });
        base += el.steps.length;
      }
    });
    return meta;
  }, [model]);

  // Hyperparameters tab
  const [hp, setHp] = useState({
    optimizer: "AdamW",
    lr: 0.01,
    momentum: 0.9,
    weightDecay: 1e-4,
    scheduler: "none",
    warmup: 0,
    epochs: 10,
    labelSmoothing: 0.0,
    mixup: 0.0,
    cutmix: 0.0,
    stochasticDepth: 0.0,
    ema: false,
    cosineRestarts: false,
    T0: 10,
    Tmult: 2,
    batchSize: 128,
    numWorkers: 4,
    loss: "CrossEntropy",
    valSplit: 0.1,
    gradClip: 0.0,
    stepSize: 30,
    gamma: 0.1,
  device: "cpu",
  precision: "fp32", // fp32 | amp_fp16 | amp_bf16
  });
  const applyPreset = (p)=>{ setHp(prev=>({ ...prev, ...p.details, cosineRestarts: p.details.scheduler==="cosine_warm_restarts" })); };

  const exportJSON = ()=>{
    const payload = { 
      block: block.map(s=>({ id:s.id, cfg:s.cfg })),
      model: model.map(el=> el.type==='layer' ? ({ type:'layer', id:el.id, cfg:el.cfg }) : ({ type:'block', name:el.name, steps: el.steps })),
  input: { H,W,Cin, mode: inputMode, datasetId, datasetPct }, stats, modelStats, hyperparams: hp };
    const blob = new Blob([JSON.stringify(payload,null,2)], { type:"application/json" });
    const url = URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='block_builder_export.json'; a.click(); URL.revokeObjectURL(url);
  };

  // keep code preview in sync when switching to code tab or when block changes
  React.useEffect(()=>{
    if(tab==='code'){
      setCode(generateTorchAll(block, model, H, W, Cin));
    }
  }, [tab, block, model, H, W, Cin]);

  // When a run starts, poll output to detect device line "DEVICE: <type>"
  React.useEffect(()=>{
    if(!deviceDetecting) return;
    let alive = true;
    let tries = 0;
    const intervalMs = 750;
    const maxTries = Math.ceil((5 * 60 * 1000) / intervalMs); // 5 minutes
    const tick = ()=>{
      fetch('/api/run-output', { cache: 'no-store' })
        .then(r=>r.text())
        .then(t=>{
          if(!alive) return;
          const s = String(t||'');
          const m = s.match(/DEVICE:\s*([A-Za-z0-9_\-:]+)/);
          if(m && m[1]){
            setDeviceUsed(m[1].toLowerCase());
            setDeviceDetecting(false);
          }
        })
        .catch(()=>{});
      tries+=1;
      // stop trying after ~5 min
      if(tries>maxTries){ alive=false; setDeviceDetecting(false); }
    };
    const id = setInterval(tick, intervalMs);
    tick();
    return ()=>{ alive=false; clearInterval(id); };
  }, [deviceDetecting, runCounter]);

  return (
    <div className="w-full h-[100dvh] bg-neutral-50 flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b bg-white flex items-center gap-3 sticky top-0 z-10">
        <LayersIcon className="w-5 h-5"/>
        <div className="text-lg font-semibold">Netkego</div>
  <Badge variant="secondary">Deep Learning Classification Toy</Badge>
        <div className="ml-auto"><Button variant="outline" onClick={exportJSON}><Download className="w-4 h-4 mr-2"/>Export</Button></div>
      </div>

  <div className="flex-1 min-h-0 overflow-auto grid grid-cols-12 gap-3 p-3">
        <div className="col-span-3 min-w-[280px] flex flex-col gap-3">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Info className="w-4 h-4"/>Input / Builder Settings</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div>
                <div className="text-xs mb-1">Mode</div>
                <Select value={inputMode} onValueChange={setInputMode}>
                  <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="dataset">Dataset (enables training)</SelectItem>
                    <SelectItem value="custom">Custom (random tensors; no training)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {inputMode==='dataset' ? (
                <>
                  <div>
                    <div className="text-xs mb-1">Dataset</div>
                    <Select value={datasetId} onValueChange={setDatasetId}>
                      <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                      <SelectContent>
                        {DATASETS.map(d=> <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="mt-2 text-[11px] p-2 rounded-md bg-neutral-100 text-neutral-700">{dsInfo?.desc}</div>
                  <div>
                    <div className="text-xs mb-1">Dataset sample</div>
                    <Slider value={[datasetPct]} min={1} max={100} step={1} onValueChange={([v])=>setDatasetPct(v)} />
                    <div className="text-xs mt-1 text-neutral-700">{datasetPct}% of training data (validation split will be applied within this sample).</div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 opacity-80">
                    <div><div className="text-xs">H</div><Input readOnly value={H}/></div>
                    <div><div className="text-xs">W</div><Input readOnly value={W}/></div>
                    <div><div className="text-xs">Channels</div><Input readOnly value={Cin}/></div>
                  </div>
                  <div className="text-xs text-neutral-600">Sizes derive from the selected dataset. Training will generate code accordingly.</div>
                </>
              ) : (
                <>
                  <div className="grid grid-cols-3 gap-2">
                    <div><div className="text-xs">H</div><Input type="number" value={H} onChange={e=>setH(parseInt(e.target.value||"56",10))}/></div>
                    <div><div className="text-xs">W</div><Input type="number" value={W} onChange={e=>setW(parseInt(e.target.value||"56",10))}/></div>
                    <div><div className="text-xs">Channels</div><Input type="number" value={Cin} onChange={e=>setCin(parseInt(e.target.value||"64",10))}/></div>
                  </div>
                  <div className="text-xs text-neutral-600">Custom shapes disable dataset training/testing. Use PyTorch tab to run forward only.</div>
                </>
              )}
            </CardContent>
          </Card>

          <Palette addLayer={tab==='model' ? addModelLayer : addLayer} mode={tab==='model' ? 'model' : 'build'} compat={tab==='model' ? null : computeNextCompat(block, stats, { C: Cin, H, W })} />
        </div>

        <div className="col-span-9">
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList>
              <TabsTrigger value="build"><Wrench className="w-4 h-4 mr-1"/>Build Blocks</TabsTrigger>
              <TabsTrigger value="model"><Boxes className="w-4 h-4 mr-1"/>Build Model</TabsTrigger>
              <TabsTrigger value="hparams"><Settings2 className="w-4 h-4 mr-1"/>Hyperparameters</TabsTrigger>
              <TabsTrigger value="training"><LineChart className="w-4 h-4 mr-1"/>Training</TabsTrigger>
              <TabsTrigger value="code"><Wrench className="w-4 h-4 mr-1"/>PyTorch</TabsTrigger>
            </TabsList>

            {/* Build Blocks tab: construct block with preset list on the right */}
            <TabsContent value="build">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  {editingModelBlockIdx!=null && model[editingModelBlockIdx]?.type==='block' && (
                    <div className="mb-2 p-2 rounded-md bg-blue-50 border border-blue-200 text-blue-800 flex items-center justify-between">
                      <span>Editing model block: <b>{model[editingModelBlockIdx]?.name || 'Block'}</b>. Changes sync automatically.</span>
                      <Button size="sm" variant="secondary" onClick={()=>setEditingModelBlockIdx(null)}>Done</Button>
                    </div>
                  )}
                  <Card>
                    <CardHeader>
                      <div className="flex items-center gap-2">
                        <CardTitle>Current Block</CardTitle>
                        <div className="ml-auto flex items-center gap-2">
                          <Input className="h-8 w-48" placeholder="Name (optional)" value={saveName} onChange={(e)=>setSaveName(e.target.value)} />
                          <Button size="sm" onClick={saveCurrentBlock} disabled={block.length===0} title="Save block"><Save className="w-4 h-4 mr-1"/>Save</Button>
                        </div>
                      </div>
                      <div className="text-xs text-neutral-600 mt-1">Use the Layer Palette on the left, or pick a Preset Block below.</div>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {block.length===0 && <div className="text-neutral-600 text-sm">Add layers from the left palette or import a preset.</div>}
            {block.map((s,i)=>{
                        const l=LAYERS.find(x=>x.id===s.id);
                        const addIssue = stats.issues.find(it => it.step===i && (it.type==="add_invalid_from" || it.type==="add_mismatch"));
                        return (
              <div key={i} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${CAT_COLORS[l.category]?.ring||''}`}>
                            <div className="flex items-start gap-2">
                              <LayerToken id={s.id} cfg={s.cfg} size="md" showHelper={true} />
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
                            </div>
                            <div className="flex items-center gap-1">
                              <Button size="icon" variant="ghost" onClick={()=>duplicateIdx(i)} title="Duplicate"><Copy className="w-4 h-4"/></Button>
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

                  {/* Save Block card removed; controls are in header */}
                </div>

                <div className="col-span-2 space-y-3">
                  <Card>
                    <CardHeader><CardTitle>Inspector & Config</CardTitle></CardHeader>
                    <CardContent className="text-sm space-y-3">
                      {selected ? (
                        <LayerConfig selected={selected} selectedIdx={selectedIdx} block={block} stats={stats} addContext={{ scope: editingModelBlockIdx!=null ? 'model' : 'block', flattenedMeta: flattenedModelMeta, baseOffset: flattenedIndexForModelIdx(editingModelBlockIdx ?? 0) - selectedIdx }} onChange={(cfg)=>{ setBlock(prev=>{ const arr=[...prev]; arr[selectedIdx]={...arr[selectedIdx], cfg}; return arr; }); }} />
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
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2"><Blocks className="w-4 h-4"/>Preset Blocks
                        <Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowPresetBlocksBuild(v=>!v)}>{showPresetBlocksBuild? 'Hide':'Show'}</Button>
                      </CardTitle>
                    </CardHeader>
                    {showPresetBlocksBuild && (
                      <CardContent className="space-y-2">
                        {PRESETS.map(p=> (
                          <div key={p.id} className="border border-neutral-200 rounded-md p-2 bg-white/90">
                            <div className="flex items-start justify-between gap-2">
                              <div>
                                <div className="text-sm font-medium">{p.name}</div>
                                <div className="text-[11px] text-neutral-600">{p.family}</div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button size="sm" variant="secondary" onClick={()=>importPreset(p)}>Use</Button>
                                <Button size="sm" variant="outline" onClick={()=>appendPresetToBlock(p)}>Append</Button>
                              </div>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1.5 items-center">
                              {p.composition.map((id, idx)=> (
                                <LayerToken key={idx} id={id} size="md" showHelper={true} />
                              ))}
                            </div>
                          </div>
                        ))}
                      </CardContent>
                    )}
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* Build Model tab */}
            <TabsContent value="model">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  <Card className="mb-3">
                    <CardHeader><CardTitle>Inspector & Config</CardTitle></CardHeader>
                    <CardContent className="text-sm">
                      {modelSelected ? (
                        modelSelected.type==='layer' ? (
                          <LayerConfig selected={modelSelected} selectedIdx={flattenedIndexForModelIdx(modelSelIdx)} block={flattenModelWithFromAdjust(model).map(s=>s)} stats={modelStats} addContext={{ scope: 'model', flattenedMeta: flattenedModelMeta, baseOffset: 0 }} onChange={(cfg)=>{ setModel(prev=>{ const arr=[...prev]; arr[modelSelIdx]={...arr[modelSelIdx], cfg}; return arr; }); }} />
                        ) : (
                          <div className="text-xs text-neutral-700">
                            <div className="font-medium mb-1">Block: {modelSelected.name}</div>
                            <div>Steps: {modelSelected.steps.map(s=>LAYERS.find(x=>x.id===s.id)?.name||s.id).join(' → ')}</div>
                            <div className="mt-2">Blocks are edited in the Build Blocks tab. Use "Edit in Blocks" in the list to open and live-edit this block.</div>
                          </div>
                        )
                      ) : (
                        <div className="text-neutral-600 text-sm">Select a layer to edit, or use "Edit in Blocks" on a block to edit its layers.</div>
                      )}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader><CardTitle>Current Model</CardTitle></CardHeader>
                    <CardContent className="space-y-2">
                      <div className="border border-dashed border-neutral-300 rounded-xl p-2 bg-white/60 text-xs text-neutral-700">Input • ({Cin}, {H}, {W}) — mandatory</div>
                      {model.length===0 && <div className="text-neutral-600 text-sm">Add saved blocks, presets, or layers from the right panel.</div>}
                      {model.map((el,i)=>{
                        const baseIdx = flattenedIndexForModelIdx(i);
                        const labelFor = (g)=>{
                          const m = flattenedModelMeta.find(mm=>mm.g===g);
                          if(!m) return `#${g}`;
                          return `#${g} ${m.blockName ? '['+m.blockName+'] ' : ''}${m.layerName}`;
                        };
                        if(el.type==='layer'){
                          const l=LAYERS.find(x=>x.id===el.id);
                          const extraDesc = (l.id==='add') ? (()=>{ const step = flattenedSteps[baseIdx]; const from = step && typeof step.cfg?.from==='number' ? step.cfg.from : null; return (from!=null) ? `Residual Add ← ${labelFor(from)}` : 'Residual Add (select source)'; })() : null;
                          return (
                            <div key={`L${i}`} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${CAT_COLORS[l.category]?.ring||''}`}>
                              <div className="flex items-start gap-2">
                                <LayerToken id={el.id} cfg={el.cfg} size="md" showHelper={true} />
                                <div>
                                  <div className="text-sm font-medium flex items-center gap-2">{l.name}<ColorChip category={l.category}/></div>
                                  <div className="text-xs text-neutral-600">{extraDesc ?? renderStepSummary(l, el)}</div>
                                </div>
                              </div>
                              <div className="flex items-center gap-1">
                                <Button size="icon" variant="ghost" onClick={()=>duplicateModelIdx(i)} title="Duplicate"><Copy className="w-4 h-4"/></Button>
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
                          <div key={`B${i}`} className={`border border-neutral-200 rounded-xl p-2 bg-neutral-100 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition`}>
                            <div className="min-w-0">
                              <div className="text-sm font-medium flex items-center gap-2"><Box className="w-4 h-4"/>{el.name}<span className="px-2 py-0.5 rounded-md text-[11px] bg-neutral-200 text-neutral-800">Block</span></div>
                              <div className="mt-1 flex flex-wrap gap-1.5 items-center">
                                {el.steps.map((s, idx)=> (
                                  <LayerToken key={idx} id={s.id} cfg={s.cfg} size="md" showHelper={true} />
                                ))}
                              </div>
                            </div>
                            <div className="flex items-center gap-1">
                              <Button size="icon" variant="ghost" onClick={()=>duplicateModelIdx(i)} title="Duplicate"><Copy className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                              <Button size="icon" variant="ghost" onClick={()=>editBlockInBuilder(i)}><Wrench className="w-4 h-4"/></Button>
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
                      <div>Est. memory (batch {hp.batchSize}): {formatMem(estimateMemoryMB(
                        modelStats,
                        hp.batchSize,
                        hp.precision,
                        hp.optimizer,
                        { Cin, H, W, datasetId, datasetPct, valSplit: hp.valSplit, numWorkers: hp.numWorkers, ema: hp.ema, inputMode }
                      ))}</div>
                      <div>Issues: {modelStats.issues.length===0 ? <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle className="w-3.5 h-3.5"/>None</span> : <span className="inline-flex items-center gap-1 text-rose-700"><AlertTriangle className="w-3.5 h-3.5"/>{modelStats.issues.length}</span>}</div>
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Model Dimension Checker</CardTitle></CardHeader>
                    <CardContent className="space-y-2 text-xs">
                      {modelStats.issues.length===0 && (
                        <div className="text-emerald-700 flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5"/>No dimensional conflicts detected.</div>
                      )}
                      {modelStats.issues.map((it, idx)=> (
                        <div key={idx} className="border border-neutral-200 rounded-md p-2 bg-white/90 backdrop-blur-sm shadow-sm">
                          <div className="font-medium flex items-center gap-2">
                            {(it.type==="add_mismatch" || it.type==="add_invalid_from") ? <AlertTriangle className="w-3.5 h-3.5 text-rose-700"/> : null}
                            Step #{it.step}: {it.msg}
                          </div>
                          {it.type==="add_mismatch" && it.ref && (
                            <div className="mt-1 text-[11px] text-neutral-700">Suggestion: insert a <b>1×1 Conv</b> (projection) or adjust <b>stride</b>/padding so both paths yield the same (C,H,W).</div>
                          )}
                          {it.type==="linear_spatial" && (
                            <div className="mt-1 text-[11px] text-neutral-700">Tip: add a <b>Global Avg Pool</b> before Linear, or rely on generator safety (auto-GAP) now enabled.</div>
                          )}
                        </div>
                      ))}
                    </CardContent>
                  </Card>

                </div>

                <div className="col-span-2">
                  <Card>
                    <CardHeader><CardTitle>Model Builder</CardTitle></CardHeader>
                    <CardContent className="space-y-3 text-sm">
                      <div className="text-xs text-neutral-600">Add blocks or layers. Input is implicit from the left settings.</div>
                      <div className="border rounded-xl p-2">
                        <div className="font-medium text-sm mb-1 flex items-center gap-2"><Library className="w-4 h-4"/>Saved Blocks<Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowSaved(v=>!v)}>{showSaved? 'Hide':'Show'}</Button></div>
                        {!showSaved ? null : savedBlocks.length===0 ? (
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
                        <div className="font-medium text-sm mb-1 flex items-center gap-2"><Blocks className="w-4 h-4"/>Preset Blocks<Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowPresetBlocks(v=>!v)}>{showPresetBlocks? 'Hide':'Show'}</Button></div>
                        {!showPresetBlocks ? null : (
                          <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                            {PRESETS.map(p=> (
                              <div key={p.id} className="border border-neutral-200 rounded-md p-2 bg-white/90">
                                <div className="flex items-start justify-between gap-2">
                                  <div>
                                    <div className="text-sm font-medium">{p.name}</div>
                                    <div className="text-[11px] text-neutral-600">{p.family}</div>
                                  </div>
                                  <Button size="sm" variant="outline" onClick={()=>addPresetToModel(p)}>Add</Button>
                                </div>
                                <div className="mt-2 flex flex-wrap gap-1.5 items-center">
                                  {p.composition.map((id, idx)=> (
                                    <LayerToken key={idx} id={id} size="md" showHelper={true} />
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Palette removed here; use the left-side palette instead */}
                      <div className="text-xs text-neutral-600 border rounded-xl p-2">
                        Use the Layer Palette on the left. It will add to
                        {" "}
                        <b>{tab==='model' ? 'Build Model' : 'Build Block'}</b>
                        {" "}
                        depending on the active tab.
                      </div>

                      <div className="border rounded-xl p-2">
                        <div className="font-medium text-sm mb-1 flex items-center gap-2"><Boxes className="w-4 h-4"/>Preset Models<Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowPresetModels(v=>!v)}>{showPresetModels? 'Hide':'Show'}</Button></div>
                        {!showPresetModels ? null : (
                          <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                            {MODEL_PRESETS.map(mp=> (
                              <div key={mp.id} className="border border-neutral-200 rounded-md p-2 bg-white/90 flex items-center justify-between">
                                <div>
                                  <div className="text-sm font-medium">{mp.name}</div>
                                  <div className="text-[11px] text-neutral-600 truncate">{mp.family} — {mp.description}</div>
                                </div>
                                <div className="flex items-center gap-2">
                                  <Button size="sm" variant="secondary" onClick={()=>applyModelPreset(mp)}>Use</Button>
                                  <Button size="sm" variant="outline" onClick={()=>appendModelPreset(mp)}>Append</Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Graph view */}
                      <Card className="mt-3">
                        <CardHeader><CardTitle>Model Graph</CardTitle></CardHeader>
                        <CardContent>
                          {(() => {
                            const N = model.length;
                            const rowH = 48; const width = 420; const height = Math.max(60, N*rowH + 20);
                            const nodes = model.map((el, i)=>({ i, y: 20 + i*rowH + 12, label: el.type==='block'? el.name : (LAYERS.find(l=>l.id===el.id)?.name || el.id), type: el.type }));
                            const adds = flattenedSteps.map((s, g)=>({s,g})).filter(o=> o.s.id==='add' && typeof o.s.cfg?.from==='number');
                            const edges = adds.map(({s,g})=> ({ from: modelIdxForFlattened(s.cfg.from), to: modelIdxForFlattened(g) })).filter(e=> e.from>=0 && e.to>=0);
                            return (
                              <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="bg-white/70 rounded-md border">
                                <defs>
                                  <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
                                    <path d="M0,0 L0,6 L6,3 z" fill="#64748b" />
                                  </marker>
                                </defs>
                                {nodes.slice(0,-1).map((n,i)=> (
                                  <line key={`seq-${i}`} x1={80} y1={n.y} x2={80} y2={nodes[i+1].y} stroke="#cbd5e1" strokeWidth="2" />
                                ))}
                                {nodes.map(n=> (
                                  <g key={`n-${n.i}`}>
                                    <circle cx={80} cy={n.y} r={8} fill={n.type==='block'? '#2563eb':'#10b981'} />
                                    <text x={96} y={n.y+4} fontSize="12" fill="#334155">{n.label}</text>
                                  </g>
                                ))}
                                {edges.map((e,idx)=>{
                                  const aY = nodes[e.from]?.y; const bY = nodes[e.to]?.y;
                                  if(aY==null || bY==null) return null;
                                  const x1 = 80, y1 = aY, x2 = 80, y2 = bY;
                                  const dx = 120 + (idx%3)*20;
                                  const path = `M ${x1} ${y1} C ${x1+dx} ${y1}, ${x2+dx} ${y2}, ${x2} ${y2}`;
                                  return <path key={`res-${idx}`} d={path} fill="none" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>;
                                })}
                              </svg>
                            );
                          })()}
                          <div className="text-[11px] text-neutral-600 mt-1">Blocks are blue nodes; layers are green nodes. Curved lines show residual adds (source → add location).</div>
                        </CardContent>
                      </Card>
                    </CardContent>
                  </Card>

                  {/* Inspector moved to top */}
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
                      <div>
                        <div className="text-xs flex items-center justify-between">
                          <span>Batch size</span>
                          <span className="text-[11px] text-neutral-600">
                            ~{formatMem(estimateMemoryMB(
                              modelStats,
                              Math.max(1, hp.batchSize),
                              hp.precision,
                              hp.optimizer,
                              { Cin, H, W, datasetId, datasetPct, valSplit: hp.valSplit, numWorkers: hp.numWorkers, ema: hp.ema, inputMode }
                            ))}
                          </span>
                        </div>
                        <Input type="number" value={hp.batchSize} onChange={(e)=>setHp({...hp, batchSize:parseInt(e.target.value||"128",10)})}/>
                      </div>
                      <div>
                        <div className="text-xs">Data workers</div>
                        <Input type="number" value={hp.numWorkers} onChange={(e)=>setHp({...hp, numWorkers:parseInt(e.target.value||"2",10)})}/>
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
                        <div className="text-xs">Loss</div>
                        <Select value={hp.loss} onValueChange={(v)=>setHp({...hp, loss:v})}>
                          <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="CrossEntropy">CrossEntropy</SelectItem>
                            <SelectItem value="MSE">MSE</SelectItem>
                            <SelectItem value="BCEWithLogits">BCEWithLogits</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <div className="text-xs">Scheduler</div>
                        <Select value={hp.scheduler} onValueChange={(v)=>setHp({...hp, scheduler:v, cosineRestarts: v==="cosine_warm_restarts" })}>
                          <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">None</SelectItem>
                            <SelectItem value="cosine">Cosine Annealing</SelectItem>
                            <SelectItem value="cosine_warm_restarts">Cosine with Warm Restarts (SGDR)</SelectItem>
                            <SelectItem value="step">Step Decay</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <div className="text-xs">Device</div>
                        <Select value={hp.device} onValueChange={(v)=>setHp({...hp, device:v})}>
                          <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="cuda">CUDA (NVIDIA GPU)</SelectItem>
                            <SelectItem value="mps">MPS (Apple Silicon)</SelectItem>
                            <SelectItem value="cpu">CPU</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <div className="text-xs">Precision</div>
                        <Select value={hp.precision} onValueChange={(v)=>setHp({...hp, precision:v})}>
                          <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="fp32">FP32 (baseline)</SelectItem>
                            <SelectItem value="amp_fp16" disabled={hp.device==='cpu'}>AMP FP16 ~1.3–2.0× (CUDA/MPS)</SelectItem>
                            <SelectItem value="amp_bf16" disabled={!(hp.device==='cpu' || hp.device==='cuda')}>AMP BF16 ~1.1–1.6× (CPU/CUDA)</SelectItem>
                          </SelectContent>
                        </Select>
                        <div className="text-[11px] text-neutral-600 mt-1">
                          {hp.precision==='fp32' ? 'Highest precision' : hp.precision==='amp_fp16' ? 'Typically 1.3–2.0× faster with ~½ memory (CUDA/MPS)' : 'Often 1.1–1.6× faster with ~½ memory (CPU/CUDA)'}
                        </div>
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
                      {hp.scheduler==="step" && (
                        <>
                          <div>
                            <div className="text-xs">Step size (epochs)</div>
                            <Input type="number" value={hp.stepSize} onChange={(e)=>setHp({...hp, stepSize:parseInt(e.target.value||"30",10)})}/>
                          </div>
                          <div>
                            <div className="text-xs">Gamma</div>
                            <Input type="number" value={hp.gamma} onChange={(e)=>setHp({...hp, gamma:parseFloat(e.target.value||"0.1")})}/>
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
                        <div className="text-xs">Validation split</div>
                        <Input type="number" value={hp.valSplit} onChange={(e)=>setHp({...hp, valSplit:Math.min(0.5, Math.max(0.01, parseFloat(e.target.value||"0.1")))})}/>
                      </div>
                      <div>
                        <div className="text-xs">Grad clip (0 to disable)</div>
                        <Input type="number" value={hp.gradClip} onChange={(e)=>setHp({...hp, gradClip:parseFloat(e.target.value||"0.0")})}/>
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
                        <div className="text-xs">Drop Path</div>
                        <Slider value={[hp.stochasticDepth]} min={0} max={0.3} step={0.01} onValueChange={([v])=>setHp({...hp, stochasticDepth:v})}/>
                        <div className="text-xs mt-1">{hp.stochasticDepth.toFixed(2)}</div>
                                           </div>
                      <div className="flex items-center gap-2">
                        <Switch checked={hp.ema} onCheckedChange={(v)=>setHp({...hp, ema:v})}/> <span>EMA</span>
                      </div>
                      {inputMode==='dataset' && dsInfo && (()=>{
                        // find last linear in model
                        const lastLinear = [...model].reverse().map((el,idx)=>({el, idx:model.length-1-idx})).find(({el})=> el.type==='layer' ? el.id==='linear' : el.steps?.some(s=>s.id==='linear'));
                        if(!lastLinear) return null;
                        let mismatch=false;
                        let fixFn=null;
                        if(lastLinear.el.type==='layer'){
                          mismatch = (lastLinear.el.cfg?.outF !== dsInfo.classes);
                          fixFn = ()=> setModel(prev=>{ const arr=[...prev]; arr[lastLinear.idx] = { ...arr[lastLinear.idx], cfg: { ...(arr[lastLinear.idx].cfg||{}), outF: dsInfo.classes } }; return arr; });
                        } else {
                          // block: set last linear inside
                          const steps = lastLinear.el.steps;
                          const j = [...steps].reverse().findIndex(s=>s.id==='linear');
                          if(j>=0){
                            const realJ = steps.length-1-j;
                            mismatch = (steps[realJ].cfg?.outF !== dsInfo.classes);
                            fixFn = ()=> setModel(prev=>{ const arr=[...prev]; const block=arr[lastLinear.idx]; const newSteps=[...block.steps]; newSteps[realJ] = { ...newSteps[realJ], cfg: { ...(newSteps[realJ].cfg||{}), outF: dsInfo.classes } }; arr[lastLinear.idx] = { ...block, steps:newSteps }; return arr; });
                          }
                        }
                        return mismatch ? (
                          <div className="mt-1 p-2 rounded-md bg-amber-50 border border-amber-200 text-amber-800 flex items-center justify-between">
                            <span>Last Linear out ({/* @ts-ignore */ lastLinear.el.type==='layer' ? lastLinear.el.cfg?.outF : 'block'}) ≠ classes ({dsInfo.classes}).</span>
                            <Button size="sm" variant="secondary" onClick={fixFn}>Set to {dsInfo.classes}</Button>
                          </div>
                        ) : null;
                      })()}
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
                      <textarea className="w-full h-[52vh] text-xs font-mono p-2 border border-neutral-200 rounded-xl bg-white/90 shadow-inner" readOnly value={JSON.stringify({ block: block.map(s=>({id:s.id,cfg:s.cfg})), input:{H,W,Cin, mode: inputMode, datasetId, datasetPct}, stats, hyperparams: hp }, null, 2)} />
                      <div className="text-xs text-neutral-500 mt-1">Copy or use Export to download JSON.</div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* Training curves/results tab */}
            <TabsContent value="training">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  <Card>
                    <CardHeader><CardTitle>Training Curves</CardTitle></CardHeader>
                    <CardContent className="text-sm">
                      <MetricsViewer />
                      <div className="text-xs text-neutral-600 mt-2">Curves parse lines starting with "METRIC:" from Run Output.</div>
                    </CardContent>
                  </Card>
                  {inputMode==='dataset' && dsInfo?.classes>1 && (
                    <Card className="mt-3">
                      <CardHeader><CardTitle>Confusion Matrix</CardTitle></CardHeader>
                      <CardContent className="text-sm">
                        <ConfusionMatrix classes={dsInfo.classes} />
                        <div className="text-xs text-neutral-600 mt-2">Computed on test set after training finishes. Saved at checkpoints/confusion.json.</div>
                      </CardContent>
                    </Card>
                  )}
                </div>
                <div className="col-span-2">
                  <Card>
                    <CardHeader><CardTitle>Testing Results</CardTitle></CardHeader>
                    <CardContent className="text-sm">
                      <MetricsSummary />
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
                        <Button onClick={()=>saveGenerated(code)} variant="outline">Save to runner/generated_block.py</Button>
                      </div>
                      <div className="flex items-center gap-2 mb-2 border-t pt-2">
                        <Button
                          variant="default"
                          className="!bg-emerald-600 hover:!bg-emerald-700 !text-white inline-flex items-center gap-2 whitespace-nowrap"
                          onClick={()=>{ setDeviceUsed(null); setDeviceDetecting(true); setRunCounter(c=>c+1); runPython(code, mainCode, hp.device); }}
                          title="Run Python training/eval"
                        >
                          <PlayCircle className="w-4 h-5"/>
                          <span className="leading-none">Run</span>
                        </Button>
                        <Button variant="destructive" onClick={()=>requestStop()} title="Signal the running training to stop early">Stop</Button>
                        <Button variant="outline" onClick={()=>setResumeOpen(true)} title="Resume from a checkpoint">Resume…</Button>
                        {deviceDetecting && (
                          <Badge className="ml-1" variant="secondary"><HelpCircle className="w-3.5 h-3.5 mr-1"/>Device: resolving…</Badge>
                        )}
                        {!deviceDetecting && deviceUsed && (
                            <Badge
                              className={`ml-1 ${
                                deviceUsed === "cpu"
                                  ? "bg-neutral-400 text-white"
                                  : deviceUsed === "cuda"
                                  ? "bg-lime-600 text-white"
                                  : deviceUsed === "mps"
                                  ? "bg-blue-600 text-white"
                                  : ""
                              }`}
                              variant="outline"
                            >
                              {deviceUsed==="cpu" && <TbCpu className="w-3.5 h-3.5 mr-1"/>}
                              {deviceUsed==="cuda" && <BsNvidia className="w-3.5 h-3.5 mr-1"/>}
                              {deviceUsed==="mps" && <FaApple className="w-3.5 h-3.5 mr-1"/>}
                              Device ({deviceUsed})
                            </Badge>
                          )}
                      </div>
                      <CheckpointPicker open={resumeOpen} onClose={()=>setResumeOpen(false)} onPick={({path, mode})=> requestResume(path, mode)} />
                      <CodeEditor language="python" value={code} onChange={setCode} className="h-[30vh]"/>
                      <div className="text-xs text-neutral-500 mt-1">This file now emits CIN/H/W, a GeneratedBlock for the Build tab, and if a Model is defined, a GeneratedModel class that flattens blocks and layers.</div>
                    </CardContent>
                  </Card>

                  <Card className="mt-3">
                    <CardHeader><CardTitle>Training/Testing Script (main.py)</CardTitle></CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-2 mb-2">
                        <Button variant="secondary" onClick={()=>copyText(mainCode)}>Copy</Button>
                        <Button variant="outline" onClick={()=>downloadText('main.py', mainCode)}>Download main.py</Button>
                        <Button onClick={()=>saveMain(mainCode)} variant="outline">Save to .runner/main.py</Button>
                        <Button
                          variant="default"
                          className="!bg-blue-600 hover:!bg-blue-700 !text-white"
                          onClick={()=> setMainCode(generateTrainingScript({ block, model, Cin, H, W, hp, inputMode, datasetId, datasetPct })) }
                          title={inputMode==='dataset'? 'Generate training code for dataset' : 'Custom mode uses random tensors; training disabled'}
                          disabled={inputMode!=='dataset'}
                        >
                          Generate Training Script
                        </Button>
                      </div>
                      <CodeEditor language="python" value={mainCode} onChange={setMainCode} className="h-[26vh]"/>
                      <div className="text-xs text-neutral-500 mt-1">Tip: use the generator to refresh code when you change datasets or hyperparameters.</div>
                    </CardContent>
                  </Card>
                </div>

                <div className="col-span-2">
                  <Card>
                    <CardHeader><CardTitle>Run Output</CardTitle></CardHeader>
                    <CardContent className="text-xs space-y-2">
                      <TrainingProgress />
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
async function runPython(text, main, device){
  try {
    if (typeof text === 'string' && text.length > 0) {
      await saveGenerated(text)
    }
    if (typeof main === 'string' && main.length > 0) {
      await saveMain(main)
    }
  } catch {}
  const params = device ? `?device=${encodeURIComponent(device)}` : ''
  fetch(`/api/run-python${params}`, { method: 'POST' }).catch(()=>{})
}

// Request the Python loop to stop by creating a STOP flag file on the backend
function requestStop(){
  // Try common endpoints; backend may support one of these
  const tryEndpoints = [
  // Prefer explicit backend endpoint, then fall back to file save
  ['/api/stop-training', { method: 'POST' }],
  ['/api/save-file?path=.runner/STOP', { method:'PUT', body:'STOP' }],
  ['/api/save-stop', { method: 'PUT', body: 'STOP' }],
  ['/api/stop', { method: 'POST' }],
  ];
  let ok=false;
  tryEndpoints.reduce((p,[url,init])=> p.catch(()=>fetch(url, init)), Promise.reject())
    .catch(()=>{})
}

// Write a resume request that the Python script will pick up at startup
function requestResume(path, mode){
  const payload = JSON.stringify({ path, mode });
  const tries = [
    ['/api/resume-training', { method:'POST', headers:{'Content-Type':'application/json'}, body: payload }],
    ['/api/save-resume', { method:'PUT', headers:{'Content-Type':'application/json'}, body: payload }],
    ['/api/save-file?path=.runner/RESUME.json', { method:'PUT', headers:{'Content-Type':'application/json'}, body: payload }],
  ];
  tries.reduce((p,[url,init])=> p.catch(()=>fetch(url, init)), Promise.reject()).catch(()=>{})
}

// Polls ANSI-colored output file and renders colorized lines
// Metrics and log moved to components; fmt moved to utils

// ---------------- Palette ----------------
// Palette moved to components

// ---------------- Preset Blocks Panel ----------------
// PresetBlocksPanel moved to components

// Compact preview of a block's composition as a list of layer rows
// BlockLayersPreview moved to components

// ---------------- Layer Config ----------------
function LayerConfig({ selected, onChange, selectedIdx, block, stats, addContext }){
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
      {(l.id==="dropout" || l.id==="droppath") && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Drop prob p</div><Input type="number" step="0.01" min="0" max="1" value={cfg.p ?? (l.id==='dropout'?0.5:0.1)} onChange={e=>set('p', Math.max(0, Math.min(1, parseFloat(e.target.value||"0"))))}/></div>
          <div className="text-[11px] text-neutral-600 col-span-2">
            {l.id==='droppath' ? 'Stochastic Depth randomly drops the residual branch with probability p (train only).' : 'Dropout zeros features with probability p (train only).'}
          </div>
        </div>
      )}
      {l.id==="add" && (
        <div className="grid grid-cols-1 gap-2">
          <div>
            <div className="text-xs">Skip from step</div>
            {addContext?.scope === 'model' ? (
              <Select
                value={(cfg.fromGlobal!=null) ? String(cfg.fromGlobal) : (cfg.from!=null ? String(cfg.from) : "")}
                onValueChange={(v)=>{
                  const g = parseInt(v,10);
                  // Store absolute index when editing at model scope; also keep relative 'from' for display safety
                  set('fromGlobal', g);
                }}
              >
                <SelectTrigger className="w-full"><SelectValue placeholder={"Choose a previous layer from the model"}/></SelectTrigger>
                <SelectContent>
                  {addContext?.flattenedMeta?.filter(m=> m.g < (addContext.baseOffset + selectedIdx)).map(m=> (
                    <SelectItem key={m.g} value={String(m.g)}>
                      #{m.g} {m.blockName ? `[${m.blockName}] `: ''}{m.layerName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Select value={cfg.from!==null && cfg.from!==undefined ? String(cfg.from) : ""} onValueChange={(v)=>set('from', parseInt(v,10))}>
                <SelectTrigger className="w-full"><SelectValue placeholder={selectedIdx>0?"Choose a previous step":"No previous steps"}/></SelectTrigger>
                <SelectContent>
                  {block.map((b,idx)=> idx<selectedIdx ? (
                    <SelectItem key={idx} value={String(idx)}>#{idx} {LAYERS.find(t=>t.id===b.id)?.name || b.id}</SelectItem>
                  ) : null)}
                </SelectContent>
              </Select>
            )}
            <div className="text-[11px] text-neutral-600 mt-1">Tip: residual requires matching (C,H,W). Choose a source before size-changing ops to form an identity path.</div>
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
  if(l.id==="droppath") return `Drop Path p=${s.cfg.p ?? 0.1}`;
  if(l.id==="add") return `Residual Add${typeof s.cfg?.from==='number' ? ` ← #${s.cfg.from}` : ''}`;
  return l.role;
}

// ---------- Compatibility engine ("pins") ----------
// Returns a map id -> { status: 'ok'|'warn'|'bad', label, reason?, synergy? }
function computeNextCompat(block, stats, baseDims){
  const out = {};
  const last = block[block.length-1] || null;
  const lastShape = stats?.outShapes?.[block.length-1] || (block.length===0 ? (baseDims||null) : null);
  const C = lastShape?.C, H = lastShape?.H, W = lastShape?.W;
  LAYERS.forEach(l=>{
    let status = 'ok'; let label = 'Fits'; let reason = '';
    // Basic shape requirements
    const needHW = ['conv','pwconv','dwconv','grpconv','dilconv','bn','gn','ln','relu','gelu','silu','hswish','prelu','maxpool','avgpool','gap','se','eca','cbam','mhsa','winattn','droppath','dropout','add','concat'];
    if(needHW.includes(l.id) && (H==null || W==null || C==null)){
      status='bad'; label='No tensor'; reason='Requires feature map input';
    }
    // Depthwise requires channels>0
    if(l.id==='dwconv' && (!Number.isFinite(C) || C<=0)){
      status='bad'; label='Missing channels';
    }
    // GroupNorm requires C%groups==0; we don't know groups until config, so warn if C not divisible by common groups
    if(l.id==='gn'){
      const commons = [32,16,8,4];
      const okAny = commons.some(g=> Number.isFinite(C) && C>0 && (C % g === 0));
      if(!okAny){ status='warn'; label='Pick groups'; reason='Set groups dividing C'; }
    }
    // Residual add needs a previous save with same shape; we check if any previous out shape matches current
    if(l.id==='add'){
      const before = stats?.outShapes||[];
      const matchIdx = [...before.slice(0, Math.max(0, (block.length)))].findIndex(s=> s && C!=null && H!=null && W!=null && s.C===C && s.H===H && s.W===W);
      if(matchIdx<0){ status='warn'; label='Needs match'; reason='Insert 1×1 or adjust stride to match shapes'; }
    }
    // Concat wants same H,W; warn if last op changed HW relative to its source (unknown source → warn generically)
    if(l.id==='concat'){
      status='warn'; label='Needs same H,W'; reason='Concatenate requires matching spatial dims';
    }
    // Linear prefers flattened features, warn if H,W > 1
    if(l.id==='linear' && Number.isFinite(H) && Number.isFinite(W) && (H>1 || W>1)){
      status='warn'; label='Flattens'; reason='Will flatten C×H×W to features';
    }
    // Drop Path only meaningful if there is or will be an Add
    if(l.id==='droppath'){
      const hasAddAhead = block.some(s=>s.id==='add');
      if(!hasAddAhead){ status='warn'; label='Residual only'; reason='Best used inside residual blocks'; }
    }
    // Synergy hint: simple last+candidate pairs
    let synergy = null;
    if(last){
      if((last.id==='conv'||last.id==='dwconv'||last.id==='grpconv'||last.id==='dilconv') && l.id==='bn') synergy = 'Conv → BN';
      else if(last.id==='bn' && ['relu','gelu','silu','hswish'].includes(l.id)) synergy = 'BN → Activation';
      else if(last.id==='dwconv' && l.id==='pwconv') synergy = 'DW → PW (separable)';
      else if(l.id==='se') synergy = 'SE improves channels';
    }
    out[l.id] = { status, label, reason, synergy };
  });
  return out;
}

// ---------- Utilities: copy/download ----------
// copy/download moved to utils

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
  const needsDropPath = has('droppath') || modelStepsForSe.some(s=>s.id==='droppath');
  const needsLN2d = has('ln') || modelStepsForSe.some(s=>s.id==='ln');
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

  if (needsDropPath){
    emit('class DropPath(nn.Module):');
    emit('    def __init__(self, p: float = 0.0):');
    emit('        super().__init__()');
    emit('        self.p = float(max(0.0, min(1.0, p)))');
    emit('');
    emit('    def forward(self, x):');
    emit('        if not self.training or self.p <= 0.0:');
    emit('            return x');
    emit('        keep = 1.0 - self.p');
    emit('        if keep <= 0.0:');
    emit('            return torch.zeros_like(x)');
    emit('        shape = (x.shape[0],) + (1,) * (x.dim() - 1)');
    emit('        noise = x.new_empty(shape).bernoulli_(keep) / keep');
    emit('        return x * noise');
    emit('');
  }

  if (needsLN2d){
    emit('class LayerNorm2d(nn.Module):');
    emit('    def __init__(self, c, eps=1e-6):');
    emit('        super().__init__()');
    emit('        self.ln = nn.LayerNorm(c, eps=eps)');
    emit('');
    emit('    def forward(self, x):');
    emit('        # apply LayerNorm over channels in channels-last order');
    emit('        x = x.permute(0, 2, 3, 1)');
    emit('        x = self.ln(x)');
    emit('        x = x.permute(0, 3, 1, 2)');
    emit('        return x');
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
      const k = cfg.k || 3; const srt = cfg.s || 1; const g = cfg.g || (s.id==='grpconv'? (cfg.g||2) : 1); const d = s.id==='dilconv' ? (cfg.d||2) : 1; const o = cfg.outC || c; emit(`        self.${name} = nn.Conv2d(${c}, ${o}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${g}, dilation=${d}, bias=False)`); c = o;
    } else if(s.id==='pwconv'){
      const o = cfg.outC || c; emit(`        self.${name} = nn.Conv2d(${c}, ${o}, kernel_size=1, stride=1, padding=0, bias=False)`); c = o;
    } else if(s.id==='dwconv'){
      const k = cfg.k || 3; const srt = cfg.s || 1; emit(`        self.${name} = nn.Conv2d(${c}, ${c}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${c}, bias=False)`);
    } else if(s.id==='bn'){
      emit(`        self.${name} = nn.BatchNorm2d(${c})`);
    } else if(s.id==='gn'){
      const groups = cfg.groups || 32; emit(`        self.${name} = nn.GroupNorm(num_groups=${groups}, num_channels=${c})`);
    } else if(s.id==='ln'){
      emit(`        self.${name} = LayerNorm2d(${c})`);
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
      const p = cfg.p ?? 0.1; emit(`        self.${name} = DropPath(p=${p})`);
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
      const from = (s.cfg && typeof s.cfg.fromGlobal==='number') ? s.cfg.fromGlobal : ((s.cfg && typeof s.cfg.from==='number') ? s.cfg.from : null);
      if(from!==null){ emit(`        x = x + ys[${from}]  # Residual add`); } else { emit(`        # TODO: set a valid source for residual add`); }
    } else if(s.id==='concat'){
      emit(`        # TODO: concat requires specifying sources; keeping x unchanged`);
    } else if(s.id==='linear'){
      emit(`        if x.dim() > 2:`);
      emit(`            x = F.adaptive_avg_pool2d(x, 1)`);
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
      const name = `m_${i}`; const cfg = s.cfg || {};
      if(['conv','grpconv','dilconv'].includes(s.id)){
        const k = cfg.k || 3; const srt = cfg.s || 1; const g = cfg.g || (s.id==='grpconv'? (cfg.g||2) : 1); const d = s.id==='dilconv' ? (cfg.d||2) : 1; const o = cfg.outC || mc; emit(`        self.${name} = nn.Conv2d(${mc}, ${o}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${g}, dilation=${d}, bias=False)`); mc = o;
      } else if(s.id==='pwconv'){
        const o = cfg.outC || mc; emit(`        self.${name} = nn.Conv2d(${mc}, ${o}, kernel_size=1, stride=1, padding=0, bias=False)`); mc = o;
      } else if(s.id==='dwconv'){
        const k = cfg.k || 3; const srt = cfg.s || 1; emit(`        self.${name} = nn.Conv2d(${mc}, ${mc}, kernel_size=${k}, stride=${srt}, padding=${Math.floor((cfg.k||3)/2)}, groups=${mc}, bias=False)`);
      } else if(s.id==='bn'){
        emit(`        self.${name} = nn.BatchNorm2d(${mc})`);
      } else if(s.id==='gn'){
        const groups = cfg.groups || 32; emit(`        self.${name} = nn.GroupNorm(num_groups=${groups}, num_channels=${mc})`);
      } else if(s.id==='ln'){
        emit(`        self.${name} = LayerNorm2d(${mc})`);
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
      } else if(s.id==='droppath'){
        const p = cfg.p ?? 0.1; emit(`        self.${name} = DropPath(p=${p})`);
      } else if(s.id==='mhsa' || s.id==='winattn' || s.id==='concat' || s.id==='deform'){
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
        const from = (s.cfg && typeof s.cfg.fromGlobal==='number') ? s.cfg.fromGlobal : ((s.cfg && typeof s.cfg.from==='number') ? s.cfg.from : null);
        if(from!==null){ emit(`        x = x + ys[${from}]  # Residual add`); } else { emit(`        # TODO: set a valid source for residual add`); }
      } else if(s.id==='concat'){
        emit(`        # TODO: concat requires specifying sources; keeping x unchanged`);
      } else if(s.id==='linear'){
        emit(`        if x.dim() > 2:`);
        emit(`            x = F.adaptive_avg_pool2d(x, 1)`);
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

// Estimate memory (MB): params + optimizer + activations (per batch) + data loader buffers
// extra context accepts: { Cin,H,W,datasetId,datasetPct,valSplit,numWorkers, ema, inputMode }
function estimateMemoryMB(stats, batch, precision = 'fp32', optimizer = 'AdamW', extra = {}){
  const bsz = Math.max(1, Number.isFinite(batch) ? batch : 1);
  const {
    Cin: inC,
    H: inH,
    W: inW,
    datasetId,
    datasetPct = 100,
    valSplit = 0.1,
    numWorkers = 0,
    ema = false,
    inputMode = 'dataset',
  } = (extra||{});

  // Activations follow selected precision; weights/optimizer usually kept in fp32 even with AMP
  const actBytesPer = precision === 'fp32' ? 4 : 2; // fp16/bf16 ~2 bytes

  // Weights (fp32) and optimizer states (approx): SGD(momentum) ~1x, AdamW ~2x
  const paramsBytes = stats.params * 4; // weights
  const optStatesMultiplier = optimizer === 'SGD' ? 1.0 : 2.0;
  const optBytes = stats.params * 4 * optStatesMultiplier;
  const emaBytes = ema ? (stats.params * 4) : 0; // EMA keeps a shadow copy of weights

  // Activation memory across layers; include rough factor for saved tensors + grads
  const actFactor = 2.0;
  const actsBytes = (stats.outShapes||[]).reduce((acc, s)=> acc + (s ? (s.C * s.H * s.W * bsz * actBytesPer) : 0), 0) * actFactor;

  // DataLoader buffered samples (CPU pinned memory); factor in dataset sample size and prefetch
  let dataBytes = 0;
  if (inputMode === 'dataset' && inC && inH && inW && datasetId){
    // Known train sizes (approximate)
    const TRAIN_SIZES = { CIFAR10: 50000, CIFAR100: 50000, MNIST: 60000, FashionMNIST: 60000, STL10: 5000 };
    const baseTrain = TRAIN_SIZES[datasetId] ?? 50000;
    const effTrain = Math.max(1, Math.floor(baseTrain * Math.max(1, Math.min(100, datasetPct)) / 100));
    const nTrain = Math.max(1, effTrain - Math.floor(effTrain * Math.min(Math.max(valSplit, 0), 0.5)));
    const prefetchFactor = 2; // PyTorch default prefetch per worker
    const pipelineBatches = Math.max(0, numWorkers) * prefetchFactor + 2; // +2 for main loop + in-flight GPU copy
    const bufferedSamples = Math.min(nTrain, bsz * Math.max(1, pipelineBatches));
    const bytesPerSample = inC * inH * inW * 4; // CPU tensors are float32 after ToTensor/Normalize
    dataBytes = bufferedSamples * bytesPerSample;
  }

  const totalMB = (paramsBytes + optBytes + emaBytes + actsBytes + dataBytes) / (1024*1024);
  return totalMB;
}
function formatMem(mb){ if(!isFinite(mb)) return '-'; return mb>1024 ? (mb/1024).toFixed(2)+ ' GB' : mb.toFixed(1)+' MB'; }

// Generate a training/testing script based on UI
function generateTrainingScript({ block, model, Cin, H, W, hp, inputMode, datasetId, datasetPct=100 }){
  const dsName = datasetId || 'CIFAR10';
  const hasModel = (model && model.length>0);
  const numClasses = (DATASETS.find(d=>d.id===dsName)?.classes)||10;
  const netClass = hasModel ? 'GeneratedModel' : 'GeneratedBlock';
  const lines=[]; const emit=(s='')=>lines.push(s);
  emit('import torch');
  emit('import torch.nn as nn');
  emit('import torch.optim as optim');
  emit('from contextlib import nullcontext');
  emit('from torch.utils.data import DataLoader, random_split');
  emit('import torchvision');
  emit('import torchvision.transforms as T');
  emit('from tqdm import tqdm');
  emit('from pathlib import Path');
  emit('import time');
  emit('from runner.generated_block import GeneratedBlock, CIN, H, W' + (hasModel? ', GeneratedModel':'') );
  emit('');
  emit('STOP_PATH = Path(".runner/STOP")');
  emit('RESUME_CFG = Path(".runner/RESUME.json")');
  emit('def should_stop():');
  emit('    try: return STOP_PATH.exists()');
  emit('    except Exception: return False');
  emit('');
  emit('def get_resume_request():');
  emit('    try:');
  emit('        import json');
  emit('        if RESUME_CFG.exists():');
  emit('            data = json.loads(RESUME_CFG.read_text())');
  emit('            RESUME_CFG.unlink(missing_ok=True)');
  emit('            return data');
  emit('    except Exception as e:');
  emit('        print("WARN: resume read error:", e)');
  emit('    return None');
  emit('');
  emit('def get_datasets(root="./data", val_split='+hp.valSplit.toString()+', sample_pct='+String(Math.max(1, Math.min(100, datasetPct)))+'):');
  emit(`    mean_std = { 'CIFAR10': ([0.4914,0.4822,0.4465],[0.247,0.243,0.261]), 'CIFAR100': ([0.507,0.487,0.441],[0.267,0.256,0.276]), 'MNIST': ([0.1307],[0.3081]), 'FashionMNIST': ([0.2860],[0.3530]), 'STL10': ([0.4467,0.4398,0.4066],[0.2603,0.2566,0.2713]) }`);
  emit(`    mean,std = mean_std.get('${dsName}', ([0.5]*${Cin}, [0.5]*${Cin}))`);
  emit('    tf_train = T.Compose([T.ToTensor(), T.Normalize(mean, std)])');
  emit('    tf_test  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])');
  if (dsName==='CIFAR10'){
    emit('    full = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf_train)');
    emit('    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf_test)');
  } else if (dsName==='CIFAR100'){
    emit('    full = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=tf_train)');
    emit('    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=tf_test)');
  } else if (dsName==='MNIST'){
    emit('    full = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tf_train)');
    emit('    test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tf_test)');
  } else if (dsName==='FashionMNIST'){
    emit('    full = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=tf_train)');
    emit('    test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=tf_test)');
  } else if (dsName==='STL10'){
    emit("    full = torchvision.datasets.STL10(root=root, split='train', download=True, transform=tf_train)");
    emit("    test = torchvision.datasets.STL10(root=root, split='test', download=True, transform=tf_test)");
  } else {
    emit(`    raise ValueError('Unsupported dataset: ${dsName}')`);
  }
  emit('    # Optional training subset sampling BEFORE val split');
  emit('    sample_pct = max(1, min(100, int(sample_pct)))');
  emit('    if sample_pct < 100:');
  emit('        g = torch.Generator().manual_seed(42)');
  emit('        idx = torch.randperm(len(full), generator=g)[: max(1, int(len(full) * (sample_pct/100.0)))]');
  emit('        full = torch.utils.data.Subset(full, idx.tolist())');
  emit('    n_val = max(1, int(len(full) * val_split))');
  emit('    n_train = max(1, len(full) - n_val)');
  emit('    train, val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))');
  emit('    return train, val, test');
  emit('');
  emit('def get_model(device):');
  emit(`    model = ${netClass}(in_channels=CIN)`);
  emit('    return model.to(device)');
  emit('');
  emit('def get_loss():');
  if(hp.loss==='MSE') emit('    return nn.MSELoss()');
  else if(hp.loss==='BCEWithLogits') emit('    return nn.BCEWithLogitsLoss()');
  else emit('    return nn.CrossEntropyLoss()');
  emit('');
  emit('def get_optimizer(model):');
  if(hp.optimizer==='AdamW') emit(`    return optim.AdamW(model.parameters(), lr=${hp.lr}, weight_decay=${hp.weightDecay})`);
  else emit(`    return optim.SGD(model.parameters(), lr=${hp.lr}, momentum=${hp.momentum??0.9}, weight_decay=${hp.weightDecay})`);
  emit('');
  emit('def get_scheduler(opt):');
  if(hp.scheduler==='none'){
    emit('    return None');
  } else if(hp.scheduler==='cosine_warm_restarts'){
    emit(`    return optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=${hp.T0}, T_mult=${hp.Tmult})`);
  } else if(hp.scheduler==='step'){
    emit(`    return optim.lr_scheduler.StepLR(opt, step_size=${hp.stepSize}, gamma=${hp.gamma})`);
  } else {
    emit('    return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)');
  }
  emit('');
  emit('def accuracy(logits, targets):');
  emit('    if logits.dim()>2: logits = logits.mean(dim=(-1,-2))');
  emit('    preds = logits.argmax(dim=1)');
  emit('    return (preds==targets).float().mean().item()');
  emit('');
  emit('def ensure_trainable(model, sample_loader, device, num_classes):');
  emit('    try:');
  emit('        nparams = sum(p.numel() for p in model.parameters())');
  emit('    except Exception:');
  emit('        nparams = 0');
  emit('    if nparams>0:');
  emit('        return model');
  emit('    print("WARN: Model has no trainable parameters; adding a small linear head.")');
  emit('    # infer feature dim with a dry forward');
  emit('    x0,_ = next(iter(sample_loader))');
  emit('    x0 = x0.to(device)[:1]');
  emit('    with torch.no_grad():');
  emit('        y0 = model(x0)');
  emit('        if y0.dim()>2: y0 = y0.mean(dim=(-1,-2))');
  emit('        feat = y0.shape[1] if y0.dim()==2 else int(y0.numel())');
  emit('    class Wrap(nn.Module):');
  emit('        def __init__(self, base, feat, num_classes):');
  emit('            super().__init__()');
  emit('            self.base = base');
  emit('            self.head = nn.Linear(feat, num_classes)');
  emit('        def forward(self, x):');
  emit('            out = self.base(x)');
  emit('            if out.dim()>2: out = out.mean(dim=(-1,-2))');
  emit('            return self.head(out)');
  emit('    w = Wrap(model, int(feat), '+String(numClasses)+').to(device)');
  emit('    return w');
  emit('');
  emit('def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0.0, global_step_start=0, precision="fp32", scaler=None, epoch_prefix=""):');
  emit('    model.train(); total=0.0; global_step=global_step_start');
  emit('    import os; os.makedirs("checkpoints", exist_ok=True)');
  emit('    desc = f"{epoch_prefix} - train" if epoch_prefix else "train"');
  emit('    for x,y in tqdm(loader, desc=desc, leave=False):');
  emit('        if should_stop():');
  emit('            print("STOP: requested — exiting train loop.")');
  emit('            try: STOP_PATH.unlink(missing_ok=True)');
  emit('            except Exception: pass');
  emit('            break');
  emit('        x=x.to(device); y=y.to(device)');
  emit('        optimizer.zero_grad()');
  emit('        with get_amp_context(device, precision):');
  emit('            out = model(x)');
  emit('            if out.dim()>2: out = out.mean(dim=(-1,-2))');
  emit('            loss = criterion(out, y)');
  emit('        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():');
  emit('            scaler.scale(loss).backward()');
  emit('            if grad_clip>0: scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)');
  emit('            scaler.step(optimizer)');
  emit('            scaler.update()');
  emit('        else:');
  emit('            loss.backward()');
  emit('            if grad_clip>0: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)');
  emit('            optimizer.step()');
  emit('        total += loss.item() * x.size(0)');
  emit('        global_step += 1');
  emit('        # per-step checkpoint')
  emit('        ckpt_step={"model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step}');
  emit('        torch.save(ckpt_step, f"checkpoints/step_{global_step:06d}.pt")');
  emit('        torch.save(ckpt_step, "checkpoints/last_step.pt")');
  emit('    return total / len(loader.dataset), global_step');
  emit('');
  emit('def evaluate(model, loader, criterion, device, precision="fp32", epoch_prefix=""):');
  emit('    model.eval(); total=0.0; accs=0.0');
  emit('    with torch.no_grad():');
  emit('        desc = f"{epoch_prefix} - val" if epoch_prefix else "val"');
  emit('        for x,y in tqdm(loader, desc=desc, leave=False):');
  emit('            if should_stop():');
  emit('                print("STOP: requested — exiting val loop.")');
  emit('                try: STOP_PATH.unlink(missing_ok=True)');
  emit('                except Exception: pass');
  emit('                break');
  emit('            x=x.to(device); y=y.to(device)');
  emit('            with get_amp_context(device, precision):');
  emit('                out = model(x)');
  emit('                if out.dim()>2: out = out.mean(dim=(-1,-2))');
  emit('                loss = criterion(out, y)');
  emit('            total += loss.item() * x.size(0)');
  emit('            accs += accuracy(out, y) * x.size(0)');
  emit('    return total/len(loader.dataset), accs/len(loader.dataset)');
  emit('');
  emit('def confusion_matrix(model, loader, device, num_classes):');
  emit('    import torch');
  emit('    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)');
  emit('    model.eval()');
  emit('    with torch.no_grad():');
  emit('        for x,y in tqdm(loader, desc="confusion", leave=False):');
  emit('            x=x.to(device); y=y.to(device)');
  emit('            out = model(x)');
  emit('            if out.dim()>2: out = out.mean(dim=(-1,-2))');
  emit('            pred = out.argmax(dim=1)');
  emit('            for t,p in zip(y.view(-1), pred.view(-1)):');
  emit('                cm[t.long(), p.long()] += 1');
  emit('    return cm');
  emit('');
  emit('def resolve_device(pref):');
  emit('    if pref=="cuda" and torch.cuda.is_available(): return torch.device("cuda")');
  emit('    if pref=="mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")');
  emit('    if pref=="cpu": return torch.device("cpu")');
  emit('    # fallback to CPU when requested device unavailable');
  emit('    return torch.device("cpu")');
  emit('');
  emit('def get_amp_context(device, precision):');
  emit('    prec = str(precision or "fp32")');
  emit('    dev = str(device)');
  emit('    if prec == "amp_fp16" and dev in ("cuda","mps"):');
  emit('        try: return torch.autocast(device_type=dev, dtype=torch.float16)');
  emit('        except Exception: return nullcontext()');
  emit('    if prec == "amp_bf16":');
  emit('        try: return torch.autocast(device_type=dev, dtype=torch.bfloat16)');
  emit('        except Exception: return nullcontext()');
  emit('    return nullcontext()');
  emit('');
  emit('def reset_peak_mem(device):');
  emit('    dev = str(device)');
  emit('    try:');
  emit('        if dev=="cuda" and torch.cuda.is_available():');
  emit('            torch.cuda.reset_peak_memory_stats()');
  emit('    except Exception:');
  emit('        pass');
  emit('');
  emit('def get_peak_gpu_mem_mb(device):');
  emit('    dev = str(device)');
  emit('    try:');
  emit('        if dev=="cuda" and torch.cuda.is_available():');
  emit('            return float(torch.cuda.max_memory_allocated())/(1024*1024)');
  emit('        if dev=="mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):');
  emit('            return float(torch.mps.current_allocated_memory())/(1024*1024)');
  emit('    except Exception:');
  emit('        return float("nan")');
  emit('    return float("nan")');
  emit('');
  emit('def get_rss_mem_mb():');
  emit('    try:');
  emit('        import os, psutil');
  emit('        return float(psutil.Process(os.getpid()).memory_info().rss)/(1024*1024)');
  emit('    except Exception:');
  emit('        return float("nan")');
  emit('');
  emit('def main():');
  emit(`    device = resolve_device("${hp.device||'cpu'}")`);
  emit('    print("DEVICE:", str(device))');
  emit(`    precision = "${hp.precision||'fp32'}"`);
  emit(`    train_ds, val_ds, test_ds = get_datasets(sample_pct=${Math.max(1, Math.min(100, datasetPct))})`);
  emit(`    train_loader = DataLoader(train_ds, batch_size=${hp.batchSize}, shuffle=True, num_workers=${hp.numWorkers}, pin_memory=True)`);
  emit(`    val_loader = DataLoader(val_ds, batch_size=${hp.batchSize}, shuffle=False, num_workers=${hp.numWorkers})`);
  emit(`    test_loader = DataLoader(test_ds, batch_size=${hp.batchSize}, shuffle=False, num_workers=${hp.numWorkers})`);
  emit('    model = get_model(device)');
  emit('    model = ensure_trainable(model, train_loader, device, '+String(numClasses)+')');
  emit('    criterion = get_loss()');
  emit('    optimizer = get_optimizer(model)');
  emit('    scheduler = get_scheduler(optimizer)');
  emit('    scaler = torch.cuda.amp.GradScaler(enabled=(precision=="amp_fp16" and str(device)=="cuda"))');
  emit('    best=0.0');
  emit('    import os; os.makedirs("checkpoints", exist_ok=True)');
  emit('    global_step = 0');
  emit('    # Resume if requested');
  emit('    resume = get_resume_request()');
  emit('    if resume:');
  emit('        from pathlib import Path as _Path');
  emit('        ckpt_path = _Path(resume.get("path","checkpoints/best.pt"))');
  emit('        if ckpt_path.exists():');
  emit('            try:');
  emit('                payload = torch.load(ckpt_path, map_location=device)');
  emit('                sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload');
  emit('                _incomp = model.load_state_dict(sd, strict=False)');
  emit('                try:');
  emit('                    miss = getattr(_incomp, "missing_keys", [])');
  emit('                    unexp = getattr(_incomp, "unexpected_keys", [])');
  emit('                except Exception:');
  emit('                    miss, unexp = [], []');
  emit('                if miss: print("WARN: missing keys:", miss)');
  emit('                if unexp: print("WARN: unexpected keys:", unexp)');
  emit('                if resume.get("mode","full") == "full":');
  emit('                    if "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])');
  emit('                    if "scheduler" in payload and scheduler:');
  emit('                        try: scheduler.load_state_dict(payload["scheduler"])');
  emit('                        except Exception: pass');
  emit('                    global_step = int(payload.get("global_step", 0))');
  emit('                    start_epoch = int(payload.get("epoch", 0)) + 1');
  emit('                else:');
  emit('                    start_epoch = 1');
  emit('                best = float(payload.get("best", 0.0))');
  emit('                temp = resume.get("mode", "full")');
  emit('                print(f"RESUME: loaded {ckpt_path} mode={temp} start_epoch={start_epoch} best={best:.4f} global_step={global_step}")');
  emit('            except Exception as e:');
  emit('                print("WARN: failed to resume:", e)');
  emit('                start_epoch = 1');
  emit('        else:');
  emit('            print(f"WARN: checkpoint not found: {ckpt_path}")');
  emit('            start_epoch = 1');
  emit('    else:');
  emit('        start_epoch = 1');
  emit(`    for epoch in range(start_epoch, ${hp.epochs}+1):`);
  emit('        if should_stop():');
  emit('            print("STOP: requested — stopping before new epoch.")');
  emit('            try: STOP_PATH.unlink(missing_ok=True)');
  emit('            except Exception: pass');
  emit('            break');
  emit(`        print("EPOCH:", epoch, "/${hp.epochs}")`);
  emit('        reset_peak_mem(device)');
  emit('        _t0 = time.time()');
  emit(`        tr_loss, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=${hp.gradClip}, global_step_start=global_step, precision=precision, scaler=scaler, epoch_prefix=f"Epoch {epoch}/${hp.epochs}")`);
  emit('        val_loss, val_acc = evaluate(model, val_loader, criterion, device, precision, epoch_prefix=f"Epoch {epoch}/'+String(hp.epochs)+'")');
  emit('        epoch_time_sec = max(1e-9, time.time() - _t0)');
  emit('        gpu_mem_mb = get_peak_gpu_mem_mb(device)');
  emit('        rss_mem_mb = get_rss_mem_mb()');
  emit('        try:\n            (scheduler.step() if scheduler else None)\n        except Exception:\n            pass');
  emit('        # track average epoch time so far');
  emit('        if epoch == start_epoch: avg_epoch_time_sec = epoch_time_sec');
  emit('        else: avg_epoch_time_sec = ((epoch - start_epoch) * avg_epoch_time_sec + epoch_time_sec) / max(1, (epoch - start_epoch + 1)) if "avg_epoch_time_sec" in locals() else epoch_time_sec');
  emit('        print(f"METRIC: epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} epoch_time_sec={epoch_time_sec:.3f} avg_epoch_time_sec={avg_epoch_time_sec:.3f} gpu_mem_mb={gpu_mem_mb:.1f} rss_mem_mb={rss_mem_mb:.1f}")')
  emit('        improved = val_acc>best')
  emit('        if improved: best=val_acc; print(f"BEST: val_acc={best:.4f}")')
  emit('        # save checkpoints each epoch and best')
  emit('        ckpt={"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best, "global_step": global_step, "val_acc": float(val_acc)}');
  emit('        if scheduler: ckpt["scheduler"]=scheduler.state_dict()');
  emit('        fname = f"checkpoints/epoch_{epoch:03d}_val{val_acc:.4f}.pt"');
  emit('        torch.save(ckpt, fname)');
  emit('        torch.save(ckpt, "checkpoints/last.pt")');
  emit('        # save best checkpoint only when improved')
  emit('        if improved:');
  emit('            torch.save(ckpt, "checkpoints/best.pt")');
  emit('            print(f"CKPT: type=best path=checkpoints/best.pt epoch={epoch} val_acc={val_acc:.4f}")')
  emit('        print(f"CKPT: type=epoch path={fname} epoch={epoch} val_acc={val_acc:.4f}")')
  emit('    tl, ta = evaluate(model, test_loader, criterion, device, precision)');
  emit('    print(f"TEST: acc={ta:.4f} loss={tl:.4f}")');
  emit('    # Save confusion matrix for classification tasks (num_classes>1)');
  emit('    try:');
  emit('        if '+String(numClasses)+' > 1:');
  emit('            cm = confusion_matrix(model, test_loader, device, '+String(numClasses)+')');
  emit('            import json');
  emit('            import os');
  emit('            os.makedirs("checkpoints", exist_ok=True)');
  emit('            counts = cm.tolist()');
  emit('            # row-normalize');
  emit('            import math');
  emit('            norm = []');
  emit('            for row in counts:');
  emit('                s = float(sum(row))');
  emit('                norm.append([ (x/s if s>0 else 0.0) for x in row ])');
  emit('            Path("checkpoints/confusion.json").write_text(json.dumps({"counts": counts, "normalized": norm}))');
  emit('    except Exception as e:');
  emit('        print("WARN: failed to save confusion matrix:", e)');
  emit('');
  emit('if __name__ == "__main__":');
  emit('    main()');
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
