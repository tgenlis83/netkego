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
import { X, ArrowUp, ArrowDown, Info, Wrench, Layers as LayersIcon, Blocks, Settings2, Download, Link2, AlertTriangle, CheckCircle, Library, Boxes, Box, PlayCircle, LineChart } from "lucide-react";
import { CAT_COLORS, LAYERS, PRESETS, MODEL_PRESETS, SYNERGIES, HP_PRESETS, DATASETS } from "@/lib/constants";
import { copyText, downloadText } from "@/lib/utils";
import Palette from "@/components/builder/Palette";
import PresetBlocksPanel from "@/components/builder/PresetBlocksPanel";
import BlockLayersPreview from "@/components/builder/BlockLayersPreview";
import LayerToken from "@/components/builder/LayerToken";
import AnsiLog from "@/components/builder/AnsiLog";
import ColorChip from "@/components/builder/ColorChip";
import { MetricsViewer, MetricsSummary } from "@/components/builder/Metrics";

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

  // Build Block state
  const [H,setH]=useState(56); const [W,setW]=useState(56); const [Cin,setCin]=useState(64);
  const [inputMode, setInputMode] = useState('custom'); // 'dataset' | 'custom'
  const [datasetId, setDatasetId] = useState('CIFAR10');
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
    plan.forEach(seg=>{
      if(seg.type==='layer'){
        out.push({ type:'layer', id: seg.id, cfg: { ...(seg.cfg||{}) } });
      } else if(seg.type==='blockRef'){
        const repeat = seg.repeat || 1;
        for(let i=0;i<repeat;i++){
          const steps = buildBlockStepsFromPreset(seg.preset, seg.outC, seg.downsample && i===0 ? 2 : 1);
          out.push({ type:'block', name: PRESETS.find(p=>p.id===seg.preset)?.name || seg.preset, steps });
        }
      }
    });
    return out;
  }

  // Use/append a model preset
  const useModelPreset = (mp)=>{
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
  const [hp,setHp]=useState({ optimizer:"SGD", lr:0.1, momentum:0.9, weightDecay:1e-4, scheduler:"cosine", warmup:5, epochs:50, labelSmoothing:0.0, mixup:0.0, cutmix:0.0, stochasticDepth:0.0, ema:false, cosineRestarts:false, T0:10, Tmult:2, batchSize:128, numWorkers:2, loss:"CrossEntropy", valSplit:0.1, gradClip:0.0, stepSize:30, gamma:0.1 });
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
  <Badge variant="secondary">Presets • Build • Hyperparameters • Train</Badge>
        <div className="ml-auto"><Button variant="outline" onClick={exportJSON}><Download className="w-4 h-4 mr-2"/>Export</Button></div>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-3 p-3">
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

          <Palette addLayer={tab==='model' ? addModelLayer : addLayer} mode={tab==='model' ? 'model' : 'build'} />
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
                  <Card>
                    <CardHeader>
                      <CardTitle>Current Block</CardTitle>
                      <div className="text-xs text-neutral-600 mt-1">Use the Layer Palette on the left, or pick a Preset Block on the right.</div>
                    </CardHeader>
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

                <div className="col-span-2 space-y-3">
                  <PresetBlocksPanel onUse={(p)=>importPreset(p)} onAppend={(p)=>appendPresetToBlock(p)} />
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
                            <div className="min-w-0">
                              <div className="text-sm font-medium flex items-center gap-2"><Box className="w-4 h-4"/>{el.name}<span className="px-2 py-0.5 rounded-md text-[11px] bg-neutral-200 text-neutral-800">Block</span></div>
                              <div className="mt-1 flex flex-wrap gap-1.5 items-center">
                                {el.steps.map((s, idx)=> (
                                  <LayerToken key={idx} id={s.id} cfg={s.cfg} size="md" />
                                ))}
                              </div>
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
                      <div>Est. memory (batch {hp.batchSize}): {formatMem(estimateMemoryMB(modelStats, hp.batchSize))}</div>
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
                                  <LayerToken key={idx} id={id} size="md" />
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
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
                        <div className="font-medium text-sm mb-1 flex items-center gap-2"><Boxes className="w-4 h-4"/>Preset Models</div>
                        <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                          {MODEL_PRESETS.map(mp=> (
                            <div key={mp.id} className="border border-neutral-200 rounded-md p-2 bg-white/90 flex items-center justify-between">
                              <div>
                                <div className="text-sm font-medium">{mp.name}</div>
                                <div className="text-[11px] text-neutral-600 truncate">{mp.family} — {mp.description}</div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button size="sm" variant="secondary" onClick={()=>useModelPreset(mp)}>Use</Button>
                                <Button size="sm" variant="outline" onClick={()=>appendModelPreset(mp)}>Append</Button>
                              </div>
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
                      <div>
                        <div className="text-xs">Batch size</div>
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
                        <Button onClick={()=>saveGenerated(code)} variant="default">Save to runner/generated_block.py</Button>
            <Button variant="secondary" onClick={()=>runPython(code, mainCode)} title="Runs in a uv-managed venv on CPU"><PlayCircle className="w-4 h-4 mr-1"/>Run</Button>
                      </div>
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
                        <Button onClick={()=>saveMain(mainCode)} variant="default">Save to .runner/main.py</Button>
                        <Button variant="secondary" onClick={()=> setMainCode(generateTrainingScript({ block, model, Cin, H, W, hp, inputMode, datasetId })) } title={inputMode==='dataset'? 'Generate training code for dataset' : 'Custom mode uses random tensors; training disabled'} disabled={inputMode!=='dataset'}>
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
// Metrics and log moved to components; fmt moved to utils

// ---------------- Palette ----------------
// Palette moved to components

// ---------------- Preset Blocks Panel ----------------
// PresetBlocksPanel moved to components

// Compact preview of a block's composition as a list of layer rows
// BlockLayersPreview moved to components

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

// Estimate memory (MB): params + activations across steps for batch size
function estimateMemoryMB(stats, batch){
  const bytesPer = 4; // float32
  const paramsMB = (stats.params * bytesPer) / (1024*1024);
  const actsMB = (stats.outShapes||[]).reduce((acc, s)=> acc + (s ? (s.C * s.H * s.W * batch * bytesPer) : 0), 0) / (1024*1024);
  return paramsMB + actsMB;
}
function formatMem(mb){ if(!isFinite(mb)) return '-'; return mb>1024 ? (mb/1024).toFixed(2)+ ' GB' : mb.toFixed(1)+' MB'; }

// Generate a training/testing script based on UI
function generateTrainingScript({ block, model, Cin, H, W, hp, inputMode, datasetId }){
  const dsName = datasetId || 'CIFAR10';
  const hasModel = (model && model.length>0);
  const numClasses = (DATASETS.find(d=>d.id===dsName)?.classes)||10;
  const netClass = hasModel ? 'GeneratedModel' : 'GeneratedBlock';
  const lines=[]; const emit=(s='')=>lines.push(s);
  emit('import torch');
  emit('import torch.nn as nn');
  emit('import torch.optim as optim');
  emit('from torch.utils.data import DataLoader, random_split');
  emit('import torchvision');
  emit('import torchvision.transforms as T');
  emit('from runner.generated_block import GeneratedBlock, CIN, H, W' + (hasModel? ', GeneratedModel':'') );
  emit('');
  emit('def get_datasets(root="./data", val_split='+hp.valSplit.toString()+'):');
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
  emit('    n_val = max(1, int(len(full) * val_split))');
  emit('    n_train = len(full) - n_val');
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
  if(hp.scheduler==='cosine_warm_restarts') emit(`    return optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=${hp.T0}, T_mult=${hp.Tmult})`);
  else if(hp.scheduler==='step') emit(`    return optim.lr_scheduler.StepLR(opt, step_size=${hp.stepSize}, gamma=${hp.gamma})`);
  else emit('    return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)');
  emit('');
  emit('def accuracy(logits, targets):');
  emit('    if logits.dim()>2: logits = logits.mean(dim=(-1,-2))');
  emit('    preds = logits.argmax(dim=1)');
  emit('    return (preds==targets).float().mean().item()');
  emit('');
  emit('def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0.0):');
  emit('    model.train(); total=0.0');
  emit('    for x,y in loader:');
  emit('        x=x.to(device); y=y.to(device)');
  emit('        optimizer.zero_grad()');
  emit('        out = model(x)');
  emit('        if out.dim()>2: out = out.mean(dim=(-1,-2))');
  emit('        loss = criterion(out, y)');
  emit('        loss.backward()');
  emit('        if grad_clip>0: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)');
  emit('        optimizer.step()');
  emit('        total += loss.item() * x.size(0)');
  emit('    return total / len(loader.dataset)');
  emit('');
  emit('def evaluate(model, loader, criterion, device):');
  emit('    model.eval(); total=0.0; accs=0.0');
  emit('    with torch.no_grad():');
  emit('        for x,y in loader:');
  emit('            x=x.to(device); y=y.to(device)');
  emit('            out = model(x)');
  emit('            if out.dim()>2: out = out.mean(dim=(-1,-2))');
  emit('            loss = criterion(out, y)');
  emit('            total += loss.item() * x.size(0)');
  emit('            accs += accuracy(out, y) * x.size(0)');
  emit('    return total/len(loader.dataset), accs/len(loader.dataset)');
  emit('');
  emit('def main():');
  emit('    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")');
  emit('    train_ds, val_ds, test_ds = get_datasets()');
  emit(`    train_loader = DataLoader(train_ds, batch_size=${hp.batchSize}, shuffle=True, num_workers=${hp.numWorkers}, pin_memory=True)`);
  emit(`    val_loader = DataLoader(val_ds, batch_size=${hp.batchSize}, shuffle=False, num_workers=${hp.numWorkers})`);
  emit(`    test_loader = DataLoader(test_ds, batch_size=${hp.batchSize}, shuffle=False, num_workers=${hp.numWorkers})`);
  emit('    model = get_model(device)');
  emit('    criterion = get_loss()');
  emit('    optimizer = get_optimizer(model)');
  emit('    scheduler = get_scheduler(optimizer)');
  emit('    best=0.0');
  emit(`    for epoch in range(1, ${hp.epochs}+1):`);
  emit(`        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=${hp.gradClip})`);
  emit('        val_loss, val_acc = evaluate(model, val_loader, criterion, device)');
  emit('        try: scheduler.step()\n        except Exception: pass');
  emit('        print(f"METRIC: epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")')
  emit('        if val_acc>best: best=val_acc; print(f"BEST: val_acc={best:.4f}")')
  emit('    tl, ta = evaluate(model, test_loader, criterion, device)');
  emit('    print(f"TEST: acc={ta:.4f} loss={tl:.4f}")');
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
