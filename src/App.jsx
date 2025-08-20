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
import { X, ArrowUp, ArrowDown, Info, Wrench, Layers as LayersIcon, Blocks, Settings2, Download, Link2, AlertTriangle, CheckCircle, Library, Boxes, Box, PlayCircle, LineChart, HelpCircle, Copy, Save, Box as BoxIcon, SlidersHorizontal, Maximize, Crop, FlipHorizontal, Sun, RotateCw, Eraser, Wand2 } from "lucide-react";
import { FaApple } from "react-icons/fa";
import { BsNvidia } from "react-icons/bs";
import { TbCpu } from "react-icons/tb";
import { CAT_COLORS, LAYERS, PRESETS, MODEL_PRESETS, SYNERGIES, HP_PRESETS, DATASETS, PREPROC_LAYERS } from "@/lib/constants";
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
import BuildBlocksTab from "@/components/tabs/BuildBlocksTab";
import BuildModelTab from "@/components/tabs/BuildModelTab";
import HyperparametersTab from "@/components/tabs/HyperparametersTab";
import TrainingTab from "@/components/tabs/TrainingTab";
import CodeTab from "@/components/tabs/CodeTab";
import LayerConfig from "@/components/tabs/LayerConfig";
import PreprocessingTab from "@/components/tabs/PreprocessingTab";
import { computeNextCompat, simulateStatsForSteps, flattenModelWithFromAdjust, modelIdxForFlattened, flattenedIndexForModelIdx, buildModelFromPreset, buildBlockStepsFromPreset } from "@/lib/builderUtils";
import { generateTorchAll } from "@/lib/codegen";

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
export default function App(){
  // UI tabs
  const [tab, setTab] = useState('build');

  // Input / dataset
  const [inputMode, setInputMode] = useState('dataset');
  const [datasetId, setDatasetId] = useState(DATASETS[0]?.id || 'CIFAR10');
  const [datasetPct, setDatasetPct] = useState(100);
  const dsInfo = useMemo(()=> DATASETS.find(d=>d.id===datasetId), [datasetId]);
  const [H, setH] = useState(dsInfo?.H ?? 32);
  const [W, setW] = useState(dsInfo?.W ?? 32);
  const [Cin, setCin] = useState(dsInfo?.C ?? 3);

  // Preprocessing pipeline (dataset transforms)
  const getDatasetMeanStd = React.useCallback((id, c)=>{
    const m = {
      CIFAR10: [[0.4914,0.4822,0.4465],[0.247,0.243,0.261]],
      CIFAR100: [[0.507,0.487,0.441],[0.267,0.256,0.276]],
      MNIST: [[0.1307],[0.3081]],
      FashionMNIST: [[0.2860],[0.3530]],
      STL10: [[0.4467,0.4398,0.4066],[0.2603,0.2566,0.2713]],
    };
    const x = m[id];
    if (x) return x;
    const ch = Math.max(1, c||3);
    return [Array.from({length: ch}, ()=>0.5), Array.from({length: ch}, ()=>0.5)];
  }, []);
  const [preproc, setPreproc] = useState(()=>{
    const [mean,std] = (DATASETS[0] ? getDatasetMeanStd(DATASETS[0].id, DATASETS[0].C) : [[0.5,0.5,0.5],[0.5,0.5,0.5]]);
    return [ { id: 'normalize', cfg: { mean, std, autoDataset: true } } ];
  });

  // Icon helper for preprocessing items
  const PreIconFor = React.useCallback((id)=>{
    switch(id){
      case 'normalize': return <SlidersHorizontal className="w-3.5 h-3.5"/>;
      case 'resize': return <Maximize className="w-3.5 h-3.5"/>;
      case 'centerCrop':
      case 'randomResizedCrop':
      case 'randomCrop': return <Crop className="w-3.5 h-3.5"/>;
      case 'randomHorizontalFlip': return <FlipHorizontal className="w-3.5 h-3.5"/>;
      case 'colorJitter': return <Sun className="w-3.5 h-3.5"/>;
      case 'randomRotation': return <RotateCw className="w-3.5 h-3.5"/>;
      case 'randomErasing': return <Eraser className="w-3.5 h-3.5"/>;
      case 'autoAugment': return <Wand2 className="w-3.5 h-3.5"/>;
      default: return <BoxIcon className="w-3.5 h-3.5"/>;
    }
  }, []);

  React.useEffect(()=>{
    if(inputMode==='dataset' && dsInfo){
      setH(dsInfo.H); setW(dsInfo.W); setCin(dsInfo.C);
    }
  }, [inputMode, dsInfo]);

  // Auto-manage Normalize step on dataset change (ToTensor is always applied in codegen)
  React.useEffect(()=>{
    if (inputMode !== 'dataset') return;
    setPreproc(prev=>{
      let a = Array.isArray(prev) ? [...prev] : [];
      // Update or add Normalize
      const [mean,std] = getDatasetMeanStd(datasetId, Cin);
      const idx = a.findIndex(s=>s.id==='normalize');
      if (idx === -1){
        a.push({ id:'normalize', cfg: { mean, std, autoDataset: true } });
      } else if (a[idx]?.cfg?.autoDataset){
        a[idx] = { ...a[idx], cfg: { ...(a[idx].cfg||{}), mean, std, autoDataset: true } };
      }
      return a;
    });
  }, [datasetId, Cin, inputMode, getDatasetMeanStd]);

  // Block builder state
  const [block, setBlock] = useState([]); // [{id,cfg}]
  const stats = useMemo(()=> simulateStatsForSteps(block, H, W, Cin), [block, Cin, H, W]);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const selected = (selectedIdx>=0 && selectedIdx<block.length) ? block[selectedIdx] : null;
  const [saveName, setSaveName] = useState('MyBlock');
  const [savedBlocks, setSavedBlocks] = useState([]);
  const [showPresetBlocksBuild, setShowPresetBlocksBuild] = useState(false);
  const [editingModelBlockIdx, setEditingModelBlockIdx] = useState(null);

  const addLayer = (step)=>{
    if(!step) return;
    const s = (typeof step === 'string') ? { id: step, cfg: {} } : { id: step?.id, cfg: { ...(step?.cfg||{}) } };
    if(!s.id) return;
    setBlock(prev=>{ const next = [...prev, s]; setSelectedIdx(next.length-1); return next; });
  };
  const moveIdx = (i, dir)=>{
    setBlock(prev=>{
      const a=[...prev]; const j=i+dir; if(i<0||i>=a.length||j<0||j>=a.length) return a; const t=a[i]; a[i]=a[j]; a[j]=t; return a;
    });
  };
  const removeIdx = (i)=> setBlock(prev=> prev.filter((_,k)=>k!==i));
  const duplicateIdx = (i)=> setBlock(prev=>{
    const a=[...prev]; if(i>=0&&i<a.length) a.splice(i+1,0, JSON.parse(JSON.stringify(a[i]))); return a;
  });
  const saveCurrentBlock = ()=>{
    const name = (saveName||'Block').trim(); if(!name) return;
    setSavedBlocks(prev=>[...prev.filter(b=>b.name!==name), { name, steps: JSON.parse(JSON.stringify(block)) }]);
  };
  const importPreset = (preset)=>{
    // Accept saved block (with steps) or catalog preset (with composition)
    let steps = [];
    if (Array.isArray(preset)) {
      steps = preset;
    } else if (preset && Array.isArray(preset.steps)) {
      steps = preset.steps;
    } else if (preset && preset.id) {
      const curC = stats?.outShapes?.[block.length-1]?.C ?? Cin;
      steps = buildBlockStepsFromPreset(preset.id, curC, 1, H, W, Cin);
    }
    if(steps.length){
      setBlock(steps.map(s=>({ id:s.id, cfg: { ...(s.cfg||{}) } })));
      setSelectedIdx(Math.max(0, steps.length-1));
    }
  };
  const appendPresetToBlock = (preset)=>{
    let steps = [];
    if (Array.isArray(preset)) {
      steps = preset;
    } else if (preset && Array.isArray(preset.steps)) {
      steps = preset.steps;
    } else if (preset && preset.id) {
      const curC = stats?.outShapes?.[block.length-1]?.C ?? Cin;
      steps = buildBlockStepsFromPreset(preset.id, curC, 1, H, W, Cin);
    }
    if(steps.length){
      setBlock(prev=>[...prev, ...steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } }))]);
      setSelectedIdx(prev=>Math.max(0, (block?.length||0) + steps.length - 1));
    }
  };

  // Model state
  const [model, setModel] = useState([]); // [{type:'layer'|'block', id?, cfg?, name?, steps?}]
  const [modelSelIdx, setModelSelIdx] = useState(-1);
  const modelSelected = (modelSelIdx>=0 && modelSelIdx<model.length) ? model[modelSelIdx] : null;
  const flattenedSteps = useMemo(()=> flattenModelWithFromAdjust(model), [model]);
  const flattenedModelMeta = useMemo(()=> (
    flattenedSteps.map((step, i)=>{
      const mi = modelIdxForFlattened(model, i);
      const el = model[mi];
      return {
        g: i,
        modelIdx: mi,
        blockName: el && el.type==='block' ? el.name : undefined,
        layerName: LAYERS.find(l=>l.id===step.id)?.name || step.id,
      };
    })
  ), [flattenedSteps, model]);
  const modelStats = useMemo(()=> simulateStatsForSteps(flattenedSteps, H, W, Cin), [flattenedSteps, Cin, H, W]);
  const [showSaved, setShowSaved] = useState(true);
  const [showPresetBlocks, setShowPresetBlocks] = useState(false);
  const [showPresetModels, setShowPresetModels] = useState(false);

  const addModelLayer = (step)=>{
    if(!step) return;
    const s = (typeof step === 'string') ? { id: step, cfg: {} } : { id: step?.id, cfg: { ...(step?.cfg||{}) } };
    if(!s.id) return;
    setModel(prev=>[...prev, { type:'layer', id: s.id, cfg: s.cfg }]);
  };
  const duplicateModelIdx = (i)=> setModel(prev=>{
    const a=[...prev]; if(i>=0&&i<a.length) a.splice(i+1,0, JSON.parse(JSON.stringify(a[i]))); return a;
  });
  const moveModelIdx = (i, dir)=> setModel(prev=>{
    const a=[...prev]; const j=i+dir; if(i<0||i>=a.length||j<0||j>=a.length) return a; const t=a[i]; a[i]=a[j]; a[j]=t; return a;
  });
  const removeModelIdx = (i)=> setModel(prev=> prev.filter((_,k)=>k!==i));
  const editBlockInBuilder = (idx)=>{
    const el = model[idx]; if(!el || el.type!=='block') return;
    setEditingModelBlockIdx(idx);
    setBlock(el.steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } })));
    setTab('build');
  };
  const addSavedBlockToModel = (name)=>{
    const b = savedBlocks.find(x=>x.name===name); if(!b) return;
    setModel(prev=>[...prev, { type:'block', name: b.name, steps: b.steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } })) }]);
  };
  const addPresetToModel = (preset)=>{
    // Accept either a preset object from PRESETS or a plain steps array
    let steps = [];
    let name = 'Preset';
    if (Array.isArray(preset)) {
      steps = preset;
    } else if (preset && Array.isArray(preset.composition)) {
      // Build steps from composition with default cfgs
      steps = preset.composition.map(id=> ({ id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults || {}) } }));
      name = preset.name || name;
    } else if (preset && Array.isArray(preset.steps)) {
      steps = preset.steps;
      name = preset.name || name;
    }
    if(steps.length){
      setModel(prev=>[...prev, { type:'block', name, steps: steps.map(s=>({ id:s.id, cfg:{ ...(s.cfg||{}) } })) }]);
    }
  };
  const applyModelPreset = (preset)=>{
    try{ setModel(buildModelFromPreset(preset.plan || preset, H, W, Cin)); }catch{ /* noop */ }
  };
  const appendModelPreset = (preset)=>{
    try{ const m = buildModelFromPreset(preset.plan || preset, H, W, Cin); setModel(prev=>[...prev, ...m]); }catch{ /* noop */ }
  };

  // Clear model handler (triggered from BuildModelTab destructive button)
  React.useEffect(()=>{
    const onClear = ()=> setModel([]);
    window.addEventListener('netkego:clear-model', onClear);
    return ()=> window.removeEventListener('netkego:clear-model', onClear);
  }, []);

  // Clear block handler
  React.useEffect(()=>{
    const onClear = ()=>{ setBlock([]); setSelectedIdx(-1); };
    window.addEventListener('netkego:clear-block', onClear);
    return ()=> window.removeEventListener('netkego:clear-block', onClear);
  }, []);

  // Hyperparameters
  const [hp, setHp] = useState({
    optimizer: 'AdamW',
    scheduler: 'cosine_warm_restarts',
    lr: 1e-3,
    momentum: 0.9,
    weightDecay: 0.01,
    epochs: 10,
    T0: 10,
    Tmult: 2,
    batchSize: 128,
    numWorkers: 4,
    loss: 'CrossEntropy',
    valSplit: 0.1,
    gradClip: 0.0,
    stepSize: 30,
    gamma: 0.1,
  // ReduceLROnPlateau defaults
  plateauMode: 'min', // metric mode: 'min' on val_loss, 'max' on val_acc
  plateauFactor: 0.1,
  plateauPatience: 10,
  plateauThreshold: 1e-4,
  plateauCooldown: 0,
  plateauMinLr: 0.0,
    device: 'cpu',
    precision: 'fp32', // fp32 | amp_fp16 | amp_bf16
  // defaults for UI controls
  warmup: 0,
  labelSmoothing: 0.0,
  mixup: 0.0,
  cutmix: 0.0,
  stochasticDepth: 0.0,
  ema: false,
  });
  const applyPreset = (p)=>{ setHp(prev=>({ ...prev, ...p.details, cosineRestarts: p.details.scheduler==="cosine_warm_restarts" })); };

  // Code / run state (declare early for effects below)
  const [code, setCode] = useState('');
  const [mainCode, setMainCode] = useState('');
  const [deviceDetecting, setDeviceDetecting] = useState(false);
  const [deviceUsed, setDeviceUsed] = useState(null);
  const [runCounter, setRunCounter] = useState(0);
  const [resumeOpen, setResumeOpen] = useState(false);

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

          { (tab==='build' || tab==='model') && (
            <Palette addLayer={tab==='model' ? addModelLayer : addLayer} mode={tab==='model' ? 'model' : 'build'} compat={tab==='model' ? null : computeNextCompat(block, stats, { C: Cin, H, W })} />
          )}
          { tab==='preproc' && (
            <Card className="mt-3">
              <CardHeader><CardTitle>Preprocessing Palette</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                {PREPROC_LAYERS.filter(l=>l.id!=='toTensor').map(l=>{
                  const supported = !!(l.supports ? l.supports(dsInfo) : true);
                  const pin = (()=>{
                    if(l.id==='colorJitter' && (dsInfo?.C||3)!==3){ return { text:'RGB only', cls:'bg-rose-100 text-rose-700 border-rose-200' }; }
                    if(!supported){ return { text:'Not supported', cls:'bg-rose-100 text-rose-700 border-rose-200' }; }
                    return { text:'OK', cls:'bg-emerald-50 text-emerald-700 border-emerald-200' };
                  })();
                  return (
                    <div key={l.id} className={`border rounded-md p-2 bg-white/90 flex items-center justify-between ${supported ? 'opacity-100' : 'opacity-60'}`}>
                      <div className="flex items-center gap-2">
                        <span className="inline-flex items-center justify-center w-6 h-6 rounded-md bg-neutral-200 text-neutral-700" title={l.name}>
                          {PreIconFor(l.id)}
                        </span>
                        <div>
                          <div className="text-sm font-medium">{l.name}</div>
                          <div className="text-[11px] text-neutral-600">{l.role}</div>
                        </div>
                        <span className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-md border ${pin.cls}`}>{pin.text}</span>
                      </div>
                      <Button size="sm" variant="outline" disabled={!supported} onClick={()=> setPreproc(prev=>[...prev, { id: l.id, cfg: { ...(l.defaults||{}) } }])}>Add</Button>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          )}
        </div>

        <div className="col-span-9">
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList>
              <TabsTrigger value="build"><Wrench className="w-4 h-4 mr-1"/>Build Blocks</TabsTrigger>
              <TabsTrigger value="model"><Boxes className="w-4 h-4 mr-1"/>Build Model</TabsTrigger>
              <TabsTrigger value="preproc"><Box className="w-4 h-4 mr-1"/>Preprocessing</TabsTrigger>
              <TabsTrigger value="hparams"><Settings2 className="w-4 h-4 mr-1"/>Hyperparameters</TabsTrigger>
              <TabsTrigger value="training"><LineChart className="w-4 h-4 mr-1"/>Training</TabsTrigger>
              <TabsTrigger value="code"><Wrench className="w-4 h-4 mr-1"/>PyTorch</TabsTrigger>
            </TabsList>

            {/* Build Blocks tab: now a dedicated component */}
            <TabsContent value="build">
              <BuildBlocksTab
                block={block}
                stats={stats}
                selectedIdx={selectedIdx}
                setSelectedIdx={setSelectedIdx}
                saveName={saveName}
                setSaveName={setSaveName}
                saveCurrentBlock={saveCurrentBlock}
                importPreset={importPreset}
                appendPresetToBlock={appendPresetToBlock}
                showPresetBlocksBuild={showPresetBlocksBuild}
                setShowPresetBlocksBuild={setShowPresetBlocksBuild}
                editingModelBlockIdx={editingModelBlockIdx}
                setEditingModelBlockIdx={setEditingModelBlockIdx}
                moveIdx={moveIdx}
                removeIdx={removeIdx}
                duplicateIdx={duplicateIdx}
                inspector={selected ? (
                  <LayerConfig selected={selected} selectedIdx={selectedIdx} block={block} stats={stats} addContext={{ scope: editingModelBlockIdx!=null ? 'model' : 'block', flattenedMeta: flattenedModelMeta, baseOffset: flattenedIndexForModelIdx(model, editingModelBlockIdx ?? 0) - selectedIdx }} onChange={(cfg)=>{ setBlock(prev=>{ const arr=[...prev]; arr[selectedIdx]={...arr[selectedIdx], cfg}; return arr; }); }} />
                ) : (
                  <div className="text-neutral-600">Select a step (wrench) to edit its parameters.</div>
                )}
                synergy={(
                  <>
                    {SYNERGIES.filter(s=>s.need.every(id=>block.some(b=>b.id===id))).map((s,i)=>(
                      <span key={i} className="px-2 py-1 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-200">{s.why}</span>
                    ))}
                    {SYNERGIES.filter(s=>!s.need.every(id=>block.some(b=>b.id===id)) && s.need.some(id=>block.some(b=>b.id===id))).map((s,i)=>(
                      <span key={i} className="px-2 py-1 rounded-md bg-neutral-50 text-neutral-700 border">Consider: {s.need.map(id=>LAYERS.find(l=>l.id===id)?.name||id).join(" + ")} — {s.why}</span>
                    ))}
                    {block.length===0 && <div className="text-neutral-600">Add layers to see synergy suggestions.</div>}
                  </>
                )}
              />
            </TabsContent>

            {/* Build Model tab */}
            <TabsContent value="model">
              <BuildModelTab
                model={model}
                setModelSelIdx={setModelSelIdx}
                modelStats={modelStats}
                H={H} W={W} Cin={Cin} hp={hp}
                inputMode={inputMode} datasetId={datasetId} datasetPct={datasetPct}
                estimateMemoryMB={estimateMemoryMB}
                formatMem={formatMem}
                duplicateModelIdx={duplicateModelIdx}
                moveModelIdx={moveModelIdx}
                removeModelIdx={removeModelIdx}
                editBlockInBuilder={editBlockInBuilder}
                addSavedBlockToModel={addSavedBlockToModel}
                addPresetToModel={addPresetToModel}
                applyModelPreset={applyModelPreset}
                appendModelPreset={appendModelPreset}
                savedBlocks={savedBlocks}
                showSaved={showSaved}
                setShowSaved={setShowSaved}
                showPresetBlocks={showPresetBlocks}
                setShowPresetBlocks={setShowPresetBlocks}
                showPresetModels={showPresetModels}
                setShowPresetModels={setShowPresetModels}
                flattenedSteps={flattenedSteps}
                flattenedModelMeta={flattenedModelMeta}
                modelIdxForFlattened={modelIdxForFlattened}
                inspector={modelSelected ? (
                  modelSelected.type==='layer' ? (
                    <LayerConfig selected={modelSelected} selectedIdx={flattenedIndexForModelIdx(model, modelSelIdx)} block={flattenModelWithFromAdjust(model).map(s=>s)} stats={modelStats} addContext={{ scope: 'model', flattenedMeta: flattenedModelMeta, baseOffset: flattenedIndexForModelIdx(model, modelSelIdx) }} onChange={(cfg)=>{ setModel(prev=>{ const arr=[...prev]; arr[modelSelIdx]={...arr[modelSelIdx], cfg}; return arr; }); }} />
                  ) : (
                    <div className="text-xs text-neutral-700">
                      <div className="font-medium mb-1">Block: {modelSelected.name}</div>
                      <div>Steps: {modelSelected.steps.map(s=>LAYERS.find(x=>x.id===s.id)?.name||s.id).join(' → ')}</div>
                      <div className="mt-2">Blocks are edited in the Build Blocks tab. Use "Edit" in the list to open and live-edit this block.</div>
                    </div>
                  )
                ) : (
                  <div className="text-neutral-600 text-sm">Select a layer to edit, or use "Edit" on a block to edit its layers.</div>
                )}
              />
            </TabsContent>

            {/* Hyperparameters tab */}
            <TabsContent value="hparams">
              <HyperparametersTab
                hp={hp}
                setHp={setHp}
                modelStats={modelStats}
                Cin={Cin} H={H} W={W}
                datasetId={datasetId}
                datasetPct={datasetPct}
                inputMode={inputMode}
                dsInfo={dsInfo}
                estimateMemoryMB={estimateMemoryMB}
                formatMem={formatMem}
                block={block}
                stats={stats}
                onApplyPreset={applyPreset}
                model={model}
                setModel={setModel}
              />
            </TabsContent>

            {/* Preprocessing tab */}
            <TabsContent value="preproc">
              <PreprocessingTab dsInfo={dsInfo} steps={preproc} setSteps={setPreproc} />
            </TabsContent>

            {/* Training curves/results tab */}
            <TabsContent value="training">
              <TrainingTab inputMode={inputMode} dsInfo={dsInfo} hp={hp} />
            </TabsContent>

            {/* PyTorch preview tab */}
            <TabsContent value="code">
              <div className="grid grid-cols-5 gap-3">
                <div className="col-span-3">
                  <CodeTab
                    code={code}
                    setCode={setCode}
                    mainCode={mainCode}
                    setMainCode={setMainCode}
                    onCopy={copyText}
                    onDownload={downloadText}
                    onSaveGenerated={(t)=>saveGenerated(t)}
                    onSaveMain={(t)=>saveMain(t)}
                    onRun={()=>{ setDeviceUsed(null); setDeviceDetecting(true); setRunCounter(c=>c+1); runPython(code, mainCode, hp.device); }}
                    onStop={()=>requestStop()}
                    onResume={({path, mode})=> requestResume(path, mode)}
                    deviceDetecting={deviceDetecting}
                    deviceUsed={deviceUsed}
                    generateTrainingScript={generateTrainingScript}
                    block={block}
                    model={model}
                    Cin={Cin}
                    H={H}
                    W={W}
                    hp={hp}
                    inputMode={inputMode}
                    datasetId={datasetId}
                    datasetPct={datasetPct}
                    preproc={preproc}
                    resumeOpen={resumeOpen}
                    setResumeOpen={setResumeOpen}
                  />
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
// LayerConfig extracted to components/tabs/LayerConfig.jsx

// renderStepSummary moved to builderUtils

// computeNextCompat moved to builderUtils

// ---------- Utilities: copy/download ----------
// copy/download moved to utils

// Codegen moved to @/lib/codegen

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
function generateTrainingScript({ model, Cin, H, W, hp, datasetId, datasetPct=100, preproc=[] }){
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
  emit(`    _fallback = ([0.5]*${Cin}, [0.5]*${Cin})`);
  emit(`    mean,std = mean_std.get('${dsName}', _fallback)`);
  // Build transforms: PIL ops first (incl. AutoAugment), then ToTensor, then tensor ops (Normalize/Erasing)
  const trainPil = [];
  const testPil = [];
  const trainTensor = [];
  const testTensor = [];
  const addTrainPil = (s)=> trainPil.push(s);
  const addTestPil = (s)=> testPil.push(s);
  const addBothPil = (s)=> { trainPil.push(s); testPil.push(s); };
  const addTrainTensor = (s)=> trainTensor.push(s);
  const addTestTensor = (s)=> testTensor.push(s);
  const addBothTensor = (s)=> { trainTensor.push(s); testTensor.push(s); };
  (Array.isArray(preproc)? preproc: []).forEach(step=>{
    const id = step?.id; const cfg = step?.cfg || {};
    if(id==='toTensor') { return; } // handled automatically between PIL and tensor ops
    if(id==='normalize') { addBothTensor('T.Normalize(mean, std)'); return; }
    if(id==='resize') { const sz = cfg.size ?? 224; addBothPil(`T.Resize(${sz})`); return; }
    if(id==='centerCrop') { const sz = cfg.size ?? 224; addBothPil(`T.CenterCrop(${sz})`); return; }
    if(id==='randomResizedCrop') { const sz = cfg.size ?? 224; const smin = cfg.scaleMin??0.8; const smax = cfg.scaleMax??1.0; addTrainPil(`T.RandomResizedCrop(${sz}, scale=(${smin}, ${smax}))`); addTestPil(`T.Resize(${sz})`); addTestPil(`T.CenterCrop(${sz})`); return; }
    if(id==='randomCrop') { const pad = cfg.padding??4; const size = cfg.size ?? Math.min(H||224, W||224); addTrainPil(`T.RandomCrop(${size}, padding=${pad})`); return; }
    if(id==='randomHorizontalFlip') { const p = cfg.p??0.5; addTrainPil(`T.RandomHorizontalFlip(p=${p})`); return; }
    if(id==='colorJitter') { const b=cfg.brightness??0.4,c=cfg.contrast??0.4,s=cfg.saturation??0.4,h=cfg.hue??0.1; addTrainPil(`T.ColorJitter(brightness=${b}, contrast=${c}, saturation=${s}, hue=${h})`); return; }
    if(id==='randomRotation') { const d = cfg.degrees??15; addTrainPil(`T.RandomRotation(${d})`); return; }
    if(id==='randomErasing') { const p = cfg.p??0.25; addTrainTensor(`T.RandomErasing(p=${p})`); return; }
    if(id==='autoAugment') { addTrainPil(`T.AutoAugment(policy=T.AutoAugmentPolicy.${(cfg.policy||'CIFAR10').toUpperCase()||'CIFAR10'})`); return; }
  });
  // Always ensure Normalize exists; insert as a tensor op after ToTensor
  if(!trainTensor.some(s=>s.startsWith('T.Normalize'))) { trainTensor.push('T.Normalize(mean, std)'); }
  if(!testTensor.some(s=>s.startsWith('T.Normalize'))) { testTensor.push('T.Normalize(mean, std)'); }
  emit('    tf_train = T.Compose(['+[...trainPil, 'T.ToTensor()', ...trainTensor].join(', ')+'])');
  emit('    tf_test  = T.Compose(['+[...testPil, 'T.ToTensor()', ...testTensor].join(', ')+'])');
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
  } else if(hp.scheduler==='plateau'){
    const mode = (hp.plateauMode||'min');
    emit(`    return optim.lr_scheduler.ReduceLROnPlateau(opt, mode='${mode}', factor=${hp.plateauFactor??0.1}, patience=${hp.plateauPatience??10}, threshold=${hp.plateauThreshold??1e-4}, cooldown=${hp.plateauCooldown??0}, min_lr=${hp.plateauMinLr??0.0})`);
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
  // Step scheduler (special case for ReduceLROnPlateau which needs a metric)
  if(hp.scheduler==='plateau'){
    emit('        try:');
    emit('            _mode = getattr(scheduler, "mode", "min") if scheduler else "min"');
    emit('            _metric = val_loss if _mode=="min" else val_acc');
    emit('            (scheduler.step(_metric) if scheduler else None)');
    emit('        except Exception:');
    emit('            pass');
  } else {
    emit('        try:\n            (scheduler.step() if scheduler else None)\n        except Exception:\n            pass');
  }
  emit('        # track average epoch time so far');
  emit('        if epoch == start_epoch: avg_epoch_time_sec = epoch_time_sec');
  emit('        else: avg_epoch_time_sec = ((epoch - start_epoch) * avg_epoch_time_sec + epoch_time_sec) / max(1, (epoch - start_epoch + 1)) if "avg_epoch_time_sec" in locals() else epoch_time_sec');
  emit('        try: cur_lr = float(optimizer.param_groups[0]["lr"])\n        except Exception: cur_lr = float("nan")')
  emit('        print(f"METRIC: epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} epoch_time_sec={epoch_time_sec:.3f} avg_epoch_time_sec={avg_epoch_time_sec:.3f} gpu_mem_mb={gpu_mem_mb:.1f} rss_mem_mb={rss_mem_mb:.1f} lr={cur_lr:.6e}")')
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
// flattenModelFromCode moved to @/lib/codegen
