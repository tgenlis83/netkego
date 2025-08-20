import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { PREPROC_LAYERS, DATASETS } from '@/lib/constants';
import { Box as BoxIcon, SlidersHorizontal, Maximize, Crop, FlipHorizontal, Sun, RotateCw, Eraser, Wand2, ArrowUp, ArrowDown, Pencil, Trash2 } from 'lucide-react';
import LayerToken from '@/components/builder/LayerToken';

function estimateImpactScore(steps){
  // Heuristic score 0..100 based on augmentations present
  let s = 0;
  steps.forEach(st=>{
    const id = st?.id;
    if(id==='normalize' || id==='toTensor') s += 2;
    if(id==='randomHorizontalFlip') s += (st.cfg?.p||0.5)*10;
    if(id==='colorJitter') s += 12;
    if(id==='randomResizedCrop'||id==='randomCrop') s += 10;
    if(id==='autoAugment') s += 20;
    if(id==='randomErasing') s += (st.cfg?.p||0.25)*20;
    if(id==='randomRotation') s += Math.min(10, (st.cfg?.degrees||0)/3);
    if(id==='resize'||id==='centerCrop') s += 3;
  });
  return Math.max(0, Math.min(100, Math.round(s)));
}

export default function PreprocessingTab({ dsInfo, steps, setSteps }){
  const [selIdx, setSelIdx] = React.useState(-1);
  const selected = selIdx>=0 && selIdx<steps.length ? steps[selIdx] : null;
  const compat = React.useMemo(()=>{
    const ds = dsInfo || DATASETS[0];
    const map={}; PREPROC_LAYERS.forEach(l=>{ map[l.id] = !!l.supports?.(ds); }); return map;
  },[dsInfo]);

  const addLayer=(id)=>{
    const def = PREPROC_LAYERS.find(l=>l.id===id);
    if(!def) return;
    setSteps(prev=>{ const next=[...prev, { id, cfg: { ...(def.defaults||{}) } }]; setSelIdx(next.length-1); return next; });
  };
  const removeIdx=(i)=> setSteps(prev=> prev.filter((_,k)=>k!==i));
  const moveIdx=(i,dir)=> setSteps(prev=>{ const a=[...prev]; const j=i+dir; if(i<0||i>=a.length||j<0||j>=a.length) return a; const t=a[i]; a[i]=a[j]; a[j]=t; return a; });

  const onCfgChange=(cfg)=>{
    if(selIdx<0) return; setSteps(prev=>{ const a=[...prev]; a[selIdx] = { ...a[selIdx], cfg }; return a; });
  };

  const impact = estimateImpactScore(steps);

  const IconFor = (id)=>{
    switch(id){
      case 'toTensor': return <BoxIcon className="w-3.5 h-3.5"/>;
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
      default: return null;
    }
  };

  const applyRecipe = (recipeId)=>{
    const size = Math.min(dsInfo?.H||224, dsInfo?.W||224);
    const base = [ { id:'normalize', cfg:{ autoDataset:true } } ];
    let arr = [];
    if (recipeId==='basic'){
      arr = base;
    } else if (recipeId==='cifar_strong'){
      arr = [ {id:'randomCrop', cfg:{ padding:4, size } }, {id:'randomHorizontalFlip', cfg:{ p:0.5 } }, {id:'normalize', cfg:{ autoDataset:true } }, {id:'randomErasing', cfg:{ p:0.25 } } ];
    } else if (recipeId==='vit_224_train'){
      arr = [ {id:'randomResizedCrop', cfg:{ size:224, scaleMin:0.8, scaleMax:1.0 } }, {id:'randomHorizontalFlip', cfg:{ p:0.5 } }, {id:'colorJitter', cfg:{ brightness:0.2, contrast:0.2, saturation:0.2, hue:0.1 } }, {id:'normalize', cfg:{ autoDataset:true } } ];
    } else if (recipeId==='eval_224'){
      arr = [ {id:'resize', cfg:{ size:224 } }, {id:'centerCrop', cfg:{ size:224 } }, ...base ];
    } else if (recipeId==='mnist_basic'){
      arr = base;
    }
    const ok = (s)=> (PREPROC_LAYERS.find(l=>l.id===s.id)?.supports?.(dsInfo||DATASETS[0]))!==false;
    const next = (arr||[]).filter(ok).map(s=> ({ id:s.id, cfg: { ...(s.cfg||{}) } }));
    setSteps(next);
    setSelIdx(next.length-1);
  };

  return (
    <div className="grid grid-cols-5 gap-3">
      <div className="col-span-3">
        <Card>
          <CardHeader><CardTitle>Preprocessing Pipeline</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {steps.length===0 && <div className="text-neutral-600 text-sm">Add preprocessing steps from the left palette.</div>}
            {steps.map((s,i)=>{
              const def = PREPROC_LAYERS.find(l=>l.id===s.id) || { name:s.id };
              return (
                <div key={i} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 flex items-center justify-between`}>
                  <div className="flex items-start gap-2">
                    <span className="inline-flex items-center justify-center w-7 h-7 rounded-md bg-neutral-200 text-neutral-700" title={def.name}>
                      {IconFor(s.id)}
                    </span>
                    <div>
                      <div className="text-sm font-medium">{def.name}</div>
                      <div className="text-xs text-neutral-600">{def.role||''}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="icon" variant="ghost" title="Move up" onClick={()=>moveIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" title="Move down" onClick={()=>moveIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" title="Edit" onClick={()=>{ setSelIdx(i); }}><Pencil className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" title="Remove" onClick={()=>removeIdx(i)}><Trash2 className="w-4 h-4"/></Button>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        <Card className="mt-3">
          <CardHeader><CardTitle>Estimated Training Impact</CardTitle></CardHeader>
          <CardContent>
            <div className="text-sm mb-1">Score: <b>{impact}</b> / 100</div>
            <div className="relative h-4 rounded-sm overflow-hidden bg-gradient-to-r from-emerald-200 via-amber-200 to-rose-200">
              <div className="absolute inset-0 border border-neutral-200/60 rounded-sm"/>
              <div
                className="absolute -bottom-1 w-0 h-0 border-l-4 border-r-4 border-t-8 border-l-transparent border-r-transparent border-t-neutral-700"
                style={{ left: `calc(${impact}% - 4px)` }}
                title={`Impact ${impact}`}
              />
            </div>
            <div className="text-[11px] text-neutral-600 mt-1">Heuristic: more augmentation generally increases regularization and training time.</div>
          </CardContent>
        </Card>
      </div>
      <div className="col-span-2">
        <Card className="mt-3">
          <CardHeader><CardTitle>Presets</CardTitle></CardHeader>
          <CardContent className="space-y-1">
            <div className="grid grid-cols-2 gap-2">
              <Button size="sm" variant="secondary" onClick={()=>applyRecipe('basic')}>Basic</Button>
              <Button size="sm" variant="secondary" onClick={()=>applyRecipe('cifar_strong')}>CIFAR strong</Button>
              <Button size="sm" variant="secondary" onClick={()=>applyRecipe('vit_224_train')}>ViT 224 train</Button>
              <Button size="sm" variant="secondary" onClick={()=>applyRecipe('eval_224')}>Eval 224</Button>
              <Button size="sm" variant="secondary" onClick={()=>applyRecipe('mnist_basic')}>MNIST basic</Button>
            </div>
          </CardContent>
        </Card>

        <Card className="mt-3">
          <CardHeader><CardTitle>Inspector & Config</CardTitle></CardHeader>
          <CardContent className="text-sm space-y-2">
            {!selected ? (
              <div className="text-neutral-600">Select a step to edit its parameters.</div>
            ) : (
              <PreprocEditor step={selected} onChange={onCfgChange} dsInfo={dsInfo} />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function ArrayInput({ label, value, onChange, lenHint }){
  const [text,setText] = React.useState((value||[]).join(','));
  React.useEffect(()=>{ setText((value||[]).join(',')); }, [value]);
  return (
    <div>
      <div className="text-xs">{label}</div>
      <Input value={text} onChange={(e)=>{ setText(e.target.value); const nums=e.target.value.split(',').map(x=>parseFloat(x.trim())).filter(v=>Number.isFinite(v)); if(!lenHint || nums.length===lenHint) onChange(nums); }} />
    </div>
  );
}

function PreprocEditor({ step, onChange, dsInfo }){
  const id = step?.id; const cfg = step?.cfg || {};
  if(id==='normalize'){
    return (
      <div className="grid grid-cols-2 gap-3">
        <ArrayInput label="Mean" value={cfg.mean} onChange={(v)=> onChange({ ...cfg, mean:v })} lenHint={dsInfo?.C||3} />
        <ArrayInput label="Std" value={cfg.std} onChange={(v)=> onChange({ ...cfg, std:v })} lenHint={dsInfo?.C||3} />
        <div className="col-span-2 text-[11px] text-neutral-600">Tip: Set to dataset stats. Auto-filled when changing dataset.</div>
      </div>
    );
  }
  if(id==='resize' || id==='centerCrop' || id==='randomResizedCrop' || id==='randomCrop'){
    return (
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-xs">Size</div>
          <Input type="number" value={cfg.size||224} onChange={(e)=> onChange({ ...cfg, size: parseInt(e.target.value||'224',10) })} />
        </div>
        {id==='randomResizedCrop' && (
          <>
            <div>
              <div className="text-xs">Scale min</div>
              <Input type="number" step="0.01" value={cfg.scaleMin??0.8} onChange={(e)=> onChange({ ...cfg, scaleMin: parseFloat(e.target.value||'0.8') })} />
            </div>
            <div>
              <div className="text-xs">Scale max</div>
              <Input type="number" step="0.01" value={cfg.scaleMax??1.0} onChange={(e)=> onChange({ ...cfg, scaleMax: parseFloat(e.target.value||'1.0') })} />
            </div>
          </>
        )}
        {id==='randomCrop' && (
          <>
            <div>
              <div className="text-xs">Padding</div>
              <Input type="number" value={cfg.padding??4} onChange={(e)=> onChange({ ...cfg, padding: parseInt(e.target.value||'4',10) })} />
            </div>
          </>
        )}
      </div>
    );
  }
  if(id==='randomHorizontalFlip'){
    return (
      <div>
        <div className="text-xs">Probability</div>
        <Input type="number" step="0.01" value={cfg.p??0.5} onChange={(e)=> onChange({ ...cfg, p: parseFloat(e.target.value||'0.5') })} />
      </div>
    );
  }
  if(id==='colorJitter'){
    return (
      <div className="grid grid-cols-2 gap-3">
        <div><div className="text-xs">Brightness</div><Input type="number" step="0.01" value={cfg.brightness??0.4} onChange={(e)=> onChange({ ...cfg, brightness: parseFloat(e.target.value||'0.4') })} /></div>
        <div><div className="text-xs">Contrast</div><Input type="number" step="0.01" value={cfg.contrast??0.4} onChange={(e)=> onChange({ ...cfg, contrast: parseFloat(e.target.value||'0.4') })} /></div>
        <div><div className="text-xs">Saturation</div><Input type="number" step="0.01" value={cfg.saturation??0.4} onChange={(e)=> onChange({ ...cfg, saturation: parseFloat(e.target.value||'0.4') })} /></div>
        <div><div className="text-xs">Hue</div><Input type="number" step="0.01" value={cfg.hue??0.1} onChange={(e)=> onChange({ ...cfg, hue: parseFloat(e.target.value||'0.1') })} /></div>
      </div>
    );
  }
  if(id==='randomRotation'){
    return (
      <div>
        <div className="text-xs">Degrees</div>
        <Input type="number" value={cfg.degrees??15} onChange={(e)=> onChange({ ...cfg, degrees: parseInt(e.target.value||'15',10) })} />
      </div>
    );
  }
  if(id==='randomErasing'){
    return (
      <div>
        <div className="text-xs">Probability</div>
        <Input type="number" step="0.01" value={cfg.p??0.25} onChange={(e)=> onChange({ ...cfg, p: parseFloat(e.target.value||'0.25') })} />
      </div>
    );
  }
  if(id==='autoAugment'){
    return (
      <div>
        <div className="text-xs">Policy</div>
        <Input value={cfg.policy||'CIFAR10'} onChange={(e)=> onChange({ ...cfg, policy: e.target.value })} />
      </div>
    );
  }
  return <div className="text-neutral-600 text-xs">No editor for {id}. Defaults will be used.</div>;
}
