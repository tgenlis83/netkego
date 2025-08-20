import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { HP_PRESETS } from "@/lib/constants";

export default function HyperparametersTab({
  hp,
  setHp,
  modelStats,
  Cin,
  H,
  W,
  datasetId,
  datasetPct,
  inputMode,
  dsInfo,
  estimateMemoryMB,
  formatMem,
  block,
  stats,
  onApplyPreset,
  model,
  setModel,
}){
  return (
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
                  <SelectItem value="plateau">ReduceLROnPlateau</SelectItem>
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
            {hp.scheduler==="plateau" && (
              <>
                <div>
                  <div className="text-xs">Mode</div>
                  <Select value={hp.plateauMode||'min'} onValueChange={(v)=>setHp({...hp, plateauMode:v})}>
                    <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="min">min (reduce when metric stops decreasing)</SelectItem>
                      <SelectItem value="max">max (reduce when metric stops increasing)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <div className="text-xs">Factor</div>
                  <Input type="number" value={hp.plateauFactor??0.1} onChange={(e)=>setHp({...hp, plateauFactor:parseFloat(e.target.value||"0.1")})}/>
                </div>
                <div>
                  <div className="text-xs">Patience (epochs)</div>
                  <Input type="number" value={hp.plateauPatience??10} onChange={(e)=>setHp({...hp, plateauPatience:parseInt(e.target.value||"10",10)})}/>
                </div>
                <div>
                  <div className="text-xs">Threshold</div>
                  <Input type="number" value={hp.plateauThreshold??0.0001} onChange={(e)=>setHp({...hp, plateauThreshold:parseFloat(e.target.value||"0.0001")})}/>
                </div>
                <div>
                  <div className="text-xs">Cooldown</div>
                  <Input type="number" value={hp.plateauCooldown??0} onChange={(e)=>setHp({...hp, plateauCooldown:parseInt(e.target.value||"0",10)})}/>
                </div>
                <div>
                  <div className="text-xs">Min LR</div>
                  <Input type="number" value={hp.plateauMinLr??0.0} onChange={(e)=>setHp({...hp, plateauMinLr:parseFloat(e.target.value||"0")})}/>
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
              <Slider value={[Number.isFinite(hp.stochasticDepth)? hp.stochasticDepth : 0]} min={0} max={0.3} step={0.01} onValueChange={([v])=>setHp({...hp, stochasticDepth:v})}/>
              <div className="text-xs mt-1">{Number.isFinite(hp.stochasticDepth) ? hp.stochasticDepth.toFixed(2) : '0.00'}</div>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={hp.ema} onCheckedChange={(v)=>setHp({...hp, ema:v})}/> <span>EMA</span>
            </div>
            {inputMode==='dataset' && dsInfo && (()=>{
              const lastLinear = [...model].reverse().map((el,idx)=>({el, idx:model.length-1-idx})).find(({el})=> el.type==='layer' ? el.id==='linear' : el.steps?.some(s=>s.id==='linear'));
              if(!lastLinear) return null;
              let mismatch=false; let fixFn=null;
              if(lastLinear.el.type==='layer'){
                mismatch = (lastLinear.el.cfg?.outF !== dsInfo.classes);
                fixFn = ()=> setModel(prev=>{ const arr=[...prev]; arr[lastLinear.idx] = { ...arr[lastLinear.idx], cfg: { ...(arr[lastLinear.idx].cfg||{}), outF: dsInfo.classes } }; return arr; });
              } else {
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

      </div>

      <div className="col-span-2">
        <Card>
          <CardHeader><CardTitle>Hyperparameter Presets</CardTitle></CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-1 gap-3">
            {HP_PRESETS.map(p=> (
              <div key={p.id} className="border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm shadow-sm hover:shadow-md transition">
                <div className="flex items-center justify-between">
                  <div className="font-medium text-sm">{p.name}</div>
                  <Button size="sm" variant="secondary" onClick={()=> onApplyPreset ? onApplyPreset(p) : setHp(prev=>({ ...prev, ...p.details, cosineRestarts: p.details.scheduler==="cosine_warm_restarts" }))}>Apply</Button>
                </div>
                <div className="text-[11px] text-neutral-600 mt-1">{Object.entries(p.details).map(([k,v])=>`${k}:${v}`).join(" · ")}</div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
