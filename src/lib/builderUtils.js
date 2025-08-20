// Builder and model utilities extracted from App.jsx for reuse
import { LAYERS, PRESETS } from '@/lib/constants'

// ---- Rough estimators (internal)
const convParams = (inC, outC, k, groups = 1) => (k * k * inC / groups) * outC
const convFLOPs = (H, W, inC, outC, k, groups = 1) => H * W * outC * (k * k * inC / groups)
const dwParams = (inC, k) => inC * k * k
const dwFLOPs = (H, W, inC, k) => H * W * inC * k * k
const pwParams = (inC, outC) => inC * outC
const pwFLOPs = (H, W, inC, outC) => H * W * inC * outC
const seParams = (C, r = 16) => { const hidden = Math.max(1, Math.floor(C / r)); return C * hidden + hidden * C }
const mhsaParams = (C) => 4 * C * C // qkv (3*C*C) + proj (C*C)
const lnBnParams = (C) => 2 * C // gamma + beta

// ---------- Compatibility engine ("pins") ----------
// Returns a map id -> { status: 'ok'|'warn'|'bad', label, reason?, synergy? }
export function computeNextCompat(block, stats, baseDims){
  const out = {}
  const last = block[block.length-1] || null
  const lastShape = stats?.outShapes?.[block.length-1] || (block.length===0 ? (baseDims||null) : null)
  const C = lastShape?.C, H = lastShape?.H, W = lastShape?.W
  LAYERS.forEach(l=>{
    let status = 'ok'; let label = 'Fits'; let reason = ''
    const needHW = ['conv','pwconv','dwconv','grpconv','dilconv','bn','gn','ln','relu','gelu','silu','hswish','prelu','maxpool','avgpool','gap','se','eca','cbam','mhsa','winattn','droppath','dropout','add','concat']
    if(needHW.includes(l.id) && (H==null || W==null || C==null)){
      status='bad'; label='No tensor'; reason='Requires feature map input'
    }
    if(l.id==='dwconv' && (!Number.isFinite(C) || C<=0)){
      status='bad'; label='Missing channels'
    }
    if(l.id==='gn'){
      const commons = [32,16,8,4]
      const okAny = commons.some(g=> Number.isFinite(C) && C>0 && (C % g === 0))
      if(!okAny){ status='warn'; label='Pick groups'; reason='Set groups dividing C' }
    }
    if(l.id==='add'){
      const before = stats?.outShapes||[]
      const matchIdx = [...before.slice(0, Math.max(0, (block.length)))].findIndex(s=> s && C!=null && H!=null && W!=null && s.C===C && s.H===H && s.W===W)
      if(matchIdx<0){ status='warn'; label='Needs match'; reason='Insert 1×1 or adjust stride to match shapes' }
    }
    if(l.id==='concat'){
      status='warn'; label='Needs same H,W'; reason='Concatenate requires matching spatial dims'
    }
    if(l.id==='linear' && Number.isFinite(H) && Number.isFinite(W) && (H>1 || W>1)){
      status='warn'; label='Flattens'; reason='Will flatten C×H×W to features'
    }
    if(l.id==='droppath'){
      const hasAddAhead = block.some(s=>s.id==='add')
      if(!hasAddAhead){ status='warn'; label='Residual only'; reason='Best used inside residual blocks' }
    }
    let synergy = null
    if(last){
      if((last.id==='conv'||last.id==='dwconv'||last.id==='grpconv'||last.id==='dilconv') && l.id==='bn') synergy = 'Conv → BN'
      else if(last.id==='bn' && ['relu','gelu','silu','hswish'].includes(l.id)) synergy = 'BN → Activation'
      else if(last.id==='dwconv' && l.id==='pwconv') synergy = 'DW → PW (separable)'
      else if(l.id==='se') synergy = 'SE improves channels'
    }
    out[l.id] = { status, label, reason, synergy }
  })
  return out
}

// Render summary for list rows
export function renderStepSummary(l, s){
  const lid = l?.id || s?.id
  const lname = l?.name || lid || 'Layer'
  const cfg = s?.cfg || {}
  if(lid==='conv'||lid==='grpconv'||lid==='dilconv') return `${lname} k${cfg.k||3} s${cfg.s||1} → C${cfg.outC||'?'}`
  if(lid==='dwconv') return `Depthwise k${cfg.k||3} s${cfg.s||1}`
  if(lid==='pwconv') return `Pointwise 1×1 → C${cfg.outC||'?'}`
  if(lid==='maxpool'||lid==='avgpool') return `${lname} k${cfg.k||2} s${cfg.s||2}`
  if(lid==='se') return `SE r=${cfg.r||16}`
  if(lid==='linear') return `Linear → ${cfg.outF||'?'}`
  if(lid==='droppath') return `Drop Path p=${cfg.p ?? 0.1}`
  if(lid==='add') return `Residual Add${typeof cfg?.from==='number' ? ` ← #${cfg.from}` : ''}`
  return l?.role || String(lid || '')
}

// Simulate stats for an arbitrary steps array
export function simulateStatsForSteps(stepsArr, H0, W0, C0){
  let h=H0,w=W0,c=C0; let params=0, flops=0; const steps=[]; const synergiesHit=new Set()
  const shapesBefore=[]; const shapesAfter=[]; const issues=[]
  stepsArr.forEach((step,i)=>{
    if(!step || !step.id){
      // Skip unknown entries but keep shapes for alignment
      shapesBefore[i] = { C:c, H:h, W:w }
      shapesAfter[i] = { C:c, H:h, W:w }
      steps.push({ i, name:String(step?.id||'?'), info:'Unknown layer', shape:`(${c}, ${h}, ${w})` })
      return
    }
    const l = LAYERS.find(x=>x.id===step.id) || { id: step.id, name: String(step.id), role: String(step.id) }
    let info=""; let outC=c; let inBefore={C:c,H:h,W:w}
    shapesBefore[i]=inBefore
    if(l.id==='patchify'){
      // Patch embedding via Conv2d(k=patch, s=patch), no padding; outC = embedC
      const patch = step.cfg?.patch ?? 16; const o = step.cfg?.embedC ?? 768;
      const pcount = patch * patch * c * o; // conv kernel params
      const hOut = Math.floor((h - patch)/patch + 1);
      const wOut = Math.floor((w - patch)/patch + 1);
      const f = Math.max(1, hOut) * Math.max(1, wOut) * o * (patch * patch * c);
      params += pcount; flops += f; outC = o; h = hOut; w = wOut; info = `Patchify ${patch} → C=${o}`
    } else if(l.id==='conv'||l.id==='grpconv'||l.id==='dilconv'||l.id==='deform'){
      const k=step.cfg.k||3; const p=Math.floor(k/2); const s=step.cfg.s||1; const g=step.cfg.g||1; const o=step.cfg.outC||c
      const pcount = convParams(c,o,k,g); const f = convFLOPs(h,w,c,o,k,g)
      params+=pcount; flops+=f; outC=o; h=Math.floor((h + 2*p - k)/s + 1); w=Math.floor((w + 2*p - k)/s + 1); info=`Conv ${k}×${k} s${s} g${g} → C=${o}`
    } else if(l.id==='dwconv'){
      const k=step.cfg.k||3; const p=Math.floor(k/2); const s=step.cfg.s||1; const pcount = dwParams(c,k); const f = dwFLOPs(h,w,c,k)
      params+=pcount; flops+=f; outC=c; h=Math.floor((h + 2*p - k)/s + 1); w=Math.floor((w + 2*p - k)/s + 1); info=`DW ${k}×${k} s${s}`
    } else if(l.id==='pwconv'){
      const o=step.cfg.outC||c; const pcount=pwParams(c,o); const f=pwFLOPs(h,w,c,o); params+=pcount; flops+=f; outC=o; info=`PW 1×1 → C=${o}`
    } else if(l.id==='bn'||l.id==='gn'||l.id==='ln'){
      // Small, but include gamma/beta to avoid zero-ish totals for tiny nets
      params += lnBnParams(c); info=l.name
    } else if(l.id==='relu'||l.id==='gelu'||l.id==='silu'||l.id==='hswish'||l.id==='prelu'){
      info=l.name
    } else if(l.id==='maxpool'||l.id==='avgpool'){
      const k=step.cfg?.k||2; const s=step.cfg?.s||2; h=Math.floor((h - k)/s + 1); w=Math.floor((w - k)/s + 1); info=`${l.name} ${k}×${k} s${s}`
    } else if(l.id==='gap'){
      h=1; w=1; info='GlobalAvgPool'
    } else if(l.id==='se'){
      const add=seParams(c, step.cfg?.r||16); params+=add; info=`SE r=${step.cfg?.r||16}`
    } else if(l.id==='mhsa'||l.id==='winattn'){
      if(l.id==='mhsa'){
        const pcount = mhsaParams(c);
        params += pcount;
        const n = Math.max(1, h*w);
        // Rough flops: projections + attention (dominant). Keep coarse to avoid huge numbers.
        flops += n * (4*c*c) + n * n * Math.max(1, Math.floor(c/2));
        info = 'MHSA (~O(N^2))'
      } else {
        info='Windowed Attention'
      }
    } else if(l.id==='droppath'||l.id==='dropout'){
      info=l.name
    } else if(l.id==='add'){
      const from = step.cfg?.from
      if(typeof from!=="number" || from<0 || from>=i){
        issues.push({ type:"add_invalid_from", step:i, msg:"Residual Add needs a valid previous step index." });
        info = 'Residual Add (select source)'
      } else {
        const ref = shapesAfter[from];
        const now = {C:c,H:h,W:w};
        const match = ref && ref.C===now.C && ref.H===now.H && ref.W===now.W;
        if(!match){
          issues.push({ type:"add_mismatch", step:i, from, ref, now, msg:`Shape mismatch: from #${from} (${ref?`${ref.C}×${ref.H}×${ref.W}`:"?"}) vs current ${now.C}×${now.H}×${now.W}` });
        }
        const refName = stepsArr[from] ? (LAYERS.find(v=>v.id===stepsArr[from].id)?.name || stepsArr[from].id) : '?'
        info = `Residual Add ← #${from} ${refName}${!match?" (mismatch)":''}`
      }
    } else if(l.id==='concat'){
      info='Concat (channels++)'; outC = c * 2
    } else if(l.id==='linear'){
      const o=step.cfg?.outF||1000; params += c*o; info=`Linear ${c}→${o}`; outC=o;
      if(h>1 || w>1){ issues.push({ type:'linear_spatial', step:i, msg:`Linear placed with spatial dims ${h}×${w}; will GAP before flatten in codegen.` }) }
    }
    ;
  steps.push({ i, name:l.name, info, shape:`(${outC}, ${h}, ${w})` })
    c=outC; shapesAfter[i]={C:c,H:h,W:w}
  })
  return { steps, outC:c, H:h, W:w, params, flops, tags:[...synergiesHit], issues, inShapes: shapesBefore, outShapes: shapesAfter }
}

// Auto-wire residuals within a steps array
export function autowireResiduals(steps, H0, W0, C0){
  let h=H0, w=W0, c=C0
  const shapesAfter = []
  steps.forEach((s,i)=>{
  const l = LAYERS.find(x=>x.id===s.id) || { id: s?.id }
    if(l.id==='conv'||l.id==='grpconv'||l.id==='dilconv'||l.id==='deform'){
      const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; const o=s.cfg?.outC||c; c=o; h=Math.floor((h + 2*p - k)/st + 1); w=Math.floor((w + 2*p - k)/st + 1)
    } else if(l.id==='dwconv'){
      const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; h=Math.floor((h + 2*p - k)/st + 1); w=Math.floor((w + 2*p - k)/st + 1)
    } else if(l.id==='pwconv'){
      const o=s.cfg?.outC||c; c=o
    } else if(l.id==='maxpool'||l.id==='avgpool'){
      const k=s.cfg?.k||2; const st=s.cfg?.s||2; h=Math.floor((h - k)/st + 1); w=Math.floor((w - k)/st + 1)
    } else if(l.id==='gap'){
      h=1; w=1
    } else if(l.id==='concat'){
      c = c * 2
    } else if(l.id==='linear'){
      c = s.cfg?.outF || c
    }
    shapesAfter[i] = { C:c, H:h, W:w }
  })
  let curC=C0, curH=H0, curW=W0
  return steps.map((s,i)=>{
  const l = LAYERS.find(x=>x.id===s.id) || { id: s?.id }
    if(l.id==='add'){
      const pre = { C:curC, H:curH, W:curW }
      let from = (typeof s.cfg?.from==='number') ? s.cfg.from : null
      const isValid = (j)=> j!=null && j>=0 && j<i && shapesAfter[j] && shapesAfter[j].C===pre.C && shapesAfter[j].H===pre.H && shapesAfter[j].W===pre.W
      if(!isValid(from)){
        from = null; for(let j=i-1;j>=0;j--){ if(isValid(j)){ from=j; break; } }
      }
      s = { ...s, cfg: { ...(s.cfg||{}), ...(from!=null ? { from } : {}) } }
    }
    if(l.id==='conv'||l.id==='grpconv'||l.id==='dilconv'||l.id==='deform'){
      const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; const o=s.cfg?.outC||curC; curC=o; curH=Math.floor((curH + 2*p - k)/st + 1); curW=Math.floor((curW + 2*p - k)/st + 1)
    } else if(l.id==='dwconv'){
      const k=s.cfg?.k||3; const p=Math.floor(k/2); const st=s.cfg?.s||1; curH=Math.floor((curH + 2*p - k)/st + 1); curW=Math.floor((curW + 2*p - k)/st + 1)
    } else if(l.id==='pwconv'){
      const o=s.cfg?.outC||curC; curC=o
    } else if(l.id==='maxpool'||l.id==='avgpool'){
      const k=s.cfg?.k||2; const st=s.cfg?.s||2; curH=Math.floor((curH - k)/st + 1); curW=Math.floor((curW - k)/st + 1)
    } else if(l.id==='gap'){
      curH=1; curW=1
    } else if(l.id==='concat'){
      curC = curC * 2
    } else if(l.id==='linear'){
      curC = s.cfg?.outF || curC
    }
    return s
  })
}

// Adjust residual 'from' indices by a base offset
export function offsetAddIndices(steps, offset){
  return steps.map(s=> (s.id==='add' && typeof s.cfg?.from==='number') ? ({ ...s, cfg: { ...s.cfg, from: s.cfg.from + offset } }) : s)
}

// Flatten model with block-local indices adjusted into global space
export function flattenModelWithFromAdjust(modelEls){
  const out=[]; let base=0
  for(const el of modelEls){
    if(el.type==='layer'){
      out.push({ id: el.id, cfg: { ...(el.cfg||{}) } })
      base += 1
    } else if(el.type==='block'){
      el.steps.forEach((s)=>{
        if(s.id==='add' && typeof s.cfg?.from==='number'){
          s = { ...s, cfg: { ...s.cfg, from: (base + s.cfg.from) } }
        }
        out.push(s)
      })
      base += el.steps.length
    }
  }
  // Resolve model-level Add elements that declare autoFromOffset
  const resolved=[]; let base2=0
  for(let i=0;i<modelEls.length;i++){
    const el = modelEls[i]
    if(el.type==='layer'){
      if(el.id==='add' && typeof el.cfg?.autoFromOffset==='number'){
  // Point to the output just before the block start: base2 - off - 1
  const from = Math.max(0, base2 - Math.max(0, el.cfg.autoFromOffset) - 1)
        resolved.push({ id:'add', cfg:{ ...(el.cfg||{}), from } })
      } else {
        resolved.push({ id: el.id, cfg: { ...(el.cfg||{}) } })
      }
      base2+=1
    } else if(el.type==='block'){
      el.steps.forEach(s=> resolved.push({ id:s.id, cfg:{ ...(s.cfg||{}) } }))
      base2 += el.steps.length
    }
  }
  return resolved
}

// Helpers mapping between model index and flattened index
export function modelIdxForFlattened(model, flatIdx){
  let base=0
  for(let i=0;i<model.length;i++){
    const el=model[i]
    const span = (el.type==='layer') ? 1 : (el.steps?.length||0)
    if(flatIdx < base + span) return i
    base += span
  }
  return -1
}
export function flattenedIndexForModelIdx(model, idx){
  let base=0
  for(let i=0;i<idx;i++){
    const el=model[i]
    base += (el.type==='layer') ? 1 : (el.steps?.length||0)
  }
  return base
}

// Naming helper
export function nextIndexedBlockName(baseName, currentModel){
  const existing = (currentModel||[]).filter(el=> el.type==='block' && typeof el.name==='string')
  const same = existing.filter(el=> el.name===baseName || el.name?.startsWith(baseName + ' #'))
  if(same.length===0) return baseName
  const indices = same.map(el=>{ const m = String(el.name||'').match(/#(\d+)$/); return m? parseInt(m[1],10) : 0 })
  const next = Math.max(0, ...indices) + 1
  return `${baseName} #${next}`
}

// Build a block's steps from a preset id, overriding channels and optional first-layer stride (for downsample in first unit)
export function buildBlockStepsFromPreset(presetId, outC, strideFirst=1, H, W, Cin){
  const preset = PRESETS.find(pp=>pp.id===presetId)
  if(!preset) return []
  // Special handling for ResNet-style blocks to match the paper
  if(presetId==='resnet_basic'){
    let convIdx=0
    const steps = preset.composition.map(id=>{
      const base = { id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }
      if(id==='conv'){
        // Two 3x3 convs, both with outC, stride on first conv when downsampling
        base.cfg.outC = outC
        base.cfg.s = (convIdx===0 ? strideFirst : 1)
        convIdx += 1
      }
      return base
    })
    return autowireResiduals(steps, H, W, Cin)
  }
  if(presetId==='resnet_bottleneck' || presetId==='resnext' || presetId==='preact_bottleneck'){
    let pwIdx=0, k3Idx=0
    const midC = Math.max(1, Math.floor(outC/4))
    const steps = preset.composition.map(id=>{
      const base = { id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }
      if(id==='pwconv'){
        // First 1x1 reduce to midC, last 1x1 expand to outC
        base.cfg.outC = (pwIdx===0 ? midC : outC)
        pwIdx += 1
      } else if(id==='grpconv' || id==='conv'){
        // 3x3 with midC, stride on the first 3x3 when downsampling
        base.cfg.outC = midC
        base.cfg.s = (k3Idx===0 ? strideFirst : 1)
        k3Idx += 1
      }
      return base
    })
    return autowireResiduals(steps, H, W, Cin)
  }
  if(presetId==='vit_encoder'){
    // Keep channels constant; no strides; autowire adds
    const steps = PRESETS.find(pp=>pp.id===presetId)?.composition.map(id=>{
      const base = { id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }
      if(id==='pwconv'){
        // Channel-preserving 1x1: let codegen use current C by omitting outC
        delete base.cfg.outC
      } else if(id==='conv' || id==='dwconv' || id==='grpconv' || id==='dilconv'){
        // No spatial stride within encoder
        base.cfg.s = 1
      }
      return base
    }) || []
    return autowireResiduals(steps, H, W, Cin)
  }
  // Default behavior: set first conv/pwconv to outC (and stride on conv), keep other pwconv at outC
  let firstConvDone=false
  const steps = preset.composition.map(id=>{
    const base = { id, cfg: { ...(LAYERS.find(l=>l.id===id)?.defaults||{}) } }
    if((id==='conv' || id==='pwconv') && !firstConvDone){
      firstConvDone = true
      base.cfg.outC = outC
      if(id==='conv') base.cfg.s = strideFirst
    } else if(id==='pwconv'){
      base.cfg.outC = outC
    }
    return base
  })
  return autowireResiduals(steps, H, W, Cin)
}

// Build a full model from a model preset plan
export function buildModelFromPreset(plan, H, W, Cin){
  const out = []
  const countByName = new Map()
  plan.forEach(seg=>{
    if(seg.type==='layer'){
      out.push({ type:'layer', id: seg.id, cfg: { ...(seg.cfg||{}) } })
    } else if(seg.type==='blockRef'){
      const repeat = seg.repeat || 1
      for(let i=0;i<repeat;i++){
        // Apply stride 2 on the first block when downsample flag set
        const steps = buildBlockStepsFromPreset(seg.preset, seg.outC, (seg.downsample && i===0) ? 2 : 1, H, W, Cin)
        const base = PRESETS.find(p=>p.id===seg.preset)?.name || seg.preset
        const idx = (countByName.get(base) || 0) + 1; countByName.set(base, idx)
        const name = `${base} #${idx}`
        out.push({ type:'block', name, steps })
        out.push({ type:'layer', id:'add', cfg: { autoFromOffset: steps.length } })
        // For ResNet v1 style blocks, apply post-activation ReLU after the residual add
        if (seg.preset==='resnet_basic' || seg.preset==='resnet_bottleneck' || seg.preset==='resnext'){
          out.push({ type:'layer', id:'relu', cfg: {} })
        }
      }
    }
  })
  return out
}
