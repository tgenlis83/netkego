import { LAYERS } from "@/lib/constants";

export function generateTorch(block, H, W, Cin){
  return generateTorchAll(block, [], H, W, Cin);
}

export function generateTorchAll(block, modelEls, H, W, Cin){
  // Helper to infer shapes through a sequence of steps
  const inferShapesForSteps = (steps, H0, W0, C0) => {
    let h = H0, w = W0, c = C0;
    const before = [], after = [];
    steps.forEach((s, i) => {
      before[i] = { C: c, H: h, W: w };
      const cfg = s?.cfg || {};
      const id = s?.id;
      if (id === 'patchify') {
        const patch = cfg.patch || 16; const embed = cfg.embedC || c; c = embed; h = Math.floor(h / patch); w = Math.floor(w / patch);
      } else if (id === 'conv' || id === 'grpconv' || id === 'dilconv') {
        const k = cfg.k || 3; const p = Math.floor(k/2); const st = cfg.s || 1; const g = cfg.g || (id==='grpconv' ? (cfg.g || 2) : 1);
        const o = cfg.outC || c; c = o; h = Math.floor((h + 2*p - k)/st + 1); w = Math.floor((w + 2*p - k)/st + 1);
      } else if (id === 'deform') {
        const k = cfg.k || 3; const p = Math.floor(k/2); const st = cfg.s || 1; const o = cfg.outC || c; c = o; h = Math.floor((h + 2*p - k)/st + 1); w = Math.floor((w + 2*p - k)/st + 1);
      } else if (id === 'dwconv') {
        const k = cfg.k || 3; const p = Math.floor(k/2); const st = cfg.s || 1; h = Math.floor((h + 2*p - k)/st + 1); w = Math.floor((w + 2*p - k)/st + 1);
      } else if (id === 'pwconv') {
        const o = cfg.outC || c; c = o;
      } else if (id === 'maxpool' || id === 'avgpool') {
  const k = cfg.k || 2; const st = cfg.s || 2; const p = (id==='maxpool') ? (Number.isFinite(cfg.p) ? cfg.p : ((k===3 && st===2) ? 1 : 0)) : (Number.isFinite(cfg.p) ? cfg.p : 0);
  h = Math.floor((h + 2*p - k)/st + 1); w = Math.floor((w + 2*p - k)/st + 1);
      } else if (id === 'gap') {
        h = 1; w = 1;
      } else if (id === 'concat') {
        c = c * 2;
      } else if (id === 'linear') {
        c = cfg.outF || c; // forward will flatten
      }
      // other ops keep shape
      after[i] = { C: c, H: h, W: w };
    });
    return { before, after };
  };

  const getAddFrom = (s) => {
    const cfg = s?.cfg || {};
    return (typeof cfg.fromGlobal === 'number') ? cfg.fromGlobal : (typeof cfg.from === 'number' ? cfg.from : null);
  };

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
  const needsMHSA = has('mhsa') || modelStepsForSe.some(s=>s.id==='mhsa');
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

  if (needsMHSA){
    emit('class MHSA2D(nn.Module):');
    emit('    def __init__(self, c, heads=8, attn_drop=0.0, proj_drop=0.0):');
    emit('        super().__init__()');
    emit('        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=int(heads), dropout=float(attn_drop), batch_first=True)');
    emit('        self.proj = nn.Linear(c, c)');
    emit('        self.proj_drop = nn.Dropout(p=float(proj_drop))');
    emit('');
    emit('    def forward(self, x):');
    emit('        # x: (N,C,H,W) -> (N,HW,C)');
    emit('        N, C, H, W = x.shape');
    emit('        x_seq = x.permute(0, 2, 3, 1).reshape(N, H*W, C)');
    emit('        out, _ = self.attn(x_seq, x_seq, x_seq)');
    emit('        out = self.proj_drop(self.proj(out))');
    emit('        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)');
    emit('        return out');
    emit('');
  }

  emit('class GeneratedBlock(nn.Module):');
  emit('    def __init__(self, in_channels='+String(Cin)+'):');
  emit('        super().__init__()');
  // Precompute shapes and which residual adds need projections
  const blkShapes = inferShapesForSteps(block, H, W, Cin);
  const blkAddProj = new Map(); // i -> { inC, outC, stride }
  let c = Cin;
  block.forEach((s, i)=>{
    const l = LAYERS.find(x=>x.id===s.id);
    const name = `layer_${i}`;
    const cfg = s.cfg || {};
    if(s.id==='add'){
      const from = getAddFrom(s);
      if(from!=null && from>=0 && from<i){
        const fromShape = blkShapes.after[from];
        const curShapeB = blkShapes.before[i];
        if(fromShape && curShapeB){
          const chDiff = fromShape.C !== curShapeB.C;
          // compute stride even when sizes aren't perfectly divisible (e.g., 7->4 should be stride 2)
          let stride = 1;
          if (fromShape.H !== curShapeB.H || fromShape.W !== curShapeB.W){
            const rH = fromShape.H / Math.max(1, curShapeB.H);
            const rW = fromShape.W / Math.max(1, curShapeB.W);
            const approx = Math.max(1, Math.round((rH + rW) / 2));
            stride = approx;
          }
          if(chDiff){ blkAddProj.set(i, { inC: fromShape.C, outC: curShapeB.C, stride }); }
        }
      }
    }
    if(s.id==='patchify'){
      const patch = cfg.patch || 16; const embed = cfg.embedC || c; emit(`        self.${name} = nn.Conv2d(${c}, ${embed}, kernel_size=${patch}, stride=${patch}, padding=0, bias=False)`); c = embed;
    } else if(['conv','grpconv','dilconv'].includes(s.id)){
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
      const k = cfg.k || 2; const srt = cfg.s || 2; const p = Number.isFinite(cfg.p) ? cfg.p : ((k===3 && srt===2) ? 1 : 0);
      emit(`        self.${name} = nn.MaxPool2d(kernel_size=${k}, stride=${srt}, padding=${p})`);
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
    } else if(s.id==='mhsa'){
      const heads = cfg.heads ?? 8; const ad = cfg.attnDrop ?? 0.0; const pd = cfg.projDrop ?? 0.0; emit(`        self.${name} = MHSA2D(${c}, heads=${heads}, attn_drop=${ad}, proj_drop=${pd})`);
    } else if(s.id==='winattn'){
      emit(`        self.${name} = nn.Identity()  # TODO: Windowed attention placeholder`);
  } else if(s.id==='linear'){
      const o = cfg.outF || 1000; emit(`        self.${name} = nn.Linear(${c}, ${o})`); c = o;
    } else if(s.id==='add' || s.id==='concat' || s.id==='deform'){
      emit(`        self.${name} = nn.Identity()  # TODO: ${l.name} handling in forward`);
      if(s.id==='add' && blkAddProj.has(i)){
        const proj = blkAddProj.get(i);
        const projName = `proj_${i}`;
        emit(`        self.${projName} = nn.Sequential(`);
        emit(`            nn.Conv2d(${proj.inC}, ${proj.outC}, kernel_size=1, stride=${proj.stride}, padding=0, bias=False),`);
        emit(`            nn.BatchNorm2d(${proj.outC})`);
        emit(`        )`);
      }
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
      const from = getAddFrom(s);
      if(from!==null){
        const projName = `proj_${i}`;
        emit(`        res = ys[${from}]`);
        if(blkAddProj.has(i)){
          emit(`        res = self.${projName}(res)`);
        } else {
          emit(`        if res.shape[2:] != x.shape[2:]:`);
          emit(`            res = F.adaptive_avg_pool2d(res, x.shape[2:])`);
        }
        emit(`        x = x + res  # Residual add`);
      } else { emit(`        # TODO: set a valid source for residual add`); }
    } else if(s.id==='concat'){
      emit(`        # TODO: concat requires specifying sources; keeping x unchanged`);
    } else if(s.id==='linear'){
      emit(`        if x.dim() > 2:`);
      emit(`            x = F.adaptive_avg_pool2d(x, 1)`);
      emit(`        x = torch.flatten(x, 1)`);
      emit(`        x = self.${name}(x)`);
      } else if(s.id==='patchify'){
        emit(`        x = self.${name}(x)`);
    } else {
      emit(`        x = self.${name}(x)`);
    }
    emit('        ys.append(x)');
  });
  emit('        return x');

  // If model is defined, also emit a GeneratedModel that flattens blocks/layers
  const modelSteps = flattenModelFromCode(modelEls);
  const mdlShapes = inferShapesForSteps(modelSteps, H, W, Cin);
  const mdlAddProj = new Map();
  modelSteps.forEach((s, i) => {
    if(s.id==='add'){
      const from = getAddFrom(s);
      if(from!=null && from>=0 && from<i){
        const fromShape = mdlShapes.after[from];
        const curShapeB = mdlShapes.before[i];
        if(fromShape && curShapeB){
          const chDiff = fromShape.C !== curShapeB.C;
          let stride = 1;
          if (fromShape.H !== curShapeB.H || fromShape.W !== curShapeB.W){
            const rH = fromShape.H / Math.max(1, curShapeB.H);
            const rW = fromShape.W / Math.max(1, curShapeB.W);
            const approx = Math.max(1, Math.round((rH + rW) / 2));
            stride = approx;
          }
          if(chDiff){ mdlAddProj.set(i, { inC: fromShape.C, outC: curShapeB.C, stride }); }
        }
      }
    }
  });
  if (modelSteps.length > 0){
    emit('');
    emit('class GeneratedModel(nn.Module):');
    emit('    def __init__(self, in_channels='+String(Cin)+'):');
    emit('        super().__init__()');
    let mc = Cin;
    modelSteps.forEach((s, i)=>{
      const name = `m_${i}`; const cfg = s.cfg || {};
      if(s.id==='patchify'){
        const patch = cfg.patch || 16; const embed = cfg.embedC || mc; emit(`        self.${name} = nn.Conv2d(${mc}, ${embed}, kernel_size=${patch}, stride=${patch}, padding=0, bias=False)`); mc = embed;
      } else if(['conv','grpconv','dilconv'].includes(s.id)){
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
        const k = cfg.k || 2; const srt = cfg.s || 2; const p = Number.isFinite(cfg.p) ? cfg.p : ((k===3 && srt===2) ? 1 : 0);
        emit(`        self.${name} = nn.MaxPool2d(kernel_size=${k}, stride=${srt}, padding=${p})`);
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
      } else if(s.id==='mhsa'){
        const heads = cfg.heads ?? 8; const ad = cfg.attnDrop ?? 0.0; const pd = cfg.projDrop ?? 0.0; emit(`        self.${name} = MHSA2D(${mc}, heads=${heads}, attn_drop=${ad}, proj_drop=${pd})`);
  } else if(s.id==='winattn' || s.id==='concat' || s.id==='deform'){
        emit(`        self.${name} = nn.Identity()  # TODO`);
      } else if(s.id==='linear'){
        const o = cfg.outF || 1000; emit(`        self.${name} = nn.Linear(${mc}, ${o})`); mc = o;
      } else if(s.id==='add'){
        emit(`        self.${name} = nn.Identity()`);
        if(mdlAddProj.has(i)){
          const proj = mdlAddProj.get(i);
          const projName = `m_proj_${i}`;
          emit(`        self.${projName} = nn.Sequential(`);
          emit(`            nn.Conv2d(${proj.inC}, ${proj.outC}, kernel_size=1, stride=${proj.stride}, padding=0, bias=False),`);
          emit(`            nn.BatchNorm2d(${proj.outC})`);
          emit(`        )`);
        }
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
        const from = getAddFrom(s);
        if(from!==null){
          const projName = `m_proj_${i}`;
          emit(`        res = ys[${from}]`);
          if(mdlAddProj.has(i)){
            emit(`        res = self.${projName}(res)`);
          } else {
            emit(`        if res.shape[2:] != x.shape[2:]:`);
            emit(`            res = F.adaptive_avg_pool2d(res, x.shape[2:])`);
          }
          emit(`        x = x + res  # Residual add`);
        } else { emit(`        # TODO: set a valid source for residual add`); }
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

// helper used inside code generator
export function flattenModelFromCode(modelEls){
  const out=[]; let base=0;
  for(const el of modelEls){
    if(el.type==='layer'){
      const s = { id: el.id, cfg: { ...(el.cfg||{}) } };
      // Resolve model-level Add elements that declare autoFromOffset
      if(s.id==='add' && typeof s.cfg.autoFromOffset === 'number'){
        const off = Math.max(0, s.cfg.autoFromOffset);
        // Point to the output just before the block start: base - off - 1
        s.cfg = { ...s.cfg, from: Math.max(0, base - off - 1) };
        delete s.cfg.autoFromOffset;
      }
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
