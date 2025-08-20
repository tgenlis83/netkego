import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CodeEditor } from "@/components/ui/code-editor";
import { PlayCircle, HelpCircle } from "lucide-react";
import { TbCpu } from "react-icons/tb";
import { BsNvidia } from "react-icons/bs";
import { FaApple } from "react-icons/fa";
import CheckpointPicker from "@/components/builder/CheckpointPicker";

export default function CodeTab({
  code,
  setCode,
  mainCode,
  setMainCode,
  onCopy,
  onDownload,
  onSaveGenerated,
  onSaveMain,
  onRun,
  onStop,
  onResume,
  deviceDetecting,
  deviceUsed,
  generateTrainingScript,
  block,
  model,
  Cin,
  H,
  W,
  hp,
  inputMode,
  datasetId,
  datasetPct,
  preproc,
  resumeOpen,
  setResumeOpen,
}){
  return (
    <div>
      <Card>
        <CardHeader><CardTitle>GeneratedBlock (auto-generated)</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 mb-2">
            <Button variant="secondary" onClick={()=>onCopy(code)}>Copy</Button>
            <Button variant="outline" onClick={()=>onDownload('generated_block.py', code)}>Download .py</Button>
            <Button onClick={()=>onSaveGenerated(code)} variant="outline">Save to runner/generated_block.py</Button>
          </div>
          <div className="flex items-center gap-2 mb-2 border-t pt-2">
            <Button
              variant="default"
              className="!bg-emerald-600 hover:!bg-emerald-700 !text-white inline-flex items-center gap-2 whitespace-nowrap"
              onClick={onRun}
              title="Run Python training/eval"
            >
              <PlayCircle className="w-4 h-5"/>
              <span className="leading-none">Run</span>
            </Button>
            <Button variant="destructive" onClick={onStop} title="Signal the running training to stop early">Stop</Button>
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
          <CheckpointPicker open={resumeOpen} onClose={()=>setResumeOpen(false)} onPick={onResume} />
          <CodeEditor language="python" value={code} onChange={setCode} className="h-[30vh]"/>
          <div className="text-xs text-neutral-500 mt-1">This file now emits CIN/H/W, a GeneratedBlock for the Build tab, and if a Model is defined, a GeneratedModel class that flattens blocks and layers.</div>
        </CardContent>
      </Card>

      <Card className="mt-3">
        <CardHeader><CardTitle>Training/Testing Script (main.py)</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 mb-2">
            <Button variant="secondary" onClick={()=>onCopy(mainCode)}>Copy</Button>
            <Button variant="outline" onClick={()=>onDownload('main.py', mainCode)}>Download main.py</Button>
            <Button onClick={()=>onSaveMain(mainCode)} variant="outline">Save to .runner/main.py</Button>
            <Button
              variant="default"
              className="!bg-blue-600 hover:!bg-blue-700 !text-white"
              onClick={()=> setMainCode(generateTrainingScript({ block, model, Cin, H, W, hp, inputMode, datasetId, datasetPct, preproc })) }
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
  );
}
