import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { MetricsViewer, MetricsSummary, ConfusionMatrix } from "@/components/builder/Metrics";

export default function TrainingTab({ inputMode, dsInfo, hp }){
  return (
    <div className="grid grid-cols-5 gap-3">
      <div className="col-span-3">
        <Card>
          <CardHeader><CardTitle>Training Curves</CardTitle></CardHeader>
          <CardContent className="text-sm">
            <MetricsViewer showLR={hp?.scheduler && hp.scheduler !== 'none'} />
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
  );
}
