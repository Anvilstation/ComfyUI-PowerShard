#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from powershard.checkpoint import Checkpoint,memory_plan
p=argparse.ArgumentParser();p.add_argument("checkpoint");p.add_argument("--output");a=p.parse_args()
c=Checkpoint(a.checkpoint);q=c.quantization()
d=dict(identity=c.identity(),config=c.model_config(),quantization=q,memory=memory_plan(c.tensors,bool(q)))
s=json.dumps(d,indent=2,ensure_ascii=False)
if a.output: Path(a.output).write_text(s)
print(s)
