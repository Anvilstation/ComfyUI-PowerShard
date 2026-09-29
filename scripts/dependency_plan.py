#!/usr/bin/env python3
"""Ничего не устанавливает. Защищает установленный torch и анализирует pip --dry-run."""
import argparse,importlib.metadata,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--requirements",default=str(Path(__file__).resolve().parents[1]/"requirements-runtime.txt"));p.add_argument("--output-dir",default="dependency-plan");a=p.parse_args()
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
protected={}
for d in importlib.metadata.distributions():
 name=d.metadata["Name"].lower().replace("_","-")
 if name in {"torch","torchvision","torchaudio"} or name.startswith(("nvidia-","triton")):
  protected[name]=d.version
if "torch" not in protected:
 raise SystemExit("PyTorch не установлен: сначала подготовьте целевую кастомную сборку. Скрипт её не устанавливает.")
(out/"protected-constraints.txt").write_text("\n".join(f"{k}=={v}" for k,v in protected.items())+"\n")
cmd=[sys.executable,"-m","pip","install","--dry-run","--report",str(out/"pip-report.json"),"-c",str(out/"protected-constraints.txt"),"-r",a.requirements]
subprocess.run(cmd,check=True)
d=json.loads((out/"pip-report.json").read_text())
planned=[x["metadata"]["name"].lower().replace("_","-") for x in d.get("install",[])]
blocked=[x for x in planned if x in protected or x in ("torch","torchvision","torchaudio") or x.startswith(("nvidia-","triton"))]
if blocked: raise SystemExit("План меняет защищённый стек: "+", ".join(blocked))
print("PASS: план не меняет torch/CUDA/NCCL. Установка не выполнялась. План:",planned)
