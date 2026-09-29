#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from powershard.diagnostics import diagnose
p=argparse.ArgumentParser(description="Диагностика без установки и изменения стека")
p.add_argument("--comfy");p.add_argument("--output")
a=p.parse_args();text=json.dumps(diagnose(a.comfy),indent=2,ensure_ascii=False)
if a.output:
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(text)
print(text)
