#!/usr/bin/env python3
"""Read existing rank history/acceptance JSON; never launch a GPU or a task."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from powershard.timing_report import summarize

parser=argparse.ArgumentParser()
parser.add_argument("paths",nargs="+",help="Session JSON, acceptance JSON, or output/powershard directories")
args=parser.parse_args()
print(json.dumps(summarize(args.paths),indent=2,ensure_ascii=False))
