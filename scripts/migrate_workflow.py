#!/usr/bin/env python3
"""Migrate a copy of an old UI/API workflow; never overwrite its source."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from powershard.workflow_migration import migrate_api, migrate_ui

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source"); parser.add_argument("destination")
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.destination)
    if source.resolve() == destination.resolve(): parser.error("Choose a new destination; keep the old graph as backup")
    graph = json.loads(source.read_text())
    result, changes = (migrate_ui if "nodes" in graph else migrate_api)(graph)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(dict(destination=str(destination), changes=changes), indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
