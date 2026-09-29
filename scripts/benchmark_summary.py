#!/usr/bin/env python3
"""Агрегация только измеренных полей. Кэш и вложенные времена не скрываются."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("worker_report", nargs="?")
p.add_argument("--prompts", nargs="*", default=[])
a = p.parse_args()
output = {"worker_stages": [], "prompts": [], "throughput": "NOT_RECORDED"}
if a.worker_report:
    for row in json.loads(Path(a.worker_report).read_text()):
        if row["sequence"] == 0:
            output["worker_stages"].append({"stage": "model_load", "ranks": [x.get("preflight") for x in row["ranks"]]})
        else:
            output["worker_stages"].append({"stage": row.get("command", "forward_or_preprocess"),
                "rpc_wall_s": row.get("ipc_and_forward_s"), "input_serialization_s": row.get("input_serialization_s"),
                "ranks": [x.get("metrics") for x in row["ranks"]]})
generations = []
for filename in a.prompts:
    report = json.loads(Path(filename).read_text())
    stages = defaultdict(float)
    for node in report["nodes"]:
        if node["status"] == "PASS":
            stages[node["stage"]] += node["wall_s"]
    did_generate = report["status"] == "PASS" and "denoising_node_including_load_and_release" in stages
    if did_generate:
        generations.append(report)
    output["prompts"].append(dict(file=filename, prompt_id=report["prompt_id"], execution_s=report["execution_s"],
                                  stages_s=dict(stages), cached_nodes=report["cached_nodes"], generated=did_generate))
if generations:
    elapsed = sum(r["execution_s"] for r in generations)
    output["throughput"] = dict(completed_generations=len(generations),
                                 jobs_per_execution_second=len(generations)/elapsed,
                                 note_ru="Только задания с фактически исполненным sampler. Сумма времени execution исключает очередь и промежутки между заданиями; не request throughput сервиса.")
    if len({r["host_pid"] for r in generations}) == 1:
        span = max(r["end_monotonic_s"] for r in generations)-min(r["start_monotonic_s"] for r in generations)
        output["throughput"]["jobs_per_observation_second"] = len(generations)/span
        output["throughput"]["observation_span_s"] = span
output["note_ru"] = "Отсутствующая стадия означает кэш/неисполнение, а не нулевую стоимость. Node times включают lazy load; worker times вложены в sampler. Collectives измеряются в опциональных torch traces. Не суммировать вложенные времена."
print(json.dumps(output, indent=2, ensure_ascii=False))
