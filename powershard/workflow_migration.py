"""Pure JSON migration of schema <=4; no CUDA, weights or filesystem writes.

API exports use named fields; LiteGraph UI exports have positional widgets.
The old generator omitted weight_placement, unlike actual node serialization.
Both known positional layouts are handled explicitly; ambiguous ones are refused.
"""
import copy

SCHEMA = 7
CONFIG_FIELDS = ("gpu_ids", "weight_placement", "precision", "attention_backend", "sequence_mode")
TUNING_DEFAULTS = dict(reserve_gib=2., prefetch_blocks=0, numa_policy="none",
                       strict_attention=False, allow_host_wrappers=False, pin_memory=True,
                       sequence_comm_dtype="fp32", prefetch_policy="auto")
OLD_CONFIG_FIELDS = ("gpu_ids", "backend", "precision", "reserve_gib", "timeout_s", "allow_unverified",
                     "release_after_sampling", "cpu_offload", "pin_memory", "prefetch_blocks", "numa_policy",
                     "weight_placement", "attention_backend", "allow_fallback", "memory_policy",
                     "sequence_mode", "sequence_comm_dtype", "allow_host_wrappers")


def config_values(old):
    for key in ("cpu_offload", "release_after_sampling", "allow_fallback"):
        if isinstance(old.get(key), list):
            raise ValueError(f"Linked legacy {key} requires manual migration of boolean policy")
    placement = old.get("weight_placement", "gpu")
    if placement == "gpu" and old.get("cpu_offload", False): placement = "cpu"
    attention = old.get("attention_backend") or ("math" if old.get("attention", "exact_chunked") == "exact_chunked" else old["attention"])
    main = dict(gpu_ids=old.get("gpu_ids", "all"), weight_placement=placement,
                precision=old.get("precision", "fp16"), attention_backend=attention,
                sequence_mode=old.get("sequence_mode", "token"))
    tuning = dict(reserve_gib=old.get("reserve_gib", 2.), prefetch_blocks=old.get("prefetch_blocks", 0),
                  numa_policy=old.get("numa_policy", "none"),
                  strict_attention=not old.get("allow_fallback", True), allow_host_wrappers=old.get("allow_host_wrappers", False),
                  pin_memory=old.get("pin_memory", True), sequence_comm_dtype=old.get("sequence_comm_dtype", "fp32"),
                  prefetch_policy=old.get("memory_policy", "auto"))
    return main, tuning


def migrate_api(graph):
    graph = copy.deepcopy(graph)
    changes = []
    if not isinstance(graph, dict) or "nodes" in graph:
        raise ValueError("Expected API graph: node_id -> class_type/inputs")
    next_id = max((int(k) for k in graph if str(k).isdigit()), default=0)

    def retain_at_loader(source,keep):
        pending=[str(source)];seen=set()
        while pending:
            key=pending.pop()
            if key in seen:continue
            seen.add(key)
            for target,node in graph.items():
                link=node.get("inputs",{}).get("config")
                if not isinstance(link,list) or len(link)!=2 or str(link[0])!=key:continue
                if node.get("class_type")=="PowerShardH3Loader":
                    node["inputs"].setdefault("keep_in_memory",keep)
                elif node.get("class_type")=="PowerShardConfigTuning":pending.append(target)

    def insert(source, kind, socket, values):
        nonlocal next_id
        next_id += 1; key = str(next_id)
        while key in graph: next_id += 1; key = str(next_id)
        for node in graph.values():
            for name, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) == str(source) and value[1] == 0:
                    node["inputs"][name] = [key, 0]
        graph[key] = dict(class_type=kind, inputs=dict(values, **{socket: [str(source), 0]}))
        changes.append(f"{source} -> {key}: {kind}")

    for key, node in list(graph.items()):
        kind, old = node.get("class_type"), dict(node.get("inputs", {}))
        if kind == "PowerShardConfig" and (set(old) != set(CONFIG_FIELDS)):
            if "release_after_sampling" in old:retain_at_loader(key,not old["release_after_sampling"])
            main, tuning = config_values(old)
            node["inputs"] = main
            if tuning != TUNING_DEFAULTS: insert(key, "PowerShardConfigTuning", "config", tuning)
            changes.append(f"{key}: Config schema {SCHEMA}; backend=fsdp2_sequence; no RPC timeout")
        elif kind=="PowerShardConfigTuning" and "keep_workers" in old:
            if isinstance(old["keep_workers"],list):raise ValueError("Linked keep_workers requires manual migration to H3 Loader")
            retain_at_loader(key,old["keep_workers"])
            node["inputs"].pop("keep_workers")
            changes.append(f"{key}: keep_workers moved to H3 Loader")
        elif kind == "PowerShardH3FP16Patcher" and any(k in old for k in ("enabled", "mlp_chunk_tokens", "mlp_chunk_mode")):
            if any(isinstance(old.get(k),list) for k in ("enabled","fp16_safe")):
                raise ValueError("Linked legacy enabled/fp16_safe requires manual migration of boolean policy")
            safe = bool(old.get("enabled", True) and old.get("fp16_safe", True))
            node["inputs"] = dict(model=old["model"], fp16_safe=safe, debug_finite=old.get("debug_finite", False))
            mode = old.get("mlp_chunk_mode", "manual") if safe else "off"
            insert(key, "PowerShardH3MLP", "model", dict(mode=mode, chunk_tokens=old.get("mlp_chunk_tokens", 512)))
        elif kind == "PowerShardH3QwenLoader" and any(k in old for k in ("mlp_chunk_mode", "mlp_chunk_tokens")):
            node["inputs"].pop("mlp_chunk_mode", None); node["inputs"].pop("mlp_chunk_tokens", None)
            insert(key, "PowerShardQwenMLP", "clip", dict(mode=old.get("mlp_chunk_mode", "auto"), chunk_tokens=old.get("mlp_chunk_tokens", 4096)))
    return graph, changes


def _old_ui_config(values):
    # Original build_workflows.py writes only 14 widgets, skipping placement.
    if len(values) == 14:
        names = OLD_CONFIG_FIELDS[:11] + ("attention_backend", "allow_fallback", "memory_policy")
    elif 15 <= len(values) <= 18 and values[11] in ("gpu", "cpu", "ats"):
        names = OLD_CONFIG_FIELDS[:len(values)]
    elif len(values) == 7:
        names = OLD_CONFIG_FIELDS[:7]
    else:
        raise ValueError("Unknown legacy PowerShardConfig widgets layout; use the original named API export")
    return dict(zip(names, values))


def migrate_ui(graph):
    graph = copy.deepcopy(graph)
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("links"), list):
        raise ValueError("Expected LiteGraph UI graph with nodes/links arrays")
    # Object links (new graph format) are deliberately not silently rewritten.
    if any(not isinstance(link, list) or len(link) < 6 for link in graph["links"]):
        raise ValueError("Migration supports classic LiteGraph link arrays; export named API JSON for other formats")
    nodes, links, changes = graph["nodes"], graph["links"], []
    next_node = max([int(n["id"]) for n in nodes]+[int(graph.get("last_node_id", 0))])
    next_link = max([int(l[0]) for l in links]+[int(graph.get("last_link_id", 0))])

    def retain_at_loader(source,keep):
        pending=[source];seen=set();by_id={node["id"]:node for node in nodes}
        while pending:
            key=pending.pop()
            if key in seen:continue
            seen.add(key)
            for link in links:
                if link[1]!=key or link[2]!=0:continue
                target=by_id[link[3]]
                if target.get("type")=="PowerShardH3Loader" and len(target.get("widgets_values",[]))==1:
                    target["widgets_values"].append(keep)
                elif target.get("type")=="PowerShardConfigTuning":pending.append(target["id"])

    def insert(source, kind, socket, dtype, values):
        nonlocal next_node, next_link
        next_node += 1; next_link += 1
        source_id = source["id"]
        outgoing = []
        for link in links:
            if link[1] == source_id and link[2] == 0:
                link[1] = next_node; outgoing.append(link[0])
        pos = source.get("pos", [0, 0])
        added = dict(id=next_node, type=kind, pos=[pos[0]+370, pos[1]+100], size=[330, 210],
                     flags={}, order=len(nodes), mode=0,
                     inputs=[dict(name=socket, type=dtype, link=next_link)],
                     outputs=[dict(name=dtype, type=dtype, links=outgoing, slot_index=0)],
                     properties={"Node name for S&R": kind, "powershard_schema": SCHEMA},
                     widgets_values=list(values.values()))
        outputs = source.setdefault("outputs", [])
        if not outputs: outputs.append(dict(name=dtype, type=dtype, links=[], slot_index=0))
        outputs[0]["links"] = [next_link]
        links.append([next_link, source_id, 0, next_node, 0, dtype]); nodes.append(added)
        changes.append(f"{source_id} -> {next_node}: {kind}")

    for node in list(nodes):
        kind, values = node.get("type"), node.get("widgets_values", [])
        if kind == "PowerShardConfig" and len(values) != len(CONFIG_FIELDS):
            old = _old_ui_config(values); main, tuning = config_values(old)
            if "release_after_sampling" in old:retain_at_loader(node["id"],not old["release_after_sampling"])
            # Linked widgets need their slot names remapped rather than dropped.
            for slot in node.get("inputs", []):
                if slot.get("name") not in CONFIG_FIELDS:
                    raise ValueError("Legacy Config has linked advanced widgets; migrate the named API export")
            node["widgets_values"] = list(main.values())
            if tuning != TUNING_DEFAULTS: insert(node, "PowerShardConfigTuning", "config", "POWERSHARD_CONFIG", tuning)
            changes.append(f"{node['id']}: Config schema {SCHEMA}")
        elif kind=="PowerShardConfigTuning" and len(values)==7:
            if any(s.get("name")=="keep_workers" for s in node.get("inputs",[])):
                raise ValueError("Linked keep_workers requires named API/manual migration")
            retain_at_loader(node["id"],values[3])
            node["widgets_values"]=values[:3]+values[4:]
            changes.append(f"{node['id']}: keep_workers moved to H3 Loader")
        elif kind == "PowerShardH3FP16Patcher" and len(values) > 2:
            if len(values) not in (4, 5): raise ValueError("Unknown legacy H3 FP16 widgets layout")
            if any(s.get("name") not in ("model", "fp16_safe", "debug_finite") for s in node.get("inputs", [])):
                raise ValueError("Legacy linked patcher controls require a named API export")
            safe = bool(values[0] and values[1])
            node["widgets_values"] = [safe, values[2]]
            mode = (values[4] if len(values) == 5 else "manual") if safe else "off"
            insert(node, "PowerShardH3MLP", "model", "MODEL", dict(mode=mode, chunk_tokens=values[3]))
        elif kind == "PowerShardH3QwenLoader" and len(values) == 6:
            if any(s.get("name") in ("mlp_chunk_mode", "mlp_chunk_tokens") for s in node.get("inputs", [])):
                raise ValueError("Legacy linked Qwen MLP controls require a named API export")
            node["widgets_values"] = values[:4]
            insert(node, "PowerShardQwenMLP", "clip", "CLIP", dict(mode=values[4], chunk_tokens=values[5]))
        if kind=="PowerShardConfigTuning":
            values=node.get("widgets_values",[])
            if len(values)==6:
                node["widgets_values"]=values+["fp32","auto"]
                changes.append(f"{node['id']}: FP32 baseline wire and prefetch policy added")
            elif len(values)!=8:
                raise ValueError("Unknown PowerShardConfigTuning widgets layout; use named API JSON")
        if kind and kind.startswith("PowerShard"):
            node.setdefault("properties", {})["powershard_schema"] = SCHEMA
    graph["last_node_id"], graph["last_link_id"] = next_node, next_link
    return graph, changes
