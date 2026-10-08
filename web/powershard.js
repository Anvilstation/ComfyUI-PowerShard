import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "PowerShard.Devices",
    async beforeConfigureGraph(graphData) {
        const legacy = graphData.nodes?.some(n =>
            (n.type === "PowerShardConfig" && n.widgets_values?.length !== 5) ||
            (n.type === "PowerShardH3FP16Patcher" && n.widgets_values?.length > 2) ||
            (n.type === "PowerShardConfigTuning" && [6, 7].includes(n.widgets_values?.length)) ||
            (n.type === "PowerShardH3QwenLoader" && n.widgets_values?.length === 6));
        if (!legacy) return;
        const response = await api.fetchApi("/powershard/migrate-workflow", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(graphData)
        });
        const result = await response.json();
        if (!response.ok || result.error) {
            alert(`PowerShard: workflow migration failed: ${result.error ?? response.status}. Export named API JSON and use scripts/migrate_workflow.py.`);
            throw new Error(result.error ?? "PowerShard migration failed");
        }
        Object.assign(graphData, result.graph);
        console.info("PowerShard workflow migrated", result.changes);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PowerShardConfig") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            this.addWidget("button", "GPU / attention / память", null, async () => {
                const widget = this.widgets.find(w => w.name === "gpu_ids");
                const dialog = document.createElement("dialog");
                dialog.style.cssText = "max-width:750px;background:#222;color:#eee;padding:20px;border:1px solid #888;";
                const title = document.createElement("h3");
                title.textContent = "PowerShard: CUDA GPU, видимые ComfyUI";
                dialog.append(title);
                const body = document.createElement("div"); dialog.append(body);
                const close = document.createElement("button"); close.textContent = "Закрыть";
                close.onclick = () => dialog.close(); dialog.append(close);
                document.body.append(dialog); dialog.onclose = () => dialog.remove(); dialog.showModal();
                try {
                    const response = await api.fetchApi("/powershard/devices");
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const data = await response.json();
                    if (data.error) throw new Error(data.error);
                    const devices = data.devices;
                    const previous = String(widget.value).split(",").map(x => x.trim());
                    let selected = previous[0] === "all" ? devices.map(d => d.user_id)
                        : previous.map(x => devices.find(d => d.user_id === x || d.uuid === x)?.user_id ?? x);
                    const count = document.createElement("p"); body.append(count);
                    const update = () => count.textContent = `Выбрано: ${selected.length}; порядок ranks: ${selected.join(", ")}`;
                    const boxes = [];
                    for (const d of devices) {
                        const label = document.createElement("label"); label.style.display = "block";
                        const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = selected.includes(d.user_id);
                        cb.onchange = () => { selected = selected.filter(x => x !== d.user_id); if (cb.checked) selected.push(d.user_id); update(); };
                        label.append(cb, document.createTextNode(`${d.user_id}: ${d.name}, ${(d.total_memory / 2**30).toFixed(1)} GiB — ${d.uuid}`));
                        body.append(label); boxes.push(cb);
                    }
                    const all = document.createElement("button"); all.textContent = "Все видимые (all)";
                    all.onclick = () => { widget.value = "all"; widget.callback?.("all"); app.graph.setDirtyCanvas(true); dialog.close(); };
                    const apply = document.createElement("button"); apply.textContent = "Применить выбранные";
                    apply.onclick = () => {
                        if (!selected.length) { count.textContent = "Выберите хотя бы одну GPU"; return; }
                        widget.value = selected.join(","); widget.callback?.(widget.value); app.graph.setDirtyCanvas(true); dialog.close();
                    };
                    body.append(all, apply); update();
                    const status = document.createElement("pre"); status.style.whiteSpace = "pre-wrap";
                    status.textContent = data.providers.map(p => `${p.name}: ${p.status}\n${p.path ?? ""}\n${p.reason}${p.legacy_shim ? "\nWARNING: старый глобальный shim" : ""}`).join("\n\n");
                    body.append(status);
                    const execution = document.createElement("pre");
                    execution.style.cssText = "white-space:pre-wrap;border-top:1px solid #888;padding-top:10px;";
                    const refresh = document.createElement("button"); refresh.textContent = "Обновить фактические вызовы и память";
                    const showStatus = async () => {
                        try {
                            const response = await api.fetchApi("/powershard/status");
                            if (!response.ok) throw new Error(`HTTP ${response.status}`);
                            const state = await response.json();
                            const gib = n => ((n ?? 0) / 2**30).toFixed(2);
                            execution.textContent = `${state.note}\n\n` + state.sessions.map(s => {
                                const lines = [`${s.role}: ${s.draining ? "отмена: текущий RPC завершается в фоне" : s.idle_on_cpu ? "CPU shards сохранены, GPU-фаза освобождена" : s.running ? "workers активны" : "workers закрыты"}; GPUs ${s.gpu_ids.join(",")}; активный режим ${s.placement}; idle-веса ${s.idle_placement ?? "нет"}`, `Requested: ${s.requested}; selected policy: ${s.selected ?? "ещё не проверено"}`];
                                for (const r of s.ranks) {
                                    lines.push(`rank ${r.rank}: CUDA tensors ${gib(r.memory?.allocated)} GiB; неактивный cache ${gib(r.memory?.inactive_cache_bytes)} GiB; CPU shards ${gib(r.weights?.cpu_shard_bytes)} GiB`, `  actual calls: ${JSON.stringify(r.attention ?? {})}; MLP chunk ${r.mlp_chunks.join(",")}`, `  implementation: ${r.provider?.implementation?.module ?? r.provider?.entrypoint_module ?? r.provider?.actual_module ?? r.provider?.module ?? "—"}; origin: ${r.provider?.path ?? "—"}`);
                                    if (r.prefetch?.requested_prefetch !== undefined) lines.push(`  prefetch ${r.prefetch.requested_prefetch} → ${r.prefetch.effective_prefetch}: ${r.prefetch.prefetch_reason}`);
                                    if (r.wall_phases?.finish_cuda_wait_s !== undefined) lines.push(`  ожидание CUDA в конце ${r.wall_phases.finish_cuda_wait_s.toFixed(3)} s; finite check ${r.wall_phases.finite_validation_s.toFixed(3)} s`);
                                }
                                for (const p of s.progress ?? []) lines.push(`progress rank ${p.rank}: RPC ${p.sequence} ${p.command} ${p.status}; MLP chunk ${p.mlp_plan?.effective_tokens ?? "—"}`);
                                return lines.join("\n");
                            }).join("\n\n");
                        } catch (error) { execution.textContent = `Статус: ${error.message}`; }
                    };
                    refresh.onclick = showStatus;
                    body.append(refresh, execution);
                    await showStatus();
                } catch (error) { body.textContent = `Диагностика: ${error.message}. gpu_ids можно указать вручную.`; }
            }, { serialize: false });
        };
    }
});
