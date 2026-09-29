import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "PowerShard.Devices",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PowerShardConfig") return;
        // C archive inserted placement before old widgets. Restore values by
        // the distinct placement enum; legacy B arrays keep their exact order.
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const values = info.widgets_values;
            if (Array.isArray(values) && ["gpu", "cpu", "ats"].includes(values[11])) {
                info = { ...info, widgets_values: [...values] };
                const placement = info.widgets_values.splice(11, 1)[0];
                info.widgets_values.splice(14, 0, placement);
                for (let i = 0; i < info.widgets_values.length; i++) {
                    if (this.widgets[i]?.type !== "button" && this.widgets[i])
                        this.widgets[i].value = info.widgets_values[i];
                }
            }
            return configure?.call(this, info);
        };
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            this.addWidget("button", "Выбрать GPU / статус attention", null, async () => {
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
                } catch (error) { body.textContent = `Диагностика: ${error.message}. gpu_ids можно указать вручную.`; }
            }, { serialize: false });
        };
    }
});

let helpSchema;
async function getHelpSchema() {
    if (!helpSchema) {
        helpSchema = api.fetchApi("/powershard/ui_schema").then(async response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        }).catch(error => { helpSchema = null; throw error; });
    }
    return helpSchema;
}

// Только дополнительные подписи/панель. widget.name, enum tokens и порядок
// сериализации не меняются. Кнопка не добавляет значения в widgets_values.
app.registerExtension({
    name: "PowerShard.ReadableSettings",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData.name.startsWith("PowerShard")) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const node = this;
            getHelpSchema().then(schema => {
                for (const widget of node.widgets ?? []) {
                    const help = schema.fields[widget.name];
                    if (help) { widget.label = help.label; widget.tooltip = help.help; }
                }
                app.graph?.setDirtyCanvas(true);
            }).catch(() => {}); // Native server tooltips остаются доступны.
            node.addWidget("button", "Параметры и пояснения", null, async () => {
                const dialog = document.createElement("dialog");
                dialog.style.cssText = "width:min(820px,90vw);max-height:85vh;overflow:auto;background:#222;color:#eee;padding:24px;border:1px solid #777;border-radius:8px;";
                const heading = document.createElement("h2"); heading.textContent = nodeData.display_name ?? nodeData.name;
                const body = document.createElement("div"), errorBox = document.createElement("p");
                errorBox.setAttribute("role", "alert"); errorBox.style.color = "#ffbd91";
                const close = document.createElement("button"); close.textContent = "Закрыть без изменений";
                close.onclick = () => dialog.close();
                dialog.append(heading, body, errorBox, close);
                document.body.append(dialog); dialog.onclose = () => dialog.remove(); dialog.showModal();
                try {
                    const schema = await getHelpSchema();
                    const about = document.createElement("p");
                    about.textContent = schema.nodes[nodeData.name]?.description ?? "";
                    body.append(about);
                    const notice = document.createElement("p");
                    notice.style.cssText = "padding:12px;background:#303e49;border-radius:4px;";
                    notice.textContent = "Минимум VRAM: CPU-шарды, prefetch 0, MLP auto, conditioning/Spectrum в RAM. Числа MiB задают бюджет временной работы, не лимит всей GPU. ATS — отдельная диагностика, не автоматическая защита от OOM.";
                    if (nodeData.name === "PowerShardConfig") body.append(notice);
                    const changes = [];
                    for (const group of ["Основное", "Память", "Дополнительно", "Совместимость"]) {
                        const section = document.createElement("details");
                        section.open = ["Основное", "Память"].includes(group);
                        const summary = document.createElement("summary"); summary.textContent = group;
                        summary.style.cssText = "font-size:1.1em;font-weight:bold;margin:18px 0 12px;cursor:pointer;";
                        section.append(summary);
                        let count = 0;
                        for (const widget of node.widgets ?? []) {
                            const help = schema.fields[widget.name];
                            if (!help || help.group !== group || widget.type === "button") continue;
                            const spec = nodeData.input?.required?.[widget.name] ?? nodeData.input?.optional?.[widget.name];
                            if (!spec || !["STRING", "BOOLEAN", "INT", "FLOAT"].includes(spec[0]) && !Array.isArray(spec[0])) continue;
                            count++;
                            const row = document.createElement("div"), label = document.createElement("label");
                            row.style.cssText = "border-top:1px solid #444;padding:12px 0;";
                            label.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:16px;font-weight:bold;";
                            const text = document.createElement("span"); text.textContent = help.label;
                            let input;
                            if (Array.isArray(spec[0])) {
                                input = document.createElement("select");
                                for (const value of spec[0]) {
                                    const option = document.createElement("option");
                                    option.value = String(value); option.textContent = schema.enums[widget.name]?.[String(value)] ?? String(value);
                                    input.append(option);
                                }
                                input.value = String(widget.value);
                            } else {
                                input = document.createElement("input");
                                input.type = spec[0] === "BOOLEAN" ? "checkbox" : ["INT", "FLOAT"].includes(spec[0]) ? "number" : "text";
                                if (input.type === "checkbox") input.checked = Boolean(widget.value);
                                else input.value = String(widget.value ?? spec[1]?.default ?? "");
                                if (input.type === "number") {
                                    input.step = spec[0] === "INT" ? "1" : "any";
                                    if (spec[1]?.min !== undefined) input.min = spec[1].min;
                                    if (spec[1]?.max !== undefined) input.max = spec[1].max;
                                }
                            }
                            input.style.cssText = "max-width:55%;padding:6px;";
                            label.append(text, input);
                            const detail = document.createElement("p"); detail.textContent = help.help;
                            detail.style.cssText = "color:#ccc;font-size:0.9em;line-height:1.45;margin:8px 0 0;";
                            row.append(label, detail); section.append(row);
                            changes.push({ widget, input, spec, help });
                        }
                        if (count) body.append(section);
                    }
                    const apply = document.createElement("button"); apply.textContent = "Применить параметры";
                    apply.style.cssText = "margin:16px 12px 0 0;padding:8px 14px;";
                    apply.onclick = () => {
                        const values = [];
                        for (const { widget, input, spec, help } of changes) {
                            if (!input.checkValidity()) { errorBox.textContent = `Проверьте параметр «${help.label}».`; input.reportValidity(); return; }
                            let value = input.type === "checkbox" ? input.checked : input.value;
                            if (Array.isArray(spec[0])) value = spec[0].find(x => String(x) === input.value);
                            if (["INT", "FLOAT"].includes(spec[0])) {
                                value = Number(input.value);
                                if (input.value === "" || !Number.isFinite(value) || spec[0] === "INT" && !Number.isInteger(value)) {
                                    errorBox.textContent = `Некорректное число: ${help.label}.`; return;
                                }
                            }
                            values.push([widget, value]);
                        }
                        for (const [widget, value] of values) {
                            if (widget.value !== value) { widget.value = value; widget.callback?.(value); }
                        }
                        app.graph?.setDirtyCanvas(true); app.graph?.change?.(); dialog.close();
                    };
                    body.append(apply);
                } catch (error) { errorBox.textContent = `Не удалось открыть пояснения: ${error.message}. Исходные widgets и подсказки ноды доступны.`; }
            }, { serialize: false });
        };
    }
});
