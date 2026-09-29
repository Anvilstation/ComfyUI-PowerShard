// Контракт frontend в минимальном DOM: не визуальная проверка ComfyUI browser.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const [,,source,metadata] = process.argv;
const {schema,input}=JSON.parse(fs.readFileSync(metadata,'utf8'));
class Element {
    constructor(tag) { this.tag=tag;this.children=[];this.style={};this.value='';this.textContent=''; }
    append(...nodes) { this.children.push(...nodes); }
    setAttribute() {}
    showModal() { this.open=true; }
    close() { this.open=false;this.onclose?.(); }
    remove() { this.removed=true; }
    checkValidity() { return true; }
    reportValidity() {}
}
const body=new Element('body'), document={body,createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text})};
const extensions=[];
const app={registerExtension:e=>extensions.push(e),graph:{setDirtyCanvas(){},change(){}}};
const api={fetchApi:async path=>({ok:true,json:async()=>{assert.equal(path,'/powershard/ui_schema');return schema;}})};
const code=fs.readFileSync(source,'utf8').replace(/^import .*;\r?\n/gm,'');
vm.runInNewContext(code,{app,api,document,console});
class Node {
    constructor() {
        this.widgets=Object.entries({...input.required,...input.optional}).map(([name,spec])=>({name,type:'widget',value:spec[1]?.default??(Array.isArray(spec[0])?spec[0][0]:undefined),callback(value){this.last=value;}}));
    }
    addWidget(type,name,value,callback,options) { this.widgets.push({type,name,value,callback,options}); }
    onConfigure(info) { this.configured=info; }
}
const visit=root=>[root,...(root.children??[]).flatMap(visit)];
(async()=>{
    for(const extension of extensions)await extension.beforeRegisterNodeDef(Node,{name:'PowerShardConfig',input});
    const node=new Node(), ids=node.widgets.map(w=>w.name);
    node.onNodeCreated();await new Promise(resolve=>setImmediate(resolve));
    assert.deepEqual(node.widgets.filter(w=>w.type!=='button').map(w=>w.name),ids);
    assert.equal(node.widgets.find(w=>w.name==='memory_profile').label,'Профиль памяти');
    assert(node.widgets.filter(w=>w.type==='button').every(w=>w.options.serialize===false));
    await node.widgets.find(w=>w.name==='Параметры и пояснения').callback();
    const dialog=body.children.at(-1), elements=visit(dialog);
    assert(elements.some(x=>x.tag==='summary' && x.textContent==='Память'));
    const label=elements.find(x=>x.tag==='label' && x.children[0].textContent==='Профиль памяти');
    const selector=label.children[1];selector.value='ram_min';
    assert(selector.children.some(x=>x.value==='ram_min' && x.textContent.includes('Минимум VRAM')));
    elements.find(x=>x.textContent==='Применить параметры').onclick();
    assert.equal(node.widgets.find(w=>w.name==='memory_profile').value,'ram_min');
    assert.equal(node.widgets.find(w=>w.name==='memory_profile').last,'ram_min');
    assert(dialog.removed);
    const legacy=node.widgets.slice(0,17).map(w=>w.value);
    node.onConfigure({widgets_values:legacy});assert.deepEqual(node.configured.widgets_values,legacy);
    const c=[...legacy];const placement=c.splice(14,1)[0];c.splice(11,0,placement);
    node.onConfigure({widgets_values:c});assert.equal(JSON.stringify(node.configured.widgets_values),JSON.stringify(legacy));
    console.log('PASS: labels, grouped panel, apply, no API/widget-order change, legacy migration');
})().catch(error=>{console.error(error);process.exitCode=1;});
