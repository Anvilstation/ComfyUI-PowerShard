#!/usr/bin/env python3
"""Отдельная явная операция: один разрешённый файл, зафиксированный revision/SHA256."""
import argparse,hashlib,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument("filename");p.add_argument("--destination",required=True);p.add_argument("--download",action="store_true");a=p.parse_args()
m=json.loads((ROOT/"models/Comfy-Org--MiniMax-H3.json").read_text());entry=next((s for s in m["siblings"] if s["rfilename"]==a.filename),None)
if entry is None or not a.filename.endswith(".safetensors"):raise SystemExit("Файл отсутствует в зафиксированном manifest")
url=f'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/{m["sha"]}/{a.filename}'
print(json.dumps({"file":a.filename,"revision":m["sha"],"size":entry.get("size"),"sha256":entry.get("lfs",{}).get("sha256"),"destination":a.destination},indent=2))
if not a.download:raise SystemExit("Только план. Для скачивания одного файла добавьте --download")
dst=Path(a.destination).resolve();dst.parent.mkdir(parents=True,exist_ok=True)
if dst.exists():raise SystemExit("Файл уже существует, перезапись запрещена")
partial=dst.with_suffix(dst.suffix+".partial")
if partial.exists():raise SystemExit("Есть partial файл предыдущей загрузки; проверьте его отдельно")
h=hashlib.sha256()
try:
 with urllib.request.urlopen(url,timeout=120) as response, partial.open("xb") as output:
  while chunk:=response.read(8*1024*1024):output.write(chunk);h.update(chunk)
 expected=entry.get("lfs",{}).get("sha256")
 if not expected or h.hexdigest()!=expected:raise RuntimeError("SHA256 mismatch; partial сохранён для диагностики")
 partial.rename(dst)
except BaseException:raise
print("Сохранён проверенный checkpoint:",dst)
