#!/usr/bin/env bash
# Запуск существующего ComfyUI, без установки/изменения файлов core.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Использование: bash scripts/launch_comfy.sh /ABS/python3.11 /ABS/ComfyUI [дополнительные аргументы ComfyUI]' >&2
  exit 2
fi
ps_python=$1
ps_comfy=$2
shift 2
if [[ ! -f "$ps_comfy/main.py" ]]; then
  echo 'Не найден ComfyUI/main.py' >&2
  exit 2
fi
"$ps_python" -c 'import sys, torch; print("Python:",sys.version,"torch:",torch.__version__,"CUDA build:",torch.version.cuda)'
cd "$ps_comfy"
exec "$ps_python" "$ps_comfy/main.py" --listen 127.0.0.1 --disable-dynamic-vram --use-pytorch-cross-attention "$@"
