# Veda Sparse Attention в PowerShard — сторонний код

* **Veda-on-ComfyUI** (<https://github.com/veda-sparse/Veda-on-ComfyUI>, MIT, (c) 2026 veda-sparse, текст лицензии —
  `LICENSE_VEDA_MIT.txt`): `tiling.py`, `h3_layout.py`, `plans.py`, `predictor.py`, `selection.py` перенесены без
  изменения арифметики; `bundle.py`, `settings.py`, `reference.py`, `engine.py`, `kernels.py` — переработанные версии
  (подмножество голов Ulysses, softmax scale FP16 Safe, без UI ComfyUI).
* **SageAttention 1.0.6** (BSD-3-Clause, Thu-ML) — INT8-ядро `kernels.py::_attention_int8_kernel` и квантование
  взяты из `veda_comfy/kernels/sage/sparse_int8.py` (производная от `quant_per_block.py` / `attn_qk_int8_per_block.py`).
* **Miowtion** (<https://github.com/veda-sparse/Miowtion>, MIT) — правила тайлинга/плана/предиктора/отбора.
* **Веса предиктора Veda** (скачиваются пользователем, не входят в архив) — MiniMax H3 Community License.

FP16-ядро (`_attention_fp16_kernel`, для V100/sm70) и бэкенд `torch-sdpa` написаны для PowerShard.
