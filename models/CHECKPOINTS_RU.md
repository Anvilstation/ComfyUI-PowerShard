# Зафиксированные H3 Pruned checkpoints

Источник целевых четырёх файлов: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/a98869194787969724c7425d95d0ed73ce9202af/diffusion_models), revision `a98869194787969724c7425d95d0ed73ce9202af`.

| Файл | SHA256 из HF LFS metadata |
|---|---|
| minimax_h3_fl2va_pruned_bf16.safetensors | `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4` |
| minimax_h3_fl2va_pruned_int8_convrot.safetensors | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| minimax_h3_ref2va_pruned_bf16.safetensors | `37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe` |
| minimax_h3_ref2va_pruned_int8_convrot.safetensors | `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779` |

Заголовки реально прочитаны через Range; всё содержимое весов не скачивалось. SHA256 получены от источника, не пересчитаны локально для отсутствующих 20/40 GB файлов. Downloader пересчитывает hash скачанного файла.

Оба варианта: 50 DiT blocks, 2 text refiner blocks, hidden 5376, heads 56×128, ffn 14336, text_dim 5120, video latent 24 channels с patch `(1,2,2)`, audio latent 32 channels×2 stereo, precomputed adaLN curve 1025×8. Значения вывели из tensor shapes, затем сравнили с полной meta-моделью ComfyUI.

- BF16 storage включает FP32 islands. Worker загружает BF16 rows в FP16, сохраняет FP32 input/output projections, adaLN curves/projections и buffers. RMSNorm и softmax accumulation выполняются FP32. Это отдельный непроверенный на реальных весах FP16 compute путь.
- INT8 storage: row-wise I8 weights/scales F32 `[out,1]`, U8 `comfy_quant` JSON. Подтверждён пример `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`. Runtime проверяет каждую quantized layer. Integer weights не превращаются целиком в FP16 и не теряют scales/rotation metadata.
- Производные файлы не создавались. Runtime-конверсия не изменяет оригинал; dtype каждого shard сохраняется в worker evidence.

## Сопутствующие компоненты

В том же revision:

- `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`: 51 506 295 256 B. Qwen3-VL-32B обрезан до 50 language layers, выдаёт layer-50 hidden states без final norm; merged vision tower формирует vision tokens. Отдельный произвольный CLIP/vision encoder не подставляется.
- `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`: 27 141 342 152 B. Native offload only, kernels на Volta/POWER9 NOT_RUN; собственный generator INT8 path не переносится автоматически на encoder.
- `vae/minimax_h3_video_vae_fp16.safetensors`: 5 207 808 496 B.
- `vae/minimax_h3_audio_vae_fp32.safetensors`: 605 254 808 B.

Tokenizer/config ресурсы используются из вашей ComfyUI (Qwen3-VL + MiniMax special token mapping), без version gate. PowerShard не делает скрытых загрузок моделей/tokenizers. Полный offline encoder проверяется отдельно на сервере. SHA/размеры сопутствующих файлов сохранены в Comfy-Org--MiniMax-H3.json.

Conditioning: native H3 node возвращает hidden states плюс `minimax_token_tags`, keyframes/refs и cond audio/video latents. Latent: joint Native NestedTensor video `[1,24,T,H/16,W/16]`, audio `[1,32,2,T_audio]`. Native `FLOW_AV` сохраняет video/audio sigma mapping и scale. Референсы могут содержать image/video/audio, родной PackedLayout сохраняется семантически, а его cache object заново создаётся в worker.

## Kijai experimental

Проверен revision `f4cac997f880e93cf6940af61ee8d58ef31ff7f3`: присутствуют W4A8 mixed FL2VA/Ref2VA, VSA fastvideo, INT8 video VAE, LoRA/controlnet варианты. Четыре целевых checkpoint из таблицы выше найдены в Comfy-Org; нельзя автоматически считать W4A8 совместимым с нашим INT8 ConvRot loader. Эти экспериментальные форматы отклоняются.

Лицензия: MiniMax H3 Community License Agreement; ссылка и original model card сохранены. Наличие файла на HF не отменяет условий исходной лицензии.
