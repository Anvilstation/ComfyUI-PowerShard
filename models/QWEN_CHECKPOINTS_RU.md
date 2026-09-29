# H3 Qwen3-VL-32B: проверенные источники

Источник: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/a98869194787969724c7425d95d0ed73ce9202af/text_encoders), revision `a98869194787969724c7425d95d0ed73ce9202af`. Скачаны только ограниченные диапазоны заголовков, **не веса**. Оригинальные checkpoints не изменялись.

| Файл в text_encoders | Размер B | SHA256 из HF metadata |
|---|---:|---|
| qwen3vl_32b_minimax_h3_bf16.safetensors | 51506295256 | 600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d |
| qwen3vl_32b_minimax_h3_int8_convrot.safetensors | 27141342152 | bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923 |

Это H3 truncated encoder: 50 language decoder layers, hidden5120, intermediate25600, Q heads64, KV heads8, head_dim128, vocab151936; native vision tower/DeepStack. Last hidden unnormalized, без final_norm и lm_head. Полная native meta-модель совпала с 902 BF16 tensor names/shapes. Текстовый tokenizer — native MiniMaxH3Tokenizer из ComfyUI с Qwen tokenizer assets и дополнительными H3 special tokens, без chat-template подмены. Input modalities: text/images/video-frame-pairs, audio labels без waveform в Qwen.

Storage BF16 — источник FP16 runtime, а не уже готовый FP16 файл. При streaming local-row load floating параметры приводятся к целевому dtype с finite/range validation. Norm/residual/vision safety islands и linear outputs FP32. INT8 ConvRot использует I8 weights + scales + comfy_quant metadata; они не превращаются в постоянную dense модель. FSDP shard axis0 разделяет output rows, ConvRot группы по input axis остаются целыми. Неполный последний shard обрабатывается FSDP padding, metadata не делится как произвольные байты.

Лицензия модели: см. MODEL_CARD_Comfy-Org.md и MiniMax-H3-Community-LICENSE в licenses; права на checkpoint определяет источник, лицензия кода PowerShard их не заменяет. Native ComfyUI code используется как зависимость GPL. Загрузка конкретного веса — отдельное действие пользователя; никаких upstream repositories целиком и автоматической переустановки пакетов.

Контрольные SHA выше взяты из metadata, **полный downloaded blob здесь не хешировался**. Header-файлы рядом дают структуру, но не позволяют запустить генерацию. FSDP/Qwen full weights и качество — NOT_RUN.
