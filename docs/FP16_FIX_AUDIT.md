# Аудит FP16 fixes, 2026-09-15

Изучен код, не только README. SHA фиксируют наблюдение, но НЕ являются version gating ComfyUI.

| Проект | Commit / лицензия | Подход и решение PowerShard |
|---|---|---|
| Amduraznak/minimax-h3-fp16-fix | b09897c5c3bf4af7262bb43d04a71a94563c1590, MIT | FP32 condition/residual, FP16 attention/MLP, out_proj /64 и fc2 /256 с компенсацией. Использованы идеи safety islands/scaling, не global class patches |
| aaalll12322/ComfyUI-MiniMaxH3-FP16Safe | 4badf245c757932ad776c0dd799500428400fb17, MIT | Structure cloning, instance patches, MLP chunking, deferred finite flags, ConvRot. Использованы эти идеи; не перенесены global mode/flag и повтор всего forward в FP32 |
| Icbears/minimax-h3-v100-patch | d869f4fc4826ed4b1b1b32c862a2193da0cf8daa, GPL-3.0-only | Изучен вариант minimax-h3-v100-l3-clean (manifest 0.1.4), FP32 audio/final islands и lifecycle. Только техническое сравнение; исходники не копировались |

Источники: [Amduraznak](https://github.com/Amduraznak/minimax-h3-fp16-fix), [FP16Safe](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe), [V100 patch](https://github.com/Icbears/minimax-h3-v100-patch).

## Почему нельзя подключить fix как есть

1. condition input.float() недостаточно: старый PowerShard Linear обратно кастовал к dtype веса. Исправлены operations и host inference dtype.
2. ModelPatcher.clone() делит module tree. H3 находится не в main process, поэтому введены immutable worker policy и structurally cloned host proxy с новой session.
3. Постоянные /64 и /256 зависят от checkpoint диапазонов. Здесь динамическая power-of-two граница для GEMM input И output, без CPU sync. Это самостоятельная реализация, не обещание большей скорости.
4. Scaling Q/K до RMSNorm без учёта epsilon меняет нормализацию. Здесь FP32 масштаб восстановлен до norm/SiLU.
5. Bias добавляется после компенсации, а не умножается на scale.
6. Finite flags принадлежат instance, нет global flags/полной FP32 fallback H3.
7. INT8 представлен frozen Parameters, без tensor subclass. Dequant — только текущие rows; scales/metadata сохранены.

Общая лицензия существующего PowerShard GPL-3.0-or-later сохранена. MIT copyright notices добавлены в licenses/NOTICE. GPL-only код в проект не переносился.
