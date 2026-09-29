# Численная стратегия FP16 Safe

## Причина регрессии

Старый Linear делал x.to(weight.dtype), даже когда вызывающий H3 передал FP32. С FP16 weight вход 100000 становился Inf до GEMM. Возможен и другой случай: вход представим в half, но сумма Linear выходит за 65504. Native extra_conds также приводил context к inference dtype до отправки worker. Детектор обнаруживал проблему, но не исправлял вычисление.

Регрессия: shape [7,24], input=100000, W=1 → старый FloatingPointError. Новый condition path даёт FP32 [7,32], каждый элемент 2400000. Реальные веса пользователя отсутствуют: воспроизведён механизм, не конкретный sample его checkpoint.

## Dtypes

| Область | Storage / исполнение |
|---|---|
| Dense weights | FP16 local shards; родные FP32 islands сохранены |
| INT8 weights / scales | I8 / FP32 frozen Parameters; ConvRot JSON — конфигурация |
| condition_proj | FP32 вход, текущий weight/dequant, GEMM и выход |
| Packed stream / residual | FP32 |
| QKV, out_proj, MLP fc1/fc2 | FP16 scaled operands и GEMM output; немедленная FP32 компенсация |
| Attention QK и PV | FP16 scaled GEMM; полное dense attention |
| RMSNorm, RoPE, softmax, SiLU, gated product | FP32; исходный epsilon/нелинейность сохраняются |
| AdaLN, video/audio safety islands, final | Native FP32; worker AV inputs/outputs FP32 |
| FSDP communication | Dtype parameter representation; I8 не распаковывается до all-gather |
| Sequence communication Safe | FP32 tokens/K/V; больше памяти, чем старый FP16 поток |

FP16 torch.matmul даёт CUDA FP16 GEMM путь без Triton/Ampere. Факт использования Tensor Core instructions на V100 требует trace/Nsight; CPU тест этого не доказывает.

## Scaling без clamp

Для A[M,K] B[K,N] берём T=16384, с запасом относительно 65504. Выбираем степень двойки sB >= max(abs(B))/T и Bh=half(B/sB).
Для каждой строки A выбираем степень двойки sA не меньше max(1, maxabs(A)/T, maxabs(A) * max_j sum_k abs(Bh[k,j]) / T).

По неравенству треугольника scaled GEMM output ограничен T до ошибки округления. Выполняем float(half(A/sA) @ Bh) * sA * sB.
Bias добавляется ПОСЛЕ компенсации: linear(x/s)*s с ненулевым bias неверно.
Все max/sum/scale остаются на GPU, нет .item() для выбора масштаба.

В вещественной арифметике линейная операция сохраняется, но FP16 rounding/underflow остаются. Большой динамический диапазон внутри строки может терять малые компоненты. Нужны сравнения реальных activations/качества. Никакое scaling не проталкивается через SiLU/RMSNorm: они получают восстановленный истинный FP32 масштаб.

## Память и проверки

MLP обрабатывает до 512 tokens за раз; SiLU×up — FP32 chunk. Выход/residual остаётся полным FP32 stream. INT8 деквантует до 256 выходных строк, выполняет inverse ConvRot и GEMM без dense cache.

Dense Linear временно создаёт FP32 текущую матрицу для bound; это может быть больше одного FP16 weight. Учитывать в peak вместе с all-gather. INT8 workspace ограничен rows. Dynamic reductions/casts/Python loops стоят времени; speedup не обещан.

FiniteTracker принадлежит worker instance. Normal mode проверяет RPC output; debug добавляет GPU flags по операциям/блокам. Один CPU read на границе RPC. Ошибки не игнорируются и не заменяются nan_to_num. Loading/preflight проверки отдельно могут синхронизироваться.

## Lifecycle

Node клонирует ComfyUI wrapper/config и маленькое дерево proxy, не веса. H3PatchConfig сериализуется JSON и fingerprint.
Worker: meta structure → INT8 representation → instance attention/safe patch → FSDP → local shards.
Три ready replies должны подтвердить одинаковый fingerprint.

Patch идемпотентен. Новая policy создаёт новую session, patch поверх активной FSDP запрещён. Disabled node создаёт чистую unpatched session; удаление node возвращает исходный loader MODEL. Глобальные классы не меняются, cached loader не получает ghost patch.
