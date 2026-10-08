# Совместимость 0.5.3

Целевой стек — существующие PyTorch 2.12, CUDA 12.4, custom FA2 на POWER9/ppc64le с V100. Нода не навязывает новый wheel или CUDA/NCCL. Фактическая совместимость должна подтверждаться server probes из README. Импорт Python package не является проверкой sm_70 kernel.

Native API тесты выполнены с ComfyUI f1072eb0350638a3390ddb6afbcaa8c6b237c6fd и torch 2.12.0+cpu на x86_64. Это не POWER9/CUDA certification. Source SHA хранится для воспроизводимости; admission проверяет signatures/capabilities. Метаданные моделей и предыдущих source audits в models/ и sources.lock.json относятся к зафиксированным исходным материалам.

ATS работает только при реальной поддержке hardware host page tables/managed/concurrent/pageable capabilities. Обычная x86 UVA не проходит ATS gate. Legacy UI/API workflows требуют migration при изменении positional widgets; fsdp2, timeout_s и allow_unverified сохраняются только как legacy parsing на внутреннем Python API.

0.5.1 сохраняет five-widget Config и две separate MLP nodes. `sequence_comm_dtype` по умолчанию fp16 с Safe normalization; явно переданный fp32 сохраняется. CPU/CUDA cache метрики раздельные. Ваши логи используют custom torch 2.12.0a1/CUDA12.4, а validation runtime — released CPU torch 2.12.0: новая CUDA численная/performance совместимость требует server probes.

0.5.2 дополняет native Qwen proxy служебными load/unpatch полями, включая clones и MLP variants. MLP defaults off; H3 Loader keep=true. UI schema 6 переносит legacy tuning keep_workers в loader, не сдвигая strict/pin controls. Явно сохранённые MLP auto/manual в старых workflows не переписываются.

Flash API shim теперь scoped: при ImportError установленного flash_attn используется локальная обёртка над существующим vllm_flash_attn, с numerical probe на каждом rank и identity реального implementation. Глобальные sys.path/sys.modules не подменяются. Это не установка независимого Dao-AILab CUDA kernel.

0.5.3 возвращает default sequence_comm_dtype=fp32; явные FP16 настройки сохраняются, в Advanced появилась возможность A/B. UI schema 7 добавляет два optional widgets в конец tuning; six-widget schema 6 и seven-widget schema 5 (с legacy keep_workers) мигрируют без сдвига existing strict/pin controls. MLP mode default off не менялся. Keep GPU/ATS теперь означает idle RAM cache с восстановлением активного placement, а не сохранение VRAM. Native CPU cache tests не сертифицируют аппаратную FSDP/ATS compatibility.

Обновлённые instructions и результаты: [README_RU.md](README_RU.md), [PERFORMANCE_FIX_0_5_3_RU.md](PERFORMANCE_FIX_0_5_3_RU.md).
