# Changelog / История изменений

## 0.6.0

### Русский

- Распределённая генерация изображений: SDXL, Flux, Z-Image, Qwen-Image и MOSS-Image. Режим распределения зависит от семейства модели.
- Veda Sparse Attention для MiniMax H3, LoRA, PDD-чекпойнты и Context Windows для H3.
- Общие ноды для LoRA, формата весов и текстовых энкодеров; обновлены отмена генерации и освобождение VRAM.
- В `example_workflows/` — готовые графы для Wan, LTX, MiniMax H3 и image-моделей. Обновлены руководства на русском и английском.

Скачайте `ComfyUI-PowerShard-0.6.0.zip` и распакуйте папку `ComfyUI-PowerShard` в `ComfyUI/custom_nodes`, затем перезапустите ComfyUI. Веса моделей скачиваются отдельно. Пайплайн проверяет CPU-тесты, включая Gloo/Ulysses, и содержимое ZIP. Новые пути для image-моделей и H3 не проверены на CUDA в рамках этого релиза.

### English

- Distributed image generation for SDXL, Flux, Z-Image, Qwen-Image and MOSS-Image. The parallel mode depends on the model family.
- Veda Sparse Attention for MiniMax H3, plus H3 LoRA, PDD checkpoints and Context Windows.
- Shared nodes for LoRA, weight formats and text encoders, with updated cancellation and VRAM cleanup.
- New examples for Wan, LTX, MiniMax H3 and image models in `example_workflows/`, with updated English and Russian guides.

Download `ComfyUI-PowerShard-0.6.0.zip`, extract the `ComfyUI-PowerShard` folder into `ComfyUI/custom_nodes`, and restart ComfyUI. Model weights are downloaded separately. The pipeline runs CPU tests, including Gloo/Ulysses, and verifies the ZIP. This release does not validate the new image and H3 paths on CUDA.
