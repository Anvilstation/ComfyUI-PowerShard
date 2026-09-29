w4a8 is experimental format using 4bit weights and int8-convrot activations, at testing phase, further info here:

https://github.com/Comfy-Org/comfy-kitchen/pull/90

Requires ComfyUI 0.31.0

---

int8_convrot VAE needs ComfyUI 0.31.0 or you get black outputs

It speeds up VAE decode times by ~1.5x

---

ref lora is the difference between fl2va and ref2va, completely experimental, I don't even know if it has a use case at this point.

---

minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors

Currently for testing with these PRs:

https://github.com/Comfy-Org/ComfyUI/pull/15958

https://github.com/Comfy-Org/comfy-kitchen/pull/117