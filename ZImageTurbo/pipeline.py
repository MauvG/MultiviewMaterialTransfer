import gc
import torch
from diffusers import ZImagePipeline

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.bfloat16 if _device == "cuda" else torch.float32
_pipe = None

promptSuffix = ", full object visible, centered in frame, entire object shown from top to bottom and edge to edge, 3/4 natural perspective, transparent background, isolated, no other objects, no cropping, no cutoff, soft even studio lighting, sharp focus, product catalog photo"


def _cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_pipe():
    global _pipe
    if _pipe is not None:
        try:
            _pipe.to("cpu")
        except Exception:
            pass
        del _pipe
        _pipe = None
    gc.collect()
    _cuda_cleanup()


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=_dtype,
            low_cpu_mem_usage=True,
        )

        if _device == "cuda":
            try:
                _pipe.enable_model_cpu_offload()
            except Exception:
                _pipe.to("cuda")

            try:
                _pipe.vae.enable_tiling()
            except Exception:
                pass
        else:
            _pipe.to(_device)

        # _pipe.enable_attention_slicing()

    return _pipe


def generate(prompt: str, height: int, width: int, steps: int, seed: int):
    pipe = get_pipe()
    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    with torch.inference_mode():
        img = pipe(
            prompt=prompt + promptSuffix,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=g,
        ).images[0]

    _cuda_cleanup()
    return img