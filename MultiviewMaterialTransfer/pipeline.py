import gc
import torch
from MultiviewMaterialTransfer.models.seva import load_pipeline

_pipeline = None

def _cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_pipeline():
    global _pipeline
    if _pipeline is not None:
        try:
            if hasattr(_pipeline, "to"):
                _pipeline.to("cpu")
        except Exception:
            pass
        del _pipeline
        _pipeline = None
    gc.collect()
    _cuda_cleanup()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline(
            device="cuda" if torch.cuda.is_available() else "cpu",
            activate_layers=True,
            resume_from="MultiviewMaterialTransfer/model.pt",
            swapping_config={"swapping_activation_type": "sigmoid_one_hot_layer_threshold_0.75"},
        )

        for obj in (
            _pipeline,
            getattr(_pipeline, "module", None),
            getattr(_pipeline, "vae", None),
            getattr(getattr(_pipeline, "module", None), "vae", None),
        ):
            if obj is None:
                continue
            try:
                obj.enable_tiling()
            except Exception:
                try:
                    obj.enable_vae_tiling()
                except Exception:
                    pass

    return _pipeline


def infer(
    input_image_path: str,
    reference_image_path: str,
    elevation: float,
    distance: float,
    fov: float,
    num_inference_steps: int,
    camera_trajectory: str = "orbit",
):
    pipe = get_pipeline()

    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    with torch.inference_mode():
        with torch.autocast(
            device_type=autocast_device,
            dtype=autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            samples = pipe.infer_demo(
                input_image_path=input_image_path,
                reference_image_path=reference_image_path,
                elevation=elevation,
                distance=distance,
                fov=fov,
                num_inference_steps=num_inference_steps,
                camera_trajectory=camera_trajectory,
            )

    frames = [f[1:] for f in samples]
    reference = frames[0]
    preds = frames[1]
    obj = frames[2]

    _cuda_cleanup()
    return reference, preds, obj