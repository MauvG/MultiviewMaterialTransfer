import torch
from MultiviewMaterialTransfer.models.seva import load_pipeline
from typing import Optional, Tuple

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline(
            device="cuda" if torch.cuda.is_available() else "cpu",
            activate_layers=True,
            resume_from="MultiviewMaterialTransfer/model.pt",
            swapping_config={"swapping_activation_type": "sigmoid_one_hot_layer_threshold_0.75"},
        )
    return _pipeline

def infer(
    input_image_path: str,
    reference_image_path: str,
    elevation: float,
    distance: float,
    fov: float,
    num_inference_steps: int,
    orbit_axis: Optional[Tuple[float, float, float]] = None,
):
    pipe = get_pipeline()
    autocast_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        samples = pipe.infer_demo(
            input_image_path=input_image_path,
            reference_image_path=reference_image_path,
            elevation=elevation,
            distance=distance,
            fov=fov,
            num_inference_steps=num_inference_steps,
            orbit_axis=orbit_axis,
        )

    frames = [f[1:] for f in samples]
    reference = frames[0]
    preds = frames[1]
    obj = frames[2]
    return reference, preds, obj
