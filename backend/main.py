import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import threading
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
GPU_LOCK = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

ALLOWED_CAMERA_TRAJECTORIES = {
    "orbit",
    "spiral",
    "lemniscate",
    "roll",
    "spiral-top",
    "spiral-down",
    "spiral-zoom-in",
    "spiral-zoom-out",
    "vertical-orbit-360",
    "vertical-orbit-180",
    "orbit-sinusoidal",
    "strafe-right",
    "strafe-left",
    "close-zoom",
}

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def downscale_image_in_place(path: Path, max_side: int = 1024):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)

        w, h = img.size
        longest = max(w, h)
        if longest <= max_side:
            return

        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))

        resized = img.resize(new_size, Image.LANCZOS)

        save_kwargs = {}
        fmt = img.format or path.suffix.replace(".", "").upper()

        if fmt == "PNG" and resized.mode not in ("RGB", "RGBA", "L", "LA"):
            resized = resized.convert("RGBA")
        elif fmt in ("JPEG", "JPG") and resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")

        resized.save(path, format=fmt, **save_kwargs)


_z_generate = None
_unload_z_pipe = None
_mv_infer = None
_unload_mv_pipe = None


def get_z_generate():
    global _z_generate
    if _z_generate is None:
        from ZImageTurbo.pipeline import generate as _gen
        _z_generate = _gen
    return _z_generate


def get_unload_z_pipe():
    global _unload_z_pipe
    if _unload_z_pipe is None:
        from ZImageTurbo.pipeline import unload_pipe as _unload
        _unload_z_pipe = _unload
    return _unload_z_pipe


def get_mv_infer():
    global _mv_infer
    if _mv_infer is None:
        from MultiviewMaterialTransfer.pipeline import infer as _infer
        _mv_infer = _infer
    return _mv_infer


def get_unload_mv_pipe():
    global _unload_mv_pipe
    if _unload_mv_pipe is None:
        from MultiviewMaterialTransfer.pipeline import unload_pipeline as _unload
        _unload_mv_pipe = _unload
    return _unload_mv_pipe


class GenRequest(BaseModel):
    prompt: str
    height: int = 512
    width: int = 512
    steps: int = 9
    seed: int = 42


@app.post("/api/generate-image")
def generate_image(req: GenRequest):
    job_id = uuid4().hex
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    with GPU_LOCK, torch.inference_mode():
        get_unload_mv_pipe()()

        img = get_z_generate()(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            steps=req.steps,
            seed=req.seed,
        )

        get_unload_z_pipe()()

    out_path = job_dir / "generated.png"
    img.save(out_path)
    cleanup_cuda()

    return {
        "job_id": job_id,
        "image_url": f"/outputs/{job_id}/generated.png",
    }


@app.post("/api/multiview-transfer")
async def multiview_transfer(
    reference: UploadFile = File(...),
    object: UploadFile = File(...),
    elevation: float = Form(10),
    distance: float = Form(2.0),
    fov: float = Form(0.7),
    steps: int = Form(50),
    max_frames: int = Form(21),
    camera_trajectory: str = Form("orbit"),
):
    job_id = uuid4().hex
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ref_path = job_dir / "reference.png"
    obj_path = job_dir / "object.png"

    ref_path.write_bytes(await reference.read())
    obj_path.write_bytes(await object.read())

    # downscale_image_in_place(ref_path, max_side=1024)
    # downscale_image_in_place(obj_path, max_side=1024)

    if camera_trajectory not in ALLOWED_CAMERA_TRAJECTORIES:
        camera_trajectory = "orbit"

    with GPU_LOCK, torch.inference_mode():
        get_unload_z_pipe()()

        _, pred_frames, obj_frames = get_mv_infer()(
            input_image_path=str(obj_path),
            reference_image_path=str(ref_path),
            elevation=elevation,
            distance=distance,
            fov=fov,
            num_inference_steps=steps,
            camera_trajectory=camera_trajectory,
        )

        get_unload_mv_pipe()()

    n = max(1, min(max_frames, len(pred_frames), len(obj_frames)))

    for i, img in enumerate(obj_frames[:n]):
        img.save(job_dir / f"obj_{i:03d}.png")

    for i, img in enumerate(pred_frames[:n]):
        img.save(job_dir / f"pred_{i:03d}.png")

    cleanup_cuda()

    return {
        "job_id": job_id,
        "reference_url": f"/outputs/{job_id}/reference.png",
        "object_url": f"/outputs/{job_id}/object.png",
        "obj_frames": [f"/outputs/{job_id}/obj_{i:03d}.png" for i in range(n)],
        "pred_frames": [f"/outputs/{job_id}/pred_{i:03d}.png" for i in range(n)],
    }