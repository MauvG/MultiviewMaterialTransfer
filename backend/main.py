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
from typing import Dict
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

app = FastAPI()
GPU_LOCK = threading.Lock()

JOB_PROGRESS: Dict[str, dict] = {}
JOB_PROGRESS_LOCK = threading.Lock()



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


def set_job_progress(job_id: str, *, step: int, total_steps: int, status: str):
    with JOB_PROGRESS_LOCK:
        JOB_PROGRESS[job_id] = {
            "job_id": job_id,
            "status": status,
            "step": step,
            "total_steps": total_steps,
            "progress": 0.0 if total_steps <= 0 else max(0.0, min(1.0, step / total_steps)),
        }


def clear_job_progress(job_id: str):
    with JOB_PROGRESS_LOCK:
        JOB_PROGRESS.pop(job_id, None)


def run_multiview_job(
    *,
    job_id: str,
    obj_path: Path,
    ref_path: Path,
    elevation: float,
    distance: float,
    fov: float,
    steps: int,
    max_frames: int,
    camera_trajectory: str,
):
    set_job_progress(job_id, step=0, total_steps=max(1, steps - 1), status="running")

    def on_progress(step: int, total_steps: int):
        set_job_progress(job_id, step=step, total_steps=total_steps, status="running")

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
            progress_callback=on_progress,
        )

        get_unload_mv_pipe()()

    n = max(1, min(max_frames, len(pred_frames), len(obj_frames)))

    saved_obj = []
    saved_pred = []

    for i, img in enumerate(obj_frames[:n]):
        out = obj_path.parent / f"obj_{i:03d}.png"
        img.save(out)
        if out.exists() and out.stat().st_size > 0:
            saved_obj.append(f"/outputs/{job_id}/obj_{i:03d}.png")

    for i, img in enumerate(pred_frames[:n]):
        out = obj_path.parent / f"pred_{i:03d}.png"
        img.save(out)
        if out.exists() and out.stat().st_size > 0:
            saved_pred.append(f"/outputs/{job_id}/pred_{i:03d}.png")

    cleanup_cuda()
    set_job_progress(
        job_id,
        step=max(1, steps - 1),
        total_steps=max(1, steps - 1),
        status="completed",
    )

    return saved_obj, saved_pred

@app.get("/api/multiview-progress/{job_id}")
def get_multiview_progress(job_id: str):
    with JOB_PROGRESS_LOCK:
        data = JOB_PROGRESS.get(job_id)

    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return data

@app.post("/api/multiview-transfer")
async def multiview_transfer(
    job_id: str = Form(...),
    reference: UploadFile = File(...),
    object: UploadFile = File(...),
    elevation: float = Form(10),
    distance: float = Form(2.0),
    fov: float = Form(0.7),
    steps: int = Form(50),
    max_frames: int = Form(21),
    camera_trajectory: str = Form("orbit"),
):
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ref_path = job_dir / "reference.png"
    obj_path = job_dir / "object.png"

    ref_path.write_bytes(await reference.read())
    obj_path.write_bytes(await object.read())

    if camera_trajectory not in ALLOWED_CAMERA_TRAJECTORIES:
        camera_trajectory = "orbit"

    # create entry immediately so early polls don't 404
    set_job_progress(job_id, step=0, total_steps=max(1, steps - 1), status="running")

    try:
        saved_obj, saved_pred = await run_in_threadpool(
            run_multiview_job,
            job_id=job_id,
            obj_path=obj_path,
            ref_path=ref_path,
            elevation=elevation,
            distance=distance,
            fov=fov,
            steps=steps,
            max_frames=max_frames,
            camera_trajectory=camera_trajectory,
        )

        return {
            "job_id": job_id,
            "reference_url": f"/outputs/{job_id}/reference.png",
            "object_url": f"/outputs/{job_id}/object.png",
            "obj_frames": saved_obj,
            "pred_frames": saved_pred,
        }

    except Exception:
        set_job_progress(
            job_id,
            step=0,
            total_steps=max(1, steps - 1),
            status="failed",
        )
        raise