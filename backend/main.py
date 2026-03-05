from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ZImageTurbo.pipeline import generate as z_generate
from MultiviewMaterialTransfer.pipeline import infer as mv_infer
import torch
from typing import Optional

app = FastAPI()

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

    img = z_generate(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        steps=req.steps,
        seed=req.seed,
    )

    out_path = job_dir / "generated.png"
    img.save(out_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    orbit_axis_x: Optional[float] = Form(None),
    orbit_axis_y: Optional[float] = Form(None),
    orbit_axis_z: Optional[float] = Form(None),
):
    job_id = uuid4().hex
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ref_path = job_dir / "reference.png"
    obj_path = job_dir / "object.png"

    ref_path.write_bytes(await reference.read())
    obj_path.write_bytes(await object.read())

    orbit_axis = None
    if orbit_axis_x is not None and orbit_axis_y is not None and orbit_axis_z is not None:
        orbit_axis = (orbit_axis_x, orbit_axis_y, orbit_axis_z)

    _, pred_frames, obj_frames = mv_infer(
        input_image_path=str(obj_path),
        reference_image_path=str(ref_path),
        elevation=elevation,
        distance=distance,
        fov=fov,
        num_inference_steps=steps,
        orbit_axis=orbit_axis,
    )

    n = max(1, min(max_frames, len(pred_frames), len(obj_frames)))

    for i, img in enumerate(obj_frames[:n]):
        img.save(job_dir / f"obj_{i:03d}.png")

    for i, img in enumerate(pred_frames[:n]):
        img.save(job_dir / f"pred_{i:03d}.png")

    return {
        "job_id": job_id,
        "reference_url": f"/outputs/{job_id}/reference.png",
        "object_url": f"/outputs/{job_id}/object.png",
        "obj_frames": [f"/outputs/{job_id}/obj_{i:03d}.png" for i in range(n)],
        "pred_frames": [f"/outputs/{job_id}/pred_{i:03d}.png" for i in range(n)],
    }
