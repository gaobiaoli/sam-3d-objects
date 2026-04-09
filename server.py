"""FastAPI service for SAM 3D Objects inference.

POST /infer accepts image+mask and optional depth/pointmap, then saves both:
- <request_id>.glb
- <request_id>.json (pose)

Return mode:
- return glb directly, and include json content in response header

Depth support:
- `depth_image` as depth image (`.png/.jpg/.jpeg/.tiff`)
- optional `K` (3x3 intrinsics JSON string)
"""

import json
import base64
import importlib.util
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import imageio.v3 as iio
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


def _load_local_inference_symbols() -> tuple[Any, Any, Any]:
    inference_py = Path(__file__).resolve().parent / "notebook" / "inference.py"
    if not inference_py.exists():
        raise RuntimeError(f"Inference module not found: {inference_py}")

    spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load inference module spec from: {inference_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Inference, module.load_image, module.load_mask


Inference, load_image, load_mask = _load_local_inference_symbols()

app = FastAPI(title="SAM 3D Objects Inference API")

inference_instance: Optional[Any] = None


def _output_dir() -> Path:
    output_dir = Path(os.environ.get("SAM3D_OUTPUT_DIR", "/root/sam-3d-objects/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _result_paths(request_id: str) -> tuple[Path, Path]:
    output_dir = _output_dir()
    return output_dir / f"{request_id}.glb", output_dir / f"{request_id}.json"


def _to_list_1d(value: Any, expected_len: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.squeeze(array).reshape(-1)
    if array.size < expected_len:
        raise ValueError(
            f"Invalid pose field length: expected {expected_len}, got {array.size}"
        )
    return array[:expected_len].astype(np.float32).tolist()


def _extract_pose_json(output: dict) -> dict:
    return {
        "object_0": {
            "scale": _to_list_1d(output["scale"], 3),
            "translation": _to_list_1d(output["translation"], 3),
            "rotation": _to_list_1d(output["rotation"], 4),
        }
    }


def _parse_k_matrix(k_text: Optional[str]) -> Optional[np.ndarray]:
    if not k_text:
        return None

    try:
        value = json.loads(k_text)
        matrix = np.asarray(value, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid K: {exc}")

    if matrix.shape != (3, 3):
        raise HTTPException(status_code=400, detail=f"K must be 3x3, got {matrix.shape}")
    return matrix


def _depth_to_pointmap(
    depth_bytes: bytes,
    k_matrix: np.ndarray,
    depth_scale: float,
    invalid_depth_value: Optional[float],
) -> torch.Tensor:
    if depth_scale <= 0:
        raise HTTPException(status_code=400, detail=f"depth_scale must be > 0, got {depth_scale}")

    depth = iio.imread(BytesIO(depth_bytes)).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    if invalid_depth_value is not None:
        depth[depth == float(invalid_depth_value)] = 0.0

    depth = depth / float(depth_scale)
    depth[depth <= 0] = np.nan

    h, w = depth.shape
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    pointmap = np.stack([-x, -y, z], axis=-1).astype(np.float32)
    return torch.from_numpy(pointmap)


def get_inference() -> Any:
    global inference_instance
    if inference_instance is None:
        config_path = os.environ.get(
            "SAM3D_PIPELINE_CONFIG",
            "/root/autodl-tmp/checkpoints/hf/pipeline.yaml",
        )
        if not os.path.exists(config_path):
            raise RuntimeError(f"Config not found: {config_path}")
        inference_instance = Inference(config_path, compile=False)
    return inference_instance


@app.on_event("startup")
async def _warmup_model() -> None:
    get_inference()


@app.post("/infer")
async def infer(
    image: UploadFile = File(..., description="RGB image"),
    mask: UploadFile = File(..., description="Binary/alpha mask"),
    depth_image: Optional[UploadFile] = File(None, description="Optional depth image file"),
    K: Optional[str] = Form(None, description="Optional 3x3 intrinsics JSON string"),
    depth_scale: float = Form(512.0),
    invalid_depth_value: Optional[float] = Form(65535.0),
    request_id: Optional[str] = Form(None),
    seed: int = Form(42),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, image.filename or "image.png")
        mask_path = os.path.join(tmpdir, mask.filename or "mask.png")

        try:
            with open(image_path, "wb") as file_obj:
                file_obj.write(await image.read())
            with open(mask_path, "wb") as file_obj:
                file_obj.write(await mask.read())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save uploads: {exc}")

        try:
            img_np = load_image(image_path)
            mask_np = load_mask(mask_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load image/mask: {exc}")

        pointmap_tensor = None
        if depth_image is not None:
            depth_bytes = await depth_image.read()
            suffix = Path(depth_image.filename or "").suffix.lower()
            valid_depth_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            if suffix not in valid_depth_suffixes:
                raise HTTPException(
                    status_code=400,
                    detail="depth_image must be image file: .png/.jpg/.jpeg/.tif/.tiff",
                )

            k_matrix = _parse_k_matrix(K)
            if k_matrix is None:
                raise HTTPException(status_code=400, detail="Depth image requires `K`")

            pointmap_tensor = _depth_to_pointmap(
                depth_bytes=depth_bytes,
                k_matrix=k_matrix,
                depth_scale=depth_scale,
                invalid_depth_value=invalid_depth_value,
            )

        try:
            output = get_inference()(img_np, mask_np, seed=seed, pointmap=pointmap_tensor)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

        mesh = output.get("glb")
        if mesh is None:
            raise HTTPException(status_code=422, detail="Inference returned no mesh")

        try:
            pose_result = _extract_pose_json(output)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to extract pose json: {exc}")

        stem = request_id or (Path(image.filename or "output").stem or "output")
        glb_path, json_path = _result_paths(stem)

        try:
            mesh.export(str(glb_path))
            json_path.write_text(json.dumps(pose_result, indent=2), encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save outputs: {exc}")

        pose_json_text = json.dumps(pose_result, separators=(",", ":"))
        pose_json_b64 = base64.b64encode(pose_json_text.encode("utf-8")).decode("ascii")
        return FileResponse(
            path=str(glb_path),
            media_type="model/gltf-binary",
            filename=f"{stem}.glb",
            headers={
                "X-Pose-Json-B64": pose_json_b64,
                "X-Json-File": str(json_path),
                "X-Request-Id": stem,
            },
        )


@app.get("/results/{request_id}/glb")
async def download_glb(request_id: str):
    glb_path, _ = _result_paths(request_id)
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail=f"GLB not found for request_id={request_id}")
    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename=f"{request_id}.glb",
    )


@app.get("/results/{request_id}/json")
async def download_json(request_id: str):
    _, json_path = _result_paths(request_id)
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"JSON not found for request_id={request_id}")
    return FileResponse(
        path=str(json_path),
        media_type="application/json",
        filename=f"{request_id}.json",
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=6006, reload=False)
