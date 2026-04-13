"""FastAPI service for SAM 3D Objects inference.

POST /infer accepts image+optional mask and optional depth/pointmap, then saves both:
- <request_id>.glb
- <request_id>.json (pose)

Mask can be provided directly or generated from bbox using SAM model:
- `mask` as binary/alpha mask image (optional)
- `bbox` as JSON string [x_min, y_min, x_max, y_max] (optional, used if mask not provided)

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


def _load_sam_model():
    """Load SAM model for mask generation from bbox."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
        
        model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
        checkpoint_path = os.environ.get(
            "SAM_CHECKPOINT_PATH",
            "/root/sam-3d-objects/sam_vit_h_4b8939.pth"
        )
        
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"SAM checkpoint not found at: {checkpoint_path}")
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device)
        predictor = SamPredictor(sam)
        return predictor
    except ImportError:
        raise RuntimeError("segment_anything not installed. Install it to use bbox-based mask generation.")


Inference, load_image, load_mask = _load_local_inference_symbols()

app = FastAPI(title="SAM 3D Objects Inference API")

inference_instance: Optional[Any] = None
sam_model: Optional[Any] = None


def _output_dir() -> Path:
    output_dir = Path(os.environ.get("SAM3D_OUTPUT_DIR", "/root/sam-3d-objects/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _result_paths(request_id: str) -> tuple[Path, Path]:
    output_dir = _output_dir()
    return output_dir / f"{request_id}.glb", output_dir / f"{request_id}.json"


def _result_mask_path(request_id: str) -> Path:
    output_dir = _output_dir()
    return output_dir / f"{request_id}_mask.png"


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


def _parse_bbox(bbox_text: Optional[str]) -> Optional[list[float]]:
    """Parse bbox from JSON string format [x_min, y_min, x_max, y_max]."""
    if not bbox_text:
        return None
    
    try:
        bbox = json.loads(bbox_text)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must have exactly 4 values")
        return [float(v) for v in bbox]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid bbox: {exc}")


def _generate_mask_from_bbox(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    """Generate mask from bbox using SAM model."""
    global sam_model
    if sam_model is None:
        raise HTTPException(status_code=500, detail="SAM model not initialized")
    
    x_min, y_min, x_max, y_max = bbox
    input_box = np.array([[x_min, y_min, x_max, y_max]])
    
    try:
        sam_model.set_image(image)
        masks, _, _ = sam_model.predict(
            box=input_box,
            multimask_output=False,
        )
        # masks shape: (1, H, W) for single mask
        mask = masks[0].astype(bool)
        return mask
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SAM inference error: {exc}")


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
    global sam_model
    get_inference()
    # Initialize SAM model
    try:
        sam_model = _load_sam_model()
    except Exception as exc:
        print(f"Warning: SAM model initialization failed: {exc}")
        print("Bbox-based mask generation will not be available")


@app.post("/infer")
async def infer(
    image: UploadFile = File(..., description="RGB image"),
    mask: Optional[UploadFile] = File(None, description="Optional binary/alpha mask image"),
    bbox: Optional[str] = Form(None, description="Optional bbox as JSON [x_min, y_min, x_max, y_max]"),
    return_mask: bool = Form(False, description="If true and using bbox, return generated mask info"),
    depth_image: Optional[UploadFile] = File(None, description="Optional depth image file"),
    K: Optional[str] = Form(None, description="Optional 3x3 intrinsics JSON string"),
    depth_scale: float = Form(512.0),
    invalid_depth_value: Optional[float] = Form(65535.0),
    request_id: Optional[str] = Form(None),
    seed: int = Form(42),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, image.filename or "image.png")

        try:
            with open(image_path, "wb") as file_obj:
                file_obj.write(await image.read())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save image: {exc}")

        try:
            img_np = load_image(image_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {exc}")

        # Load or generate mask
        mask_np = None
        mask_generated_from_bbox = False
        if mask is not None:
            # Mask file provided directly
            mask_path = os.path.join(tmpdir, mask.filename or "mask.png")
            try:
                with open(mask_path, "wb") as file_obj:
                    file_obj.write(await mask.read())
                mask_np = load_mask(mask_path)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to load mask: {exc}")
        elif bbox is not None:
            # Generate mask from bbox using SAM
            bbox_list = _parse_bbox(bbox)
            try:
                mask_np = _generate_mask_from_bbox(img_np, bbox_list)
                mask_generated_from_bbox = True
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to generate mask from bbox: {exc}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'mask' file or 'bbox' parameter must be provided"
            )

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

        mask_path = None
        if return_mask and mask_generated_from_bbox and mask_np is not None:
            mask_path = _result_mask_path(stem)
            mask_uint8 = (mask_np.astype(np.uint8) * 255)
            try:
                iio.imwrite(str(mask_path), mask_uint8)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to save generated mask: {exc}")

        pose_json_text = json.dumps(pose_result, separators=(",", ":"))
        pose_json_b64 = base64.b64encode(pose_json_text.encode("utf-8")).decode("ascii")
        extra_headers = {
            "X-Pose-Json-B64": pose_json_b64,
            "X-Json-File": str(json_path),
            "X-Request-Id": stem,
        }
        if mask_path is not None:
            try:
                mask_bytes = Path(mask_path).read_bytes()
                extra_headers["X-Mask-Png-B64"] = base64.b64encode(mask_bytes).decode("ascii")
            except Exception:
                pass

        return FileResponse(
            path=str(glb_path),
            media_type="model/gltf-binary",
            filename=f"{stem}.glb",
            headers=extra_headers,
        )


@app.post("/infer_sam")
async def infer_sam(
    image: UploadFile = File(..., description="RGB image"),
    bbox: str = Form(..., description="Bbox as JSON [x_min, y_min, x_max, y_max]"),
):
    """Generate mask from image and bbox using SAM model.
    
    Args:
        image: RGB image file
        bbox: Bbox as JSON string [x_min, y_min, x_max, y_max]
    
    Returns:
        PNG mask image
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, image.filename or "image.png")

        try:
            with open(image_path, "wb") as file_obj:
                file_obj.write(await image.read())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save image: {exc}")

        try:
            img_np = load_image(image_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {exc}")

        # Parse and validate bbox
        bbox_list = _parse_bbox(bbox)
        
        try:
            mask_np = _generate_mask_from_bbox(img_np, bbox_list)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate mask from bbox: {exc}")

        # Convert mask to PNG image and return
        mask_uint8 = (mask_np.astype(np.uint8) * 255)
        mask_png_bytes = BytesIO()
        iio.imwrite(mask_png_bytes, mask_uint8, extension=".png")
        mask_png_bytes.seek(0)

        return FileResponse(
            content=mask_png_bytes.getvalue(),
            media_type="image/png",
            filename="mask.png",
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


@app.get("/results/{request_id}/mask")
async def download_mask(request_id: str):
    mask_path = _result_mask_path(request_id)
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail=f"Mask not found for request_id={request_id}")
    return FileResponse(
        path=str(mask_path),
        media_type="image/png",
        filename=f"{request_id}_mask.png",
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=6006, reload=False)
