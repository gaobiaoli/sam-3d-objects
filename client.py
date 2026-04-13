import base64
import json
from pathlib import Path

import requests


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run_test(
    url,
    image_path,
    mask_path=None,
    bbox=None,
    depth_path=None,
    K=None,
    request_id=None,
    depth_scale=1000.0,
    invalid_depth_value=65535.0,
    seed=42,
    return_mask=False,
    output_dir=".",
) -> None:
    depth_file_obj = None
    mask_file_obj = None
    files = {}
    with open(image_path, "rb") as image_file:
        files["image"] = (Path(image_path).name, image_file, "image/png")

        if mask_path is not None:
            mask_path = Path(mask_path)
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask file: {mask_path}")
            mask_file_obj = mask_path.open("rb")
            files["mask"] = (mask_path.name, mask_file_obj, "image/png")

        if mask_path is None and bbox is None:
            raise ValueError("Either mask_path or bbox must be provided")

        if depth_path is not None:
            depth_path = Path(depth_path)
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth file: {depth_path}")
            depth_file_obj = depth_path.open("rb")
            files["depth_image"] = (depth_path.name, depth_file_obj, "image/png")

        data = {
            "request_id": request_id,
            "depth_scale": depth_scale,
            "invalid_depth_value": invalid_depth_value,
            "seed": seed,
        }
        if K is not None:
            data["K"] = json.dumps(_jsonable(K))
        if bbox is not None:
            data["bbox"] = json.dumps(_jsonable(bbox))
            if return_mask:
                data["return_mask"] = "true"

        try:
            resp = requests.post(url, files=files, data=data, timeout=600, stream=True)
        finally:
            if depth_file_obj is not None:
                depth_file_obj.close()
            if mask_file_obj is not None:
                mask_file_obj.close()

        if resp.status_code != 200:
            raise SystemExit(f"Request failed: {resp.status_code} {resp.text}")

        output_folder = Path(output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)

        request_key = request_id or "output"
        local_glb = output_folder / f"{request_key}.glb"
        local_json = output_folder / f"{request_key}.json"
        local_mask = None

        payload = resp.json()
        request_key = payload.get("request_id") or request_key
        local_glb = output_folder / f"{request_key}.glb"
        local_json = output_folder / f"{request_key}.json"

        glb_b64 = payload.get("glb_b64")
        if not glb_b64:
            raise SystemExit("Request failed: JSON body missing glb_b64")
        local_glb.write_bytes(base64.b64decode(glb_b64))

        pose_obj = payload.get("pose")
        if pose_obj is not None:
            local_json.write_text(
                json.dumps(pose_obj, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        mask_b64 = payload.get("mask_png_b64")
        if mask_b64:
            local_mask = output_folder / f"{request_key}_mask.png"
            local_mask.write_bytes(base64.b64decode(mask_b64))
            print(f"Saved Mask to: {local_mask.resolve()}")

        print(f"Saved GLB to: {local_glb.resolve()}")
        print(f"Saved JSON to: {local_json.resolve()}")

        return local_glb, local_json, local_mask