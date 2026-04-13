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

        request_key = resp.headers.get("X-Request-Id") or request_id or "output"
        local_glb = output_folder / f"{request_key}.glb"
        with local_glb.open("wb") as file_obj:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)

        poses_b64 = resp.headers.get("X-Pose-Json-B64")
        local_json = output_folder / f"{request_key}.json"
        if poses_b64:
            try:
                pose_text = base64.b64decode(poses_b64).decode("utf-8")
                try:
                    pose_obj = json.loads(pose_text)
                    local_json.write_text(
                        json.dumps(pose_obj, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except json.JSONDecodeError:
                    local_json.write_text(pose_text, encoding="utf-8")
            except Exception as exc:
                print(f"Warning: failed to decode X-Pose-Json-B64: {exc}")

        print(f"Saved GLB to: {local_glb.resolve()}")
        print(f"Saved JSON to: {local_json.resolve()}")

        return local_glb, local_json