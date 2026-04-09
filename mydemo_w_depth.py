# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask
from pathlib import Path
import json
import torch
import numpy as np
import imageio.v3 as iio


IMAGE_PATH = Path("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_rgb.png")
MASK_PATH = Path("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_rgb_mask_1.png")
DEPTH_PATH = Path("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_depth.png")
POSE_PATH = Path("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_pose.json")
OUT_JSON_PATH = Path("/root/sam-3d-objects/demo/demo.json")
OUT_GLB_PATH = Path("/root/sam-3d-objects/demo/demo.glb")

def _to_list_1d(value, expected_len: int):
	if isinstance(value, torch.Tensor):
		arr = value.detach().cpu().numpy()
	else:
		arr = np.asarray(value)
	arr = np.squeeze(arr)
	arr = arr.reshape(-1)
	if arr.size < expected_len:
		raise ValueError(
			f"Invalid pose field length: expected {expected_len}, got {arr.size}"
		)
	return arr[:expected_len].astype(np.float32).tolist()


def load_camera_k_matrix(pose_path: Path) -> np.ndarray:
	pose_data = json.loads(pose_path.read_text(encoding="utf-8"))
	k_matrix = np.asarray(pose_data["camera_k_matrix"], dtype=np.float32)
	if k_matrix.shape != (3, 3):
		raise ValueError(f"camera_k_matrix must be 3x3, got {k_matrix.shape}")
	return k_matrix


def depth_to_pointmap(depth_path: Path, k_matrix: np.ndarray, depth_scale: float = 512.0,
	invalid_depth_value: int = 65535) -> torch.Tensor:
	depth = iio.imread(depth_path).astype(np.float32)
	
	if depth.ndim == 3:
		depth = depth[..., 0]

	depth[depth == invalid_depth_value] = 0.0

	depth = depth / depth_scale

	depth[depth <= 0] = np.nan

	h, w = depth.shape
	fx = k_matrix[0, 0]
	fy = k_matrix[1, 1]
	cx = k_matrix[0, 2]
	cy = k_matrix[1, 2]

	u = np.arange(w, dtype=np.float32)
	v = np.arange(h, dtype=np.float32)
	uu, vv = np.meshgrid(u, v)

	z = depth
	x = (uu - cx) * z / fx
	y = (vv - cy) * z / fy

	pointmap = np.stack([-x, -y, z], axis=-1).astype(np.float32)
	return torch.tensor(pointmap, dtype=torch.float32)

# load model
tag = "hf"
config_path = f"../autodl-tmp/checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
k_matrix = load_camera_k_matrix(POSE_PATH)
pointmap = depth_to_pointmap(DEPTH_PATH, k_matrix)
image = load_image(str(IMAGE_PATH))
mask = load_mask(str(MASK_PATH))

# run model
output = inference(image, mask, seed=42, pointmap=pointmap)


# save
pose_json = {
    "object_0": {
        "scale": _to_list_1d(output["scale"], 3),
        "translation": _to_list_1d(output["translation"], 3),
        "rotation": _to_list_1d(output["rotation"], 4),
    }
}

OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON_PATH.write_text(json.dumps(pose_json, indent=2), encoding="utf-8")
output["glb"].export(str(OUT_GLB_PATH))

print(f"Saved GLB to {OUT_GLB_PATH}")
print(f"Saved pose JSON to {OUT_JSON_PATH}")
