# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from pathlib import Path
import json
import torch
import numpy as np
import imageio.v3 as iio
import trimesh
from utils import build_o3d_transform_from_p3d
from pose_optimizer import optimize_pose_from_loaded_mesh
import open3d as o3d

IMAGE_PATH = Path("tmp/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_25_domain_rgb.png")
MASK_PATH = Path("tmp/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_25_domain_rgb_chair_5_mask.png")
DEPTH_PATH = Path("tmp/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_25_domain_depth.png")
POSE_PATH = Path("tmp/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_25_domain_pose.json")
# IMAGE_PATH = Path("demo/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_14_domain_rgb.png")
# MASK_PATH = Path("demo/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_14_domain_rgb_sofa_4_mask.png")
# DEPTH_PATH = Path("demo/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_14_domain_depth.png")
# POSE_PATH = Path("demo/camera_127fc8dfeaa8426fb28965db5171b14a_lounge_1_frame_14_domain_pose.json")
OUT_JSON_PATH = Path("/root/sam-3d-objects/demo/demo.json")
OUT_OPTIMIZED_JSON_PATH = Path("/root/sam-3d-objects/demo/demo_optimized.json")
OUT_GLB_PATH = Path("/root/sam-3d-objects/demo/demo.glb")

OPTIMIZE_NUM_ITERATIONS = 300
IOU_LOSS = True


def _load_cached_glb(glb_path: Path):
	mesh = o3d.io.read_triangle_mesh(str(glb_path))
	
	return mesh


def _load_cached_output(glb_path: Path, json_path: Path) -> dict:
	pose_json = json.loads(json_path.read_text(encoding="utf-8"))
	if "object_0" not in pose_json:
		raise KeyError(f"object_0 not found in cached pose JSON: {json_path}")
	pose = pose_json["object_0"]
	return {
		"glb": _load_cached_glb(glb_path),
		"scale": np.asarray(pose["scale"], dtype=np.float32),
		"translation": np.asarray(pose["translation"], dtype=np.float32),
		"rotation": np.asarray(pose["rotation"], dtype=np.float32),
	}

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

def euler_xyz_to_matrix(eul):
        rx, ry, rz = eul

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rx_mat = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx,  cx]
        ], dtype=np.float32)

        ry_mat = np.array([
            [ cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy]
        ], dtype=np.float32)

        rz_mat = np.array([
            [cz, -sz, 0.0],
            [sz,  cz, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return rz_mat @ ry_mat @ rx_mat

def load_camera_k_matrix(pose_path: Path) -> np.ndarray:
	pose_data = json.loads(pose_path.read_text(encoding="utf-8"))
	in_matrix = np.asarray(pose_data["camera_k_matrix"], dtype=np.float32)
	if in_matrix.shape != (3, 3):
		raise ValueError(f"camera_k_matrix must be 3x3, got {in_matrix.shape}")
	rt = np.asarray(pose_data['camera_rt_matrix'], dtype=np.float32)
	R = rt[:3, :3]

	A = euler_xyz_to_matrix(
		np.asarray(pose_data['final_camera_rotation'], dtype=np.float32)
	)
	cam_loc = np.asarray(pose_data['camera_location'], dtype=np.float32).reshape(1, 3)

	transform = np.eye(4, dtype=np.float32)
	transform[:3, :3] = A @ R @ A
	transform[:3, 3] = cam_loc.flatten()
	return in_matrix, transform


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


def process_pointmap_for_optimization(pointmap: torch.Tensor) -> np.ndarray:
	pointmap_np = pointmap.detach().cpu().numpy()
	z = pointmap_np[..., 2]
	x = pointmap_np[..., 0]
	y = pointmap_np[..., 1]
	valid = np.isfinite(z)
	x[~valid] = 0.0
	y[~valid] = 0.0
	z[~valid] = 0.0
	return np.stack([-x, -y, z], axis=-1).astype(np.float32)


def _delta_pose_to_json(delta_pose: dict) -> dict:
	return {
		"object_0": {
			"scale": np.asarray(delta_pose["scale"], dtype=np.float32).reshape(3).tolist(),
			"translation": np.asarray(delta_pose["translation"], dtype=np.float32).reshape(3).tolist(),
			"rotation": np.asarray(delta_pose["rotation_wxyz"], dtype=np.float32).reshape(4).tolist(),
		}
	}


def run_pose_optimization(mesh, pointmap_tensor: torch.Tensor, pose_json: dict,
						  	intrinsics: np.ndarray = None,extrinsics: np.ndarray = None) -> dict:
	base_pose = pose_json["object_0"]
	base_transform = build_o3d_transform_from_p3d(
		quat_wxyz=np.array(base_pose["rotation"]),
		translation=np.array(base_pose["translation"]),
		scale=np.array(base_pose["scale"]),
		for_glb=True,
	)
	print("Base transform for optimization:", base_transform)

	optimize_result = optimize_pose_from_loaded_mesh(
		glb_mesh=mesh,
		point_map=process_pointmap_for_optimization(pointmap_tensor),
		mask_path=str(MASK_PATH),
		device="cuda" if torch.cuda.is_available() else "cpu",
		optimize_scale=True,
		trim_ratio=0.8,
		num_iterations=OPTIMIZE_NUM_ITERATIONS,
		optimize_independent_scale=False,
		base_transform=base_transform,
		iou_loss_weight=1,
		K=extrinsics,
		intrinsics=intrinsics,
		iou_loss=IOU_LOSS,
	)
	return _delta_pose_to_json(optimize_result["delta_pose"])

# load depth pointmap for pose optimization
k_matrix, ex_matrix = load_camera_k_matrix(POSE_PATH)
pointmap = depth_to_pointmap(DEPTH_PATH, k_matrix)

cache_exists = OUT_GLB_PATH.exists() and OUT_JSON_PATH.exists()
if cache_exists:
	print(f"Loading cached output from {OUT_GLB_PATH} and {OUT_JSON_PATH}")
	output = _load_cached_output(OUT_GLB_PATH, OUT_JSON_PATH)
else:
	from inference import Inference, load_image, load_mask
	# load model
	tag = "hf"
	config_path = f"../autodl-tmp/checkpoints/{tag}/pipeline.yaml"
	inference = Inference(config_path, compile=False)

	# load image (RGBA only, mask is embedded in the alpha channel)
	image = load_image(str(IMAGE_PATH))
	mask = load_mask(str(MASK_PATH))

	# run model
	output = inference(image, mask, seed=42)


# save initial prediction and server-default optimized pose
pose_json = {
    "object_0": {
        "scale": _to_list_1d(output["scale"], 3),
        "translation": _to_list_1d(output["translation"], 3),
        "rotation": _to_list_1d(output["rotation"], 4),
    }
}
pose_json_optimized = run_pose_optimization(output["glb"], pointmap, pose_json,intrinsics=k_matrix
												,extrinsics=ex_matrix)

OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
if not cache_exists:
	OUT_JSON_PATH.write_text(json.dumps(pose_json, indent=2), encoding="utf-8")
OUT_OPTIMIZED_JSON_PATH.write_text(json.dumps(pose_json_optimized, indent=2), encoding="utf-8")
if not cache_exists:
	output["glb"].export(str(OUT_GLB_PATH))

print(f"Saved GLB to {OUT_GLB_PATH}")
print(f"Saved pose JSON to {OUT_JSON_PATH}")
print(f"Saved optimized pose JSON to {OUT_OPTIMIZED_JSON_PATH}")
