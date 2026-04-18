import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

GLB_YUP_TO_ZUP = np.array(
	[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
	dtype=np.float64,
)
PYTORCH3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
GLB_YUP_TO_ZUP_EXTENDED = np.eye(4, dtype=np.float64)
GLB_YUP_TO_ZUP_EXTENDED[:3, :3] = GLB_YUP_TO_ZUP


def build_o3d_transform_from_p3d(quat_wxyz, translation, scale, for_glb = True):
    rot = quat_wxyz_to_rotmat([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]]).T
    trans = translation.copy()
    rot = PYTORCH3D_TO_OPENCV @ rot 
    trans = (PYTORCH3D_TO_OPENCV @ trans.reshape(3, 1)).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot @ np.diag(scale)
    transform[:3, 3] = trans
    if for_glb:
        transform = transform @ GLB_YUP_TO_ZUP_EXTENDED 
    return transform

def transform2pose(transform):
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    scale = np.linalg.norm(rot, axis=0)

    # Normalize the rotation matrix
    rot = rot / scale

    quat = R.from_matrix(rot).as_quat()  # [x, y, z, w]
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转换为 [w, x, y, z]

    return quat_wxyz, trans, scale

def pose2transform(quat_wxyz, translation, scale):
    rot = quat_wxyz_to_rotmat([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot @ np.diag(scale)
    transform[:3, 3] = translation
    return transform

def quat_wxyz_to_rotmat(qwxyz):
    """
    四元数转旋转矩阵（输入为 [qw, qx, qy, qz]）
    """
    qw, qx, qy, qz = qwxyz
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def load_pose_json(pose_json: Path, object_key: str = "object_0"):
	if not pose_json.exists():
		raise FileNotFoundError(f"pose json not found: {pose_json}")

	with open(pose_json, "r", encoding="utf-8") as f:
		data = json.load(f)

	if object_key not in data:
		raise KeyError(f"object key '{object_key}' not found in {pose_json}")

	entry = data[object_key]
	quat = np.asarray(entry.get("rotation"), dtype=np.float64).reshape(-1)
	trans = np.asarray(entry.get("translation"), dtype=np.float64).reshape(-1)
	scale = np.asarray(entry.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)

	if quat.size != 4:
		raise ValueError(f"rotation must be [w,x,y,z], got shape={quat.shape}")
	if trans.size != 3:
		raise ValueError(f"translation must be 3 values, got shape={trans.shape}")
	if scale.size == 1:
		scale = np.repeat(scale, 3)
	if scale.size != 3:
		raise ValueError(f"scale must be scalar or 3 values, got shape={scale.shape}")

	return quat, trans, scale