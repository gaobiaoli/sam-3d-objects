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


import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_rotation_change_xyz(
    quat: np.ndarray,
    init_quat: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """
    计算当前姿态相对于初始姿态的 XYZ 欧拉角变化。

    Parameters
    ----------
    quat:
        当前四元数，顺序为 [w, x, y, z]。
    init_quat:
        初始四元数，顺序为 [w, x, y, z]。
    degrees:
        True 返回角度，False 返回弧度。

    Returns
    -------
    np.ndarray
        [delta_x, delta_y, delta_z]
    """
    quat = np.asarray(quat, dtype=float)
    init_quat = np.asarray(init_quat, dtype=float)

    if quat.shape != (4,) or init_quat.shape != (4,):
        raise ValueError("四元数必须为长度为 4 的数组：[w, x, y, z]")

    if np.linalg.norm(quat) < 1e-12:
        raise ValueError("当前四元数的模不能为 0")

    if np.linalg.norm(init_quat) < 1e-12:
        raise ValueError("初始四元数的模不能为 0")

    # 归一化
    quat = quat / np.linalg.norm(quat)
    init_quat = init_quat / np.linalg.norm(init_quat)

    # scipy 的四元数顺序是 [x, y, z, w]
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    init_quat_xyzw = np.array([
        init_quat[1],
        init_quat[2],
        init_quat[3],
        init_quat[0],
    ])

    current_rotation = Rotation.from_quat(quat_xyzw)
    initial_rotation = Rotation.from_quat(init_quat_xyzw)

    # 当前姿态相对于初始姿态的相对旋转
    relative_rotation = current_rotation * initial_rotation.inv()

    # 转换为 XYZ 欧拉角
    delta_xyz = relative_rotation.as_euler("xyz", degrees=degrees)

    return delta_xyz