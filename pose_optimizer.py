import json
from pathlib import Path
import copy
import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image
try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False

def load_mask(mask_path, threshold=128):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > threshold).astype(np.uint8)
    return mask


def erode_mask(mask, kernel_size=3, iterations=1):
    if kernel_size <= 1 or not has_cv2:
        return mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)


def depth_edge_from_point_map(point_map, rtol=0.03):
    z = point_map[..., 2]
    valid = np.isfinite(z) & (z > 0)

    edge = np.zeros_like(valid, dtype=bool)

    dz_right = np.zeros_like(z, dtype=np.float32)
    dz_down = np.zeros_like(z, dtype=np.float32)

    dz_right[:, :-1] = np.abs(z[:, 1:] - z[:, :-1])
    dz_down[:-1, :] = np.abs(z[1:, :] - z[:-1, :])

    rel_right = dz_right / np.maximum(np.abs(z), 1e-6)
    rel_down = dz_down / np.maximum(np.abs(z), 1e-6)

    edge |= rel_right > rtol
    edge |= rel_down > rtol

    return edge & valid


def extract_target_points_from_point_map(
    point_map,
    mask,
    erode_kernel=3,
    remove_depth_edges=True,
    depth_edge_rtol=0.03,
    max_points=20000,
    remove_outliers=True,
    outlier_percentile=98.0,
):
    mask = mask.astype(np.uint8)

    if erode_kernel > 1:
        mask = erode_mask(mask, kernel_size=erode_kernel)

    #去除0，0，0的点
    valid = np.isfinite(point_map).all(axis=-1) & (point_map[..., 2] > 0)
    # valid = np.isfinite(point_map).all(axis=-1)

    if remove_depth_edges:
        edge = depth_edge_from_point_map(point_map, rtol=depth_edge_rtol)
        valid = valid & (~edge)

    valid = valid & (mask > 0)

    target_points = point_map[valid]

    if len(target_points) == 0:
        raise ValueError("no valid target points found inside mask")

    if remove_outliers and len(target_points) >= 50:
        center = np.median(target_points, axis=0, keepdims=True)
        dist = np.linalg.norm(target_points - center, axis=1)
        th = np.percentile(dist, outlier_percentile)
        target_points = target_points[dist <= th]

    if len(target_points) > max_points:
        idx = np.random.choice(len(target_points), max_points, replace=False)
        target_points = target_points[idx]

    return target_points


def open3d_to_trimesh(mesh_o3d):
    vertices = np.asarray(mesh_o3d.vertices)
    triangles = np.asarray(mesh_o3d.triangles)

    if len(vertices) == 0:
        raise ValueError("open3d mesh has no vertices")

    if len(triangles) == 0:
        raise ValueError("open3d mesh has no triangles")

    mesh_tm = trimesh.Trimesh(
        vertices=vertices,
        faces=triangles,
        process=False
    )
    return mesh_tm

def scene_to_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh.copy()

    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.dump(concatenate=True)
        if isinstance(mesh, trimesh.trimesh):
            return mesh
        raise ValueError("failed to concatenate trimesh.scene into trimesh")

    if isinstance(scene_or_mesh, o3d.geometry.TriangleMesh):
        return open3d_to_trimesh(scene_or_mesh)

    raise ValueError(f"unsupported type: {type(scene_or_mesh)}")


def sample_source_points_from_mesh(mesh, num_points=30000):
    mesh = scene_to_mesh(mesh)

    if mesh.faces is not None and len(mesh.faces) > 0:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    else:
        verts = np.asarray(mesh.vertices)
        if len(verts) > num_points:
            idx = np.random.choice(len(verts), num_points, replace=False)
            points = verts[idx]
        else:
            points = verts.copy()

    return points


def quaternion_to_matrix_torch(q):
    q = q / (torch.norm(q) + 1e-12)
    w, x, y, z = q[0], q[1], q[2], q[3]

    r = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ],
        dim=0,
    )
    return r


class delta_pose_optimizer:
    def __init__(
        self,
        source_points,
        target_points,
        device="cuda",
        optimize_scale=True,
        trim_ratio=0.9,
        init_quat=None,
        init_trans=None,
        init_log_scale=None,
        K=None  
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.optimize_scale = optimize_scale
        self.trim_ratio = trim_ratio
        self.K = K
        self.source = torch.tensor(source_points, dtype=torch.float32, device=self.device)
        self.target = torch.tensor(target_points, dtype=torch.float32, device=self.device)
        if K is not None:
            self.K = torch.tensor(K, dtype=torch.float32, device=self.device)
        if init_quat is None:
            init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        else:
            init_quat = torch.tensor(init_quat, dtype=torch.float32, device=self.device)
        if init_trans is None:
            init_trans = torch.zeros(3, dtype=torch.float32, device=self.device)
        else:
            init_trans = torch.tensor(init_trans, dtype=torch.float32, device=self.device)
        if init_log_scale is None:
            init_log_scale = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            init_log_scale = torch.tensor(init_log_scale, dtype=torch.float32, device=self.device)

        self.source_center = self.source.mean(dim=0, keepdim=True)
    
        if optimize_scale:
            self.log_scale = torch.nn.Parameter(init_log_scale.clone())
        else:
            self.log_scale = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.quat = torch.nn.Parameter(init_quat.clone())
        self.trans = torch.nn.Parameter(init_trans.clone())

        self.init_quat = self.quat.data.clone()
        self.init_trans = self.trans.data.clone()
        self.init_log_scale = self.log_scale.data.clone() if optimize_scale else self.log_scale.clone()

    def transform_source(self):
        # centered = self.source - self.source_center
        scale = torch.exp(self.log_scale)
        r = quaternion_to_matrix_torch(self.quat) @ torch.diag(scale.expand(3))
        transformed = (self.source) @ r.t() + self.trans
        return transformed

    def one_sided_trimmed_cd(self, batch_size=4096):
        src = self.transform_source()
        self.new_source = src
        all_min_dists = []

        for i in range(0, len(self.target), batch_size):
            tgt_batch = self.target[i:i + batch_size]
            d = torch.cdist(tgt_batch, src)
            min_d, _ = torch.min(d, dim=1)
            all_min_dists.append(min_d)

        min_dists = torch.cat(all_min_dists, dim=0)

        if 0 < self.trim_ratio < 1.0 and len(min_dists) > 10:
            k = max(1, int(len(min_dists) * self.trim_ratio))
            min_dists, _ = torch.topk(min_dists, k=k, largest=False)

        return min_dists.mean()
    def main_axis_loss(self):
        if self.K is None:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        rot =quaternion_to_matrix_torch(self.quat)
        rot_world = self.K[:3, :3] @ rot
        z_world = torch.tensor([0.0, 0.0, 1.0], device=rot_world.device, dtype=rot_world.dtype)
        dots = rot_world.t() @ z_world                       # (3,)

        # 只关心“平行”，不区分 +z / -z
        loss = 1.0 - torch.max(dots ** 2)
        return loss * 0.5
    def regularization_loss(self):
        quat_reg = 0.001 * torch.sum((self.quat - self.init_quat) ** 2)
        trans_reg = 0.001 * torch.sum((self.trans - self.init_trans) ** 2)

        if self.optimize_scale:
            scale_reg = 0.001 * (self.log_scale - self.init_log_scale) ** 2
        else:
            scale_reg = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return quat_reg + trans_reg + scale_reg

    def depth_scale_constraint_loss(self):
        if not self.optimize_scale:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)


        delta_depth = self.trans[2] - self.init_trans[2]
        delta_log_scale = self.log_scale - self.init_log_scale
        same_direction = torch.relu(-delta_depth * delta_log_scale)
        return 200.0 * same_direction

    def bbox_scale_regularization_loss(self):
        if not self.optimize_scale:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # 使用长宽高表征尺度：计算 target/reference 在 xyz 三轴上的尺寸比，
        # 取最大比值作为期望缩放因子，并约束当前 scale 接近该值。
        src = self.new_source
        src_extent = torch.amax(src, dim=0) - torch.amin(src, dim=0)
        tgt_extent = torch.amax(self.target, dim=0) - torch.amin(self.target, dim=0)

        ratios = tgt_extent / torch.clamp(src_extent, min=1e-6)
        desired_scale = torch.min(ratios)
        desired_log_scale = torch.log(torch.clamp(desired_scale, min=1e-6))

        return 0.1 * ((self.log_scale - desired_log_scale) ** 2)
    
    def compute_loss(self):
        cd = self.one_sided_trimmed_cd()
        reg = self.regularization_loss()
        axis = self.main_axis_loss()
        depth_scale = self.depth_scale_constraint_loss()
        bbox_scale = self.bbox_scale_regularization_loss()
        loss = cd + reg + axis+ depth_scale+ bbox_scale
        return loss, cd, axis, depth_scale, bbox_scale

    def optimize(self, num_iterations=300, lr=0.01, patience=60):
        # 前半程锁定 quat，仅优化平移/尺度；后半程再解锁 quat
        quat_group = {"name": "quat", "params": [self.quat], "lr": 0.0}
        trans_group = {"name": "trans", "params": [self.trans], "lr": lr * 5.0}
        params = [quat_group, trans_group]
        if self.optimize_scale:
            params.insert(0, {"name": "scale", "params": [self.log_scale], "lr": lr * 5.0})

        optimizer = torch.optim.Adam(params)
        step_size = num_iterations
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        unlock_iter = max(1, num_iterations // 2)

        history = {"loss": [], "cd": [], "scale": []}

        best_cd = float("inf")
        best_state = None
        bad_steps = 0

        for it in range(num_iterations):
            if it == unlock_iter:
                for group in optimizer.param_groups:
                    if group.get("name") == "quat":
                        group["lr"] = lr
                        print(f"unlock quat at iter {it + 1}, quat lr={group['lr']}")

            optimizer.zero_grad()
            loss, cd, axis, depth_scale, bbox_scale = self.compute_loss()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.quat[:] = self.quat / (torch.norm(self.quat) + 1e-12)

            scale = float(torch.exp(self.log_scale).detach().cpu().item())
            cd_val = float(cd.detach().cpu().item())
            axis_val = float(axis.detach().cpu().item())
            depth_scale_val = float(depth_scale.detach().cpu().item())
            bbox_scale_val = float(bbox_scale.detach().cpu().item())
            loss_val = float(loss.detach().cpu().item())

            history["loss"].append(loss_val)
            history["cd"].append(cd_val)
            history["scale"].append(scale)

            if cd_val < best_cd or it < unlock_iter + 10:
                best_cd = cd_val
                best_state = {
                    "quat": self.quat.detach().clone(),
                    "trans": self.trans.detach().clone(),
                    "log_scale": self.log_scale.detach().clone() if self.optimize_scale else self.log_scale.clone(),
                }
                bad_steps = 0
            else:
                bad_steps += 1

            if (it + 1) % 20 == 0:
                print(
                    f"iter {it + 1:04d} | "
                    f"loss={loss_val:.6f} | cd={cd_val:.6f} | axis={axis_val:.6f} | depth_scale={depth_scale_val:.6f} | bbox_scale={bbox_scale_val:.6f} | scale={scale:.5f}"
                )

            if bad_steps >= patience:
                print(f"early stopping at iter {it + 1}")
                break

        if best_state is not None:
            with torch.no_grad():
                self.quat[:] = best_state["quat"]
                self.trans[:] = best_state["trans"]
                if self.optimize_scale:
                    self.log_scale.copy_(best_state["log_scale"])

        return history

    def get_delta_pose(self):
        with torch.no_grad():
            scale = float(torch.exp(self.log_scale).cpu().item())
            quat = (self.quat / (torch.norm(self.quat) + 1e-12)).cpu().numpy()
            trans = self.trans.cpu().numpy()

        return {
            "scale": np.array([scale, scale, scale], dtype=np.float32),
            "rotation_wxyz": quat.astype(np.float32),
            "translation": trans.astype(np.float32),
        }

def quat_wxyz_to_rotmat(qwxyz):
    """
    四元数转旋转矩阵（输入为 [qw, qx, qy, qz]）
    """
    qw, qx, qy, qz = qwxyz
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def pose2transform(quat_wxyz, translation, scale):
    rot = quat_wxyz_to_rotmat([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot @ np.diag(scale)
    transform[:3, 3] = translation
    return transform

def save_pose_json(save_path, delta_pose):
    payload = {
        "object_0": {
            "scale": delta_pose["scale"].tolist(),
            "rotation": delta_pose["rotation_wxyz"].tolist(),
            "translation": delta_pose["translation"].tolist(),
        }
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def transform2pose(transform):
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    scale = np.linalg.norm(rot, axis=0)

    # Normalize the rotation matrix
    rot = rot / scale

    quat = R.from_matrix(rot).as_quat()  # [x, y, z, w]
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转换为 [w, x, y, z]

    return quat_wxyz, trans, scale


def optimize_pose_from_loaded_mesh(
    glb_mesh,
    point_map,
    mask_path,
    device="cuda",
    optimize_scale=True,
    target_max_points=20000,
    source_num_points=30000,
    erode_kernel=3,
    remove_depth_edges=True,
    depth_edge_rtol=0.03,
    trim_ratio=0.9,
    num_iterations=300,
    lr=0.01,
    patience=60,
    save_refined_glb_path=None,
    save_delta_pose_path=None,
    base_transform=np.eye(4, dtype=np.float32),
    K=None,
):  
    #copy glb: open3d.cpu.pybind.geometry.TriangleMesh
    raw_glb = copy.deepcopy(glb_mesh)
    init_quat_wxyz, init_translation, scale = transform2pose(base_transform)
    init_log_scale = np.log(scale[0]) if optimize_scale else None

    mask = load_mask(mask_path)

    target_points = extract_target_points_from_point_map(
        point_map=point_map,
        mask=mask,
        erode_kernel=erode_kernel,
        remove_depth_edges=remove_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        max_points=target_max_points,
        remove_outliers=True,
        outlier_percentile=98.0,
    )
    
    source_points = sample_source_points_from_mesh(
        glb_mesh,
        num_points=source_num_points,
    )

    src_center = source_points.mean(axis=0)
    tgt_center = target_points.mean(axis=0)
    print("initial scale:", scale[0])
    print("source center:", src_center)
    print("target center:", tgt_center)
    print("center distance before opt:", np.linalg.norm(src_center - tgt_center))

    optimizer = delta_pose_optimizer(
        source_points=source_points,
        target_points=target_points,
        device=device,
        optimize_scale=optimize_scale,
        trim_ratio=trim_ratio,
        init_quat=init_quat_wxyz,
        init_trans=init_translation,
        init_log_scale=init_log_scale,
        K=K,
    )

    history = optimizer.optimize(
        num_iterations=num_iterations,
        lr=lr,
        patience=patience,
    )

    delta_pose = optimizer.get_delta_pose()
    # refined_mesh = apply_delta_to_mesh(glb_mesh, delta_pose)
    refined_source_points = sample_source_points_from_mesh(raw_glb, num_points=min(source_num_points, 10000))
    refined_center = refined_source_points.mean(axis=0)
    print("refined source center:", refined_center)
    print("center distance after opt:", np.linalg.norm(refined_center - tgt_center))
    print("estimated scale:", delta_pose["scale"][0])
    print("estimated rotation (wxyz):", delta_pose["rotation_wxyz"])
    print("estimated translation:", delta_pose["translation"])
    
    return {
        "target_points": target_points,
        "delta_pose": delta_pose,
        "history": history,
        "refined_mesh": raw_glb,
    }


if __name__ == "__main__":
    pass