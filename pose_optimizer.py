import json
from pathlib import Path
import copy
import numpy as np
import torch
import trimesh
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
    TexturesVertex,
)
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from PIL import Image
import random
try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False

def load_mask(mask_path, threshold=128):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > threshold).astype(np.uint8)
    return mask


def _denormalize_f(norm_K, height, width):
    cx_norm = norm_K[0][2]
    cy_norm = norm_K[1][2]
    fx_norm = norm_K[0][0]
    fy_norm = norm_K[1][1]
    s_norm = norm_K[0][1]

    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    s_abs = s_norm * width

    return np.array(
        [[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _build_mask_renderer(mask_np, device,intrinsics=None):
    raw_h, raw_w = mask_np.shape
    target_size = 512
    h, w = target_size, target_size
    if intrinsics is not None:
        #norm
        intrinsics[0,0] = intrinsics[0,0] / raw_w
        intrinsics[1,1] = intrinsics[1,1] / raw_h
        intrinsics[0,2] = intrinsics[0,2] / raw_w
        intrinsics[1,2] = intrinsics[1,2] / raw_h
    else:
        intrinsics = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    
    intrinsics = _denormalize_f(intrinsics, h, w)
    cameras = PerspectiveCameras(
        focal_length=torch.tensor(
            [[intrinsics[0, 0], intrinsics[1, 1]]], device=device, dtype=torch.float32
        ),
        principal_point=torch.tensor(
            [[intrinsics[0, 2], intrinsics[1, 2]]], device=device, dtype=torch.float32
        ),
        image_size=torch.tensor([[h, w]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=1e-6,
        faces_per_pixel=20,
        max_faces_per_bin=200000,
        bin_size=0,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    if mask_t.shape[0] != h or mask_t.shape[1] != w:
        mask_t = F.interpolate(
            mask_t[None, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return renderer, mask_t


def _build_depth_rasterizer(depth_np, mask_np, device, intrinsics=None):
    raw_h, raw_w = depth_np.shape
    if intrinsics is not None:
        intrinsics = np.array(intrinsics, dtype=np.float32, copy=True)
        if np.max(np.abs(intrinsics[:2, :3])) <= 2.0:
            intrinsics = _denormalize_f(intrinsics, raw_h, raw_w)
    else:
        intrinsics = _denormalize_f(
            np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
            raw_h,
            raw_w,
        )

    cameras = PerspectiveCameras(
        focal_length=torch.tensor(
            [[intrinsics[0, 0], intrinsics[1, 1]]], device=device, dtype=torch.float32
        ),
        principal_point=torch.tensor(
            [[intrinsics[0, 2], intrinsics[1, 2]]], device=device, dtype=torch.float32
        ),
        image_size=torch.tensor([[raw_h, raw_w]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(raw_h, raw_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        max_faces_per_bin=200000,
        bin_size=0,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    depth_t = torch.tensor(depth_np, dtype=torch.float32, device=device)
    valid_np = np.isfinite(depth_np) & (depth_np > DEPTH_RESIDUAL_MIN_DEPTH)
    if mask_np is not None:
        valid_np = valid_np & (mask_np > 0)
    valid_t = torch.tensor(valid_np, dtype=torch.bool, device=device)
    return rasterizer, depth_t, valid_t


def _render_depth_residual_loss(verts, faces, quat, trans, log_scale, rasterizer, depth_t, valid_t):
    scale = torch.exp(log_scale)
    r = quaternion_to_matrix_torch(quat) @ torch.diag(scale.expand(3))
    verts_t = verts @ r.t() + trans
    opencv_to_p3d = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        device=verts_t.device,
        dtype=verts_t.dtype,
    )
    verts_t = verts_t @ opencv_to_p3d
    mesh_p3d = Meshes(verts=[verts_t], faces=[faces])
    fragments = rasterizer(mesh_p3d)
    pred_depth = fragments.zbuf[0, ..., 0]
    render_valid = fragments.pix_to_face[0, ..., 0] >= 0
    valid = render_valid & valid_t & torch.isfinite(pred_depth)

    valid_count = int(valid.sum().detach().cpu().item())
    if valid_count < DEPTH_RESIDUAL_MIN_POINTS:
        return torch.tensor(0.0, dtype=torch.float32, device=verts.device)

    residual = pred_depth[valid] - depth_t[valid]
    if residual.numel() > DEPTH_RESIDUAL_MAX_POINTS:
        idx = torch.randperm(residual.numel(), device=residual.device)[:DEPTH_RESIDUAL_MAX_POINTS]
        residual = residual[idx]
    robust = torch.sqrt(residual ** 2 + DEPTH_RESIDUAL_EPS ** 2)
    return DEPTH_RESIDUAL_WEIGHT * robust.mean()


def _simplify_mesh(mesh, target_triangles=3000):
    if isinstance(mesh, trimesh.Trimesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    else:
        mesh_o3d = copy.deepcopy(mesh)
    mesh_o3d = copy.deepcopy(mesh_o3d)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_non_manifold_edges()
    if len(mesh_o3d.triangles) > target_triangles:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_triangles)
    return mesh_o3d


def _build_p3d_mesh(mesh, device, target_triangles=3000):
    
    mesh_tm = open3d_to_trimesh(_simplify_mesh(mesh, target_triangles))
    
    verts = torch.tensor(mesh_tm.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_tm.faces, dtype=torch.int64, device=device)
    textures = TexturesVertex(
        verts_features=torch.ones((1, verts.shape[0], 3), device=device, dtype=verts.dtype)
    )
    return verts, faces, textures


def _render_soft_iou_loss(verts, faces, textures, quat, trans, log_scale, renderer, mask_t):
    scale = torch.exp(log_scale)
    r = quaternion_to_matrix_torch(quat) @ torch.diag(scale.expand(3))
    verts_t = verts @ r.t() + trans
    # OpenCV -> PyTorch3D (flip X/Y) for rendering
    opencv_to_p3d = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        device=verts_t.device,
        dtype=verts_t.dtype,
    )
    verts_t = verts_t @ opencv_to_p3d
    mesh_p3d = Meshes(verts=[verts_t], faces=[faces], textures=textures)
    rendered = renderer(mesh_p3d)
    pred = rendered[..., 3][0].clamp(0.0, 1.0)
    gt = mask_t.clamp(0.0, 1.0)
    # 保存gt约pred 以黑白图
    # cv2.imwrite("gt.png", (gt * 255).detach().cpu().numpy().astype(np.uint8))
    # cv2.imwrite("pred.png", (pred * 255).detach().cpu().numpy().astype(np.uint8))
    # mask iou loss
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    iou = intersection / (union + 1e-8)
    # iou loss
    loss = 1.0 - iou
    return loss, iou
    # 保存pred、gt
    # cv2.imwrite("gt.png", (gt * 255).detach().cpu().numpy().astype(np.uint8))
    # cv2.imwrite("pred.png", (pred * 255).detach().cpu().numpy().astype(np.uint8))
    # h, w = pred.shape
    # x = torch.linspace(0.0, 1.0, w, device=pred.device, dtype=pred.dtype)
    # y = torch.linspace(0.0, 1.0, h, device=pred.device, dtype=pred.dtype)

    # def _soft_bounds(proj, coords):
    #     log_proj = torch.log(proj.clamp_min(1e-8))
    #     inv_temp = 50.0
    #     lo = (torch.softmax(log_proj - coords * inv_temp, dim=0) * coords).sum()
    #     hi = (torch.softmax(log_proj + coords * inv_temp, dim=0) * coords).sum()
    #     return lo, hi

    # pred_x0, pred_x1 = _soft_bounds(pred.amax(dim=0), x)
    # pred_y0, pred_y1 = _soft_bounds(pred.amax(dim=1), y)
    # gt_x0, gt_x1 = _soft_bounds(gt.amax(dim=0), x)
    # gt_y0, gt_y1 = _soft_bounds(gt.amax(dim=1), y)

    # inter_w = (torch.minimum(pred_x1, gt_x1) - torch.maximum(pred_x0, gt_x0)).clamp_min(0.0)
    # inter_h = (torch.minimum(pred_y1, gt_y1) - torch.maximum(pred_y0, gt_y0)).clamp_min(0.0)
    # inter_box = inter_w * inter_h
    # pred_box = (pred_x1 - pred_x0).clamp_min(0.0) * (pred_y1 - pred_y0).clamp_min(0.0)
    # gt_box = (gt_x1 - gt_x0).clamp_min(0.0) * (gt_y1 - gt_y0).clamp_min(0.0)
    # bbox_iou = inter_box / (pred_box + gt_box - inter_box + 1e-8)

    # return 1.0 - bbox_iou


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
        optimize_independent_scale=False,
        trim_ratio=0.9,
        init_quat=None,
        init_trans=None,
        init_log_scale=None,
        ex=None,
        intrinsics = None,
        render_iou_loss_fn=None,
        render_depth_loss_fn=None,
        depth_map=None,
        depth_mask=None,
        scale_range_min_ratio=0.5,
        scale_range_max_ratio=2,
        scale_range_weight=10.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.optimize_scale = optimize_scale
        self.optimize_independent_scale = optimize_independent_scale
        self.scale_range_min_ratio = scale_range_min_ratio
        self.scale_range_max_ratio = scale_range_max_ratio
        self.scale_range_weight = scale_range_weight
        self.trim_ratio = trim_ratio
        self.K = ex
        self.intrinsics = intrinsics
        self.source = torch.tensor(source_points, dtype=torch.float32, device=self.device)
        self.target = torch.tensor(target_points, dtype=torch.float32, device=self.device)
        if ex is not None:
            self.K = torch.tensor(ex, dtype=torch.float32, device=self.device)
        if init_quat is None:
            init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        else:
            init_quat = torch.tensor(init_quat, dtype=torch.float32, device=self.device)
        if init_trans is None:
            init_trans = torch.zeros(3, dtype=torch.float32, device=self.device)
        else:
            init_trans = torch.tensor(init_trans, dtype=torch.float32, device=self.device)
        if init_log_scale is None:
            if optimize_independent_scale:
                init_log_scale = torch.zeros(3, dtype=torch.float32, device=self.device)
            else:
                init_log_scale = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            init_log_scale = torch.tensor(init_log_scale, dtype=torch.float32, device=self.device)
            if optimize_independent_scale and init_log_scale.numel() == 1:
                init_log_scale = init_log_scale.reshape(1).expand(3).clone()

        self.source_center = self.source.mean(dim=0, keepdim=True)
    
        if optimize_scale:
            self.log_scale = torch.nn.Parameter(init_log_scale.clone())
        else:
            if optimize_independent_scale:
                self.log_scale = torch.zeros(3, dtype=torch.float32, device=self.device)
            else:
                self.log_scale = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.quat = torch.nn.Parameter(init_quat.clone())
        self.trans = torch.nn.Parameter(init_trans.clone())

        self.init_quat = self.quat.data.clone()
        self.init_trans = self.trans.data.clone()
        self.init_log_scale = self.log_scale.data.clone() if optimize_scale else self.log_scale.clone()
        self.render_iou_loss_fn = render_iou_loss_fn
        self.render_depth_loss_fn = render_depth_loss_fn
        self.depth_map = None
        self.depth_valid = None
        if depth_map is not None:
            depth_np = np.asarray(depth_map, dtype=np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np[..., 2]
            valid_np = np.isfinite(depth_np) & (depth_np > DEPTH_RESIDUAL_MIN_DEPTH)
            if depth_mask is not None:
                valid_np = valid_np & (np.asarray(depth_mask) > 0)
            self.depth_map = torch.tensor(depth_np, dtype=torch.float32, device=self.device)
            self.depth_valid = torch.tensor(valid_np, dtype=torch.bool, device=self.device)
        self.desired_log_scale = self._compute_fixed_desired_log_scale()

    def _compute_fixed_desired_log_scale(self):
        src_extent = torch.amax(self.source, dim=0) - torch.amin(self.source, dim=0)
        tgt_extent = torch.amax(self.target, dim=0) - torch.amin(self.target, dim=0)
        ratios = tgt_extent / torch.clamp(src_extent, min=1e-6)
        desired_scale = ratios if self.optimize_independent_scale else torch.min(ratios)
        print("desired scale:", desired_scale.detach().cpu().numpy())
        return torch.log(torch.clamp(desired_scale, min=1e-6)).detach()

    def transform_source(self):
        # centered = self.source - self.source_center
        scale = torch.exp(self.log_scale)
        r = quaternion_to_matrix_torch(self.quat) @ torch.diag(scale.expand(3))
        transformed = (self.source) @ r.t() + self.trans
        return transformed

    def _visible_source_points(self, src):
        if not VISIBLE_SOURCE_CHAMFER:
            return src
        if src.shape[0] <= VISIBLE_MIN_SOURCE_POINTS:
            return src

        visible_idx = None
        with torch.no_grad():
            z = src[:, 2]
            valid = torch.isfinite(src).all(dim=1) & (z > VISIBLE_MIN_DEPTH)

            if self.intrinsics is not None:
                k = torch.as_tensor(self.intrinsics, dtype=src.dtype, device=src.device)
                if k.shape == (3, 3):
                    fx, fy = k[0, 0], k[1, 1]
                    cx, cy = k[0, 2], k[1, 2]
                    if torch.max(torch.abs(k[:2, :3])) <= 2.0:
                        fx = fx * VISIBLE_ZBUFFER_WIDTH
                        fy = fy * VISIBLE_ZBUFFER_HEIGHT
                        cx = cx * VISIBLE_ZBUFFER_WIDTH
                        cy = cy * VISIBLE_ZBUFFER_HEIGHT

                    u = torch.round(fx * src[:, 0] / z.clamp_min(1e-6) + cx).long()
                    v = torch.round(fy * src[:, 1] / z.clamp_min(1e-6) + cy).long()
                    valid = (
                        valid
                        & (u >= 0)
                        & (u < VISIBLE_ZBUFFER_WIDTH)
                        & (v >= 0)
                        & (v < VISIBLE_ZBUFFER_HEIGHT)
                    )

                    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
                    if valid_idx.numel() >= VISIBLE_MIN_SOURCE_POINTS:
                        linear = v[valid_idx] * VISIBLE_ZBUFFER_WIDTH + u[valid_idx]
                        z_valid = z[valid_idx]
                        zbuf = torch.full(
                            (VISIBLE_ZBUFFER_WIDTH * VISIBLE_ZBUFFER_HEIGHT,),
                            float("inf"),
                            dtype=src.dtype,
                            device=src.device,
                        )
                        zbuf.scatter_reduce_(0, linear, z_valid, reduce="amin", include_self=True)
                        visible = z_valid <= (zbuf[linear] + VISIBLE_DEPTH_TOLERANCE)
                        candidate_idx = valid_idx[visible]
                        if candidate_idx.numel() >= VISIBLE_MIN_SOURCE_POINTS:
                            if candidate_idx.numel() > VISIBLE_MAX_SOURCE_POINTS:
                                perm = torch.randperm(candidate_idx.numel(), device=src.device)[:VISIBLE_MAX_SOURCE_POINTS]
                                candidate_idx = candidate_idx[perm]
                            visible_idx = candidate_idx

            if visible_idx is None:
                front_idx = torch.nonzero(valid, as_tuple=False).flatten()
                if front_idx.numel() >= VISIBLE_MIN_SOURCE_POINTS:
                    if front_idx.numel() > VISIBLE_MAX_SOURCE_POINTS:
                        perm = torch.randperm(front_idx.numel(), device=src.device)[:VISIBLE_MAX_SOURCE_POINTS]
                        front_idx = front_idx[perm]
                    visible_idx = front_idx

        if visible_idx is not None:
            return src[visible_idx]
        return src

    def one_sided_trimmed_cd(self, batch_size=4096):
        src = self.transform_source()
        # src = self._visible_source_points(src)
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

    def trimmed_cd(self, batch_size=4096):
        """Two-sided trimmed Chamfer distance (symmetric).

        Computes target->source and source->target minimal distances in batches,
        applies trimming controlled by `self.trim_ratio` to each side, and
        returns the mean of the two trimmed means.
        """
        src = self.transform_source()
        # src = self._visible_source_points(src)

        # target -> source
        all_min_dists_t2s = []
        for i in range(0, len(self.target), batch_size):
            tgt_batch = self.target[i : i + batch_size]
            d = torch.cdist(tgt_batch, src)
            min_d, _ = torch.min(d, dim=1)
            all_min_dists_t2s.append(min_d)
        if len(all_min_dists_t2s) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        min_dists_t2s = torch.cat(all_min_dists_t2s, dim=0)

        # source -> target
        all_min_dists_s2t = []
        for i in range(0, len(src), batch_size):
            src_batch = src[i : i + batch_size]
            d = torch.cdist(src_batch, self.target)
            min_d, _ = torch.min(d, dim=1)
            all_min_dists_s2t.append(min_d)
        if len(all_min_dists_s2t) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        min_dists_s2t = torch.cat(all_min_dists_s2t, dim=0)

        # trimming (apply independently to each side)
        if 0 < self.trim_ratio < 1.0 and len(min_dists_t2s) > 10:
            k_t = max(1, int(len(min_dists_t2s) * self.trim_ratio /2))
            min_dists_t2s, _ = torch.topk(min_dists_t2s, k=k_t, largest=False)
        if 0 < self.trim_ratio < 1.0 and len(min_dists_s2t) > 10:
            k_s = max(1, int(len(min_dists_s2t) * self.trim_ratio /2))
            min_dists_s2t, _ = torch.topk(min_dists_s2t, k=k_s, largest=False)

        # guard empty
        if min_dists_t2s.numel() == 0 or min_dists_s2t.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return 0.5 * (min_dists_t2s.mean() + min_dists_s2t.mean())
    def main_axis_loss(self):
        if self.K is None:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        rot = quaternion_to_matrix_torch(self.quat)
        rot_world = self.K[:3, :3] @ rot
        z_world = torch.tensor([0.0, 0.0, 1.0], device=rot_world.device, dtype=rot_world.dtype)
        dots = rot_world.t() @ z_world                       # (3,)

        # 只关心“平行”，不区分 +z / -z
        loss = 1.0 - torch.max(dots ** 2)
        return loss 
    def regularization_loss(self):
        quat_reg = 0.001 * torch.sum((self.quat - self.init_quat) ** 2)
        trans_reg = 0.001 * torch.sum((self.trans - self.init_trans) ** 2)

        if self.optimize_scale:
            scale_reg = 0.001 * torch.sum((self.log_scale - self.init_log_scale) ** 2)
        else:
            scale_reg = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return quat_reg + trans_reg + scale_reg

    # def depth_scale_constraint_loss(self):
    #     if not self.optimize_scale:
    #         return torch.tensor(0.0, dtype=torch.float32, device=self.device)


    #     delta_depth = self.trans[2] - self.init_trans[2]
    #     delta_log_scale = torch.mean(self.log_scale - self.init_log_scale)
    #     same_direction = torch.relu(-delta_depth * delta_log_scale)
    #     return 200.0 * same_direction

    def depth_scale_constraint_loss(self):
        if not self.optimize_scale:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        eps = torch.tensor(1e-6, dtype=torch.float32, device=self.device)

        init_depth = torch.clamp(self.init_trans[2], min=eps)
        cur_depth = torch.clamp(self.trans[2], min=eps)

        depth_ratio = cur_depth / init_depth
        target_delta_log_scale = torch.log(depth_ratio)

        delta_log_scale = torch.mean(self.log_scale - self.init_log_scale)

        ratio_loss = (delta_log_scale - target_delta_log_scale) ** 2

        # 可选：仍然保留方向约束，防止优化早期跑反方向
        direction_loss = torch.relu(-delta_log_scale * (cur_depth - init_depth)) ** 2

        return 0.01 * ratio_loss + 200 * direction_loss

    def bbox_scale_regularization_loss(self):
        if not self.optimize_scale:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return 0.02 * torch.sum((self.log_scale - self.desired_log_scale) ** 2)

    def scale_range_loss(self):
        if not self.optimize_scale:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        delta = self.log_scale - self.init_log_scale
        lo = torch.log(torch.tensor(self.scale_range_min_ratio, dtype=torch.float32, device=self.device))
        hi = torch.log(torch.tensor(self.scale_range_max_ratio, dtype=torch.float32, device=self.device))
        loss = torch.relu(lo - delta) ** 2 + torch.relu(delta - hi) ** 2
        return self.scale_range_weight * torch.sum(loss)

    

    def depth_residual_loss(self):
        if not DEPTH_RESIDUAL_LOSS:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if self.render_depth_loss_fn is not None:
            return self.render_depth_loss_fn(self.quat, self.trans, self.log_scale)
        if self.depth_map is None or self.intrinsics is None:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        src = self._visible_source_points(self.transform_source())
        if src.shape[0] < DEPTH_RESIDUAL_MIN_POINTS:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        h, w = self.depth_map.shape
        idx = None
        with torch.no_grad():
            z = src[:, 2]
            k = torch.as_tensor(self.intrinsics, dtype=src.dtype, device=src.device)
            if k.shape == (3, 3):
                fx, fy = k[0, 0], k[1, 1]
                cx, cy = k[0, 2], k[1, 2]
                if torch.max(torch.abs(k[:2, :3])) <= 2.0:
                    fx = fx * w
                    fy = fy * h
                    cx = cx * w
                    cy = cy * h

                u = torch.round(fx * src[:, 0] / z.clamp_min(1e-6) + cx).long()
                v = torch.round(fy * src[:, 1] / z.clamp_min(1e-6) + cy).long()
                valid = (
                    torch.isfinite(src).all(dim=1)
                    & (z > DEPTH_RESIDUAL_MIN_DEPTH)
                    & (u >= 0)
                    & (u < w)
                    & (v >= 0)
                    & (v < h)
                )
                if valid.any():
                    valid = valid & self.depth_valid[v.clamp(0, h - 1), u.clamp(0, w - 1)]
                idx = torch.nonzero(valid, as_tuple=False).flatten()
                if idx.numel() > DEPTH_RESIDUAL_MAX_POINTS:
                    perm = torch.randperm(idx.numel(), device=src.device)[:DEPTH_RESIDUAL_MAX_POINTS]
                    idx = idx[perm]

        if idx is None or idx.numel() < DEPTH_RESIDUAL_MIN_POINTS:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        pred_z = src[idx, 2]
        gt_z = self.depth_map[v[idx], u[idx]]
        residual = pred_z - gt_z
        robust = torch.sqrt(residual ** 2 + DEPTH_RESIDUAL_EPS ** 2)
        return DEPTH_RESIDUAL_WEIGHT * robust.mean()

    def compute_loss(self, iou_loss_weight=0.0):
        cd = self.trimmed_cd()
        reg = self.regularization_loss()
        axis = self.main_axis_loss()
        depth_scale = self.depth_scale_constraint_loss()
        bbox_scale = self.bbox_scale_regularization_loss()
        scale_range = self.scale_range_loss()
        depth_residual = self.depth_residual_loss()
        iou_loss = None
        iou_score = None
        if self.render_iou_loss_fn is not None and iou_loss_weight > 0.0:
            iou_out = self.render_iou_loss_fn(self.quat, self.trans, self.log_scale)
            if isinstance(iou_out, (tuple, list)):
                iou_loss_raw, iou_score = iou_out
            else:
                iou_loss_raw = iou_out
            iou_loss = iou_loss_weight * iou_loss_raw
        loss = cd + reg + depth_scale  + axis + bbox_scale + depth_residual
        if iou_loss is not None:
            loss = loss + iou_loss
        return loss, cd, axis, depth_scale, bbox_scale, scale_range, depth_residual, iou_loss, iou_score

    def optimize(
        self,
        num_iterations=300,
        lr=0.01,
        render_iou_every=20,
        iou_loss_weight=0.03,
    ):
        # 前半程锁定 quat，仅优化平移/尺度；后半程再解锁 quat
        quat_group = {"name": "quat", "params": [self.quat], "lr": 0}
        trans_group = {"name": "trans", "params": [self.trans], "lr": lr * 5.0}
        params = [quat_group, trans_group]
        if self.optimize_scale:
            params.insert(0, {"name": "scale", "params": [self.log_scale], "lr": lr * 5.0})

        optimizer = torch.optim.Adam(params)
        step_size = num_iterations //3
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        unlock_iter = max(1, num_iterations // 2)
        # unlock_iter = 0

        history = {"loss": [], "cd": [], "scale": []}

        for it in range(num_iterations):
            if it == unlock_iter:
                for group in optimizer.param_groups:
                    if group.get("name") == "quat":
                        group["lr"] = lr
                        print(f"unlock quat at iter {it + 1}, quat lr={group['lr']}")

            iou_weight = iou_loss_weight if self.render_iou_loss_fn is not None else 0.0

            optimizer.zero_grad()
            loss, cd, axis, depth_scale, bbox_scale, scale_range, depth_residual, iou_loss, iou_score = self.compute_loss(
                iou_loss_weight=iou_weight,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.quat[:] = self.quat / (torch.norm(self.quat) + 1e-12)

            scale_np = torch.exp(self.log_scale).detach().cpu().numpy()
            if np.ndim(scale_np) == 0:
                scale_history = float(scale_np)
                scale_text = f"{scale_history:.5f}"
            else:
                scale_history = scale_np.astype(float).tolist()
                scale_text = "[" + ", ".join(f"{v:.5f}" for v in scale_history) + "]"
            cd_val = float(cd.detach().cpu().item())
            axis_val = float(axis.detach().cpu().item())
            depth_scale_val = float(depth_scale.detach().cpu().item())
            bbox_scale_val = float(bbox_scale.detach().cpu().item())
            scale_range_val = float(scale_range.detach().cpu().item())
            depth_residual_val = float(depth_residual.detach().cpu().item())
            loss_val = float(loss.detach().cpu().item())

            history["loss"].append(loss_val)
            history["cd"].append(cd_val)
            history["scale"].append(scale_history)

            iou_val = None
            if iou_score is not None and render_iou_every is not None:
                if (it + 1) % render_iou_every == 0:
                    iou_val = float(iou_score.detach().cpu().item())

            if (it + 1) % 20 == 0:
                msg = (
                    f"iter {it + 1:04d} | "
                    f"loss={loss_val:.6f} | cd={cd_val:.6f} | axis={axis_val:.6f} | "
                    f"depth_scale={depth_scale_val:.6f} | bbox_scale={bbox_scale_val:.6f} | "
                    f"scale_range={scale_range_val:.6f} | depth_res={depth_residual_val:.6f} | "
                    f"scale={scale_text}"
                )
                if iou_val is not None:
                    msg += f" | iou={iou_val:.4f}"
                if iou_loss is not None:
                    msg += f" | iou_loss={iou_loss.item():.6f} | iou_w={iou_weight:.4f}"
                print(msg)

        return history

    def get_delta_pose(self):
        with torch.no_grad():
            scale = torch.exp(self.log_scale).cpu().numpy()
            if np.ndim(scale) == 0:
                scale = np.array([float(scale), float(scale), float(scale)], dtype=np.float32)
            else:
                scale = scale.astype(np.float32).reshape(3)
            quat = (self.quat / (torch.norm(self.quat) + 1e-12)).cpu().numpy()
            trans = self.trans.cpu().numpy()

        return {
            "scale": scale,
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
    optimize_independent_scale=False,
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
    intrinsics = None,
    render_iou=True,
    render_iou_every=20,
    render_iou_device=None,
    iou_loss=False,
    iou_loss_weight=0.5,
    scale_range_min_ratio=0.5,
    scale_range_max_ratio=2,
    scale_range_weight=0.0,
    seed =1,
):  
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    #copy glb: open3d.cpu.pybind.geometry.TriangleMesh
    raw_glb = copy.deepcopy(glb_mesh)
    init_quat_wxyz, init_translation, scale = transform2pose(base_transform)
    if optimize_scale:
        init_log_scale = np.log(scale if optimize_independent_scale else scale[0])
    else:
        init_log_scale = None

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
    print("initial scale:", scale if optimize_independent_scale else scale[0])
    print("source center:", src_center)
    print("target center:", tgt_center)
    print("center distance before opt:", np.linalg.norm(src_center - tgt_center))

    optimizer = delta_pose_optimizer(
        source_points=source_points,
        target_points=target_points,
        device=device,
        optimize_scale=optimize_scale,
        optimize_independent_scale=optimize_independent_scale,
        trim_ratio=trim_ratio,
        init_quat=init_quat_wxyz,
        init_trans=init_translation,
        init_log_scale=init_log_scale,
        ex=K,
        intrinsics=intrinsics,
        render_iou_loss_fn=None,
        render_depth_loss_fn=None,
        depth_map=point_map,
        depth_mask=mask,
        scale_range_min_ratio=scale_range_min_ratio,
        scale_range_max_ratio=scale_range_max_ratio,
        scale_range_weight=scale_range_weight,
    )

    render_depth_loss_fn = None
    if DEPTH_RESIDUAL_LOSS:
        depth_np = np.asarray(point_map, dtype=np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 2]
        depth_device = optimizer.device
        depth_rasterizer, depth_t, depth_valid_t = _build_depth_rasterizer(
            depth_np,
            mask,
            depth_device,
            intrinsics,
        )
        depth_verts, depth_faces, _ = _build_p3d_mesh(glb_mesh, depth_device)

        def _depth_loss_fn(quat, trans, log_scale):
            return _render_depth_residual_loss(
                depth_verts,
                depth_faces,
                quat,
                trans,
                log_scale,
                depth_rasterizer,
                depth_t,
                depth_valid_t,
            )

        render_depth_loss_fn = _depth_loss_fn

    if render_depth_loss_fn is not None:
        optimizer.render_depth_loss_fn = render_depth_loss_fn

    render_iou_loss_fn = None
    if iou_loss:
        iou_device = (
            torch.device(render_iou_device)
            if render_iou_device is not None
            else optimizer.device
        )
        renderer, mask_t = _build_mask_renderer(mask, iou_device, intrinsics)
        verts, faces, textures = _build_p3d_mesh(glb_mesh, iou_device)

        def _iou_loss_fn(quat, trans, log_scale):
            return _render_soft_iou_loss(
                verts,
                faces,
                textures,
                quat,
                trans,
                log_scale,
                renderer,
                mask_t,
            )

        render_iou_loss_fn = _iou_loss_fn

    if render_iou_loss_fn is not None:
        optimizer.render_iou_loss_fn = render_iou_loss_fn

    history = optimizer.optimize(
        num_iterations=num_iterations,
        lr=lr,
        render_iou_every=render_iou_every if render_iou else None,
        iou_loss_weight=iou_loss_weight,
    )

    delta_pose = optimizer.get_delta_pose()
    # refined_mesh = apply_delta_to_mesh(glb_mesh, delta_pose)
    refined_source_points = sample_source_points_from_mesh(raw_glb, num_points=min(source_num_points, 10000))
    refined_center = refined_source_points.mean(axis=0)
    print("refined source center:", refined_center)
    print("center distance after opt:", np.linalg.norm(refined_center - tgt_center))
    print("estimated scale:", delta_pose["scale"] if optimize_independent_scale else delta_pose["scale"][0])
    print("estimated rotation (wxyz):", delta_pose["rotation_wxyz"])
    print("estimated translation:", delta_pose["translation"])
    
    return {
        "target_points": target_points,
        "delta_pose": delta_pose,
        "history": history,
        "refined_mesh": raw_glb,
    }





VISIBLE_SOURCE_CHAMFER = True
VISIBLE_ZBUFFER_HEIGHT = 512
VISIBLE_ZBUFFER_WIDTH = 512
VISIBLE_MIN_DEPTH = 1e-6
VISIBLE_DEPTH_TOLERANCE = 0.01
VISIBLE_MIN_SOURCE_POINTS = 256
VISIBLE_MAX_SOURCE_POINTS = 12000

DEPTH_RESIDUAL_LOSS = True
DEPTH_RESIDUAL_WEIGHT = 0.2
DEPTH_RESIDUAL_EPS = 0.01
DEPTH_RESIDUAL_MIN_DEPTH = 1e-6
DEPTH_RESIDUAL_MIN_POINTS = 256
DEPTH_RESIDUAL_MAX_POINTS = 12000


if __name__ == "__main__":
    pass
