"""BIM prior -> metric depth for SAM3D: minimal inference port of the Area1 mixed BIM adapter.

MoGe-1 and SAM3D stay frozen. A small CNN (88,835 parameters) reads MoGe decoder features and the
BIM depth/normals rendered in the image's camera, and predicts

    D = D_MoGe * exp(log_scale + residual),   |residual| <= 0.3

Ported unchanged from bim_adapter/{model,geometry,prepare,area3}.py on codex/bim-pointmap-adapter.
Weights: weights/bim_adapter_mixed_e40.pt (Area1 real IFC + scan-fitted BIM, epoch 40).

BIM prior input: BIM depth in metres (camera Z, OpenCV camera), 0 or NaN where no BIM surface is hit,
rendered with the same K as the RGB image (any resolution; resized to 256x256 nearest).
Optional BIM normals (3xHxW or HxWx3, camera frame, facing the camera); if omitted they are
computed from the BIM depth. If no BIM pixel is valid the adapter falls back to MoGe depth.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ADAPTER_SIZE = 256
MOGE_NUM_TOKENS = 2500
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "bim_adapter_mixed_e40.pt"


class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class BIMAdapter(nn.Module):
    def __init__(self, local_residual=True, use_bim=True, width=32, residual_limit=.3, feature_channels=64):
        super().__init__()
        self.config = dict(local_residual=local_residual, use_bim=use_bim,
                           width=width, residual_limit=residual_limit, feature_channels=feature_channels)
        self.encoder = nn.Sequential(nn.Conv2d(feature_channels + 11, width, 3, padding=1), nn.GELU(),
                                     *[ResBlock(width) for _ in range(3)])
        self.visibility_head = nn.Conv2d(width, 1, 1)
        self.scale_head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        self.local = nn.Sequential(nn.Conv2d(width + 1, width, 3, padding=1), nn.GELU())
        self.residual_head = nn.Conv2d(width, 1, 1)

    def forward(self, batch):
        depth = batch["depth"].float()
        valid_depth = torch.isfinite(depth) & (depth > 0)
        safe_depth = torch.where(valid_depth, depth, 1.)
        bim = batch["bim_depth"].float()
        valid = torch.isfinite(bim) & (bim > 0)
        if not self.config["use_bim"]:
            valid = torch.zeros_like(valid)
        log_bim = torch.where(valid, bim.clamp_min(1e-4).log(), 0.)
        normal = torch.nan_to_num(batch["normal"].float())
        bim_normal = torch.where(valid.expand(-1, 3, -1, -1),
                                 torch.nan_to_num(batch["bim_normal"].float()), 0.)
        disagreement = torch.where(valid, log_bim - safe_depth.log(), 0.).clamp(-5, 5)
        condition = torch.cat((safe_depth.log() / 3., log_bim / 3., disagreement / 3.,
                               valid.float(), normal, bim_normal,
                               (normal * bim_normal).sum(1, keepdim=True).abs()), 1)
        feature = batch["features"].float()
        size = (max(4, depth.shape[-2] // 8), max(4, depth.shape[-1] // 8))
        x = self.encoder(torch.cat((F.interpolate(feature, size, mode="bilinear", align_corners=False),
                                    F.adaptive_avg_pool2d(condition, size)), 1))
        visibility_logits = self.visibility_head(x)
        pooling_valid = valid if self.config["use_bim"] else valid_depth
        weight = visibility_logits.sigmoid() * F.adaptive_avg_pool2d(pooling_valid.float(), size)
        weighted = (x * weight).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1e-6)
        descriptor = torch.cat((x.mean((2, 3)), weighted), 1)
        log_scale = 4. * torch.tanh(self.scale_head(descriptor).view(-1, 1, 1, 1) / 4.)
        gate = valid.flatten(1).any(1).view(-1, 1, 1, 1).float()
        if not self.config["use_bim"]:
            gate = torch.ones_like(gate)
        log_scale = log_scale * gate
        calibrated = torch.where(valid, log_bim - safe_depth.log() - log_scale.detach(), 0.)
        local_x = self.local(torch.cat((x, F.adaptive_avg_pool2d(calibrated.clamp(-5, 5) / 3., size)), 1))
        residual = self.config["residual_limit"] * torch.tanh(self.residual_head(local_x))
        residual = F.interpolate(residual, depth.shape[-2:], mode="bilinear", align_corners=False) * gate
        if not self.config["local_residual"]:
            residual = residual * 0.
        prediction = safe_depth * torch.exp(log_scale + residual)
        return {"depth": torch.where(valid_depth, prediction, 0.), "log_scale": log_scale,
                "residual": residual, "visibility_logits": visibility_logits}


def depth_to_points(depth, k):
    b, _, h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, device=depth.device),
                          torch.arange(w, device=depth.device), indexing="ij")
    pixels = torch.stack((u, v, torch.ones_like(u)), -1).to(depth.dtype)
    rays = torch.einsum("bij,hwj->bhwi", torch.linalg.inv(k.to(depth.dtype)), pixels)
    points = rays * depth[:, 0, :, :, None]
    valid = torch.isfinite(depth[:, 0]) & (depth[:, 0] > 0)
    return torch.where(valid[..., None], points, float("nan"))


def depth_normals(depth, k, edge_rtol=.1):
    points = depth_to_points(depth, k)
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    normal = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1, eps=1e-8)
    center = points[:, 1:-1, 1:-1]
    normal = torch.where((normal * center).sum(-1, keepdim=True) > 0, -normal, normal)
    z = depth[:, 0]
    neighbors = torch.stack((z[:, 1:-1, 2:], z[:, 1:-1, :-2],
                             z[:, 2:, 1:-1], z[:, :-2, 1:-1]), -1)
    zcenter = z[:, 1:-1, 1:-1, None]
    good = torch.isfinite(center).all(-1) & torch.isfinite(neighbors).all(-1)
    good &= (neighbors > 0).all(-1)
    good &= ((neighbors - zcenter).abs() <= edge_rtol * zcenter.clamp_min(.01)).all(-1)
    normal = torch.where(good[..., None], torch.nan_to_num(normal), 0.)
    return F.pad(normal.permute(0, 3, 1, 2), (1, 1, 1, 1))


def load_adapter(path=DEFAULT_WEIGHTS, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=True)
    model = BIMAdapter(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def _resize_nearest(array, size=ADAPTER_SIZE):
    return cv2.resize(array, (size, size), interpolation=cv2.INTER_NEAREST)


@torch.inference_mode()
def bim_corrected_depth(moge, adapter, rgb, K, bim_depth, bim_normal=None):
    """rgb: HxWx3 uint8; K: 3x3 in pixels of rgb; moge: MoGe-1 MoGeModel (e.g. SAM3D's depth model).

    Returns (HxW float32 metric depth with 0 = invalid, info dict).
    """
    device = next(adapter.parameters()).device
    rgb = np.asarray(rgb)[..., :3]
    h, w = rgb.shape[:2]
    K = np.asarray(K, np.float32)
    low_k = K.copy()
    low_k[0] *= ADAPTER_SIZE / w
    low_k[1] *= ADAPTER_SIZE / h

    captured = {}

    def hook(module, inputs, output):
        captured["features"] = F.adaptive_avg_pool2d(output.detach().float(), (16, 16))[:1].half().float()

    # MoGe Head.forward calls the children of each Sequential directly; hook the last called child.
    handle = moge.head.upsample_blocks[-1][-1].register_forward_hook(hook)
    try:
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().to(device) / 255.
        native = moge.infer(image, num_tokens=MOGE_NUM_TOKENS, force_projection=False)["depth"].float().cpu().numpy()
    finally:
        handle.remove()
    native[~np.isfinite(native) | (native <= 0)] = 0
    low = _resize_nearest(native)[None]
    normal = depth_normals(torch.from_numpy(low)[None], torch.from_numpy(low_k)[None])[0].numpy()

    bim = np.nan_to_num(np.asarray(bim_depth, np.float32).squeeze(), nan=0., posinf=0., neginf=0.)
    if bim.ndim != 2:
        raise ValueError(f"BIM depth must be HxW, got {bim.shape}")
    bim[bim < 0] = 0
    if bim.shape != (ADAPTER_SIZE, ADAPTER_SIZE):
        bim = _resize_nearest(bim)
    if bim_normal is None:
        bim_n = depth_normals(torch.from_numpy(bim)[None, None], torch.from_numpy(low_k)[None])[0].numpy()
    else:
        bim_n = np.nan_to_num(np.asarray(bim_normal, np.float32))
        if bim_n.ndim != 3 or 3 not in (bim_n.shape[0], bim_n.shape[-1]):
            raise ValueError(f"BIM normal must be 3xHxW or HxWx3, got {bim_n.shape}")
        if bim_n.shape[0] == 3 and bim_n.shape[-1] != 3:
            bim_n = bim_n.transpose(1, 2, 0)
        if bim_n.shape[:2] != (ADAPTER_SIZE, ADAPTER_SIZE):
            bim_n = _resize_nearest(bim_n)
        bim_n = np.ascontiguousarray(bim_n.transpose(2, 0, 1))

    batch = {"depth": torch.from_numpy(low)[None], "normal": torch.from_numpy(normal)[None],
             "features": captured["features"], "bim_depth": torch.from_numpy(bim)[None, None],
             "bim_normal": torch.from_numpy(bim_n)[None]}
    out = adapter({k: v.to(device).float() for k, v in batch.items()})
    log_field = F.interpolate(out["log_scale"] + out["residual"], (h, w), mode="bilinear", align_corners=False)
    native_t = torch.from_numpy(native)[None, None].to(device)
    corrected = torch.where(native_t > 0, native_t * log_field.exp(), 0.)[0, 0].cpu().numpy()
    info = {"log_scale": float(out["log_scale"].flatten()[0]), "bim_coverage": float((bim > 0).mean())}
    return corrected.astype(np.float32), info
