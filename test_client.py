import argparse
import json
from pathlib import Path

import numpy as np

from client import run_test


def load_camera_k_matrix(pose_json_path: Path) -> np.ndarray:
    pose_data = json.loads(pose_json_path.read_text(encoding="utf-8"))
    k_matrix = np.asarray(pose_data["camera_k_matrix"], dtype=np.float32)
    if k_matrix.shape != (3, 3):
        raise ValueError(f"camera_k_matrix must be 3x3, got {k_matrix.shape}")
    return k_matrix


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SAM3D client API")
    parser.add_argument("--url", default="http://127.0.0.1:6006/infer")
    parser.add_argument(
        "--image",
        default="demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_rgb.png",
    )
    parser.add_argument("--mask", default=None)
    parser.add_argument("--bbox", default="[100,120,300,360]", help='JSON bbox, e.g. "[100,120,300,360]"')
    parser.add_argument("--return-mask", action="store_true")
    parser.add_argument("--request-id", default="api_test")

    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument(
        "--depth",
        default="demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_depth.png",
    )
    parser.add_argument(
        "--pose-json",
        default="demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_pose.json",
    )
    parser.add_argument("--depth-scale", type=float, default=512.0)
    parser.add_argument("--invalid-depth-value", type=float, default=65535.0)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = build_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

    mask_path = None
    if args.mask is not None:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")

    bbox = None
    if args.bbox is not None:
        bbox = json.loads(args.bbox)

    if mask_path is None and bbox is None:
        raise ValueError("Either --mask or --bbox must be provided")

    depth_input = None
    k_matrix = None
    if args.use_depth:
        depth_path = Path(args.depth)
        pose_json_path = Path(args.pose_json)
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth file: {depth_path}")
        if not pose_json_path.exists():
            raise FileNotFoundError(f"Missing pose json file: {pose_json_path}")

        depth_input = depth_path
        k_matrix = load_camera_k_matrix(pose_json_path)

    run_test(
        url=args.url,
        image_path=str(image_path),
        mask_path=str(mask_path) if mask_path else None,
        bbox=bbox,
        depth_path=str(depth_input) if depth_input else None,
        K=k_matrix,
        request_id=args.request_id,
        depth_scale=args.depth_scale,
        invalid_depth_value=args.invalid_depth_value,
        return_mask=args.return_mask,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
