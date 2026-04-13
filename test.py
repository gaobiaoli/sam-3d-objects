import base64
import json
from pathlib import Path
from typing import Any

import requests


ANN_JSON = "demo/camera_bbda26272dd346e595ed8e87a2d91020_office_3_frame_30_domain_ann.json"
URL = "http://127.0.0.1:6006/infer"
OUT_DIR = "demo/out_assets"
IOU_THRESHOLD = 0.6
IMAGE_ROOT = None
TIMEOUT = 600


def _resolve_image_path(ann_json_path: Path, image_path_value: str, image_root: str | None) -> Path:
	normalized = image_path_value.replace("\\", "/")
	candidates: list[Path] = []

	rel_path = Path(normalized)
	candidates.append(rel_path)
	candidates.append(ann_json_path.parent / rel_path)

	if image_root:
		candidates.append(Path(image_root) / rel_path)

	candidates.append(ann_json_path.parent / rel_path.name)

	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()

	joined = "\n".join(f"- {path}" for path in candidates)
	raise FileNotFoundError(f"Image not found. Tried:\n{joined}")


def _decode_pose_header(response: requests.Response) -> dict[str, Any]:
	pose_b64 = response.headers.get("X-Pose-Json-B64")
	if not pose_b64:
		raise ValueError("Missing response header: X-Pose-Json-B64")
	pose_text = base64.b64decode(pose_b64).decode("utf-8")
	return json.loads(pose_text)


def _infer_with_bbox(url: str, image_path: Path, bbox: list[float], request_id: str, timeout: int) -> requests.Response:
	with image_path.open("rb") as image_file:
		files = {
			"image": (image_path.name, image_file, "image/png"),
		}
		data = {
			"bbox": json.dumps([float(v) for v in bbox]),
			"request_id": request_id,
		}
		response = requests.post(url, files=files, data=data, timeout=timeout)
	return response


def main() -> None:
	ann_json_path = Path(ANN_JSON)
	if not ann_json_path.exists():
		raise FileNotFoundError(f"Annotation JSON not found: {ann_json_path}")

	raw = ann_json_path.read_text(encoding="utf-8")
	if not raw.strip():
		raise ValueError(
			f"Annotation JSON is empty: {ann_json_path}. "
			"Please set ANN_JSON to a valid non-empty annotation file."
		)
	ann_data = json.loads(raw)
	image_path = _resolve_image_path(
		ann_json_path=ann_json_path,
		image_path_value=ann_data["image_path"],
		image_root=IMAGE_ROOT,
	)

	image_stem = image_path.stem
	out_dir = Path(OUT_DIR)
	out_dir.mkdir(parents=True, exist_ok=True)

	annotations = ann_data.get("annotations", [])
	selected = [
		anno
		for anno in annotations
		if float(anno.get("bbox_iou_with_uncropped", 0.0)) > float(IOU_THRESHOLD)
	]

	if not selected:
		print("No annotations passed threshold, nothing to run.")
		return

	print(f"Using image: {image_path}")
	print(f"Selected {len(selected)}/{len(annotations)} annotations (iou > {IOU_THRESHOLD})")

	success_count = 0
	for anno in selected:
		instance_name = str(anno["instance_name"])
		bbox = anno["bbox_xyxy"]
		output_name = f"{image_stem}_{instance_name}"

		response = _infer_with_bbox(
			url=URL,
			image_path=image_path,
			bbox=bbox,
			request_id=output_name,
			timeout=TIMEOUT,
		)

		if response.status_code != 200:
			print(
				f"[FAILED] {instance_name} status={response.status_code} detail={response.text[:500]}"
			)
			continue

		glb_path = out_dir / f"{output_name}.glb"
		json_path = out_dir / f"{output_name}.json"

		glb_path.write_bytes(response.content)
		pose_obj = _decode_pose_header(response)
		json_path.write_text(json.dumps(pose_obj, indent=2, ensure_ascii=False), encoding="utf-8")

		success_count += 1
		print(f"[OK] {instance_name} -> {glb_path.name}, {json_path.name}")

	print(f"Done. Success {success_count}/{len(selected)}")


if __name__ == "__main__":
	main()
