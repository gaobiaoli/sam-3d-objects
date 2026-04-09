# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask
from pathlib import Path


def _to_list_1d(value, expected_len: int):
	if isinstance(value, torch.Tensor):
		arr = value.detach().cpu().numpy()
	else:
		arr = np.asarray(value)
	arr = np.squeeze(arr)
	arr = arr.reshape(-1)
	if arr.size < expected_len:
		raise ValueError(f"Invalid pose field length: expected {expected_len}, got {arr.size}")
	return arr[:expected_len].astype(np.float32).tolist()

# load model
tag = "hf"
config_path = f"../autodl-tmp/checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_rgb.png")
mask = load_mask("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_rgb_mask_0.png", index=14)
out_json = Path("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_rgb_mask_0.json")
out_glb = Path("demo/camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain_rgb_mask_0.glb")
# run model
output = inference(image, mask, seed=42)


# save
pose_json = {
		"object_0": {
			"scale": _to_list_1d(output["scale"], 3),
			"translation": _to_list_1d(output["translation"], 3),
			"rotation": _to_list_1d(output["rotation"], 4),
		}
	}

out_json.write_text(json.dumps(pose_json, indent=2), encoding="utf-8")
output["glb"].export(str(out_glb))
