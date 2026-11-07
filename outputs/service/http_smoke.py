import base64
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import requests
import trimesh

root = Path('/home/bgao491/MySAM3d')
out = root / 'outputs/service/smoke-test'
out.mkdir(parents=True, exist_ok=True)
stem = 'camera_5c2959c3089c4ac5a7c5c913cc0df78d_office_5_frame_41_domain'
image = root / 'demo' / (stem + '_rgb.png')
depth = root / 'demo' / (stem + '_depth.png')
pose = json.loads((root / 'demo' / (stem + '_pose.json')).read_text())
bbox = [506, 478, 729, 750]
base = 'http://127.0.0.1:6006'
session = requests.Session()
session.trust_env = False

response = session.get(base + '/docs', timeout=15)
assert response.status_code == 200
spec = session.get(base + '/openapi.json', timeout=15)
spec.raise_for_status()
(out / 'openapi.json').write_text(json.dumps(spec.json(), indent=2))
print('HTTP docs and OpenAPI: OK', flush=True)

with image.open('rb') as image_file:
    response = session.post(
        base + '/infer_sam',
        files={'image': (image.name, image_file, 'image/png')},
        data={'bbox': json.dumps(bbox)}, timeout=180)
assert response.status_code == 200, response.text[:2000]
assert response.headers['content-type'] == 'image/png'
mask = np.asarray(Image.open(BytesIO(response.content)))
assert mask.shape == (1080, 1080) and np.any(mask > 0) and np.any(mask == 0)
(out / 'sam-mask.png').write_bytes(response.content)
print('Real SAM bbox segmentation: OK', flush=True)

started = time.perf_counter()
with image.open('rb') as image_file, depth.open('rb') as depth_file:
    response = session.post(
        base + '/infer',
        files={'image': (image.name, image_file, 'image/png'),
               'depth_image': (depth.name, depth_file, 'image/png')},
        data={'bbox': json.dumps(bbox), 'K': json.dumps(pose['camera_k_matrix']),
              'use_depth': 'true', 'depth_scale': '512',
              'return_mask': 'true', 'seed': '42',
              'request_id': 'local_service_smoke'}, timeout=600)
elapsed = time.perf_counter() - started
assert response.status_code == 200, response.text[:2000]
payload = response.json()
glb = base64.b64decode(payload['glb_b64'])
assert glb[:4] == b'glTF'
asset = out / 'local_service_smoke.glb'
asset.write_bytes(glb)
(out / 'local_service_smoke.json').write_text(json.dumps(payload['pose'], indent=2))
scene = trimesh.load(asset, force='scene')
assert len(scene.geometry) > 0
vertices = sum(len(mesh.vertices) for mesh in scene.geometry.values())
faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
assert vertices > 0 and faces > 0
for mesh in scene.geometry.values():
    assert np.isfinite(mesh.vertices).all()
for name, length in [('scale', 3), ('translation', 3), ('rotation', 4)]:
    values = payload['pose']['object_0'][name]
    assert len(values) == length and np.isfinite(values).all()
assert base64.b64decode(payload['mask_png_b64']).startswith(b'\x89PNG')
for kind in ('glb', 'json', 'mask'):
    downloaded = session.get(base + '/results/local_service_smoke/' + kind, timeout=30)
    assert downloaded.status_code == 200, downloaded.text[:500]
    if kind == 'glb':
        assert downloaded.content == glb
summary = {'base_url': base, 'http_status': response.status_code,
           'request': 'bbox + RGB + depth + intrinsics',
           'inference_seconds': round(elapsed, 2), 'glb_bytes': len(glb),
           'mesh_vertices': vertices, 'mesh_faces': faces,
           'sam_mask_pixels': int((mask > 0).sum()),
           'result_downloads': 'glb/json/mask all HTTP 200'}
(out / 'result.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
