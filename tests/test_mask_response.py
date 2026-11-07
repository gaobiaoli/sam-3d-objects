from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import server


def test_infer_sam_returns_decodable_png(monkeypatch):
    # Exercise HTTP upload and response encoding without loading a GPU model.
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:12, 6:18] = True
    monkeypatch.setattr(server, "_generate_mask_from_bbox", lambda image, bbox: mask)
    image = BytesIO()
    Image.new("RGB", (30, 20), "white").save(image, format="PNG")

    client = TestClient(server.app, raise_server_exceptions=False)
    response = client.post(
        "/infer_sam",
        files={"image": ("image.png", image.getvalue(), "image/png")},
        data={"bbox": "[6, 4, 18, 12]"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert 'filename="mask.png"' in response.headers["content-disposition"]
    decoded = np.asarray(Image.open(BytesIO(response.content)))
    np.testing.assert_array_equal(decoded, mask.astype(np.uint8) * 255)
