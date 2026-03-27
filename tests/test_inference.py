import pytest
import base64
from PIL import Image
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.main import app
from backend.schemas.detection import DetectionResponse, BehaviorCount

client = TestClient(app)


def create_test_image() -> bytes:
    """Cria uma imagem verde 640x480 para testes."""
    img = Image.new("RGB", (640, 480), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


MOCK_RESPONSE = DetectionResponse(
    success=True,
    total_animals=2,
    behaviors=BehaviorCount(eating=1, standing=1),
    detections=[],
    inference_time_ms=120.0,
    model_used="roboflow",
)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "version" in data


@patch("backend.routers.inference.run_inference", new_callable=AsyncMock, return_value=MOCK_RESPONSE)
def test_detect_base64(mock_infer):
    img_bytes = create_test_image()
    img_b64 = base64.b64encode(img_bytes).decode()
    response = client.post("/api/v1/detect/base64", json={"image": img_b64})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "total_animals" in data
    assert "behaviors" in data
    assert "model_used" in data


@patch("backend.routers.inference.run_inference", new_callable=AsyncMock, return_value=MOCK_RESPONSE)
def test_detect_upload(mock_infer):
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_detect_base64_invalid():
    response = client.post("/api/v1/detect/base64", json={"image": "nao_e_base64_valido!!!"})
    assert response.status_code == 400


def test_detect_upload_too_large():
    # Imagem de 6MB (acima do limite de 5MB)
    large_content = b"x" * (6 * 1024 * 1024)
    response = client.post(
        "/api/v1/detect",
        files={"file": ("big.jpg", large_content, "image/jpeg")}
    )
    assert response.status_code == 413
