import pytest
from io import BytesIO
from PIL import Image
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = ["opacity"]
    return mock_model

@pytest.fixture
def mock_mlflow(mock_model):
    mock_mlflow_server = MagicMock()
    mock_pytorch = MagicMock()
    mock_pytorch.load_model.return_value = mock_model
    mock_mlflow_server.pytorch = mock_pytorch
    with patch("src.predict.mlflow", mock_mlflow_server):
        yield mock_mlflow_server

def test_get_model(mock_mlflow, mock_model):
    from src import predict
    predict._model = None
    model = predict.get_model()
    assert model == mock_model
    result = model.predict(None)
    assert result == ["opacity"]

def test_predict(mock_mlflow, mock_model):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient 
    from src.predict import retinal_router

    app = FastAPI()
    app.include_router(retinal_router)
    client = TestClient(app)
    # create image fake
    img = Image.new("RGB", (1, 1), color="white")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    buf.name = "test.jpg"
    response = client.post("/retinal/predict",
                        files={"file": ("test.jpg", buf, "image/jpeg")}
                        )
    assert response.status_code == 200
    print("response:", response.json())