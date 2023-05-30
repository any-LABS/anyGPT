from fastapi.testclient import TestClient

from anyGPT.service.app import app, config

client = TestClient(app)
config.model = "pretrained_models/pre-trained-10M-char.pt"


def test_notfound():
    response = client.get("/")
    assert response.status_code == 404

def test_badrequest():
    response = client.post("/infer", json={"fake": "hello world"})
    assert response.status_code == 422

def test_infer():
    response = client.post("/infer", json={"data": "hello world", "temperature": 0.5, "top_k": 150, "max_new_tokens": 450})
    assert response.status_code == 200
    assert response.text != ""