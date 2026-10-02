"""
tests/test_auth.py
---------------------
Basic auth flow tests using FastAPI's TestClient.
Run with: pytest -v
NOTE: these hit your real configured database (see backend/config.py) -
use a separate test database in a real CI setup.
"""
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_register_and_login():
    import uuid
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    payload = {
        "username": username,
        "email": f"{username}@example.com",
        "password": "testpass123",
    }
    r1 = client.post("/auth/register", json=payload)
    assert r1.status_code == 201

    r2 = client.post("/auth/login", data={"username": username, "password": "testpass123"})
    assert r2.status_code == 200
    assert "access_token" in r2.json()


def test_login_wrong_password():
    r = client.post("/auth/login", data={"username": "nonexistent_user", "password": "wrong"})
    assert r.status_code == 401
