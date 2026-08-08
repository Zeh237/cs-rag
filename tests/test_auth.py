import pytest

from app.services.auth import auth_service


class TestPasswordHashing:
    def test_round_trip(self):
        hashed = auth_service.hash_password("correct horse battery staple")
        assert hashed != "correct horse battery staple"
        assert auth_service.verify_password("correct horse battery staple", hashed)

    def test_wrong_password_rejected(self):
        hashed = auth_service.hash_password("right-password")
        assert not auth_service.verify_password("wrong-password", hashed)

    def test_hashes_are_salted_and_unique(self):
        h1 = auth_service.hash_password("same-password")
        h2 = auth_service.hash_password("same-password")
        assert h1 != h2


class TestAccessTokens:
    def test_create_and_decode(self):
        token = auth_service.create_access_token(42, "alice")
        payload = auth_service.decode_token(token)
        assert payload["sub"] == "42"
        assert payload["username"] == "alice"

    def test_decode_invalid_token_raises_401(self):
        with pytest.raises(Exception) as exc_info:
            auth_service.decode_token("not-a-real-token")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired token"


class TestAuthEndpoints:
    def test_register_login_me(self, client):
        r = client.post("/auth/register", json={"username": "bob", "password": "pw-1234"})
        assert r.status_code == 201
        assert r.json()["username"] == "bob"

        login = client.post("/auth/login", json={"username": "bob", "password": "pw-1234"})
        assert login.status_code == 200
        token = login.json()["access_token"]
        assert token

        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["username"] == "bob"

    def test_register_duplicate_username_conflict(self, client):
        payload = {"username": "carol", "password": "pw-5678"}
        assert client.post("/auth/register", json=payload).status_code == 201
        r = client.post("/auth/register", json=payload)
        assert r.status_code == 409
        assert "already taken" in r.json()["detail"]

    def test_login_wrong_password(self, client):
        client.post("/auth/register", json={"username": "dave", "password": "correct"})
        r = client.post("/auth/login", json={"username": "dave", "password": "wrong"})
        assert r.status_code == 401

    def test_me_requires_token(self, client):
        r = client.get("/auth/me")
        assert r.status_code in (401, 403)

    def test_me_rejects_invalid_token(self, client):
        r = client.get("/auth/me", headers={"Authorization": "Bearer tampered.token.value"})
        assert r.status_code in (401, 403)