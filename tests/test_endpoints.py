class TestSearchEndpoints:
    def test_keyword_search_returns_canned_results(self, client):
        r = client.get("/search/keyword", params={"q": "recursion", "limit": 3})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 2
        assert results[0]["title"] == "Recursion Notes"
        assert results[0]["chunk_id"] == "p1_chunk_0"
        assert results[0]["page_number"] == 3

    def test_semantic_search(self, client):
        r = client.get("/search/semantic", params={"q": "sorting"})
        assert r.status_code == 200
        assert r.json()["results"][0]["paper_id"] == "p1"


class TestChatEndpoints:
    def test_create_list_get_delete(self, client, auth_headers):
        created = client.post("/chats", json={"title": "Algorithms Qs"}, headers=auth_headers)
        assert created.status_code == 201
        chat_id = created.json()["id"]

        listed = client.get("/chats", headers=auth_headers)
        assert any(c["id"] == chat_id for c in listed.json())

        got = client.get(f"/chats/{chat_id}", headers=auth_headers)
        assert got.status_code == 200
        assert got.json()["title"] == "Algorithms Qs"
        assert got.json()["history"] == []

        deleted = client.delete(f"/chats/{chat_id}", headers=auth_headers)
        assert deleted.status_code == 204
        assert client.get(f"/chats/{chat_id}", headers=auth_headers).status_code == 404

    def test_cannot_access_another_users_chat(self, client):
        client.post("/auth/register", json={"username": "user_one", "password": "pw1"})
        client.post("/auth/register", json={"username": "user_two", "password": "pw2"})
        t1 = client.post("/auth/login", json={"username": "user_one", "password": "pw1"}).json()["access_token"]
        t2 = client.post("/auth/login", json={"username": "user_two", "password": "pw2"}).json()["access_token"]

        chat_id = client.post("/chats", json={"title": "secret"}, headers={"Authorization": f"Bearer {t1}"}).json()["id"]
        r = client.get(f"/chats/{chat_id}", headers={"Authorization": f"Bearer {t2}"})
        assert r.status_code == 404

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["elasticsearch"] is True