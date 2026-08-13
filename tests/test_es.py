from app.services.elasticsearch import ElasticsearchService


class FakeClient:
    def __init__(self, docs):
        self.docs = {d["_id"]: d for d in docs}

    def get(self, index, id, ignore=None):
        doc = self.docs.get(id)
        if doc is None:
            if ignore is not None:
                return {"found": False}
            raise KeyError(id)
        return {"found": True, "_source": doc["_source"]}

    def mget(self, index, body):
        docs = []
        for doc_id in body.get("ids", []):
            doc = self.docs.get(doc_id)
            docs.append({"found": bool(doc), "_source": doc["_source"] if doc else None})
        return {"docs": docs}

    def index(self, index, id, document):
        self.docs[id] = {"_id": id, "_source": document}

    def search(self, index, body=None, size=None):
        return {"hits": {"hits": [{"_score": 1.0, "_source": d["_source"]} for d in self.docs.values()]}}


def _svc(docs) -> ElasticsearchService:
    svc = ElasticsearchService()
    svc.client = FakeClient(docs)
    return svc


def test_parse_chunk_id():
    assert ElasticsearchService()._parse_chunk_id("paper_1_chunk_7") == ("paper_1_chunk_", 7)
    assert ElasticsearchService()._parse_chunk_id("nochunk") == (None, None)


def test_index_chunk_roundtrip():
    svc = _svc([])
    svc.index_chunk("p1_chunk_0", {"paper_id": "p1", "text": "hello"})
    assert svc.client.get(index="i", id="p1_chunk_0")["_source"] == {"paper_id": "p1", "text": "hello"}


def test_format_results_normalizes_fields():
    response = {
        "hits": {
            "hits": [
                {
                    "_score": 2.5,
                    "_source": {
                        "paper_id": "p1",
                        "title": "Notes",
                        "authors": ["A"],
                        "category": "C",
                        "chunk_id": "p1_chunk_0",
                        "page_number": 4,
                        "text": "body",
                        "source_url": "u",
                        "local_pdf": True,
                    },
                }
            ]
        }
    }
    results = ElasticsearchService()._format_results(response)
    assert results[0]["score"] == 2.5
    assert results[0]["page_number"] == 4
    assert results[0]["local_pdf"] is True
    assert results[0]["source_url"] == "u"


def _chunk_docs():
    return [
        {"_id": "p_chunk_0", "_source": {"chunk_id": "p_chunk_0", "text": "first chunk"}},
        {"_id": "p_chunk_1", "_source": {"chunk_id": "p_chunk_1", "text": "second chunk"}},
        {"_id": "p_chunk_2", "_source": {"chunk_id": "p_chunk_2", "text": "third chunk"}},
    ]


def test_get_chunk_window_orders_neighbors():
    svc = _svc(_chunk_docs())
    window = svc.get_chunk_window("p_chunk_1", window_size=1)
    assert window == "first chunk\n\nsecond chunk\n\nthird chunk"


def test_get_chunk_window_entries_returns_docs_with_pages():
    docs = [
        {"_id": "p_chunk_0", "_source": {"chunk_id": "p_chunk_0", "text": "a", "page_start": 1, "page_end": 1}},
        {"_id": "p_chunk_1", "_source": {"chunk_id": "p_chunk_1", "text": "b", "page_start": 2, "page_end": 2}},
        {"_id": "p_chunk_2", "_source": {"chunk_id": "p_chunk_2", "text": "c", "page_start": 4, "page_end": 5}},
    ]
    svc = _svc(docs)
    entries = svc.get_chunk_window_entries("p_chunk_1", window_size=1)
    assert [e["chunk_id"] for e in entries] == ["p_chunk_0", "p_chunk_1", "p_chunk_2"]
    assert entries[2]["page_end"] == 5


def test_get_chunk_window_entries_missing_id_returns_empty():
    svc = _svc(_chunk_docs())
    assert svc.get_chunk_window_entries("unknown_id", window_size=1) == []


def test_search_pages_dedupes_and_sorts_by_page():
    docs = [
        {"_id": "a_chunk_0", "_source": {"chunk_id": "a_chunk_0", "paper_id": "a", "page_start": 3, "text": "hello foo world"}},
        {"_id": "a_chunk_1", "_source": {"chunk_id": "a_chunk_1", "paper_id": "a", "page_start": 3, "text": "another foo"}},
        {"_id": "a_chunk_2", "_source": {"chunk_id": "a_chunk_2", "paper_id": "a", "page_start": 7, "text": "baz foo"}},
    ]
    svc = _svc(docs)
    results = svc.search_pages("a", "foo")
    assert [r["page"] for r in results] == [3, 7]
    assert results[0]["chunk_id"] == "a_chunk_0"
    assert "hello foo world" in results[0]["snippet"]


def test_get_chunk_window_missing_id_returns_empty():
    svc = _svc(_chunk_docs())
    assert svc.get_chunk_window("unknown_id", window_size=1) == ""


def test_get_chunk_window_caps_at_bounds():
    svc = _svc(_chunk_docs())
    window = svc.get_chunk_window("p_chunk_0", window_size=1)
    assert window == "first chunk\n\nsecond chunk"