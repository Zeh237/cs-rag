import pytest

from app.services.evaluation import (
    average_precision,
    bootstrap_ci,
    ndcg_at_k,
    paired_bootstrap_diff,
    precision_at_k,
    mean_average_precision,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_exact():
    ranks = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(ranks, relevant, 1) == 1.0
    assert precision_at_k(ranks, relevant, 2) == 0.5
    assert precision_at_k(ranks, relevant, 4) == 0.5
    assert precision_at_k(ranks, relevant, 0) == 0.0


def test_recall_at_k_exact():
    ranks = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    assert recall_at_k(ranks, relevant, 1) == 1 / 3
    assert recall_at_k(ranks, relevant, 2) == 1 / 3
    assert recall_at_k(ranks, relevant, 3) == 2 / 3
    assert recall_at_k(ranks, relevant, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
    assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0


def test_ndcg_at_k_perfect_rank_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    assert ndcg_at_k(["a", "b"], {"a"}, 1) == 1.0


def test_ndcg_at_k_relevant_late_is_less_than_one():
    ndcg = ndcg_at_k(["x", "b", "a"], {"a", "b"}, 3)
    assert 0.0 < ndcg < 1.0


def test_average_precision():
    expected = (1 / 1 + 2 / 3) / 2
    assert average_precision(["a", "x", "b"], {"a", "b"}) == expected


def test_mean_average_precision_means():
    rankings = [["a", "b"], ["z", "y"]]
    relevance = [{"a"}, {"z"}]
    assert mean_average_precision(rankings, relevance) == pytest.approx(1.0)


class TestBootstrap:
    def test_ci_bounds(self):
        values = [0.5, 0.4, 0.6, 0.55, 0.45, 0.5, 0.48, 0.52]
        mean, low, high = bootstrap_ci(values, n_resamples=200, seed=1)
        assert low <= mean <= high
        assert 0.0 < mean < 1.0

    def test_ci_deterministic_for_same_seed(self):
        values = [0.5, 0.4, 0.6]
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_empty_input(self):
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


class TestPairedBootstrap:
    def test_significant_positive_difference(self):
        a = [0.9] * 20
        b = [0.2] * 20
        mean, low, high = paired_bootstrap_diff(a, b, seed=3)
        assert mean > 0
        assert low > 0

    def test_ci_covers_zero_when_similar(self):
        values = [0.5 + (i % 3) * 0.01 for i in range(40)]
        mean, low, high = paired_bootstrap_diff(values, list(values), seed=4)
        assert mean == 0.0
        assert low <= 0.0 <= high


class TestDatasetHelpers:
    def test_sample_chunks_respects_count_and_seed(self):
        from scripts.build_eval_dataset import sample_chunks

        chunks = [
            {"chunk_id": f"p1_chunk_{i}", "paper_id": "p1", "text": "word " * 400}
            for i in range(4)
        ] + [
            {"chunk_id": f"p2_chunk_{i}", "paper_id": "p2", "text": "word " * 400}
            for i in range(4)
        ]
        first = sample_chunks(chunks, count=6, per_paper=2, min_len=300, seed=42)
        second = sample_chunks(chunks, count=6, per_paper=2, min_len=300, seed=42)
        assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
        assert len(first) <= 6

    def test_sample_excludes_noise_and_short_chunks(self):
        from scripts.build_eval_dataset import sample_chunks

        chunks = [
            {"chunk_id": "p1_chunk_0", "paper_id": "p1", "text": "short"},
            {"chunk_id": "p1_chunk_1", "paper_id": "p1", "text": "References\n" + "x " * 400},
            {"chunk_id": "p1_chunk_2", "paper_id": "p1", "text": "real content " * 40},
        ]
        selected = sample_chunks(chunks, count=10, per_paper=5, min_len=300, seed=1)
        assert [c["chunk_id"] for c in selected] == ["p1_chunk_2"]

    def test_neighboring_chunk_ids(self):
        from scripts.build_eval_dataset import neighboring_chunk_ids

        ids = ["p_chunk_0", "p_chunk_1", "p_chunk_2", "p_chunk_5"]
        assert neighboring_chunk_ids("p_chunk_1", ids) == ["p_chunk_0", "p_chunk_2"]
        assert neighboring_chunk_ids("p_chunk_5", ids) == []


class TestDatasetSchema:
    def test_valid_entry_passes_validation(self):
        from scripts.evaluate_retrieval import evaluate_query

        ranks = {"keyword": ["a", "b", "c"], "semantic": ["b", "a", "c"], "hybrid": ["a", "b", "c"], "hybrid+rerank": ["a", "b", "c"]}
        metrics = evaluate_query(ranks, set({"a"}), [1, 3])
        assert metrics["keyword"]["p@1"] == 1.0
        assert metrics["keyword"]["r@3"] == 1.0

    def test_evaluate_query_handles_empty_relevant(self):
        from scripts.evaluate_retrieval import evaluate_query

        ranks = {"keyword": ["a", "b"], "semantic": ["a", "b"], "hybrid": ["a", "b"], "hybrid+rerank": ["a", "b"]}
        metrics = evaluate_query(ranks, set(), [1, 3])
        assert metrics["keyword"]["map"] == 0.0