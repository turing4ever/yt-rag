"""Evaluation and metrics for RAG quality."""

import uuid
from dataclasses import dataclass

from .db import Database
from .models import Feedback, TestCase
from .search import search


@dataclass
class EvalResult:
    """Result of running a single test case."""

    test_id: str
    query: str
    precision_at_k: float
    recall: float
    mrr: float  # Mean Reciprocal Rank
    keyword_match: float
    latency_ms: int
    passed: bool
    retrieved_video_ids: list[str]
    expected_video_ids: list[str]


@dataclass
class BenchmarkResult:
    """Result of running the full benchmark."""

    total_tests: int
    passed: int
    failed: int
    avg_precision: float
    avg_recall: float
    avg_mrr: float
    avg_keyword_match: float
    avg_latency_ms: float
    results: list[EvalResult]


def precision_at_k(retrieved: list[str], expected: list[str], k: int = 5) -> float:
    """Calculate precision at K.

    Args:
        retrieved: List of retrieved IDs (ordered by relevance)
        expected: List of expected relevant IDs
        k: Number of top results to consider

    Returns:
        Precision score between 0 and 1
    """
    if not expected:
        return 1.0 if not retrieved else 0.0

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    relevant = sum(1 for r in top_k if r in expected)
    return relevant / len(top_k)


def recall(retrieved: list[str], expected: list[str]) -> float:
    """Calculate recall.

    Args:
        retrieved: List of retrieved IDs
        expected: List of expected relevant IDs

    Returns:
        Recall score between 0 and 1
    """
    if not expected:
        return 1.0

    found = sum(1 for e in expected if e in retrieved)
    return found / len(expected)


def mean_reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """Calculate Mean Reciprocal Rank.

    Args:
        retrieved: List of retrieved IDs (ordered by relevance)
        expected: List of expected relevant IDs

    Returns:
        MRR score between 0 and 1
    """
    if not expected:
        return 1.0

    for i, r in enumerate(retrieved, 1):
        if r in expected:
            return 1.0 / i

    return 0.0


def keyword_match_score(answer: str | None, keywords: list[str] | None) -> float:
    """Calculate keyword match score.

    Args:
        answer: Generated answer text
        keywords: List of expected keywords

    Returns:
        Score between 0 and 1
    """
    if not keywords:
        return 1.0
    if not answer:
        return 0.0

    answer_lower = answer.lower()
    found = sum(1 for k in keywords if k.lower() in answer_lower)
    return found / len(keywords)


def run_test_case(
    test: TestCase,
    db: Database,
    top_k: int = 10,
) -> EvalResult:
    """Run a single test case.

    Args:
        test: Test case to run
        db: Database instance
        top_k: Number of results to retrieve

    Returns:
        EvalResult with metrics
    """
    # Run search
    result = search(
        query=test.query,
        db=db,
        top_k=top_k,
        video_id=test.scope_id if test.scope_type == "video" else None,
        channel_id=test.scope_id if test.scope_type == "channel" else None,
        generate_answer=bool(test.expected_keywords),
        log_query=False,
    )

    # Extract video IDs from results
    retrieved_video_ids = list(dict.fromkeys(h.video_id for h in result.hits))
    expected_video_ids = test.expected_video_ids or []

    # Calculate metrics
    p_at_k = precision_at_k(retrieved_video_ids, expected_video_ids, k=5)
    rec = recall(retrieved_video_ids, expected_video_ids)
    mrr = mean_reciprocal_rank(retrieved_video_ids, expected_video_ids)
    kw_match = keyword_match_score(result.answer, test.expected_keywords)

    # Determine pass/fail
    passed = True
    if expected_video_ids and p_at_k < 0.2:
        passed = False
    if test.expected_keywords and kw_match < 0.5:
        passed = False

    return EvalResult(
        test_id=test.id,
        query=test.query,
        precision_at_k=p_at_k,
        recall=rec,
        mrr=mrr,
        keyword_match=kw_match,
        latency_ms=result.latency_ms,
        passed=passed,
        retrieved_video_ids=retrieved_video_ids,
        expected_video_ids=expected_video_ids,
    )


def run_benchmark(
    db: Database | None = None,
    top_k: int = 10,
) -> BenchmarkResult:
    """Run all test cases.

    Args:
        db: Database instance
        top_k: Number of results to retrieve per test

    Returns:
        BenchmarkResult with aggregate metrics
    """
    if db is None:
        db = Database()
        db.init()

    tests = db.get_test_cases()
    if not tests:
        return BenchmarkResult(
            total_tests=0,
            passed=0,
            failed=0,
            avg_precision=0.0,
            avg_recall=0.0,
            avg_mrr=0.0,
            avg_keyword_match=0.0,
            avg_latency_ms=0.0,
            results=[],
        )

    results = []
    for test in tests:
        result = run_test_case(test, db, top_k)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    return BenchmarkResult(
        total_tests=len(results),
        passed=passed,
        failed=failed,
        avg_precision=sum(r.precision_at_k for r in results) / len(results),
        avg_recall=sum(r.recall for r in results) / len(results),
        avg_mrr=sum(r.mrr for r in results) / len(results),
        avg_keyword_match=sum(r.keyword_match for r in results) / len(results),
        avg_latency_ms=sum(r.latency_ms for r in results) / len(results),
        results=results,
    )


def add_test_case(
    query: str,
    expected_video_ids: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    reference_answer: str | None = None,
    scope_type: str | None = None,
    scope_id: str | None = None,
    db: Database | None = None,
) -> TestCase:
    """Add a new test case.

    Args:
        query: Test query
        expected_video_ids: Expected video IDs in results
        expected_keywords: Expected keywords in answer
        reference_answer: Reference answer for comparison
        scope_type: 'video' or 'channel' filter
        scope_id: ID for scope filter
        db: Database instance

    Returns:
        Created TestCase
    """
    if db is None:
        db = Database()
        db.init()

    test = TestCase(
        id=str(uuid.uuid4())[:8],
        query=query,
        scope_type=scope_type,
        scope_id=scope_id,
        expected_video_ids=expected_video_ids,
        expected_keywords=expected_keywords,
        reference_answer=reference_answer,
    )

    db.add_test_case(test)
    return test


def add_feedback(
    query_id: str,
    helpful: bool | None = None,
    source_rating: int | None = None,
    comment: str | None = None,
    db: Database | None = None,
) -> Feedback:
    """Add feedback for a query.

    Args:
        query_id: ID of the query to rate
        helpful: Whether the answer was helpful
        source_rating: Rating 1-5 for source quality
        comment: Optional comment
        db: Database instance

    Returns:
        Created Feedback
    """
    if db is None:
        db = Database()
        db.init()

    feedback = Feedback(
        query_id=query_id,
        helpful=helpful,
        source_rating=source_rating,
        comment=comment,
    )

    db.add_feedback(feedback)
    return feedback
