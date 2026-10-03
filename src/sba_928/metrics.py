"""Dependency-free text metrics and subgroup aggregation."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from sba_928.schema import TARGET_LABELS

_LABEL_PATTERN = re.compile(
    r"Positive feature:\s*(.*?)\s*Primary problem:\s*(.*?)\s*Recommendation:\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)
_STOP = frozenset(
    {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "it", "its",
     "this", "that", "of", "to", "for", "in", "on", "with"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-token F1 between two strings."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def parse_fields(text: str) -> dict[str, str] | None:
    """Split a labelled answer into its three fields, or None if labels are missing."""
    match = _LABEL_PATTERN.search(text)
    if not match:
        return None
    keys = ("positive", "problem", "recommendation")
    return {k: v.strip() for k, v in zip(keys, match.groups(), strict=True)}


def format_adherence(prediction: str) -> float:
    """1.0 when all three labels appear once and in the required order."""
    positions = [prediction.find(label) for label in TARGET_LABELS]
    ok = all(p >= 0 for p in positions) and positions == sorted(positions)
    return float(ok)


def grounding(prediction: str, review: str) -> float:
    """Share of content words in the observed fields that occur in the review.

    The recommendation field is excluded because advice is not in the review.
    """
    fields = parse_fields(prediction)
    if fields is None:
        return 0.0
    review_tokens = set(tokenize(review))
    words = [
        w
        for part in (fields["positive"], fields["problem"])
        for w in tokenize(part)
        if w not in _STOP and w != "none" and w != "reported"
    ]
    if not words:
        return 1.0
    return sum(w in review_tokens for w in words) / len(words)


def score_example(prediction: str, target: str, review: str) -> dict[str, float]:
    """All per-example metrics."""
    pred_fields, ref_fields = parse_fields(prediction), parse_fields(target)
    if pred_fields and ref_fields:
        pos = token_f1(pred_fields["positive"], ref_fields["positive"])
        prob = token_f1(pred_fields["problem"], ref_fields["problem"])
        rec = token_f1(pred_fields["recommendation"], ref_fields["recommendation"])
    else:
        pos = prob = rec = 0.0
    return {
        "token_f1": token_f1(prediction, target),
        "positive_f1": pos,
        "problem_f1": prob,
        "recommendation_f1": rec,
        "format_adherence": format_adherence(prediction),
        "grounding": grounding(prediction, review),
        "exact_match": float(prediction.strip().lower() == target.strip().lower()),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every metric across rows."""
    if not rows:
        return {}
    return {key: round(mean(r[key] for r in rows), 4) for key in rows[0]}


def rating_band(rating: int) -> str:
    """Bucket a 1-5 rating."""
    return "low (1-2)" if rating <= 2 else "mid (3)" if rating == 3 else "high (4-5)"


def subgroup_metrics(
    rows: list[dict[str, Any]],
    group_field: str,
    metric: str = "token_f1",
    min_n: int = 10,
) -> list[dict[str, Any]]:
    """Mean metric per group with sample size, gap vs overall, and a small-n flag."""
    if not rows:
        return []
    overall = mean(r[metric] for r in rows)
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = rating_band(row[group_field]) if group_field == "rating" else str(row[group_field])
        buckets[key].append(row[metric])
    return [
        {
            "group_field": group_field,
            "group": name,
            "n": len(values),
            "metric": metric,
            "mean": round(mean(values), 4),
            "gap_vs_overall": round(mean(values) - overall, 4),
            "small_sample": len(values) < min_n,
        }
        for name, values in sorted(buckets.items())
    ]