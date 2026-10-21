"""Dependency-free text metrics and subgroup aggregation."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from sba_928.schema import DEFAULT_TASK, TARGET_LABELS, TASK_TARGET_LABELS

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


def detect_task(text: str) -> str | None:
    """Find the task family whose three labels all appear in a (target) text."""
    lower = text.lower()
    for task, labels in TASK_TARGET_LABELS.items():
        if all(label.lower() in lower for label in labels):
            return task
    return None


def parse_labelled(text: str, labels: tuple[str, ...]) -> list[str] | None:
    """Split an answer into the values after each label, or None if a label is missing."""
    lower = text.lower()
    positions: list[int] = []
    start = 0
    for label in labels:
        pos = lower.find(label.lower(), start)
        if pos < 0:
            return None
        positions.append(pos)
        start = pos + len(label)
    values: list[str] = []
    for i, label in enumerate(labels):
        begin = positions[i] + len(label)
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        values.append(text[begin:end].strip())
    return values


def parse_fields(text: str) -> dict[str, str] | None:
    """Split a review-insight answer into its three fields, or None if labels are missing."""
    match = _LABEL_PATTERN.search(text)
    if not match:
        return None
    keys = ("positive", "problem", "recommendation")
    return {k: v.strip() for k, v in zip(keys, match.groups(), strict=True)}


def format_adherence(prediction: str, labels: tuple[str, ...] = TARGET_LABELS) -> float:
    """1.0 when all labels appear once and in the required order."""
    positions = [prediction.find(label) for label in labels]
    ok = all(p >= 0 for p in positions) and positions == sorted(positions)
    return float(ok)


def grounding(
    prediction: str, review: str, labels: tuple[str, ...] = TARGET_LABELS
) -> float:
    """Share of content words in the first two fields that occur in the source text.

    The last field (the recommendation or implication) is excluded because advice is
    not in the source. For the numeric task families this score is lower by
    construction, so compare it within a task rather than across tasks.
    """
    values = parse_labelled(prediction, labels)
    if values is None:
        return 0.0
    source_tokens = set(tokenize(review))
    words = [
        w
        for part in values[:2]
        for w in tokenize(part)
        if w not in _STOP and w != "none" and w != "reported"
    ]
    if not words:
        return 1.0
    return sum(w in source_tokens for w in words) / len(words)


def _normalize(value: str) -> str:
    return value.strip(" .").lower()


def score_example(prediction: str, target: str, review: str) -> dict[str, float]:
    """All per-example metrics.

    The task family is read from the target's labels. The three *_f1 columns score the
    first, second and third labelled field (for review_insight these are the positive
    feature, the problem and the recommendation).
    """
    task = detect_task(target) or DEFAULT_TASK
    labels = TASK_TARGET_LABELS[task]
    pred_values = parse_labelled(prediction, labels)
    ref_values = parse_labelled(target, labels)
    if pred_values is not None and ref_values is not None:
        first, second, third = (
            token_f1(p, r) for p, r in zip(pred_values, ref_values, strict=True)
        )
        key_match = float(_normalize(pred_values[0]) == _normalize(ref_values[0]))
    else:
        first = second = third = key_match = 0.0
    return {
        "token_f1": token_f1(prediction, target),
        "positive_f1": first,
        "problem_f1": second,
        "recommendation_f1": third,
        "format_adherence": format_adherence(prediction, labels),
        "grounding": grounding(prediction, review, labels),
        "exact_match": float(prediction.strip().lower() == target.strip().lower()),
        "key_field_match": key_match,
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