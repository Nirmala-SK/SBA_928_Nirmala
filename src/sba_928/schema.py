"""Record schema and validation for the market research dataset."""

from __future__ import annotations

from typing import Any

REQUIRED_STRING_FIELDS = (
    "review_id",
    "product",
    "category",
    "region",
    "instruction",
    "context",
    "target",
    "split",
)
VALID_SPLITS = ("train", "validation", "test")

# Kept under its old name so other files that import it keep working.
TARGET_LABELS = ("Positive feature:", "Primary problem:", "Recommendation:")

DEFAULT_TASK = "review_insight"

# The labels every target must contain, for each task family.
TASK_TARGET_LABELS: dict[str, tuple[str, ...]] = {
    "review_insight": TARGET_LABELS,
    "consumer_behavior": ("Buying driver:", "Pain point:", "Sentiment:"),
    "market_trend": ("Trend:", "Change:", "Implication:"),
    "competitor_analysis": ("Leader:", "Rating gap:", "Recommendation:"),
    "product_demand": ("Demand level:", "Reason:", "Recommendation:"),
    "customer_segmentation": ("Largest segment:", "Smallest segment:", "Recommendation:"),
}


def validate_record(record: dict[str, Any]) -> None:
    """Raise ValueError when a record is malformed."""
    task = record.get("task_type", DEFAULT_TASK)
    if task not in TASK_TARGET_LABELS:
        raise ValueError(f"{record.get('review_id', '?')}: unknown task_type '{task}'")
    required = REQUIRED_STRING_FIELDS
    if task == DEFAULT_TASK:
        required = required + ("review",)
    for field in required:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{record.get('review_id', '?')}: bad field '{field}'")
    rating = record.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError(f"{record['review_id']}: rating must be an int from 1 to 5")
    if record["split"] not in VALID_SPLITS:
        raise ValueError(f"{record['review_id']}: unknown split '{record['split']}'")
    for label in TASK_TARGET_LABELS[task]:
        if label not in record["target"]:
            raise ValueError(f"{record['review_id']}: target missing '{label}'")