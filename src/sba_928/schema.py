"""Record schema and validation for the market research dataset."""

from __future__ import annotations

from typing import Any

REQUIRED_STRING_FIELDS = (
    "review_id",
    "product",
    "category",
    "region",
    "review",
    "instruction",
    "context",
    "target",
    "split",
)
VALID_SPLITS = ("train", "validation", "test")
TARGET_LABELS = ("Positive feature:", "Primary problem:", "Recommendation:")


def validate_record(record: dict[str, Any]) -> None:
    """Raise ValueError when a record is malformed."""
    for field in REQUIRED_STRING_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{record.get('review_id', '?')}: bad field '{field}'")
    rating = record.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError(f"{record['review_id']}: rating must be an int from 1 to 5")
    if record["split"] not in VALID_SPLITS:
        raise ValueError(f"{record['review_id']}: unknown split '{record['split']}'")
    for label in TARGET_LABELS:
        if label not in record["target"]:
            raise ValueError(f"{record['review_id']}: target missing '{label}'")