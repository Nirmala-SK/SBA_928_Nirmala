"""Tests for the multi-task market research dataset."""

import re

from sba_928.metrics import score_example
from sba_928.schema import TASK_TARGET_LABELS, validate_record
from sba_928.step_01_generate_dataset import CATALOG, generate_records
from sba_928.tasks import build_dataset, split_by_task


def _dataset():
    return build_dataset(generate_records(600, 42), CATALOG, 100, 42)


def test_every_task_family_is_present():
    assert {r["task_type"] for r in _dataset()} == set(TASK_TARGET_LABELS)


def test_dataset_is_reproducible():
    assert _dataset() == _dataset()


def test_new_task_contexts_are_unique():
    seen = set()
    for r in _dataset():
        if r["task_type"] == "review_insight":
            continue
        key = (r["task_type"], r["context"])
        assert key not in seen
        seen.add(key)


def test_split_covers_all_tasks_without_overlap():
    parts = split_by_task(_dataset(), 42)
    id_sets = [{r["review_id"] for r in rows} for rows in parts.values()]
    assert len(set().union(*id_sets)) == sum(len(s) for s in id_sets) == 600
    for name, rows in parts.items():
        assert {r["task_type"] for r in rows} == set(TASK_TARGET_LABELS), name
        for record in rows:
            validate_record(record)


def test_segmentation_always_has_a_clear_largest_segment():
    for r in _dataset():
        if r["task_type"] == "customer_segmentation":
            counts = re.findall(r"Rated [\d-]+: (\d+)", r["context"])
            assert len(set(counts)) == 3


def test_perfect_prediction_scores_one_for_every_task():
    for r in _dataset():
        scores = score_example(r["target"], r["target"], r["review"] or r["context"])
        assert scores["exact_match"] == 1.0
        assert scores["format_adherence"] == 1.0
        assert scores["key_field_match"] == 1.0