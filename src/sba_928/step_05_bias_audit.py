"""Step 5: audit the dataset and the models for gaps between groups."""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path
from statistics import mean

from sba_928.io_utils import read_jsonl, write_json
from sba_928.metrics import subgroup_metrics, tokenize
from sba_928.settings import BIAS_DIR, EVAL_DIR, TRAIN_FILE

LOGGER = logging.getLogger(__name__)
GROUP_FIELDS = ("category", "region", "rating", "task_type")
NO_REGION = "All regions"  # tasks that pool several regions carry no single region
LOADED_TERMS = frozenset(
    {"cheap", "ghetto", "exotic", "primitive", "tribal", "elderly", "youthful",
     "emotional", "aggressive"}
)


def task_of(record: dict) -> str:
    """Task family of a record (old review records have no task_type)."""
    return record.get("task_type", "review_insight")


def audit_dataset(train: list[dict]) -> dict:
    """Check the training data for representation and wording imbalance."""
    total = len(train)
    regional = [r for r in train if r["region"] != NO_REGION]
    reviews = [r for r in regional if task_of(r) == "review_insight"]
    shares = {
        "task_type": {
            k: round(v / total, 3)
            for k, v in sorted(Counter(task_of(r) for r in train).items())
        },
        "category": {
            k: round(v / total, 3)
            for k, v in sorted(Counter(r["category"] for r in train).items())
        },
        "region": {
            k: round(v / len(regional), 3)
            for k, v in sorted(Counter(r["region"] for r in regional).items())
        },
    }
    regions = sorted({r["region"] for r in reviews})
    # Review length and rating are compared on review_insight records only, so the
    # comparison is like for like (the other task families have no review text).
    avg_len = {
        g: round(mean(len(tokenize(r["review"])) for r in reviews if r["region"] == g), 2)
        for g in regions
    }
    avg_rating = {
        g: round(mean(r["rating"] for r in reviews if r["region"] == g), 2)
        for g in regions
    }
    flagged = sorted(
        {
            w
            for r in train
            for w in tokenize(
                r["instruction"] + " " + r["target"] + " " + (r["review"] or r["context"])
            )
            if w in LOADED_TERMS
        }
    )
    smallest = min(shares["region"], key=shares["region"].get)
    return {
        "shares": shares,
        "records_with_a_single_region": len(regional),
        "records_pooling_regions": total - len(regional),
        "review_insight_records_used_for_length_and_rating": len(reviews),
        "avg_review_length_by_region": avg_len,
        "avg_rating_by_region": avg_rating,
        "loaded_terms_found": flagged,
        "smallest_region": smallest,
        "smallest_region_share": shares["region"][smallest],
    }


def main() -> None:
    """Write subgroup metrics for both models plus the dataset audit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--eval-dir", default=str(EVAL_DIR))
    parser.add_argument("--output-dir", default=str(BIAS_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    rows = read_jsonl(Path(args.eval_dir) / "model_comparison.jsonl")
    table: list[dict] = []
    for model_key in ("base", "fine_tuned"):
        flat = [
            {"category": r["category"], "region": r["region"], "rating": r["rating"],
             "task_type": r["task_type"], **r[model_key]}
            for r in rows
        ]
        for field in GROUP_FIELDS:
            subset = flat
            if field == "region":
                # Pooled-region records say nothing about any one region.
                subset = [r for r in flat if r["region"] != NO_REGION]
            for entry in subgroup_metrics(subset, field, "token_f1", args.min_n):
                table.append({"model": model_key, **entry})

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    # Gaps between task families are expected (the tasks differ), so they are kept
    # out of the fairness ranking and reported separately.
    tuned = [
        t for t in table
        if t["model"] == "fine_tuned" and not t["small_sample"] and t["group_field"] != "task_type"
    ]
    largest = sorted(tuned, key=lambda t: abs(t["gap_vs_overall"]), reverse=True)[:3]
    by_task = [t for t in table if t["model"] == "fine_tuned" and t["group_field"] == "task_type"]
    write_json(
        out_dir / "bias_audit.json",
        {
            "dataset_audit": audit_dataset(read_jsonl(TRAIN_FILE)),
            "largest_fine_tuned_gaps": largest,
            "fine_tuned_by_task_type": by_task,
            "small_sample_groups": [
                f'{t["model"]}/{t["group_field"]}/{t["group"]} (n={t["n"]})'
                for t in table if t["small_sample"]
            ],
            "limitations": [
                "Groups are synthetic labels, not real demographic evidence.",
                "Small groups produce unstable means; do not over-read their gaps.",
                "A metric gap is an observation, not proof of unfairness or its cause.",
                                (
                    "Region is only a real attribute for 3 of 6 task families; the others "
                    "pool all regions and are left out of the region comparison."
                ),
                "Gaps between task families reflect task difficulty, not group fairness.",
            ],
        },
    )
    LOGGER.info("Largest gaps: %s", [(g["group"], g["gap_vs_overall"]) for g in largest])
    for t in by_task:
        LOGGER.info("task %-22s n=%d token_f1=%.3f", t["group"], t["n"], t["mean"])


if __name__ == "__main__":
    main()