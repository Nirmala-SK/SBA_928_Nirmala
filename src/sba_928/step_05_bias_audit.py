"""Step 5: audit the dataset and the models for gaps between groups."""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from statistics import mean

from sba_928.io_utils import read_jsonl, write_json
from sba_928.metrics import subgroup_metrics, tokenize
from sba_928.settings import BIAS_DIR, EVAL_DIR, TRAIN_FILE

LOGGER = logging.getLogger(__name__)
GROUP_FIELDS = ("category", "region", "rating")
LOADED_TERMS = frozenset(
    {"cheap", "ghetto", "exotic", "primitive", "tribal", "elderly", "youthful",
     "emotional", "aggressive"}
)


def audit_dataset(train: list[dict]) -> dict:
    """Check the training data for representation and wording imbalance."""
    total = len(train)
    shares = {}
    for field in ("region", "category"):
        counts = Counter(r[field] for r in train)
        shares[field] = {k: round(v / total, 3) for k, v in sorted(counts.items())}
    avg_len = {
        region: round(mean(len(tokenize(r["review"])) for r in train if r["region"] == region), 2)
        for region in sorted({r["region"] for r in train})
    }
    avg_rating = {
        region: round(mean(r["rating"] for r in train if r["region"] == region), 2)
        for region in sorted({r["region"] for r in train})
    }
    flagged = sorted(
        {
            w
            for r in train
            for w in tokenize(r["instruction"] + " " + r["target"] + " " + r["review"])
            if w in LOADED_TERMS
        }
    )
    smallest = min(shares["region"], key=shares["region"].get)
    return {
        "shares": shares,
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
    args = parser.parse_args()

    rows = read_jsonl(EVAL_DIR / "model_comparison.jsonl")
    table: list[dict] = []
    for model_key in ("base", "fine_tuned"):
        flat = [
            {"category": r["category"], "region": r["region"], "rating": r["rating"],
             **r[model_key]}
            for r in rows
        ]
        for field in GROUP_FIELDS:
            for entry in subgroup_metrics(flat, field, "token_f1", args.min_n):
                table.append({"model": model_key, **entry})

    BIAS_DIR.mkdir(parents=True, exist_ok=True)
    with (BIAS_DIR / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    tuned = [t for t in table if t["model"] == "fine_tuned" and not t["small_sample"]]
    largest = sorted(tuned, key=lambda t: abs(t["gap_vs_overall"]), reverse=True)[:3]
    write_json(
        BIAS_DIR / "bias_audit.json",
        {
            "dataset_audit": audit_dataset(read_jsonl(TRAIN_FILE)),
            "largest_fine_tuned_gaps": largest,
            "small_sample_groups": [
                f'{t["model"]}/{t["group_field"]}/{t["group"]} (n={t["n"]})'
                for t in table if t["small_sample"]
            ],
            "limitations": [
                "Groups are synthetic labels, not real demographic evidence.",
                "Small groups produce unstable means; do not over-read their gaps.",
                "A metric gap is an observation, not proof of unfairness or its cause.",
            ],
        },
    )
    LOGGER.info("Largest gaps: %s", [(g["group"], g["gap_vs_overall"]) for g in largest])


if __name__ == "__main__":
    main()