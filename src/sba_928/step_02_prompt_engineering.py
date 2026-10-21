"""Step 2: compare prompt variants on a pretrained model. No training happens here."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any

from sba_928.io_utils import read_jsonl, write_json, write_jsonl
from sba_928.metrics import aggregate, score_example
from sba_928.modeling import generate_from_model_path
from sba_928.prompts import build_prompt_variants
from sba_928.settings import BASE_MODEL, PROMPT_DIR, VALIDATION_FILE

LOGGER = logging.getLogger(__name__)


def task_of(record: dict[str, Any]) -> str:
    """Task family of a record (old review records have no task_type)."""
    return record.get("task_type", "review_insight")


def balanced_sample(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Pick records round-robin across task families so every task is covered."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[task_of(record)].append(record)
    tasks = sorted(by_task)
    picked: list[dict[str, Any]] = []
    index = 0
    while len(picked) < limit and any(index < len(rows) for rows in by_task.values()):
        for task in tasks:
            if index < len(by_task[task]) and len(picked) < limit:
                picked.append(by_task[task][index])
        index += 1
    return picked


def main() -> None:
    """Run every prompt variant on the same validation records."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--data", default=str(VALIDATION_FILE))
    args = parser.parse_args()

    # Validation split is used so the test split stays untouched until step 4.
    records = balanced_sample(read_jsonl(args.data), args.limit)
    per_variant: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_variant_task: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    saved: list[dict[str, object]] = []

    names = list(build_prompt_variants("x"))
    for name in names:
        prompts = [
            build_prompt_variants(r["context"], task_of(r))[name] for r in records
        ]
        responses = generate_from_model_path(args.model, prompts, args.device)
        for record, prompt, response in zip(records, prompts, responses, strict=True):
            source = record["review"] or record["context"]
            metrics = score_example(response, record["target"], source)
            per_variant[name].append(metrics)
            per_variant_task[name][task_of(record)].append(metrics)
            saved.append(
                {
                    "review_id": record["review_id"],
                    "task_type": task_of(record),
                    "variant": name,
                    "prompt": prompt,
                    "response": response,
                    "target": record["target"],
                    **metrics,
                }
            )

    summary = {
        "model": args.model,
        "records": len(records),
        "note": "Prompt engineering only. No model parameters were changed.",
        "variants": {name: aggregate(rows) for name, rows in per_variant.items()},
        "by_task": {
            name: {task: aggregate(rows) for task, rows in tasks.items()}
            for name, tasks in per_variant_task.items()
        },
    }
    write_jsonl(PROMPT_DIR / "prompt_predictions.jsonl", saved)
    write_json(PROMPT_DIR / "prompt_summary.json", summary)
    for name, metrics in summary["variants"].items():
        LOGGER.info("%-11s %s", name, metrics)


if __name__ == "__main__":
    main()