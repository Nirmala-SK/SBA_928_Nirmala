"""Step 4: compare the base and fine-tuned models on the held-out test split."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from sba_928.io_utils import read_jsonl, write_json, write_jsonl
from sba_928.metrics import aggregate, score_example
from sba_928.modeling import generate_from_model_path
from sba_928.prompts import format_training_input
from sba_928.settings import BASE_MODEL, DEFAULT_FINE_TUNED, EVAL_DIR, TEST_FILE

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Send identical inputs and decoding settings to both models."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--fine-tuned-model", default=str(DEFAULT_FINE_TUNED))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(EVAL_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    records = read_jsonl(TEST_FILE)[: args.limit]
    prompts = [format_training_input(r["instruction"], r["context"]) for r in records]
    base_out = generate_from_model_path(args.base_model, prompts, args.device)
    tuned_out = generate_from_model_path(args.fine_tuned_model, prompts, args.device)

    rows, base_scores, tuned_scores = [], [], []
    base_by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    tuned_by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for rec, base, tuned in zip(records, base_out, tuned_out, strict=True):
        task = rec.get("task_type", "review_insight")
        source = rec["review"] or rec["context"]
        b = score_example(base, rec["target"], source)
        t = score_example(tuned, rec["target"], source)
        base_scores.append(b)
        tuned_scores.append(t)
        base_by_task[task].append(b)
        tuned_by_task[task].append(t)
        rows.append(
            {
                "review_id": rec["review_id"],
                "task_type": task,
                "category": rec["category"],
                "region": rec["region"],
                "rating": rec["rating"],
                "input": prompts[len(rows)],
                "target": rec["target"],
                "base_response": base,
                "fine_tuned_response": tuned,
                "base": b,
                "fine_tuned": t,
                "token_f1_change": round(t["token_f1"] - b["token_f1"], 4),
            }
        )

    base_agg, tuned_agg = aggregate(base_scores), aggregate(tuned_scores)
    by_task = {}
    for task in sorted(base_by_task):
        task_base = aggregate(base_by_task[task])
        task_tuned = aggregate(tuned_by_task[task])
        by_task[task] = {
            "records": len(base_by_task[task]),
            "base": task_base,
            "fine_tuned": task_tuned,
            "change": {k: round(task_tuned[k] - task_base[k], 4) for k in task_base},
        }
    summary = {
        "records": len(rows),
        "base_model": args.base_model,
        "fine_tuned_model": args.fine_tuned_model,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 96},
        "base": base_agg,
        "fine_tuned": tuned_agg,
        "change": {k: round(tuned_agg[k] - base_agg[k], 4) for k in base_agg},
        "by_task": by_task,
        "improved": sum(r["token_f1_change"] > 0 for r in rows),
        "unchanged": sum(r["token_f1_change"] == 0 for r in rows),
        "regressed": sum(r["token_f1_change"] < 0 for r in rows),
        "caveat": "Synthetic, templated data. Gains here may not transfer to real reviews.",
    }
    write_jsonl(out_dir / "model_comparison.jsonl", rows)
    write_json(out_dir / "model_comparison_summary.json", summary)
    LOGGER.info("base=%s", base_agg)
    LOGGER.info("tuned=%s", tuned_agg)
    for task, result in by_task.items():
        LOGGER.info(
            "%-22s token_f1 %.3f -> %.3f | exact_match %.3f -> %.3f | key_field %.3f -> %.3f",
            task,
            result["base"]["token_f1"],
            result["fine_tuned"]["token_f1"],
            result["base"]["exact_match"],
            result["fine_tuned"]["exact_match"],
            result["base"]["key_field_match"],
            result["fine_tuned"]["key_field_match"],
        )


if __name__ == "__main__":
    main()