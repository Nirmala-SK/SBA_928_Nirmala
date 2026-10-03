"""Step 2: compare prompt variants on a pretrained model. No training happens here."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

from sba_928.io_utils import read_jsonl, write_json, write_jsonl
from sba_928.metrics import aggregate, score_example
from sba_928.modeling import generate_from_model_path
from sba_928.prompts import build_prompt_variants
from sba_928.settings import BASE_MODEL, PROMPT_DIR, VALIDATION_FILE

LOGGER = logging.getLogger(__name__)


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
    records = read_jsonl(args.data)[: args.limit]
    per_variant: dict[str, list[dict[str, float]]] = defaultdict(list)
    saved: list[dict[str, object]] = []

    names = list(build_prompt_variants("x"))
    for name in names:
        prompts = [build_prompt_variants(r["context"])[name] for r in records]
        responses = generate_from_model_path(args.model, prompts, args.device)
        for record, prompt, response in zip(records, prompts, responses, strict=True):
            metrics = score_example(response, record["target"], record["review"])
            per_variant[name].append(metrics)
            saved.append(
                {
                    "review_id": record["review_id"],
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
    }
    write_jsonl(PROMPT_DIR / "prompt_predictions.jsonl", saved)
    write_json(PROMPT_DIR / "prompt_summary.json", summary)
    for name, metrics in summary["variants"].items():
        LOGGER.info("%-11s %s", name, metrics)


if __name__ == "__main__":
    main()