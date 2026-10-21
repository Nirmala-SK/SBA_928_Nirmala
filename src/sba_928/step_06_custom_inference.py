"""Step 6: run the saved model on one new input (any of the six task families)."""

from __future__ import annotations

import argparse
import json
import logging

from sba_928.metrics import grounding
from sba_928.modeling import generate_from_model_path
from sba_928.prompts import INSTRUCTIONS, format_context, format_training_input
from sba_928.schema import TASK_TARGET_LABELS
from sba_928.settings import DEFAULT_FINE_TUNED
from sba_928.tasks import TASK_INSTRUCTIONS

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Format the input exactly as in training and print the prediction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_FINE_TUNED))
    parser.add_argument("--task", default="review_insight", choices=list(TASK_TARGET_LABELS))
    parser.add_argument("--product", help="review_insight only")
    parser.add_argument("--rating", type=int, choices=range(1, 6), help="review_insight only")
    parser.add_argument("--review", help="review_insight only")
    parser.add_argument("--context", help="all other tasks: the data text, as in training")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.task == "review_insight":
        if not (args.product and args.rating and args.review):
            parser.error("review_insight needs --product, --rating and --review")
        context = format_context(args.product, args.rating, args.review)
        instruction = INSTRUCTIONS[0]
        source = args.review
    else:
        if not args.context:
            parser.error(f"{args.task} needs --context")
        context = args.context
        instruction = TASK_INSTRUCTIONS[args.task][0]
        source = args.context

    prompt = format_training_input(instruction, context)
    response = generate_from_model_path(args.model, [prompt], args.device)[0]
    print(
        json.dumps(
            {
                "model": args.model,
                "task": args.task,
                "input": prompt,
                "response": response,
                "grounding_score": round(
                    grounding(response, source, TASK_TARGET_LABELS[args.task]), 3
                ),
                "note": "Prediction only. Requires human review.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()