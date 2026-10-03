"""Step 6: run the saved model on one new review."""

from __future__ import annotations

import argparse
import json
import logging

from sba_928.metrics import grounding
from sba_928.modeling import generate_from_model_path
from sba_928.prompts import INSTRUCTIONS, format_context, format_training_input
from sba_928.settings import DEFAULT_FINE_TUNED

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Format the input exactly as in training and print the prediction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_FINE_TUNED))
    parser.add_argument("--product", required=True)
    parser.add_argument("--rating", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--review", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    context = format_context(args.product, args.rating, args.review)
    prompt = format_training_input(INSTRUCTIONS[0], context)
    response = generate_from_model_path(args.model, [prompt], args.device)[0]
    print(
        json.dumps(
            {
                "model": args.model,
                "input": prompt,
                "response": response,
                "grounding_score": round(grounding(response, args.review), 3),
                "note": "Prediction only. Requires human review.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()