"""Step 1: generate a synthetic market research review dataset and split it."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Any

from sba_928.io_utils import sha256_file, write_json, write_jsonl
from sba_928.prompts import INSTRUCTIONS, format_context
from sba_928.schema import validate_record
from sba_928.settings import (
    MANIFEST_FILE,
    RAW_CSV,
    SEED,
    TEST_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
)

LOGGER = logging.getLogger(__name__)

# product -> (category, positive features, [(problem clause, recommendation)])
CATALOG: dict[str, tuple[str, list[str], list[tuple[str, str]]]] = {
    "Reusable Water Bottle": (
        "drinkware",
        ["keeps drinks cold for hours", "feels sturdy in the hand"],
        [("the lid leaks", "Redesign the lid seal"),
         ("the cap is hard to unscrew", "Add a wider, grippier cap")],
    ),
    "Insulated Coffee Mug": (
        "drinkware",
        ["keeps coffee hot all morning", "fits in a car cup holder"],
        [("the paint chips after a month", "Use a more durable coating"),
         ("the lid is hard to clean", "Make the lid easier to disassemble")],
    ),
    "Wireless Mouse": (
        "electronics",
        ["is comfortable for long sessions", "has a quiet click"],
        [("it disconnects several times a day", "Improve the wireless connection stability"),
         ("the battery dies within two weeks", "Extend battery life or add recharging")],
    ),
    "Bluetooth Speaker": (
        "electronics",
        ["sounds clear at high volume", "is easy to pair"],
        [("the battery drains too fast", "Increase battery capacity"),
         ("the bass sounds muddy", "Retune the low-frequency audio")],
    ),
    "Running Shoes": (
        "apparel",
        ["feel light on long runs", "have great cushioning"],
        [("the sole wore down in two months", "Strengthen the outsole rubber"),
         ("they run a full size small", "Publish a more accurate size guide")],
    ),
    "Rain Jacket": (
        "apparel",
        ["blocks heavy rain", "packs down small"],
        [("the zipper keeps snagging", "Switch to a smoother zipper"),
         ("the inside feels sticky and hot", "Improve breathability")],
    ),
    "Organic Granola": (
        "food",
        ["tastes fresh and crunchy", "has a good amount of nuts"],
        [("it is too sweet", "Offer a reduced-sugar version"),
         ("the bag arrived half empty", "Improve package filling and sealing")],
    ),
    "Protein Bars": (
        "food",
        ["keep me full between meals", "come in great flavors"],
        [("the texture is chalky", "Improve the texture of the recipe"),
         ("the price is high for the size", "Review pricing against competitors")],
    ),
    "Face Moisturizer": (
        "beauty",
        ["absorbs quickly", "leaves skin soft"],
        [("the scent is overpowering", "Offer a fragrance-free option"),
         ("the pump stopped working", "Replace the pump mechanism")],
    ),
    "Yoga Mat": (
        "fitness",
        ["has excellent grip", "is thick and supportive"],
        [("it smells strongly of rubber", "Reduce odor before shipping"),
         ("the edges started to peel", "Improve the edge stitching and bonding")],
    ),
    "Smart Thermostat": (
        "home",
        ["saves on my energy bill", "has a clean app"],
        [("setup took over an hour", "Simplify the installation guide"),
         ("the schedule resets after power cuts", "Store schedules in local memory")],
    ),
    "Desk Lamp": (
        "home",
        ["gives bright, even light", "has a sleek design"],
        [("the switch feels loose", "Use a firmer switch"),
         ("the arm will not stay in position", "Strengthen the arm joint")],
    ),
}

# Regions are metadata only. They are NOT in the model input. Writing style varies by
# region so the bias audit has something real to measure in this synthetic data.
REGIONS = ("North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa")
REGION_WEIGHTS = (0.34, 0.26, 0.20, 0.12, 0.08)  # deliberately unbalanced
REGION_OPENERS = {
    "North America": ["", "Honestly, ", "Overall, "],
    "Europe": ["", "To be fair, ", "In general, "],
    "Asia-Pacific": ["", "Quick note: ", ""],
    "Latin America": ["", "Pretty good, but also: ", ""],
    "Middle East & Africa": ["", "Delivery was fine. ", ""],
}


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def make_record(index: int, rng: random.Random) -> dict[str, Any]:
    """Create one internally consistent synthetic review and its target."""
    product = rng.choice(sorted(CATALOG))
    category, features, problems = CATALOG[product]
    region = rng.choices(REGIONS, weights=REGION_WEIGHTS)[0]
    kind = rng.choices(("mixed", "positive", "negative"), weights=(0.6, 0.2, 0.2))[0]
    feature = rng.choice(features)
    problem, advice = rng.choice(problems)
    opener = rng.choice(REGION_OPENERS[region])
    noun = product.lower()

    subject = f"The {noun}"
    if kind == "mixed":
        review = f"{opener}{subject} {feature}, but {problem}."
        rating = rng.choice((2, 3, 3, 4))
        target = (f"Positive feature: {_cap(feature)}. Primary problem: {_cap(problem)}. "
                  f"Recommendation: {advice}.")
    elif kind == "positive":
        review = f"{opener}{subject} {feature}. No complaints so far."
        rating = rng.choice((4, 5, 5))
        target = (f"Positive feature: {_cap(feature)}. Primary problem: None reported. "
                  "Recommendation: Keep the current design.")
    else:
        review = f"{opener}I regret buying this. {_cap(problem)}."
        rating = rng.choice((1, 1, 2))
        target = (f"Positive feature: None reported. Primary problem: {_cap(problem)}. "
                  f"Recommendation: {advice}.")
    review = review.strip()
    return {
        "review_id": f"SYN-{index:04d}",
        "product": product,
        "category": category,
        "region": region,
        "rating": rating,
        "review": review,
        "instruction": rng.choice(INSTRUCTIONS),
        "context": format_context(product, rating, review),
        "target": target,
    }


def generate_records(count: int, seed: int = SEED) -> list[dict[str, Any]]:
    """Generate a reproducible list of records (no split assigned yet)."""
    rng = random.Random(seed)
    return [make_record(i + 1, rng) for i in range(count)]


def split_records(
    records: list[dict[str, Any]], seed: int = SEED
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle with a fixed seed and split 80/10/10 with disjoint IDs."""
    shuffled = [dict(r) for r in records]
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.80)
    validation_end = int(len(shuffled) * 0.90)
    parts = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    for name, rows in parts.items():
        for row in rows:
            row["split"] = name
            validate_record(row)
    return parts


def write_raw_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the unsplit source data as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["review_id", "product", "category", "region", "rating", "review"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in fields})


def main() -> None:
    """Generate data, split it, and write files plus a hash manifest."""
    from sba_928.tasks import build_dataset, split_by_task

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    pool = generate_records(args.count, args.seed)
    write_raw_csv(RAW_CSV, pool)
    per_task = args.count // 6
    records = build_dataset(pool, CATALOG, per_task, args.seed)
    parts = split_by_task(records, args.seed)
    for rows in parts.values():
        for row in rows:
            validate_record(row)

    paths = {"train": TRAIN_FILE, "validation": VALIDATION_FILE, "test": TEST_FILE}
    manifest: dict[str, Any] = {"seed": args.seed, "files": {}}
    for name, path in paths.items():
        write_jsonl(path, parts[name])
        manifest["files"][name] = {
            "path": str(path),
            "records": len(parts[name]),
            "sha256": sha256_file(path),
        }
        LOGGER.info("%s: %d records -> %s", name, len(parts[name]), path)
    write_json(MANIFEST_FILE, manifest)


if __name__ == "__main__":
    main()