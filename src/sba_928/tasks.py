"""Market research task families beyond review extraction.

Every target is computed from the context by a fixed rule, so labels are
deterministic and can be checked by tests.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from statistics import mean
from typing import Any

TASK_INSTRUCTIONS: dict[str, list[str]] = {
    "consumer_behavior": [
        "Analyze the consumer behavior in these reviews.",
        "Summarize what drives buying and what frustrates these customers.",
    ],
    "market_trend": [
        "Describe the market trend shown by this rating data.",
        "Identify the trend direction and what it means for the business.",
    ],
    "competitor_analysis": [
        "Compare these two competing products and name the leader.",
        "Run a competitor analysis on the products below.",
    ],
    "product_demand": [
        "Estimate the demand level for this product.",
        "Assess product demand and recommend an inventory action.",
    ],
    "customer_segmentation": [
        "Segment this customer group by satisfaction.",
        "Identify the largest and smallest customer segments and recommend an action.",
    ],
}


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _by_product(pool: list[dict[str, Any]], product: str) -> list[dict[str, Any]]:
    return [r for r in pool if r["product"] == product]


def _avg(rows: list[dict[str, Any]]) -> float:
    return round(mean(r["rating"] for r in rows), 1)


def _sentiment(avg: float) -> str:
    if avg >= 4.0:
        return "Satisfied customers"
    if avg >= 3.0:
        return "Mixed sentiment"
    return "Dissatisfied customers"


def consumer_behavior(pool: list[dict[str, Any]], catalog: Any, rng: random.Random) -> dict[str, Any]:
    product = rng.choice(sorted(catalog))
    category, features, problems = catalog[product]
    sample = rng.sample(_by_product(pool, product), 4)
    lines = [f"{i}. (rating {r['rating']}) {r['review']}" for i, r in enumerate(sample, 1)]

    def count(phrase: str) -> int:
        return sum(phrase.lower() in r["review"].lower() for r in sample)

    best_feature = max(features, key=count)
    best_problem = max((p for p, _ in problems), key=count)
    driver = _cap(best_feature) if count(best_feature) else "None reported"
    pain = _cap(best_problem) if count(best_problem) else "None reported"
    avg = _avg(sample)
    return {
        "product": product,
        "category": category,
        "region": "All regions",
        "rating": round(avg),
        "context": f"Product: {product}\nReviews:\n" + "\n".join(lines),
        "target": (f"Buying driver: {driver}. Pain point: {pain}. "
                   f"Sentiment: {_sentiment(avg)} (average rating {avg})."),
    }


def market_trend(pool: list[dict[str, Any]], catalog: Any, rng: random.Random) -> dict[str, Any]:
    category = rng.choice(sorted({v[0] for v in catalog.values()}))
    region = rng.choice(sorted({r["region"] for r in pool}))
    direction = rng.choice(("up", "down", "flat"))
    value = rng.uniform(3.0, 4.0)
    series: list[float] = []
    for _ in range(4):
        series.append(round(min(5.0, max(1.0, value)), 1))
        if direction == "up":
            value += rng.uniform(0.2, 0.4)
        elif direction == "down":
            value -= rng.uniform(0.2, 0.4)
        else:
            value += rng.uniform(-0.05, 0.05)
    change = round(series[-1] - series[0], 1) + 0.0
    if change >= 0.3:
        label, implication = "Rising", f"Increase marketing and stock for {category}"
    elif change <= -0.3:
        label, implication = "Falling", f"Investigate quality issues before expanding {category}"
    else:
        label, implication = "Stable", f"Maintain the current strategy for {category}"
    quarters = ", ".join(f"Q{i} {v}" for i, v in enumerate(series, 1))
    return {
        "product": "All products",
        "category": category,
        "region": region,
        "rating": round(mean(series)),
        "context": f"Category: {category}. Region: {region}. Average rating by quarter: {quarters}.",
        "target": f"Trend: {label}. Change: {change:+.1f} points. Implication: {implication}.",
    }


def competitor_analysis(pool: list[dict[str, Any]], catalog: Any, rng: random.Random) -> dict[str, Any]:
    by_category: dict[str, list[str]] = {}
    for name, (cat, _, _) in catalog.items():
        by_category.setdefault(cat, []).append(name)
    category = rng.choice(sorted(c for c, p in by_category.items() if len(p) >= 2))
    first, second = rng.sample(sorted(by_category[category]), 2)
    stats = {}
    for p in (first, second):
        sample = rng.sample(_by_product(pool, p), 8)
        stats[p] = (_avg(sample), sum(r["rating"] <= 2 for r in sample))
    (avg1, low1), (avg2, low2) = stats[first], stats[second]
    if avg1 == avg2:
        leader, gap = "Tie", 0.0
        rec = "Compete on price, features, and service"
    else:
        leader, trailer = (first, second) if avg1 > avg2 else (second, first)
        gap = abs(avg1 - avg2)
        rec = f"{trailer} should fix its most common complaints to close the gap with {leader}"
    context = (
        f"Category: {category}. Compare two competing products using 8 reviews each.\n"
        f"{first}: average rating {avg1}, low ratings (1-2 stars) {low1} of 8.\n"
        f"{second}: average rating {avg2}, low ratings (1-2 stars) {low2} of 8."
    )
    return {
        "product": f"{first} vs {second}",
        "category": category,
        "region": "All regions",
        "rating": round((avg1 + avg2) / 2),
        "context": context,
        "target": f"Leader: {leader}. Rating gap: {gap:.1f}. Recommendation: {rec}.",
    }


def product_demand(pool: list[dict[str, Any]], catalog: Any, rng: random.Random) -> dict[str, Any]:
    product = rng.choice(sorted(catalog))
    category = catalog[product][0]
    avg = _avg(rng.sample(_by_product(pool, product), 10))
    volume = rng.randint(5, 80)
    low_volume, weak = volume < 15, avg < 2.8
    if low_volume or weak:
        level = "Low"
        reason = ("Low review volume and weak ratings" if low_volume and weak
                  else "Low review volume" if low_volume else "Weak ratings")
        action = "Reduce inventory and review the marketing plan"
    elif volume >= 40 and avg >= 3.2:
        level, reason = "High", "High review volume with solid ratings"
        action = "Increase inventory"
    else:
        level, reason = "Medium", "Moderate review volume or ratings"
        action = "Maintain current inventory"
    return {
        "product": product,
        "category": category,
        "region": "All regions",
        "rating": round(avg),
        "context": (f"Product: {product}. Category: {category}. "
                    f"Reviews last month: {volume}. Average rating: {avg}."),
        "target": f"Demand level: {level}. Reason: {reason}. Recommendation: {action}.",
    }


def customer_segmentation(pool: list[dict[str, Any]], catalog: Any, rng: random.Random) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in pool:
        cells.setdefault((r["category"], r["region"]), []).append(r)
    keys = sorted(k for k, v in cells.items() if len(v) >= 6)
    while True:
        category, region = rng.choice(keys)
        group = cells[(category, region)]
        size = rng.randint(6, len(group))
        sample = rng.sample(group, size)
        counts = {
            "Promoters": sum(r["rating"] >= 4 for r in sample),
            "Passives": sum(r["rating"] == 3 for r in sample),
            "Detractors": sum(r["rating"] <= 2 for r in sample),
        }
        if len(set(counts.values())) == 3:  # no ties, so the answer is clear
            break
    pct = {name: round(100 * c / size) for name, c in counts.items()}
    largest = max(counts, key=lambda n: counts[n])
    smallest = min(counts, key=lambda n: counts[n])
    rec = {
        "Promoters": "Launch a loyalty or referral program",
        "Passives": "Offer targeted incentives to move customers toward loyalty",
        "Detractors": "Fix product issues and follow up with unhappy customers",
    }[largest]
    return {
        "product": "All products",
        "category": category,
        "region": region,
        "rating": round(mean(r["rating"] for r in sample)),
        "context": (f"Customer group: {category} buyers in {region}. Total customers: {size}. "
                    f"Rated 4-5: {counts['Promoters']} ({pct['Promoters']}%). "
                    f"Rated 3: {counts['Passives']} ({pct['Passives']}%). "
                    f"Rated 1-2: {counts['Detractors']} ({pct['Detractors']}%)."),
        "target": (f"Largest segment: {largest} ({pct[largest]}%). Smallest segment: {smallest}. "
                   f"Recommendation: {rec}."),
    }


Builder = Callable[[list[dict[str, Any]], Any, random.Random], dict[str, Any]]

BUILDERS: dict[str, tuple[str, Builder]] = {
    "consumer_behavior": ("CB", consumer_behavior),
    "market_trend": ("MT", market_trend),
    "competitor_analysis": ("CA", competitor_analysis),
    "product_demand": ("PD", product_demand),
    "customer_segmentation": ("CS", customer_segmentation),
}


def generate_task_records(
    pool: list[dict[str, Any]], catalog: Any, per_task: int, seed: int
) -> list[dict[str, Any]]:
    """Build `per_task` unique records for each task family from the review pool."""
    rng = random.Random(seed + 1)
    records: list[dict[str, Any]] = []
    for task, (prefix, builder) in BUILDERS.items():
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_task:
            attempts += 1
            if attempts > per_task * 50:
                raise RuntimeError(f"Could not build {per_task} unique records for {task}")
            built = builder(pool, catalog, rng)
            if built["context"] in seen:
                continue
            seen.add(built["context"])
            records.append({
                "review_id": f"{prefix}-{len(seen):04d}",
                "task_type": task,
                **built,
                "review": "",
                "instruction": rng.choice(TASK_INSTRUCTIONS[task]),
            })
    return records

def build_dataset(
    pool: list[dict[str, Any]], catalog: Any, per_task: int, seed: int
) -> list[dict[str, Any]]:
    """Mix review extraction with the five new task families (per_task records each)."""
    insight = []
    for row in pool[:per_task]:
        record = dict(row)
        record["task_type"] = "review_insight"
        insight.append(record)
    return insight + generate_task_records(pool, catalog, per_task, seed)


def split_by_task(
    records: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Split 80/10/10 inside every task family so each split covers all tasks."""
    parts: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task_type"], []).append(dict(record))
    for task in sorted(by_task):
        rows = by_task[task]
        random.Random(seed).shuffle(rows)
        train_end = int(len(rows) * 0.80)
        validation_end = int(len(rows) * 0.90)
        chunks = {
            "train": rows[:train_end],
            "validation": rows[train_end:validation_end],
            "test": rows[validation_end:],
        }
        for name, chunk in chunks.items():
            for row in chunk:
                row["split"] = name
            parts[name].extend(chunk)
    for rows in parts.values():
        random.Random(seed).shuffle(rows)
    return parts