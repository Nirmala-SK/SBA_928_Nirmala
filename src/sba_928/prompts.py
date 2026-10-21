"""Prompt templates: training format, controlled variants, and scenario variations."""

from __future__ import annotations

INSTRUCTIONS = (
    "Identify the positive feature, problem, and recommendation.",
    "Summarize what the customer liked, what went wrong, and what the company should do.",
    "Extract the positive feature, primary problem, and a recommendation from this review.",
)

# Labels the model must produce, for each task family.
TASK_LABELS = {
    "review_insight": "Positive feature, Primary problem, and Recommendation",
    "consumer_behavior": "Buying driver, Pain point, and Sentiment",
    "market_trend": "Trend, Change, and Implication",
    "competitor_analysis": "Leader, Rating gap, and Recommendation",
    "product_demand": "Demand level, Reason, and Recommendation",
    "customer_segmentation": "Largest segment, Smallest segment, and Recommendation",
}

# Opening line of the "minimal" prompt, for each task family.
TASK_MINIMAL = {
    "review_insight": "Analyze this customer review.",
    "consumer_behavior": "Analyze the consumer behavior in these reviews.",
    "market_trend": "Describe the market trend in this data.",
    "competitor_analysis": "Compare these two competing products.",
    "product_demand": "Assess the demand for this product.",
    "customer_segmentation": "Segment this customer group.",
}

# One worked example per task family, used by the few-shot prompt.
TASK_EXAMPLES = {
    "review_insight": (
        "Example\nReview: The lamp is bright, but the switch feels loose.\n"
        "Positive feature: Bright light. Primary problem: Loose switch. "
        "Recommendation: Use a firmer switch.\n\n"
    ),
    "consumer_behavior": (
        "Example\nProduct: Desk Lamp\nReviews:\n"
        "1. (rating 4) The desk lamp gives bright, even light, but the switch feels loose.\n"
        "Buying driver: Gives bright, even light. Pain point: The switch feels loose. "
        "Sentiment: Satisfied customers (average rating 4.0).\n\n"
    ),
    "market_trend": (
        "Example\nCategory: food. Region: Europe. Average rating by quarter: "
        "Q1 3.1, Q2 3.4, Q3 3.7, Q4 4.0.\n"
        "Trend: Rising. Change: +0.9 points. Implication: Increase marketing and stock "
        "for food.\n\n"
    ),
    "competitor_analysis": (
        "Example\nCategory: drinkware. Compare two competing products using 8 reviews each.\n"
        "Reusable Water Bottle: average rating 3.8, low ratings (1-2 stars) 1 of 8.\n"
        "Insulated Coffee Mug: average rating 3.2, low ratings (1-2 stars) 2 of 8.\n"
        "Leader: Reusable Water Bottle. Rating gap: 0.6. Recommendation: Insulated Coffee "
        "Mug should fix its most common complaints to close the gap with Reusable Water "
        "Bottle.\n\n"
    ),
    "product_demand": (
        "Example\nProduct: Yoga Mat. Category: fitness. Reviews last month: 55. "
        "Average rating: 3.6.\n"
        "Demand level: High. Reason: High review volume with solid ratings. "
        "Recommendation: Increase inventory.\n\n"
    ),
    "customer_segmentation": (
        "Example\nCustomer group: home buyers in Europe. Total customers: 10. "
        "Rated 4-5: 6 (60%). Rated 3: 3 (30%). Rated 1-2: 1 (10%).\n"
        "Largest segment: Promoters (60%). Smallest segment: Detractors. "
        "Recommendation: Launch a loyalty or referral program.\n\n"
    ),
}


def format_context(product: str, rating: int, review: str) -> str:
    """Build the context string stored in every review record."""
    return f"Product: {product} | Rating: {rating}/5 | Review: {review}"


def format_training_input(instruction: str, context: str) -> str:
    """Single input format used for training, evaluation, and inference."""
    return f"{instruction.strip()}\n\nContext:\n{context.strip()}"


def build_prompt_variants(
    context: str, task: str = "review_insight"
) -> dict[str, str]:
    """Prompt-engineering variants. Only the wording differs between them."""
    labels = TASK_LABELS[task]
    example = TASK_EXAMPLES[task]
    return {
        "minimal": f"{TASK_MINIMAL[task]}\n\n{context}",
        "structured": (
            f"Identify the {labels}. Use those labels exactly.\n\n{context}"
        ),
        "grounded": (
            "Use only the supplied data. Do not invent facts. "
            f"Identify the {labels}.\n\n{context}"
        ),
        "role_based": (
            "You are a market research analyst preparing findings for a product "
            f"team. Report the {labels}, using those labels.\n\n{context}"
        ),
        "few_shot": (
            f"Identify the {labels}. Use those labels exactly.\n\n{example}{context}"
        ),
    }


# Scenario templates show how one dataset record can drive different market
# research questions. They are used in the report and in step 2 for illustration.
SCENARIO_TEMPLATES = {
    "consumer_behavior": (
        "What purchase motivation or frustration does this {rating}/5 review of "
        "{product} reveal about the customer?\n\n{context}"
    ),
    "market_trend": (
        "If many {category} customers wrote reviews like this one, which product "
        "trend would it point to?\n\n{context}"
    ),
    "competitor_analysis": (
        "Which weakness here could a competitor in {category} exploit, and what "
        "should {product}'s maker fix first?\n\n{context}"
    ),
}


def build_scenario_prompts(
    product: str, category: str, rating: int, context: str
) -> dict[str, str]:
    """Fill every scenario template for one record."""
    return {
        name: template.format(
            product=product, category=category, rating=rating, context=context
        )
        for name, template in SCENARIO_TEMPLATES.items()
    }