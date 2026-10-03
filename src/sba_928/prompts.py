"""Prompt templates: training format, controlled variants, and scenario variations."""

from __future__ import annotations

INSTRUCTIONS = (
    "Identify the positive feature, problem, and recommendation.",
    "Summarize what the customer liked, what went wrong, and what the company should do.",
    "Extract the positive feature, primary problem, and a recommendation from this review.",
)


def format_context(product: str, rating: int, review: str) -> str:
    """Build the context string stored in every record."""
    return f"Product: {product} | Rating: {rating}/5 | Review: {review}"


def format_training_input(instruction: str, context: str) -> str:
    """Single input format used for training, evaluation, and inference."""
    return f"{instruction.strip()}\n\nContext:\n{context.strip()}"


def build_prompt_variants(context: str) -> dict[str, str]:
    """Prompt-engineering variants. Only the wording differs between them."""
    labels = "Positive feature, Primary problem, and Recommendation"
    example = (
        "Example\nReview: The lamp is bright, but the switch feels loose.\n"
        "Positive feature: Bright light. Primary problem: Loose switch. "
        "Recommendation: Use a firmer switch.\n\n"
    )
    return {
        "minimal": f"Analyze this customer review.\n\n{context}",
        "structured": (
            f"Identify the {labels}. Use those labels exactly.\n\n{context}"
        ),
        "grounded": (
            "Use only the supplied review. Do not invent facts. "
            f"Identify the {labels}.\n\n{context}"
        ),
        "role_based": (
            "You are a market research analyst summarizing customer feedback for a "
            f"product team. Report the {labels}, using those labels.\n\n{context}"
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