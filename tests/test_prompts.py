from sba_928.prompts import (
    build_prompt_variants,
    build_scenario_prompts,
    format_training_input,
)


def test_variants_present_and_different():
    variants = build_prompt_variants("CTX")
    assert {"minimal", "structured", "grounded"} <= set(variants)
    assert len(set(variants.values())) == len(variants)
    assert all("CTX" in v for v in variants.values())


def test_scenarios_cover_three_research_angles():
    prompts = build_scenario_prompts("Desk Lamp", "home", 3, "CTX")
    assert set(prompts) == {"consumer_behavior", "market_trend", "competitor_analysis"}


def test_training_input_format():
    assert format_training_input(" Do it. ", " ctx ") == "Do it.\n\nContext:\nctx"