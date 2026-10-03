from sba_928.metrics import (
    format_adherence,
    grounding,
    parse_fields,
    score_example,
    subgroup_metrics,
    token_f1,
)

GOOD = (
    "Positive feature: Quiet click. Primary problem: Battery dies. "
    "Recommendation: Add recharging."
)


def test_token_f1_basics():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("x y", "a b") == 0.0
    assert 0 < token_f1("a b", "a c") < 1


def test_format_and_parse():
    assert format_adherence(GOOD) == 1.0
    assert format_adherence("Analysis: fine") == 0.0
    assert parse_fields(GOOD)["problem"] == "Battery dies."


def test_grounding_penalizes_invented_facts():
    review = "The mouse has a quiet click but the battery dies."
    assert grounding(GOOD, review) == 1.0
    invented = (
        "Positive feature: Waterproof. Primary problem: Screen cracks. "
        "Recommendation: Fix."
    )
    assert grounding(invented, review) == 0.0


def test_perfect_prediction_scores_one():
    scores = score_example(GOOD, GOOD, "quiet click battery dies")
    assert scores["token_f1"] == scores["exact_match"] == scores["format_adherence"] == 1.0


def test_subgroup_metrics_flags_small_groups():
    rows = [{"region": "A", "token_f1": 1.0}] * 3 + [{"region": "B", "token_f1": 0.0}] * 12
    result = {r["group"]: r for r in subgroup_metrics(rows, "region", min_n=10)}
    assert result["A"]["small_sample"] and not result["B"]["small_sample"]
    assert result["A"]["gap_vs_overall"] > 0 > result["B"]["gap_vs_overall"]