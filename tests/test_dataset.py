from sba_928.schema import validate_record
from sba_928.step_01_generate_dataset import generate_records, split_records


def test_counts_and_split_sizes():
    parts = split_records(generate_records(600))
    assert [len(parts[k]) for k in ("train", "validation", "test")] == [480, 60, 60]


def test_ids_do_not_overlap():
    parts = split_records(generate_records(600))
    ids = [{r["review_id"] for r in rows} for rows in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(s) for s in ids) == 600


def test_reproducible():
    assert generate_records(50, seed=7) == generate_records(50, seed=7)
    assert generate_records(50, seed=7) != generate_records(50, seed=8)


def test_records_valid_and_target_supported_by_review():
    for rows in split_records(generate_records(200)).values():
        for r in rows:
            validate_record(r)
            problem_part = r["target"].split("Primary problem:")[1]
            if not problem_part.strip().startswith("None reported"):
                problem = problem_part.split(". Recommendation")[0]
                assert problem.strip().lower() in r["review"].lower()