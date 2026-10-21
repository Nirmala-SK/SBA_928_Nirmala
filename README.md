# SBA 928: Prompt Engineering and Fine-Tuning for Market Research

A packaged `uv` application. Prompt engineering (Step 2) changes no model
parameters. Fine-tuning (Step 3) does. The two are reported separately. The
model is fine-tuned on six market research task families, not on a single
review-extraction task.

Full report (version 2): `docs/v2/Nirmala_SBA_928_Project_Report_v2.docx`
Version 1 report: `docs/v1/Nirmala_SBA_928_Project_Report_v1.docx`

## Reproduce

```bash
uv sync --group dev
uv run python -m pytest
uv run python -m ruff check .
uv run python -m sba_928.step_01_generate_dataset
uv run python -m sba_928.step_02_prompt_engineering --limit 12 --device cpu
uv run python -m sba_928.step_03_fine_tune --config config/experiment3.json
uv run python -m sba_928.step_04_compare_models --fine-tuned-model artifacts/models/flan-t5-sba-928-run3/final --output-dir artifacts/evaluation_run3 --device cpu
uv run python -m sba_928.step_05_bias_audit --eval-dir artifacts/evaluation_run3 --output-dir artifacts/bias_audit_run3
uv run python -m sba_928.step_06_custom_inference --model artifacts/models/flan-t5-sba-928-run3/final --product "Wireless Mouse" --rating 3 --review "The mouse is comfortable, but it disconnects several times each day." --device cpu
uv run python -m sba_928.step_06_custom_inference --model artifacts/models/flan-t5-sba-928-run3/final --task product_demand --context "Product: Wireless Mouse. Category: electronics. Reviews last month: 62. Average rating: 3.9." --device cpu
```

The first model command downloads google/flan-t5-small. Model checkpoints are
not committed to GitHub. The selected model is written to
`artifacts/models/flan-t5-sba-928-run3/final`.
Results for this version: `artifacts/prompt_baseline`,
`artifacts/evaluation_run3`, `artifacts/bias_audit_run3`.
`artifacts/evaluation_run1`, `artifacts/evaluation_run2` and
`artifacts/bias_audit` hold results from the earlier single-task version of the
project (review extraction only). They are kept for reference and are not
comparable with the scores below.

## 1. Dataset and task families

**Source data.** 600 synthetic customer reviews (12 products, 7 categories,
5 regions, ratings 1 to 5), generated with seed 42. Data is synthetic as
specified in the project guide. This pool of reviews is the source for all six
task families.

**Task families.** The fine-tuning dataset has 600 records: 100 for each of six
task families. Every record has an instruction, a context and a labelled
target. Targets are computed from the context by fixed rules, so labels are
deterministic.

| Task family | Input (context) | Target labels | Rule |
|---|---|---|---|
| Review insight | One review | Positive feature, Primary problem, Recommendation | Templated review text |
| Consumer behavior | 4 reviews of one product | Buying driver, Pain point, Sentiment | Most frequent feature and problem; sentiment from the average rating (4.0 or more satisfied, 3.0 to 3.9 mixed, below 3.0 dissatisfied) |
| Market trends | Average rating by quarter for a category and region | Trend, Change, Implication | Change of +0.3 or more is Rising, -0.3 or less is Falling, otherwise Stable. The quarterly series is synthetic and generated from a seeded random trend |
| Competitor analysis | Two products in one category, 8 reviews each | Leader, Rating gap, Recommendation | Higher average rating leads; equal averages are a Tie |
| Product demand | Reviews last month and average rating | Demand level, Reason, Recommendation | Low if under 15 reviews or rating below 2.8; High if 40 or more reviews and rating 3.2 or higher; otherwise Medium |
| Customer segmentation | Counts and percentages of customers rated 4-5, 3 and 1-2 | Largest segment, Smallest segment, Recommendation | Groups with tied counts are excluded so the largest segment is always clear |

**Split.** 80/10/10 inside every task family, giving 480 train, 60 validation
and 60 test records (10 test records per family) with disjoint IDs. Every split
therefore covers all six families.

**Prompt variations.** `prompts.py` builds five prompt variants (minimal,
structured, grounded, role-based, few-shot) for each of the six families, each
with its own labels and its own worked example. The original scenario templates
for consumer behavior, market trends and competitor analysis remain in
`prompts.py`; those three scenarios and two more (product demand, customer
segmentation) are now trained and evaluated directly.

## 2. Prompt engineering (no training)

Unmodified FLAN-T5-small on 12 validation records, two from each task family.

| Variant | token F1 | format adherence |
|---|---|---|
| minimal | 0.086 | 0.0 |
| structured | 0.050 | 0.0 |
| grounded | 0.084 | 0.0 |
| role_based | 0.089 | 0.0 |
| few_shot | 0.228 | 0.167 |

Four of the five variants never produced the required labelled format. The
few-shot prompt scored highest, and 2 of its 12 answers followed the format.
This step did not train or modify the model, and it shows that changing the
wording of the prompt alone barely moves a small base model.

## 3. Fine-tuning

**Preparation.** Input is `instruction + "\n\nContext:\n" + context`; the label
is the target. Max lengths 384 (source) and 128 (target) tokens.

**Training.** google/flan-t5-small, Seq2SeqTrainer, CPU, seed 42, 10 epochs,
learning rate 1e-4, batch size 4 with gradient accumulation 2. All six task
families are trained together in one model. The best checkpoint is chosen on
validation loss. The test file is never read during training.

**Comparison on 60 held-out test records (identical inputs and decoding).**

| Metric | Base | Fine-tuned |
|---|---|---|
| token F1 | 0.078 | 0.818 |
| format adherence | 0.0 | 1.0 |
| exact match | 0.0 | 0.267 |
| key-field accuracy | 0.0 | 0.683 |
| first / second / third field F1 | 0.0 / 0.0 / 0.0 | 0.725 / 0.776 / 0.644 |

Key-field accuracy is an exact match on the first labelled field: the positive
feature (review insight), buying driver (consumer behavior), trend (market
trends), leader (competitor analysis), demand level (product demand) and
largest segment (customer segmentation).

**Results by task family (10 test records each).**

| Task family | token F1 base to tuned | exact match base to tuned | key field base to tuned |
|---|---|---|---|
| Review insight | 0.179 to 0.840 | 0.0 to 0.2 | 0.0 to 1.0 |
| Consumer behavior | 0.011 to 0.925 | 0.0 to 0.4 | 0.0 to 0.8 |
| Market trends | 0.055 to 0.754 | 0.0 to 0.0 | 0.0 to 0.6 |
| Competitor analysis | 0.162 to 0.885 | 0.0 to 0.1 | 0.0 to 0.7 |
| Product demand | 0.053 to 0.817 | 0.0 to 0.6 | 0.0 to 0.6 |
| Customer segmentation | 0.008 to 0.687 | 0.0 to 0.3 | 0.0 to 0.4 |

Every family improved over the base model, and the model followed the required
label format on all 60 test records.

**Iteration on customer segmentation.** In the first multi-task run, customer
segmentation scored key-field accuracy 0.0 and token F1 0.490. Inspecting the
answers showed the model was not comparing the counts: it gave almost the same
answer for every input and invented percentages. Two causes in the task design
were found. The percentages had to be calculated rather than being given, and
some groups had tied counts, which made the target an arbitrary tie-break. The
generator was changed to include percentages in the context and to exclude
ties, the dataset was regenerated and the model retrained. Segmentation then
reached token F1 0.687 and key-field accuracy 0.4. Only the segmentation
records changed, so the other five families are comparable between runs; their
small shifts (for example consumer behavior token F1 0.956 to 0.925) are within
the noise expected from retraining on 10 test records.

**Custom inference.** Two examples from the final model:

- A reworded Wireless Mouse review ("The mouse is comfortable, but it
  disconnects several times each day.") returned the correct positive feature
  and problem, but the recommendation was "Keep the current design". That is
  the answer the training data gives for reviews with no problem, so the
  advice does not fit this review.
- A product demand input (62 reviews last month, average rating 3.9) returned
  "Demand level: Medium". The rule gives High, so this is an error. The model
  has trouble applying numeric thresholds.

Both are single examples and show where the model fails.

**Limitations.**

- Each task family has only 10 test records, so one record moves a family's
  score by 10 percentage points. The per-family numbers show direction, not
  precise performance.
- Customer segmentation (key-field accuracy 0.4) and product demand (0.6) are
  the weakest families. A small model only partly learns to compare numbers
  and apply thresholds from 80 training examples per family.
- Data is synthetic and templated, and targets come from fixed rules. The
  families are built from one shared pool of 600 reviews, so a test record can
  reuse reviews that also appear in a training record's context. Contexts are
  unique within each family (tested), but overlap in the underlying reviews was
  not measured. High scores should not be read as performance on real
  customers or real reviews.
- The grounding score checks whether words in the answer appear in the source
  text. It is meaningful for review insight and consumer behavior, but not for
  the numeric families, whose answers use words such as "Medium" and "Rising"
  that never appear in the input. It is not used as a headline result.
- Market trend series are generated from a seeded random trend, not from the
  review data.

## 4. Bias analysis and fairness

**Training data audit (480 training records).**

- Task families are balanced (16.7% each).
- Categories: apparel 18.5%, home 18.1%, food 17.1%, electronics 15.8%,
  drinkware 15.2%, fitness 7.7%, beauty 7.5%.
- Region is a real attribute for 240 of the 480 records (review insight,
  market trends and customer segmentation). The other 240 pool all regions and
  are left out of region comparisons. Among the 240: North America 30.0%,
  Europe 24.2%, Asia-Pacific 18.8%, Latin America 15.0%, Middle East & Africa
  12.1%.
- On the 80 review-insight training records, average rating by region ranged
  from 2.75 (Middle East & Africa) and 2.79 (Asia-Pacific) to 3.27 (North
  America); average review length ranged from 11.9 to 14.0 words. These are
  small samples per region.
- No loaded terms were found with the project word list, which is a weak check.

**Model results (fine-tuned, token F1).** The largest gaps among groups with at
least 10 test records were electronics (-0.107, n=10), apparel (+0.098, n=12)
and rating 4-5 (+0.059, n=13). Groups below 10 test records, which cannot be
assessed reliably: beauty 8, drinkware 8, fitness 4, food 9 and home 9; every
region (Asia-Pacific 6, Europe 8, Latin America 6, Middle East & Africa 4,
North America 6); and rating 1-2 (5). Only 30 of the 60 test records have a
single region, so no regional gap could be assessed.

The six task families differ in difficulty (family token F1 ranged from 0.687
to 0.925), so a category's score can be affected by which families happen to
fall in it. This was not tested. The gaps are observations on small samples,
not evidence of unfairness or of its cause. Groups are synthetic labels, not
real demographic evidence.

**Fairness strategies tied to these findings.**
- Stratify splits by region and category and add examples for the smallest
  groups (Middle East & Africa, fitness, beauty).
- Build a larger test set (about 30 or more records per family and per group)
  and report uncertainty.
- Keep region out of the model input (done here) and check predictions do not
  shift with sentiment.
- Extend the loaded-term list and have a person review sample prompts and
  answers.
- Report each task family and each industry category separately.

## Layout

`src/sba_928/` has `step_01` to `step_06` plus helpers (`tasks.py` builds the
five new task families). Tests are in `tests/` (18 tests, including checks on
the multi-task dataset). The report is in `docs/`.