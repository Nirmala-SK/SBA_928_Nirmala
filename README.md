# SBA 928: Prompt Engineering and Fine-Tuning for Market Research

A packaged `uv` application. Prompt engineering (Step 2) changes no model
parameters. Fine-tuning (Step 3) does. The two are reported separately.

## Reproduce

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run python -m sba_928.step_01_generate_dataset
uv run python -m sba_928.step_02_prompt_engineering --limit 12 --device cpu
uv run python -m sba_928.step_03_fine_tune --config config/default.json
uv run python -m sba_928.step_03_fine_tune --config config/experiment2.json
uv run python -m sba_928.step_04_compare_models --fine-tuned-model artifacts/models/flan-t5-sba-928-run2/final --device cpu
uv run python -m sba_928.step_05_bias_audit
uv run python -m sba_928.step_06_custom_inference --model artifacts/models/flan-t5-sba-928-run2/final --product "Wireless Mouse" --rating 3 --review "The mouse is comfortable, but it disconnects several times each day."
```

The first model command downloads google/flan-t5-small. Model checkpoints are
not committed to GitHub. They are written to `artifacts/models/`: run 1 in
`flan-t5-sba-928/final`, run 2 (selected model) in `flan-t5-sba-928-run2/final`.
Results: `artifacts/evaluation_run1`, `artifacts/evaluation_run2`,
`artifacts/prompt_baseline`, `artifacts/bias_audit`.

## 1. Dataset and prompt variations

**Dataset.** 600 synthetic customer reviews (12 products, 7 categories,
5 regions, ratings 1 to 5), generated with seed 42 and split 80/10/10 into
480 train, 60 validation and 60 test records with disjoint IDs. Each record has
an instruction, context and a labelled target (Positive feature, Primary
problem, Recommendation). Data is synthetic as specified in the project guide.

**Market research angles.** `prompts.py` provides scenario templates for
consumer behavior, market trends and competitor analysis built from the same
records.

**Prompt variants (12 validation reviews, unmodified FLAN-T5-small).**

| Variant | token F1 | format adherence |
|---|---|---|
| minimal | 0.154 | 0.0 |
| structured | 0.163 | 0.0 |
| grounded | 0.171 | 0.0 |
| role_based | 0.267 | 0.0 |
| few_shot | 0.035 | 0.0 |

No variant produced the required three-label format. In the four validation reviews inspected, answers were short and often not the task: minimal and grounded gave sentiment-like or generic phrases ("Five Stars", "The product is not good for the skin."), structured and few_shot mostly returned one or two words ("Positive feature", "Positive", "Negative"), and role_based repeated the list of labels from the prompt ("Positive feature, Primary problem, and Recommendation") instead of answering. That repetition explains why role_based had the highest token F1 (0.267): the correct answers contain those label words, so copying them from the prompt earns overlap without analysing the review. Some answers were not supported by the review, such as "The product is not good for the skin" for a review praising how quickly the moisturizer absorbs. Only one answer touched the real problem, and only partly ("The running shoes are very small and the size is small."). No variant gave a recommendation. This step did not train or modify the model.

## 2. Fine-tuning

**Preparation.** Input is `instruction + "\n\nContext:\n" + context`; the label
is the target. Max lengths 256 (source) and 96 (target) tokens.

**Training.** google/flan-t5-small, Seq2SeqTrainer, CPU, seed 42, batch size 4
with gradient accumulation 2. The best checkpoint is chosen on validation loss.
The test file is never read during training.

| | Run 1 | Run 2 (selected) |
|---|---|---|
| Epochs | 3 | 8 |
| Learning rate | 5e-5 | 1e-4 |
| Final validation loss | 0.3629 | 0.0034 |

Run 2 was selected on validation loss. Two settings changed together, so the
improvement cannot be attributed to either one alone.

**Comparison on 60 held-out test reviews (identical inputs and decoding).**

| Metric | Base | Run 1 | Run 2 |
|---|---|---|---|
| token F1 | 0.176 | 0.809 | 0.994 |
| recommendation F1 | 0.0 | 0.282 | 0.974 |
| format adherence | 0.0 | 1.0 | 1.0 |
| grounding | 0.0 | 0.996 | 1.0 |
| exact match | 0.0 | 0.0 | 0.967 |

Run 1 identified the positive feature and problem well but often gave
recommendations that did not fit the problem (for example "Improve the grip"
for a rubber-smell complaint). Run 2 largely fixed this.

**Limitations of these scores.** Because the generator reuses phrases, 38 of
60 test reviews exactly duplicate a training review. Scored separately, the 38
seen reviews had token F1 0.9947 and the 22 unseen reviews 0.9914 (exact match
0.974 vs 0.955; one error in each group). Both groups share the same
product and problem vocabulary, and 22 reviews is too few for firm
conclusions. The two non-exact answers (SYN-0371 and SYN-0081, both Bluetooth
Speaker reviews) identified the positive feature and the problem correctly but
gave unsuitable recommendations for "the bass sounds muddy" ("Increase the bass
volume" and "Extend the bass noise" instead of "Retune the low-frequency
audio"). These are real errors, not wording differences, and both involve the
same problem phrase.

**Custom inference.** A reworded Wireless Mouse review produced a correct,
grounded answer. A review outside the templates (Electric Kettle, two
problems, problem-first wording) failed: the model returned "Positive
feature: None reported", missing "Boils quickly", and recommended "Keep the lid
intact" although the review called the lid flimsy. The grounding score was
still 1.0, so that metric does not detect omissions or irrelevant advice.
Rerunning the same review with a rating of 4 returned the identical answer ("Positive feature: None reported", "Keep the lid intact"), so the low rating does not explain the missed positive feature. Candidate causes, none of them tested, are the bare opening sentence ("Boils quickly."), which differs from the "The product has..." phrasing in the training data, the two-problem structure, and the unseen product. This is a single example. Overall, high scores on the
templated test set should not be read as performance on real reviews.

## 3. Bias analysis and fairness

**Training data audit.** Regions are imbalanced: North America 36.7%, Europe
28.1%, Asia-Pacific 17.5%, Latin America 11.5%, Middle East & Africa 6.2%.
Categories are uneven (apparel 18.8%, beauty 7.7%, fitness 7.1%). Middle East &
Africa reviews have a higher average rating (3.43 vs about 2.9 to 3.0). No
loaded terms were found with the project word list, which is a weak check.

**Model results (run 2, token F1).** The largest gaps among groups with at
least 10 reviews were electronics (-0.032, n=10), Middle East & Africa (-0.029,
n=11) and rating 3 (-0.012, n=21). Five groups had fewer than 10 test reviews
(fitness 3, beauty 6, drinkware 6, Latin America 6, apparel 8) and cannot be
assessed reliably.

Inspection showed the electronics and Middle East & Africa gaps come from the
same two records (SYN-0371 and SYN-0081), both Bluetooth Speaker reviews with
the problem "the bass sounds muddy". The recommendations were real errors, not
wording differences. That phrase appears in 11 of 480 training reviews (2.3%),
close to the roughly 3.3% expected if all problem phrases were equally common,
so rarity does not explain the errors. Because both groups contain the same two
answers, the data cannot separate a category effect from a region effect, and
the cause is not established.

These gaps are small and rest on the same two answers. They are observations,
not evidence of unfairness. Groups are synthetic labels, not real demographic
evidence.

**Fairness strategies tied to these findings.**
- Stratify splits by region and category and add examples for the smallest
  groups (Middle East & Africa, fitness, beauty).
- Build a larger test set (about 30 or more reviews per group) and report
  uncertainty.
- Keep region out of the model input (done here) and check predictions do not
  shift with sentiment.
- Extend the loaded-term list and have a person review sample prompts and
  answers.
- Report each industry category separately.

## Layout

`src/sba_928/` has `step_01` to `step_06` plus helpers. Tests are in `tests/`.

