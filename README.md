# VLMHallucinationAudit

Sentence-level hallucination risk classifier for VLM-generated medical
reports.

## Scope, disclosed upfront

This project analyzes **real VLM-generated outputs about real images**, with
expert-verified hallucination labels — but it does **not** train on the raw
image pixels itself. The source images (Kvasir-v2 gastrointestinal images)
are hosted at datasets.simula.no, which the build environment cannot reach;
only the text annotations, hosted on GitHub, were downloadable. So this is a
genuine hallucination-detection project grounded in real multimodal model
outputs, honestly scoped as text-only rather than framed as a full
image+text joint model. A production version would add the image as a
second input to a joint classifier — noted as the clear next step, not
hidden.

## Data

Real, public dataset: [Gut-VLM](https://github.com/bhattarailab/Hallucination-Aware-VLM)
(Khanal et al., MICCAI 2025, early accept / top 9% of submissions). GPT-4o
generated 1,816 descriptive medical reports from real Kvasir-v2 endoscopy
images; medical experts then labeled each sentence `correct` or `incorrect`
(hallucinated) and supplied corrections. Official train/test split used as
published (1,450 / 366 reports → 5,324 / 1,330 sentences after splitting
reports into sentences).

Class balance: **27.3%** of sentences are hallucinated in the test set
(359/1,330) — real and imbalanced, not synthetically balanced.

## Result

| Model | ROC-AUC | F1 (hallucinated class) |
|---|---|---|
| Majority-class baseline | 0.500 | 0.000 |
| TF-IDF + Logistic Regression (text-only) | **0.799** | **0.587** |

The text-only model clears the majority baseline by a wide margin — a real
and slightly surprising finding, since a sentence's hallucination status is,
in principle, only fully verifiable against the image. The explanation is
in the interpretability analysis below: this VLM (GPT-4o on GI endoscopy
images) hallucinates in **patterned, not random, ways**, and those patterns
leave a lexical fingerprint even without the image.

## Interpretability

Top n-grams driving hallucination predictions (logistic regression
coefficients):

**Predicts hallucination:** `brown`, `yellowish`, `dark`, `black`, `light blue`,
`instrument`, `distinctly colored`

**Predicts correct/grounded:** `normal`, `blue`, `pink`, `healthy`, `red`,
`no instruments`, `no foreign`

Reading: this VLM systematically over-hallucinates specific color/attribute
claims (browns, yellows, darker tones, describing an instrument as present)
while claims using `normal`, `blue`, `pink`, `red`, or negations (`no
instruments`, `no foreign [bodies]`) are much more often grounded. This
matches a well-documented VLM failure mode — models tend to guess a
plausible-sounding specific attribute when uncertain rather than abstaining
or hedging. Full ranked lists in `results/interpretability.json`.

## Serving layer

`src/04_api.py` serves a `/score` endpoint scoring hallucination risk for
any sentence, verified live:

- "The anatomical landmarks are brown colored." → risk 0.710, flagged for review
- "No polyps or signs of infection are visible." → risk 0.336, not flagged

Framed explicitly as a **fast text-only pre-filter** in front of a full
image-grounded verifier, not a substitute for one — consistent with the
scope disclosure above.

## Files

- `src/01_extract_sentences.py` — parse sentence-level labels from the raw annotations, official split
- `src/02_textonly_baseline.py` — TF-IDF + Logistic Regression vs. majority-class baseline
- `src/03_interpretability.py` — top predictive n-grams for hallucination vs. correct
- `src/04_api.py` — FastAPI serving layer
- `results/textonly_classifier_metrics.json`, `results/interpretability.json` — raw metrics
