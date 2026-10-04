# riskship-rna

Data preparation and model training for a shipment-loss neural network (RNA
= "red neuronal artificial"). The model predicts `is_lost` **at shipment
creation, before dispatch**, so only information available at that moment is
used as a feature; anything recorded later (tracking status, loss cause,
cashout amount) is treated as label leakage and excluded.

## Repository layout

- `datos/` -- raw CSV/JSON exports (input).
- `unify_datos.py` -- PEP 723 script that unifies `datos/` into `dataset.parquet` (polars, own inline dependencies).
- `dataset.parquet`, `dataset_dictionary.md` -- generated dataset and its column/type dictionary. Committed, never hand-edited.
- `notebooks/rna_shipment_loss.ipynb` -- exploration, cleaning, training, and evaluation of the loss-prediction model.
- `AutogluonModels/` -- trained model artifacts (generated, gitignored).
- `export_model.py` -- exports the best per-seed predictor as one self-contained pickle into `exports/`.
- `predict_risk.py` -- scores shipments (JSON or dataset rows) with the exported pickle and flags the ones above a risk threshold.
- `load_neon.py`, `sql/shipments.sql` -- create the `shipments` table in Neon (a mirror of `dataset.parquet`) and bulk-load it; the Hermes `riskship` profile reads it from a cron job.
- `informe/` -- the project report in IEEE format (`riskship_rna.tex`, LaTeX, in Spanish) and `figuras/`, exported by the notebook. Compile with `cd informe && Rscript -e 'tinytex::pdflatex("riskship_rna.tex")'`.

## Requirements and setup

```bash
uv sync
```

Python is pinned to 3.12 (`.python-version`, `requires-python = ">=3.12,<3.14"`
in `pyproject.toml`). This is a hard constraint of `autogluon-tabular` 1.6.3,
which requires Python `<3.14` and `pandas<2.4`; do not bump either without
first checking AutoGluon's supported ranges.

## Usage

Rebuild the dataset from the raw exports:

```bash
uv run unify_datos.py
```

Open the notebook interactively:

```bash
uv run jupyter lab
```

Run it headlessly (for CI or a quick smoke test):

```bash
RNA_TIME_LIMIT=90 RNA_N_SEEDS=2 RNA_FIGURES_DIR=<output-dir>/figuras uv run jupyter nbconvert --to notebook --execute \
  notebooks/rna_shipment_loss.ipynb --output-dir <output-dir> \
  --output executed.ipynb --ExecutePreprocessor.timeout=1800
```

`RNA_TIME_LIMIT` (seconds, default 600) caps AutoGluon's training time **per
seed**, and `RNA_N_SEEDS` (default 5) sets how many seeds are trained; keep
both low for a smoke test. Every figure is also exported at 300 dpi to
`RNA_FIGURES_DIR` (default `informe/figuras/`, which the report includes), so
a smoke test must point it somewhere else or it overwrites the report's
figures.

The notebook is committed executed, so its figures and metrics can be read
directly on GitHub. Those results come from a full run with the default time
limit and seed count.

Export the best trained model as a single self-contained pickle:

```bash
uv run export_model.py                     # -> exports/riskship_rna_model.pickle (+ .json)
uv run export_model.py --seed 46 --out model.pickle
```

The script reads the per-seed predictors the notebook left under
`AutogluonModels/`, picks the seed with the best **validation** PR-AUC (the same
rule the notebook uses for its reference model), loads every network into
memory and pickles the whole predictor, so the file predicts without the
model directory. It then reloads the pickle in a fresh interpreter with that
directory hidden and requires identical scores. The sidecar `.json` records
the seed, feature list and library versions: unpickling needs the same
`autogluon.tabular`, `torch`, `fastai` and `pandas` versions. Use
`export_model.prepare_features` to derive `day_of_week` and fill `TIPO_BULKY`
from a raw export before calling `predict_proba`.

Score shipments with the exported model from the command line (the Hermes
agent profile `riskship` ships its own copy of the pickle and of this script in
its `riesgo-envio` skill; copy `exports/` there after re-exporting):

```bash
uv run predict_risk.py shipment.json              # JSON object or list of objects
uv run predict_risk.py --shipment-id 46411999224  # a row from dataset.parquet
uv run predict_risk.py --sample 3 --threshold 0.65
```

It prints one JSON document with the loss probability per shipment and an
`at_risk` flag against the threshold (default 0.65). The probability is on the
scale of the case-control sample, not of production prevalence (see "How to
read these numbers" below).

Mirror the dataset into Neon for the Hermes agent's hourly cron (`riesgo-envio-neon`):

```bash
NEON_DATABASE_URL='postgresql://...' uv run load_neon.py   # role with write access
```

`sql/shipments.sql` has the same 58 columns as the parquet (quoted, case preserved)
plus `id` (the cron's watermark) and `created_at`. `load_neon.py` is a PEP 723
script (`pyarrow`, `psycopg`), so `uv run` resolves it outside the project env.
Regenerate the DDL if the dataset's columns change.

## Data

`dataset.parquet` holds 262,317 rows x 58 columns, one row per shipment
**item** (not per shipment -- a shipment with several items spans several
rows; see `dataset_dictionary.md` for the full column list). The label is
`is_lost` (Int, 0/1), about 24% positive before cleaning. After the notebook's
cleaning steps (missingness-artifact rows dropped, restricted to the months
where both classes are observed and the label has matured), the usable
dataset is smaller and closer to a 37/63 split; see the notebook for the exact numbers, computed from the
data rather than hardcoded here.

## Key modeling decisions

| Decision | Why |
|---|---|
| Prediction moment: at creation, before dispatch | Only columns known at that moment may be features; post-creation tracking/status/loss-cause columns are dropped. |
| Leakage columns dropped (`CAUSA_BPP`, `CLASSIFICATION_LM`, `F_CLASIFICATION*`, `BPP_CASHOUT_USD`, `status_*`/`date_status_1..10`, ...) | Non-null in ~100% of lost rows and 0% of non-lost rows, or a post-dispatch timeline -- pure restatements of the label. |
| `LG_REGION` dropped | The same facility is exported under different region names depending on the row's class, so the column encodes the label through an inconsistent taxonomy; `SHP_LG_FACILITY_ID` carries the geographic signal safely instead. |
| Temporal window restricted to months where both classes exist and the label has matured | Recent months have zero lost labels because the label has not matured yet (extraction artifact), not because risk dropped to zero. A trailing month can hold a few positives and still be immature (2026-04: ~1% lost vs 31-47% before), so trailing months below `MATURITY_RATIO` x the median monthly lost rate are dropped too. |
| No absolute date features, only day-of-week | Using year/month would let the model re-learn the sampling artifact above. |
| Rows with null core numerics or null `date_status_0` dropped | These nulls are ~100% lost -- an artifact of how the two classes were extracted, not a real relationship. |
| Temporal split by month of `date_status_0` (oldest months train, latest months validate and test) | No future information reaches training, and the test answers the operational question: how well does the model rank shipments that did not exist yet. Months are chosen from the data (`MIN_HOLDOUT_SHARE`). A shipment can span multiple item rows, so the notebook asserts no `SHIPMENT_ID` crosses partitions. |
| Several seeds, fixed partition | Training repeats with `N_SEEDS` seeds on the same split and test set, so metrics are reported as mean and standard deviation instead of a single run. |
| Ranking metrics (`recall_en_K`, `precision_en_K`) next to PR-AUC / ROC-AUC | The model is a prioritization tool: what matters is how many real losses fall inside the share of shipments a team can actually review. |
| AutoGluon restricted to `NN_TORCH` + `FASTAI` | The goal is specifically a neural network, not AutoGluon's default tree ensemble. |
| `eval_metric="average_precision"` | The label is imbalanced (~24% positive raw, about a third after cleaning); accuracy is misleadingly high for a model that just predicts the majority class. |

## Experimental protocol and results

These are the figures the project report must quote. They come from the
committed notebook run (default `RNA_TIME_LIMIT=600`, `RNA_N_SEEDS=5`) and are
computed on the held-out test month. If the notebook is re-run and any number
changes, update this section and `CLAUDE.md` in the same commit.

### Protocol

| Item | Value |
|---|---|
| Model | AutoGluon `TabularPredictor`, neural networks only: `NN_TORCH` (PyTorch MLP) + `FASTAI`, combined by a weighted ensemble; default hyperparameters |
| Optimized metric | `average_precision` (PR-AUC) |
| Clean dataset | 165,002 rows (items), 2025-09 to 2026-03, ~37% positive |
| Split | Temporal, by month of `date_status_0`; months chosen from the data (`MIN_HOLDOUT_SHARE = 0.10`) |
| Seeds | 5 seeds: 42, 43, 44, 45, 46 (`RANDOM_STATE = 42`, consecutive) |
| What a seed changes | Only the networks' training (weight init, batch order, dropout): `seed_value` for `NN_TORCH`, `random_seed` for `FASTAI`. The partition and the test set are identical across seeds |
| Dispersion | Sample standard deviation across the 5 seeds (`ddof = 1`) |
| Classification threshold | 0.5 on the predicted probability, positive class = lost |
| Reference model | Seed 46, chosen by best PR-AUC on **validation** (0.7671), never by test score |
| Training cost | ~100 s per seed (78-128 s); wall-clock time, so it varies between runs while every metric below reproduces exactly. Early stopping ends well before the 600 s cap |

| Partition | Months | Rows | Share | Shipments | Positive rate |
|---|---|---|---|---|---|
| Train | 2025-09 to 2026-01 | 125,082 | 75.8% | 112,303 | 39.0% |
| Validation | 2026-02 | 18,831 | 11.4% | 17,297 | 31.6% |
| Test | 2026-03 | 21,089 | 12.8% | 19,506 | 31.2% |

### Test metrics across the 5 seeds

| Metric | Mean | Std |
|---|---|---|
| PR-AUC (average precision) | 0.7496 | 0.0020 |
| ROC-AUC | 0.8241 | 0.0020 |
| Recall (lost, threshold 0.5) | 0.5339 | 0.0136 |
| Precision (lost, threshold 0.5) | 0.7568 | 0.0157 |
| F1 (lost, threshold 0.5) | 0.6259 | 0.0044 |
| False positive rate (threshold 0.5) | 0.0781 | 0.0089 |
| PR-AUC on validation | 0.7644 | 0.0019 |

The PR-AUC baseline is the test positive rate, 0.312 (a model that ranks at
random). Dispersion across seeds is small, so the result does not depend on
the seed.

### Ranking metrics (prioritization)

`recall_en_K` is the share of all real losses found inside the top K% of test
rows by score; `precision_en_K` is the share of that top K% that really was
lost. Mean and std across the 5 seeds:

| Reviewed share (K) | Rows reviewed | Recall@K | Precision@K | Max possible Recall@K |
|---|---|---|---|---|
| 1% | 211 | 0.0321 (std 0.0000) | 1.0000 (std 0.0000) | 0.032 |
| 5% | 1,054 | 0.1581 (std 0.0006) | 0.9873 (std 0.0038) | 0.160 |
| 10% | 2,109 | 0.3019 (std 0.0020) | 0.9421 (std 0.0061) | 0.320 |
| 20% | 4,218 | 0.5060 (std 0.0024) | 0.7895 (std 0.0037) | 0.641 |

Reviewing the top 20% captures about 51% of the losses, 2.5 times better than
reviewing at random. Recall@K is capped by prevalence: with 31.2% positives,
even a perfect ranking cannot capture more than K / 0.312 of the losses.

Reference model (seed 46), confusion matrix on test at threshold 0.5:

| | Predicted not lost | Predicted lost |
|---|---|---|
| Actually not lost | 13,425 | 1,082 |
| Actually lost | 3,094 | 3,488 |

### Projection to the real prevalence

Reference model (seed 46), projected with `REAL_PREVALENCE = 0.001` (the
team's estimate of the real loss rate, not measured in this dataset). For any
score threshold: reviewed share = pi x recall + (1 - pi) x FPR, and real
precision = pi x recall / reviewed share.

| Minimum score | Recall | False positive rate | Shipments reviewed | Reviewed per real loss | Lift over random |
|---|---|---|---|---|---|
| 0.91 | 0.159 | 0.0008 | 0.09% | 6 | 173x |
| 0.78 | 0.299 | 0.0097 | 1.00% | 33 | 30x |
| 0.53 | 0.507 | 0.0606 | 6.10% | 120 | 8.3x |
| 0.50 | 0.530 | 0.0746 | 7.50% | 142 | 7.1x |

Headline for the report: reviewing the top 1% of real shipments would capture
about 30% of the losses, one real loss per 33 shipments reviewed. The
projection assumes the sampled negatives represent the real ones, which the
extraction artifacts put in doubt, and the rows with a near-zero false
positive rate are the most exposed to the suspected `DOM_DOMAIN_ID` artifact.
It is a projection, not a measurement.

### How to read these numbers

- **The sample is case-control, on purpose.** Real loss is around 0.1%, so the
  export keeps the losses and only a fraction of the healthy shipments
  (~31-39% positive here). That design is valid. Under it, only recall and the
  false positive rate carry over to production unchanged (each is computed
  inside one class), so ROC-AUC is valid as is. Precision, PR-AUC, Precision@K,
  Recall@K and lift over random all depend on prevalence: they describe
  ranking quality **inside this sample** and none is a production figure (the
  lift ceiling is 1/prevalence: ~3 here, ~1000 in production). See
  "Projection to the real prevalence" below.
- **The artifacts come from extracting the two classes differently**, not from
  the class ratio: `LG_REGION` naming, the class-revealing nulls,
  `DATEPARAMETER` and probably `DOM_DOMAIN_ID`.
- **Precision@1% = 1.000 is suspicious, not a triumph.** Six `DOM_DOMAIN_ID`
  values are 100% lost with 200-834 rows each (e.g.
  `CELLPHONE_CASES_AND_COVERS`), which looks like the same class-keyed taxonomy
  artifact found in `LG_REGION`. It is an open issue and likely inflates the
  very top of the ranking.
- **Not comparable with the earlier run.** A previous version used a random
  group split and kept the immature month 2026-04; it scored ROC-AUC 0.848 /
  PR-AUC 0.800. The drop to 0.824 / 0.750 is the cost of an honest temporal
  test, not a regression.
- **Item level.** Rows are shipment items, so all metrics are per item, not
  per shipment.
- **One test month.** The test set is a single month; it measures
  generalization to the next month, not across seasons.

## How the data should be re-extracted

The class ratio is not the problem; extracting the two classes through
different paths is. A future export should:

1. Use **one query for both classes**: same population, filters, source tables
   and creation-date window. Compute the label afterwards on that single set.
2. **Downsample only the negatives, at random, at a recorded rate** `r` (keep
   every loss in the window). Without `r` the predicted probabilities cannot be
   corrected to the real scale (real odds = predicted odds x `r`).
3. Include only shipments whose **label has matured**: created more than N days
   ago, N being the time by which ~99% of losses are declared. Apply it to both
   classes.
4. Take attributes **as they were at shipment creation**, not their current
   value.
5. Sample **by shipment**, not by item.

See `CLAUDE.md` for the full list of constraints a future change must not break.
