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
RNA_TIME_LIMIT=90 uv run jupyter nbconvert --to notebook --execute \
  notebooks/rna_shipment_loss.ipynb --output-dir <output-dir> \
  --output executed.ipynb --ExecutePreprocessor.timeout=1800
```

`RNA_TIME_LIMIT` (seconds, default 600) caps AutoGluon's training time; keep
it low for a smoke test and raise it for a real training run.

The notebook is committed executed, so its figures and metrics can be read
directly on GitHub. Those results come from a full run with the default time
limit.

## Data

`dataset.parquet` holds 262,317 rows x 58 columns, one row per shipment
**item** (not per shipment -- a shipment with several items spans several
rows; see `dataset_dictionary.md` for the full column list). The label is
`is_lost` (Int, 0/1), about 24% positive before cleaning. After the notebook's
cleaning steps (missingness-artifact rows dropped, restricted to the months
where both classes are observed), the usable dataset is smaller and closer
to a 34/66 split; see the notebook for the exact numbers, computed from the
data rather than hardcoded here.

## Key modeling decisions

| Decision | Why |
|---|---|
| Prediction moment: at creation, before dispatch | Only columns known at that moment may be features; post-creation tracking/status/loss-cause columns are dropped. |
| Leakage columns dropped (`CAUSA_BPP`, `CLASSIFICATION_LM`, `F_CLASIFICATION*`, `BPP_CASHOUT_USD`, `status_*`/`date_status_1..10`, ...) | Non-null in ~100% of lost rows and 0% of non-lost rows, or a post-dispatch timeline -- pure restatements of the label. |
| `LG_REGION` dropped | The same facility is exported under different region names depending on the row's class, so the column encodes the label through an inconsistent taxonomy; `SHP_LG_FACILITY_ID` carries the geographic signal safely instead. |
| Temporal window restricted to months where both classes exist | Recent months have zero lost labels because the label has not matured yet (extraction artifact), not because risk dropped to zero. |
| No absolute date features, only day-of-week | Using year/month would let the model re-learn the sampling artifact above. |
| Rows with null core numerics or null `date_status_0` dropped | These nulls are ~100% lost -- an artifact of how the two classes were extracted, not a real relationship. |
| Group split by `SHIPMENT_ID` (`GroupShuffleSplit`, ~70/15/15) | A shipment can span multiple item rows; a random row split would leak a shipment across train/val/test. |
| AutoGluon restricted to `NN_TORCH` + `FASTAI` | The goal is specifically a neural network, not AutoGluon's default tree ensemble. |
| `eval_metric="average_precision"` | The label is imbalanced (~24-34% positive); accuracy is misleadingly high for a model that just predicts the majority class. |

See `CLAUDE.md` for the full list of constraints a future change must not break.
