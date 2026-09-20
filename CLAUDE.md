# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Data preparation and model training for a shipment-risk neural network (RNA = "red neuronal artificial"). Raw shipment exports in `datos/` are unified into `dataset.parquet` (`unify_datos.py`), and `notebooks/rna_shipment_loss.ipynb` cleans that dataset and trains a model predicting `is_lost` **at shipment creation, before dispatch** -- only information known at that moment may be a feature. Tests do not exist yet.

## Commands

```bash
uv run unify_datos.py                       # datos/ -> dataset.parquet + dataset_dictionary.md
uv run unify_datos.py datos -o dataset.csv  # CSV output instead
```

`unify_datos.py` is a PEP 723 single-file script: its dependencies (`polars`, Python >= 3.13) are declared in the inline `# /// script` header, so `uv run` resolves them in isolation and ignores the project environment.

```bash
uv sync                # create .venv from pyproject.toml + uv.lock
uv run jupyter lab     # exploration notebooks
```

Run the notebook headlessly (smoke test or CI):

```bash
RNA_TIME_LIMIT=90 uv run jupyter nbconvert --to notebook --execute \
  notebooks/rna_shipment_loss.ipynb --output-dir <output-dir> \
  --output executed.ipynb --ExecutePreprocessor.timeout=1800
```

`RNA_TIME_LIMIT` (seconds, default 600) caps AutoGluon's `fit` time inside the notebook. A smoke test must write its executed copy to an output directory outside `notebooks/` (`--output-dir` above): the committed notebook holds the results of a full run and a 90-second run must not replace them.

Never use `jupyter nbconvert --clear-output` to make a stripped copy: it implies `--inplace`, ignores `--output-dir`, and erases the outputs of the working notebook.

`pyproject.toml` defines the project's uv environment (`autogluon-tabular[fastai]`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, plus a `dev` group with `jupyterlab` and `ipykernel`). There is no lint or test setup.

## Environment

Python is pinned to 3.12 (`.python-version`, `requires-python = ">=3.12,<3.14"`). This is required by `autogluon-tabular` 1.6.3, which needs Python `<3.14` and `pandas<2.4`. Do not bump the Python version or `pandas` without first checking AutoGluon's current supported ranges -- the project was previously on Python 3.14 / pandas 3, where AutoGluon could not install at all.

## Data flow

`datos/*.csv|*.json` -> `unify_datos.py` -> `dataset.parquet` + `dataset_dictionary.md`

- `dataset.parquet` and `dataset_dictionary.md` are **generated, but committed**. Never edit them by hand; re-run the script after changing it or the raw files, and commit the regenerated outputs together.
- `datos/` contains overlapping exports: the same job exported as both CSV and JSON, and browser re-downloads such as `job_X (1).csv`. Overlap is expected and handled by deduplication, not by removing files.

## How unification works

The order of steps in `unify()` matters:

1. Every file is read with **all columns as strings** (`read_raw`), so heterogeneous files concatenate safely (`how="diagonal"` fills missing columns with null). JSON is parsed with the stdlib `json` module because `pl.read_json` does not preserve key order.
2. Values are stripped and `""` becomes null, because CSV yields null for empty cells while JSON yields `""`.
3. Types are inferred per column over the **whole** unified column (`infer_column`), trying Int64 -> Float64 -> Date -> Datetime and falling back to String. Columns whose name contains `_ID` are deliberately kept as strings (leading zeros, joins).
4. Rows are deduplicated on all data columns except `source_file` and `batch_num` (`DEDUP_IGNORED_COLUMNS`): those are export bookkeeping, so the same shipment pulled in two batches counts as one row.

Because inference needs every value to fit, a single dirty value keeps a whole column as String. Note that an all-null column (e.g. `CANT_BULTOS`, 100% null in the current dictionary) is a different case: `infer_column` returns an empty/all-null column untouched by the `values.is_empty()` guard, so it stays String too, but not because of a dirty value. Check `dataset_dictionary.md` after regenerating to spot type regressions.

## Dataset shape

~262k rows x 58 columns. Rows are **not** unique per shipment: a row is one **item** of a shipment, not the shipment itself. ~246k distinct `SHIPMENT_ID` values mean ~8.8k shipments span more than one row (their items differ in `ITE_ITEM_ID`, `ITE_ITEM_DESC`, `BRAND_NAME`, category, etc.); the label never disagrees across a shipment's rows. Beyond the shipment/item attributes, note:

- `status_0..10` / `date_status_0..10`: the shipment's status timeline, stored wide.
- `is_lost` (Int64 0/1): the loss label, imbalanced (~24% are 1).
- `source_file`, `batch_num`: provenance only, not features.

Column names are a mix of Spanish and English and come from the upstream export; keep them as-is.

## Modeling constraints

Non-obvious rules the notebook (and any future inference script) must keep. Full reasoning and evidence live in `notebooks/rna_shipment_loss.ipynb`; this is the tight version so a future change does not silently reintroduce a problem already solved there.

- **Prediction moment**: the model predicts `is_lost` at shipment creation, before dispatch. Any column only known later is leakage, regardless of how predictive it looks.
- **Leakage column families**: `CAUSA_BPP`/`L1_CAUSA_BPP`/`L2_CAUSA_BPP`/`DATE_BPP`/`BPP_CASHOUT_USD`/`CLASSIFICATION_LM`/`SUBSTATUS_CLASIFICATION`/`F_CLASIFICATION`/`F_CLASIFICATION_STD` are non-null in ~100% of lost rows and 0% of non-lost rows -- pure label restatement. `status_0..10`/`date_status_1..10` are the post-creation tracking timeline -- never available at creation time.
- **`DATEPARAMETER` vs `date_status_0`**: `DATEPARAMETER` equals `date_status_0`'s date for non-lost rows but runs ~9 days later on average for lost rows -- it silently encodes the label and must be dropped. `date_status_0` is the reliable shipment-start timestamp.
- **Label maturity / temporal sampling bias**: lost rate by month of `DATEPARAMETER` collapses to exactly 0 for the most recent months -- the label has not matured yet for recent shipments, not because risk dropped. Training data must be restricted to months (of `date_status_0`) where both classes are observed, computed from the data, never hardcoded. No absolute date feature (year/month/day-of-month) may be used, only day-of-week of `date_status_0` -- an absolute date would let the model re-learn this artifact.
- **Missingness artifacts**: rows with null `PESO_TOTAL_KG`/`DIM_MAX_CM`/`VALOR_TOTAL_USD`/`CANTIDAD_ITEMS`/`date_status_0` are ~100% lost -- an extraction artifact, not signal -- and are dropped rather than imputed.
- **`LG_REGION` dropped (taxonomy artifact)**: the same facility is exported under more than one region name, and the name used depends on whether the row is lost or not -- the column encodes the label, not geography. Use `SHP_LG_FACILITY_ID` for geographic signal instead. The notebook's "purity audit" cell (section 3) is the general check for this class of problem: it flags any kept categorical where a large share of rows sit in a value with a near-0%/near-100% lost rate.
- **Group split by `SHIPMENT_ID`**: a row is one item, not one shipment (see Dataset shape). Every train/val/test split must group by `SHIPMENT_ID` (`GroupShuffleSplit`), or the same shipment can leak across splits.
- **NN-only hyperparameters**: `TabularPredictor.fit` is restricted to `{"NN_TORCH": {}, "FASTAI": {}}` -- the goal is specifically a neural network ("RNA"), not AutoGluon's default tree ensemble.
- **Metric**: `eval_metric="average_precision"`, not accuracy -- the label is imbalanced enough that a majority-class model already scores well on accuracy.

## Generated directories

`AutogluonModels/` (trained model artifacts) is gitignored. `notebooks/*.ipynb` are committed **executed, with outputs**, so the figures and metrics render on GitHub. Commit a notebook only after a clean top-to-bottom run with the default `RNA_TIME_LIMIT`; do not commit partial or out-of-order executions.
