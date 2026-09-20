# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Data preparation for a shipment-risk neural network (RNA = "red neuronal artificial"). The repo currently holds only the ingestion step: raw shipment exports in `datos/` are unified into `dataset.parquet`. Preprocessing, training, and tests do not exist yet.

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

`pyproject.toml` defines a separate uv project (Python >= 3.14, `pandas`, `pyarrow`, plus a `dev` group with `jupyterlab` and `ipykernel`) used only for notebook exploration. There is no lint or test setup.

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

Because inference needs every value to fit, a single dirty value keeps a whole column as String (e.g. `CANT_BULTOS` in the current dictionary). Check `dataset_dictionary.md` after regenerating to spot type regressions.

## Dataset shape

~262k rows x 58 columns. Rows are **not** unique per shipment: there are ~246k distinct `SHIPMENT_ID` values, so a shipment can span several rows that differ in some data column. Beyond the shipment/item attributes, note:

- `status_0..10` / `date_status_0..10`: the shipment's status timeline, stored wide.
- `is_lost` (Int64 0/1): the loss label, imbalanced (~24% are 1).
- `source_file`, `batch_num`: provenance only, not features.

Column names are a mix of Spanish and English and come from the upstream export; keep them as-is.
