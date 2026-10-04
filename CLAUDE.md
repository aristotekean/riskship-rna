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

```bash
uv run export_model.py   # best-validation seed from AutogluonModels/ -> exports/riskship_rna_model.pickle + .json
```

`export_model.py` runs in the project environment (it needs AutoGluon). An AutoGluon predictor is a directory, not a file: it calls `persist(models="all")` so the pickle carries the networks and predicts without `AutogluonModels/`, then verifies the round trip in a fresh interpreter with the source directory temporarily renamed. The pickle is only loadable with the library versions recorded in the sidecar JSON.

```bash
uv run predict_risk.py --shipment-id 46411999224   # or a JSON file / '-' for stdin; --threshold 0.65 default
```

`predict_risk.py` is the inference entry point consumed by the Hermes agent profile at `~/.hermes/profiles/riskship` (skill `riesgo-envio`, script `skills/riesgo-envio/scripts/evaluar_riesgo.sh`), which creates a Linear issue per shipment above the threshold. It imports `prepare_features` from `export_model.py`, so keep the two derived features (`TIPO_BULKY` fill, `day_of_week`) in sync with the notebook's `preprocess`. Scores are on the case-control sample scale, never production probabilities.

Run the notebook headlessly (smoke test or CI):

```bash
RNA_TIME_LIMIT=90 RNA_N_SEEDS=2 RNA_FIGURES_DIR=<output-dir>/figuras uv run jupyter nbconvert --to notebook --execute \
  notebooks/rna_shipment_loss.ipynb --output-dir <output-dir> \
  --output executed.ipynb --ExecutePreprocessor.timeout=1800
```

`RNA_TIME_LIMIT` (seconds, default 600) caps AutoGluon's `fit` time inside the notebook **per seed**; `RNA_N_SEEDS` (default 5) sets how many seeds are trained. `finish_figure` exports every figure at 300 dpi to `RNA_FIGURES_DIR` (default `../informe/figuras`, relative to `notebooks/`), which is what the report includes -- a smoke test must override it, or it replaces the report's figures with smoke-run ones. A smoke test must write its executed copy to an output directory outside `notebooks/` (`--output-dir` above): the committed notebook holds the results of a full run and a 90-second run must not replace them.

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
- **Label maturity / temporal sampling bias**: lost rate by month of `DATEPARAMETER` collapses to exactly 0 for the most recent months -- the label has not matured yet for recent shipments, not because risk dropped. Training data must be restricted to months (of `date_status_0`) where both classes are observed, computed from the data, never hardcoded. "Both classes observed" is not enough on its own: 2026-04 has both classes but ~1% lost against 31-47% in earlier months, so `preprocess` also drops trailing months whose lost rate is below `MATURITY_RATIO` x the median monthly rate. No absolute date feature (year/month/day-of-month) may be used, only day-of-week of `date_status_0` -- an absolute date would let the model re-learn this artifact.
- **Missingness artifacts**: rows with null `PESO_TOTAL_KG`/`DIM_MAX_CM`/`VALOR_TOTAL_USD`/`CANTIDAD_ITEMS`/`date_status_0` are ~100% lost -- an extraction artifact, not signal -- and are dropped rather than imputed.
- **`LG_REGION` dropped (taxonomy artifact)**: the same facility is exported under more than one region name, and the name used depends on whether the row is lost or not -- the column encodes the label, not geography. Use `SHP_LG_FACILITY_ID` for geographic signal instead. The notebook's "purity audit" cell (section 3) is the general check for this class of problem: it flags any kept categorical where a large share of rows sit in a value with a near-0%/near-100% lost rate.
- **Temporal split, no shipment across partitions**: train/val/test are split by month of `date_status_0` -- oldest months train, the latest months validate and test, with the months chosen from the data (`MIN_HOLDOUT_SHARE`), never hardcoded. The month is only a split key, never a feature. A row is one item, not one shipment (see Dataset shape), so no `SHIPMENT_ID` may appear in two partitions; all rows of a shipment share one `date_status_0`, which makes the temporal split satisfy this, and the notebook asserts it. Any other split (e.g. a random one) must group by `SHIPMENT_ID`.
- **Seeds vary the model, not the split**: training repeats over `SEEDS` on the same partition and test set, and results are reported as mean and standard deviation. The detailed evaluation uses the seed with the best **validation** score; never pick it by test score.
- **NN-only hyperparameters**: `TabularPredictor.fit` is restricted to `{"NN_TORCH": {}, "FASTAI": {}}` -- the goal is specifically a neural network ("RNA"), not AutoGluon's default tree ensemble.
- **Metric**: `eval_metric="average_precision"`, not accuracy -- the label is imbalanced enough that a majority-class model already scores well on accuracy. The model is a prioritization tool, so the notebook also reports `recall_en_K`/`precision_en_K` (share of real losses inside the top K% by score). The sample is case-control on purpose (real loss is ~0.1%; the export keeps the losses and a fraction of the healthy shipments). Under that design only recall and the false positive rate carry over to production unchanged, so ROC-AUC is valid as is; precision, PR-AUC, `precision_en_K`, `recall_en_K` and lift over random all depend on prevalence and describe ranking quality inside the sample only. Never call lift "comparable with production" (its ceiling is 1/prevalence: ~3 in the sample, ~1000 in production). Production figures come only from the notebook's projection (`project_to_prevalence`, `REAL_PREVALENCE`), which assumes the sampled negatives represent the real ones -- exactly what the extraction artifacts put in doubt.
- **Re-extraction rules**: the class ratio is fine; extracting the two classes through different paths is what created the artifacts (`LG_REGION`, class-revealing nulls, `DATEPARAMETER`, probably `DOM_DOMAIN_ID`). A new export must use one query for both classes (same population, filters, tables, creation-date window), downsample only negatives at random at a **recorded** rate, include only shipments with a matured label, take attributes as of creation, and sample by shipment. `README.md` has the full list.

## Reported results

The project report quotes these numbers, so they are recorded here and in `README.md` ("Experimental protocol and results", which has the full tables). They come from the committed notebook run with the defaults (`RNA_TIME_LIMIT=600`, `RNA_N_SEEDS=5`). **If a re-run changes any of them, update the notebook, `README.md`, and this section in the same commit** -- never leave the docs quoting numbers the committed notebook no longer shows.

- **Seeds**: 5 seeds, 42-46 (`SEEDS = [RANDOM_STATE + i for i in range(N_SEEDS)]`). A seed only changes network training: `seed_value` for `NN_TORCH`, `random_seed` for `FASTAI` (AutoGluon names them differently per model, and defaults both to 0 if unset). The partition and test set are identical across seeds. Dispersion is the sample standard deviation (`ddof=1`). Fewer than 5 seeds is a smoke test, not a reportable result.
- **Partition**: train 2025-09..2026-01 (125,082 rows, 39.0% positive), validation 2026-02 (18,831 rows, 31.6%), test 2026-03 (21,089 rows, 31.2%). Clean dataset: 165,002 rows.
- **Test metrics, mean (std) over 5 seeds**: PR-AUC 0.7496 (0.0020), ROC-AUC 0.8241 (0.0020), recall 0.5339 (0.0136), precision 0.7568 (0.0157), F1 0.6259 (0.0044) -- the last three for the lost class at threshold 0.5. Validation PR-AUC 0.7644 (0.0019). PR-AUC random baseline = test positive rate, 0.312.
- **Ranking metrics, mean (std)**: Recall@1% 0.0321 (0.0000), @5% 0.1581 (0.0006), @10% 0.3019 (0.0020), @20% 0.5060 (0.0024). Precision@1% 1.0000 (0.0000), @5% 0.9873 (0.0038), @10% 0.9421 (0.0061), @20% 0.7895 (0.0037). Recall@K is capped at K / 0.312 by prevalence.
- **Projection to real prevalence** (`REAL_PREVALENCE = 0.001`, the team's estimate, not measured here; reference model): minimum score 0.78 -> 1.00% of shipments reviewed, recall 0.299, one real loss per 33 reviewed, 30x lift; score 0.50 -> 7.50% reviewed, recall 0.530, one per 142, 7.1x lift; score 0.91 -> 0.09% reviewed, recall 0.159, one per 6, 173x. False positive rate at 0.5, mean (std) over seeds: 0.0781 (0.0089). These are the only production-scale figures the report may quote, always labelled as a projection.
- **Reference model**: seed 46, picked by best validation PR-AUC (0.7671). Test confusion matrix at 0.5: TN 13,425, FP 1,082, FN 3,094, TP 3,488. Training takes ~100 s per seed (early stopping), far below the 600 s cap; that wall-clock time varies between runs, while every metric above reproduced exactly across two full runs.
- **Not comparable with the pre-temporal-split run** (random group split, immature 2026-04 kept): ROC-AUC 0.848 / PR-AUC 0.800. The lower current numbers are the honest ones; do not quote the old ones.
- **Open issue -- suspected `DOM_DOMAIN_ID` taxonomy artifact**: six domain values are 100% lost with 200-834 rows each (`CELLPHONE_CASES_AND_COVERS`, `CAT_AND_DOG_FOODS`, `CHRISTMAS_AND_DECORATIVE_STRINGS_LIGHTS`, `FLOOR_CEILING_AND_WALL_LIGHTS`, `CHARMS`, `VEHICLE_SPEAKERS`). This mirrors the `LG_REGION` problem and likely explains Precision@1% = 1.000. Investigate before presenting top-of-ranking precision as real signal.

## Report

`informe/riskship_rna.tex` is the project report (IEEEtran conference, neutral Spanish, decimal comma). Compile with `cd informe && Rscript -e 'tinytex::pdflatex("riskship_rna.tex")'` (TinyTeX through R; there is no system LaTeX). Its figures in `informe/figuras/` are generated by the notebook -- never edit them by hand. Every number in it comes from the committed notebook run, so the same-commit rule in "Reported results" covers the report too.

## Agent shell gotchas

- **Never run `eza` without an explicit path** (`eza .`, `eza -l informe/`). With no path argument, `eza` (v0.23.5) reads file names from stdin whenever stdin is not a terminal; in an agent's non-interactive shell stdin is a pipe that never closes, so a bare `eza` or `eza -l` hangs forever. Verified: `eza` hangs, `eza .` returns at once, `eza < /dev/null` returns empty. It is not specific to any folder.
- **A hung trailing command looks like a slow build.** Compiling the report takes seconds. Twice a bare `eza -l` chained after `tinytex::pdflatex(...)` kept the command open for 10 minutes and was misdiagnosed as TinyTeX installing packages. Do not chain a listing after a build; if a command seems slow, check whether its real work already finished (output file timestamp, `ps`) before blaming the build.

## Generated directories

`AutogluonModels/` (trained model artifacts) is gitignored. `notebooks/*.ipynb` are committed **executed, with outputs**, so the figures and metrics render on GitHub. Commit a notebook only after a clean top-to-bottom run with the default `RNA_TIME_LIMIT` and `RNA_N_SEEDS`; do not commit partial or out-of-order executions.
