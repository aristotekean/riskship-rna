"""Export the best trained predictor as a single self-contained pickle.

The notebook trains one AutoGluon predictor per seed under
``AutogluonModels/rna_shipment_loss/seed_<seed>``. AutoGluon stores a predictor
as a directory: ``predictor.pkl`` there is only a pointer and the weights live in
``models/``. This script picks the seed with the best validation PR-AUC (never
the test score), loads every model into memory and pickles the whole predictor
into one file that predicts without the directory.

Usage (inside the project environment)::

    uv run export_model.py                    # -> exports/riskship_rna_model.pickle
    uv run export_model.py --seed 46          # force a seed
    uv run export_model.py --out path.pickle  # custom output path

Loading the export elsewhere requires the same ``autogluon.tabular``, ``torch``,
``fastai`` and ``pandas`` versions as recorded in the sidecar ``.json``::

    import pickle
    with open("exports/riskship_rna_model.pickle", "rb") as f:
        predictor = pickle.load(f)
    scores = predictor.predict_proba(prepare_features(rows), as_multiclass=False)
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR = Path("AutogluonModels/rna_shipment_loss")
DEFAULT_OUT = Path("exports/riskship_rna_model.pickle")
DATA_PATH = Path("dataset.parquet")
DATE_COLUMN = "date_status_0"
# Same names and order as the notebook: the model learned these category values.
DAY_NAMES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
VERIFY_SAMPLE_SIZE = 2000


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the two notebook features that do not exist in the raw export.

    This is the inference-time part of the notebook's ``preprocess``: it does not
    apply the training-only row filters (missingness artifacts, label maturity).
    Columns the model does not use are ignored by AutoGluon, so passing the
    whole raw row is fine.
    """
    frame = raw.copy()
    frame["TIPO_BULKY"] = frame["TIPO_BULKY"].fillna("NONE")
    frame["day_of_week"] = pd.Categorical(
        pd.to_datetime(frame[DATE_COLUMN]).dt.dayofweek.map(dict(enumerate(DAY_NAMES))),
        categories=DAY_NAMES,
        ordered=True,
    )
    return frame


def seed_dirs(model_dir: Path) -> dict[int, Path]:
    found = {int(p.name.removeprefix("seed_")): p for p in model_dir.glob("seed_*") if p.is_dir()}
    if not found:
        sys.exit(f"no seed_* directories under {model_dir}; run the notebook first")
    return found


def validation_score(path: Path) -> float:
    """Validation PR-AUC of the final ensemble, as stored by AutoGluon at fit time."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(str(path), require_version_match=False)
    board = predictor.leaderboard(silent=True)
    return float(board.loc[board["model"] == predictor.model_best, "score_val"].iloc[0])


def pick_best_seed(dirs: dict[int, Path]) -> tuple[int, dict[int, float]]:
    scores = {seed: validation_score(path) for seed, path in sorted(dirs.items())}
    best = max(scores, key=scores.get)
    return best, scores


def export(seed: int, path: Path, out: Path) -> dict:
    import autogluon.tabular
    import torch
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(str(path), require_version_match=False)
    # Load every model (and the ensemble's ancestors) into memory so the pickled
    # object never goes back to disk.
    predictor.persist(models="all", with_ancestors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(predictor, f, protocol=pickle.HIGHEST_PROTOCOL)

    board = predictor.leaderboard(silent=True)
    metadata = {
        "seed": seed,
        "source_dir": str(path),
        "model_best": predictor.model_best,
        "validation_pr_auc": float(board.loc[board["model"] == predictor.model_best, "score_val"].iloc[0]),
        "eval_metric": predictor.eval_metric.name,
        "label": predictor.label,
        "features": predictor.features(),
        "day_of_week_categories": DAY_NAMES,
        "pickle_bytes": out.stat().st_size,
        "versions": {
            "python": platform.python_version(),
            "autogluon.tabular": autogluon.tabular.__version__,
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with out.with_suffix(".json").open("w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


VERIFY_SNIPPET = """
import pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
pickle_path, sample_path, ref_path = sys.argv[1:4]
with open(pickle_path, "rb") as f:
    predictor = pickle.load(f)
scores = predictor.predict_proba(pd.read_parquet(sample_path), as_multiclass=False).to_numpy()
ref = np.load(ref_path)
diff = float(np.abs(scores - ref).max())
print(f"round trip: {len(scores)} rows, max abs diff {diff:.2e}")
sys.exit(0 if np.allclose(scores, ref, atol=1e-6) else 1)
"""


def verify(seed_dir: Path, out: Path) -> None:
    """Load the pickle in a fresh interpreter, with the source directory hidden,
    and require identical scores to the directory-backed predictor."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(str(seed_dir), require_version_match=False)
    raw = pd.read_parquet(DATA_PATH)
    sample = prepare_features(raw.sample(n=min(VERIFY_SAMPLE_SIZE, len(raw)), random_state=0))
    sample = sample[predictor.features()]
    ref = predictor.predict_proba(sample, as_multiclass=False).to_numpy()

    work = out.parent / ".verify"
    work.mkdir(exist_ok=True)
    sample_path, ref_path = work / "sample.parquet", work / "ref.npy"
    sample.to_parquet(sample_path)
    np.save(ref_path, ref)

    hidden = seed_dir.with_name(seed_dir.name + ".hidden-during-verify")
    seed_dir.rename(hidden)
    try:
        result = subprocess.run(
            [sys.executable, "-c", VERIFY_SNIPPET, str(out), str(sample_path), str(ref_path)],
            capture_output=True,
            text=True,
        )
    finally:
        hidden.rename(seed_dir)
        sample_path.unlink(missing_ok=True)
        ref_path.unlink(missing_ok=True)
        work.rmdir()
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        out.unlink(missing_ok=True)
        out.with_suffix(".json").unlink(missing_ok=True)
        sys.exit(f"verification failed, export removed:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, help="seed to export (default: best validation PR-AUC)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output path (default: {DEFAULT_OUT})")
    parser.add_argument("--no-verify", action="store_true", help="skip the fresh-process round-trip check")
    args = parser.parse_args()

    dirs = seed_dirs(MODEL_DIR)
    if args.seed is None:
        seed, scores = pick_best_seed(dirs)
        for s, score in scores.items():
            print(f"seed {s}: validation PR-AUC {score:.4f}" + ("  <- best" if s == seed else ""))
    elif args.seed in dirs:
        seed = args.seed
    else:
        sys.exit(f"seed {args.seed} not found; available: {sorted(dirs)}")

    metadata = export(seed, dirs[seed], args.out)
    print(f"exported seed {seed} ({metadata['model_best']}) -> {args.out} ({metadata['pickle_bytes'] / 1e6:.1f} MB)")
    print(f"metadata -> {args.out.with_suffix('.json')}")
    if not args.no_verify:
        verify(dirs[seed], args.out)


if __name__ == "__main__":
    main()
