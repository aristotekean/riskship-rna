"""Score shipments with the exported model and flag the ones above a risk threshold.

Reads shipments as JSON (an object or a list of objects with the raw export
columns the model uses, see ``exports/riskship_rna_model.json``) and prints one
JSON document with the loss probability per shipment and whether it exceeds the
threshold. Meant to be called by an agent or a cron job::

    uv run predict_risk.py shipment.json                 # file
    echo '{...}' | uv run predict_risk.py -              # stdin
    uv run predict_risk.py --shipment-id 44879512345     # a row from dataset.parquet
    uv run predict_risk.py --sample 3 --threshold 0.65   # random rows, demo

Exit code 0 always when scoring succeeds; read ``any_at_risk`` in the output.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

from export_model import prepare_features

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "exports" / "riskship_rna_model.pickle"
DATA_PATH = PROJECT_DIR / "dataset.parquet"
DEFAULT_THRESHOLD = 0.65
ID_COLUMN = "SHIPMENT_ID"


def load_predictor(path: Path):
    if not path.exists():
        sys.exit(f"model not found at {path}; run `uv run export_model.py` first")
    with path.open("rb") as f:
        return pickle.load(f)


def read_shipments(source: str) -> pd.DataFrame:
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    payload = json.loads(text)
    rows = payload if isinstance(payload, list) else [payload]
    if not rows:
        sys.exit("no shipments in input")
    return pd.DataFrame(rows)


def rows_from_dataset(shipment_ids: list[str] | None, sample: int | None, seed: int) -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH)
    if shipment_ids:
        rows = raw[raw[ID_COLUMN].astype(str).isin(shipment_ids)]
        if rows.empty:
            sys.exit(f"no rows for SHIPMENT_ID in {shipment_ids}")
        return rows
    return raw.dropna(subset=["PESO_TOTAL_KG", "DIM_MAX_CM", "VALOR_TOTAL_USD", "CANTIDAD_ITEMS", "date_status_0"]).sample(
        n=sample, random_state=seed
    )


def score(predictor, shipments: pd.DataFrame, threshold: float) -> dict:
    features = predictor.features()
    missing = [c for c in features if c not in shipments.columns and c != "day_of_week"]
    if "date_status_0" not in shipments.columns:
        missing.append("date_status_0")
    if missing:
        sys.exit(f"input is missing columns the model needs: {missing}")
    prepared = prepare_features(shipments)
    probabilities = predictor.predict_proba(prepared[features], as_multiclass=False).to_numpy()
    results = []
    for (_, row), p in zip(shipments.iterrows(), probabilities):
        results.append({
            "shipment_id": None if ID_COLUMN not in row or pd.isna(row[ID_COLUMN]) else str(row[ID_COLUMN]),
            "loss_probability": round(float(p), 4),
            "at_risk": bool(p > threshold),
            "actual_is_lost": None if "is_lost" not in row or pd.isna(row["is_lost"]) else int(row["is_lost"]),
            "date_status_0": None if pd.isna(row["date_status_0"]) else str(pd.Timestamp(row["date_status_0"])),
            "features": {c: (None if pd.isna(row[c]) else row[c]) for c in features if c in row.index},
        })
    return {
        "threshold": threshold,
        "model": str(MODEL_PATH.relative_to(PROJECT_DIR)),
        "n_shipments": len(results),
        "n_at_risk": sum(r["at_risk"] for r in results),
        "any_at_risk": any(r["at_risk"] for r in results),
        "shipments": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", nargs="?", help="JSON file with shipment(s), or '-' for stdin")
    parser.add_argument("--shipment-id", action="append", help="score this SHIPMENT_ID from dataset.parquet (repeatable)")
    parser.add_argument("--sample", type=int, help="score N random shipments from dataset.parquet (demo)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for --sample (default: nondeterministic)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"risk threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    if args.source:
        shipments = read_shipments(args.source)
    elif args.shipment_id or args.sample:
        shipments = rows_from_dataset(args.shipment_id, args.sample, args.seed)
    else:
        parser.error("give a JSON source, --shipment-id or --sample")

    predictor = load_predictor(args.model)
    print(json.dumps(score(predictor, shipments, args.threshold), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
