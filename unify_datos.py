# /// script
# requires-python = ">=3.13"
# dependencies = ["polars>=1.20"]
# ///
"""Unify every CSV and JSON export in a directory into a single dataset.

Usage:
    uv run unify_datos.py                      # datos/ -> dataset.parquet
    uv run unify_datos.py datos -o dataset.csv

A column name/type dictionary is always written next to the output
(dataset.parquet -> dataset_dictionary.md).
"""

import argparse
import json
from pathlib import Path

import polars as pl

SOURCE_COLUMN = "source_file"
BOOLEAN_VALUES = {"true", "false"}
# Export bookkeeping, not shipment data: the same row pulled in two batches is one row.
DEDUP_IGNORED_COLUMNS = {SOURCE_COLUMN, "batch_num"}


def read_raw(path: Path) -> pl.DataFrame:
    """Load one export with every column as a string, so files concat safely."""
    match path.suffix.lower():
        case ".csv":
            frame = pl.read_csv(path, infer_schema=False)
        case ".json":
            # pl.read_json does not preserve key order; the stdlib parser does.
            records = json.loads(path.read_text(encoding="utf-8"))
            frame = pl.from_dicts(records, infer_schema_length=None).cast(pl.String)
        case suffix:
            raise ValueError(f"Unsupported file type: {suffix}")
    return frame.with_columns(pl.lit(path.name).alias(SOURCE_COLUMN))


def infer_column(column: pl.Series) -> pl.Series:
    """Cast a string column to the narrowest type that fits every value."""
    values = column.drop_nulls()
    # Identifiers stay as strings: leading zeros and joins matter more than math.
    if values.is_empty() or "_ID" in column.name.upper():
        return column
    if set(values.unique()) <= BOOLEAN_VALUES:
        return column == "true"
    casts = (
        lambda s: s.cast(pl.Int64),
        lambda s: s.cast(pl.Float64),
        lambda s: s.str.to_date("%Y-%m-%d"),
        lambda s: s.str.to_datetime(),
    )
    for cast in casts:
        try:
            return cast(column)
        except pl.exceptions.PolarsError:
            continue
    return column


def unify(input_dir: Path) -> pl.DataFrame:
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".csv", ".json"})
    if not paths:
        raise SystemExit(f"No CSV or JSON files found in {input_dir}")

    raw = pl.concat([read_raw(path) for path in paths], how="diagonal")
    data_columns = [name for name in raw.columns if name != SOURCE_COLUMN]

    # CSV yields null for empty cells while JSON yields "": normalize before typing.
    normalized = raw.with_columns(
        pl.col(data_columns).str.strip_chars().replace("", None)
    )
    typed = normalized.with_columns(infer_column(normalized[name]) for name in data_columns)
    dedup_columns = [name for name in data_columns if name not in DEDUP_IGNORED_COLUMNS]
    unified = typed.unique(subset=dedup_columns, keep="first", maintain_order=True)

    print(f"files: {len(paths)}  rows read: {raw.height:,}  duplicates dropped: "
          f"{raw.height - unified.height:,}  final: {unified.height:,} x {unified.width}")
    return unified


def write_dictionary(frame: pl.DataFrame, dataset_path: Path) -> Path:
    """Write a markdown table of column names and types next to the dataset."""
    path = dataset_path.with_name(f"{dataset_path.stem}_dictionary.md")
    rows = "\n".join(
        f"| `{name}` | {dtype.base_type().__name__} |" for name, dtype in frame.schema.items()
    )
    path.write_text(
        f"# Data dictionary: {dataset_path.name}\n\n"
        f"{frame.height:,} rows x {frame.width} columns\n\n"
        f"| Column | Type |\n|---|---|\n{rows}\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("datos"))
    parser.add_argument("-o", "--output", type=Path, default=Path("dataset.parquet"),
                        help="output file; .parquet or .csv (default: dataset.parquet)")
    args = parser.parse_args()

    unified = unify(args.input_dir)
    match args.output.suffix.lower():
        case ".parquet":
            unified.write_parquet(args.output)
        case ".csv":
            unified.write_csv(args.output)
        case suffix:
            raise SystemExit(f"Unsupported output format: {suffix}")
    print(f"wrote {args.output}")
    print(f"wrote {write_dictionary(unified, args.output)}")


if __name__ == "__main__":
    main()
