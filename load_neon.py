# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=17", "psycopg[binary]>=3.2"]
# ///
"""Create the shipments table in Neon and bulk-load dataset.parquet into it.

    uv run load_neon.py                 # dataset.parquet -> table shipments
    uv run load_neon.py --truncate      # reload from scratch

The table is what the Hermes profile `riskship` reads through its `riesgo-envio-neon`
cron. Needs a connection string WITH write permission in NEON_DATABASE_URL (or
--url); the read-only URL the cron uses is not enough. Rows go in through COPY, in the
column order of the parquet file; `id` and `created_at` take their defaults.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent
DDL_PATH = PROJECT_DIR / "sql" / "shipments.sql"
DEFAULT_PARQUET = PROJECT_DIR / "dataset.parquet"
BATCH = 20_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET, help=f"parquet to load (default {DEFAULT_PARQUET.name})")
    parser.add_argument("--url", default=os.environ.get("NEON_DATABASE_URL"), help="Postgres URL with write access")
    parser.add_argument("--truncate", action="store_true", help="empty the table before loading")
    args = parser.parse_args()
    if not args.url:
        sys.exit("set NEON_DATABASE_URL or pass --url (a connection string with write access)")

    table = pq.read_table(args.parquet)
    columns = ", ".join(f'"{name}"' for name in table.column_names)
    started = time.perf_counter()
    with psycopg.connect(args.url) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_PATH.read_text())
            if args.truncate:
                cur.execute("TRUNCATE shipments RESTART IDENTITY")
            cur.execute("SELECT count(*) FROM shipments")
            before = cur.fetchone()[0]
            with cur.copy(f"COPY shipments ({columns}) FROM STDIN") as copy:
                # Declare types so psycopg encodes dates/timestamps/bools, not their repr.
                copy.set_types([_pg_type(str(t)) for t in table.schema.types])
                for batch in table.to_batches(max_chunksize=BATCH):
                    for row in zip(*(col.to_pylist() for col in batch.columns)):
                        copy.write_row(row)
            cur.execute("SELECT count(*) FROM shipments")
            after = cur.fetchone()[0]
        conn.commit()
    print(f"rows in parquet: {table.num_rows:,}; table before: {before:,}; after: {after:,}; {time.perf_counter() - started:.0f} s")


def _pg_type(arrow_type: str) -> str:
    return {"large_string": "text", "string": "text", "bool": "bool", "int64": "int8", "double": "float8",
            "date32[day]": "date", "timestamp[us]": "timestamp"}[arrow_type]


if __name__ == "__main__":
    main()
