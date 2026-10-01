"""Land raw ERCOT telemetry in Snowflake bronze.

Target: ``{SNOWFLAKE_DATABASE}.BRONZE.RAW_ERCOT_TELEMETRY``. The database defaults
to ``ERCOT_LAKEHOUSE``. Set ``SNOWFLAKE_ROLE`` to a role that can merge bronze
and read silver. Do not rely on ACCOUNTADMIN outside a personal demo.

Rows are merged on ``(SCED_TIMESTAMP_UTC, SETTLEMENT_POINT)``. The connection is
opened with ``client_session_keep_alive=True``. Simulator rows are loaded with
``AUTHORITATIVE=FALSE`` and must not drive a curtailment breaker.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion.fetch_ercot import fetch_telemetry
from src.api.schemas import RawErcotTelemetry

logger = logging.getLogger("ercot.ingestion.snowflake")

TABLE_NAME = "RAW_ERCOT_TELEMETRY"
STAGE_TABLE_NAME = "RAW_ERCOT_TELEMETRY_LOAD"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

# Column order is the bronze contract. Names are stored unquoted and therefore
# upper-case in Snowflake.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("SCED_TIMESTAMP_UTC", "TIMESTAMP_NTZ NOT NULL"),
    ("INTERVAL_START_UTC", "TIMESTAMP_NTZ NOT NULL"),
    ("INTERVAL_END_UTC", "TIMESTAMP_NTZ NOT NULL"),
    ("REPEATED_HOUR_FLAG", "VARCHAR(1) NOT NULL"),
    ("SETTLEMENT_POINT", "VARCHAR(32) NOT NULL"),
    ("SETTLEMENT_POINT_TYPE", "VARCHAR(8) NOT NULL"),
    ("LMP_USD_MWH", "NUMBER(18,6) NOT NULL"),
    ("OPERATING_RESERVE_MW", "NUMBER(18,3) NOT NULL"),
    ("SYSTEM_DEMAND_MW", "NUMBER(18,3) NOT NULL"),
    ("ONLINE_CAPACITY_MW", "NUMBER(18,3) NOT NULL"),
    ("WIND_GENERATION_MW", "NUMBER(18,3) NOT NULL"),
    ("SOLAR_GENERATION_MW", "NUMBER(18,3) NOT NULL"),
    ("NET_LOAD_MW", "NUMBER(18,3) NOT NULL"),
    ("EEA_LEVEL", "NUMBER(3,0)"),
    ("GRID_STATE", "VARCHAR(64)"),
    ("HEAT_INDEX_F", "NUMBER(8,3) NOT NULL"),
    ("TEMPERATURE_F", "NUMBER(8,3) NOT NULL"),
    ("RELATIVE_HUMIDITY_PCT", "NUMBER(8,3) NOT NULL"),
    ("WEATHER_STATION_ID", "VARCHAR(16) NOT NULL"),
    ("WEATHER_OBSERVED_AT_UTC", "TIMESTAMP_NTZ NOT NULL"),
    ("EMIL_ID", "VARCHAR(32)"),
    ("REPORT_TYPE_ID", "VARCHAR(16)"),
    ("DOCUMENT_ID", "VARCHAR(32)"),
    ("PUBLISHED_AT_UTC", "TIMESTAMP_NTZ"),
    ("SOURCE", "VARCHAR(32) NOT NULL"),
    ("AUTHORITATIVE", "BOOLEAN NOT NULL"),
    ("FALLBACK_REASON", "VARCHAR(4000)"),
    ("INGESTED_AT_UTC", "TIMESTAMP_NTZ NOT NULL"),
)
_KEY_COLUMNS = ("SCED_TIMESTAMP_UTC", "SETTLEMENT_POINT")
_TIMESTAMP_COLUMNS = {
    "SCED_TIMESTAMP_UTC",
    "INTERVAL_START_UTC",
    "INTERVAL_END_UTC",
    "WEATHER_OBSERVED_AT_UTC",
    "PUBLISHED_AT_UTC",
    "INGESTED_AT_UTC",
}
_FIELD_BY_COLUMN = {name: name.lower() for name, _ddl in COLUMNS}


@dataclass(frozen=True)
class BronzeLoadReceipt:
    """Result of one bronze merge. ``loaded`` is false for a dry run."""

    database: str
    schema_name: str
    table: str
    rows_submitted: int
    rows_merged: int
    source: str
    authoritative: bool
    loaded: bool


def load_environment() -> None:
    """Load the repository ``.env`` without overriding variables already set."""

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def load_bronze(
    records: list[RawErcotTelemetry],
    *,
    dry_run: bool = False,
) -> BronzeLoadReceipt:
    """Create the bronze table if needed and merge ``records`` into it."""

    if not records:
        raise ValueError("refusing to load an empty telemetry batch")
    sources = {record.source for record in records}
    if len(sources) != 1:
        raise ValueError("a bronze batch must be entirely live or entirely simulated")
    authoritative_flags = {record.authoritative for record in records}
    if len(authoritative_flags) != 1:
        raise ValueError("a bronze batch must not mix authoritative and simulator rows")

    load_environment()
    database = _identifier(os.environ.get("SNOWFLAKE_DATABASE", "ERCOT_LAKEHOUSE"), "SNOWFLAKE_DATABASE")
    schema = _identifier(os.environ.get("SNOWFLAKE_SCHEMA", "BRONZE"), "SNOWFLAKE_SCHEMA")
    source = next(iter(sources))
    authoritative = next(iter(authoritative_flags))
    logger.info(
        "Bronze target %s.%s.%s rows=%s source=%s authoritative=%s",
        database,
        schema,
        TABLE_NAME,
        len(records),
        source,
        authoritative,
    )
    if dry_run:
        return BronzeLoadReceipt(
            database=database,
            schema_name=schema,
            table=TABLE_NAME,
            rows_submitted=len(records),
            rows_merged=0,
            source=source,
            authoritative=authoritative,
            loaded=False,
        )

    connection = _connect(database, schema)
    try:
        merged = _merge(connection, database, schema, records)
    finally:
        connection.close()
    return BronzeLoadReceipt(
        database=database,
        schema_name=schema,
        table=TABLE_NAME,
        rows_submitted=len(records),
        rows_merged=merged,
        source=source,
        authoritative=authoritative,
        loaded=True,
    )


def _connect(database: str, schema: str):
    import snowflake.connector

    required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing Snowflake environment variables: " + ", ".join(missing))
    role = os.environ.get("SNOWFLAKE_ROLE", "").strip()
    warehouse = _identifier(os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"), "SNOWFLAKE_WAREHOUSE")
    try:
        return snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=warehouse,
            database=database,
            schema=schema,
            role=role or None,
            client_session_keep_alive=True,
            login_timeout=60,
            network_timeout=120,
            application="ERCOT_Lakehouse",
        )
    except snowflake.connector.errors.Error as exc:
        raise RuntimeError(f"Snowflake connection failed ({exc.errno})") from exc


def _merge(connection, database: str, schema: str, records: list[RawErcotTelemetry]) -> int:
    import snowflake.connector

    target = f"{database}.{schema}.{TABLE_NAME}"
    stage = f"{database}.{schema}.{STAGE_TABLE_NAME}"
    column_ddl = ",\n    ".join(f"{name} {ddl}" for name, ddl in COLUMNS)
    create_sql = f"""
        CREATE SCHEMA IF NOT EXISTS {database}.{schema}
    """
    table_sql = f"""
        CREATE TABLE IF NOT EXISTS {target} (
            {column_ddl},
            CONSTRAINT PK_RAW_ERCOT_TELEMETRY PRIMARY KEY (SCED_TIMESTAMP_UTC, SETTLEMENT_POINT) NOT ENFORCED
        )
        CLUSTER BY (SCED_TIMESTAMP_UTC)
    """
    stage_sql = f"CREATE OR REPLACE TEMPORARY TABLE {stage} LIKE {target}"
    comment_sql = (
        f"COMMENT ON TABLE {target} IS "
        "'Bronze NP6-788-CD hub LMPs joined to PRC, system demand, fuel mix, and Harris County heat index.'"
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(create_sql)
            cursor.execute(table_sql)
            cursor.execute(comment_sql)
            cursor.execute(stage_sql)
            _fill_stage(connection, cursor, database, schema, records)
            cursor.execute(_merge_sql(target, stage))
            merged = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else len(records)
        connection.commit()
    except snowflake.connector.errors.Error as exc:
        connection.rollback()
        raise RuntimeError(f"Snowflake bronze merge failed ({exc.errno})") from exc
    logger.info("Merged %s rows into %s", merged, target)
    return int(merged)


def _fill_stage(connection, cursor, database: str, schema: str, records: list[RawErcotTelemetry]) -> None:
    frame_rows = [_row_values(record) for record in records]
    try:
        _write_with_pandas(connection, database, schema, frame_rows)
        return
    except Exception as exc:
        logger.warning(
            "write_pandas failed (%s). The bulk path needs pandas and pyarrow; loading the stage with bound INSERT statements",
            exc,
        )
        stage = f"{database}.{schema}.{STAGE_TABLE_NAME}"
        cursor.execute(f"DELETE FROM {stage}")
        _insert_executemany(cursor, stage, frame_rows)


def _write_with_pandas(connection, database: str, schema: str, rows: list[tuple]) -> None:
    import pandas as pd
    from snowflake.connector.pandas_tools import write_pandas

    frame = pd.DataFrame(rows, columns=[name for name, _ddl in COLUMNS])
    for name in _TIMESTAMP_COLUMNS:
        parsed = pd.to_datetime(frame[name], utc=False)
        frame[name] = [None if pd.isna(item) else item.to_pydatetime() for item in parsed]
    success, _chunks, written, _output = write_pandas(
        connection,
        frame,
        STAGE_TABLE_NAME,
        database=database,
        schema=schema,
        quote_identifiers=False,
        auto_create_table=False,
        overwrite=False,
        use_logical_type=True,
    )
    if not success or written != len(rows):
        raise RuntimeError(f"write_pandas wrote {written} of {len(rows)} rows")


def _insert_executemany(cursor, stage: str, rows: list[tuple]) -> None:
    column_list = ", ".join(name for name, _ddl in COLUMNS)
    placeholders = ", ".join(["%s"] * len(COLUMNS))
    statement = f"INSERT INTO {stage} ({column_list}) VALUES ({placeholders})"
    chunk_size = 200
    for offset in range(0, len(rows), chunk_size):
        cursor.executemany(statement, rows[offset : offset + chunk_size])


def _merge_sql(target: str, stage: str) -> str:
    updates = ",\n    ".join(
        f"target.{name} = source.{name}" for name, _ddl in COLUMNS if name not in _KEY_COLUMNS
    )
    insert_columns = ", ".join(name for name, _ddl in COLUMNS)
    insert_values = ", ".join(f"source.{name}" for name, _ddl in COLUMNS)
    return f"""
        MERGE INTO {target} AS target
        USING {stage} AS source
            ON target.SCED_TIMESTAMP_UTC = source.SCED_TIMESTAMP_UTC
           AND target.SETTLEMENT_POINT = source.SETTLEMENT_POINT
        WHEN MATCHED THEN UPDATE SET
            {updates}
        WHEN NOT MATCHED THEN INSERT ({insert_columns})
            VALUES ({insert_values})
    """


def _row_values(record: RawErcotTelemetry) -> tuple:
    payload = record.model_dump()
    values = []
    for column, _ddl in COLUMNS:
        value = payload[_FIELD_BY_COLUMN[column]]
        if column in _TIMESTAMP_COLUMNS:
            value = _naive_utc(value)
        values.append(value)
    return tuple(values)


def _naive_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Snowflake TIMESTAMP_NTZ values must originate from timezone-aware UTC")
    return value.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)


def _identifier(value: str, env_name: str) -> str:
    if not _IDENTIFIER.match(value):
        raise ValueError(f"{env_name} is not a safe Snowflake identifier")
    return value.upper()


def main(argv: list[str] | None = None) -> int:
    """Fetch a telemetry batch and merge it into bronze."""

    import json

    parser = argparse.ArgumentParser(description="Load HB_HOUSTON and HB_WEST telemetry into Snowflake bronze.")
    parser.add_argument("--intervals", type=int, default=1)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve the target and fetch rows without connecting.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    records = fetch_telemetry(
        intervals=args.intervals,
        allow_simulation_fallback=not args.no_fallback,
        simulate=args.simulate,
        seed=args.seed,
    )
    receipt = load_bronze(records, dry_run=args.dry_run)
    print(json.dumps(asdict(receipt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
