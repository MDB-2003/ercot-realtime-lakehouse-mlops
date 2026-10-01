"""Refuses to overwrite bronze with planted spike prices.

The classifier target is a real HB_HOUSTON LMP above $250 inside the next 60
minutes of ercot_live telemetry. A synthetic overwrite marked authoritative
teaches the model a clock pattern and destroys the landed SCED history.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "seed_spikes.py does not run. Land NP6-788-CD with "
        "python -m src.ingestion.load_snowflake_bronze, build "
        "is_spike_250_within_60min with dbt, then train on ercot_live rows only.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
