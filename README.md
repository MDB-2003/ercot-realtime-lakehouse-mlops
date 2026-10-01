# ERCOT Houston hub lakehouse

Five-minute HB_HOUSTON telemetry, a 60-minute price-spike forecast, and a deterministic curtailment desk for a 50 MW Houston Ship Channel olefins unit.

The classifier estimates the chance that HB_HOUSTON prints above $250/MWh sometime in the next 60 minutes. It does not score the price it was just given. Same-row flags (`is_spike_250` and the $500 / $1,000 / $5,000 flags) stay on the silver table for auditing. The training label is `is_spike_250_within_60min`, and it is null until that future hour has actually been observed. Training rows whose 60-minute label window reaches into a test fold are dropped before that fold is fit. Only `ercot_live` rows are used.

Curtailment is not chosen by the model. `src/models/dispatch_optimizer.py` compares three positions on caller-supplied probabilities: take the price, shed up to 20 MW, or discharge the 10 MW battery and shed the rest. The language-model flag on that result is always false.

## Layout

| Path | Role |
| --- | --- |
| `src/ingestion/fetch_ercot.py` | NP6-788-CD LMPs, PRC, load, fuel mix, and the La Porte heat index |
| `src/ingestion/load_snowflake_bronze.py` | Merge into `ERCOT_LAKEHOUSE.BRONZE.RAW_ERCOT_TELEMETRY` |
| `dbt_ercot/` | Silver features, including the forward spike label |
| `src/models/train_classifier.py` | Walk-forward LightGBM and the SHAP explainer |
| `src/models/dispatch_optimizer.py` | Baseline, curtailment, and battery-plus-shed economics |
| `src/api/main.py` | `/health`, `/predict`, `/explain`, forecast, and simulate |
| `src/dashboard/app.py` | Streamlit desk. Plots come from `/explain` |
| `config/facility_profile.yaml` | 50 MW load, 20 MW curtailable, 10 MW / 20 MWh battery, $6,500/hr downtime |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with a Snowflake role that can merge bronze and read silver. The default database is `ERCOT_LAKEHOUSE`. `.env` is gitignored.

## Run

```bash
python -m src.ingestion.load_snowflake_bronze
dbt run --project-dir dbt_ercot --profiles-dir dbt_ercot --select stg_ercot_features
python -m src.models.train_classifier
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
streamlit run src/dashboard/app.py
```

`train_classifier` writes `models/spike_lgb_v1.pkl` and `models/explainer.pkl` only after at least three walk-forward folds contain both a future spike and a future non-spike. Until that history exists, `/predict` and `/explain` return 503. `/health` still reports whether the files are on disk. A model trained on the same-row price flag is rejected.

`python seed_spikes.py` exits without writing. It does not plant prices or mark simulator rows as authoritative.

Dispatch economics for one 5-minute interval at $400/MWh with P(spike) = 0.90:

| Strategy | Energy | Downtime | Total | 4CP risk |
| --- | ---: | ---: | ---: | ---: |
| Unmitigated | $1,666.67 | $0 | $1,666.67 | $2,700,000 |
| Shed 20 MW | $1,000.00 | $541.67 | $1,541.67 | $1,620,000 |
| 10 MW battery + 20 MW shed | $666.67 | $541.67 | $1,208.34 | $1,080,000 |

```bash
python -m unittest tests.test_dispatch_optimizer
```
