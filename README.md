# ERCOT Houston Hub Price-Spike Case Study

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Understanding Price Spikes](#understanding-price-spikes)
- [Business Aspect](#business-aspect)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [Directory Tree](#directory-tree)
- [Technologies Used](#technologies-used)
- [Credits](#credits)

## Overview

This project looks at a familiar ERCOT problem: Houston hub prices that stay quiet for long stretches and then jump. It has two parts.

The first is a classifier that estimates the chance HB_HOUSTON prints above $250/MWh sometime in the next 60 minutes. It reads five-minute telemetry (prices, reserves, weather, and the time of day) and returns a probability for the hour ahead. The second part is a small dispatch desk for a 50 MW olefins unit on the Houston Ship Channel. Given that probability and a price path, the desk compares three ways to get through the interval: buy the full load, shed up to 20 MW of discretionary equipment, or discharge a 10 MW battery and shed the rest.

The model stops at the probability. Whether anything actually comes offline is decided by the dispatch rules and the limits in the facility profile.

## Motivation

In ERCOT, a large industrial load settles against a hub price that can move from ordinary levels to scarcity pricing inside a single afternoon. A continuous chemical plant feels that immediately. The energy for the interval is already purchased by the time the price prints, and a process that has to stay up (reactors, catalytic cooling, flares) cannot be switched off just because the hub got expensive.

Replacing lost production is costly, and so is sitting through a spike on the full 50 MW. The useful moment is the hour before the print, while there is still time to drop air compressors, pelletizers, or secondary chillers, or to cover part of the import with the battery.

For a plant like this, the practical goal is narrow. Keep the critical 30 MW online, and use the flexible 20 MW and the battery only when the next hour looks expensive enough to justify the downtime.

## Understanding Price Spikes

A spike, in this project, means the Houston hub locational marginal price clears above a fixed dollar threshold.

The working threshold is $250/MWh. The silver table also marks $500, $1,000, and $5,000 so those rarer hours can be checked later. Each of those flags describes the price on the same row. They are useful as a record of what just happened. They are a poor training target, because the Houston price is already one of the inputs. Asking the model whether that price is above $250 mostly asks it to read the number it was given.

The label used for training is `is_spike_250_within_60min`. It is 1 when any later five-minute interval in the next hour prints a Houston price above $250. It stays empty until that hour has actually shown up in the data, so the unfinished tail of the series is never treated as a calm hour.

Two money figures sit next to that label, and they answer different questions.

Energy cost is straightforward. It is the megawatts still taken from the grid, times the length of the interval, times the hub price.

The 4CP figure is a transmission-planning exposure. ERCOT's four highest summer system peaks set a large part of the following year's transmission charge. This profile uses $54,000 per MW-year as that planning value. If the simulated horizon is the one that sets a summer peak, the desk reports the annual charge attached to the grid import during that horizon. It is a way to see how much peak demand the strategy left on the meter. It is not the bill for those five-minute intervals.

## Business Aspect

The case is built around one site: Houston Ship Channel Olefins Complex, Unit 3, in Harris County, interconnected through CenterPoint at 138 kV and settled at HB_HOUSTON. HB_WEST is kept alongside it so Houston-specific congestion is visible as a spread, rather than looking like a system-wide price move. Weather is the heat index and temperature for the Ship Channel site in the facility profile.

People at a plant like this usually have a little time. They do not decide to shed load in the same minute the price spikes. The day tends to fall into three stretches.

**The ordinary stretch.** Reserves are healthy, the heat is manageable, and Houston is priced like a normal hour. The plant runs all 50 MW. A forecast that is sitting near zero should leave the process alone.

**The decision stretch.** Reserves start to fall, the heat index climbs, or Houston pulls away from the West hub. This is the next 60 minutes, which is the horizon the model is built for. Curtailable equipment still has duration left, and the battery still has state of charge. Acting here changes the import. Waiting until the price has already cleared means the energy and the downtime land on the same interval.

**The spike itself.** Houston is above $250, sometimes well above it. Training tags this window as the event, then asks the model to see it from earlier rows. When you score a live interval, that future price is not available as an input. The open hour stays unlabeled until it has been observed.

Curtailment has its own cost. Lost production is priced at $6,500 per hour. Shedding the full 20 MW for five minutes is one twelfth of that, about $542. Discharging the battery does not add that downtime charge, because the process load is still being served. The desk runs all three positions on the prices and probabilities you supply and keeps the one with the lower total cost.

A single five-minute interval at $400/MWh, with a spike probability of 0.90, comes out like this:

| Strategy | Energy | Downtime | Total | 4CP risk |
| --- | ---: | ---: | ---: | ---: |
| Unmitigated | $1,666.67 | $0 | $1,666.67 | $2,700,000 |
| Shed 20 MW | $1,000.00 | $541.67 | $1,541.67 | $1,620,000 |
| 10 MW battery + 20 MW shed | $666.67 | $541.67 | $1,208.34 | $1,080,000 |

The facility profile also sets `llm_may_authorize_curtailment` to false. A language model can sit beside the desk. It does not get to trip load or change the megawatt limits.

## Technical Aspect

The project is split the same way as the business problem: one path that forecasts the spike, and one path that turns a probability into a cost comparison.

**Forecasting the next hour**

Live rows are pulled every five minutes from public sources and joined on the SCED interval:

- ERCOT MIS report NP6-788-CD, for HB_HOUSTON and HB_WEST LMPs
- ERCOT dashboards for physical responsive capability, system demand, and the fuel mix
- National Weather Service observations at the Ship Channel site

If a live feed is blocked or comes back incomplete, the fetch path can emit simulator rows with the same shape and `authoritative` set to false. Those rows keep the pipeline moving. Training ignores them and uses only `ercot_live`.

The rows are merged into Snowflake bronze (`ERCOT_LAKEHOUSE.BRONZE.RAW_ERCOT_TELEMETRY`). A dbt model builds one silver row per Houston interval. Along the way it adds reserve-depletion velocity over 30 minutes, the Houston-minus-West price spread, net-load acceleration, and the forward spike label. Same-row spike flags stay on that table for auditing.

The classifier is LightGBM. Evaluation is walk-forward, so every test row comes after the rows it was trained on. Any training row whose 60-minute label window reaches into the test fold is dropped before that fold is fit, and the class weight for a fold is computed only from that fold's training rows. The model and the SHAP explainer are saved only after at least three test folds contain both a future spike and a future non-spike. Until that history exists, `/predict` and `/explain` return 503. `/health` still tells you whether the files are on disk.

SHAP attributions are on the probability scale. The background is a sample of training rows, so a single call can show which inputs pushed the probability up or down: thin reserves, a hot hour, Houston already rich versus the West, and so on.

Features used by the model:

- Houston and West hub prices
- operating reserves and how fast those reserves are falling
- temperature and heat index
- hour of day and day of week

**Reading the drivers, then costing the response**

`/explain` returns every feature contribution, largest absolute value first. The Streamlit desk plots that response. It does not invent a second explanation.

Dispatch is a separate function. `simulate_dispatch` does not load the model. You pass the interval prices and the spike probabilities. It scores three strategies on that path:

1. Buy the full 50 MW at the hub price.
2. Shed discretionary load, up to 20 MW, when the probability is above the threshold you set. Each curtailable asset still has a maximum duration.
3. Do that shed and also discharge the 10 MW / 20 MWh battery, down to a 15% state of charge. Round-trip efficiency is 88%.

The strategy with the lower energy-plus-downtime cost is the one the desk returns. A missing model file does not block simulation, and a model score never closes a breaker by itself.

## Installation

The code is written for current Python 3. Python 3.11 or newer matches the package pins in `requirements.txt`. If Python is missing, install it from [python.org](https://www.python.org/downloads/). From the project directory, after cloning:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and fill in the Snowflake account, user, password, and role. The role needs to merge into bronze and read silver. Warehouse, database, and schema have defaults (`COMPUTE_WH`, `ERCOT_LAKEHOUSE`, `BRONZE`). `.env` is gitignored.

Then, in order:

```bash
python -m src.ingestion.load_snowflake_bronze
dbt run --project-dir dbt_ercot --profiles-dir dbt_ercot --select stg_ercot_features
python -m src.models.train_classifier
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
streamlit run src/dashboard/app.py
```

The API exposes `/health`, `/predict`, `/explain`, `/api/v1/grid/forecast`, and `/api/v1/grid/simulate`. Interactive docs are at `/docs` once the server is up.

Dispatch tests, with no Snowflake connection required:

```bash
python -m unittest tests.test_dispatch_optimizer
```

## Directory Tree

```text
├── config
│   └── facility_profile.yaml
├── dbt_ercot
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models
│       ├── sources.yml
│       └── silver
│           ├── schema.yml
│           └── stg_ercot_features.sql
├── src
│   ├── api
│   │   ├── main.py
│   │   └── schemas.py
│   ├── dashboard
│   │   └── app.py
│   ├── ingestion
│   │   ├── fetch_ercot.py
│   │   └── load_snowflake_bronze.py
│   └── models
│       ├── dispatch_optimizer.py
│       └── train_classifier.py
├── tests
│   └── test_dispatch_optimizer.py
├── requirements.txt
└── README.md
```

`train_classifier` writes `models/spike_lgb_v1.pkl` and `models/explainer.pkl` locally. Those files are gitignored.

## Technologies Used

- Python
- pandas
- LightGBM
- SHAP
- scikit-learn
- FastAPI and Uvicorn
- Streamlit
- dbt and Snowflake
- Pydantic

## Credits

Settlement prices, reserves, system demand, and fuel mix come from ERCOT's public MIS and dashboard feeds. Temperature and heat index come from the National Weather Service. The megawatts, battery size, and downtime cost in `config/facility_profile.yaml` are a planning profile for this case study.
