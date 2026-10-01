{{ config(
    materialized="table",
    cluster_by=["sced_timestamp_utc"],
    tags=["silver", "features"]
) }}

/*
    Silver feature grain: one HB_HOUSTON SCED interval.

    Reserve depletion velocity is the 30-minute reserve delta divided by 30,
    in MW per minute. On a complete 5-minute series the 6-row lag is that window.
    A negative value means physical responsive capability is falling.

    Nodal congestion spread is the HB_HOUSTON LMP minus the HB_WEST LMP.

    Net-load acceleration is the per-minute change in net-load velocity.
    Velocity is (net_load_t - net_load_t-1) / 5. Acceleration divides the next
    velocity delta by 5 again, so the unit is MW per minute squared.

    Same-row spike flags use a strict greater-than test against 250, 500, 1000,
    and 5000. Those flags are not model targets. is_spike_250_within_60min is 1
    when any later SCED timestamp in (t+5 minutes, t+60 minutes] prints a Houston
    LMP above $250. The label is null unless that window is actually observed:
    the next point is at most 10 minutes ahead, and some point reaches at least
    55 minutes ahead. The window is partitioned by source, so a simulator
    price cannot label an ercot_live interval. The open tail stays unlabeled.
*/

with source_rows as (
    select
        sced_timestamp_utc,
        interval_start_utc,
        interval_end_utc,
        repeated_hour_flag,
        settlement_point,
        lmp_usd_mwh,
        operating_reserve_mw,
        system_demand_mw,
        wind_generation_mw,
        solar_generation_mw,
        net_load_mw,
        heat_index_f,
        temperature_f,
        relative_humidity_pct,
        eea_level,
        grid_state,
        source,
        ingested_at_utc
    from {{ source('bronze', 'raw_ercot_telemetry') }}
    where settlement_point in ('HB_HOUSTON', 'HB_WEST')
),

deduped as (
    select *
    from source_rows
    qualify row_number() over (
        partition by sced_timestamp_utc, settlement_point
        order by ingested_at_utc desc
    ) = 1
),

houston as (
    select
        sced_timestamp_utc,
        interval_start_utc,
        interval_end_utc,
        repeated_hour_flag,
        settlement_point,
        lmp_usd_mwh as lmp_hb_houston_usd_mwh,
        operating_reserve_mw,
        system_demand_mw,
        wind_generation_mw,
        solar_generation_mw,
        net_load_mw,
        heat_index_f,
        temperature_f,
        relative_humidity_pct,
        eea_level,
        grid_state,
        source
    from deduped
    where settlement_point = 'HB_HOUSTON'
),

west as (
    select
        sced_timestamp_utc,
        lmp_usd_mwh as lmp_hb_west_usd_mwh
    from deduped
    where settlement_point = 'HB_WEST'
),

joined as (
    select
        houston.sced_timestamp_utc,
        houston.interval_start_utc,
        houston.interval_end_utc,
        houston.repeated_hour_flag,
        houston.settlement_point,
        houston.lmp_hb_houston_usd_mwh,
        west.lmp_hb_west_usd_mwh,
        houston.lmp_hb_houston_usd_mwh - west.lmp_hb_west_usd_mwh as nodal_congestion_spread_usd_mwh,
        houston.operating_reserve_mw,
        houston.system_demand_mw,
        houston.wind_generation_mw,
        houston.solar_generation_mw,
        houston.net_load_mw,
        houston.heat_index_f,
        houston.temperature_f,
        houston.relative_humidity_pct,
        houston.eea_level,
        houston.grid_state,
        houston.source,
        extract(
            hour from convert_timezone('UTC', 'America/Chicago', houston.sced_timestamp_utc)
        ) as hour_ct
    from houston
    inner join west
        on houston.sced_timestamp_utc = west.sced_timestamp_utc
),

lagged as (
    select
        joined.*,
        lag(joined.operating_reserve_mw, 6) over (
            order by joined.sced_timestamp_utc
        ) as reserve_mw_30min_prior,
        lag(joined.sced_timestamp_utc, 6) over (
            order by joined.sced_timestamp_utc
        ) as sced_timestamp_30min_prior,
        lag(joined.net_load_mw, 1) over (
            order by joined.sced_timestamp_utc
        ) as net_load_mw_prior,
        lag(joined.sced_timestamp_utc, 1) over (
            order by joined.sced_timestamp_utc
        ) as sced_timestamp_prior
    from joined
),

with_velocity as (
    select
        lagged.*,
        case
            when datediff('second', lagged.sced_timestamp_30min_prior, lagged.sced_timestamp_utc) between 25 * 60 and 35 * 60
                then (lagged.operating_reserve_mw - lagged.reserve_mw_30min_prior) / 30.0
            else null
        end as reserve_depletion_velocity_mw_per_min,
        case
            when datediff('second', lagged.sced_timestamp_prior, lagged.sced_timestamp_utc) between 4 * 60 and 6 * 60
                then (lagged.net_load_mw - lagged.net_load_mw_prior) / 5.0
            else null
        end as net_load_velocity_mw_per_min
    from lagged
),

with_acceleration as (
    select
        with_velocity.*,
        lag(with_velocity.net_load_velocity_mw_per_min, 1) over (
            order by with_velocity.sced_timestamp_utc
        ) as net_load_velocity_prior,
        lag(with_velocity.sced_timestamp_utc, 1) over (
            order by with_velocity.sced_timestamp_utc
        ) as acceleration_prior_timestamp
    from with_velocity
),

realized as (
    select
        sced_timestamp_utc,
        interval_start_utc,
        interval_end_utc,
        repeated_hour_flag,
        settlement_point,
        lmp_hb_houston_usd_mwh,
        lmp_hb_west_usd_mwh,
        nodal_congestion_spread_usd_mwh,
        operating_reserve_mw,
        reserve_depletion_velocity_mw_per_min,
        system_demand_mw,
        wind_generation_mw,
        solar_generation_mw,
        net_load_mw,
        net_load_velocity_mw_per_min,
        case
            when net_load_velocity_mw_per_min is not null
                and net_load_velocity_prior is not null
                and datediff('second', acceleration_prior_timestamp, sced_timestamp_utc) between 4 * 60 and 6 * 60
                then (net_load_velocity_mw_per_min - net_load_velocity_prior) / 5.0
            else null
        end as net_load_acceleration_mw_per_min2,
        heat_index_f,
        temperature_f,
        relative_humidity_pct,
        eea_level,
        grid_state,
        hour_ct,
        source,
        case when lmp_hb_houston_usd_mwh > 250 then 1 else 0 end as is_spike_250,
        case when lmp_hb_houston_usd_mwh > 500 then 1 else 0 end as is_spike_500,
        case when lmp_hb_houston_usd_mwh > 1000 then 1 else 0 end as is_spike_1000,
        case when lmp_hb_houston_usd_mwh > 5000 then 1 else 0 end as is_spike_5000,
        date_part(epoch_second, sced_timestamp_utc) as sced_epoch
    from with_acceleration
),

forward_window as (
    select
        realized.*,
        max(is_spike_250) over (
            partition by source
            order by sced_epoch
            range between 300 following and 3600 following
        ) as spike_flag_within_60min,
        min(sced_epoch) over (
            partition by source
            order by sced_epoch
            range between 300 following and 3600 following
        ) as first_future_epoch,
        max(sced_epoch) over (
            partition by source
            order by sced_epoch
            range between 300 following and 3600 following
        ) as last_future_epoch
    from realized
)

select
    sced_timestamp_utc,
    interval_start_utc,
    interval_end_utc,
    repeated_hour_flag,
    settlement_point,
    lmp_hb_houston_usd_mwh,
    lmp_hb_west_usd_mwh,
    nodal_congestion_spread_usd_mwh,
    operating_reserve_mw,
    reserve_depletion_velocity_mw_per_min,
    system_demand_mw,
    wind_generation_mw,
    solar_generation_mw,
    net_load_mw,
    net_load_velocity_mw_per_min,
    net_load_acceleration_mw_per_min2,
    heat_index_f,
    temperature_f,
    relative_humidity_pct,
    eea_level,
    grid_state,
    hour_ct,
    source,
    is_spike_250,
    is_spike_500,
    is_spike_1000,
    is_spike_5000,
    case
        when first_future_epoch is null then null
        when first_future_epoch - sced_epoch > 600 then null
        when last_future_epoch - sced_epoch < 3300 then null
        else spike_flag_within_60min
    end as is_spike_250_within_60min
from forward_window
