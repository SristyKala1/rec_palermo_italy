"""Build hourly consumption, production and price profiles for the 9 Palermo
REC entities in the CSV format SmartCer reads.

Inputs: PVGIS hourly PV output (1 kWp, Palermo) and OPSD Sicily zonal load
and day-ahead price, 2015-2020.

Usage: python scripts/build_consumption_profiles.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "data" / "config" / "config.json"
NODES_DIR = REPO_ROOT / "data" / "config" / "nodes"

SIM_START = "2015-01-01 00:00:00"
SIM_END = "2020-09-30 23:00:00"


def load_pvgis(path: Path = DATA_RAW / "pvgis_palermo_2015_2020.csv") -> pd.Series:
    """Hourly PV output per kWp in kWh, timestamps floored to the hour."""
    df = pd.read_csv(path)
    df = df[df["time"].str.match(r"^\d{8}:\d{4}$", na=False)]
    dt = pd.to_datetime(df["time"], format="%Y%m%d:%H%M").dt.floor("h")
    production_kwh_per_kwp = df["P"].astype(float).to_numpy() / 1000.0
    series = pd.Series(production_kwh_per_kwp, index=dt, name="pv_per_kwp").sort_index()
    series = series[~series.index.duplicated(keep="first")]
    return series.loc[SIM_START:SIM_END]


def load_opsd(path: Path = DATA_RAW / "opsd_sicily_2015_2020.csv") -> pd.DataFrame:
    """Hourly Sicily zonal load (MW) and day-ahead price (EUR/MWh), local time."""
    df = pd.read_csv(path)
    dt = (
        pd.to_datetime(df["utc_timestamp"], utc=True)
        .dt.tz_convert("Europe/Rome")
        .dt.tz_localize(None)
    )
    out = pd.DataFrame(
        {
            "load_mw": df["IT_SICI_load_actual_entsoe_transparency"].to_numpy(),
            "price_eur_per_mwh": df["IT_SICI_price_day_ahead"].to_numpy(),
        },
        index=dt,
    ).sort_index()
    out = out[~out.index.duplicated(keep="first")]

    full_timeline = pd.date_range(SIM_START, SIM_END, freq="h")
    out = out.reindex(full_timeline).ffill().bfill()
    return out


STANDBY_FLOOR = 0.08
SCHOOL_TERM_MONTHS = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12}


def _hours(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy()


def _weekday(index: pd.DatetimeIndex) -> np.ndarray:
    return index.dayofweek.to_numpy() < 5


def residential_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Unchanged zonal shape for a single household."""
    return np.ones(len(index))


def dwelling2_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Larger household: higher daytime use at weekends."""
    is_weekend = index.dayofweek.to_numpy() >= 5
    in_daytime = (_hours(index) >= 9) & (_hours(index) < 18)
    return np.where(is_weekend & in_daytime, 1.15, 1.0)


def apartment_block_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Common services: lower evening peak, higher daytime base load."""
    hours = _hours(index)
    is_evening_peak = (hours >= 19) & (hours < 22)
    is_daytime = (hours >= 8) & (hours < 18)
    mask = np.ones(len(index))
    mask = np.where(is_evening_peak, 0.7, mask)
    mask = np.where(is_daytime, 1.15, mask)
    return mask


def school_mask(index: pd.DatetimeIndex) -> np.ndarray:
    in_term = np.isin(index.month, list(SCHOOL_TERM_MONTHS))
    in_hours = (_hours(index) >= 8) & (_hours(index) < 16)
    active = in_term & _weekday(index) & in_hours
    return np.where(active, 1.0, STANDBY_FLOOR)


def town_hall_mask(index: pd.DatetimeIndex) -> np.ndarray:
    in_hours = (_hours(index) >= 8) & (_hours(index) < 18)
    active = _weekday(index) & in_hours
    return np.where(active, 1.0, STANDBY_FLOOR)


def factory_mask(index: pd.DatetimeIndex) -> np.ndarray:
    in_hours = (_hours(index) >= 6) & (_hours(index) < 22)
    active = _weekday(index) & in_hours
    return np.where(active, 1.0, 0.25)


def retail_shop_mask(index: pd.DatetimeIndex) -> np.ndarray:
    in_hours = (_hours(index) >= 9) & (_hours(index) < 20)
    is_sunday = index.dayofweek.to_numpy() == 6
    active = in_hours & ~is_sunday
    return np.where(active, 1.0, STANDBY_FLOOR)


def wholesale_biz_mask(index: pd.DatetimeIndex) -> np.ndarray:
    in_hours = (_hours(index) >= 7) & (_hours(index) < 18)
    is_saturday = index.dayofweek.to_numpy() == 5
    is_sunday = index.dayofweek.to_numpy() == 6
    active_weekday = _weekday(index) & in_hours
    active_saturday = is_saturday & (_hours(index) >= 7) & (_hours(index) < 13)
    active = active_weekday | active_saturday
    return np.where(active & ~is_sunday, 1.0, STANDBY_FLOOR)


def farm_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Daytime use, higher from April to September (irrigation)."""
    in_daytime = (_hours(index) >= 6) & (_hours(index) < 20)
    seasonal = np.where(np.isin(index.month, [4, 5, 6, 7, 8, 9]), 1.0, 0.6)
    return np.where(in_daytime, seasonal, STANDBY_FLOOR * seasonal)


ENTITY_MASKS = {
    "dwelling1": residential_mask,
    "dwelling2": dwelling2_mask,
    "apartment_block": apartment_block_mask,
    "school": school_mask,
    "town_hall": town_hall_mask,
    "factory": factory_mask,
    "retail_shop": retail_shop_mask,
    "wholesale_biz": wholesale_biz_mask,
    "farm": farm_mask,
}


RETAIL_PRICE_EUR_PER_KWH = {
    "dwelling1": 0.2245,
    "dwelling2": 0.2245,
    "apartment_block": 0.2114,
    "school": 0.2114,
    "town_hall": 0.2114,
    "retail_shop": 0.1770,
    "wholesale_biz": 0.1770,
    "farm": 0.1770,
    "factory": 0.1529,
}


def build_import_price(entity_id: str, price: pd.Series) -> pd.Series:
    """Zonal price plus a fixed retail add-on so the average matches Eurostat."""
    adder = RETAIL_PRICE_EUR_PER_KWH[entity_id] - price.mean()
    return price + adder


ANNUAL_CONSUMPTION_KWH = {
    "dwelling1": 2_700,
    "dwelling2": 3_500,
    "apartment_block": 15_000,
}


def build_consumption(entity_id: str, contract_power: float, opsd_load: pd.Series) -> pd.Series:
    """Zonal load shape x entity mask, scaled to annual kWh for residential
    entities and to peak = contract power for the rest."""
    mask = ENTITY_MASKS[entity_id](opsd_load.index)
    shaped = opsd_load.to_numpy() * mask
    if entity_id in ANNUAL_CONSUMPTION_KWH:
        mean_hourly_kwh = ANNUAL_CONSUMPTION_KWH[entity_id] / 8760
        scaled = shaped * (mean_hourly_kwh / shaped.mean())
    else:
        scaled = shaped * (contract_power / shaped.max())
    return pd.Series(scaled, index=opsd_load.index, name="consumption")


def build_production(pv_power: float, pv_per_kwp: pd.Series) -> pd.Series:
    return (pv_per_kwp * pv_power).rename("production")


def build_prices(opsd_price: pd.Series) -> pd.Series:
    """EUR/MWh to EUR/kWh, negative prices set to 0."""
    return (opsd_price.clip(lower=0) / 1000.0).rename("price")


def build_entity_profile(entity_id: str, spec: dict, opsd: pd.DataFrame, pv_per_kwp: pd.Series) -> pd.DataFrame:
    """Hourly profile for one entity from its config.json entry."""
    price = build_prices(opsd["price_eur_per_mwh"])
    df = pd.DataFrame(index=opsd.index)
    df.index.name = "datetime"

    has_consumption = "contract_power" in spec
    has_production = "Photovoltaic" in spec

    if has_consumption:
        opsd_load = pd.Series(opsd["load_mw"].to_numpy(), index=opsd.index)
        df["consumption"] = build_consumption(entity_id, spec["contract_power"], opsd_load)
    if has_production:
        df["production"] = build_production(spec["Photovoltaic"]["pv_power"], pv_per_kwp)
    if has_consumption:
        df["price_imp"] = build_import_price(entity_id, price)
    if has_production:
        df["price_exp"] = price

    return df


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


GROUP_TO_SUBFOLDER = {
    "Producers": "producers",
    "Consumers": "consumers",
    "Prosumers": "prosumers",
}


def main():
    config = load_config()
    opsd = load_opsd()
    pv_per_kwp = load_pvgis()

    for group, subfolder in GROUP_TO_SUBFOLDER.items():
        entities = config.get(group, {})
        if not entities:
            continue
        out_dir = NODES_DIR / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        for entity_id, spec in entities.items():
            df = build_entity_profile(entity_id, spec, opsd, pv_per_kwp)
            out_path = out_dir / f"{entity_id}.csv"
            df.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S")
            print(f"wrote {out_path.relative_to(REPO_ROOT)}  ({len(df)} rows, columns={list(df.columns)})")


if __name__ == "__main__":
    main()
