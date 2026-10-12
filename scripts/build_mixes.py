"""Build the community mixes for the load-diversity study.

Every mix has the same nine members with the same PV, batteries, tariffs and
annual demand as the full community. The only change is that some members
get a household load shape instead of their own, so differences between
mixes come from how varied the load profiles are.

Usage: python scripts/build_mixes.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from build_consumption_profiles import (
    CONFIG_PATH,
    GROUP_TO_SUBFOLDER,
    build_entity_profile,
    load_config,
    load_opsd,
    load_pvgis,
    residential_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MIXES_DIR = REPO_ROOT / "data" / "mixes"

# members that take a household load shape in each mix
MIXES = {
    "full": [],
    "no_industry": ["factory", "farm"],
    "homes_and_public": ["factory", "farm", "retail_shop", "wholesale_biz"],
    "households_only": ["factory", "farm", "retail_shop", "wholesale_biz", "school", "town_hall", "apartment_block"],
}


def cer_path(mix):
    """Config folder SmartCer should load for a mix."""
    return CONFIG_PATH.parent if mix == "full" else MIXES_DIR / mix


def household_shaped(consumption, opsd_load):
    """Household load shape with the same total demand as `consumption`."""
    shape = opsd_load.to_numpy() * residential_mask(opsd_load.index)
    return shape * (consumption.sum() / shape.sum())


def main():
    config = load_config()
    opsd = load_opsd()
    pv_per_kwp = load_pvgis()
    opsd_load = pd.Series(opsd["load_mw"].to_numpy(), index=opsd.index)

    for mix, swapped in MIXES.items():
        if mix == "full":
            continue
        out = MIXES_DIR / mix
        if out.exists():
            shutil.rmtree(out)
        for group, subfolder in GROUP_TO_SUBFOLDER.items():
            for entity_id, spec in config.get(group, {}).items():
                df = build_entity_profile(entity_id, spec, opsd, pv_per_kwp)
                if entity_id in swapped:
                    df["consumption"] = household_shaped(df["consumption"], opsd_load)
                (out / "nodes" / subfolder).mkdir(parents=True, exist_ok=True)
                df.to_csv(out / "nodes" / subfolder / f"{entity_id}.csv", date_format="%Y-%m-%d %H:%M:%S")
        with open(out / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"wrote {out.relative_to(REPO_ROOT)}  (household shape: {', '.join(swapped)})")


if __name__ == "__main__":
    main()
