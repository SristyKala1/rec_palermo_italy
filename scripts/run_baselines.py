"""Run the no-storage and rule-based (RBC) baselines over the full simulation
period and save SmartCer's per-hour results and KPIs.

Usage: python scripts/run_baselines.py
"""

import collections
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from smartcer.cer_env import SmartCer
from smartcer.cer_metrics.scores import compute_kpis
from smartcer.cer_utils import save_results_to_csv
from smartcer.storage_controllers.rbc_controller import StandardRBC

REPO_ROOT = Path(__file__).resolve().parents[1]
CER_PATH = REPO_ROOT / "data" / "config"
RESULTS_DIR = REPO_ROOT / "results" / "simulations"

SIM_START = "2015-01-01 00:00:00"
SIM_END = "2020-09-30 23:00:00"
SEED = 42

CER_CONFIG = {
    "simulation": {
        "start": SIM_START,
        "end": SIM_END,
        "delta_input": "1h",
        "delta_output": "YE",
        "seed": SEED,
    },
    "reward": {
        "type": "profit",
        "single_agent": False,
        "constraints": {"penalty": True, "coefficient": 0.3, "max_ratio": 0.99},
        "scaler": {"apply": True, "clip_min": None, "clip_max": None, "reset_interval": None},
    },
}


class FixedRBC(StandardRBC):
    """StandardRBC normalises its action by capacity, but the env scales actions
    by rated power. Rescale so the requested energy is what the rule intended."""

    def choose_actions(self, state):
        actions = super().choose_actions(state)
        for node_id, action in actions.items():
            ratio = state[node_id]["capacity"] / state[node_id]["rated_power"]
            actions[node_id] = np.clip(action * ratio, -1.0, 1.0).astype(np.float32)
        return actions


def no_storage_actions(cer, info):
    return collections.OrderedDict(
        (node_id, np.array([0.0], dtype=np.float32)) for node_id in sorted(cer.storage_nodes_ids)
    )


def run(cer, choose_actions, label):
    """Simulate one full episode and return SmartCer's results dict."""
    _, info = cer.reset(seed=SEED)
    for _ in tqdm(range(cer.sim_hours), desc=label):
        actions = choose_actions(cer, info)
        _, _, terminated, _, info, _ = cer.step(actions)
        if terminated:
            break
    return cer.get_sim_results()


def save(results, name):
    out_dir = RESULTS_DIR / name
    os.makedirs(out_dir, exist_ok=True)
    save_results_to_csv(csv_path=str(out_dir), results=results)
    kpis = compute_kpis(results_cer=results["cer"])
    with open(out_dir / "kpi.json", "w") as f:
        json.dump(kpis, f, indent=4)
    with open(out_dir / "sim_config.json", "w") as f:
        json.dump(CER_CONFIG, f, indent=4)
    return kpis


def main():
    cer = SmartCer(cer_path=str(CER_PATH), cer_config=CER_CONFIG)
    rbc = FixedRBC(storage_node=cer.storage_nodes, producer_ids=cer.producers_ids, prosumer_ids=cer.prosumers_ids)

    strategies = {
        "no_storage": no_storage_actions,
        "rbc": lambda cer, info: rbc.choose_actions(info),
    }
    for name, choose_actions in strategies.items():
        results = run(cer, choose_actions, name)
        kpis = save(results, name)
        print(name, json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
