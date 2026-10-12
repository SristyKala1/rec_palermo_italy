"""Train a SAC agent to run the community's batteries on 2015-2018, then
compare it with no storage and the rule-based controller on 2019-2020.

Usage:
    python scripts/train_sac.py --mix full --seed 42
    python scripts/train_sac.py --mix households_only --seed 42 --eval-only
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC
from tqdm import tqdm

from smartcer.cer_env import SmartCer
from build_mixes import MIXES, cer_path
from rl_env import CommunityBatteryEnv, hourly_profit
from run_baselines import RESULTS_DIR, SEED, FixedRBC, make_config, no_storage_actions, run, save

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

TRAIN = ("2015-01-01 00:00:00", "2018-12-31 23:00:00")
TEST = ("2019-01-01 00:00:00", "2020-09-30 23:00:00")
TEST_DIR = "test_2019_2020"


def model_path(mix, seed):
    return MODELS_DIR / f"sac_{mix}_seed{seed}"


def build(mix, period):
    """SmartCer env for a mix and period plus its no-storage run and hourly profit."""
    config = make_config(*period)
    cer = SmartCer(cer_path=str(cer_path(mix)), cer_config=config)
    results = run(cer, no_storage_actions, f"{mix} no storage {period[0][:4]}-{period[1][:4]}")
    return cer, config, results, hourly_profit(cer)


def train(mix, seed, steps):
    cer, _, _, baseline = build(mix, TRAIN)
    env = CommunityBatteryEnv(cer, baseline, seed=seed)
    model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    model.save(model_path(mix, seed))
    return cer


def evaluate(mix, seed, train_cer=None):
    if train_cer is None:
        train_cer = SmartCer(cer_path=str(cer_path(mix)), cer_config=make_config(*TRAIN))
    cer, config, no_storage, baseline = build(mix, TEST)
    out = f"{TEST_DIR}/{mix}"

    if not (RESULTS_DIR / out / "rbc" / "kpi.json").exists():
        save(no_storage, f"{out}/no_storage", config)
        rbc = FixedRBC(storage_node=cer.storage_nodes, producer_ids=cer.producers_ids, prosumer_ids=cer.prosumers_ids)
        save(run(cer, lambda c, info: rbc.choose_actions(info), f"{mix} rbc"), f"{out}/rbc", config)

    env = CommunityBatteryEnv(cer, baseline, scaling_from=train_cer, seed=SEED)
    model = SAC.load(model_path(mix, seed), device="cpu")
    obs, _ = env.reset()
    for _ in tqdm(range(cer.sim_hours), desc=f"{mix} sac seed {seed}"):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    kpis = save(cer.get_sim_results(), f"{out}/sac_seed{seed}", config)
    print(json.dumps(kpis, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=MIXES, default="full")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    train_cer = None if args.eval_only else train(args.mix, args.seed, args.steps)
    evaluate(args.mix, args.seed, train_cer)


if __name__ == "__main__":
    main()
