import numpy as np
import pytest

from smartcer.cer_env import SmartCer
from rl_env import CommunityBatteryEnv, hourly_profit
from run_baselines import CER_PATH, FixedRBC, make_config, no_storage_actions, run

WEEK = ("2019-07-01 00:00:00", "2019-07-07 23:00:00")


@pytest.fixture(scope="module")
def env():
    cer = SmartCer(cer_path=str(CER_PATH), cer_config=make_config(*WEEK))
    run(cer, no_storage_actions, "baseline")
    return CommunityBatteryEnv(cer, hourly_profit(cer), seed=0)


def rollout(env, policy):
    env.reset()
    rewards = []
    while True:
        _, reward, terminated, _, _ = env.step(policy())
        rewards.append(reward)
        if terminated:
            return np.array(rewards)


def total_violation(env):
    return sum(sum(node.storage.constraint_violation) for node in env.cer.storage_nodes.values())


def test_idle_batteries_earn_zero_reward(env):
    rewards = rollout(env, lambda: np.zeros(env.action_space.shape))
    assert np.allclose(rewards, 0.0)


@pytest.mark.parametrize("fill", [-1.0, 1.0])
def test_extreme_actions_are_always_feasible(env, fill):
    rollout(env, lambda: np.full(env.action_space.shape, fill))
    assert total_violation(env) == pytest.approx(0.0, abs=1e-3)


def test_random_actions_are_always_feasible(env):
    rng = np.random.default_rng(0)
    rollout(env, lambda: rng.uniform(-1, 1, env.action_space.shape))
    assert total_violation(env) == pytest.approx(0.0, abs=1e-3)


def test_soc_stays_within_limits(env):
    rng = np.random.default_rng(1)
    rollout(env, lambda: rng.uniform(-1, 1, env.action_space.shape))
    for node in env.cer.storage_nodes.values():
        soc = np.array(node.storage.soc_final_list)
        assert soc.max() <= node.storage.capacity + 1e-6
        assert soc.min() >= node.storage.soc_min - 1e-3


def test_full_charge_uses_all_available_pv(env):
    env.reset()
    for _ in range(12):
        env.step(np.zeros(env.action_space.shape))
    node = env.cer.nodes["town_hall"]
    battery = node.storage
    expected = min(battery.rated_power, battery.capacity - battery.soc, node.get_current_data("production"))
    env.step(np.ones(env.action_space.shape))
    assert expected > 0
    assert battery.actual_charges[-1] == pytest.approx(expected, rel=1e-4)


def test_fixed_rbc_requests_the_intended_energy():
    cer = SmartCer(cer_path=str(CER_PATH), cer_config=make_config(*WEEK))
    rbc = FixedRBC(storage_node=cer.storage_nodes, producer_ids=cer.producers_ids, prosumer_ids=cer.prosumers_ids)
    _, info = cer.reset(seed=0)
    for _ in range(cer.sim_hours):
        actions = rbc.choose_actions(info)
        for node_id, action in actions.items():
            s = info[node_id]
            surplus = s["production"] - s["consumption"]
            if surplus > 0:
                intended = min(surplus, s["capacity"] - s["soc"], s["rated_power"])
                assert action[0] * s["rated_power"] == pytest.approx(intended, rel=1e-4, abs=1e-4)
        _, _, terminated, _, info, _ = cer.step(actions)
        if terminated:
            break


def test_fixed_rbc_keeps_soc_above_minimum():
    cer = SmartCer(cer_path=str(CER_PATH), cer_config=make_config(*WEEK))
    rbc = FixedRBC(storage_node=cer.storage_nodes, producer_ids=cer.producers_ids, prosumer_ids=cer.prosumers_ids)
    run(cer, lambda c, info: rbc.choose_actions(info), "rbc")
    for node in cer.storage_nodes.values():
        assert min(node.storage.soc_final_list) >= node.storage.soc_min - 1e-3
        assert sum(node.storage.constraint_violation) == pytest.approx(0.0, abs=1e-3)
