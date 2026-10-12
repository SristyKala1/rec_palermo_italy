"""Gymnasium wrapper that lets stable-baselines3 agents control SmartCer's
batteries with one community-level reward."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartcer.cer_preprocessing.preprocessing import RemoveFeature
from smartcer.cer_utils import create_action_dict, flatten_observation


def hourly_profit(cer):
    """Community revenue - expense + incentive for every simulated hour."""
    return (
        np.array(cer.cer_energy_revenue)
        - np.array(cer.cer_energy_expense)
        + np.array(cer.cer_actual_tip)
    )


class CommunityBatteryEnv(gym.Env):
    """Each action is a fraction of what the battery can do this hour
    (+1 = charge as much as possible, -1 = discharge down to soc_min).
    Reward = hourly profit minus the profit with idle batteries.
    The year feature is dropped so the model can run on unseen years."""

    def __init__(self, cer, baseline_profit, scaling_from=None, seed=None):
        super().__init__()
        self.cer = cer
        self.baseline_profit = np.asarray(baseline_profit)
        self.seed_ = seed
        self.storage_ids = sorted(cer.storage_nodes_ids)

        if scaling_from is not None:
            cer.encoder = {node: dict(enc) for node, enc in scaling_from.encoder.items()}
        cer.encoder["cer"]["year"] = RemoveFeature()

        obs, _ = cer.reset(seed=seed)
        n_obs = flatten_observation(obs).shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(self.storage_ids),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self.cer.reset(seed=self.seed_ if seed is None else seed)
        return flatten_observation(obs), {}

    def feasible_actions(self, action):
        """Map [-1, 1] fractions to SmartCer actions (energy / rated power)."""
        scaled = []
        for a, node_id in zip(np.clip(action, -1.0, 1.0), self.storage_ids):
            node = self.cer.nodes[node_id]
            battery = node.storage
            if a >= 0:
                limit = min(battery.rated_power, battery.capacity - battery.soc, node.get_current_data("production"))
            else:
                # discharging x kWh lowers the SoC by x / efficiency
                limit = min(battery.rated_power, max(0.0, (battery.soc - battery.soc_min) * battery.ETA))
            scaled.append(a * limit / battery.rated_power)
        return create_action_dict(np.array(scaled, dtype=np.float32), self.storage_ids)

    def step(self, action):
        actions = self.feasible_actions(action)
        obs, _, terminated, _, _, _ = self.cer.step(actions)
        t = self.cer.time_step - 1
        profit = (
            self.cer.cer_energy_revenue[t]
            - self.cer.cer_energy_expense[t]
            + self.cer.cer_actual_tip[t]
        )
        penalty = self.cer.penalties[t] or 0.0
        reward = float(profit - self.baseline_profit[t] - penalty)
        return flatten_observation(obs), reward, bool(terminated), False, {}
