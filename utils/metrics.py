"""Rollout and aggregation helpers shared by evaluation and dashboard."""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
import pandas as pd
from environment.microgrid_env import MicrogridEnv


def rule_based_action(observation: np.ndarray) -> np.ndarray:
    """Charge all usable surplus, otherwise discharge to meet deficit."""
    solar, load = observation[1], observation[2]
    return np.array([1.0 if solar > load else -1.0], dtype=np.float32)


def rollout(env: MicrogridEnv, policy: Any, day_index: int, deterministic: bool = True) -> pd.DataFrame:
    obs, _ = env.reset(options={"day_index": day_index})
    records: list[dict[str, Any]] = []
    done = False
    while not done:
        if callable(policy):
            action = policy(obs)
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        records.append(info)
        done = terminated or truncated
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> dict[str, float]:
    solar_total = float(records["solar"].sum())
    solar_utilized = float((records["solar_to_load"] + records["battery_charge"]).sum())
    return {"total_cost": float(records["cost"].sum()), "grid_import": float(records["grid_import"].sum()),
            "grid_export": float(records["grid_export"].sum()), "solar_utilization": solar_utilized,
            "solar_waste": float(records["energy_wasted"].sum()), "battery_usage": float((records["battery_charge"] + records["battery_discharge"]).sum()),
            "battery_cycles": float((records["battery_charge"] + records["battery_discharge"]).sum() / (2 * env_capacity(records))),
            "unmet_load": float(records["unmet_load"].sum()), "renewable_energy_ratio": solar_utilized / max(float(records["load"].sum()), 1e-9),
            "solar_available": solar_total}


def env_capacity(records: pd.DataFrame) -> float:
    # Capacity is a fixed project assumption; helper keeps summary serializable.
    return 10.0
