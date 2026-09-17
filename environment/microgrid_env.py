"""Gymnasium environment for one-day battery control in a solar microgrid."""
from __future__ import annotations

from typing import Any, Sequence
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class MicrogridEnv(gym.Env[np.ndarray, np.ndarray]):
    """Hourly microgrid simulator with a continuous charge/discharge action.

    Positive actions request charging from solar surplus; negative actions request
    discharge to cover a solar shortfall.  Grid import/export is resolved after
    that battery decision.
    """
    metadata = {"render_modes": []}
    battery_capacity = 10.0
    min_soc = 0.10
    max_soc = 0.90
    initial_soc = 0.50
    max_charge_rate = 5.0
    max_discharge_rate = 5.0
    charge_efficiency = 0.95
    discharge_efficiency = 0.95
    export_price_factor = 0.35

    def __init__(self, data: pd.DataFrame, day_indices: Sequence[int] | None = None, seed: int | None = None):
        super().__init__()
        required = {"timestamp", "solar_generation_kwh", "load_kwh", "electricity_price"}
        if not required.issubset(data.columns):
            raise ValueError(f"Data is missing columns: {required - set(data.columns)}")
        self.data = data.copy().reset_index(drop=True)
        if len(self.data) % 24:
            raise ValueError("Data must contain complete 24-hour days")
        self.days = [self.data.iloc[i:i + 24].reset_index(drop=True) for i in range(0, len(self.data), 24)]
        self.day_indices = list(day_indices) if day_indices is not None else list(range(len(self.days)))
        if not self.day_indices:
            raise ValueError("day_indices cannot be empty")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Values are normalized: SOC [0,1], solar/load [0,1], price [0,1], hour [0,1].
        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(5,), dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_day_index = self.day_indices[0]
        self.hour = 0
        self.soc = self.initial_soc

    def _row(self) -> pd.Series:
        return self.days[self.current_day_index].iloc[self.hour]

    def _observation(self) -> np.ndarray:
        row = self._row()
        return np.array([self.soc, row.solar_generation_kwh / 6.0, row.load_kwh / 5.0, row.electricity_price / 10.0, self.hour / 23.0], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        requested_day = options.get("day_index")
        if requested_day is not None:
            if requested_day not in self.day_indices:
                raise ValueError("Requested day is not available in this environment")
            self.current_day_index = int(requested_day)
        else:
            self.current_day_index = int(self.np_random.choice(self.day_indices))
        self.hour = 0
        self.soc = self.initial_soc
        return self._observation(), {"day_index": self.current_day_index}

    def step(self, action: np.ndarray):
        action_value = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
        row = self._row()
        solar, load, price = float(row.solar_generation_kwh), float(row.load_kwh), float(row.electricity_price)
        solar_to_load = min(solar, load)
        surplus = max(0.0, solar - load)
        deficit = max(0.0, load - solar)
        soc_energy = self.soc * self.battery_capacity
        charge_room_input = (self.max_soc * self.battery_capacity - soc_energy) / self.charge_efficiency
        discharge_available = (soc_energy - self.min_soc * self.battery_capacity) * self.discharge_efficiency
        battery_charge = 0.0
        battery_discharge = 0.0
        if action_value > 0 and surplus > 0:
            battery_charge = min(action_value * self.max_charge_rate, surplus, max(0.0, charge_room_input))
            self.soc += battery_charge * self.charge_efficiency / self.battery_capacity
        elif action_value < 0 and deficit > 0:
            battery_discharge = min(-action_value * self.max_discharge_rate, deficit, max(0.0, discharge_available))
            self.soc -= battery_discharge / (self.discharge_efficiency * self.battery_capacity)
        self.soc = float(np.clip(self.soc, self.min_soc, self.max_soc))
        grid_import = max(0.0, deficit - battery_discharge)
        grid_export = max(0.0, surplus - battery_charge)
        # Export is unrestricted in this small model, so curtailment is zero.
        energy_wasted = 0.0
        grid_cost = grid_import * price
        export_revenue = grid_export * price * self.export_price_factor
        solar_waste_penalty = energy_wasted * 1.0
        cycle_penalty = (battery_charge + battery_discharge) * 0.025
        renewable_reward = (solar_to_load + battery_charge) * 0.02
        reward = -grid_cost + export_revenue - solar_waste_penalty - cycle_penalty + renewable_reward
        info = {"hour": self.hour, "timestamp": str(row.timestamp), "solar": solar, "load": load, "price": price, "action": action_value,
                "soc": self.soc, "solar_to_load": solar_to_load, "battery_charge": battery_charge, "battery_discharge": battery_discharge,
                "grid_import": grid_import, "grid_export": grid_export, "energy_wasted": energy_wasted, "unmet_load": 0.0,
                "cost": grid_cost - export_revenue, "reward": reward}
        self.hour += 1
        terminated = self.hour >= 24
        if terminated:
            # Gymnasium expects a valid observation even at terminal state.
            observation = np.zeros(5, dtype=np.float32)
        else:
            observation = self._observation()
        return observation, float(reward), terminated, False, info
