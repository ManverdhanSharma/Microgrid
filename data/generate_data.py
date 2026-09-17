"""Create reproducible, realistic-looking hourly microgrid data."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_energy_data(days: int = 40, seed: int = 42, output_path: str | Path = "data/energy_data.csv") -> pd.DataFrame:
    """Generate hourly solar, demand, and time-of-use tariff data.

    The final seven days are conventionally kept for evaluation, leaving about a
    month of varied days for PPO training.
    """
    if days < 2:
        raise ValueError("days must be at least 2")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    rows: list[dict[str, object]] = []
    for day in range(days):
        # Weather affects every solar hour in a day, not just independent noise.
        weather = rng.uniform(0.58, 1.08)
        demand_scale = rng.normal(1.0, 0.07)
        for hour in range(24):
            solar_shape = max(0.0, np.sin(np.pi * (hour - 6) / 12))
            solar = max(0.0, 5.2 * solar_shape * weather + rng.normal(0, 0.12))
            morning_peak = 1.15 * np.exp(-((hour - 8) / 2.1) ** 2)
            evening_peak = 2.0 * np.exp(-((hour - 19) / 2.8) ** 2)
            base_load = 0.72 + morning_peak + evening_peak + (0.25 if 11 <= hour <= 16 else 0)
            load = max(0.3, base_load * demand_scale + rng.normal(0, 0.12))
            price = 4.2
            if 7 <= hour <= 10:
                price = 6.4
            if 18 <= hour <= 22:
                price = 8.2
            price = max(2.5, price + rng.normal(0, 0.25))
            rows.append({"timestamp": timestamps[day * 24 + hour], "solar_generation_kwh": round(solar, 3), "load_kwh": round(load, 3), "electricity_price": round(price, 3)})
    data = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return data
