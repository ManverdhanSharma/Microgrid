"""Train PPO on the simulated microgrid."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from environment.microgrid_env import MicrogridEnv
from data.generate_data import generate_energy_data


def train(timesteps: int = 50_000, data_path: str | Path = ROOT / "data/energy_data.csv", model_path: str | Path = ROOT / "models/ppo_microgrid") -> Path:
    """Train and save a PPO policy. Final seven days are held out."""
    data_file = Path(data_path)
    if not data_file.exists():
        generate_energy_data(output_path=data_file)
    data = pd.read_csv(data_file, parse_dates=["timestamp"])
    day_count = len(data) // 24
    train_days = list(range(max(1, day_count - 7)))
    env = Monitor(MicrogridEnv(data, day_indices=train_days, seed=42))
    model = PPO("MlpPolicy", env, verbose=1, seed=42, n_steps=256, batch_size=64, learning_rate=3e-4, gamma=0.99)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    env.close()
    return target.with_suffix(".zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO microgrid controller")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--data", default=str(ROOT / "data/energy_data.csv"))
    args = parser.parse_args()
    print(f"Saved PPO model to {train(args.timesteps, args.data)}")
