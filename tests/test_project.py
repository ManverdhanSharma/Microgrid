from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from data.generate_data import generate_energy_data
from environment.microgrid_env import MicrogridEnv


def test_dataset_generation(tmp_path: Path):
    frame = generate_energy_data(days=3, seed=7, output_path=tmp_path / "energy.csv")
    assert len(frame) == 72
    assert {"timestamp", "solar_generation_kwh", "load_kwh", "electricity_price"}.issubset(frame.columns)


def test_reset_step_soc_and_energy_balance(tmp_path: Path):
    frame = generate_energy_data(days=2, output_path=tmp_path / "energy.csv")
    env = MicrogridEnv(frame)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (5,)
    for action in (np.array([1.0]), np.array([-1.0])):
        _, _, _, _, info = env.step(action)
        assert env.min_soc <= info["soc"] <= env.max_soc
        assert info["solar_to_load"] + info["grid_import"] + info["battery_discharge"] == pytest.approx(info["load"])
        assert info["solar_to_load"] + info["battery_charge"] + info["grid_export"] + info["energy_wasted"] == pytest.approx(info["solar"])


import pytest


def test_model_save_and_load(tmp_path: Path):
    frame = generate_energy_data(days=2, output_path=tmp_path / "energy.csv")
    model = PPO("MlpPolicy", MicrogridEnv(frame), n_steps=8, batch_size=8)
    model_path = tmp_path / "ppo_test"
    model.save(model_path)
    loaded = PPO.load(model_path)
    action, _ = loaded.predict(MicrogridEnv(frame).reset()[0])
    assert action.shape == (1,)
