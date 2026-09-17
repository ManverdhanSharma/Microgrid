"""Compare PPO against the rule-based controller on held-out days."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from environment.microgrid_env import MicrogridEnv
from utils.metrics import rollout, rule_based_action, summarize


def evaluate(data_path: str | Path = ROOT / "data/energy_data.csv", model_path: str | Path = ROOT / "models/ppo_microgrid.zip", output_path: str | Path = ROOT / "plots/evaluation_summary.csv") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both policies on the final seven unseen days and save a metric table."""
    data = pd.read_csv(data_path, parse_dates=["timestamp"])
    days = len(data) // 24
    eval_days = list(range(max(0, days - 7), days))
    env = MicrogridEnv(data, day_indices=eval_days, seed=123)
    model = PPO.load(model_path)
    base_frames = [rollout(env, rule_based_action, d) for d in eval_days]
    ppo_frames = [rollout(env, model, d) for d in eval_days]
    baseline_records, ppo_records = pd.concat(base_frames, ignore_index=True), pd.concat(ppo_frames, ignore_index=True)
    baseline, ppo = summarize(baseline_records), summarize(ppo_records)
    metrics = ["total_cost", "grid_import", "grid_export", "solar_utilization", "solar_waste", "battery_usage", "battery_cycles", "unmet_load", "renewable_energy_ratio"]
    table = pd.DataFrame({"Metric": metrics, "Baseline": [baseline[m] for m in metrics], "PPO": [ppo[m] for m in metrics]})
    cost_change = (baseline["total_cost"] - ppo["total_cost"]) / max(abs(baseline["total_cost"]), 1e-9) * 100
    print("\nEvaluation on held-out simulated days\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nPPO cost difference vs baseline: {cost_change:+.2f}% (positive means PPO costs less)")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    return table, baseline_records, ppo_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/energy_data.csv"))
    parser.add_argument("--model", default=str(ROOT / "models/ppo_microgrid.zip"))
    args = parser.parse_args()
    evaluate(args.data, args.model)
