"""Matplotlib reporting for a representative evaluated day."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(baseline: pd.DataFrame, ppo: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path = "plots") -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    day = ppo.iloc[:24].copy(); hours = day["hour"]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(hours, day["solar"], label="Solar generation", color="#f4b400"); ax.plot(hours, day["load"], label="Load", color="#2463a6"); ax.set(xlabel="Hour", ylabel="kWh", title="Solar Generation and Load"); ax.legend(); fig.tight_layout(); fig.savefig(output / "solar_vs_load.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(hours, day["soc"] * 100, marker="o", color="#2e8b57"); ax.set(xlabel="Hour", ylabel="SOC (%)", title="Battery State of Charge", ylim=(0, 100)); fig.tight_layout(); fig.savefig(output / "battery_soc.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.step(hours, day["price"], where="mid", color="#8e44ad"); ax.set(xlabel="Hour", ylabel="Price (Rs/kWh)", title="Electricity Price"); fig.tight_layout(); fig.savefig(output / "electricity_price.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4));
    for column, label in [("solar_to_load", "Solar to load"), ("battery_charge", "Solar to battery"), ("battery_discharge", "Battery to load"), ("grid_import", "Grid to load"), ("grid_export", "Solar to grid")]: ax.plot(hours, day[column], label=label)
    ax.set(xlabel="Hour", ylabel="kWh", title="PPO Energy Flow"); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(output / "energy_flow.png", dpi=150); plt.close(fig)
    costs = summary.set_index("Metric").loc["total_cost"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["Rule-based", "PPO"], [costs["Baseline"], costs["PPO"]], color=["#95a5a6", "#2e8b57"]); ax.set(ylabel="Net electricity cost (Rs)", title="Cost Comparison"); fig.tight_layout(); fig.savefig(output / "cost_comparison.png", dpi=150); plt.close(fig)
