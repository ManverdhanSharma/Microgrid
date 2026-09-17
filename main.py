"""Simple end-to-end command line workflow."""
from __future__ import annotations
import argparse
from pathlib import Path
from data.generate_data import generate_energy_data
from training.train_ppo import train
from evaluation.evaluate import evaluate
from utils.plots import generate_plots

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-based smart microgrid RL project")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not any([args.generate_data, args.train, args.evaluate, args.all]): parser.print_help(); return
    if args.generate_data or args.all:
        data = generate_energy_data(seed=args.seed, output_path=ROOT / "data/energy_data.csv")
        print(f"Generated {len(data)} hourly records in data/energy_data.csv")
    if args.train or args.all:
        print(f"Training PPO for {args.timesteps:,} timesteps...")
        print(f"Saved model: {train(args.timesteps)}")
    if args.evaluate or args.all:
        summary, baseline, ppo = evaluate()
        generate_plots(baseline, ppo, summary, ROOT / "plots")
        print("Saved five plots in plots/")


if __name__ == "__main__": main()
