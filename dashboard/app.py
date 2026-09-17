"""Interactive Streamlit view of a trained microgrid policy."""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from environment.microgrid_env import MicrogridEnv
from utils.metrics import rollout, summarize

st.set_page_config(page_title="Smart Microgrid RL", page_icon="⚡", layout="wide")
st.title("⚡ AI Smart Microgrid Energy Management")
data_path, model_path = ROOT / "data/energy_data.csv", ROOT / "models/ppo_microgrid.zip"
if not data_path.exists() or not model_path.exists():
    st.warning("Generate data and train the model first: `python main.py --all`"); st.stop()
data = pd.read_csv(data_path, parse_dates=["timestamp"])
total_days = len(data) // 24
selected_day = st.sidebar.selectbox("Simulated day", range(total_days), index=total_days - 1, format_func=lambda x: f"Day {x + 1} ({data.iloc[x * 24].timestamp.date()})")
if st.sidebar.button("Run RL agent", type="primary") or "records" not in st.session_state:
    env = MicrogridEnv(data, day_indices=list(range(total_days)))
    st.session_state.records = rollout(env, PPO.load(model_path), selected_day)
records = st.session_state.records
metrics = summarize(records)
cols = st.columns(5)
for col, label, value in zip(cols, ["Electricity cost", "Grid consumption", "Solar utilization", "Battery usage", "Renewable energy"], [f"₹{metrics['total_cost']:.2f}", f"{metrics['grid_import']:.1f} kWh", f"{metrics['solar_utilization']:.1f} kWh", f"{metrics['battery_usage']:.1f} kWh", f"{metrics['renewable_energy_ratio'] * 100:.1f}%"]): col.metric(label, value)
st.subheader("Solar vs Load"); st.line_chart(records.set_index("hour")[["solar", "load"]])
c1, c2 = st.columns(2)
with c1: st.subheader("Battery SOC"); st.line_chart(records.set_index("hour")[["soc"]] * 100)
with c2: st.subheader("Electricity Price"); st.line_chart(records.set_index("hour")[["price"]])
st.subheader("Energy Flow"); st.area_chart(records.set_index("hour")[["solar_to_load", "battery_charge", "battery_discharge", "grid_import", "grid_export"]])
st.subheader("RL Agent Decisions")
def decision(row: pd.Series) -> str:
    if row.action > 0.05: return "Charge Battery"
    if row.action < -0.05: return "Discharge Battery"
    return "Hold Battery"
view = records[["hour", "solar", "load", "soc", "price", "action"]].copy(); view["decision"] = records.apply(decision, axis=1); view["soc"] = (view["soc"] * 100).round(1); view.columns = ["Hour", "Solar (kWh)", "Load (kWh)", "Battery SOC (%)", "Price (₹/kWh)", "Action", "RL Agent Decision"]
st.dataframe(view, use_container_width=True, hide_index=True)
