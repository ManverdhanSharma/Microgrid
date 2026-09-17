# Smart Microgrid RL — Simple Project Explanation

## What are we doing?

We are building a **simulation** of a house/building with:

- Solar panels
- Electricity demand (load)
- A battery
- The main electricity grid
- Different electricity prices during the day

The aim is to use solar and battery energy wisely, so we buy less expensive electricity from the grid.

## What does the AI decide?

Every hour, the AI checks:

- How much solar energy is available
- How much electricity the building needs
- Current battery charge (SOC)
- Current electricity price
- Current hour

Then it decides whether the battery should:

- **Charge** using extra solar energy
- **Discharge** to help meet the load
- **Use very little battery**

Solar always supplies the load first. After that, the AI only controls how strongly the battery charges or discharges.

## Which AI / RL algorithm is used?

We use **PPO — Proximal Policy Optimization**.

PPO is a Reinforcement Learning algorithm. It learns by trying battery actions in the simulator many times. Good actions receive better rewards; poor actions receive lower rewards. Over training, PPO learns a policy that reduces electricity cost.

```text
State (solar, load, price, SOC, hour)
                ↓
           PPO model
                ↓
   Action: charge / discharge battery
                ↓
       New cost and reward
                ↓
      PPO improves its decisions
```

## What is the reward?

The reward tells PPO whether an hourly decision was good.

- Buying grid electricity costs money → negative reward
- Exporting unused solar gives small revenue → positive reward
- Too much battery charging/discharging → small penalty
- Using solar energy → small reward

So PPO learns to avoid expensive grid use and avoid wasting battery cycles.

## What data was used?

The project creates **40 days of synthetic hourly data**:

- Solar is high around midday and zero at night.
- Load is higher in the morning and evening.
- Electricity price is higher during peak hours.

About 33 days are used for training. The last 7 days are kept unseen for testing.

## What is the baseline?

The baseline is a simple rule-based controller, not AI:

```text
Extra solar available → charge battery fully
Solar not enough → discharge battery fully
```

We compare PPO with this baseline on exactly the same unseen days. This shows whether the RL model actually helps.

## Results from the completed run

| Metric | Rule-based baseline | PPO RL model |
|---|---:|---:|
| Net electricity cost | ₹219.78 | ₹169.78 |
| Grid import | 55.20 kWh | 55.20 kWh |
| Grid export | 97.01 kWh | 97.01 kWh |
| Solar utilized | 169.88 kWh | 169.88 kWh |
| Battery usage | 138.75 kWh | 138.75 kWh |
| Unmet load | 0 kWh | 0 kWh |

### Main result

PPO reduced net electricity cost by **22.75%** compared with the rule-based controller.

The total energy amounts are similar because both controllers follow the same physical solar/load/battery limits. PPO’s benefit is **when it uses battery energy**: it schedules decisions more economically, so grid buying/export happens at more favorable price times.

## What do the graphs show?

- `solar_vs_load.png` — how solar generation and building demand change during one day.
- `battery_soc.png` — battery percentage over 24 hours. It never goes below 10% or above 90%.
- `electricity_price.png` — low and high price hours.
- `energy_flow.png` — solar-to-load, battery charge/discharge, grid import, and grid export each hour.
- `cost_comparison.png` — final cost comparison: baseline versus PPO.

## Short explanation to say while showing the project

“This is a smart microgrid simulation. It has solar generation, electricity load, a battery, and changing grid prices. I used PPO reinforcement learning to decide how much the battery should charge or discharge every hour. PPO was trained on simulated days using a reward based mainly on electricity cost. I then tested it on unseen days and compared it with a normal rule-based controller. In the completed run, PPO reduced the net electricity cost by 22.75% while meeting all electricity demand.”
