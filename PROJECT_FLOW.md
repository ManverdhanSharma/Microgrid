# Smart Microgrid RL — Project Working Flow

This document explains how the project works from data generation to final results.

## Complete project flow

```mermaid
flowchart TD
    A[1. Generate synthetic energy data] --> B[Hourly records for 40 days]
    B --> B1[Solar generation]
    B --> B2[Electricity load]
    B --> B3[Electricity price]

    B --> C[2. Create Microgrid Environment]
    C --> C1[One episode = 24 hours = one day]
    C --> C2[Battery model + energy-flow rules]

    C --> D[3. PPO observes current state]
    D --> D1[Battery SOC]
    D --> D2[Current solar]
    D --> D3[Current load]
    D --> D4[Current grid price]
    D --> D5[Current hour]

    D --> E[4. PPO chooses one battery action]
    E --> E1[-1 = strong discharge]
    E --> E2[0 = hold / little battery use]
    E --> E3[+1 = strong charge]

    E --> F[5. Environment applies energy rules]
    F --> G[6. Calculate cost and reward]
    G --> H[7. PPO learns from the reward]
    H --> D

    H --> I[After training: save PPO model]
    I --> J[8. Test on unseen days]
    J --> K[Compare PPO against rule-based controller]
    K --> L[Metrics, graphs, Streamlit dashboard]
```

## What happens in one hour?

```mermaid
flowchart TD
    A[Start hour: read solar, load, price, and battery SOC] --> B[Solar supplies load first]
    B --> C{Is solar greater than load?}

    C -->|Yes: solar surplus| D[PPO action decides how much surplus solar to charge into battery]
    D --> E[Remaining solar is exported to the grid]

    C -->|No: solar deficit| F[PPO action decides how much battery energy to discharge]
    F --> G[Remaining demand is bought from the grid]

    E --> H[Update battery SOC]
    G --> H
    H --> I[Calculate grid cost, export revenue, battery-use penalty, and reward]
    I --> J[Move to next hour]
```

## Example: sunny afternoon

```text
Solar = 4.8 kWh
Load  = 2.1 kWh

1. Solar directly serves the load: 2.1 kWh
2. Solar surplus: 2.7 kWh
3. PPO gives a positive action
4. Battery charges with the allowed amount of surplus
5. Any solar left over is exported to the grid
6. Battery SOC is increased after 95% charging efficiency
```

## Example: expensive evening hour

```text
Solar = 0.0 kWh
Load  = 3.0 kWh
Price = high

1. There is no solar, so the load has a 3.0 kWh deficit.
2. PPO may give a negative action to discharge the battery.
3. Battery supplies as much as it safely can.
4. The grid supplies only what remains.
5. Less grid import at a high price gives a better reward.
```

## Battery flow

```mermaid
flowchart LR
    S[Solar surplus] -->|Positive PPO action| B[Battery]
    B -->|Negative PPO action| L[Load deficit]
    B --> SOC[Battery SOC: kept between 10% and 90%]
    SOC --> B
```

Battery limits used in this project:

| Item | Value |
|---|---:|
| Capacity | 10 kWh |
| Starting SOC | 50% |
| Minimum safe SOC | 10% |
| Maximum safe SOC | 90% |
| Maximum charge rate | 5 kWh/hour |
| Maximum discharge rate | 5 kWh/hour |
| Charge/discharge efficiency | 95% |

## RL terms used in this project

| Term | Meaning here |
|---|---|
| Reinforcement Learning | Learning decisions by trying actions in a simulator and receiving a reward. |
| Agent | The PPO model that decides the battery charge/discharge action. |
| Environment | `MicrogridEnv`: the simulated solar, load, battery, grid, and price system. |
| State / observation | The information shown to PPO: SOC, solar, load, price, and hour. |
| Action | A number from `-1` to `+1`. Negative requests discharge; positive requests charge. |
| Reward | A score after every hour. Higher reward means a better energy-management decision. |
| Episode | One full simulated day of 24 hourly decisions. |
| Timestep | One hour in an episode. |
| Policy | The neural network inside PPO that maps the state to an action. |
| Training | Repeating many simulated days and updating PPO so it gets better rewards. |
| Evaluation | Running the already-trained PPO model on days it did not train on. |
| Baseline | The simple non-RL controller used as a fair comparison. |

## Why the reward makes sense

```mermaid
flowchart LR
    GI[Grid import] -->|cost: bad| R[Hourly reward]
    EX[Grid export] -->|small revenue: good| R
    BC[Excess battery cycling] -->|small penalty| R
    SU[Using solar directly or charging with solar] -->|small bonus| R
```

The reward is mainly based on money:

```text
reward = - cost of grid import
         + revenue from grid export
         - small battery-cycling penalty
         + small solar-use reward
```

So the agent gradually learns that buying grid electricity is undesirable—especially at expensive hours—and that battery energy should be used carefully.

## Training versus evaluation

```mermaid
flowchart LR
    A[40 days of generated data] --> B[First ~33 days: training]
    B --> C[PPO repeatedly interacts and updates its policy]
    A --> D[Final 7 days: unseen evaluation]
    C --> E[Freeze trained PPO model]
    E --> D
    D --> F[Run PPO and rule-based controller]
    F --> G[Compare cost, grid import, solar use, battery use]
```

The model is not given correct actions. It learns because actions that lower grid cost receive better rewards over many training episodes.

## Rule-based baseline flow

```text
If solar > load: charge battery as much as possible.
If solar < load: discharge battery as much as possible.
Otherwise: use the grid for anything left.
```

This rule does not learn from price or previous experience. PPO is evaluated against it on the same unseen days, using the same battery and energy rules.

## Output produced by the project

```mermaid
flowchart LR
    T[Trained PPO model] --> E[Evaluation]
    E --> M[Metrics table]
    E --> P[Plots]
    E --> D[Streamlit dashboard]
    M --> M1[Cost]
    M --> M2[Grid import/export]
    M --> M3[Solar utilization]
    M --> M4[Battery usage]
    P --> P1[Solar vs load]
    P --> P2[Battery SOC]
    P --> P3[Price]
    P --> P4[Energy flow]
    P --> P5[Baseline vs PPO cost]
```
