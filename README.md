# India EV Supply Chain: Action Projection RL Decision Support System

This repository contains a Deep Reinforcement Learning (DRL) based Decision Support System (DSS) designed to optimize Electric Vehicle (EV) battery-material procurement in India. It simulates a digital twin of an EV supply chain subjected to severe stochastic disruptions, managing procurement under a strict corporate financial cap using an Action Projection Layer. 

The project uses Proximal Policy Optimization (PPO) and compares the trained AI agent against traditional rule-based algorithms (Greedy/JIT and Min-Max Inventory) through an interactive, multi-seed Monte Carlo Streamlit dashboard.

---

## 🚀 Features

* **Digital Twin Environment (`action_procure.py`):** A custom Gymnasium environment simulating monthly battery-material sourcing. It features stochastic market volatility and four disruption types:
  * Port Delays
  * Supplier Defaults
  * Geopolitical Tariff Shocks
  * Recycled Material Quality Issues
* **Action Projection Layer:** Enforces hard budget constraints ($120M cap). If an action exceeds the budget, the projection layer mathematically scales down the raw procurement values before environment execution, preventing financial violations while allowing the agent to learn from its intent.
* **Complex Action Space:** The agent jointly controls virgin material sourcing, domestic recycled sourcing (PLI subsidy eligible), expedited air-freight logistics, and LFP chemistry pivoting (reducing critical mineral dependence).
* **Multi-Seed Training (`train.py`):** Utilizes `stable-baselines3` to train a highly robust PPO agent across parallel vectorized environments.
* **Interactive Command Center (`dashboard1.py`):** A fully featured Streamlit dashboard providing descriptive narratives, financial trajectory comparisons, and multi-seed statistical baselines (Mean, Std Dev, Variance Box Plots).

---

## 📂 Repository Structure

* **`action_procure.py`**: The Gymnasium environment (`IndiaEVProcurementEnv`) defining the MDP, reward function, and action projection mechanics.
* **`train.py`**: The training script leveraging Stable Baselines3 to train the PPO model for 150k timesteps.
* **`dashboard1.py`**: The Streamlit web application for evaluating the RL agent against baselines.
* **`minerals_trade.csv` & `recycling_flows.csv`**: External datasets for dynamically sizing virgin and recycled baseline capacities (the environment falls back to defaults if these are empty).
* **`ppo_india_action_proj.zip`**: The saved PPO model weights (generated after running `train.py`).

---

## 🛠️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/india-ev-supply-chain.git
   cd india-ev-supply-chain
   ```

2. Install the required dependencies:
   * Ensure you have Python 3.8+ installed.
   ```bash
   pip install gymnasium stable-baselines3 numpy pandas streamlit plotly
   ```

---

## 🧠 Usage

### 1. Train the RL Agent
Before running the dashboard, you must train the PPO model so the system has weights to evaluate:
```bash
python train.py
```
*This will launch 5 parallel vectorized environments to prevent overfitting and output the `ppo_india_action_proj.zip` model file.*

### 2. Launch the Command Center Dashboard
Once the model is trained, start the interactive Streamlit DSS:
```bash
streamlit run dashboard1.py
```
* **Simulation Horizon:** Adjust the slider to test different month spans (3 to 24 months).
* **Evaluation Runs:** Adjust the seed slider to test statistical rigor across multiple parallel stochastic futures.
* **Execute:** Click "Execute Prescriptive Analysis" to view financial trajectories, baseline variance box-plots, and autonomous agent directives.

---

## 📊 Technical Details

### Observation Space (State)
A continuous 6-dimensional space tracking:
* `[Current Inventory, Virgin Capacity, Recycled Capacity, Disruption Code, Virgin Spot Price, Budget Limit]`

### Action Space
A continuous 4-dimensional array (`[0.0, 1.0]`):
* **Virgin Ratio:** Volume of imported virgin material to order.
* **Recycled Ratio:** Volume of domestic recycled material to order.
* **Expedite Effort:** Activation trigger (>0.6) for emergency premium air-freight.
* **LFP Pivot:** Percentage of production shifted to Lithium Iron Phosphate (LFP) chemistry to slash immediate critical mineral burn rate.

---

## 🤖 Baseline Strategies Evaluated

* **RL Agent (Proposed):** Anticipates disruptions, balances PLI subsidies, pivots chemistry, and stays within budget utilizing action projection.
* **Greedy Heuristic (JIT):** Only orders exact deficits using virgin materials. Minimizes baseline spend but highly vulnerable to starvation during tariff shocks.
* **Min-Max Inventory:** Automatically orders maximum capacity when inventory falls below 6,000 tons. Protects production but repeatedly triggers budget vetoes.
