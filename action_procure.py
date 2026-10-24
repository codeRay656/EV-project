import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class IndiaEVProcurementEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super(IndiaEVProcurementEnv, self).__init__()

        self.inventory = 1000.0
        self.current_disruption = 0
        self.virgin_price = 30000.0
        self.current_step = 0

        try:
            trade_df = pd.read_csv('minerals_trade.csv')
            recycling_df = pd.read_csv('recycling_flows.csv')
            self.base_virgin_capacity = float(trade_df['lithium_export_t'].mean() / 12)
            self.base_recycled_capacity = float(recycling_df['collected_tonnes'].mean() / 12)
        except Exception:
            self.base_virgin_capacity = 5000.0
            self.base_recycled_capacity = 800.0

        self.monthly_consumption = 3000.0
        self.max_inventory = 12000.0
        self.base_virgin_price = 30000.0
        self.base_recycled_price = 35000.0
        self.pli_subsidy_rate = 0.15

        self.disruption_probabilities = {
            'port_delay': 0.08,
            'supplier_default': 0.05,
            'tariff_shock': 0.03,
            'quality_issue': 0.07
        }

        baseline_cost = self.monthly_consumption * self.base_virgin_price
        self.historical_costs = [baseline_cost] * 50
        self.alpha_level = 0.95
        self.hard_budget_limit = 120_000_000.0
        self.rolling_var_95 = baseline_cost
        self.max_steps = 24

        # Track the *actual* available materials at any given moment
        self.current_virgin_available = self.base_virgin_capacity
        self.current_recycled_available = self.base_recycled_capacity

        # Action Space: [Virgin Ratio, Recycled Ratio, Expedite, LFP Pivot]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation Space: Updated to track current available capacities
        self.observation_space = spaces.Box(
            low=np.float32([0, 0, 0, 0, 0, 0]),
            high=np.float32([self.max_inventory, self.base_virgin_capacity * 2,
                             self.base_recycled_capacity * 2, 4, 100000, 500_000_000.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = 3000.0
        self.current_disruption = 0
        self.virgin_price = self.base_virgin_price

        self.current_virgin_available = self.base_virgin_capacity
        self.current_recycled_available = self.base_recycled_capacity

        baseline_cost = self.monthly_consumption * self.base_virgin_price
        self.historical_costs = [baseline_cost] * 50
        self.rolling_var_95 = baseline_cost

        return self._get_obs(), {}

    def _get_obs(self):
        # Allow the agent to "see" the exact current capacities so it isn't flying blind
        return np.array([
            float(self.inventory),
            float(self.current_virgin_available),
            float(self.current_recycled_available),
            float(self.current_disruption),
            float(self.virgin_price),
            float(self.hard_budget_limit)
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        self.current_virgin_available = self.base_virgin_capacity
        self.current_recycled_available = self.base_recycled_capacity

        # Organic market volatility using Gymnasium's isolated RNG (FIXED HERE)
        self.virgin_price = self.base_virgin_price * self.np_random.uniform(0.95, 1.05)

        # Disruption trigger using Gymnasium's isolated RNG (FIXED HERE)
        d = self.np_random.random()

        if d < self.disruption_probabilities['port_delay']:
            self.current_disruption = 1
            self.current_virgin_available = 0.0
        elif d < (self.disruption_probabilities['port_delay'] + self.disruption_probabilities['supplier_default']):
            self.current_disruption = 2
            self.current_virgin_available *= 0.2
        elif d < (self.disruption_probabilities['port_delay'] + self.disruption_probabilities['supplier_default'] +
                  self.disruption_probabilities['tariff_shock']):
            self.current_disruption = 3
            self.virgin_price *= 2.5
        elif d < (self.disruption_probabilities['port_delay'] + self.disruption_probabilities['supplier_default'] +
                  self.disruption_probabilities['tariff_shock'] + self.disruption_probabilities['quality_issue']):
            self.current_disruption = 4
        else:
            self.current_disruption = 0

        # Actions
        virgin_ordered = action[0] * self.monthly_consumption * 1.5
        recycled_ordered = action[1] * self.monthly_consumption * 1.0
        expedite_effort = action[2]
        lfp_pivot = action[3]

        consumption_reduction = lfp_pivot * 0.25
        actual_consumption = self.monthly_consumption * (1.0 - consumption_reduction)

        # Expedite logic
        if self.current_disruption == 1 and expedite_effort > 0.6:
            emergency_supply = self.base_virgin_capacity * 0.5
            virgin_received = min(virgin_ordered, emergency_supply)
            expedite_cost = 5_000_000
        else:
            virgin_received = min(virgin_ordered, self.current_virgin_available)
            expedite_cost = 0

        recycled_received = min(recycled_ordered, self.current_recycled_available)

        yield_loss = False
        if self.current_disruption == 4:
            recycled_received *= 0.7
            yield_loss = True
        elif recycled_received > self.base_recycled_capacity * 0.8:
            recycled_received *= 0.85
            yield_loss = True

        virgin_cost = virgin_received * self.virgin_price
        recycled_cost = recycled_received * self.base_recycled_price
        pli_rebate = recycled_cost * self.pli_subsidy_rate

        gross_cost = virgin_cost + recycled_cost + expedite_cost
        net_cost = gross_cost - pli_rebate
        intended_cost = net_cost

        # ACTION PROJECTION LAYER
        clamped = False
        if net_cost > self.hard_budget_limit:
            clamped = True
            reduction_needed = net_cost / self.hard_budget_limit
            virgin_received = virgin_received / reduction_needed
            recycled_received = recycled_received / reduction_needed

            virgin_cost = virgin_received * self.virgin_price
            recycled_cost = recycled_received * self.base_recycled_price
            pli_rebate = recycled_cost * self.pli_subsidy_rate
            net_cost = virgin_cost + recycled_cost - pli_rebate + expedite_cost
            if net_cost > self.hard_budget_limit:
                net_cost = self.hard_budget_limit

        total_received = virgin_received + recycled_received
        self.inventory = min(self.max_inventory, self.inventory + total_received)
        production = min(actual_consumption, self.inventory)
        self.inventory -= production
        unmet = max(0, actual_consumption - production)

        # CVaR Tracking
        self.historical_costs.append(net_cost)
        if len(self.historical_costs) > 2000:
            self.historical_costs.pop(0)
        self.rolling_var_95 = float(np.percentile(self.historical_costs, 95))

        # Reward Function
        production_reward = (production / self.monthly_consumption) * 100
        unmet_penalty = (unmet / self.monthly_consumption) * 300

        cvar_penalty = 0
        if net_cost > self.rolling_var_95:
            cvar_penalty = ((net_cost - self.rolling_var_95) / self.hard_budget_limit) * 50

        lfp_penalty = lfp_pivot * 15
        pli_bonus = pli_rebate / 1_000_000 * 10

        veto_penalty = 0
        if clamped:
            excess_ratio = (intended_cost - self.hard_budget_limit) / self.hard_budget_limit
            veto_penalty = (excess_ratio ** 2) * 500

        reward = (
                production_reward
                + pli_bonus
                - unmet_penalty
                - cvar_penalty
                - veto_penalty
                - lfp_penalty
        )

        info = {
            "actual_cost": net_cost,
            "unmet": unmet,
            "pli_rebate": pli_rebate,
            "intended_cost": intended_cost,
            "budget_clamped": clamped,
            "disruption_code": self.current_disruption,
            "virgin_rec": virgin_received,
            "recycled_rec": recycled_received,
            "expedited": expedite_effort > 0.6,
            "lfp_pivot": lfp_pivot,
            "inventory_level": self.inventory,
            "production": production,
            "yield_loss": yield_loss,
            "reward": reward,
            "total_received": total_received,
            "consumption": actual_consumption
        }

        return self._get_obs(), float(reward), False, self.current_step >= self.max_steps, info