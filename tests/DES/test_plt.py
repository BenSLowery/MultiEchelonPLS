import multiechelonpls.des.discreteeventsystem as des
import scipy.stats as sp
import numpy as np

instance = {
    "T": 36,
    "n": 5,
    "store_lt": 2,
    "warehouse_lt": 2,
    "capacity": 10000,
    "dfw_chance": 0.8,
    "dfw_cost": 0,
    "holding_cost_store": 3,
    "holding_cost_warehouse": 1,
    "shortage_cost": 18,
    "initial_stock_warehouse": 20,
    "initial_stock_store": [
        5 for _ in range(5)
    ],  # Make sure length of number of stores
}

# Test positive lead-time case

# Pre-generate the demand with store following poisson(10) and online following poisson(2) for this test
np.random.seed(42)
demand = [
    [sp.poisson(2).rvs()] + [sp.poisson(10).rvs() for i in range(instance["n"])]
    for t in range(instance["T"])
]

sim_plt = des.PLT_speedy(
    instance["T"],
    instance["n"],
    instance["store_lt"],
    instance["warehouse_lt"],
    instance["capacity"],
    instance["capacity"],
    instance["capacity"],
    instance["dfw_chance"],
    instance["dfw_cost"],
    instance["holding_cost_store"],
    instance["holding_cost_warehouse"],
    instance["shortage_cost"],
    instance["shortage_cost"],
    instance["initial_stock_warehouse"],
    instance["initial_stock_store"],
)

# Set desired online demand for allocation (this is the newsvendor quantile for online demand)
online_demand_allocation = [
    sp.poisson(2).ppf(
        instance["holding_cost_warehouse"]
        / (instance["holding_cost_warehouse"] + instance["shortage_cost"])
    )
    for t in range(instance["T"])
]
sim_plt.set_order_q(
    [132 for s in range(instance["T"])],
    [[12 for s in range(instance["T"])] for k in range(instance["n"])],
    [[2 for s in range(instance["T"])] for k in range(instance["n"])],
)
sim_plt.run(demand, online_demand_allocation)
print(sim_plt.period_cost)
sim_plt.reset()
