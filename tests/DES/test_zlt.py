import multiechelonpls.des.discreteeventsystem as des
import scipy.stats as sp
import numpy as np

instance = {
    "T": 36,
    "n": 5,
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

# Pre-generate the demand with store following poisson(10) and online following poisson(2) for this test
np.random.seed(42)
demand = [
    [sp.poisson(2).rvs()] + [sp.poisson(10).rvs() for i in range(instance["n"])]
    for t in range(instance["T"])
]

sim_zlt = des.ZLT(
    instance["T"],
    [instance["shortage_cost"] for n in range(instance["n"] + 1)],
    [instance["holding_cost_warehouse"]]
    + [instance["holding_cost_store"] for n in range(instance["n"])],
    instance["dfw_cost"],
    0,  # Salvage
    [instance["initial_stock_warehouse"]] + instance["initial_stock_store"],
    instance["dfw_chance"],
    "OUT",  # Order up to policy (i.e. base-stock)
)
order_quantities = [[100, 11, 11, 11, 11, 11] for t in range(instance["T"])]
print(order_quantities[35][4])
sim_zlt.set_order_q(order_quantities)
sim_zlt.run(demand)
print(sim_zlt.period_cost)
sim_zlt.reset()
