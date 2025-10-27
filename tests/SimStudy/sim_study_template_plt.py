# Template to run a simulation study for inventory management models.
# This is the structure on how to find the best parameters. After this has been used, then you would use another simulation model where
# the results from eah instance/heuristic is run on the same simulation environment to compare performance.
# Positive Lead Time
import pickle
import itertools
import numpy as np
import scipy.stats as sp
import multiechelonpls.des.discreteeventsystem as des
# In practice you want to do this in parallel but for illustration we just show one process.
# import multiprocessing as mp


instance = {
    "T": 36,
    "n": 5,
    "store_lt": 1,
    "warehouse_lt": 1,
    "capacity": 10000,
    "dfw_chance": 0.8,
    "dfw_cost": 0,
    "holding_cost_store": 1,
    "holding_cost_warehouse": 1,
    "shortage_cost": 18,
    "initial_stock_warehouse": 20,
    "initial_stock_store": [
        10 for _ in range(5)
    ],  # Make sure length of number of stores
    "demand": [[2] + [10 for _ in range(5)] for _ in range(36)],
}

SIMULATION_RUNS = (
    10000  # how many simulation runs to average over for each parameter set
)

# For online allocation at the warehouse
online_allocation = int(
    sp.poisson(instance["demand"][0][0]).ppf(
        instance["shortage_cost"] / (instance["shortage_cost"] + 1)
    )
)

# Determine some sensible bounds to search over
# In practice you would want to run the echelon base-stock policy first which is only over 2 dimensions an use a sensible region around that to create the
# 3-dimensional search space when including the order cap.

# Store
ub_st = int(
    sp.poisson(instance["demand"][0][1] * (3 + instance["store_lt"])).ppf(
        instance["shortage_cost"] / (instance["shortage_cost"] + 1)
    )
)
lb_st = int(instance["demand"][0][1])
st_bounds = [i for i in range(lb_st, ub_st)]

# WH
# Just try with 20*(LTD of warehouse_2) - this is sensible for most instances but we can adjust if needed
wh_bounds = [
    i
    for i in range(
        20 + 5 * instance["n"] * (2 + instance["store_lt"] + instance["warehouse_lt"])
    )
]

# r boundaries (set minimum as 5, as realistically given the scenario we won't set a bound less than half demand)
# set arbitrary max of 20 for this example, since we will get rid of all ones where this is larger than store order up to level
r_range = [i for i in range(5, 20)]

# Only take options where the cap is less or equal than the store value
st_wh_r_actions = [
    combo
    for combo in itertools.product(st_bounds, wh_bounds, r_range)
    if combo[2] <= combo[0]
]

# Pregenerate demand for simulation
np.random.seed(42)
demand = np.array(
    [
        [
            [
                np.random.poisson(instance["demand"][t][i])
                for i in range(instance["n"] + 1)
            ]
            for t in range(instance["T"])
        ]
        for i in range(SIMULATION_RUNS)
    ]
)


costs_all_actions = {action: [] for action in st_wh_r_actions}
for action in st_wh_r_actions:
    print(action)

    print(
        "Store Base Stock: {}, Warehouse Base Stock: {}, r cap: {}".format(
            action[0], action[1], action[2]
        )
    )
    wh = action[0]
    st = action[1]
    r = action[2]

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
    sim_plt.set_order_q(
        [wh for s in range(instance["T"])],
        [[st for s in range(instance["T"])] for k in range(instance["n"])],
        [[r for s in range(instance["T"])] for k in range(instance["n"])],
    )
    for demand_run in demand:
        sim_plt.run(demand_run, [online_allocation for t in range(instance["T"])])
        costs_all_actions[action].append(np.sum(sim_plt.period_cost))
        sim_plt.reset()
