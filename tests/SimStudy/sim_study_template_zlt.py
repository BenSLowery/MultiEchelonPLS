# Template to run a simulation study for inventory management models.
# Zero Lead Time
#   tbd.
import scipy.stats as sp
import numpy as np

# AdBS method template
def adbs_evaluation():
    pass


# LC method template
def lc_evaluation():
    pass

# CBS method template
def cbs_evaluation():
    pass



instance = {
    "T": 36,
    "n": 10,
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
    "demand": [[2] + [10 for _ in range(10)] for _ in range(36)],
}

nv = int(sp.poisson(instance['demand'][0][1]).ppf(instance['cu'][1]/(instance['cu'][1]+1)))
bound= int((1-instance['c_dfw']/instance['cu'][1])*instance['p']*nv)
store_bounds = [i for i in range(max(nv-bound-5,0), nv+5)] # Use local control bounds and add a little leeway either side (just in case)

# Warehouse bound (lower bound in online demand)
od = instance['demand'][0][0]

# Upper bound is about 1/2 the expected store order being withheld in the warehouse 

nv_od = sp.poisson(od).ppf(instance['cu'][0]/(instance['cu'][0]+instance['co'][0]))

wh_bounds = [i for i in range(int(od),int((nv/2)*instance['n']+nv_od))]

best_bs, all_vals = constant_base_stock(instance['n'],instance['demand'],36, instance['cu'], instance['co'], [instance['c_dfw'] for d in range(instance['n'])], instance['p'], [0 for i in range(instance['n']+1)], 10000, store_bounds, wh_bounds, 92)
instance['central_control_order_stationary'] = best_bs[1][1]
pickle.dump(non_stationary_instances, open('/beegfs/client/default/loweryb/CentralControlHeuristic/Revisions/new_instances/non_stationary_vary_p.pickle', 'wb'))
