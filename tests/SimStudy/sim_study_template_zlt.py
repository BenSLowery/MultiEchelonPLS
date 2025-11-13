# Template to run a simulation study for inventory management models.
# Zero Lead Time
#   tbd.
import scipy.stats as sp
import numpy as np
import itertools
import multiechelonpls.des.discreteeventsystem as des
import multiechelonpls.optimalpolicy.twoechelonserial as tes


# AdBS method template
# Change can be in demand or overage cost
def adbs_evaluation(wh_range, store_range, params, change='overage'):

    scores = {}
    # Iterate over all possible combinations
    combinations = list(
        itertools.product([i for i in wh_range], [j for j in store_range])
    )
    
    # Pre-generate demands for all periods - usa same CRN for each parameter selection
    simulation_len = 500
    demands = [[
            [sp.poisson(params["demand_params"][t][0]).rvs()]
            + [
                sp.poisson(params["demand_params"][t][n + 1]).rvs()
                for n in range(params["stores"])
            ]
            for t in range(params["periods"])
    ] for sim in range(simulation_len)]

    # Calculate the most common parameter 
    if change == 'demand':
        store_demands_collapsed = [d for d_t in params["demand_params"] for d in d_t[1:]]
        mode_store = max(store_demands_collapsed, key=store_demands_collapsed.count)
    elif change == 'overage':
        mode_overage = max(params['co'][1:], key=params['co'][1:].count)
    else:
        raise ValueError("Change must be either 'overage' or 'demand'")
    
    for (wh, st) in combinations:
        combo_costs = []
        sim_zlt = des.ZLT(
            params["periods"],
            params['cu'],
            params['co'],
            0, # DFW cost
            0,  # Salvage
            [params['demand_params'][0][location] for location in range(params["stores"] + 1)],
            params["p"],
            "OUT",  # Order up to policy (i.e. base-stock)
        )
        # For each store length
        if change == 'overage':
            st_out_one_time_period = []
            # Adjust the different parameter values
            for c_o in params['co'][1:]:
                if c_o == mode_overage:
                    st_out_one_time_period.append(st)
                else:
                    CF_mode = params['cu'][1]/(params['cu'][1]+mode_overage)
                    CF =  params['cu'][1]/(params['cu'][1]+c_o)
                    if c_o <= mode_overage:
                    
                        st_out_one_time_period.append(np.floor(st-st*(CF_mode-CF)))
                        
                    elif c_o >= mode_overage:
                        st_out_one_time_period.append(np.ceil(st+st*(CF-CF_mode)))
            st_out = [st_out_one_time_period for t in range(T)]

            wh_online_demand = [sp.poisson(params['demand_params'][t][0]).ppf(params['cu'][0]/(params['cu'][0]+params['cu'][0])) for t in range(T)]

            order_up_to = [[wh+wh_online_demand[t] + np.sum(st_out[t])] + st_out[t] for t in range(T)]
                
            sim_zlt.set_order_q(order_up_to)

        elif change == 'demand':

            safety_stock_perc = ((st-mode_store)/mode_store)

            st_out = []
            for t in range(params['periods']):
                store_out_t = []
                for store in range(params['stores']):
                    if params['demand_params'][t][store+1] == mode_store:
                        store_out_t.append(st)
                    else:
                        store_out_t.append(np.round(params['demand_params'][t][store+1]+params['demand_params'][t][store+1]*safety_stock_perc))
                st_out.append(store_out_t)

            wh_online_demand = [sp.poisson(params['demand_params'][t][0]).ppf(params['cu'][0]/(params['cu'][0]+params['co'][0])) for t in range(params['periods'])]

            order_up_to = [[wh+wh_online_demand[t] + np.sum(st_out[t])] + st_out[t] for t in range(params['periods'])]
                
            # Run instance
            sim_zlt.set_order_q(order_up_to)

        # Run simulations
        for demand in demands:
            sim_zlt.run(demand)
            combo_costs.append(sim_zlt.period_cost.sum())
            sim_zlt.reset()
        scores[(wh, st)] = (order_up_to, np.mean(combo_costs))
    # Return best
    scores_cost = {k: v[1] for k, v in scores.items()}
    return scores, min(scores_cost, key=scores_cost.get)


# LC method template
def lc_evaluation(params):
    """
    Search for the best policy for the NV heuristic.
    """
    n = params["stores"]
    cu = params["cu"]
    co = params["co"]
    p = params["p"]
    cdfw = [0 for i in range(n)]  # We never really want to charge for DFW
    T = params["periods"]
    demand_params = np.array(params["demand_params"])

    lc_orders = np.zeros([T, n + 1])
    # Go through each stores configurations. The warehouse will be the same for all, the store will be different,
    for store in range(1, n + 1):
        # Critical Fractile
        cf = cu[store] / (cu[store] + co[store])
        ub = []
        online_ub = []
        # Generate bounds for each time period
        for t in range(len(demand_params)):
            d_t = demand_params[t][store]  # Get mean demand parameter

            # Calculate the bound (the lower bound is calcualted within the NV search class)
            ub.append(int(sp.poisson(d_t).ppf(cf)))
            online_ub.append(15)

        ### NV SEARCH ###
        lc_sdp = tes.ZeroLTNVSearch(
            T,
            [cu[0], cu[store]],
            [co[0], co[store]],
            cdfw[store - 1],
            p,
            0.995,
            demand_params[:, [0, store]],
            (max(demand_params[:, store]) * 3, max(online_ub)),
        )
        lc_sdp.terminal_state()
        lc_sdp.FindPolicy(ub, online_ub)
        ### END ###

        for time_index, order in lc_sdp.opt_action.items():
            lc_orders[time_index - 1][store] = order[1]
            lc_orders[time_index - 1][0] += order[0]
    return lc_orders


# CBS method template
def cbs_evaluation(wh_range, store_range, params):

    scores = {}
    # Iterate over all possible combinations
    combinations = list(
        itertools.product([i for i in wh_range], [j for j in store_range])
    )
    
    # Pre-generate demands for all periods - usa same CRN for each parameter selection
    simulation_len = 500
    demands = [[
            [sp.poisson(params["demand_params"][t][0]).rvs()]
            + [
                sp.poisson(params["demand_params"][t][n + 1]).rvs()
                for n in range(params["stores"])
            ]
            for t in range(params["periods"])
        ] for sim in range(simulation_len)]
    
    # Run through combinations on the DES
    # Pre-generate the demand with store following poisson(10) and online following poisson(2) for this test
    for (wh, st) in combinations:
        combo_costs = []
        sim_zlt = des.ZLT(
            params["periods"],
            params['cu'],
            params['co'],
            0, # DFW cost
            0,  # Salvage
            [params['demand_params'][0][location] for location in range(params["stores"] + 1)],
            params["p"],
            "OUT",  # Order up to policy (i.e. base-stock)
        )
        order_quantities = [[wh] + [st for n in range(params['stores'])] for t in range(params["periods"])]
        

        for demand in demands:
            sim_zlt.set_order_q(order_quantities)
            sim_zlt.run(demand)
            combo_costs.append(sim_zlt.period_cost.sum())
            sim_zlt.reset()
        scores[(wh, st)] = np.mean(combo_costs) # random numver for now just illustration
    # Return best
    return scores, min(scores, key=scores.get)


if __name__ == "__main__":
    # This is just an illustrating example - you can replace with the parameters within the simulation study paper to verify they align up.
    np.random.seed(42)
    params = {
        "periods": 5,
        "stores": 10,
        "p": 0.8,
        "cu": [18 for i in range(11)],
        "co": [1] + [2 for i in range(10)],
        "demand_params": [
            [10] + [10-T*2 for n in range(10)] for T in range(5)
        ],  # Format demand_params[time period][0 if online else store index]
    }

    # Run each method on the des
    print('Calculating LC evaluation...')
    #print(lc_evaluation(params))

    print('Calculating AdBS evaluation...')
    print(adbs_evaluation(range(2,40, 5), range(4, 16, 2), params, change='demand')[1])
    

    print('Calculating CBS evaluation...')
    print(cbs_evaluation(range(120, 180, 5), range(4, 16, 2), params))
    
    
