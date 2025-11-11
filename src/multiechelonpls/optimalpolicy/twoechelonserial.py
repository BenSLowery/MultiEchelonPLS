# Two echelon serial system for Local Control Method
# Uses backwards dynamic programming to calculate the optimal local control, then aggregates. You could use a simulation approach too, just this is an exact method.
import numpy as np
import scipy.stats as sp
import itertools
import multiprocessing as mp
import functools


# Allows for parrellisation to speed up
class ZeroLTNVSearch:
    """
    Search for the best policy for the NV heuristic.
    """

    def __init__(
        self,
        periods,
        underage_cost,
        overage_cost,
        dfw_cost,
        dfw_rate,
        discount_factor,
        demand_params,
        demand_trunc,
        cores=3,
    ):
        self.verbose = True  # If showing output
        self.T = periods
        self.cu = underage_cost
        self.co = overage_cost
        self.cdfw = dfw_cost
        self.p = dfw_rate
        self.gamma = discount_factor
        self.N = 1
        self.params = demand_params[:, 1]
        self.online_params = demand_params[:, 0]
        self.max_d = int(demand_trunc[0])
        self.num_cores = cores

        # Generate demands
        self.demand_val, self.demand_pmf = self.gen_demand(self.params)
        self.online_demand_val, self.online_demand_pmf = self.gen_demand(
            self.online_params
        )

        # Maximum order up to levels we need to look at. Sensible cap is just the maximum demand we look at for stores, and for warehouse, it'll be the sum of all these
        self.max_y_store = self.max_d
        self.max_y_wh = self.max_d * (self.N + 1) + int(demand_trunc[1])  # Add online

        # State space
        st_ss = range(self.max_y_store)  # Store
        wh_ss = range(self.max_y_wh)  # Warehouse
        self.state_space = [
            x for x in itertools.product(*([wh_ss] + [st_ss for i in range(self.N)]))
        ]

        # Save previous value functions and optimal policies
        self.optimal_pol = []
        self.V = []  # Previous states

    def gen_demand(self, params):
        # Pre-allocate array to generate demand
        vals = np.array(
            [i for i in range(self.max_d + 1)]
        )  # Vals are just integer demand values
        pmf = np.empty(
            self.T, object
        )  # pmf has a list of pmf values for the warehouse and N stores. Rows are period and columns are demand

        # Allocate the pmf
        for t in range(self.T):
            pmf[t] = [sp.poisson(params[t]).pmf(d) for d in range(self.max_d + 1)]

        return vals, pmf

    # Cache results (set it arbitrarily big)
    @functools.lru_cache(maxsize=2**14)
    def calc_immediate_cost(self, t, y, q):
        Exp = 0
        for d_online, d_online_pmf in zip(
            self.online_demand_val, self.online_demand_pmf[t - 1]
        ):
            for d, d_pmf in zip(
                self.demand_val, self.demand_pmf[t - 1]
            ):  # Get t-1 index as starting from 1 not 0
                # binomial parameter for the store
                beta_n = max(d - y[1], 0)
                beta_val = [b for b in range(beta_n + 1)]
                beta_pmf = [
                    sp.binom._pmf([b for b in range(beta_n + 1)], beta_n, self.p)
                ]
                # Calculate values inside the expectation
                for b, b_pmf in zip(beta_val, beta_pmf[0]):
                    # Warehouse Cost
                    inner_cost_wh = self.cu[0] * max(
                        q + b + d_online - y[0], 0
                    ) + self.co[0] * max(y[0] - b - q - d_online, 0)

                    # Store Cost
                    inner_cost_st = (
                        self.cu[1] * (max(d - y[1], 0) - b)
                        + self.co[1] * max(y[1] - d, 0)
                        + self.cdfw * b
                    )
                    Exp += (
                        d_online_pmf * b_pmf * d_pmf * (inner_cost_st + inner_cost_wh)
                    )
        return Exp

    @functools.lru_cache(maxsize=2**14)
    def calc_future_cost(self, t, y, q):
        # Keep track of cost
        Exp = 0
        for d_online, d_online_pmf in zip(
            self.online_demand_val, self.online_demand_pmf[t - 1]
        ):
            for d, d_pmf in zip(
                self.demand_val, self.demand_pmf[t - 1]
            ):  # Get t-1 index as starting from 1 not 0
                # binomial parameter for the store
                beta_n = max(d - y[1], 0)
                beta_val = [b for b in range(beta_n + 1)]
                beta_pmf = [
                    sp.binom.pmf([b for b in range(beta_n + 1)], beta_n, self.p)
                ]
                # Calculate values inside the expectation
                for b, b_pmf in zip(beta_val, beta_pmf[0]):
                    # Warehouse next period
                    x_t_plus_1_wh = max(y[0] - q - b - d_online, 0)

                    x_t_plus_1_st = max(y[1] - d, 0)

                    Exp += (
                        d_online_pmf
                        * b_pmf
                        * d_pmf
                        * self.V_t_plus_1[tuple([x_t_plus_1_wh, x_t_plus_1_st])]
                    )

        return Exp

    def terminal_state(self, salvage_cost=0):
        """
        Appends terminal condition, inventory can be salvaged for purchase price c.
        Negative values cost c from ordering commitment
        """
        for x in self.state_space:
            self.V.append((tuple(x), -sum(x) * salvage_cost))

    def parallel_eval(self, y, period):
        # Assume we start in state (0,0) - as it doesn't matter where we start (as base-stock optimal)
        q_store = max(y[1], 0)  # Order quantities needed for cost function
        im_cost = self.calc_immediate_cost(period, tuple(y), q_store)
        fut_cost = self.gamma * self.calc_future_cost(period, tuple(y), q_store)
        cost = im_cost + fut_cost
        return y, cost

    def update_states(self, x, opt, period):
        # Calculate the order up to level for the state
        y_st = max(opt[1], x[1])  # Stores are easy

        if max(opt[0], x[0]) == x[0]:
            y_wh = x[0]
        else:
            # Calculate y based on how much we need to send to the stores
            y_wh = opt[0] - x[1]

        y = [y_wh, y_st]
        q_store = max(y[1] - x[1], 0)  # Order quantities needed for cost function

        im_cost = self.calc_immediate_cost(period, tuple(y), q_store)
        fut_cost = self.gamma * self.calc_future_cost(period, tuple(y), q_store)
        cost = im_cost + fut_cost
        return x, y, cost

    def FindPolicy(self, st_nv_order, wh_nv_order):
        """
        NV order gives the newsvendor order for each period
        """
        self.opt_action = {}

        # Step 1. Iterate backwards recursively through all periods
        for period in range(self.T, 0, -1):
            # Keep value functions to use
            self.V_t_plus_1 = {val_func[0]: val_func[1] for val_func in self.V}

            self.V = []

            self.total_cost = {}

            # Generate actions
            ub = st_nv_order[period - 1]
            lb = ub - int((1 - self.cdfw / self.cu[1]) * self.p * ub)

            actions = [
                i
                for i in itertools.product(
                    *[
                        [i for i in range(ub + 1, ub + 1 + wh_nv_order[period - 1])],
                        [i for i in range(lb, ub + 1)],
                    ]
                )
            ]
            # print(actions)
            cl = mp.Pool(self.num_cores)
            total_cost_par = cl.starmap(
                self.parallel_eval, list(zip(actions, itertools.repeat(period)))
            )
            cl.close()

            for t_c in total_cost_par:
                self.total_cost[t_c[0]] = t_c[1]

            # Get the order up to level for this period
            # Assume base stock so can start in arbitrary state
            opt = min(self.total_cost, key=self.total_cost.get)

            self.opt_action[period] = opt

            # Update states
            cl = mp.Pool(self.num_cores)
            updated_states = cl.starmap(
                self.update_states,
                zip(self.state_space, itertools.repeat(opt), itertools.repeat(period)),
            )
            cl.close()

            for x, y, cost in updated_states:
                self.V.append((x, cost))
                self.optimal_pol.append((period, x, y))
