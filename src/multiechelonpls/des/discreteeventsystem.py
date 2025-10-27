import numpy as np
import pandas as pd

"""
    Contains discrete event systems for:
    * ZLT (Zero Lead Time) model
    * PLT_speedy (Positive Lead Time) model without logging
"""


class PLT_speedy:
    def __init__(
        self,
        periods,
        N,
        store_lead_time,
        warehouse_lead_time,
        production_capacity,
        warehouse_capacity,
        store_capacity,
        p,
        cost_dfw,
        holding_warehouse,
        holding_store,
        shortage_warehouse,
        shortage_store,
        warehouse_initial,
        store_initial,
    ):
        self.T = periods
        self.N = N
        self.l_s = store_lead_time
        self.l_w = warehouse_lead_time
        self.cap_prod = production_capacity
        self.cap_w = warehouse_capacity
        self.cap_s = store_capacity
        self.p = p
        self.c_dfw = cost_dfw
        self.co_w = holding_warehouse
        self.co_s = holding_store
        self.cu_w = shortage_warehouse
        self.cu_s = shortage_store
        self.init_warehouse = warehouse_initial
        self.init_store = store_initial

        # Set up the inventory levels
        self.x_warehouse = np.zeros((self.T + 1, self.l_w + 1))
        self.x_store = np.zeros((self.N, self.T + 1, self.l_s + 1))

        # Populate the initial inventory evenely across the inventory pipeline (excluding the L position which is for orders)
        if self.l_w > 0:
            for lead_time in range(self.l_w):
                self.x_warehouse[0][lead_time] = int(
                    self.init_warehouse / (self.l_w + 1)
                )
        else:
            self.x_warehouse[0][0] = self.init_warehouse

        for store in range(self.N):
            if self.l_s > 0:
                for lead_time in range(self.l_s):
                    self.x_store[store][0][lead_time] = int(
                        self.init_store[store] / (self.l_s + 1)
                    )
            else:
                self.x_store[store][0][0] = self.init_store[store]

    def calc_store_orders(self, t):
        desired_orders = []  # Store "desired" orders for each store
        for store in range(self.N):
            # Applies the order quantity
            # Workout current inventory pipeline
            inv_pipeline_store = np.sum(self.x_store[store][t])
            # Base-stock policy
            desired_q = max(self.st_out[store][t] - inv_pipeline_store, 0)
            # Apply the storage capacity
            desired_q = min(desired_q, self.cap_s - inv_pipeline_store)

            # Apply order cap
            desired_q = min(desired_q, self.r[store][t])

            desired_orders.append(desired_q)

        # Check we have enough in thw warehouse to fulfil the order
        total_orders = np.sum(desired_orders)
        if total_orders <= self.x_warehouse[t][0]:
            return desired_orders
        else:
            allocated_orders = self.allocate_stock(
                desired_orders, self.x_warehouse[t][0], t
            )
            return allocated_orders

    def calc_warehouse_orders(self, t):
        # Calculate the echelon order up-to-level
        # i.e. sum inventory position across the entire network
        inv_pipeline_all = sum(self.x_warehouse[t]) + sum(
            [sum(self.x_store[store][t]) for store in range(self.N)]
        )

        warehouse_q = max(self.ech_out[t] - inv_pipeline_all, 0)

        # Apply the production capacity constraint
        warehouse_q = min(warehouse_q, self.cap_prod)

        # Apply the warehouse capacity
        warehouse_q = min(warehouse_q, self.cap_w - inv_pipeline_all)

        return warehouse_q

    def allocate_stock(self, desired_orders, available_stock, t):
        # Add at the start of the allocation the online desired orders
        desired_orders = [self.online_demand_alloc[t]] + desired_orders

        actual_allocation = [0 for i in range(len(desired_orders))]

        # Go through how much we have available and assign one by one
        for i in range(int(available_stock)):
            # Get index of largest shortfall
            idx = np.argmax(desired_orders)
            actual_allocation[idx] += 1
            desired_orders[idx] -= 1
        return actual_allocation[1:]

    def set_order_q(
        self, echelon_base_stock_level, base_stock_level_st, order_cap=None
    ):
        # In the echelon inventory we must have that ECH > sum of Store Base Stocks
        # The order cap is set less than the store base stock level.
        for t in range(self.T):
            if (
                np.sum([base_stock_level_st[store][t] for store in range(self.N)])
                > echelon_base_stock_level[t]
            ):
                raise ValueError(
                    "Sum of store base stocks must be less than echelon base stock level"
                )
        self.ech_out = echelon_base_stock_level
        self.st_out = base_stock_level_st
        if order_cap:
            self.r = order_cap
        else:
            # If no order cap is givng, not having an order cap is equivalent to setting the order cap at the base-stock since that's
            # the largest it can be
            self.r = base_stock_level_st

    def reset(self):
        """Reset the inventory simulation to its initial state."""
        self.x_warehouse = np.zeros((self.T + 1, self.l_w + 1))
        self.x_store = np.zeros((self.N, self.T + 1, self.l_s + 1))
        if self.l_w > 0:
            for lead_time in range(self.l_w):
                self.x_warehouse[0][lead_time] = int(
                    self.init_warehouse / (self.l_w + 1)
                )
        else:
            self.x_warehouse[0][0] = self.init_warehouse

        for store in range(self.N):
            if self.l_s > 0:
                for lead_time in range(self.l_s):
                    self.x_store[store][0][lead_time] = int(
                        self.init_store[store] / (self.l_s + 1)
                    )
            else:
                self.x_store[store][0][0] = self.init_store[store]

        self.period_cost = np.zeros(self.T + 1)

    def run(self, demand, online_demand_allocation):
        self.demand = demand
        self.online_demand_alloc = online_demand_allocation

        # Save period costs
        self.period_cost = np.zeros([self.T + 1, self.N + 1])

        for t in range(self.T):
            # Append starting inventory to log

            # Step 1. Get orders into the system
            # We do the warehouse first (as this makes zero lead time at the warehouse easier to calculate)

            # Update the warehouse pipeline vector
            Q_wh = self.calc_warehouse_orders(t)
            self.x_warehouse[t][self.l_w] += Q_wh

            # For this, we assume a base-stock policy for stores.
            Q_stores = self.calc_store_orders(t)

            # Update the store pipeline vectors
            for store in range(self.N):
                self.x_store[store][t][self.l_s] += Q_stores[store]

            self.x_warehouse[t][0] -= np.sum(Q_stores)

            # Experince demand and DFW fulfilment
            dfw_fulfillment = []
            # Warehouse
            self.x_warehouse[t][0] -= self.demand[t][0]

            # Stores
            for store in range(self.N):
                self.x_store[store][t][0] -= self.demand[t][store + 1]

                # Is DFW available?
                if self.x_store[store][t][0] < 0:
                    dfw_request = np.random.binomial(
                        np.abs(self.x_store[store][t][0]), self.p
                    )

                    # Here we can only give DFW from what's available in the warehouse
                    dfw_request = min(dfw_request, max(self.x_warehouse[t][0], 0))
                    dfw_fulfillment.append(dfw_request)

                    # Take DFW from warehouse and lesser the stockout at store
                    self.x_warehouse[t][0] -= dfw_request
                    self.x_store[store][t][0] += dfw_request
                else:
                    dfw_fulfillment.append(0)

            # Calculate the period costs

            # Warehouse
            self.period_cost[t][0] = (
                np.abs(self.x_warehouse[t][0]) * self.cu_w
                if self.x_warehouse[t][0] <= 0
                else self.x_warehouse[t][0] * self.co_w
            )

            # Store
            for store in range(self.N):
                self.period_cost[t][store + 1] = (
                    np.abs(self.x_store[store][t][0]) * self.cu_s
                    if self.x_store[store][t][0] <= 0
                    else self.x_store[store][t][0] * self.co_s
                )
                # Add dfw
                self.period_cost[t][store + 1] += self.c_dfw * dfw_fulfillment[store]

            # Carry inventory to next period
            # Warehouse
            if self.l_w > 0:
                self.x_warehouse[t + 1][0] = (
                    self.x_warehouse[t][0] + self.x_warehouse[t][1]
                )
                for pos in range(1, self.l_w):
                    self.x_warehouse[t + 1][pos] = self.x_warehouse[t][pos + 1]
                self.x_warehouse[t + 1][self.l_w] = 0
            else:
                self.x_warehouse[t + 1][0] = self.x_warehouse[t][0]

            # Store
            if self.l_s > 0:
                for store in range(self.N):
                    self.x_store[store][t + 1][0] = (
                        max(self.x_store[store][t][0], 0) + self.x_store[store][t][1]
                    )
                    for pos in range(1, self.l_s):
                        self.x_store[store][t + 1][pos] = self.x_store[store][t][
                            pos + 1
                        ]
                    self.x_store[store][t + 1][self.l_s] = 0
            else:
                for store in range(self.N):
                    self.x_store[store][t + 1][0] = max(self.x_store[store][t][0], 0)


class ZLT:
    def __init__(
        self,
        periods,
        underage,
        overage,
        dfw_cost,
        salvage,
        initial_inventory,
        prob_dfw,
        order_rule,
        log=False,
    ):
        self.T = periods
        self.cu = underage
        self.co = overage
        self.c = salvage
        self.c_dfw = dfw_cost
        self.p = prob_dfw
        self.order_rule = order_rule
        self.init_inv = initial_inventory
        self.N = len(initial_inventory) - 1

        self.log_data = log
        if self.log_data:
            # Create empty dictionary to store information which will eventually become a dataframe
            self.log = {}

        # Set up inventory levels
        self.x = np.zeros((self.T + 1, self.N + 1))
        self.x[0] = initial_inventory

        if self.log_data:
            # Create empty dictionary to store information which will eventually become a dataframe
            self.log = {
                "StartingInv": [],
                "Order": [],
                "Allocation_Req": [],
                "PostOrder": [],
                "Demand": [],
                "DFW_Fulfillment": [],
                "DFW_total": [],
                "EndingInventory": [],
                "PeriodCost": [],
            }

    def set_order_q(self, order):
        if self.order_rule == "FQ":  # If a fixed order quantity
            self.orders = lambda t: [order[t][i] for i in range(self.N + 1)]
        elif (
            self.order_rule == "OUT"
        ):  # If an order-up-to level, we have a special case for the warehouse where we need to subtract the stroe inventory less than the order level from the warehouse OuT
            self.orders = lambda t: [
                order[t][0]
                - sum([min(order[t][i], self.x[t][i]) for i in range(1, self.N + 1)])
                - self.x[t][0]
            ] + [max(order[t][i] - self.x[t][i], 0) for i in range(1, self.N + 1)]
        elif (
            self.order_rule == "OPT"
        ):  # If an optimal policy, we have a different order rule based on the current inventory
            # Rename columns to lower case incase we entered them incorrectly with camel case
            order.rename(str.lower, axis="columns")
            self.orders = lambda t: [
                max(
                    int(
                        order[
                            (order["period"] == t + 1)
                            & (order["inventory"] == tuple(map(int, self.x[t])))
                        ]["order-up-to"].values[0][i]
                    )
                    - self.x[t][i],
                    0,
                )
                for i in range(self.N + 1)
            ]  # very very silly method, should improve

    def reset(self):
        """Reset the inventory simulation to its initial state."""
        self.x = np.zeros((self.T + 1, self.N + 1))
        self.x[0] = self.init_inv
        self.period_cost = np.zeros(self.T + 1)
        if self.log_data:
            self.log = {
                "StartingInv": [],
                "Order": [],
                "Allocation_Req": [],
                "PostOrder": [],
                "Demand": [],
                "DFW_Fulfillment": [],
                "DFW_total": [],
                "EndingInventory": [],
                "PeriodCost": [],
            }

    def allocate_stock(self, t, Q, allocation_policy="max_req"):
        """
        Allow more allocation policies
        """
        x = self.x[t].copy()

        # Check if we need to allocate in the first place
        x_wh = x[0] + Q[0] - np.sum(Q[1:])

        # This case we don't need to allocate stock
        if x_wh >= 0:
            x[0] += Q[0] - np.sum(Q[1:])
            x[1:] += Q[1:]
            if self.log_data:
                self.log["Allocation_Req"].append(False)
            return x

        # If not then we enact an allocation policy
        if allocation_policy == "max_req":
            if self.log_data:
                self.log["Allocation_Req"].append(True)
            # Get available stock
            avail = x[0] + Q[0]
            store_q = Q[1:].copy()  # keep track of how much we can allocate

            # Allocate one by one to each store with highest need
            while avail > 0:
                store_q[np.argmax(store_q)] -= 1
                avail -= 1

            # Allocate deliveries and shipments
            x[0] = 0  # Warehouse will obviously have no stock now.
            x[1:] += np.subtract(Q[1:], store_q)  # Store stock

        return x

    def run(self, demand):
        # demand
        self.demand = demand

        # period costs
        self.period_cost = np.zeros([self.T + 1, self.N + 1])

        for t in range(self.T):
            if self.log_data:
                si = self.x[t].copy()
                self.log["StartingInv"].append(si)

            # Step 1. Find stocking decision for the period
            Q = self.orders(t)

            # Step 2. Echelons recieve stock and warehouse loses stock

            # Set an allocation procedure to make sure max amount of stock is sent.
            self.x[t] = self.allocate_stock(t, Q)

            if self.log_data:
                po = self.x[t].copy()
                self.log["PostOrder"].append(po)

            # Step 3. Demand in each channel realised
            self.x[t] -= self.demand[t]

            # Step 4. Calculate DFW fulfilment
            dfw_fulfillment = np.zeros(self.N)

            for idx, s in enumerate(self.x[t][1:]):
                if s < 0:
                    dfw_fulfillment[idx] = np.random.binomial(np.abs(s), self.p)
                    self.x[t][idx + 1] += dfw_fulfillment[idx]
                else:
                    dfw_fulfillment[idx] = 0

            # Take off DFW fulfilment from warehouse
            self.x[t][0] -= np.sum(dfw_fulfillment)

            # Step 5. Calculate period costs

            # Warehouse cost
            self.period_cost[t][0] = (
                np.abs(self.x[t][0] * self.cu[0])
                if self.x[t][0] <= 0
                else self.x[t][0] * self.co[0]
            )

            # Store cost
            self.period_cost[t][1:] = self.c_dfw * dfw_fulfillment  # DFW cost
            for s in range(1, self.N + 1):
                self.period_cost[t][s] += (
                    np.abs(self.x[t][s] * self.cu[s])
                    if self.x[t][s] <= 0
                    else self.x[t][s] * self.co[s]
                )

            # Step 6. Carry Inventory to next period
            self.x[t + 1][0] = self.x[t][0]
            self.x[t + 1][1:] = np.maximum(self.x[t][1:], 0)

            if self.log_data:
                self.log["Order"].append(Q)
                self.log["Demand"].append(self.demand[t])
                self.log["DFW_Fulfillment"].append(dfw_fulfillment)
                self.log["DFW_total"].append(np.sum(dfw_fulfillment))
                self.log["EndingInventory"].append(
                    self.x[t]
                )  # i.e. record inventory after pipeline movements
                self.log["PeriodCost"].append(np.sum(self.period_cost[t]))

        # Salvage remaining inventory:
        self.period_cost[self.T] = -self.c * self.x[self.T]
        if self.log_data:
            self.log["StartingInv"].append(self.x[self.T])
            self.log["Order"].append(0)
            self.log["PostOrder"].append(0)
            self.log["Demand"].append(0)
            self.log["Allocation_Req"].append(0)
            self.log["DFW_Fulfillment"].append(0)
            self.log["DFW_total"].append(0)
            self.log["EndingInventory"].append(0)
            self.log["PeriodCost"].append(np.sum(self.period_cost[self.T]))

        # Export log to dataframe
        if self.log_data:
            self.log = pd.DataFrame(self.log)
            self.log.index += 1
            self.log.index.name = "Period"
