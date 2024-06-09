import numpy as np

class GeneralFlippingDecoder():
    def base_cost(self, marked_checks: np.ndarray[bool]):
        return 1

    def cost(self, marked_checks: np.ndarray[bool], max_cost, max_function_calls=np.inf, estimate=None):
        function_calls = 1

        cost_gauge = self.base_cost(marked_checks)
        if estimate is not None:
            cost_gauge = estimate
        if cost_gauge == 0 or cost_gauge > max_cost:
            return cost_gauge, [], function_calls
        
        if cost_gauge == 1:
            faults_to_check = np.nonzero(self.fault_graph[marked_checks].all(axis=0) & np.logical_not(self.fault_graph[~marked_checks].any(axis=0)))[0]
            if len(faults_to_check)  >= 1:
                return cost_gauge, [faults_to_check[0]], function_calls
            
            cost_gauge += 1

        while cost_gauge <= max_cost:
            # calculate faults to check by how they would reduce
            faults_to_check = np.nonzero(self.fault_graph[marked_checks].any(axis=0))[0] # any touching fault
            # sort them based on something
            sorter = self.fault_graph[marked_checks].sum(axis=0) - np.logical_not(self.fault_graph[marked_checks].sum(axis=0)).astype(np.int8)
            sorter = sorter[faults_to_check]

            faults_to_check = faults_to_check[np.argsort(sorter)[::-1]]

            for fault in faults_to_check:
                marked_checks_prime = marked_checks ^ (self.fault_graph[:, fault] == 1)

                # need to check here
                if max_function_calls - function_calls <= 0:
                    return cost_gauge, [], function_calls
                
                if cost_gauge == estimate:
                    cost, path, new_function_calls = self.cost(marked_checks_prime, max_cost = cost_gauge-1, max_function_calls=max_function_calls - function_calls, estimate=estimate-1)
                else:
                    cost, path, new_function_calls = self.cost(marked_checks_prime, max_cost = cost_gauge-1, max_function_calls=max_function_calls - function_calls)
                function_calls += new_function_calls

                if cost == cost_gauge-1:
                    return cost_gauge, [fault] + path, function_calls
            
            # if at this point we haven't returned that means the actual cost is higher
            # so we increase our estimate of the cost (by 2, as parity is fixed)
            cost_gauge += 1
            
        return cost_gauge, [], function_calls

    def __init__(self, fault_graph: np.ndarray):
        self.fault_graph = fault_graph

class FlippingDecoder():
    def idx2color(self, idx):
        return (idx//12 + idx) % 3

    def syndrome2color_count(self, syndrome):
        color, count = np.unique(self.idx2color(np.where(syndrome)[0]), return_counts=True)
        count[np.argsort(color)]
        return count[np.argsort(color)]
    
    def base_cost(self, marked_checks: np.ndarray[bool]):
        num_checks = marked_checks.sum()
        if num_checks == 2:
            return 6 # a weight 2 syndrome can only be caused by min a weight 6 error
        return np.max(self.syndrome2color_count(marked_checks))
        # elif num_checks % 2 == 0:
        #     return ((num_checks+4)//6)*2
        # # else
        # return ((num_checks+2)//6)*2 + 1

    def cost_optimized_further(self, marked_checks: np.ndarray[bool], max_cost, max_function_calls=np.inf):
        """
        Returns cost estimate, the faults to flip and the number of function calls
        """
        function_calls = 1

        graph = self.fault_graph
        cost_gauge = self.base_cost(marked_checks)

        if cost_gauge == 0 or cost_gauge > max_cost:
            return cost_gauge, [], function_calls
        
        #fault_pattern = graph[marked_checks].sum(axis=0)
        #con3_faults = np.nonzero(fault_pattern == 3)[0]
        #con2_faults = np.nonzero(fault_pattern == 2)[0]
        #con1_faults = np.nonzero(fault_pattern == 1)[0]
        ##con32_faults = np.nonzero(fault_pattern >= 2)[0]
        ##con_any_faults = np.nonzero(fault_pattern)[0]
        #con32_faults = np.hstack([con3_faults, con2_faults]) # smarter order
        #con_any_faults = np.hstack([con32_faults, con1_faults]) # smarter order

        #del fault_pattern

        #S = marked_checks.sum()

        if cost_gauge == 1:
            faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0] # WHAAAT
            #if len(con3_faults) == 1:
            if len(faults_to_check):
                return cost_gauge, [faults_to_check[0]], function_calls
                #return cost_gauge, [con3_faults[0]], function_calls
            cost_gauge = 3

        # if cost_gauge > max_cost:
        #     return cost_gauge, []
        
        # if depth == 0: # after this we for sure have to go a recursion deeper, so check if we are allowed
        #     # ACTUALLY NOT NECESSARILY TRUE BUT WHATEVER
        #     return cost_gauge, [], function_calls
        
        while cost_gauge <= max_cost:
            if marked_checks.sum() <= cost_gauge:
                if cost_gauge == 4: # small optimization, only look around 1 particular check
                    faults_to_check = np.nonzero(graph[marked_checks][0])[0]
                    
                else:
                    #faults_to_check = con_any_faults
                    faults_to_check = np.nonzero(graph[marked_checks].any(axis=0))[0] # WHAAAT
                    #faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 1)[0]]) # WHAAAT


            elif marked_checks.sum() <= 2*cost_gauge:
                #faults_to_check = con32_faults
                faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0]]) # WHAAAT
                #faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 2)[0]
                # TODO: feasability filter
            else: #marked_checks.sum() <= 3*cost_gauge
                #faults_to_check = con3_faults
                faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
                if len(faults_to_check) < marked_checks.sum() - 2*cost_gauge: # feasability filter
                    faults_to_check = []

            # now that we now what faults to check, we iterate through them
            for fault in faults_to_check:
                marked_checks_prime = marked_checks ^ (graph[:, fault] == 1)

                # need to check here
                if max_function_calls - function_calls <= 0:
                    return cost_gauge, [], function_calls
                
                cost, path, new_function_calls = self.cost_optimized_further(marked_checks_prime, max_cost = cost_gauge-1, max_function_calls=max_function_calls - function_calls)
                function_calls += new_function_calls

                if cost == cost_gauge-1:
                    return cost_gauge, [fault] + path, function_calls
            
            # if at this point we haven't returned that means the actual cost is higher
            # so we increase our estimate of the cost (by 2, as parity is fixed)
            cost_gauge += 2
            
        return cost_gauge, [], function_calls

    def cost_optimized(self, marked_checks: np.ndarray[bool], max_cost, max_function_calls=np.inf):
        """
        Returns cost estimate, the faults to flip and the number of function calls
        """
        function_calls = 1

        graph = self.fault_graph
        cost_gauge = self.base_cost(marked_checks)

        if cost_gauge == 0 or cost_gauge > max_cost:
            return cost_gauge, [], function_calls
        
        #fault_pattern = graph[marked_checks].sum(axis=0)
        #con3_faults = np.nonzero(fault_pattern == 3)[0]
        #con2_faults = np.nonzero(fault_pattern == 2)[0]
        #con1_faults = np.nonzero(fault_pattern == 1)[0]
        ##con32_faults = np.nonzero(fault_pattern >= 2)[0]
        ##con_any_faults = np.nonzero(fault_pattern)[0]
        #con32_faults = np.hstack([con3_faults, con2_faults]) # smarter order
        #con_any_faults = np.hstack([con32_faults, con1_faults]) # smarter order

        #del fault_pattern

        #S = marked_checks.sum()

        if cost_gauge == 1:
            faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0] # WHAAAT
            #if len(con3_faults) == 1:
            if len(faults_to_check) == 1:
                return cost_gauge, [faults_to_check[0]], function_calls
                #return cost_gauge, [con3_faults[0]], function_calls
            cost_gauge = 3

        # if cost_gauge > max_cost:
        #     return cost_gauge, []
        
        # if depth == 0: # after this we for sure have to go a recursion deeper, so check if we are allowed
        #     # ACTUALLY NOT NECESSARILY TRUE BUT WHATEVER
        #     return cost_gauge, [], function_calls
        
        while cost_gauge <= max_cost:
            if marked_checks.sum() <= cost_gauge:
                if cost_gauge == 4: # small optimization, only look around 1 particular check
                    faults_to_check = np.nonzero(graph[marked_checks][0])[0]
                else:
                    #faults_to_check = con_any_faults
                    #faults_to_check = np.nonzero(graph[marked_checks].any(axis=0))[0] # WHAAAT
                    faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 1)[0]]) # WHAAAT
            elif marked_checks.sum() <= 2*cost_gauge:
                #faults_to_check = con32_faults
                #faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 2)[0] # WHAAAT
                faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0]])
                # TODO: feasability filter
            else: #marked_checks.sum() <= 3*cost_gauge
                #faults_to_check = con3_faults
                faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
                if len(faults_to_check) < marked_checks.sum() - 2*cost_gauge: # feasability filter
                    faults_to_check = []

            # now that we now what faults to check, we iterate through them
            for fault in faults_to_check:
                marked_checks_prime = marked_checks ^ (graph[:, fault] == 1)

                # need to check here
                if max_function_calls - function_calls <= 0:
                    return cost_gauge, [], function_calls
                
                cost, path, new_function_calls = self.cost_optimized(marked_checks_prime, max_cost = cost_gauge-1, max_function_calls=max_function_calls - function_calls)
                function_calls += new_function_calls

                if cost == cost_gauge-1:
                    return cost_gauge, [fault] + path, function_calls
            
            # if at this point we haven't returned that means the actual cost is higher
            # so we increase our estimate of the cost (by 2, as parity is fixed)
            cost_gauge += 2
            
        return cost_gauge, [], function_calls

    def cost_debug(self, marked_checks: np.ndarray[bool], depth: int = 0, max_cost: int = 1000, history=None):
        self.function_calls_left -= 1

        graph = self.fault_graph
        cost_gauge = self.base_cost(marked_checks)

        if history is None:
            history = []
        
        if self.function_calls_left < 0:
            return cost_gauge, [], history

        new_node = True

        if cost_gauge == 0 or cost_gauge > max_cost:
            history.append([new_node, depth, cost_gauge, max_cost, np.nan]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            return cost_gauge, [], history
        # we could handle this case in the loop, but not necessary, 
        # will be called often, so good to increase performance
        # but will check if difference
        if cost_gauge == 1:
            # then there are exactly three marked checks, let's see if doable
            faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
            if len(faults_to_check) == 1:
                history.append([new_node, depth, cost_gauge, max_cost, 1]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
                return cost_gauge, list(faults_to_check), history
            else:
                history.append([new_node, depth, cost_gauge, max_cost, 0]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            cost_gauge = 3
            new_node = False
        if cost_gauge > max_cost:
            history.append([new_node, depth, cost_gauge, max_cost, np.nan]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            return cost_gauge, [], history
        
        if depth == 0: # after this we for sure have to go a recursion deeper, so check if we are allowed
            history.append([new_node, depth, cost_gauge, max_cost, np.nan]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            return cost_gauge, [], history
        
        while True:
            if marked_checks.sum() <= cost_gauge:
                if cost_gauge == 4: # small optimization, only look around 1 particular check
                    faults_to_check = np.nonzero(graph[marked_checks][0])[0]
                else:
                    #faults_to_check = np.nonzero(graph[marked_checks].any(axis=0))[0]
                    faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 1)[0]]) # WHAAAT
            elif marked_checks.sum() <= 2*cost_gauge:
                #faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 2)[0]
                faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0]])
                # TODO: feasability filter
            else: #marked_checks.sum() <= 3*cost_gauge
                faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
                if len(faults_to_check) < marked_checks.sum() - 2*cost_gauge: # feasability filter
                    faults_to_check = np.array([],dtype=int)

            # now that we now what faults to check, we iterate through them
            history.append([new_node, depth, cost_gauge, max_cost, len(faults_to_check)]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            new_node = False
            for fault in faults_to_check:
                # apply that fault to get new pattern
                marked_checks_prime = marked_checks ^ (graph[:, fault] == 1)
                # go a recursion step deeper, with depth=depth-1 (to stop the recursion at some depth)
                # and with max_cost_cost_gauge-1, we stop the recursion too if the best case scenario is not good enough
                # (branch cutting)
                # if self.function_calls_left <= 0:
                #     return cost_gauge, [], history # somehow return no path is worse than returning one additonal random node
                # might need 
                cost, path, lower_hist = self.cost_debug(marked_checks_prime, depth=depth-1, max_cost = cost_gauge-1)
                history.extend(lower_hist)
                if cost == cost_gauge-1 or self.function_calls_left < 0:
                    return cost_gauge, [fault] + path, history
            
            # if at this point we haven't returned that means the actual cost is higher
            # so we increase our estimate of the cost (by 2, as parity is fixed)
            cost_gauge += 2
            # if now we are above the max_cost we return
            if cost_gauge > max_cost:
                history.append([new_node, depth, cost_gauge, max_cost, np.nan])
                return cost_gauge, [], history
            
            # I'm sure this sob will terminate
    
    def decode(self, syndrome: np.ndarray, max_cost=5, max_function_calls=np.inf, debug=False, **kwargs):
        if debug:
            max_depth = kwargs.get('max_depth', 10)
            self.function_calls_left = max_function_calls
            cost_lower_bound, path, history = self.cost_debug(marked_checks=syndrome.astype(bool), depth=max_depth)
            full_error = np.zeros(self.fault_graph.shape[1], dtype=np.int8)
            #success = cost_lower_bound <= max_depth + 1
            for idx in path:
                full_error[idx] = not full_error[idx]
            return full_error, history
        
        else:
            cost_lower_bound, path, function_calls = self.cost_optimized(marked_checks=syndrome.astype(bool),
                                                                 max_cost=max_cost,
                                                                 max_function_calls=max_function_calls)
            full_error = np.zeros(self.fault_graph.shape[1], dtype=np.int8)
            #success = cost_lower_bound <= max_depth + 1
            for idx in path:
                full_error[idx] = not full_error[idx]
            return full_error, function_calls

    def __init__(self, fault_graph: np.ndarray) -> None:
        self.fault_graph = fault_graph

        self.check_to_fault =[[],[]]
        self.fault_to_check = [[],[]]        