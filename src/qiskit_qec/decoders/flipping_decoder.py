import numpy as np
from ldpc import bposd_decoder, bp_decoder

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

    def is_base_case(self, marked_checks: np.ndarray[bool]):
        graph = self.fault_graph
        if marked_checks.sum() == 3:
            faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
            if len(faults_to_check) == 1:
                return True, faults_to_check[0]
        return False, None
        
    def weight_iterative_with_bp(self, syndrome: np.ndarray, max_weight, max_iterations = np.inf):
        """
        Iterative minimum weight calculations.
        Returns weight_estimate (weight lower bound), minimum_weight_error, number_of_calls
        Raises MaxIterationsExceededException if a solution could not be found within max_iterations steps
        """

        graph = self.fault_graph
        syndrome = syndrome.astype(bool)

        is_base_case, sol = self.is_base_case(syndrome)
        if is_base_case:
            return 1, [sol]

        weight_estimate = self.base_cost(syndrome)

        if weight_estimate == 0 or weight_estimate > max_weight:
            return weight_estimate, []
        
        if weight_estimate == 1 and not is_base_case:
            weight_estimate = 3

        num_iterations = 0

        while weight_estimate <= max_weight:
            num_iterations += 1
            if num_iterations >= max_iterations:
                raise ValueError(f"MaxIterations {max_iterations} exceeded (Will be a custom exception)")

            # assume weight = weight_estimate, dynamic programming using that assumption
            nodes = [()]
            parent_lookup = {0: None}
            children_lookup = {0: []}

            current_node = 0

            while True:
                if weight_estimate <= 1:
                    faults_to_check_mask = np.array([0])
                elif syndrome.sum() <= weight_estimate:
                    if weight_estimate == 4: # small optimization, only look around 1 particular check
                        faults_to_check_mask = graph[syndrome][0] == 1
                        #faults_to_check_mask = faults_to_check[len(children_lookup[current_node]):]
                    else:
                        faults_to_check_mask = graph[syndrome].any(axis=0)
                        #faults_to_check = np.hstack([np.nonzero(graph[syndrome].sum(axis=0) == 3)[0], np.nonzero(graph[syndrome].sum(axis=0) == 2)[0], np.nonzero((graph[syndrome].sum(axis=0) == 1) & graph[np.nonzero(syndrome)[0][0]])[0]])
                        #faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                elif syndrome.sum() <= 2*weight_estimate:
                    faults_to_check_mask = graph[syndrome].sum(axis=0) >= 2

                    #faults_to_check = np.hstack([np.nonzero(graph[syndrome].sum(axis=0) == 3)[0], np.nonzero(graph[syndrome].sum(axis=0) == 2)[0]])
                    #faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                    # TODO: feasability filter
                else: #marked_checks.sum() <= 3*weight_estimate
                    faults_to_check_mask = graph[syndrome].sum(axis=0) >= 3
                    #faults_to_check = np.nonzero(graph[syndrome].sum(axis=0) >= 3)[0]
                    #faults_to_check = faults_to_check[len(children_lookup[current_node]):]

                num_remaining_candidates = faults_to_check_mask.sum() - len(children_lookup[current_node]) #total - the ones checked already
                if num_remaining_candidates < syndrome.sum() - 2*weight_estimate: # feasability filter
                    faults_to_check_mask = np.array([0])

                if faults_to_check_mask.sum() > 0:
                    #use bpd to find likelihoods of faults being in Fhat
                    self.bpd.decode(syndrome)
                    # calculate actual candidate fault indices and sort them according to their likelihood
                    faults_to_check = np.where(faults_to_check_mask)[0][np.argsort(self.bpd.log_prob_ratios[faults_to_check_mask])]
                    # the fault we check is the next one we haven't checked
                    fault = faults_to_check[len(children_lookup[current_node])]
                    new_node_idx = len(nodes)
                    new_fault_set = nodes[current_node] + (fault,)
                    children_lookup[current_node].append(new_node_idx)
                    parent_lookup[new_node_idx] = current_node
                    nodes.append(new_fault_set)
                    children_lookup[new_node_idx] = []
                    current_node = new_node_idx
                    syndrome = syndrome ^ (graph[:, fault] == 1)
                    is_base_case, sol = self.is_base_case(syndrome)
                    weight_estimate -= 1

                    if is_base_case:
                        new_node_idx = len(nodes)
                        new_fault_set = nodes[current_node] + (sol,)
                        nodes.append(new_fault_set)
                        parent_lookup[new_node_idx] = current_node
                        children_lookup[new_node_idx] = []
                        return new_fault_set, new_node_idx, nodes, parent_lookup, children_lookup
                    
                    if self.base_cost(syndrome) > weight_estimate:
                        #raise ValueError("this conidtion is met")
                        parent_idx = parent_lookup[current_node]
                        if parent_idx is None:
                            break
                        fault = nodes[current_node][-1]
                        syndrome = syndrome ^ (graph[:, fault] == 1)
                        current_node = parent_idx
                        weight_estimate += 1
                    
                    
                else:
                    parent_idx = parent_lookup[current_node]
                    if parent_idx is None:
                        break
                    fault = nodes[current_node][-1]
                    syndrome = syndrome ^ (graph[:, fault] == 1)
                    current_node = parent_idx
                    weight_estimate += 1

            weight_estimate += 2

    def weight_iterative(self, syndrome: np.ndarray, max_weight, max_iterations = np.inf):
        """
        Iterative minimum weight calculations.
        Returns weight_estimate (weight lower bound), minimum_weight_error, number_of_calls
        Raises MaxIterationsExceededException if a solution could not be found within max_iterations steps
        """
        graph = self.fault_graph
        syndrome = syndrome.astype(bool)

        is_base_case, sol = self.is_base_case(syndrome)
        if is_base_case:
            return 1, [sol]

        weight_estimate = self.base_cost(syndrome)

        if weight_estimate == 0 or weight_estimate > max_weight:
            return weight_estimate, []
        
        if weight_estimate == 1 and not is_base_case:
            weight_estimate = 3

        num_iterations = 0

        while weight_estimate <= max_weight:
            num_iterations += 1
            if num_iterations >= max_iterations:
                raise ValueError(f"MaxIterations {max_iterations} exceeded (Will be a custom exception)")

            # assume weight = weight_estimate, dynamic programming using that assumption
            nodes = [()]
            parent_lookup = {0: None}
            children_lookup = {0: []}

            current_node = 0

            while True:
                if weight_estimate <= 1:
                    faults_to_check = []
                elif syndrome.sum() <= weight_estimate:
                    if weight_estimate == 4: # small optimization, only look around 1 particular check
                        faults_to_check = np.nonzero(graph[syndrome][0])[0]
                        faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                    else:
                        faults_to_check = np.hstack([np.nonzero(graph[syndrome].sum(axis=0) == 3)[0], np.nonzero(graph[syndrome].sum(axis=0) == 2)[0], np.nonzero((graph[syndrome].sum(axis=0) == 1) & graph[np.nonzero(syndrome)[0][0]])[0]])
                        faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                elif syndrome.sum() <= 2*weight_estimate:
                    faults_to_check = np.hstack([np.nonzero(graph[syndrome].sum(axis=0) == 3)[0], np.nonzero(graph[syndrome].sum(axis=0) == 2)[0]])
                    faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                    # TODO: feasability filter
                else: #marked_checks.sum() <= 3*weight_estimate
                    faults_to_check = np.nonzero(graph[syndrome].sum(axis=0) >= 3)[0]
                    faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                    if len(faults_to_check) < syndrome.sum() - 2*weight_estimate: # feasability filter
                        faults_to_check = np.array([],dtype=int)

                if len(faults_to_check) > 0:
                    fault = faults_to_check[0]
                    new_node_idx = len(nodes)
                    new_fault_set = nodes[current_node] + (fault,)
                    children_lookup[current_node].append(new_node_idx)
                    parent_lookup[new_node_idx] = current_node
                    nodes.append(new_fault_set)
                    children_lookup[new_node_idx] = []
                    current_node = new_node_idx
                    syndrome = syndrome ^ (graph[:, fault] == 1)
                    is_base_case, sol = self.is_base_case(syndrome)
                    weight_estimate -= 1

                    if is_base_case:
                        new_node_idx = len(nodes)
                        new_fault_set = nodes[current_node] + (sol,)
                        nodes.append(new_fault_set)
                        parent_lookup[new_node_idx] = current_node
                        children_lookup[new_node_idx] = []
                        return new_fault_set, new_node_idx, nodes, parent_lookup, children_lookup
                    
                    if self.base_cost(syndrome) > weight_estimate:
                        #raise ValueError("this conidtion is met")
                        parent_idx = parent_lookup[current_node]
                        if parent_idx is None:
                            break
                        fault = nodes[current_node][-1]
                        syndrome = syndrome ^ (graph[:, fault] == 1)
                        current_node = parent_idx
                        weight_estimate += 1
                    
                    
                else:
                    parent_idx = parent_lookup[current_node]
                    if parent_idx is None:
                        break
                    fault = nodes[current_node][-1]
                    syndrome = syndrome ^ (graph[:, fault] == 1)
                    current_node = parent_idx
                    weight_estimate += 1

            weight_estimate += 2

    def cost_iterative_total(self, marked_checks: np.ndarray[bool], max_cost):
        graph = self.fault_graph

        is_base_case, sol = self.is_base_case(marked_checks)
        if is_base_case:
            return 1, [sol]

        cost_gauge = self.base_cost(marked_checks)

        if cost_gauge == 0 or cost_gauge > max_cost:
            return cost_gauge, []
        
        if cost_gauge == 1 and not is_base_case:
            cost_gauge = 3

        while cost_gauge <= max_cost:
            res = self.cost_iterative_try(marked_checks, cost_gauge)
            if res == 'fail':
                cost_gauge += 2
                continue
            new_fault_set, new_node_idx, nodes, parent_lookup, children_lookup = res
            error = list(new_fault_set)
            return len(error), error, new_node_idx
        
        return cost_gauge, [], -1

    def cost_iterative_try(self, marked_checks: np.ndarray[bool], cost_gauge):
        """ This works if cost==cost_gauge. Need wrapper for cost_gauge increase, or one function for all """
        nodes = [()]
        parent_lookup = {0: None}
        children_lookup = {0: []}

        current_node = 0

        graph = self.fault_graph

        while True:
            if cost_gauge <= 1:
                faults_to_check = []
            elif marked_checks.sum() <= cost_gauge:
                if cost_gauge == 4: # small optimization, only look around 1 particular check
                    faults_to_check = np.nonzero(graph[marked_checks][0])[0]
                    faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                else:
                    #faults_to_check = np.nonzero(graph[marked_checks].any(axis=0))[0]
                    #faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 1)[0]]) # WHAAAT
                    faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero((graph[marked_checks].sum(axis=0) == 1) & graph[np.nonzero(marked_checks)[0][0]])[0]]) # WHAAAT
                    faults_to_check = faults_to_check[len(children_lookup[current_node]):]
            elif marked_checks.sum() <= 2*cost_gauge:
                #faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 2)[0]
                faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0]])
                faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                # TODO: feasability filter
            else: #marked_checks.sum() <= 3*cost_gauge
                faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
                faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                if len(faults_to_check) < marked_checks.sum() - 2*cost_gauge: # feasability filter
                    faults_to_check = np.array([],dtype=int)

            if len(faults_to_check) > 0:
                fault = faults_to_check[0]
                new_node_idx = len(nodes)
                new_fault_set = nodes[current_node] + (fault,)
                children_lookup[current_node].append(new_node_idx)
                parent_lookup[new_node_idx] = current_node
                nodes.append(new_fault_set)
                children_lookup[new_node_idx] = []
                current_node = new_node_idx
                marked_checks = marked_checks ^ (graph[:, fault] == 1)
                is_base_case, sol = self.is_base_case(marked_checks)
                cost_gauge -= 1
                if is_base_case:
                    new_node_idx = len(nodes)
                    new_fault_set = nodes[current_node] + (sol,)
                    nodes.append(new_fault_set)
                    parent_lookup[new_node_idx] = current_node
                    children_lookup[new_node_idx] = []
                    return new_fault_set, new_node_idx, nodes, parent_lookup, children_lookup
                
            else:
                parent_idx = parent_lookup[current_node]
                if parent_idx is None:
                    return 'fail'
                fault = nodes[current_node][-1]
                marked_checks = marked_checks ^ (graph[:, fault] == 1)
                current_node = parent_idx
                cost_gauge += 1

    def cost_optimized(self, marked_checks: np.ndarray[bool], max_cost, max_function_calls=np.inf, pre_estimate=None):
        """
        Returns cost estimate, the faults to flip and the number of function calls
        """
        function_calls = 1

        graph = self.fault_graph
        cost_gauge = self.base_cost(marked_checks)
        if pre_estimate is not None and pre_estimate > cost_gauge:
            cost_gauge = pre_estimate

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
                    #faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 1)[0]]) # WHAAAT
                    faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero((graph[marked_checks].sum(axis=0) == 1) & graph[np.nonzero(marked_checks)[0][0]])[0]])
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
                
                cost, path, new_function_calls = self.cost_optimized(marked_checks_prime, max_cost = cost_gauge-1, max_function_calls=max_function_calls - function_calls, pre_estimate=cost_gauge-1)
                function_calls += new_function_calls

                if cost == cost_gauge-1:
                    return cost_gauge, [fault] + path, function_calls
            
            # if at this point we haven't returned that means the actual cost is higher
            # so we increase our estimate of the cost (by 2, as parity is fixed)
            cost_gauge += 2
            
        return cost_gauge, [], function_calls

    def cost_debug(self, marked_checks: np.ndarray[bool], depth: int = 0, max_cost: int = 1000, history=None, pre_estimate=None):
        self.function_calls_left -= 1

        graph = self.fault_graph
        cost_gauge = self.base_cost(marked_checks)
        if pre_estimate is not None and pre_estimate > cost_gauge:
            cost_gauge = pre_estimate

        if history is None:
            history = []
        
        if self.function_calls_left < 0:
            return cost_gauge, [], history

        new_node = True

        if cost_gauge == 0 or cost_gauge > max_cost:
            history.append([[], new_node, depth, cost_gauge, max_cost, np.nan]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            return cost_gauge, [], history
        # we could handle this case in the loop, but not necessary, 
        # will be called often, so good to increase performance
        # but will check if difference
        if cost_gauge == 1:
            # then there are exactly three marked checks, let's see if doable
            faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
            if len(faults_to_check) == 1:
                history.append([[], new_node, depth, cost_gauge, max_cost, 1]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
                return cost_gauge, list(faults_to_check), history
            else:
                history.append([[], new_node, depth, cost_gauge, max_cost, 0]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            cost_gauge = 3
            new_node = False
        if cost_gauge > max_cost:
            history.append([[], new_node, depth, cost_gauge, max_cost, np.nan]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            return cost_gauge, [], history
        
        if depth == 0: # after this we for sure have to go a recursion deeper, so check if we are allowed
            history.append([[], new_node, depth, cost_gauge, max_cost, np.nan]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
            return cost_gauge, [], history
        
        while True:
            if marked_checks.sum() <= cost_gauge:
                if cost_gauge == 4: # small optimization, only look around 1 particular check
                    faults_to_check = np.nonzero(graph[marked_checks][0])[0]
                else:
                    #faults_to_check = np.nonzero(graph[marked_checks].any(axis=0))[0]
                    #faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 1)[0]]) # WHAAAT
                    faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0], np.nonzero((graph[marked_checks].sum(axis=0) == 1) & graph[np.nonzero(marked_checks)[0][0]])[0]]) # WHAAAT
            elif marked_checks.sum() <= 2*cost_gauge:
                #faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 2)[0]
                faults_to_check = np.hstack([np.nonzero(graph[marked_checks].sum(axis=0) == 3)[0], np.nonzero(graph[marked_checks].sum(axis=0) == 2)[0]])
                # TODO: feasability filter
            else: #marked_checks.sum() <= 3*cost_gauge
                faults_to_check = np.nonzero(graph[marked_checks].sum(axis=0) >= 3)[0]
                if len(faults_to_check) < marked_checks.sum() - 2*cost_gauge: # feasability filter
                    faults_to_check = np.array([],dtype=int)

            # now that we now what faults to check, we iterate through them
            history.append([[], new_node, depth, cost_gauge, max_cost, len(faults_to_check)]) # [new_node, depth_left, cost_gauge, max_cost, num_childs]
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
                cost, path, lower_hist = self.cost_debug(marked_checks_prime, depth=depth-1, max_cost = cost_gauge-1, pre_estimate=cost_gauge-1)
                for step in lower_hist:
                    step[0].append(fault)
                history.extend(lower_hist)
                if cost == cost_gauge-1 or self.function_calls_left < 0:
                    return cost_gauge, [fault] + path, history
            
            # if at this point we haven't returned that means the actual cost is higher
            # so we increase our estimate of the cost (by 2, as parity is fixed)
            cost_gauge += 2
            # if now we are above the max_cost we return
            if cost_gauge > max_cost:
                history.append([[], new_node, depth, cost_gauge, max_cost, np.nan])
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

    def __init__(self, fault_graph: np.ndarray, max_iter=1024) -> None:
        self.fault_graph = fault_graph

        bpd=bp_decoder(
            self.fault_graph,#the parity check matrix
            error_rate=0.01,# dummy error rate
            channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
            max_iter=max_iter, #the maximum number of iterations for BP)
            bp_method="ms",
            ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
            )
        self.bpd = bpd

        self.check_to_fault =[[],[]]
        self.fault_to_check = [[],[]]

    def parse_debug_history(self, history):
        num_nodes = 0
        num_true_nodes = 0
        nodes = {

        }
        tree_lib_nodes = {

        }

        def find_parent(level):
            parent = None
            for node_idx in nodes:
                if nodes[node_idx]['level'] == level:
                    parent = node_idx
            return parent

        def find_parent_treelib(level):
            parent = None
            for node_idx in tree_lib_nodes:
                if tree_lib_nodes[node_idx]['level'] == level:
                    parent = node_idx
            return parent

        for node in history:
            if node[1]:
                nodes[num_true_nodes] = {'disp': f'NODE: {node[0][::-1]}, initial_gauge:{node[3]}, max_gauge:{node[4]}', 'level': node[2], 'parent': find_parent(node[2]+1), 'path': np.sort(node[0]), 'path_sequence': node[0][::-1]}
                tree_lib_nodes[num_nodes] = {'disp': f'NODE: {node[0][::-1]}, initial_gauge:{node[3]}, max_gauge:{node[4]}', 'level': node[2], 'parent': find_parent_treelib(node[2]+1), 'path': np.sort(node[0]), 'path_sequence': node[0][::-1]}
                num_true_nodes +=1
                num_nodes += 1
                tree_lib_nodes[num_nodes] = {'disp': f'{node[3]}->{node[5]}', 'level': node[2]-1, 'parent': num_nodes-1, 'path': np.sort(node[0]), 'path_sequence': node[0][::-1]}
                num_nodes +=1
            else:
                tree_lib_nodes[num_nodes] = {'disp': f'{node[3]}->{node[5]}', 'level': node[2]-1, 'parent': find_parent_treelib(node[2]), 'path': np.sort(node[0]), 'path_sequence': node[0][::-1]}
                num_nodes +=1

        return nodes, num_true_nodes, tree_lib_nodes
    
    def print_debug_history_tree(self, history, resolve_weight_updates=True):
        from treelib import Node, Tree # type: ignore

        sparse_nodes, num_true_nodes, tree_lib_nodes = self.parse_debug_history(history=history)

        tree = Tree()

        if resolve_weight_updates:
            for node_idx in tree_lib_nodes:
                #print(tree_lib_nodes[node_idx]['disp'])
                tree.create_node(tree_lib_nodes[node_idx]['disp'], node_idx, parent=tree_lib_nodes[node_idx]['parent'])

        else:
            for node_idx in sparse_nodes:
                #print(tree_lib_nodes[node_idx]['disp'])
                tree.create_node(sparse_nodes[node_idx]['disp'], node_idx, parent=sparse_nodes[node_idx]['parent'])

        print(tree.show(stdout=False, sorting=False))

    def plot_subproblem_graph_from_iterative(self, nodes, parent_lookup, fig_size = (12,4), full_search_tree = True, **nx_kwargs):
        import networkx as nx
        import matplotlib.pyplot as plt

        graph = nx.DiGraph()

        if full_search_tree:
            for node in nodes:
                graph.add_node(node)

            for node_idx, parent_idx in parent_lookup.items():
                if parent_idx is None:
                    continue
                graph.add_edge(nodes[parent_idx], nodes[node_idx])

        else:
            for node in nodes:
                graph.add_node(tuple(np.sort(node)))

            for node_idx, parent_idx in parent_lookup.items():
                if parent_idx is None:
                    continue
                graph.add_edge(tuple(np.sort(nodes[parent_idx])), tuple(np.sort(nodes[node_idx])))

        # Define a function to get the level of each node
        def get_levels(graph):
            levels = {}
            for node, data in graph.nodes(data=True):
                levels[node] = len(node) #data['level']
            return levels

        # Get the levels of nodes
        levels = get_levels(graph)

        # Assign positions for nodes (y-coordinate fixed by level, x-coordinate free)
        pos = {}
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        for level in level_nodes:
            num_nodes = len(level_nodes[level])
            for i, node in enumerate(level_nodes[level]):
                pos[node] = (i*8- 4*len(level_nodes[level]), -level)  # x is free (i), y is fixed (-level to draw downward)
        print(graph.number_of_nodes())
        plt.figure(figsize=fig_size)
        nx_kwargs['with_labels'] = nx_kwargs.get('with_labels', False)
        nx_kwargs['arrows'] = nx_kwargs.get('arrows', True)
        nx_kwargs['node_size'] = nx_kwargs.get('node_size', 50)
        nx_kwargs['node_color'] = nx_kwargs.get('node_color', "lightblue")
        nx_kwargs['font_size'] = nx_kwargs.get('font_size', 10)
        nx.draw(graph, pos, **nx_kwargs)
        plt.show()

    def plot_subproblem_graph_from_debug_history(self, history, fig_size = (12,4), full_search_tree = True, **nx_kwargs):
        import networkx as nx
        import matplotlib.pyplot as plt

        graph = nx.DiGraph()

        nodes, num_true_nodes, tree_lib_nodes = self.parse_debug_history(history=history)

        if full_search_tree:
            path_identifier = 'path_sequence'
        else:
            path_identifier = 'path'

        for node_idx, node in nodes.items():
            graph.add_node(tuple(node[path_identifier]), level=node['level'])

        for node_idx, node in nodes.items():
            if node['parent'] is None:
                continue
            graph.add_edge(tuple(nodes[node['parent']][path_identifier]), tuple(node[path_identifier]))

        # Define a function to get the level of each node
        def get_levels(graph):
            levels = {}
            for node, data in graph.nodes(data=True):
                levels[node] = data['level']
            return levels

        # Get the levels of nodes
        levels = get_levels(graph)

        # Assign positions for nodes (y-coordinate fixed by level, x-coordinate free)
        pos = {}
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        for level in level_nodes:
            num_nodes = len(level_nodes[level])
            for i, node in enumerate(level_nodes[level]):
                pos[node] = (i*8- 4*len(level_nodes[level]), level)  # x is free (i), y is fixed (-level to draw downward)

        print(graph.number_of_nodes())
        plt.figure(figsize=fig_size)
        nx_kwargs['with_labels'] = nx_kwargs.get('with_labels', False)
        nx_kwargs['arrows'] = nx_kwargs.get('arrows', True)
        nx_kwargs['node_size'] = nx_kwargs.get('node_size', 50)
        nx_kwargs['node_color'] = nx_kwargs.get('node_color', "lightblue")
        nx_kwargs['font_size'] = nx_kwargs.get('font_size', 10)
        nx.draw(graph, pos, **nx_kwargs)
        plt.show()
