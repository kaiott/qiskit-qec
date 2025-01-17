import numpy as np
import heapq
from ldpc import bposd_decoder, bp_decoder


class MinimumWeightNoBPDecisionTreeDecoder():
    def __init__(self, fault_graph: np.ndarray, idx2color: np.ndarray) -> None:
        self.fault_graph = fault_graph
        self.idx2color = idx2color

        # initialize adjacency matrices
        self.check_to_fault = dict()
        for check_index in range(self.fault_graph.shape[0]):
            fault_indices = frozenset(np.where(self.fault_graph[check_index])[0])
            self.check_to_fault[check_index] = fault_indices

        self.fault_to_check = dict()
        for fault_index in range(self.fault_graph.shape[1]):
            check_indices = frozenset(np.where(self.fault_graph[:,fault_index])[0])
            self.fault_to_check[fault_index] = check_indices

    def get_descendants(self, c, F, s):
        """ Implements the growing step of a leaf efficiently (instead of iterating over faults) """
        # choose check index next to syndrome with lowest degree still available
        idx = np.argmin([len(self.check_to_fault[check_index] - F) for check_index in s])
        check_index = list(s)[idx]

        # look at those faults, they are branches down to descendants
        down_branches = list(self.check_to_fault[check_index] - F)

        relevant_costs = np.zeros(len(down_branches))
        
            
        return down_branches, relevant_costs

    def decode(self, syndrome: np.ndarray[bool], max_depth = np.inf, max_iterations = np.inf):
        """ Fast, non-provable version of decision tree decoding. Relies on BP and does not take structure into account.
        Decimation (once implemented in ldpc fork) done around lowest degree check node.
        Parameters:
            early_exit: whether we return a solution direclty in BP converged. default False (but will be True at some point).
            max_depth: limit depth of decision tree, i.e. nodes of that depth will not have any descendants.
            max_iterations: maximum nodes that will be explored, if exceeded will raise Exception.
            bp_buffer_method: whether and how to include the buffer of BP LLRs """
        
        # initialization
        s = frozenset(np.where(syndrome)[0]) # sparse version of syndrome
        if len(s) == 0:
            return frozenset(), 'initial_exit', np.zeros(self.fault_graph.shape[1],dtype=bool), set(), []
        tree_nodes = set([frozenset()]) # initialize identified nodes with root (empty fault set)
        unexplored_leaves = [] # min-heap / priority queue for unexplored but identified leaves
        wlb = self.weight_lower_bound(s, frozenset())
        heapq.heappush(unexplored_leaves, ((wlb, 0), frozenset(), s)) # add empty set / root with cost 0 to it.

        # decoding
        while len(unexplored_leaves) > 0:

            if len(tree_nodes)-len(unexplored_leaves) >= max_iterations:
                return frozenset(), 'fail', np.zeros(self.fault_graph.shape[1],dtype=bool), tree_nodes, unexplored_leaves

            c, F, s = heapq.heappop(unexplored_leaves) # get next unexplored leaf

            # now get children... runs BP
            down_branches, relevant_costs = self.get_descendants(c,F,s)
            
            # else add children
            for f, dc in zip(down_branches, relevant_costs):
                F_prime = F | frozenset([f])
                if F_prime in tree_nodes: # if node already discovered, ignore
                    continue
                c_prime = c[1] + dc # c[1] is the cost from BP
                s_prime = s ^ self.fault_to_check[f]
                if len(s_prime) == 0: # if we have a solution we return it right away (before this is actually explored)
                    decoded_error = np.zeros(self.fault_graph.shape[1],dtype=bool)
                    decoded_error[list(F_prime)] = True
                    return F_prime, 'late_exit', decoded_error, tree_nodes, unexplored_leaves
                
                # calculate weight lower bound
                wlb = self.weight_lower_bound(s_prime, F_prime)

                # otherwise add the new leaf to data structures
                tree_nodes.add(F_prime)
                heapq.heappush(unexplored_leaves, ((wlb, c_prime), F_prime, s_prime))

    def weight_lower_bound(self, s, F):
        bound_w = int(np.ceil((len(s)-self.count_f3(s,F))/2))
        bound_c = np.max(self.syndrome2color_count(s))

        bound = max(bound_c, bound_w)
        bound_with_history = bound + len(F)

        return bound_with_history
    
    def count_f3(self, s, F):
        neigbour_faults = set()
        for check_index in s:
            neigbour_faults |= self.check_to_fault[check_index]
        neigbour_faults -= F

        f3s = 0
        for fault_index in neigbour_faults:
            if len(self.fault_to_check[fault_index] & s) == 3:
                f3s += 1
        return f3s
    
    def syndrome2color_count(self, syndrome):
        color, count = np.unique(self.idx2color[list(syndrome)], return_counts=True)
        return count[np.argsort(color)]

class BreadthFirstDecisionTreeDecoder():
    def __init__(self, fault_graph: np.ndarray) -> None:
        self.fault_graph = fault_graph

        # initialize adjacency matrices
        self.check_to_fault = dict()
        for check_index in range(self.fault_graph.shape[0]):
            fault_indices = frozenset(np.where(self.fault_graph[check_index])[0])
            self.check_to_fault[check_index] = fault_indices

        self.fault_to_check = dict()
        for fault_index in range(self.fault_graph.shape[1]):
            check_indices = frozenset(np.where(self.fault_graph[:,fault_index])[0])
            self.fault_to_check[fault_index] = check_indices

    def get_descendants(self, c, F, s):
        """ Implements the growing step of a leaf efficiently (instead of iterating over faults) """
        # choose check index next to syndrome with lowest degree still available
        idx = np.argmin([len(self.check_to_fault[check_index] - F) for check_index in s])
        check_index = list(s)[idx]

        # look at those faults, they are branches down to descendants
        down_branches = list(self.check_to_fault[check_index] - F)
        
        relevant_costs = np.ones(len(down_branches))
            
        return down_branches, relevant_costs
    
    def decode(self, syndrome: np.ndarray[bool], max_depth = np.inf, max_iterations = np.inf):
        """ Fast, non-provable version of decision tree decoding. Relies on BP and does not take structure into account.
        Decimation (once implemented in ldpc fork) done around lowest degree check node.
        Parameters:
            early_exit: whether we return a solution direclty in BP converged. default False (but will be True at some point).
            max_depth: limit depth of decision tree, i.e. nodes of that depth will not have any descendants.
            max_iterations: maximum nodes that will be explored, if exceeded will raise Exception."""
        
        # initialization
        s = frozenset(np.where(syndrome)[0]) # sparse version of syndrome
        if len(s) == 0:
            return frozenset(), 'initial_exit', np.zeros(self.fault_graph.shape[1],dtype=bool), set(), []
        tree_nodes = set([frozenset()]) # initialize identified nodes with root (empty fault set)
        unexplored_leaves = [] # min-heap / priority queue for unexplored but identified leaves
        heapq.heappush(unexplored_leaves, (0, frozenset(), s)) # add empty set / root with cost 0 to it.

        # decoding
        while len(unexplored_leaves) > 0:

            if len(tree_nodes)-len(unexplored_leaves) >= max_iterations:
                return frozenset(), 'fail', np.zeros(self.fault_graph.shape[1],dtype=bool), tree_nodes, unexplored_leaves

            c, F, s = heapq.heappop(unexplored_leaves) # get next unexplored leaf

            # now get children... runs BP
            down_branches, relevant_costs = self.get_descendants(c,F,s)
            
            # else add children
            for f, dc in zip(down_branches, relevant_costs):
                F_prime = F | frozenset([f])
                if F_prime in tree_nodes: # if node already discovered, ignore
                    continue
                c_prime = c + dc
                s_prime = s ^ self.fault_to_check[f]
                if len(s_prime) == 0: # if we have a solution we return it right away (before this is actually explored)
                    decoded_error = np.zeros(self.fault_graph.shape[1],dtype=bool)
                    decoded_error[list(F_prime)] = True
                    return F_prime, 'late_exit', decoded_error, tree_nodes, unexplored_leaves

                # otherwise add the new leaf to data structures
                tree_nodes.add(F_prime)
                heapq.heappush(unexplored_leaves, (c_prime, F_prime, s_prime))

class MinimumWeightDecisionTreeDecoder():
    def __init__(self, fault_graph: np.ndarray, idx2color: np.ndarray, max_iter=1024, buffer_length=1, priors=None, init_max_iter=None, learning_rate=1) -> None:
        self.fault_graph = fault_graph
        self.idx2color = idx2color

        if priors is None: priors = [None]

        self.max_iter = max_iter
        if init_max_iter is None:
            init_max_iter = max_iter
        self.init_max_iter = init_max_iter

        bpd=bp_decoder(
            self.fault_graph,#the parity check matrix
            error_rate=0.01,# dummy error rate
            channel_probs=priors, #assign error_rate to each qubit. This will override "error_rate" input variable
            max_iter=max_iter, #the maximum number of iterations for BP)
            bp_method="ms",
            ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
            buffer_length=buffer_length,
            learning_rate=learning_rate,
            )
        self.bpd = bpd

        # initialize adjacency matrices
        self.check_to_fault = dict()
        for check_index in range(self.fault_graph.shape[0]):
            fault_indices = frozenset(np.where(self.fault_graph[check_index])[0])
            self.check_to_fault[check_index] = fault_indices

        self.fault_to_check = dict()
        for fault_index in range(self.fault_graph.shape[1]):
            check_indices = frozenset(np.where(self.fault_graph[:,fault_index])[0])
            self.fault_to_check[fault_index] = check_indices
        
        self.c = max([len(checks) for checks in self.fault_to_check.values()]) #max number of checks per fault

    def get_descendants(self, c, F, s, bp_buffer_method):
        """ Implements the growing step of a leaf efficiently (instead of iterating over faults) """
        # choose check index next to syndrome with lowest degree still available
        idx = np.argmin([len(self.check_to_fault[check_index] - F) for check_index in s])
        check_index = list(s)[idx]

        # look at those faults, they are branches down to descendants
        down_branches = list(self.check_to_fault[check_index] - F)
        
        # create full syndrome for BP
        syndrome = np.zeros(self.fault_graph.shape[0],dtype=bool)
        syndrome[list(s)] = True

        # run BP
        #self.bpd.decode(syndrome)
        if len(F) == 0:
            self.bpd.max_iter = self.init_max_iter
        else:
            self.bpd.max_iter = self.max_iter
            
        self.bpd.decode_decimated(syndrome, F)

        if bp_buffer_method is None or self.bpd.converge:
            llrs = self.bpd.log_prob_ratios[down_branches]
            relevant_costs = llrs + np.log(1+np.exp(-llrs))
            #relevant_costs = np.piecewise(llrs, 
            #                            [llrs <= -32, (llrs > 32) & (llrs < 32), llrs >= 32],
            #                            [lambda llr: 0, lambda llr: llr + np.log(1+np.exp(-llr)), lambda llr: llr])
        else:
            if bp_buffer_method == 'max':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].max(axis=0)
                relevant_costs = llrs + np.log(1+np.exp(-llrs))
            elif bp_buffer_method == 'min':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].min(axis=0)
                relevant_costs = llrs + np.log(1+np.exp(-llrs))
            elif bp_buffer_method == 'mean':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].mean(axis=0)
                #relevant_costs = np.piecewise(llrs, 
                #                              [llrs <= -32, (llrs > 32) & (llrs < 32), llrs >= 32],
                #                              [lambda llr: 0, lambda llr: llr + np.log(1+np.exp(-llr)), lambda llr: llr])
                relevant_costs = llrs + np.log(1+np.exp(-llrs))
            elif (type(bp_buffer_method) == int or type(bp_buffer_method) == float) and bp_buffer_method > 0:
                zcr = np.count_nonzero(np.diff(np.sign(self.bpd.log_prob_ratios_buffer[:,down_branches]), axis=0), axis=0)/len(self.bpd.log_prob_ratios_buffer)
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].mean(axis=0)
                cost_unweighted = llrs + np.log(1+np.exp(-llrs))
                t = zcr**(1/bp_buffer_method)
                relevant_costs = -np.log(0.5)*t + cost_unweighted*(1-t)
            else:
                raise TypeError(f'Invalid bp_buffer_method {bp_buffer_method}. Must be one of Union[None, "max", "min", "mean"] or a numeric value')
            
        return down_branches, relevant_costs

    def decode(self, syndrome: np.ndarray[bool], early_exit = False, max_depth = np.inf, max_iterations = np.inf, bp_buffer_method=None):
        """ Fast, non-provable version of decision tree decoding. Relies on BP and does not take structure into account.
        Decimation (once implemented in ldpc fork) done around lowest degree check node.
        Parameters:
            early_exit: whether we return a solution direclty in BP converged. default False (but will be True at some point).
            max_depth: limit depth of decision tree, i.e. nodes of that depth will not have any descendants.
            max_iterations: maximum nodes that will be explored, if exceeded will raise Exception.
            bp_buffer_method: whether and how to include the buffer of BP LLRs """
        
        # initialization
        s = frozenset(np.where(syndrome)[0]) # sparse version of syndrome
        if len(s) == 0:
            return frozenset(), 'initial_exit', np.zeros(self.fault_graph.shape[1],dtype=bool), set(), []
        tree_nodes = set([frozenset()]) # initialize identified nodes with root (empty fault set)
        unexplored_leaves = [] # min-heap / priority queue for unexplored but identified leaves
        # wlb = self.weight_lower_bound(s, frozenset())
        if len(self.idx2color)!=0:
            wlb = self.weight_lower_bound(s, frozenset())
        else:
            wlb = self.weight_lower_bound_2(s, frozenset())

        heapq.heappush(unexplored_leaves, ((wlb, 0), frozenset(), s)) # add empty set / root with cost 0 to it.

        # decoding
        while len(unexplored_leaves) > 0:

            if len(tree_nodes)-len(unexplored_leaves) >= max_iterations:
                return frozenset(), 'fail', np.zeros(self.fault_graph.shape[1],dtype=bool), tree_nodes, unexplored_leaves

            c, F, s = heapq.heappop(unexplored_leaves) # get next unexplored leaf

            # now get children... runs BP
            down_branches, relevant_costs = self.get_descendants(c,F,s, bp_buffer_method)
            
            # if BP converged and we allow early exit return solution
            if early_exit and self.bpd.converge and self.bpd.bp_decoding.sum() + len(F) == c[0]:
                decoded_error = self.bpd.bp_decoding # decoding from bp
                for f in F:
                    decoded_error[f] = 1
                return frozenset(np.where(decoded_error)[0]), 'early_exit', decoded_error, tree_nodes, unexplored_leaves
            
            # else add children
            for f, dc in zip(down_branches, relevant_costs):
                F_prime = F | frozenset([f])
                if F_prime in tree_nodes: # if node already discovered, ignore
                    continue
                c_prime = c[1] + dc # c[1] is the cost from BP
                s_prime = s ^ self.fault_to_check[f]
                if len(s_prime) == 0: # if we have a solution we return it right away (before this is actually explored)
                    decoded_error = np.zeros(self.fault_graph.shape[1],dtype=bool)
                    decoded_error[list(F_prime)] = True
                    return F_prime, 'late_exit', decoded_error, tree_nodes, unexplored_leaves
                
                # calculate weight lower bound
                if len(self.idx2color)!=0:
                    wlb = self.weight_lower_bound(s_prime, F_prime)
                else:
                    wlb = self.weight_lower_bound_2(s_prime, F_prime)

                # otherwise add the new leaf to data structures
                tree_nodes.add(F_prime)
                heapq.heappush(unexplored_leaves, ((wlb, c_prime), F_prime, s_prime))

    def weight_lower_bound_2(self, s, F):
        """returns the more refined genral bound h2"""
        a = np.zeros(self.c)
        for check_index in s:
            sensitivity = 1
            neigbour_faults = self.check_to_fault[check_index]
            for fault in neigbour_faults:
                sensitivity = max(sensitivity, len(self.fault_to_check[fault] & s))
            a[sensitivity-1] += 1
        q=0 #q_c
        h2=0
        l=self.c
        while l>0:
            h2 += np.floor((q+a[l-1])/l)
            q = (q+a[l-1])%l
            l-=1
        bound = h2
        bound_with_history = bound + len(F)

        return bound_with_history
    

    def weight_lower_bound(self, s, F):
        bound_w = int(np.ceil((len(s)-self.count_f3(s,F))/2))
        bound_c = np.max(self.syndrome2color_count(s))

        bound = max(bound_c, bound_w)
        bound_with_history = bound + len(F)

        return bound_with_history
    
    def count_f3(self, s, F):
        neigbour_faults = set()
        for check_index in s:
            neigbour_faults |= self.check_to_fault[check_index]
        neigbour_faults -= F

        f3s = 0
        for fault_index in neigbour_faults:
            if len(self.fault_to_check[fault_index] & s) == 3:
                f3s += 1
        return f3s
    
    def syndrome2color_count(self, syndrome):
        color, count = np.unique(self.idx2color[list(syndrome)], return_counts=True)
        return count[np.argsort(color)]
    
class GeneralFlippingDecoder():

    def get_descendants(self, c, F, s, bp_buffer_method):
        """ Implements the growing step of a leaf efficiently (instead of iterating over faults) """
        # choose check index next to syndrome with lowest degree still available
        idx = np.argmin([len(self.check_to_fault[check_index] - F) for check_index in s])
        check_index = list(s)[idx]

        # look at those faults, they are branches down to descendants
        down_branches = list(self.check_to_fault[check_index] - F)
        
        # create full syndrome for BP
        syndrome = np.zeros(self.fault_graph.shape[0],dtype=bool)
        syndrome[list(s)] = True

        # run BP
        #self.bpd.decode(syndrome)
        if len(F) == 0:
            self.bpd.max_iter = self.init_max_iter
        else:
            self.bpd.max_iter = self.max_iter
            
        self.bpd.decode_decimated(syndrome, F)

        if bp_buffer_method is None or self.bpd.converge:
            llrs = self.bpd.log_prob_ratios[down_branches]
            relevant_costs = llrs + np.log(1+np.exp(-llrs))
            #relevant_costs = np.piecewise(llrs, 
            #                            [llrs <= -32, (llrs > 32) & (llrs < 32), llrs >= 32],
            #                            [lambda llr: 0, lambda llr: llr + np.log(1+np.exp(-llr)), lambda llr: llr])
        else:
            if bp_buffer_method == 'max':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].max(axis=0)
                relevant_costs = llrs + np.log(1+np.exp(-llrs))
            elif bp_buffer_method == 'min':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].min(axis=0)
                relevant_costs = llrs + np.log(1+np.exp(-llrs))
            elif bp_buffer_method == 'mean':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].mean(axis=0)
                #relevant_costs = np.piecewise(llrs, 
                #                              [llrs <= -32, (llrs > 32) & (llrs < 32), llrs >= 32],
                #                              [lambda llr: 0, lambda llr: llr + np.log(1+np.exp(-llr)), lambda llr: llr])
                relevant_costs = llrs + np.log(1+np.exp(-llrs))
            elif (type(bp_buffer_method) == int or type(bp_buffer_method) == float) and bp_buffer_method > 0:
                zcr = np.count_nonzero(np.diff(np.sign(self.bpd.log_prob_ratios_buffer[:,down_branches]), axis=0), axis=0)/len(self.bpd.log_prob_ratios_buffer)
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].mean(axis=0)
                cost_unweighted = llrs + np.log(1+np.exp(-llrs))
                t = zcr**(1/bp_buffer_method)
                relevant_costs = -np.log(0.5)*t + cost_unweighted*(1-t)
            else:
                raise TypeError(f'Invalid bp_buffer_method {bp_buffer_method}. Must be one of Union[None, "max", "min", "mean"] or a numeric value')
            
        return down_branches, relevant_costs


    def decode(self, syndrome: np.ndarray[bool], early_exit = False, max_depth = np.inf, max_iterations = np.inf, bp_buffer_method=None):
        """ Fast, non-provable version of decision tree decoding. Relies on BP and does not take structure into account.
        Decimation (once implemented in ldpc fork) done around lowest degree check node.
        Parameters:
            early_exit: whether we return a solution direclty in BP converged. default False (but will be True at some point).
            max_depth: limit depth of decision tree, i.e. nodes of that depth will not have any descendants.
            max_iterations: maximum nodes that will be explored, if exceeded will raise Exception.
            bp_buffer_method: whether and how to include the buffer of BP LLRs """
        
        # initialization
        s = frozenset(np.where(syndrome)[0]) # sparse version of syndrome
        if len(s) == 0:
            return frozenset(), 'initial_exit', np.zeros(self.fault_graph.shape[1],dtype=bool), set(), []
        tree_nodes = set([frozenset()]) # initialize identified nodes with root (empty fault set)
        unexplored_leaves = [] # min-heap / priority queue for unexplored but identified leaves
        heapq.heappush(unexplored_leaves, (0, frozenset(), s)) # add empty set / root with cost 0 to it.

        # decoding
        while len(unexplored_leaves) > 0:

            if len(tree_nodes)-len(unexplored_leaves) >= max_iterations:
                return frozenset(), 'fail', np.zeros(self.fault_graph.shape[1],dtype=bool), tree_nodes, unexplored_leaves

            c, F, s = heapq.heappop(unexplored_leaves) # get next unexplored leaf

            # now get children... runs BP
            down_branches, relevant_costs = self.get_descendants(c,F,s, bp_buffer_method)
            
            # if BP converged and we allow early exit return solution
            if early_exit and self.bpd.converge:
                decoded_error = self.bpd.bp_decoding # decoding from bp
                for f in F:
                    decoded_error[f] = 1
                return frozenset(np.where(decoded_error)[0]), 'early_exit', decoded_error, tree_nodes, unexplored_leaves
            
            # else add children
            for f, dc in zip(down_branches, relevant_costs):
                F_prime = F | frozenset([f])
                if F_prime in tree_nodes: # if node already discovered, ignore
                    continue
                c_prime = c + dc
                s_prime = s ^ self.fault_to_check[f]
                if len(s_prime) == 0: # if we have a solution we return it right away (before this is actually explored)
                    decoded_error = np.zeros(self.fault_graph.shape[1],dtype=bool)
                    decoded_error[list(F_prime)] = True
                    return F_prime, 'late_exit', decoded_error, tree_nodes, unexplored_leaves
                # otherwise add the new leaf to data structures
                tree_nodes.add(F_prime)
                heapq.heappush(unexplored_leaves, (c_prime, F_prime, s_prime))
                   
    def __init__(self, fault_graph: np.ndarray, max_iter=1024, buffer_length=1, priors=None, init_max_iter=None, learning_rate=1):
        self.fault_graph = fault_graph

        if priors is None: priors = [None]

        self.max_iter = max_iter
        if init_max_iter is None:
            init_max_iter = max_iter
        self.init_max_iter = init_max_iter

        bpd=bp_decoder(
            self.fault_graph,#the parity check matrix
            error_rate=0.01,# dummy error rate
            channel_probs=priors, #assign error_rate to each qubit. This will override "error_rate" input variable
            max_iter=max_iter, #the maximum number of iterations for BP)
            bp_method="ms",
            ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
            buffer_length=buffer_length,
            learning_rate=learning_rate,
            )
        self.bpd = bpd

        # initialize adjacency matrices
        self.check_to_fault = dict()
        for check_index in range(self.fault_graph.shape[0]):
            fault_indices = frozenset(np.where(self.fault_graph[check_index])[0])
            self.check_to_fault[check_index] = fault_indices

        self.fault_to_check = dict()
        for fault_index in range(self.fault_graph.shape[1]):
            check_indices = frozenset(np.where(self.fault_graph[:,fault_index])[0])
            self.fault_to_check[fault_index] = check_indices

class GeneralOracleFlippingDecoder():

    def incremental_cost(self, llrs):
        alpha = 6
        beta = 1
        c1 = (alpha+beta) / np.pi
        c2 = 0.5
        c3 = 2
        c4 = (alpha-beta) / 2
        return c1*np.arctan(c2*(llrs - c3)) + c4

    def get_descendants(self, c, F, s, bp_buffer_method):
        """ Implements the growing step of a leaf efficiently (instead of iterating over faults) """
        # choose check index next to syndrome with lowest degree still available
        idx = np.argmin([len(self.check_to_fault[check_index] - F) for check_index in s])
        check_index = list(s)[idx]

        # look at those faults, they are branches down to descendants
        down_branches = list(self.check_to_fault[check_index] - F)
        
        # create full syndrome for BP
        syndrome = np.zeros(self.fault_graph.shape[0],dtype=bool)
        syndrome[list(s)] = True

        # run BP
        #self.bpd.decode(syndrome)
        if len(F) == 0:
            self.bpd.max_iter = self.init_max_iter
        else:
            self.bpd.max_iter = self.max_iter
            
        self.bpd.decode_decimated(syndrome, F)

        if bp_buffer_method is None or self.bpd.converge:
            llrs = self.bpd.log_prob_ratios[down_branches]
            relevant_costs = llrs + np.log(1+np.exp(-llrs))
            #relevant_costs = np.piecewise(llrs, 
            #                            [llrs <= -32, (llrs > 32) & (llrs < 32), llrs >= 32],
            #                            [lambda llr: 0, lambda llr: llr + np.log(1+np.exp(-llr)), lambda llr: llr])
        else:
            if bp_buffer_method == 'max':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].max(axis=0)
                relevant_costs = self.incremental_cost(llrs)
            elif bp_buffer_method == 'min':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].min(axis=0)
                relevant_costs = self.incremental_cost(llrs)
            elif bp_buffer_method == 'mean':
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].mean(axis=0)
                #relevant_costs = np.piecewise(llrs, 
                #                              [llrs <= -32, (llrs > 32) & (llrs < 32), llrs >= 32],
                #                              [lambda llr: 0, lambda llr: llr + np.log(1+np.exp(-llr)), lambda llr: llr])
                relevant_costs = self.incremental_cost(llrs)
            elif (type(bp_buffer_method) == int or type(bp_buffer_method) == float) and bp_buffer_method > 0:
                zcr = np.count_nonzero(np.diff(np.sign(self.bpd.log_prob_ratios_buffer[:,down_branches]), axis=0), axis=0)/len(self.bpd.log_prob_ratios_buffer)
                llrs = self.bpd.log_prob_ratios_buffer[:,down_branches].mean(axis=0)
                cost_unweighted = self.incremental_cost(llrs)
                t = zcr**(1/bp_buffer_method)
                relevant_costs = -np.log(0.5)*t + cost_unweighted*(1-t)
            else:
                raise TypeError(f'Invalid bp_buffer_method {bp_buffer_method}. Must be one of Union[None, "max", "min", "mean"] or a numeric value')
            
        return down_branches, relevant_costs

    def decode(self, syndrome: np.ndarray[bool], early_exit = False, max_depth = np.inf, max_iterations = np.inf, bp_buffer_method=None):
        """ Fast, non-provable version of decision tree decoding. Relies on BP and does not take structure into account.
        Decimation (once implemented in ldpc fork) done around lowest degree check node.
        Parameters:
            early_exit: whether we return a solution direclty in BP converged. default False (but will be True at some point).
            max_depth: limit depth of decision tree, i.e. nodes of that depth will not have any descendants.
            max_iterations: maximum nodes that will be explored, if exceeded will raise Exception.
            bp_buffer_method: whether and how to include the buffer of BP LLRs """
        
        # initialization
        s = frozenset(np.where(syndrome)[0]) # sparse version of syndrome
        if len(s) == 0:
            return frozenset(), 'initial_exit', np.zeros(self.fault_graph.shape[1],dtype=bool), set(), []
        tree_nodes = set([frozenset()]) # initialize identified nodes with root (empty fault set)
        unexplored_leaves = [] # min-heap / priority queue for unexplored but identified leaves
        heapq.heappush(unexplored_leaves, (0, frozenset(), s)) # add empty set / root with cost 0 to it.

        # decoding
        while len(unexplored_leaves) > 0:

            if len(tree_nodes)-len(unexplored_leaves) >= max_iterations:
                return frozenset(), 'fail', np.zeros(self.fault_graph.shape[1],dtype=bool), tree_nodes, unexplored_leaves

            c, F, s = heapq.heappop(unexplored_leaves) # get next unexplored leaf

            # now get children... runs BP
            down_branches, relevant_costs = self.get_descendants(c,F,s, bp_buffer_method)
            
            # if BP converged and we allow early exit return solution
            if early_exit and self.bpd.converge:
                decoded_error = self.bpd.bp_decoding # decoding from bp
                for f in F:
                    decoded_error[f] = 1
                return frozenset(np.where(decoded_error)[0]), 'early_exit', decoded_error, tree_nodes, unexplored_leaves
            
            # else add children
            for f, dc in zip(down_branches, relevant_costs):
                F_prime = F | frozenset([f])
                if F_prime in tree_nodes: # if node already discovered, ignore
                    continue
                c_prime = c + dc
                s_prime = s ^ self.fault_to_check[f]
                if len(s_prime) == 0: # if we have a solution we return it right away (before this is actually explored)
                    decoded_error = np.zeros(self.fault_graph.shape[1],dtype=bool)
                    decoded_error[list(F_prime)] = True
                    return F_prime, 'late_exit', decoded_error, tree_nodes, unexplored_leaves
                # otherwise add the new leaf to data structures
                tree_nodes.add(F_prime)
                heapq.heappush(unexplored_leaves, (c_prime, F_prime, s_prime))
                   
    def __init__(self, fault_graph: np.ndarray, max_iter=1024, buffer_length=1, priors=None, init_max_iter=None, learning_rate=1):
        self.fault_graph = fault_graph

        if priors is None: priors = [None]

        self.max_iter = max_iter
        if init_max_iter is None:
            init_max_iter = max_iter
        self.init_max_iter = init_max_iter

        bpd=bp_decoder(
            self.fault_graph,#the parity check matrix
            error_rate=0.01,# dummy error rate
            channel_probs=priors, #assign error_rate to each qubit. This will override "error_rate" input variable
            max_iter=max_iter, #the maximum number of iterations for BP)
            bp_method="ms",
            ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
            buffer_length=buffer_length,
            learning_rate=learning_rate,
            )
        self.bpd = bpd

        # initialize adjacency matrices
        self.check_to_fault = dict()
        for check_index in range(self.fault_graph.shape[0]):
            fault_indices = frozenset(np.where(self.fault_graph[check_index])[0])
            self.check_to_fault[check_index] = fault_indices

        self.fault_to_check = dict()
        for fault_index in range(self.fault_graph.shape[1]):
            check_indices = frozenset(np.where(self.fault_graph[:,fault_index])[0])
            self.fault_to_check[fault_index] = check_indices

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
        
    def get_next_edge_idx(self, node_idx, nodes, children_lookup, syndrome, weight_lb, use_bp=True):
        """ assume use_bp always true, will implement the false case later
         Returns index of edge in children lookup """
        # if children have already been calculated, use that
        if node_idx in children_lookup:
            edge_list = children_lookup[node_idx]
            # find first edge that points to non-exisiting child node
            for edge_idx, (edge_label, child_idx) in enumerate(edge_list):
                if child_idx is None:
                    return edge_idx
            # else if no such edge exists anymore return None
            return None

        # else if the true lower bound of this syndrome is already higher than the desired one, we stop evaluating it
        if self.base_cost(syndrome) > weight_lb:
            children_lookup[node_idx] = [] # this node will not have any children
            return None # so we return None
        
        # else we calculate all possible candidate faults, sort them, and store them
        # based on the syndrome "weight" and the weight lower bound restrict candidates
        if syndrome.sum() <= weight_lb:
            if weight_lb == 4:
                candidate_faults_mask = self.fault_graph[syndrome][0] == 1
            else:
                candidate_faults_mask = self.fault_graph[syndrome].any(axis=0)
        elif syndrome.sum() <= 2*weight_lb:
            candidate_faults_mask = self.fault_graph[syndrome].sum(axis=0) >= 2
        else: #syndrome.sum() <= 3*weight_lb
            candidate_faults_mask = self.fault_graph[syndrome].sum(axis=0) >= 3
        
        # remove candidates that are already in fault sequence (do not take double click paths)
        for fault in nodes[node_idx]:
            candidate_faults_mask[fault] = 0

        # if we do memoization, we should filter out some more candidates here
        # for debugging purposes (being able to not do memoization, or do memoization but see subproblem graph)
        # we do not do this here, but in the main algorithm, only linear overhead in num iterations

        minimum_needed_errors = max(1, syndrome.sum() - 2*weight_lb)
        if candidate_faults_mask.sum() < minimum_needed_errors:
            children_lookup[node_idx] = [] # this node will not have any children
            return None # so we return None
        
        # else we sort them according to BP
        self.bpd.decode(syndrome)
        # calculate actual candidate fault indices and sort them according to their likelihood
        candidate_fault_indices = np.where(candidate_faults_mask)[0][np.argsort(self.bpd.log_prob_ratios[candidate_faults_mask])]
        
        # if the minimum_needed_errors is e.g. 3, in the worse case there are 3 errors in the end of the list
        # while before there are none. But that means we can safely discard the last 2 faults
        if minimum_needed_errors > 1:
            candidate_fault_indices = candidate_fault_indices[:-minimum_needed_errors+1]

        # now that we got all candidate faults we have to check in the right order
        # store this information for later
        children_lookup[node_idx] = [(edge_label, None) for edge_label in candidate_fault_indices]
        return 0 # and we return 0 to imply that we start with the first edge in the list        

    def weight_priority_queue(self, syndrome: np.ndarray[bool], max_weight, max_iterations = np.inf, memoize=True, increase=2, initial_lb=0, bp_buffer_method=None):
        # fuck the base case...
        syndrome = syndrome.astype(bool) # just in case I forget...

        weight_lb = self.base_cost(syndrome)
        weight_lb = max(weight_lb, initial_lb)

        num_iterations = 0
        debug_object = []

        if weight_lb == 0:
            return 0, (), num_iterations, debug_object
        
        def update_nodes(current_node_idx, nodes, syndrome, weight_lb):
            current_node_label, _, current_logperr, _ = nodes[current_node_idx]
            weight_lb -= len(current_node_label)

            # if the true lower bound of this syndrome is already higher than the desired one, no children
            if self.base_cost(syndrome) > weight_lb:
                return
            
            # else we calculate all possible candidate faults, sort them, and store them
            # based on the syndrome "weight" and the weight lower bound restrict candidates
            if syndrome.sum() <= weight_lb:
                if weight_lb == 4:
                    candidate_faults_mask = self.fault_graph[syndrome][0] == 1
                else:
                    candidate_faults_mask = self.fault_graph[syndrome].any(axis=0)
                    # should not be like this, but let's leave it for now
            elif syndrome.sum() <= 2*weight_lb:
                candidate_faults_mask = self.fault_graph[syndrome].sum(axis=0) >= 2
            else: #syndrome.sum() <= 3*weight_lb
                candidate_faults_mask = self.fault_graph[syndrome].sum(axis=0) >= 3
            
            # remove candidates that are already in fault sequence (do not take double click paths)
            for fault in current_node_label:
                candidate_faults_mask[fault] = 0

            # calculate candidate fault indices
            candidate_fault_indices = np.where(candidate_faults_mask)[0]

            # remove candidates that lead to already visited node (memoization)
            for idx, (node_label, _, logperr, visited) in enumerate(nodes):
                for fault_idx in candidate_fault_indices:
                    if tuple(np.sort(current_node_label + (fault_idx,))) == node_label:
                        # if not visited:
                            # update logperr

                        candidate_faults_mask[fault_idx] = 0

            minimum_needed_errors = max(1, syndrome.sum() - 2*weight_lb)
            if candidate_faults_mask.sum() < minimum_needed_errors:
                return # this node will not have any children
            
            # else we sort them according to BP
            self.bpd.decode(syndrome)
            if bp_buffer_method is None or self.bpd.converge:
                llrs = self.bpd.log_prob_ratios[candidate_faults_mask]
                relevant_costs = llrs + np.log(1+np.exp(-llrs))
            else:
                if bp_buffer_method == 'max':
                    llrs = self.bpd.log_prob_ratios_buffer[:,candidate_faults_mask].max(axis=0)
                    relevant_costs = llrs + np.log(1+np.exp(-llrs))
                elif bp_buffer_method == 'min':
                    llrs = self.bpd.log_prob_ratios_buffer[:,candidate_faults_mask].min(axis=0)
                    relevant_costs = llrs + np.log(1+np.exp(-llrs))
                elif bp_buffer_method == 'mean':
                    llrs = self.bpd.log_prob_ratios_buffer[:,candidate_faults_mask].mean(axis=0)
                    relevant_costs = llrs + np.log(1+np.exp(-llrs))
                elif (type(bp_buffer_method) == int or type(bp_buffer_method) == float) and bp_buffer_method > 0:
                    zcr = np.count_nonzero(np.diff(np.sign(self.bpd.log_prob_ratios_buffer[:,candidate_faults_mask]), axis=0), axis=0)/len(self.bpd.log_prob_ratios_buffer)
                    llrs = self.bpd.log_prob_ratios_buffer[:,candidate_faults_mask].mean(axis=0)
                    cost_unweighted = llrs + np.log(1+np.exp(-llrs))
                    t = zcr**(1/bp_buffer_method)
                    relevant_costs = -np.log(0.5)*t + cost_unweighted*(1-t)
                else:
                    raise TypeError(f'Invalid bp_buffer_method {bp_buffer_method}. Must be one of Union[None, "max", "min", "mean"] or a numeric value')
            # calculate actual candidate fault indices and sort them according to their likelihood
            #sorter = np.argsort(self.bpd.log_prob_ratios[candidate_faults_mask])
            candidate_fault_cost = np.sort(relevant_costs)
            candidate_fault_indices = np.where(candidate_faults_mask)[0][np.argsort(relevant_costs)]
            
            # if the minimum_needed_errors is e.g. 3, in the worse case there are 3 errors in the end of the list
            # while before there are none. But that means we can safely discard the last 2 faults
            if minimum_needed_errors > 1:
                candidate_fault_indices = candidate_fault_indices[:-minimum_needed_errors+1]

            # now that we got all candidate faults we have to check in the right order
            # store this information for later
            for edge_cost, candidate_fault in zip(candidate_fault_cost, candidate_fault_indices):
                node_label = tuple(np.sort(current_node_label + (candidate_fault,)))
                logperr = current_logperr + edge_cost
                nodes.append((node_label, current_node_idx, logperr, False))

        def get_next_unvisited_idx(nodes):
            best_idx = None
            best_logperr = np.inf
            for idx, (node_label, parent_idx, logperr, visited) in enumerate(nodes):
                if not visited and logperr < best_logperr:
                    best_logperr = logperr
                    best_idx = idx
            return best_idx


        original_syndrome = syndrome.copy()

        while weight_lb <= max_weight:
            nodes = [((), None, 0, False)] # list of nodes, a node is a tuple (node_label, parent_idx, logperr, visited)
            #parent_lookup = {0: (None, None)} # node_idx: (edge_label, parent_idx)
            #children_lookup = {} # node_idx: [(edge_label, child_idx), (edge_label, child_idx), ..., (edge_label, None)]

            tree_object = {
                'weight_bound': weight_lb,
                'num_iterations': 0,
                'nodes': nodes,
            }

            debug_object.append(tree_object)

            while True: # The backtracking dynamic programming part tries to find a solution with weight <= weight_lb
                tree_object['num_iterations'] += 1 # of the current subroutine
                num_iterations += 1 # total
                if num_iterations >= max_iterations:
                    raise ValueError(f"MaxIterations {max_iterations} exceeded (Will be a custom exception)")
                
                current_node_idx = get_next_unvisited_idx(nodes)

                if current_node_idx is None:
                    # no more nodes to visit
                    break
                
                # else we go visit this node
                node_label, parent_idx, logperr, visited = nodes[current_node_idx]
                nodes[current_node_idx] = (node_label, parent_idx, logperr, True) # update that we visited this node
                syndrome = original_syndrome ^ (self.fault_graph[:, node_label].sum(axis=1) % 2 == 1)
                if syndrome.sum() == 0: #syndrome.any()
                    return len(node_label), node_label, num_iterations, debug_object
                # if self.base_cost(syndrome) > weight_lb - len(node_label): # in this case do not look at kids
                #    continue
                update_nodes(current_node_idx, nodes, syndrome, weight_lb)

                #this should be everything, no????                    

            # if this failed, increase the weight lower bound
            weight_lb += increase

        # if unsuccessful, return
        return weight_lb, None, num_iterations, debug_object

    def weight_reworked(self, syndrome: np.ndarray[bool], max_weight, max_iterations = np.inf, memoize=True, increase=2, initial_lb=0):
        # fuck the base case...
        syndrome = syndrome.astype(bool) # just in case I forget...

        weight_lb = self.base_cost(syndrome)
        weight_lb = max(weight_lb, initial_lb)

        num_iterations = 0
        debug_object = []

        if weight_lb == 0:
            return 0, (), num_iterations, debug_object

        while weight_lb <= max_weight:
            nodes = [()] # [(), (f1), (f1,f2), (f1,f3), (f4), ...]
            parent_lookup = {0: (None, None)} # node_idx: (edge_label, parent_idx)
            children_lookup = {} # node_idx: [(edge_label, child_idx), (edge_label, child_idx), ..., (edge_label, None)]

            tree_object = {
                'weight_bound': weight_lb,
                'num_iterations': 0,
                'nodes': nodes,
                'parent_lookup': parent_lookup,
                'children_lookup': children_lookup
            }

            debug_object.append(tree_object)
            
            current_node_idx = 0

            while True: # The backtracking dynamic programming part tries to find a solution with weight <= weight_lb
                tree_object['num_iterations'] += 1 # of the current subroutine
                num_iterations += 1 # total
                if num_iterations >= max_iterations:
                    raise ValueError(f"MaxIterations {max_iterations} exceeded (Will be a custom exception)")
                
                next_edge_idx = self.get_next_edge_idx(current_node_idx, nodes, children_lookup, syndrome, weight_lb)
                if next_edge_idx is not None:
                    next_fault = children_lookup[current_node_idx][next_edge_idx][0]
                    new_node_idx = len(nodes)
                    new_fault_set = nodes[current_node_idx] + (next_fault,)
                    if memoize:
                        new_fault_set = tuple(np.sort(new_fault_set))
                        children_lookup[current_node_idx][next_edge_idx] = (next_fault, new_node_idx)
                        if new_fault_set in nodes:
                            continue
                    else:
                        children_lookup[current_node_idx][next_edge_idx] = (next_fault, new_node_idx)
                    
                    # update parent
                    parent_lookup[new_node_idx] = (next_fault, current_node_idx) # update parent dictionary
                    nodes.append(new_fault_set) # update nodes list
                    current_node_idx = new_node_idx # go to the new node
                    syndrome = syndrome ^ (self.fault_graph[:, next_fault] == 1) # by altering the syndrome accordingly

                    # if syndrome is trivial, we found a solution
                    if syndrome.sum() == 0:
                        return len(new_fault_set), new_fault_set, num_iterations, debug_object
                    
                    # else start loop again, as we went down a node, decrease weight lower bound
                    weight_lb -= 1

                else: # this node has no more children to explore, go up (backtracking step)
                    fault, parent_idx = parent_lookup[current_node_idx]
                    if parent_idx is None: # we are at root, we stop trying this weight lower bound and increase it
                        break
                    
                    current_node_idx = parent_idx # go up to the parent node
                    syndrome = syndrome ^ (self.fault_graph[:, fault] == 1) # by undoing the flip on the syndrome
                    weight_lb += 1 # as we went up, increase the weight lower bound

            # if this failed, increase the weight lower bound
            weight_lb += increase

        # if unsuccessful, return
        return weight_lb, None, num_iterations, debug_object

    def weight_iterative_with_bp(self, syndrome: np.ndarray, max_weight, max_iterations = np.inf, memoize=True):
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
                        # assume weight = weight_estimate, dynamic programming using that assumption
            num_iterations_i = 0
            nodes = [()]
            parent_lookup = {0: None}
            children_lookup = {0: []}

            current_node = 0

            while True:
                num_iterations_i +=1
                num_iterations += 1
                if num_iterations >= max_iterations:
                    raise ValueError(f"MaxIterations {max_iterations} exceeded (Will be a custom exception)")

                if weight_estimate <= 1: # comes from the fact that the base case has been checked. go up the tree
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

                for f in nodes[current_node]:
                    if len(faults_to_check_mask) > 1: #otherwise it is empty...
                        faults_to_check_mask[f] = 0

                num_remaining_candidates = faults_to_check_mask.sum() - len(children_lookup[current_node]) #total - the ones checked already
                if num_remaining_candidates < max(1, syndrome.sum() - 2*weight_estimate): # feasability filter
                    faults_to_check_mask = np.array([0])

                if faults_to_check_mask.sum() > 0: # not good...
                    #use bpd to find likelihoods of faults being in Fhat
                    self.bpd.decode(syndrome) # technically belong into the get next fault function
                    # calculate actual candidate fault indices and sort them according to their likelihood
                    faults_to_check = np.where(faults_to_check_mask)[0][np.argsort(self.bpd.log_prob_ratios[faults_to_check_mask])]
                    # the fault we check is the next one we haven't checked
                    fault = faults_to_check[len(children_lookup[current_node])]
                    new_node_idx = len(nodes)
                    new_fault_set = nodes[current_node] + (fault,)
                    if memoize:
                        new_fault_set = tuple(np.sort(new_fault_set))
                        children_lookup[current_node].append(new_node_idx) # add to children lookup in any case, because otherwise endless loop..
                        if new_fault_set in nodes:
                            continue
                    else:
                        children_lookup[current_node].append(new_node_idx)
                    #children_lookup[current_node].append(new_node_idx)
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
                        #fault = nodes[current_node][-1]
                        if memoize:
                            # then fault is set difference
                            fault = [f for f in nodes[current_node] if f not in nodes[parent_idx]][0]
                        else:
                            fault = nodes[current_node][-1]
                        syndrome = syndrome ^ (graph[:, fault] == 1)
                        current_node = parent_idx
                        weight_estimate += 1
                    
                    
                else:
                    parent_idx = parent_lookup[current_node]
                    if parent_idx is None:
                        break
                    #fault = nodes[current_node][-1]
                    if memoize:
                        # then fault is set difference
                        fault = [f for f in nodes[current_node] if f not in nodes[parent_idx]][0]
                    else:
                        fault = nodes[current_node][-1]
                    syndrome = syndrome ^ (graph[:, fault] == 1)
                    current_node = parent_idx
                    weight_estimate += 1

            weight_estimate += 2

    def weight_iterative(self, syndrome: np.ndarray, max_weight, max_iterations = np.inf, memoize=True):
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
                        #faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                    else:
                        faults_to_check = np.hstack([np.nonzero(graph[syndrome].sum(axis=0) == 3)[0], np.nonzero(graph[syndrome].sum(axis=0) == 2)[0], np.nonzero((graph[syndrome].sum(axis=0) == 1) & graph[np.nonzero(syndrome)[0][0]])[0]])
                        #faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                elif syndrome.sum() <= 2*weight_estimate:
                    faults_to_check = np.hstack([np.nonzero(graph[syndrome].sum(axis=0) == 3)[0], np.nonzero(graph[syndrome].sum(axis=0) == 2)[0]])
                    #faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                    # TODO: feasability filter
                else: #marked_checks.sum() <= 3*weight_estimate
                    faults_to_check = np.nonzero(graph[syndrome].sum(axis=0) >= 3)[0]
                    #faults_to_check = faults_to_check[len(children_lookup[current_node]):]
                    #if len(faults_to_check) < syndrome.sum() - 2*weight_estimate: # feasability filter
                    #    faults_to_check = np.array([],dtype=int)

                # remove duplicates
                faults_to_check = [f for f in faults_to_check if f not in nodes[current_node]]

                num_remaining_candidates = len(faults_to_check) - len(children_lookup[current_node]) #total - the ones checked already
                if num_remaining_candidates < max(1, syndrome.sum() - 2*weight_estimate): # feasability filter
                    faults_to_check = []


                if len(faults_to_check) > 0:
                    fault = faults_to_check[len(children_lookup[current_node])]
                    new_node_idx = len(nodes)
                    new_fault_set = nodes[current_node] + (fault,)
                    if memoize:
                        new_fault_set = tuple(np.sort(new_fault_set))
                        children_lookup[current_node].append(new_node_idx) # add to children lookup in any case, because otherwise endless loop..
                        if new_fault_set in nodes:
                            continue
                    else:
                        children_lookup[current_node].append(new_node_idx)

                    parent_lookup[new_node_idx] = current_node # only if new node, as can only have one parent now. might change that
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
                        #fault = nodes[current_node][-1]
                        if memoize:
                            # then fault is set difference
                            fault = [f for f in nodes[current_node] if f not in nodes[parent_idx]][0]
                        else:
                            fault = nodes[current_node][-1]
                        syndrome = syndrome ^ (graph[:, fault] == 1)
                        current_node = parent_idx
                        weight_estimate += 1
                    
                    
                else:
                    parent_idx = parent_lookup[current_node]
                    if parent_idx is None:
                        break
                    if memoize:
                        # then fault is set difference
                        fault = [f for f in nodes[current_node] if f not in nodes[parent_idx]][0]
                    else:
                        fault = nodes[current_node][-1]
                    syndrome = syndrome ^ (graph[:, fault] == 1)
                    current_node = parent_idx
                    weight_estimate += 1

            weight_estimate += 2
        
        return 'fails'

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

    def __init__(self, fault_graph: np.ndarray, max_iter=1024, buffer_length=1) -> None:
        self.fault_graph = fault_graph

        bpd=bp_decoder(
            self.fault_graph,#the parity check matrix
            error_rate=0.01,# dummy error rate
            channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
            max_iter=max_iter, #the maximum number of iterations for BP)
            bp_method="ms",
            ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
            buffer_length=buffer_length,
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

    def plot_subproblem_graph_from_iterative_tree_object(self, tree_object, fig_size = (12,4), full_search_tree = True, **nx_kwargs):
        import networkx as nx
        import matplotlib.pyplot as plt

        nodes = tree_object['nodes']
        parent_lookup = tree_object['parent_lookup']

        graph = nx.DiGraph()

        if full_search_tree:
            for node in nodes:
                graph.add_node(node)

            edge_labels = {}
            for node_idx, (edge_label, parent_idx) in parent_lookup.items():
                if parent_idx is None:
                    continue
                graph.add_edge(nodes[parent_idx], nodes[node_idx])
                edge_labels[(nodes[parent_idx], nodes[node_idx])] = edge_label

        else:
            for node in nodes:
                graph.add_node(tuple(np.sort(node)))

            edge_labels = {}
            for node_idx, parent_idx in parent_lookup.items():
                if parent_idx is None:
                    continue
                graph.add_edge(tuple(np.sort(nodes[parent_idx])), tuple(np.sort(nodes[node_idx])))
                graph.add_edge(nodes[parent_idx], nodes[node_idx])

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
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.show()

    def plot_subproblem_graph_from_priority_queue(self, tree_object, fig_size = (12,4), full_search_tree = True, **nx_kwargs):
        import networkx as nx
        import matplotlib.pyplot as plt

        nodes = tree_object['nodes']

        graph = nx.DiGraph()

        edge_labels = {}
        for node_label, parent_idx, _, visited in nodes:
            if visited:
                graph.add_node(node_label)
                if parent_idx is not None:
                    try:
                        edge_label = list(set(node_label)-set(nodes[parent_idx][0]))[0]
                    except IndexError:
                        print(nodes[parent_idx][0])
                        print(node_label)
                        print(set(nodes[parent_idx][0]) - set(node_label))
                        print(list(set(nodes[parent_idx][0]) - set(node_label)))
                    graph.add_edge(nodes[parent_idx][0], node_label)
                    edge_labels[(nodes[parent_idx][0], node_label)] = edge_label

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
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
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
