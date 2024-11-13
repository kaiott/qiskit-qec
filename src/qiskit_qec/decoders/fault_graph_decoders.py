# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Hard decision renormalization group decoders."""

from abc import ABC
from copy import copy, deepcopy
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Dict, List, Set, Tuple

import numpy as np
from rustworkx import PyGraph, connected_components, distance_matrix

from qiskit_qec.analysis.lse_solvers import solve
from qiskit_qec.linear.matrix import rank
from qiskit_qec.decoders.decoding_graph import DecodingGraph
from qiskit_qec.utils import DecodingGraphEdge
from qiskit_qec.circuits import CSSCodeCircuit
from qiskit_qec.utils.stim_tools import get_stim_circuits

class WeightedFaultGraphUnionFindDecoderWrapper:
    CONFIG = {
            'name': 'Guided UF',
            'extra_connect_clusters': True,
            'lse_solver': 'numba',
            'do_initially_add_connecting_faults': False,
            'minimize_weight': False,
    }

    def __init__(self, code_circuit: CSSCodeCircuit):
        detectors, logicals = code_circuit.stim_detectors()
        stim_circuit = get_stim_circuits(code_circuit.noisy_circuit["0"], detectors=detectors, logicals=logicals)[0][0]
        error_model = stim_circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        self.num_checks, self.num_faults = error_model.num_detectors, error_model.num_errors
        print(f'Created error model {self.num_checks} detectors and {self.num_faults} errors')
        self.fault_graph, self.error_props, self.outcome_matrix = self.parse_error_model(error_model=error_model)

        weights = - np.log(self.error_props)
        self.wfguf = WeightedFaultGraphUnionFindDecoder(weights=weights, fault_graph=self.fault_graph)

    def get_random_error(self, suppression_factor = 1):
        return (np.random.random(self.num_faults) <= self.error_props*suppression_factor).astype(np.int8)
    
    def make_error(self, error_indices):
        error = np.zeros(self.num_faults, dtype=np.int8) # create error
        error[error_indices] = 1 # set weight bits to 1
        return error

    def get_syndrome(self, error):
        return self.fault_graph @ error % 2
    
    def get_logical_outcome(self, error):
        return self.outcome_matrix @ error % 2
    
    def error_equivalence(self, error_1, error_2):
        logical_1 = self.get_logical_outcome(error_1)
        logical_2 = self.get_logical_outcome(error_2)
        return np.array_equal(logical_1, logical_2)
    
    def decode(self, syndrome: np.ndarray):
        decoded_error, history = self.wfguf.decode(syndrome, **self.CONFIG)
        return decoded_error, history
    
    def process(self, outcome: str):
        # TODO outcome to syndrome
        syndrome = self.parse_outcome(outcome=outcome)
        decoded_error, history = self.decode(syndrome=syndrome)
        return list(self.get_logical_outcome(decoded_error))        
    
    def parse_error_model(self, error_model):
        prob_fault_graph = np.zeros((error_model.num_detectors, error_model.num_errors))
        outcome_matrix = np.zeros((error_model.num_observables, error_model.num_errors), dtype=np.int8)
        error_probs = np.zeros(error_model.num_errors)

        error_index = 0
        for d in error_model:
            if d.type=='error':
                error_prob = d.args_copy()[0]
                for target in d.targets_copy():
                    if target.is_relative_detector_id():
                        prob_fault_graph[target.val, error_index] = error_prob
                    if target.is_logical_observable_id():
                        outcome_matrix[target.val, error_index] = 1
                error_probs[error_index] = error_prob
                error_index += 1

        fault_graph = (prob_fault_graph > 0).astype(np.int8)
        return fault_graph, error_probs, outcome_matrix

    def parse_outcome(self, outcome: str) -> np.ndarray:
        pass

class WeightedFaultGraphUnionFindDecoder:
    def __init__(self, weights: np.ndarray, fault_graph: np.ndarray) -> None:
        """ Initializes the Tanner graph (simply with reduced adjacency matrix) """
        self.prime_weights = weights
        self.fault_graph = fault_graph

    class Cluster:
        """
        Always check surface
        """
        def __init__(self, checks_mask: np.ndarray, faults_mask: np.ndarray, biadjacency_matrix: np.ndarray, fault_weights: np.ndarray) -> None:
            self.checks_mask = checks_mask
            self.faults_mask = faults_mask
            self.reachable_faults_mask = biadjacency_matrix[checks_mask].any(axis=0) &~ self.faults_mask
            self.fault_weights = fault_weights
            self.valid = None
            self.internal_error = None
            self.biadjacency_matrix = biadjacency_matrix # for stuff like growing, finding true interior etc.
            self.internal_dof = np.nan
            self.minimized_internal = None

        @property
        def checks(self) -> np.ndarray:
            return np.where(self.checks_mask)[0]
        
        @property
        def faults(self) -> np.ndarray:
            return np.where(self.faults_mask)[0]

        def contains_check(self, check_idx) -> bool:
            return self.checks_mask[check_idx]
        
        def contains_fault(self, fault_idx) -> bool:
            return self.faults_mask[fault_idx]

        def touches(self, other: "WeightedFaultGraphUnionFindDecoder.Cluster") -> bool:
            return np.any(self.checks_mask & other.checks_mask)
        
        def overlaps(self, other: "WeightedFaultGraphUnionFindDecoder.Cluster") -> bool:
            return np.any(self.checks_mask | other.checks_mask) or np.any(self.faults_mask | other.faults_mask)

        def merge(self, other: "WeightedFaultGraphUnionFindDecoder.Cluster") -> None:
            """ merging happens in place. """
            self.checks_mask |= other.checks_mask
            self.faults_mask |= other.faults_mask
            self.reachable_faults_mask |= other.reachable_faults_mask
            self.valid = None
            self.internal_error = None

        def num_checks(self):
            return self.checks_mask.sum()
        
        def num_faults(self):
            return self.faults_mask.sum()
        
        def size(self):
            return self.num_checks() + self.num_faults()
        
        def shape(self):
            return self.num_checks(), self.num_faults()
        
        def is_valid(self, syndrome=None, **kwargs) -> bool:

            # if no syndrome provided return self.valid independent of whether computed or not
            if syndrome is None:
                return self.valid

            # if self.valid is set, assume validity has not changed
            if self.valid is not None:
                return self.valid

            # # get interior fault nodes
            # self.interior_faults_mask = self.get_interior_faults_mask(biadjacency_matrix)

            # system of equation extraction
            a = self.biadjacency_matrix[self.checks_mask][:,self.faults_mask]
            b = syndrome[self.checks_mask]

            # cluster is valid iff exists x s.t. ax = b
            solvable, x, stats = solve(a, b,
                                       minimize_weight = kwargs.get("minimize_weight", True),
                                       max_dof = kwargs.get("max_dof", 9),
                                       stats = True,
                                       lse_solver=kwargs.get("lse_solver"))
            self.internal_dof = stats['dof']
            self.minimized_internal = stats['minimized']
 
            if not solvable:
                self.valid = False
                return False
            
            self.valid = True

            # x is a mask of indices of error with respect to interior data
            # convert it to mask with respect to all fault nodes
            self.internal_error = np.zeros_like(self.faults_mask)
            self.internal_error[self.faults_mask] = x

            return True
        
        def decode_cluster_bposd(self, syndrome):
            from ldpc import bposd_decoder
            max_iter=1024
            # get interior fault nodes
            #self.interior_faults_mask = self.get_interior_faults_mask(self.biadjacency_matrix)

            # system of equation extraction
            a = self.biadjacency_matrix[self.checks_mask][:,self.faults_mask]
            b = syndrome[self.checks_mask]

            osd_order= 0
            while True:
                try:
                    bpd=bposd_decoder(
                        a ,#the parity check matrix
                        error_rate=0.01,# dummy error rate
                        channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
                        max_iter=max_iter, #the maximum number of iterations for BP)
                        bp_method="ms",
                        ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
                        osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                        osd_order=osd_order #the osd search depth
                        )
                    break
                except ValueError as e:
                    osd_order -= 1

            #return self.internal_error
            x = bpd.decode(b)

            # x is a mask of indices of error with respect to interior data
            # convert it to mask with respect to all fault nodes
            internal_error = np.zeros_like(self.faults_mask)
            internal_error[self.faults_mask] = x
            return internal_error

        # def neighbour_faults_mask(self) -> np.ndarray:
        #     """ if check surface return next faults that would be added with grow
        #     if fault surface, then return just the surface"""
        #     if self.cluster_surface_type == 'c':
        #         return self.biadjacency_matrix[self.cluster_surface_mask].any(axis=0) &~self.faults_mask
        #     else:
        #         return self.cluster_surface_mask
            
        # def neighbour_checks_mask(self) -> np.ndarray:
        #     if self.cluster_surface_type == 'f':
        #         return self.biadjacency_matrix[:, self.cluster_surface_mask].any(axis=1) &~self.checks_mask
        #     else:
        #         return self.cluster_surface_mask

        def grow(self) -> None:
            step_size = self.fault_weights[self.reachable_faults_mask].min()
            for fault in np.nonzero(self.reachable_faults_mask)[0]:
                self.fault_weights[fault] -= step_size
                if self.fault_weights[fault] == 0:
                    # we add this fault plus all it's neighbours
                    self.faults_mask[fault] = True
                    self.checks_mask = self.checks_mask | self.biadjacency_matrix[:,fault]
            
            # update reachable faults
            self.reachable_faults_mask = self.biadjacency_matrix[self.checks_mask].any(axis=0) &~ self.faults_mask
            self.valid = None

        def __repr__(self) -> str:
            string = f'checks: {list(self.checks)}\n'
            string += f'faults: {list(self.faults)}\n'
            if self.valid:
                string += f'internal error: {list(np.where(self.internal_error)[0])}\n'
            else:
                string += f'Not valid'
            return string
        
        def snapshot(self) -> Dict:
            return {'valid': self.valid,
                 'checks': list(self.checks),
                 'faults': list(self.faults),
                 'size': self.size(),
                 'num_checks': self.num_checks(),
                 'num_faults': self.num_faults(),
                 'dof': self.internal_dof,
                 'minimized': self.minimized_internal}

    def decode(self, syndrome: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Tuple[Tuple[int,int], float, int, List[Tuple[Tuple[int,int], float, int, float, float, int, float]], float]]]:
        """ Main entry point for decoding, takes syndrome and initialized clusters. 
        kwargs can be:
            predecode: bool, default is False. 
            do_initially_add_connecting_faults: bool, default is False
            do_initially_add_surrounded_faults: bool, default is False
            initial_cluster_strategy: str: Union[None, "connected_components"], default is None. Deprecated
            do_add_surrounded_faults: bool, default is False
            do_add_connecting_faults: bool, default is False
            grow_strategy: str = Union[None, "individual"], default is None
            extra_merging_strategy: str = Union[None, "extra"], default is None, Deprecated
            lse_solver: str = Union[None, "python", "numba", "c++"], default is None (which later resolves to numba)
            minimize_weight: bool, default is False
            max_dof: int, default is None (which later resolves to 9)
        """
        m, n = self.fault_graph.shape
        l = len(syndrome)
        cluster_history = [] # to record how clusters grow

        if l != m:
            raise ValueError(f"syndrome of length {l} is incompatible with number of checks {m} in fault graph")
        
        clusters = self._create_initial_clusters(syndrome, **kwargs)
        all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)

        while not all_valid:
            # TODO: add option and implementation to reweight faults depending on some factors
            clusters = self._grow_step(clusters=clusters, syndrome=syndrome, **kwargs)
            all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)
                    
        decoded_error = self._find_correction(clusters, syndrome, kwargs.get('cluster_decoding'))

        return decoded_error, cluster_history

    def _check_clusters(self, clusters: List["WeightedFaultGraphUnionFindDecoder.Cluster"], syndrome, cluster_history: List = None, **kwargs):
        """ 
        Goes through all clusters and checks if they are valid given the syndrome.
        If provided, will also append a snapshot of the current clusters to cluster_history.
        Returns all_valid: a bool indicating if all clusters are valid
        """
        clusters_snapshot = {'num_cluster': len(clusters), 'clusters': []}
        all_valid = True
        for idx, cluster in enumerate(clusters):
            cluster_is_valid = cluster.is_valid(syndrome, **kwargs)
            all_valid &= cluster_is_valid

            clusters_snapshot['clusters'].append(cluster.snapshot())

        if cluster_history is not None:
            cluster_history.append(clusters_snapshot)

        return all_valid
    
    def _get_smallest_invalid(self, clusters: List["WeightedFaultGraphUnionFindDecoder.Cluster"], syndrome, **kwargs):
        """ returns the index of the smallest invalid cluster """
        # find smallest invalid cluster
        smallest_invalid_index = -1
        for idx, cluster in enumerate(clusters):
            cluster_is_valid = cluster.is_valid(syndrome, **kwargs)

            if not cluster_is_valid and (smallest_invalid_index == -1 or cluster.size() < clusters[smallest_invalid_index].size()):
                smallest_invalid_index = idx

        if smallest_invalid_index == -1:
            raise ValueError("all clusters are valid")
        
        return smallest_invalid_index
    
    def _grow_step(self, clusters: List["WeightedFaultGraphUnionFindDecoder.Cluster"], syndrome, **kwargs):
        """
        growing chooses smallest cluster, and growing it, then merging
        """
        # TODO: different strategies here too
        smallest_invalid_index = self._get_smallest_invalid(clusters=clusters, syndrome=syndrome, **kwargs)
        clusters[smallest_invalid_index].grow()
        merged_away = []
        for idx, other in enumerate(clusters):
            if idx == smallest_invalid_index:
                continue
            if clusters[smallest_invalid_index].touches(other): # these 2 clusters are now connected
                # keep track of merged clusters
                merged_away.append(idx)
                clusters[smallest_invalid_index].merge(other) # merges other into cluster in place

        for i in merged_away[::-1]: # delete the merged clusters in reverse order so indices keep making sense
            del clusters[i]

        return clusters

    def _create_initial_clusters(self, syndrome, **kwargs):
        #initial_cluster_strategy: str: Union[None, "connected_components"], default is None
        # TODO: implement "connected_components" initial cluster strategy
        do_initially_add_connecting_faults = kwargs.get("do_initially_add_connecting_faults", False)
        do_initially_add_surrounded_faults = kwargs.get("do_initially_add_connecting_faults", False)

        clusters = []

        m, n = self.fault_graph.shape

        fault_weights = self.prime_weights.copy() # weights get consumed in a decoding

        # create cluster for each non-trivial check node
        for check in np.where(syndrome)[0]:
            checks_mask = np.zeros(m, dtype=np.bool_)
            faults_mask = np.zeros(n, dtype=np.bool_)
            checks_mask[check] = True
            cluster = WeightedFaultGraphUnionFindDecoder.Cluster(checks_mask=checks_mask,
                                                                 faults_mask=faults_mask,
                                                                 biadjacency_matrix=self.fault_graph,
                                                                 fault_weights=fault_weights)
            clusters.append(cluster)

        if do_initially_add_connecting_faults:
            clusters, _ = self._add_connecting_faults(clusters=clusters)
        elif do_initially_add_surrounded_faults:
            clusters, _ = self._add_surrounded_faults(clusters=clusters)
        
        return clusters

    def _add_surrounded_faults(self, clusters: List["WeightedFaultGraphUnionFindDecoder.Cluster"]):
        raise NotImplementedError

    def _add_connecting_faults(self, clusters: List["WeightedFaultGraphUnionFindDecoder.Cluster"]):
        """ 
        Will connect and merge cluster based on rule:
        if a fault node is not in any cluster but connected to at least two different invalid clusters
        or a fault node is a surface node of a cluster and connected to at least one different cluster
        add all neighbour checks of that fault node to the clusters and merge them accordingly.
        This does affect fault surface clusters.
        This does affect valid clusters (maybe add option to make sure it doesn't)
        maybe add different options
        """

        invalid_clusters_to_faults_connectivity = np.vstack([cluster.reachable_faults_mask for cluster in clusters if not cluster.is_valid()])
        connecting_faults_mask = invalid_clusters_to_faults_connectivity.sum(axis=0) >= 2
        connecting_faults_indices = np.where(connecting_faults_mask)[0]

        m, n = self.fault_graph.shape

        for fault_idx in connecting_faults_indices:
            # find neighbour checks of a connecting fault, cannot be done with invalid_clusters_to_faults_connectivity
            # because we also want checks in valid clusters
            neighbour_checks_mask = self.fault_graph[:, fault_idx].astype(np.bool_)
            this_fault_mask = np.zeros(n, dtype=np.bool_)
            new_cluster = WeightedFaultGraphUnionFindDecoder.Cluster(checks_mask=neighbour_checks_mask,
                                                                     faults_mask=this_fault_mask,
                                                                     biadjacency_matrix=self.fault_graph,
                                                                     fault_weights=clusters[0].fault_weights)
            
            #clusters[smallest_invalid_index].grow()
            clusters.append(new_cluster)
            merged_away = []
            for idx, other in enumerate(clusters[:-1]):
                if new_cluster.touches(other): # these 2 clusters are now connected
                    # keep track of merged clusters
                    merged_away.append(idx)
                    new_cluster.merge(other) # merges other into cluster in place

            for i in merged_away[::-1]: # delete the merged clusters in reverse order so indices keep making sense
                del clusters[i]

            return clusters

        return clusters, len(connecting_faults_indices) > 0

    def _find_correction(self, clusters: List["WeightedFaultGraphUnionFindDecoder.Cluster"], syndrome, cluster_decoding):
        decoded_error = np.zeros(self.fault_graph.shape[1], dtype=int)
        for cluster in clusters:
            if cluster_decoding == 'bposd':
                decoded_error |= cluster.decode_cluster_bposd(syndrome)
            else:
                decoded_error |= cluster.internal_error
        return decoded_error

class FaultGraphUnionFindDecoderWrapper:
    def __init__(self, code_circuit) -> None:
        self.code = code_circuit
        self.fault_graph = self.get_foliated_tanner(code_circuit.T, code_circuit.basis)
        self.uf_decoder = FaultGraphUnionFindDecoder(self.fault_graph)

    def process(self, output_string: str):
        pass

class Cluster:
        def __init__(self, checks_interior, faults_interior, cluster_surface, cluster_surface_type, biadjacency_matrix=None) -> None:
            self.checks_mask = checks_interior
            self.faults_mask = faults_interior
            self.cluster_surface_mask = cluster_surface
            self.cluster_surface_type = cluster_surface_type
            self.valid = None
            self.internal_error = None
            self.biadjacency_matrix = biadjacency_matrix # for stuff like growing, finding true interior etc.
            self.interior_faults_mask: np.ndarray = None
            self.internal_dof = np.nan
            self.minimized_internal = None

        @property
        def checks(self) -> np.ndarray:
            return np.where(self.checks_mask)[0]
        
        @property
        def faults(self) -> np.ndarray:
            return np.where(self.faults_mask)[0]

        def get_interior_faults_mask(self, biadjacency_matrix: np.ndarray = None) -> np.ndarray:
            if biadjacency_matrix is None:
                biadjacency_matrix = self.biadjacency_matrix
                if biadjacency_matrix is None:
                    raise ValueError("please provide the relevant biadjacency matrix")
            non_interior_faults_mask = biadjacency_matrix[~self.checks_mask].any(axis=0) # mask for all fault nodes in the graph that touch a check outside cluster
            interior_faults = ~non_interior_faults_mask & self.faults_mask # the & is only necessary if there are faults not connected to any check so never but whatever
            return interior_faults

        def contains_check(self, check_idx) -> bool:
            return self.checks_mask[check_idx]
        
        def contains_fault(self, fault_idx) -> bool:
            return self.faults_mask[fault_idx]

        def touches(self, other: "Cluster") -> bool:
            return self.cluster_surface_type == other.cluster_surface_type and np.any(self.cluster_surface_mask & other.cluster_surface_mask)
        
        def overlaps(self, other: "Cluster") -> bool:
            # not really used, just for completeness
            return np.any(self.checks_mask | other.checks_mask) or np.any(self.faults_mask | other.faults_mask)

        def merge(self, other: "Cluster") -> None:
            """ merging happens in place. This assumes that touches is true """
            self.checks_mask |= other.checks_mask
            self.faults_mask |= other.faults_mask
            self.cluster_surface_mask |= other.cluster_surface_mask
            # self.cluster_surface_type not affected
            self.valid = None
            self.internal_error = None

        def num_checks(self):
            return self.checks_mask.sum()
        
        def num_faults(self):
            return self.faults_mask.sum()
        
        def size(self):
            return self.num_checks() + self.num_faults()
        
        def shape(self):
            return self.num_checks(), self.num_faults()
        
        def is_valid(self, syndrome=None, biadjacency_matrix=None, **kwargs) -> bool:

            if syndrome is None:
                return self.valid # if no syndrome provided return self.valid independent of whether computed or not

            # if self.valid is set and we use default biadjacency_matrix, assume validity has not changed
            if self.valid is not None and biadjacency_matrix is None:
                return self.valid

            if biadjacency_matrix is None:
                biadjacency_matrix = self.biadjacency_matrix
                if biadjacency_matrix is None:
                    raise ValueError("please provide the relevant biadjacency matrix")
            
            # get interior fault nodes
            self.interior_faults_mask = self.get_interior_faults_mask(biadjacency_matrix)

            # system of equation extraction
            a = biadjacency_matrix[self.checks_mask][:,self.interior_faults_mask]
            b = syndrome[self.checks_mask]

            # cluster is valid iff exists x s.t. ax = b
            solvable, x, stats = solve(a, b,
                                       minimize_weight = kwargs.get("minimize_weight", True),
                                       max_dof = kwargs.get("max_dof", 9),
                                       stats = True,
                                       lse_solver=kwargs.get("lse_solver"))
            self.internal_dof = stats['dof']
            self.minimized_internal = stats['minimized']
 
            if not solvable:
                self.valid = False
                return False
            
            self.valid = True

            # x is a mask of indices of error with respect to interior data
            # convert it to mask with respect to all fault nodes
            self.internal_error = np.zeros_like(self.faults_mask)
            self.internal_error[self.interior_faults_mask] = x

            return True
        
        def decode_cluster_bposd(self, syndrome):
            from ldpc import bposd_decoder
            max_iter=1024
            # get interior fault nodes
            self.interior_faults_mask = self.get_interior_faults_mask(self.biadjacency_matrix)

            # system of equation extraction
            a = self.biadjacency_matrix[self.checks_mask][:,self.interior_faults_mask]
            b = syndrome[self.checks_mask]

            osd_order= 0
            while True:
                try:
                    bpd=bposd_decoder(
                        a ,#the parity check matrix
                        error_rate=0.01,# dummy error rate
                        channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
                        max_iter=max_iter, #the maximum number of iterations for BP)
                        bp_method="ms",
                        ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
                        osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                        osd_order=osd_order #the osd search depth
                        )
                    break
                except ValueError as e:
                    osd_order -= 1

            #return self.internal_error
            x = bpd.decode(b)

            # x is a mask of indices of error with respect to interior data
            # convert it to mask with respect to all fault nodes
            internal_error = np.zeros_like(self.faults_mask)
            internal_error[self.interior_faults_mask] = x
            return internal_error

        def neighbour_faults_mask(self) -> np.ndarray:
            """ if check surface return next faults that would be added with grow
            if fault surface, then return just the surface"""
            if self.cluster_surface_type == 'c':
                return self.biadjacency_matrix[self.cluster_surface_mask].any(axis=0) &~self.faults_mask
            else:
                return self.cluster_surface_mask
            
        def neighbour_checks_mask(self) -> np.ndarray:
            if self.cluster_surface_type == 'f':
                return self.biadjacency_matrix[:, self.cluster_surface_mask].any(axis=1) &~self.checks_mask
            else:
                return self.cluster_surface_mask

        def grow(self) -> None:            
            if self.cluster_surface_type == 'c':
                #self.cluster_surface_mask = np.any(biadjacency_matrix[self.cluster_surface_mask], axis=0) &~self.faults_mask
                self.cluster_surface_mask = self.neighbour_faults_mask()
                self.faults_mask |= self.cluster_surface_mask
                self.cluster_surface_type = 'f'

            elif self.cluster_surface_type == 'f':
                #self.cluster_surface_mask = np.any(biadjacency_matrix[:, self.cluster_surface_mask], axis=1) &~self.checks_mask
                self.cluster_surface_mask = self.neighbour_checks_mask()
                self.checks_mask |= self.cluster_surface_mask
                self.cluster_surface_type = 'c'

            self.valid = None # need to reset

        def __repr__(self) -> str:
            string = f'checks: {list(self.checks)}\n'
            string += f'faults: {list(self.faults)}\n'
            if self.valid:
                string += f'internal error: {list(np.where(self.internal_error)[0])}\n'
            else:
                string += f'Not valid'
            return string
        
        def snapshot(self) -> Dict:
            return {'valid': self.valid,
                 'checks': list(self.checks),
                 'faults': list(self.faults),
                 'size': self.size(),
                 'num_checks': self.num_checks(),
                 'num_faults': self.num_faults(),
                 'num_internal_faults': self.interior_faults_mask.sum(),
                 'dof': self.internal_dof,
                 'minimized': self.minimized_internal}

class FaultGraphUnionFindDecoder:
    def __init__(self, fault_graph: np.ndarray) -> None:
        """ Initializes the Tanner graph (simply with reduced adjacency matrix) """
        self.fault_graph = fault_graph

    def decode(self, syndrome: np.ndarray, **kwargs):
        """ Main entry point for decoding, takes syndrome and initialized clusters. 
        kwargs can be (but for uniform growth, ignore most of them...):
            predecode: bool, default is False. 
            do_initially_add_connecting_faults: bool, default is False
            do_initially_add_surrounded_faults: bool, default is False
            do_add_surrounded_faults: bool, default is False
            do_add_connecting_faults: bool, default is False
            grow_strategy: str = Union[None, "individual"], default is None
            lse_solver: str = Union[None, "python", "numba", "c++"], default is None (which later resolves to numba)
            minimize_weight: bool, default is False
            max_dof: int, default is None (which later resolves to 9)
        """
        m, n = self.fault_graph.shape
        l = len(syndrome)
        cluster_history = [] # to record how clusters grow

        if l != m:
            raise ValueError(f"syndrome of length {l} is incompatible with number of checks {m} in fault graph")
        
        clusters = self._create_initial_clusters(syndrome, **kwargs)
        all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)

        while not all_valid:
            clusters = self._grow_step(clusters=clusters, syndrome=syndrome, **kwargs)
            all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)

        decoded_error = self._find_correction(clusters, syndrome, kwargs.get('cluster_decoding'))

        return decoded_error, cluster_history

    def _check_clusters(self, clusters: List[Cluster], syndrome, cluster_history=None, **kwargs):
        """ 
        Goes through all clusters and checks if they are valid given the syndrome.
        If provided, will also append a snapshot of the current clusters to cluster_history.
        Returns all_valid: a bool indicating if all clusters are valid
        """
        clusters_snapshot = {'num_cluster': len(clusters), 'clusters': []}
        all_valid = True
        for idx, cluster in enumerate(clusters):
            cluster_is_valid = cluster.is_valid(syndrome, **kwargs)
            all_valid &= cluster_is_valid

            clusters_snapshot['clusters'].append(cluster.snapshot())

        if cluster_history is not None:
            cluster_history.append(clusters_snapshot)

        return all_valid
    
    def _get_smallest_invalid(self, clusters: List[Cluster], syndrome, **kwargs):
        """ returns the index of the smallest invalid cluster """
        # find smallest invalid cluster
        smallest_invalid_index = -1
        for idx, cluster in enumerate(clusters):
            cluster_is_valid = cluster.is_valid(syndrome, **kwargs)

            if not cluster_is_valid and (smallest_invalid_index == -1 or cluster.size() < clusters[smallest_invalid_index].size()):
                smallest_invalid_index = idx

        if smallest_invalid_index == -1:
            raise ValueError("all clusters are valid")
        
        return smallest_invalid_index
    
    def _grow_step(self, clusters: List[Cluster], syndrome, **kwargs):
        """ grows clusters depending on the rule specified, handles growing related merging,
         now according to smallest first """
        grow_strategy = kwargs.get("grow_strategy", None)
        if grow_strategy == 'individual':
            smallest_invalid_index = self._get_smallest_invalid(clusters=clusters, syndrome=syndrome, **kwargs)
            clusters[smallest_invalid_index].grow()
            merged_away = []
            for idx, other in enumerate(clusters):
                if idx == smallest_invalid_index:
                    continue
                if clusters[smallest_invalid_index].touches(other): # these 2 clusters are now connected
                    # keep track of merged clusters
                    merged_away.append(idx)
                    clusters[smallest_invalid_index].merge(other) # merges other into cluster in place

            for i in merged_away[::-1]: # delete the merged clusters in reverse order so indices keep making sense
                del clusters[i]

            return clusters
        
        # else grow all invalid ones
        grown_clusters = []
        for cluster in clusters:        
            if not cluster.is_valid(syndrome, **kwargs):
                # grow cluster if invalid
                cluster.grow()

            # merge directly
            merged_away = []
            for i, other in enumerate(grown_clusters):
                if cluster.touches(other): # these 2 clusters are now connected
                    # keep track of merged
                    merged_away.append(i)
                    cluster.merge(other) # merges other into cluster in place

            for i in merged_away[::-1]: # delete the merged clusters in reverse order so indices keep making sense
                del grown_clusters[i]

            grown_clusters.append(cluster)
        
        clusters = grown_clusters
        return clusters

    def _create_initial_clusters(self, syndrome, **kwargs):
        do_initially_add_connecting_faults = kwargs.get("do_initially_add_connecting_faults", False)

        clusters = []

        m, n = self.fault_graph.shape

        # create cluster for each non-trivial check node
        for check in np.where(syndrome)[0]:
            checks = np.zeros(m, dtype=np.bool_)
            data = np.zeros(n, dtype=np.bool_)
            cluster_surface = np.zeros(m, dtype=np.bool_)
            cluster_surface[check] = True
            checks |= cluster_surface
            cluster_surface_type = 'c'
            cluster = Cluster(checks, data, cluster_surface, cluster_surface_type, self.fault_graph)
            clusters.append(cluster)

        if do_initially_add_connecting_faults:
            clusters, _ = self._add_connecting_faults(clusters=clusters)
        
        return clusters

    def _add_connecting_faults(self, clusters: List[Cluster]):
        """ 
        Will connect and merge cluster based on rule:
        if a fault node is not in any cluster but connected to at least two different invalid clusters
        or a fault node is a surface node of a cluster and connected to at least one different cluster
        add all neighbour checks of that fault node to the clusters and merge them accordingly.
        This does affect fault surface clusters.
        This does affect valid clusters (maybe add option to make sure it doesn't)
        not the same as _form_connecting_clusters_from_checks, which will add faults that have twice the same cluster as neighbour
        """

        invalid_clusters_to_faults_connectivity = np.vstack([cluster.neighbour_faults_mask() for cluster in clusters if not cluster.is_valid()])
        connecting_faults_mask = invalid_clusters_to_faults_connectivity.sum(axis=0) >= 2
        # This explicitely includes fault nodes that are part of a fault surface cluster, I am not sure if it should
        # TODO: figure out if we should filter or not, or if we get rid of fault surface clusters altogether.
        # For now, accept it
        connecting_faults_indices = np.where(connecting_faults_mask)[0]

        for fault_idx in connecting_faults_indices:
            # find neighbour checks of a connecting fault, cannot be done with invalid_clusters_to_faults_connectivity
            # because we also want checks in valid clusters
            neighbour_checks_mask = self.fault_graph[:, fault_idx].astype(np.bool_)
            neighbour_checks_indices = np.where(neighbour_checks_mask)[0]

            merged_cluster_indices = []
            # find first cluster this fault is connected to
            for idxA, neighbour_check_index in product(range(len(clusters)), neighbour_checks_indices):
                if clusters[idxA].contains_check(neighbour_check_index): break
            # find all other clusters this fault is connected to and merge them
            for idxB in range(idxA + 1, len(clusters)):
                for neighbour_check_index in neighbour_checks_indices:
                    if clusters[idxB].contains_check(neighbour_check_index):
                        clusters[idxA].merge(clusters[idxB])
                        merged_cluster_indices.append(idxB)
                        break

            # add connecting fault to merged cluster
            clusters[idxA].faults_mask[fault_idx] = True

            # add all neighbour checks to the merged cluster too
            clusters[idxA].checks_mask |= neighbour_checks_mask
            clusters[idxA].cluster_surface_mask |= neighbour_checks_mask

            # delete the merged away clusters
            for cluster_index in merged_cluster_indices[::-1]:
                del clusters[cluster_index]

        return clusters, len(connecting_faults_indices) > 0

    def _find_correction(self, clusters: List[Cluster], syndrome, cluster_decoding):
        decoded_error = np.zeros(self.fault_graph.shape[1], dtype=int)
        for cluster in clusters:
            if cluster_decoding == 'bposd':
                decoded_error |= cluster.decode_cluster_bposd(syndrome)
            else:
                decoded_error |= cluster.internal_error
        return decoded_error
