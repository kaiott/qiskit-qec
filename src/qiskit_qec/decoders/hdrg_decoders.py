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


class ClusteringDecoder(ABC):
    """
    Generic base class for clustering decoders.
    """

    def __init__(
        self,
        code_circuit,
        decoding_graph: DecodingGraph = None,
    ):
        self.code = code_circuit

        if hasattr(self.code, "code_index"):
            self.code_index = self.code.code_index
        else:
            self.code_index = {j: j for j in range(self.code.n)}

        if decoding_graph:
            self.decoding_graph = decoding_graph
        else:
            self.decoding_graph = DecodingGraph(self.code)

    def get_corrections(self, string, clusters):
        """
        Turn a set of neutral clusters into corrections.

        Args:
            string (str): Output string of the code
            clusters (dict): Dictionary with the indices of the given node
            as keys and an integer specifying their cluster as the corresponding
            value.
        Returns:
            corrected_logicals (list): A list of integers that are 0 or 1.
        These are the corrected values of the final transversal
        measurement, in the same form as given by the code's `string2raw_logicals`.
        """

        # get the list of bulk nodes for each cluster
        cluster_nodes = {c: [] for c in clusters.values()}
        for n, c in clusters.items():
            node = self.decoding_graph.graph[n]
            if not node.is_logical:
                cluster_nodes[c].append(node)

        # get the list of required logicals for each cluster
        cluster_logicals = {}
        for c, nodes in cluster_nodes.items():
            _, logical_nodes, _ = self.code.check_nodes(nodes, minimal=True)
            log_indexes = [node.index for node in logical_nodes]
            cluster_logicals[c] = log_indexes

        # get the net effect on each logical
        net_logicals = {node.index: 0 for node in self.decoding_graph.logical_nodes}
        for c, log_indexes in cluster_logicals.items():
            for log_index in log_indexes:
                net_logicals[log_index] += 1
        for log_index, num in net_logicals.items():
            net_logicals[log_index] = num % 2

        corrected_logicals = self.code.string2raw_logicals(string)
        for log_index, log_value in enumerate(corrected_logicals):
            corrected_logicals[log_index] = (net_logicals[log_index] + int(log_value)) % 2

        return corrected_logicals


class BravyiHaahDecoder(ClusteringDecoder):
    """Decoder based on finding connected components within the decoding graph."""

    def _cluster(self, ns, dist_max):
        """
        Finds connected components in the given nodes, for nodes connected by at most the given distance
        in the given decoding graph.
        """

        # calculate distance for the graph
        dg = self.decoding_graph.graph
        distance = distance_matrix(dg)

        # create empty `DecodingGraph`
        cluster_graph = DecodingGraph(None)
        cg = cluster_graph.graph
        # add all the given nodes to cg
        d2c = {}
        c2g = {}
        for n in ns:
            node = dg.nodes()[n]
            d2c[n] = cg.add_node(node)
            c2g[d2c[n]] = n
        # add an edge between a pair of the given nodes if their distance is small enough
        for n0 in ns:
            for n1 in ns:
                if n0 < n1:
                    dist = distance[n0, n1]
                    if dist <= dist_max:
                        cg.add_edge(d2c[n0], d2c[n1], {"distance": dist})
        # find the connected components of cg
        con_comps = connected_components(cg)

        # use these to define clusters
        clusters = {}
        con_comp_dict = {}
        for c, con_comp in enumerate(con_comps):
            con_comp_dict[c] = []

            # check the neutrality of each connected component
            con_nodes = [cg[n] for n in con_comp]
            neutral, logicals, num_errors = self.code.check_nodes(
                con_nodes, ignore_extra_logical=True
            )

            # it's fully neutral if no extra logicals are needed
            # and if the error num is less than the max dist
            fully_neutral = neutral and logicals == []
            if num_errors:
                fully_neutral = fully_neutral and num_errors < dist_max

            # if a cluster is neutral, all nodes are labelled with c
            # otherwise, it gets a None
            for n in con_comp:
                if fully_neutral:
                    clusters[c2g[n]] = c
                else:
                    clusters[c2g[n]] = None
                con_comp_dict[c].append(c2g[n])

        return clusters, con_comp_dict

    def cluster(self, nodes):
        """

        Args:
            nodes (list): List of nodes, of the type produced by `string2nodes`.
        Returns:
            final_clusters (dict): Dictionary with the indices of the given node
            as keys and an integer specifying their cluster as the corresponding
            value.
        """

        # get indices for nodes and logical nodes
        dg = self.decoding_graph.graph
        ns = set(dg.nodes().index(node) for node in nodes)
        lns = set(dg.nodes().index(node) for node in self.decoding_graph.logical_nodes)

        dist_max = 0
        final_clusters = {}
        con_comps = []
        clusterss = []
        while ns and dist_max <= self.code.d:
            dist_max += 1
            # add logical nodes to unpaired nodes
            ns = set(ns).union(lns)

            # cluster nodes and contract decoding graph given the current distance
            clusters, con_comp = self._cluster(ns, dist_max)
            # record the clustered and unclustered nodes
            ns = []
            for n, c in clusters.items():
                if c is not None:
                    final_clusters[n] = c
                else:
                    if not dg[n].is_logical:
                        ns.append(n)
            con_comps.append(con_comp)
            clusterss.append(clusters)

        return final_clusters

    def process(self, string, predecoder=None):
        """
        Process an output string and return corrected final outcomes.

        Args:
            string (str): Output string of the code.
            predecoder (callable): Function that takes in and returns
            a list of nodes. Used to do preprocessing on the nodes
            corresponding to the input string.

        Returns:
            corrected_logicals (list): A list of integers that are 0 or 1.
        These are the corrected values of the final transversal
        measurement, in the same form as given by the code's `string2raw_logicals`.
        """

        # turn string into nodes and cluster
        nodes = self.code.string2nodes(string, all_logicals=True)
        # apply predecoder if one is given
        if predecoder:
            nodes = predecoder(nodes)
        # then cluster
        clusters = self.cluster(nodes)

        return self.get_corrections(string, clusters)


@dataclass
class SpanningForest:
    """
    Spanning forest for the peeling decoder.
    """

    vertices: Dict[int, List[int]]
    edges: List[int]


@dataclass
class BoundaryEdge:
    """
    Boundary edge for the boundary of a UnionFindDecoderCluster.
    """

    index: int
    cluster_vertex: int
    neighbour_vertex: int
    data: DecodingGraphEdge

    def reverse(self):
        """
        Returns a reversed version of the boundary edge (cluster and neighbour vertex flipped)
        """
        return BoundaryEdge(
            index=self.index,
            cluster_vertex=self.neighbour_vertex,
            neighbour_vertex=self.cluster_vertex,
            data=self.data,
        )


@dataclass
class UnionFindDecoderCluster:
    """
    Cluster for the UnionFindDecoder
    """

    boundary: List[BoundaryEdge]
    atypical_nodes: Set[int]
    boundary_nodes: Set[int]
    nodes: Set[int]
    fully_grown_edges: Set[int]
    size: int


@dataclass
class FusionEntry:
    """
    Entry for the fusion list between the growing and merging of the union find decoder.
    """

    u: int
    v: int
    connecting_edge: BoundaryEdge


class UnionFindDecoder(ClusteringDecoder):
    """
    Decoder based on growing clusters around syndrome errors to
    "convert" them into erasure errors, which can be corrected easily,
    by the peeling decoder for compatible codes or by the standard HDRG
    method in general.

    TODO: Add weights to edges of graph according to Huang et al (see. arXiv:2004.04693, section III)

    See arXiv:1709.06218v3 for more details.
    """

    def __init__(self, code, decoding_graph: DecodingGraph = None, use_peeling=True) -> None:
        super().__init__(code, decoding_graph=decoding_graph)
        self.graph = deepcopy(self.decoding_graph.graph)
        self.clusters: Dict[int, UnionFindDecoderCluster] = {}
        self.odd_cluster_roots: List[int] = []
        self.use_peeling = use_peeling
        self._clusters4peeling = []

    def process(self, string: str, predecoder=None):
        """
        Process an output string and return corrected final outcomes.

        Args:
            string (str): Output string of the code.
            predecoder (callable): Function that takes in and returns
            a list of nodes. Used to do preprocessing on the nodes
            corresponding to the input string.
        Returns:
            corrected_logicals (list): A list of integers that are 0 or 1.
        These are the corrected values of the final logical measurement.
        """

        if self.use_peeling:
            self.graph = deepcopy(self.decoding_graph.graph)
            highlighted_nodes = self.code.string2nodes(string, all_logicals=True)
            if predecoder:
                highlighted_nodes = predecoder(highlighted_nodes)

            # call cluster to do the clustering, but actually use the peeling form
            self.cluster(highlighted_nodes)
            clusters = self._clusters4peeling

            # determine the net logical z
            measured_logicals = {}
            for node in self.decoding_graph.logical_nodes:
                measured_logicals[node.index] = node.qubits
            net_z_logicals = {tuple(z_logical): 0 for z_logical in measured_logicals.values()}
            for cluster_nodes, _ in clusters:
                erasure = self.graph.subgraph(cluster_nodes)
                flipped_qubits = self.peeling(erasure)
                for qubit_to_be_corrected in flipped_qubits:
                    for z_logical in net_z_logicals:
                        if qubit_to_be_corrected in z_logical:
                            net_z_logicals[z_logical] += 1
            for z_logical, num in net_z_logicals.items():
                net_z_logicals[z_logical] = num % 2

            # apply this to the raw readout
            corrected_z_logicals = []
            raw_logicals = self.code.string2raw_logicals(string)
            for j, z_logical in measured_logicals.items():
                raw_logical = int(raw_logicals[j])
                corrected_logical = (raw_logical + net_z_logicals[tuple(z_logical)]) % 2
                corrected_z_logicals.append(corrected_logical)
            return corrected_z_logicals
        else:
            # turn string into nodes and cluster
            nodes = self.code.string2nodes(string, all_logicals=True)
            if predecoder:
                nodes = predecoder(nodes)
            clusters = self.cluster(nodes)
            return self.get_corrections(string, clusters)

    def cluster(self, nodes: List):
        """
        Create clusters using the union-find algorithm.

        Args:
            nodes (List): List of non-typical nodes in the syndrome graph,
            of the type produced by `string2nodes`.

        Returns:
            clusters (dict): Dictionary with the indices of
            the given node as keys and an integer specifying their cluster as the corresponding
            value.
        """
        node_indices = [self.decoding_graph.node_index(node) for node in nodes]
        for node_index, _ in enumerate(self.graph.nodes()):
            self.graph[node_index].properties["syndrome"] = node_index in node_indices
            self.graph[node_index].properties["root"] = node_index

        for edge in self.graph.edges():
            edge.properties["growth"] = 0
            edge.properties["fully_grown"] = False

        self.clusters: Dict[int, UnionFindDecoderCluster] = {}
        self.odd_cluster_roots = []
        for node_index in node_indices:
            self._create_new_cluster(node_index)

        j = 0
        while self.odd_cluster_roots and j < 2 * self.code.d * (self.code.T + 1):
            self._grow_and_merge_clusters()
            j += 1

        # compile info into standard clusters dict
        clusters = {}
        for c, cluster in self.clusters.items():
            # determine which nodes exactly are in the neutral cluster
            neutral_nodes = list(cluster.atypical_nodes | cluster.boundary_nodes)
            # put them in the required dict
            for n in neutral_nodes:
                clusters[n] = c

        # also compile into form required for peeling
        self._clusters4peeling = []
        for _, cluster in self.clusters.items():
            if not cluster.atypical_nodes:
                continue
            self._clusters4peeling.append(
                (list(cluster.nodes), list(cluster.atypical_nodes | cluster.boundary_nodes))
            )

        return clusters

    def find(self, u: int) -> int:
        """
        Find() function as described in the paper that returns the root
        of the cluster of a node, including path compression.

        Args:
            u (int): The index of the node in the decoding graph.

        Returns:
            root (int): The root of the cluster of node u.
        """
        if self.graph[u].properties["root"] == u:
            return self.graph[u].properties["root"]

        self.graph[u].properties["root"] = self.find(self.graph[u].properties["root"])
        return self.graph[u].properties["root"]

    def _create_new_cluster(self, node_index):
        node = self.graph[node_index]
        if not node.is_logical:
            self.odd_cluster_roots.insert(0, node_index)
        boundary_edges = []
        for edge_index, neighbour, data in self.neighbouring_edges(node_index):
            boundary_edges.append(BoundaryEdge(edge_index, node_index, neighbour, data))
        self.clusters[node_index] = UnionFindDecoderCluster(
            boundary=boundary_edges,
            fully_grown_edges=set(),
            atypical_nodes=set([node_index]) if not node.is_logical else set([]),
            boundary_nodes=set([node_index]) if node.is_logical else set([]),
            nodes=set([node_index]),
            size=1,
        )

    def _grow_and_merge_clusters(self) -> Set[int]:
        fusion_edge_list = self._grow_clusters()
        return self._merge_clusters(fusion_edge_list)

    def _grow_clusters(self) -> List[FusionEntry]:
        """
        Grow every "odd" cluster by half an edge.

        Returns:
            fusion_edge_list (List[FusionEntry]): List of edges that connect two
            clusters that will be merged in the next step.
        """
        fusion_edge_list: List[FusionEntry] = []
        for root in self.odd_cluster_roots:
            cluster = self.clusters[root]
            for edge in cluster.boundary:
                edge.data.properties["growth"] += 0.5
                if (
                    edge.data.properties["growth"] >= edge.data.weight
                    and not edge.data.properties["fully_grown"]
                ):
                    neighbour_root = self.find(edge.neighbour_vertex)
                    if not neighbour_root in self.clusters:
                        boundary_edges = []
                        for edge_index, neighbour_neighbour, data in self.neighbouring_edges(
                            edge.neighbour_vertex
                        ):
                            boundary_edges.append(
                                BoundaryEdge(
                                    edge_index, edge.neighbour_vertex, neighbour_neighbour, data
                                )
                            )
                        self.graph[edge.neighbour_vertex].properties["root"] = edge.neighbour_vertex
                        self.clusters[edge.neighbour_vertex] = UnionFindDecoderCluster(
                            boundary=boundary_edges,
                            fully_grown_edges=set(),
                            atypical_nodes=set(),
                            boundary_nodes=set([edge.neighbour_vertex])
                            if self.graph[edge.neighbour_vertex].is_logical
                            else set([]),
                            nodes=set([edge.neighbour_vertex]),
                            size=1,
                        )
                    fusion_entry = FusionEntry(
                        u=edge.cluster_vertex, v=edge.neighbour_vertex, connecting_edge=edge
                    )
                    fusion_edge_list.append(fusion_entry)
        return fusion_edge_list

    def _merge_clusters(self, fusion_edge_list: List[FusionEntry]):
        """
        Merges the clusters based on the fusion_edge_list computed in _grow_clusters().
        Updates the odd_clusters list by recomputing the neutrality of the newly merged clusters.

        Args:
            fusion_edge_list (List[FusionEntry]): List of edges that connect two
            clusters that was computed in _grow_clusters().
        Returns:
            new_neutral_cluster_roots (List[int]): List of roots of newly neutral clusters
        """
        new_neutral_clusters = []
        for entry in fusion_edge_list:
            root_u, root_v = self.find(entry.u), self.find(entry.v)
            if root_u == root_v:
                continue
            new_root = root_v if self.clusters[root_v].size > self.clusters[root_u].size else root_u
            root_to_update = root_v if new_root == root_u else root_u

            if new_root in new_neutral_clusters or root_to_update in new_neutral_clusters:
                continue

            cluster = self.clusters[new_root]
            other_cluster = self.clusters.pop(root_to_update)

            entry.connecting_edge.data.properties["growth"] = 0
            entry.connecting_edge.data.properties["fully_grown"] = True
            cluster.fully_grown_edges.add(entry.connecting_edge.index)

            # Merge boundaries
            cluster.boundary += other_cluster.boundary
            cluster.boundary.remove(entry.connecting_edge)
            cluster.boundary.remove(entry.connecting_edge.reverse())

            cluster.nodes |= other_cluster.nodes
            cluster.atypical_nodes |= other_cluster.atypical_nodes
            cluster.boundary_nodes |= other_cluster.boundary_nodes
            cluster.fully_grown_edges |= other_cluster.fully_grown_edges
            cluster.size += other_cluster.size

            # update odd_cluster_roots
            if self.code.is_cluster_neutral(
                [self.graph[node] for node in cluster.atypical_nodes]
            ) or self.code.is_cluster_neutral(
                [
                    self.graph[node]
                    for node in cluster.atypical_nodes
                    | (set(list(cluster.boundary_nodes)[:1]) if cluster.boundary_nodes else set())
                ]
            ):
                if new_root in self.odd_cluster_roots:
                    self.odd_cluster_roots.remove(new_root)
                    new_neutral_clusters.append(new_root)
            else:
                if not new_root in self.odd_cluster_roots:
                    self.odd_cluster_roots.append(new_root)

            if root_to_update in self.odd_cluster_roots:
                self.odd_cluster_roots.remove(root_to_update)
            self.graph[root_to_update].properties["root"] = new_root
            self.odd_cluster_roots = sorted(
                self.odd_cluster_roots, key=lambda c: self.clusters[c].size
            )

    def peeling(self, erasure: PyGraph) -> List[int]:
        """ "
        Runs the peeling decoder on the erasure provided.
        Assumes that the erasure is one connected component, if not it will run in an
        infinite loop in the tree construction.
        It works by first producing a spanning forest of the erasure and then
        going backwards through the edges of the tree computing the error based on the syndrome.
        Based on arXiv:1703.01517.

        Args:
            erasure (PyGraph): subgraph of the syndrome graph that represents the erasure.

        Returns:
            errors (List[int]): List of qubit indices on which Pauli errors occurred.
        """
        tree = SpanningForest(vertices={}, edges=[])

        # Construct spanning forest
        # Pick starting vertex
        for vertex in erasure.node_indices():
            if erasure[vertex].is_logical and erasure[vertex].properties["syndrome"]:
                tree.vertices[vertex] = []
                break

        if not tree.vertices:
            for vertex in erasure.node_indices():
                if erasure[vertex].properties["syndrome"]:
                    tree.vertices[vertex] = []
                    break

        # Expand forest |V| - 1 times, constructing it
        while len(tree.edges) < len(erasure.nodes()) - 1:
            vertices = copy(tree.vertices)
            for node in vertices.keys():
                if len(tree.edges) >= len(erasure.nodes()) - 1:
                    break
                for edge, (_, neighbour, _) in dict(erasure.incident_edge_index_map(node)).items():
                    if not neighbour in tree.vertices.keys():
                        tree.edges.append(edge)
                        tree.vertices[neighbour] = []
                        tree.vertices[node].append(edge)
                        break

        edges = set()
        for edge in tree.edges[::-1]:
            endpoints = erasure.get_edge_endpoints_by_index(edge)
            pendant_vertex = endpoints[0] if not tree.vertices[endpoints[0]] else endpoints[1]
            tree_vertex = endpoints[0] if pendant_vertex == endpoints[1] else endpoints[1]
            tree.vertices[tree_vertex].remove(edge)
            if erasure[pendant_vertex].properties["syndrome"]:
                edges.add(edge)
                erasure[tree_vertex].properties["syndrome"] = not erasure[tree_vertex].properties[
                    "syndrome"
                ]
                erasure[pendant_vertex].properties["syndrome"] = False

        return [erasure.edges()[edge].qubits[0] for edge in edges if erasure.edges()[edge].qubits]

    def neighbouring_edges(self, node_index) -> List[Tuple[int, int, DecodingGraphEdge]]:
        """Returns all of the neighbouring edges of a node in the decoding graph.

        Args:
            node_index (int): The index of the node in the graph.

        Returns:
            neighbouring_edges (List[Tuple[int, int, DecodingGraphEdge]]): List of neighbouring edges

            In following format::

                {
                    index of edge in graph,
                    index of neighbour node in graph,
                    data payload of the edge
                }

        """
        return [
            (edge, neighbour, data)
            for edge, (_, neighbour, data) in dict(
                self.graph.incident_edge_index_map(node_index)
            ).items()
        ]






class FaultGraphUnionFindDecoderWrapper:
    def __init__(self, code_circuit) -> None:
        self.code = code_circuit
        self.fault_graph = self.get_foliated_tanner(code_circuit.T, code_circuit.basis)
        self.uf_decoder = FaultGraphUnionFindDecoder(self.fault_graph)

    def process(self, output_string: str):
        pass

class FaultGraphUnionFindDecoder:
    def __init__(self, fault_graph: np.ndarray) -> None:
        """ Initializes the Tanner graph (simply with reduced adjacency matrix) """
        self.fault_graph = fault_graph

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

        def touches(self, other: "TannerUnionFindOptimized.Cluster") -> bool:
            return self.cluster_surface_type == other.cluster_surface_type and np.any(self.cluster_surface_mask & other.cluster_surface_mask)
        
        def overlaps(self, other: "TannerUnionFindOptimized.Cluster") -> bool:
            # not really used, just for completeness
            return np.any(self.checks_mask | other.checks_mask) or np.any(self.faults_mask | other.faults_mask)

        def merge(self, other: "TannerUnionFindOptimized.Cluster") -> None:
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

    def decode_new(self, syndrome: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Tuple[Tuple[int,int], float, int, List[Tuple[Tuple[int,int], float, int, float, float, int, float]], float]]]:
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

        do_add_surrounded_faults = kwargs.get('do_add_surrounded_faults', False)
        do_add_connecting_faults = kwargs.get('do_add_connecting_faults', False)

        while not all_valid:
            changing = True
            while (do_add_surrounded_faults or do_add_connecting_faults) and changing and not all_valid:
                while do_add_surrounded_faults and changing and not all_valid:
                    clusters, changing = self._add_surrounded_faults(clusters=clusters)
                    if changing:
                        all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)
                if do_add_connecting_faults and not all_valid:
                    clusters, changing = self._add_connecting_faults(clusters=clusters)
                    if changing:
                        all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)

            if not all_valid:
                clusters = self._grow_step(clusters=clusters, syndrome=syndrome, **kwargs)
                all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)

        # now all clusters are valid
        # as a sideeffect the is valid also found all the errors within the clusters. Let's combine them
                    
        decoded_error = self._find_correction(clusters, syndrome, kwargs.get('cluster_decoding'))

        return decoded_error, cluster_history

    def decode(self, syndrome: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Tuple[Tuple[int,int], float, int, List[Tuple[Tuple[int,int], float, int, float, float, int, float]], float]]]:
        """ Main entry point for decoding, takes syndrome and initialized clusters. 
        kwargs can be:
            predecode: bool, default is False
            initial_cluster_strategy: str: Union[None, "connected_components"], default is None
            grow_strategy: str = Union[None, "individual"], default is None
            extra_merging_strategy: str = Union[None, "extra"], default is None
            lse_solver: str = Union[None, "python", "numba", "c++"], default is None
            minimize_weight: bool, default is False
            max_dof: int, default is 9
        """
        extra_connect_clusters = kwargs.get("extra_connect_clusters", False)
        if extra_connect_clusters:
            return self.decode_new(syndrome, **kwargs)

        m, n = self.fault_graph.shape
        l = len(syndrome)

        if l != m:
            raise ValueError(f"syndrome of length {l} is incompatible with number of checks {m} in fault graph")

        predecode = kwargs.pop("predecode", False)
        grow_strategy = kwargs.get("grow_strategy", None)
        extra_connect_clusters = kwargs.get("extra_connect_clusters", False)

        if predecode:
            pre_error, syndrome = self._predecode(syndrome=syndrome)

        initial_clusters = self._create_initial_clusters(syndrome, **kwargs)


        if extra_connect_clusters:
            decoded_error, history = self._decode_rarely_grow(initial_clusters, syndrome, **kwargs)
        elif grow_strategy == 'individual':
            decoded_error, history = self._decode_individual(initial_clusters, syndrome, **kwargs)
        else:
            decoded_error, history = self._decode_recursive(initial_clusters, syndrome, **kwargs)

        if predecode:
            decoded_error ^= pre_error

        return decoded_error, history

    def _predecode(self, syndrome):
        pre_error = self.fault_graph.T @ syndrome == 3
        pre_syndrome = self.fault_graph @ pre_error % 2 ^ syndrome
        return pre_error, pre_syndrome

    def _check_clusters(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"], syndrome, cluster_history=None, **kwargs):
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
    
    def _get_smallest_invalid(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"], syndrome, **kwargs):
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
    
    def _grow_step(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"], syndrome, **kwargs):
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

    def _decode_rarely_grow(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"], syndrome, **kwargs):
        cluster_history = [] # benchmarking
        extra_merging_strategy = kwargs.pop("extra_merging_strategy", None)
        extra_connect_clusters = kwargs.pop("extra_connect_clusters", None)

        do_extra_onec_clusters_temp_disable_variable_REMOVE_ME = False

        while True:
            # check if current clusters are all valid
            all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)
            
            if all_valid: # if so, stop
                break
            while do_extra_onec_clusters_temp_disable_variable_REMOVE_ME: # otherwise add new faults and chekcs to clusters, if so desired
                clusters, changed = self._form_connecting_clusters_from_checks(clusters=clusters)
                all_valid = self._check_clusters(clusters=clusters, syndrome=syndrome, cluster_history=cluster_history, **kwargs)
                if not changed or not all_valid:
                    break
            if all_valid:
                break
            clusters = self._grow_step(clusters=clusters, syndrome=syndrome, **kwargs)
            # validity of the grown clusters will be checked at beginning of loop again

        # now all clusters are valid
        # as a sideeffect the is valid also found all the errors within the clusters. Let's combine them
        decoded_error = np.zeros(self.fault_graph.shape[1], dtype=int)
        for cluster in clusters:
            decoded_error |= cluster.internal_error

        return decoded_error, cluster_history

    def _decode_individual(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"], syndrome, **kwargs):
        cluster_history = [] # benchmarking
        extra_merging_strategy = kwargs.pop("extra_merging_strategy", None)
        extra_connect_clusters = kwargs.pop("extra_connect_clusters", None)

        while True:
            # find smallest invalid cluster
            smallest_invalid_index = -1
            growth_step_history = {'num_cluster': len(clusters), 'clusters': []}
            for idx, cluster in enumerate(clusters):
                valid_cluster = cluster.is_valid(syndrome, **kwargs)

                growth_step_history['clusters'].append({'valid': valid_cluster,
                                                        'checks': list(cluster.checks),
                                                        'faults': list(cluster.faults),
                                                        'size': cluster.size(),
                                                        'num_checks': cluster.num_checks(),
                                                        'num_faults': cluster.num_faults(),
                                                        'num_internal_faults': cluster.interior_faults_mask.sum(),
                                                        'dof': cluster.internal_dof,
                                                        'minimized': cluster.minimized_internal})

                if not valid_cluster and (smallest_invalid_index == -1 or cluster.size() < clusters[smallest_invalid_index].size()):
                    smallest_invalid_index = idx

            cluster_history.append(growth_step_history)

            if smallest_invalid_index == -1:
                break # no invalid cluster. Exit loop

            # grow cluster
            clusters[smallest_invalid_index].grow()

            # merge directly
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

            if extra_merging_strategy == 'extra':
                clusters = self._extra_merging_strategy(clusters)
            
        # now all clusters are valid
        # as a sideeffect the is valid also found all the errors within the clusters. Let's combine them
        decoded_error = np.zeros(self.fault_graph.shape[1], dtype=int)
        for cluster in clusters:
            decoded_error |= cluster.internal_error

        return decoded_error, cluster_history

    def _decode_recursive(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"], syndrome, cluster_history=None, **kwargs):
        
        if cluster_history is None:# benchmarking
            cluster_history = []# benchmarking

        growth_step_history = {'num_cluster': len(clusters), 'clusters': []} # benchmarking

        all_valid = True
        grown_clusters = []

        for cluster in clusters:
            valid_cluster = cluster.is_valid(syndrome, **kwargs)

            growth_step_history['clusters'].append({'valid': valid_cluster,
                                                    'checks': list(cluster.checks),
                                                    'faults': list(cluster.faults),
                                                    'size': cluster.size(),
                                                    'num_checks': cluster.num_checks(),
                                                    'num_faults': cluster.num_faults(),
                                                    'num_internal_faults': cluster.interior_faults_mask.sum(),
                                                    'dof': cluster.internal_dof,
                                                    'minimized': cluster.minimized_internal})
            
            
            if not valid_cluster:
                all_valid = False

                # grow cluster
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
        cluster_history.append(growth_step_history)

        if kwargs.get("extra_merging_strategy"):
            clusters = self._extra_merging_strategy(clusters)

        if all_valid:
            # now all clusters are valid
            # as a sideeffect the is valid also found all the errors within the clusters. Let's combine them
            decoded_error = np.zeros(self.fault_graph.shape[1], dtype=int)
            for cluster in clusters:
                decoded_error |= cluster.internal_error

            return decoded_error, cluster_history

        return self._decode_recursive(clusters, syndrome, cluster_history=cluster_history, **kwargs)

    def _create_initial_clusters(self, syndrome, **kwargs):
        #initial_cluster_strategy: str: Union[None, "connected_components"], default is None
        # TODO: implement "connected_components" initial cluster strategy
        do_initially_add_connecting_faults = kwargs.get("do_initially_add_connecting_faults", False)
        do_initially_add_surrounded_faults = kwargs.get("do_initially_add_connecting_faults", False)

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
            cluster = FaultGraphUnionFindDecoder.Cluster(checks, data, cluster_surface, cluster_surface_type, self.fault_graph)
            clusters.append(cluster)

        if do_initially_add_connecting_faults:
            clusters, _ = self._add_connecting_faults(clusters=clusters)
        elif do_initially_add_surrounded_faults:
            clusters, _ = self._add_surrounded_faults(clusters=clusters)
        
        # if kwargs.get("do_initially_add_connecting_faults", False) == 'connected_components':
        #     clusters = self._form_initial_connecting_clusters(clusters)

        # if kwargs.get("extra_merging_strategy") == 'extra':
        #     clusters = self._extra_merging_strategy(clusters)
        
        return clusters
    
    def _form_initial_connecting_clusters(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"]):
        """ Assumes all clusters have check surface, like in beginning of algorithm, and no faults in clusters
         Need to make sure this works even if some valid ones have fault surface OR remove fault surface from valid clusters """
        all_cluster_checks_mask = np.zeros(self.fault_graph.shape[0], dtype=np.bool_)
        all_cluster_faults_mask = np.zeros(self.fault_graph.shape[1], dtype=np.bool_)
        for cluster in clusters:
            all_cluster_checks_mask |= cluster.checks_mask
            all_cluster_faults_mask |= cluster.faults_mask

        connecting_faults_indices = np.where(self.fault_graph[all_cluster_checks_mask].sum(axis=0) >= 2)[0]
        # the former assumes no faults in any cluster yet

        for fault_idx in connecting_faults_indices:
            # find neighbour checks of a connecting fault
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

            # add fault to merged cluster
            clusters[idxA].faults_mask[fault_idx] = True

            # add all neighbour checks to the merged cluster too
            clusters[idxA].checks_mask |= neighbour_checks_mask
            clusters[idxA].cluster_surface_mask |= neighbour_checks_mask

            # delete the merged away clusters
            for cluster_index in merged_cluster_indices[::-1]:
                del clusters[cluster_index]

        return clusters
    
    def _form_connecting_clusters_from_checks(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"]):
        """ 
        Will connect and merge cluster based on rule:
        if a fault node is not in any cluster but connected to at least two different clusters, those cluster
        are merged, and the fault and the rest of its neighbour checks are added to the resulting cluster.
        This leaves fault surface clusters invariant.
        This does affect valid clusters (maybe add option to make sure it doesn't)
        """

        use_valid_clusters = True
        something_changed = False

        all_cluster_checks_mask = np.zeros(self.fault_graph.shape[0], dtype=np.bool_)
        all_cluster_faults_mask = np.zeros(self.fault_graph.shape[1], dtype=np.bool_)
        for cluster in clusters:
            if not cluster.valid or use_valid_clusters:
                all_cluster_checks_mask |= cluster.checks_mask
            all_cluster_faults_mask |= cluster.faults_mask

        connecting_faults_mask = self.fault_graph[all_cluster_checks_mask].sum(axis=0) >= 2
        connecting_faults_mask = connecting_faults_mask & ~all_cluster_faults_mask
        connecting_faults_indices = np.where(self.fault_graph[all_cluster_checks_mask].sum(axis=0) >= 2)[0]
        # the former assumes no faults in any cluster yet
        
        for fault_idx in connecting_faults_indices:
            # find neighbour checks of a connecting fault
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

            # add fault to merged cluster
            clusters[idxA].faults_mask[fault_idx] = True

            # add all neighbour checks to the merged cluster too
            clusters[idxA].checks_mask |= neighbour_checks_mask
            clusters[idxA].cluster_surface_mask |= neighbour_checks_mask

            # delete the merged away clusters
            for cluster_index in merged_cluster_indices[::-1]:
                del clusters[cluster_index]

        return clusters, something_changed

    def _extra_merging_strategy(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"]):
            # This strategy will be removed again...
            # TODO: add again, remove, or make different version
            return clusters
            data_in_clusters = np.zeros_like(clusters[0].faults)
            check_in_clusters = np.zeros_like(clusters[0].checks)
            for cluster in clusters:
                data_in_clusters |= cluster.data
                check_in_clusters |= cluster.checks
            data_outside_clusters = ~data_in_clusters
            checks_outside_clusters = ~check_in_clusters
            data_touching_checks_outside = np.any(self.adj_mat_T[checks_outside_clusters], axis=0)
            data_to_merge = data_outside_clusters & ~data_touching_checks_outside
            # data_to_merge contains all data nodes that only touch clustesr but aren't in clusters
            # iterate through all of them
            data_to_merge = np.where(data_to_merge)[0]
            for data_node in data_to_merge:
                # find connected checks (grow to checks):
                conn_checks = self.adj_mat_T[:, data_node]
                # find what clusters that corresponds too
                corresponding_cluster_indices = []
                for i, cluster in enumerate(clusters):
                    if np.any(conn_checks & cluster.checks):
                        corresponding_cluster_indices.append(i)
                # corresponding_cluster indices must contain at least one entry
                this_idx = corresponding_cluster_indices[0]
                clusters[this_idx].data_interior[data_node] = 1
                # merge others into this
                for other_idx in corresponding_cluster_indices[1:]:
                    clusters[this_idx].merge(clusters[other_idx])
                for i in corresponding_cluster_indices[1:][::-1]: # delete the merged clusters in reverse order so indices keep making sense
                    del clusters[i]
            return clusters

    def _add_surrounded_faults(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"]):
        raise NotImplementedError

    def _add_connecting_faults(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"]):
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

    def _find_correction(self, clusters: List["FaultGraphUnionFindDecoder.Cluster"], syndrome, cluster_decoding):
        decoded_error = np.zeros(self.fault_graph.shape[1], dtype=int)
        for cluster in clusters:
            if cluster_decoding == 'bposd':
                decoded_error |= cluster.decode_cluster_bposd(syndrome)
            else:
                decoded_error |= cluster.internal_error
        return decoded_error

class TannerUnionFindOptimized:
    def __init__(self, adj_mat: np.ndarray) -> None:
        """ Initializes the Tanner graph (simply with reduced adjacency matrix) """
        self.adj_mat = adj_mat
        self.decoders = {}

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

        def touches(self, other: "TannerUnionFindOptimized.Cluster") -> bool:
            return self.cluster_surface_type == other.cluster_surface_type and np.any(self.cluster_surface_mask & other.cluster_surface_mask)
        
        def overlaps(self, other: "TannerUnionFindOptimized.Cluster") -> bool:
            # not really used, just for completeness
            return np.any(self.checks_mask | other.checks_mask) or np.any(self.faults_mask | other.faults_mask)

        def merge(self, other: "TannerUnionFindOptimized.Cluster") -> None:
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
        
        def is_valid(self, syndrome, biadjacency_matrix=None, lse_solver=None) -> bool:

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
            solvable, x, stats = solve(a, b, minimize_weight=True, max_dof=0, stats=True, lse_solver=lse_solver)
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

        def grow(self, biadjacency_matrix = None) -> None:
            if biadjacency_matrix is None:
                biadjacency_matrix = self.biadjacency_matrix
                if biadjacency_matrix is None:
                    raise ValueError("please provide the relevant biadjacency matrix")
            
            if self.cluster_surface_type == 'c':
                #self.cluster_surface_mask = np.any(biadjacency_matrix[self.cluster_surface_mask], axis=0) &~self.faults_mask
                self.cluster_surface_mask = biadjacency_matrix[self.cluster_surface_mask].any(axis=0) &~self.faults_mask
                self.faults_mask |= self.cluster_surface_mask
                self.cluster_surface_type = 'f'

            elif self.cluster_surface_type == 'f':
                #self.cluster_surface_mask = np.any(biadjacency_matrix[:, self.cluster_surface_mask], axis=1) &~self.checks_mask
                biadjacency_matrix[:, self.cluster_surface_mask].any(axis=1) &~self.checks_mask
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
        
    class Decoder:
        def __init__(self, adj_mat: np.ndarray, T: int) -> None:
            m, n = adj_mat.shape
            if T > 1: # in this case (more than two consecutive SM rounds) assume with measurement error
                self.adj_mat_T = np.zeros((T*m, (T+1)*m + T*n), dtype=bool) # reduced adjacency matrix of T-foliated Tanner graph
                for i in range(T):
                    self.adj_mat_T[i*m:i*m+m, i*(m+n): i*(m+n) + m] = np.eye(m, dtype=bool)
                    self.adj_mat_T[i*m:i*m+m, i*(m+n) + m: i*(m+n) + m+n] = adj_mat
                    self.adj_mat_T[i*m:i*m+m, i*(m+n) + m+n: i*(m+n) + 2*m+n] = np.eye(m, dtype=bool)
            else:
                self.adj_mat_T = adj_mat # otherwise assume no measurement errors -> decode on unfoliated (in particular no boundary virtual qubits) Tanner graph
            self.m = m
            self.n = n
            self.T = T
            #self.adj_mat_T = adj_mat

        def predecode(self, syndrome):
            pre_error = self.adj_mat_T.T @ syndrome == 3
            pre_syndrome = self.adj_mat_T @ pre_error % 2 ^ syndrome
            return pre_error, pre_syndrome

        def decode(self, clusters: List["TannerUnionFindOptimized.Cluster"], syndrome, history=None, lse_solver=None):
            # kinda horrible place to put it but whatever
            clusters = self.extra_merging_strategy(clusters)
            
            if history is None:# benchmarking
                history = []# benchmarking

            step_history = {}# benchmarking
            step_history['num_clust'] = len(clusters)# benchmarking

            cluster_figs = []  # benchmarking
            all_valid = True
            decoded_error = np.zeros(self.adj_mat_T.shape[1], dtype=np.uint8)
            #new_clusters = []
            for cluster in clusters:
                if cluster.valid:
                    decoded_error |= cluster.internal_error
                    cluster_figs.append({'valid': True, 'cluster_nodes': (list(np.where(cluster.checks_mask)[0]), list(np.where(cluster.faults_mask)[0]))})
                    continue

                valid, x, figs = self.is_valid(cluster, syndrome, lse_solver=lse_solver)  # benchmarking
                cluster_figs.append(figs)  # benchmarking
                if not valid:
                    all_valid = False
                else:
                    decoded_error |= x
                    cluster.valid = True
                    cluster.internal_error = x

            step_history['clusters'] =  cluster_figs# benchmarking
            
            if all_valid:
                step_history['t_grow'] = 0
                #figs = (size, t_con, num_clust, cluster_figs, None) # benchmarking
                history.append(step_history)
                #history.append(figs) # benchmarking
                return decoded_error, history # benchmarking
            t0 = perf_counter() # benchmarking
            clusters = self.grow(clusters)
            t_grow = perf_counter() - t0 # benchmarking
            #figs = (size, t_con, num_clust, cluster_figs, t_grow) # benchmarking
            step_history['t_grow'] = t_grow# benchmarking
            history.append(step_history) # benchmarking
            return self.decode(clusters, syndrome, history=history, lse_solver=lse_solver)      # benchmarking

        def decode_non_recursive(self, clusters: List["TannerUnionFindOptimized.Cluster"], syndrome, lse_solver=None):            
            cluster_history = [] # benchmarking

            while True:
                # find smallest invalid cluster
                smallest_invalid_index = -1
                growth_step_history = {'num_cluster': len(clusters), 'clusters': []}
                for idx, cluster in enumerate(clusters):

                    growth_step_history['clusters'].append({'valid': cluster.is_valid(syndrome, lse_solver=lse_solver),
                                                            'checks': list(cluster.checks), #(list(np.where(cluster.checks)[0]), list(np.where(cluster.faults)[0])),
                                                            'faults': list(cluster.faults),
                                                            'size': cluster.size(),
                                                            'num_checks': cluster.num_checks(),
                                                            'num_faults': cluster.num_faults(),
                                                            'num_internal_faults': cluster.interior_faults_mask.sum(),
                                                            'dof': cluster.internal_dof,
                                                            'minimized': cluster.minimized_internal})

                    if not cluster.is_valid(syndrome, lse_solver=lse_solver) and smallest_invalid_index == -1 or cluster.size() < clusters[smallest_invalid_index].size():
                        smallest_invalid_index = idx

                cluster_history.append(growth_step_history)

                if smallest_invalid_index == -1:
                    break # no invalid cluster. Exit loop

                # grow cluster
                clusters[smallest_invalid_index].grow()

                # merge directly
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
                
            # now all clusters are valid
            # as a sideeffect the is valid also found all the errors within the clusters. Let's combine them
            decoded_error = np.zeros(self.adj_mat_T.shape[1], dtype=int)
            for cluster in clusters:
                decoded_error |= cluster.internal_error

            return decoded_error, cluster_history

        def is_valid(self, cluster: "TannerUnionFindOptimized.Cluster", syndrome, lse_solver=None):
            # checks_interior, data_interior, cluster_surface, cluster_surface_type, valid, internal_error = cluster
            # if cluster_surface_type == 'c':
            #     checks = checks_interior | cluster_surface
            #     data = data_interior
            # else:
            #     checks = checks_interior
            #     data = data_interior | cluster_surface

            clust_size = cluster.shape() # benchmarking

            t0 = perf_counter() # benchmarking
            a, true_interior_data = self.get_cluster_LSE(cluster) # benchmarking
            b = syndrome[cluster.checks_mask]
            t_int = perf_counter() - t0 # t_int really doesn't say much, just means how long to copy the array

            int_size = true_interior_data.sum() # benchmarking

            #t0 = perf_counter() # benchmarking

            valid, x, solve_stats = solve(a, b, minimize_weight=True, max_dof=0, stats=True, lse_solver=lse_solver)
            if valid:
            # x is sparse, contains true indices of error within interior
                int_x = np.zeros_like(cluster.faults_mask)
                int_x[true_interior_data] = x
            else: int_x = None
            # now int_x contains the interior error in dense format with respect to whole graph
            
            #t_valid = perf_counter() - t0 # benchmarking

            stats = {'valid': valid, 'clust_size': clust_size, 'cluster_nodes': (list(np.where(cluster.checks_mask)[0]), list(np.where(cluster.faults_mask)[0])), 'int_size': int_size, 't_int': t_int}
            for solve_stat in solve_stats:
                stats[solve_stat] = solve_stats[solve_stat]

            # if figs is not None: # benchmarking
            #     t_valid -= (figs[0] + figs[2]) # benchmarking
            #     figs = (clust_size, t_int, int_size, t_valid, *figs) # benchmarking
            # else:
            #     figs = (clust_size, t_int, int_size, t_valid, None, None, None)
            
            return valid, int_x, stats
            
        def get_cluster_LSE(self, cluster: "TannerUnionFindOptimized.Cluster"):
            """ajd_mat is the reduced (bipartite) adjaceny matrix (n//2 x n) of the whole tanner (X or Z) graph.
            cluster = (check_selector, data_selector)
            check_selector is a binary array of size n//2, where a 1 means the corresponing check node is in E.
            data_selector is a binary array of size n, where a 1 means the corresponing data node is in E."""
            # checks_interior, data_interior, cluster_surface, cluster_surface_type, valid, internal_error = cluster
            # if cluster_surface_type == 'c':
            #     check_selector = checks_interior | cluster_surface
            #     data_selector = data_interior
            # else:
            #     check_selector = checks_interior
            #     data_selector = data_interior | cluster_surface

            # First we want to find the interior of E. But we only care about the data qubits in Int(E)
            no_outside = self.adj_mat_T[~cluster.checks_mask].sum(axis=0) == 0 # selector for all data qubits that do not have connection to any check outside of E
            interior = cluster.faults_mask & no_outside
            return self.adj_mat_T[cluster.checks_mask][:,interior], interior
        
        def grow(self, clusters: List["TannerUnionFindOptimized.Cluster"], grow_valid=False, merging_strategy = None):
            grown_clusters: List[TannerUnionFindOptimized.Cluster] = []
            for cluster in clusters:
                #checks_interior, data_interior, cluster_surface, cluster_surface_type, valid, internal_error = cluster
                if not cluster.valid or grow_valid:
                    if cluster.cluster_surface_type == 'c':
                        cluster.checks_mask |= cluster.cluster_surface_mask
                        cluster.cluster_surface_mask = np.any(self.adj_mat_T[cluster.cluster_surface_mask], axis=0) & ~cluster.faults_mask
                        cluster.faults_mask |= cluster.cluster_surface_mask
                        cluster.cluster_surface_type = 'f'
                    elif cluster.cluster_surface_type == 'f':
                        cluster.faults_mask |= cluster.cluster_surface_mask
                        cluster.cluster_surface_mask =  np.any(self.adj_mat_T[:, cluster.cluster_surface_mask], axis=1) & ~cluster.checks_mask
                        cluster.checks_mask |= cluster.cluster_surface_mask
                        cluster.cluster_surface_type = 'c'

                # merge directly
                merged_away = []
                for i, other in enumerate(grown_clusters):
                    if cluster.touches(other): # these 2 clusters are now connected
                        # keep track of merged
                        merged_away.append(i)
                        cluster.merge(other) # merges other into cluster in place

                for i in merged_away[::-1]: # delete the merged clusters in reverse order so indices keep making sense
                    del grown_clusters[i]
                # append the merged cluster
                grown_clusters.append(cluster)
            # very inefficient, but let's just hack this in:
            # want to add and merge all clusters when a data node is outside of any cluster but touches only clusters
            # get all such data nodes:
            # all data nodes not in clusters
            if merging_strategy is None:
                grown_clusters = self.extra_merging_strategy(grown_clusters)
            return grown_clusters
    
        def extra_merging_strategy(self, clusters: List["TannerUnionFindOptimized.Cluster"]):
            # This strategy will be removed again...
            return clusters
            data_in_clusters = np.zeros_like(clusters[0].faults)
            check_in_clusters = np.zeros_like(clusters[0].checks)
            for cluster in clusters:
                data_in_clusters |= cluster.data
                check_in_clusters |= cluster.checks
            data_outside_clusters = ~data_in_clusters
            checks_outside_clusters = ~check_in_clusters
            data_touching_checks_outside = np.any(self.adj_mat_T[checks_outside_clusters], axis=0)
            data_to_merge = data_outside_clusters & ~data_touching_checks_outside
            # data_to_merge contains all data nodes that only touch clustesr but aren't in clusters
            # iterate through all of them
            data_to_merge = np.where(data_to_merge)[0]
            for data_node in data_to_merge:
                # find connected checks (grow to checks):
                conn_checks = self.adj_mat_T[:, data_node]
                # find what clusters that corresponds too
                corresponding_cluster_indices = []
                for i, cluster in enumerate(clusters):
                    if np.any(conn_checks & cluster.checks):
                        corresponding_cluster_indices.append(i)
                # corresponding_cluster indices must contain at least one entry
                this_idx = corresponding_cluster_indices[0]
                clusters[this_idx].data_interior[data_node] = 1
                # merge others into this
                for other_idx in corresponding_cluster_indices[1:]:
                    clusters[this_idx].merge(clusters[other_idx])
                for i in corresponding_cluster_indices[1:][::-1]: # delete the merged clusters in reverse order so indices keep making sense
                    del clusters[i]
            return clusters

    def get_decoder(self, T: int) -> "TannerUnionFindOptimized.Decoder":
        if T not in self.decoders:
            self.decoders[T] = TannerUnionFindOptimized.Decoder(self.adj_mat, T)
        return self.decoders[T]

    def decode(self, syndrome: np.ndarray, predecode=False, lse_solver=None, grow_strategy=None) -> Tuple[np.ndarray, List[Tuple[Tuple[int,int], float, int, List[Tuple[Tuple[int,int], float, int, float, float, int, float]], float]]]:
        """ Main entry point for decoding, takes syndrome and initialized clusters. In optimized version actually creates the clusters. """

        "Syndrome can be from multiple measurement rounds"
        m, n = self.adj_mat.shape
        l = len(syndrome)
        # syndrome length should be T*m where T+1 is number of SM rounds (i.e T number of parity comparisons between them)
        if l % m != 0:
            raise ValueError("Syndrome not of valid length (valid <=> multiple of number of checks)")
        
        T = l // m

        decoder = self.get_decoder(T)

        if predecode:
            pre_error, syndrome = decoder.predecode(syndrome=syndrome)

        clusters = []
        for check in np.where(syndrome)[0]:
            checks = np.zeros(T*m, dtype=bool)
            if T > 1:
                data = np.zeros((T+1)*m + T*n, dtype=bool)
            else:
                data = np.zeros(n, dtype=bool)
            cluster_surface = np.zeros(T*m, dtype=bool)
            cluster_surface[check] = True
            checks |= cluster_surface # new versiiiiiiion
            cluster_surface_type = 'c'
            valid = False
            internal_error = None
            cluster = TannerUnionFindOptimized.Cluster(checks, data, cluster_surface, cluster_surface_type, decoder.adj_mat_T)
            clusters.append(cluster)

        if grow_strategy == 'individual':
            decoded_error, history = decoder.decode_non_recursive(clusters, syndrome, lse_solver=lse_solver)
        else:
            decoded_error, history = decoder.decode(clusters, syndrome, lse_solver=lse_solver)

        if predecode:
            decoded_error ^= pre_error

        return decoded_error, history

    def _decode_individual(self, clusters: List["TannerUnionFindOptimized.Cluster"], syndrome, history=None, lse_solver=None):
        pass

    def _decode_recursive(self, clusters: List["TannerUnionFindOptimized.Cluster"], syndrome, history=None, lse_solver=None):
        pass

    def _create_initial_clusters(self, syndrome, kwargs):
        pass