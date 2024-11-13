# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Define the bivariate bicylce code."""

from typing import List, Optional, Sequence, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rustworkx as rx
from IPython.display import display
from matplotlib.patches import Circle
from rustworkx.visualization import graphviz_draw
from scipy import sparse

from qiskit_qec.linear.matrix import rank
from qiskit_qec.linear.symplectic import normalizer
from qiskit_qec.operators.pauli_list import PauliList


def cs_pow(l, power=1):
        """
        Calculates and returns a power of the cyclic shift matrix of size lxl (C_l)

        Parameters
        ----------
        l     : int
            size of cyclic shift matrix.
        power : int
            Power to which Cl is raised. Defaults to 1.

        Returns
        --------
        np.array
            C_l^power

        Examples
        --------

        >>> cs_pow(3, 2)
        (C_3)^2

        """

        return np.roll(np.eye(l, dtype=np.uint8), shift=power, axis=1)

def arr_to_indices(arr: np.array) -> List[List[int]]:
        """ Converts a numpy array to a list of list of indices where it is non-zero """
        return [np.where(row)[0].tolist() for row in arr]

class ToricCode:
    """Toric code data.

    The X and Z gauge operator lists are given as lists of supports.
    There is a consistent qubit ordering of these lists so that
    we can construct gate schedules for circuits.
    """

    def __init__(
        self,
        d: int,
    ) -> None:
        """Initializes a toric code on a grid

        Args:
            d: distance of code,

        Examples:
            Example 1:
            >>> code = ToricCode(d=12)
            [[144,2,12]] toric code
        """

        self._d = d
        self.n = d*d
        self.s = self.n//2
        self._k = 2

        self._create_check_matrices()

        #self._x_stabilizers = arr_to_indices(self.hx)
        self._z_stabilizers = arr_to_indices(self.hz)

        #self.symplectic_stabilizers = np.vstack([np.hstack([self.hx, np.zeros((self.n//2, self.n))]),
        #                                         np.hstack([np.zeros((self.n//2, self.n)), self.hz])])

        self._logical_x = None
        self._logical_z = None

        #self.create_tanner_graph()

    def __str__(self) -> str:
        """Formatted string."""
        #return f"(l={self.l},m={self.m}) bivariate bicycle code"
        return f"[[{self.n}, {self.k}, ≤{self.d}]] bivariate bicyle code"

    def __repr__(self) -> str:
        """String representation."""
        val = str(self)
        val += f"\nx_stabilizers = {self.x_stabilizers}"
        val += f"\nz_stabilizers = {self.z_stabilizers}"
        val += f"\nlogical_x = {self.logical_x}"
        val += f"\nlogical_z = {self.logical_z}"
        return val
    
    def __eq__(self, other: "BBCodeVector") -> bool:
        return np.array_equal(self.hx, other.hx) and np.array_equal(self.hz, other.hz)

    @property
    def x_stabilizers(self) -> PauliList:
        raise NotImplementedError
        return self._x_stabilizers
    
    @property
    def z_stabilizers(self) -> PauliList:
        return self._z_stabilizers
    
    @property
    def x_gauges(self) -> PauliList:
        raise NotImplementedError
        return self.x_stabilizers
    
    @property
    def z_gauges(self) -> PauliList:
        return self.z_stabilizers
    
    @property
    def logical_z(self) -> List[List[int]]:
        raise NotImplementedError
        if self._logical_z is None:
            self._create_logicals()
        return BBCode.arr_to_indices(self._logical_z.matrix[:, self.n:])
    
    @property
    def logical_x(self) -> List[List[int]]:
        raise NotImplementedError
        if self._logical_x is None:
            self._create_logicals()
        return BBCode.arr_to_indices(self._logical_x.matrix[:, :self.n])
    
    @property
    def k(self):
        return self._k
        if self._k is None:
            self._k = self.n - 2*rank(self.hx)
        return self._k

    @property
    def d(self):
        return self._d
    
    def get_foliated_tanner_x(self, T, boundary_start=True, boundary_end=False):
        m, n = self.hx.shape

        shape = (T*m, (T+1)*m + T*n)

        foliated = np.zeros(shape, dtype=np.bool_)

        for i in range(T):
            foliated[i*m:i*m+m, i*(m+n): i*(m+n) + m] = np.eye(m)
            foliated[i*m:i*m+m, i*(m+n) + m: i*(m+n) + m+n] = self.hx
            foliated[i*m:i*m+m, i*(m+n) + m+n: i*(m+n) + 2*m+n] = np.eye(m)

        if not boundary_start:
            foliated = foliated[m:]

        if not boundary_end:
            foliated = foliated[:-m]

        return foliated
    
    def get_foliated_tanner_z(self, T, boundary_start=True, boundary_end=False):
        m, n = self.hz.shape

        shape = (T*m, (T+1)*m + T*n)

        foliated = np.zeros(shape, dtype=np.bool_)

        for i in range(T):
            foliated[i*m:i*m+m, i*(m+n): i*(m+n) + m] = np.eye(m)
            foliated[i*m:i*m+m, i*(m+n) + m: i*(m+n) + m+n] = self.hz
            foliated[i*m:i*m+m, i*(m+n) + m+n: i*(m+n) + 2*m+n] = np.eye(m)

        if not boundary_start:
            foliated = foliated[m:]

        if not boundary_end:
            foliated = foliated[:-m]

        return foliated

    def get_syndrome(self, physical_error: np.ndarray, base: str='z') -> np.ndarray:
        if base == 'z':
            return self.hz @ physical_error % 2
        elif base == 'x':
            return self.hx @ physical_error % 2
        else:
            raise ValueError(f'"base" must be one of {{"x", "y"}}, not {base}.')
        
    def get_logical_error(self, physical_error: np.ndarray, base: str='z') -> np.ndarray:
        if self._logical_x is None:
            self._create_logicals()
        if base == 'z':
            return self._logical_z.matrix[:, self.n:] @ physical_error % 2
        elif base == 'x':
            return self._logical_x.matrix[:, :self.n] @ physical_error % 2
        else:
            raise ValueError(f'"base" must be one of {{"x", "y"}}, not {base}.')
    
    def plot_code_connections(self, show=True, figsize=(12,8)):
        """ Assumes z checks, can be customized later """
        fig, ax = plt.subplots(figsize = figsize)        

        #x, y = self._fault_idx2coord(np.arange(self.n))
        
        ax.scatter(*self._fault_idx2coord(np.arange(self.n)), color='blue', marker='o', label='L-Datas')
        ax.scatter(*self._check_index2coord(np.arange(self.s), check_type='z'), color='green', marker='s', label='Z-Checks')
        ax.scatter(*self._check_index2coord(np.arange(self.s), check_type='x'), color='red', marker='s', label='X-Checks')
        #ax.scatter(x, y+0.5, color='red', marker='s', label='X-Checks')
        #ax.scatter(x+0.5, y+0.5, color='orange', marker='o', label='R-Datas')
        ax.set_aspect('equal')
        # if title is not None:
        #     ax.set_title(title)
        # ax.set_xlabel('x (d)')
        # ax.set_ylabel('y (d)')
        # ax.set_xticks(np.arange(0,self.d))
        # ax.set_yticks(np.arange(0,self.d))

        # Store the current limits after plotting the important data
        # x_limits = ax.get_xlim()
        # y_limits = ax.get_ylim()

        # # add errors
        # if faults is not None:
        #     x_faults, y_faults = self._fault_idx2coord(faults)
        #     ax.scatter(x_faults, y_faults, color='darkred', marker="$\u26A1$", s=324, label='actual_error')

        # cluster_colors = ['lightblue', 'lightpink', 'lightgreen', 'palegoldenrod', 'thistle', 'rosybrown', 'navajowhite', 'lightgray', 'mistyrose']

        # # add clusters
        # if clusters is not None:
        #     for i, cluster in enumerate(clusters):
        #         checks, datas = cluster
        #         # assume z checks
        #         #check nodes
        #         for xi, yi in zip(*self._check_index2coord(checks, check_type='z')):
        #             circle = Circle((xi, yi), 0.3, color=cluster_colors[i % len(cluster_colors)], label='Scaled Circle', zorder=-1)
        #             ax.add_patch(circle)
        #         # fault nodes
        #         for xi, yi in zip(*self._fault_idx2coord(datas)):
        #             circle = Circle((xi, yi), 0.3, color=cluster_colors[i % len(cluster_colors)], label='Scaled Circle', zorder=-1)
        #             ax.add_patch(circle)

        # # Reapply the original limits
        # ax.set_xlim(x_limits)
        # ax.set_ylim(y_limits)
        if show:
            plt.show()
        else:
            plt.close(fig=fig)
        return fig 

    def plot_code(self, faults=None, clusters=None, title=None, show=True, figsize = (12,8)):
        """ Assumes z checks, can be customized later """
        fig, ax = plt.subplots(figsize = figsize)        

        #x, y = self._fault_idx2coord(np.arange(self.n))
        
        ax.scatter(*self._fault_idx2coord(np.arange(self.n)), color='blue', marker='o', label='L-Datas')
        ax.scatter(*self._check_index2coord(np.arange(self.s), check_type='z'), color='green', marker='s', label='Z-Checks')
        #ax.scatter(x, y+0.5, color='red', marker='s', label='X-Checks')
        #ax.scatter(x+0.5, y+0.5, color='orange', marker='o', label='R-Datas')
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('x (d)')
        ax.set_ylabel('y (d)')
        ax.set_xticks(np.arange(0,self.d))
        ax.set_yticks(np.arange(0,self.d))

        # Store the current limits after plotting the important data
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        # add errors
        if faults is not None:
            x_faults, y_faults = self._fault_idx2coord(faults)
            ax.scatter(x_faults, y_faults, color='darkred', marker="$\u26A1$", s=324, label='actual_error')

        cluster_colors = ['lightblue', 'lightpink', 'lightgreen', 'palegoldenrod', 'thistle', 'rosybrown', 'navajowhite', 'lightgray', 'mistyrose']

        # add clusters
        if clusters is not None:
            for i, cluster in enumerate(clusters):
                checks, datas = cluster
                # assume z checks
                #check nodes
                for xi, yi in zip(*self._check_index2coord(checks, check_type='z')):
                    circle = Circle((xi, yi), 0.3, color=cluster_colors[i % len(cluster_colors)], label='Scaled Circle', zorder=-1)
                    ax.add_patch(circle)
                # fault nodes
                for xi, yi in zip(*self._fault_idx2coord(datas)):
                    circle = Circle((xi, yi), 0.3, color=cluster_colors[i % len(cluster_colors)], label='Scaled Circle', zorder=-1)
                    ax.add_patch(circle)

        # Reapply the original limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        if show:
            plt.show()
        else:
            plt.close(fig=fig)
        return fig 

    def plot_interactive_decoding(self, cluster_history: pd.DataFrame, actual_error, decoded_error, figsize=(12,8)):
        step_count = cluster_history['step'].max() + 1
        plots = []
        for step_index in range(cluster_history['step'].max() + 1):
            plots.append(
                self.plot_code(
                    faults=actual_error,
                    clusters=cluster_history[cluster_history['step']==step_index]['cluster_nodes'].values,
                    title=f'Growth step {step_index} and actual error',
                    show=False,
                    figsize=figsize))
        plots.append(
            self.plot_code(
                faults=decoded_error,
                clusters=cluster_history[cluster_history['step']==cluster_history['step'].max()]['cluster_nodes'].values,
                title=f'Final clusters and decoded error',
                show=False,
                figsize=figsize)) 

        output = widgets.Output()

        def show_plot(step_index):
            output.clear_output(wait=True)
            with output:
                display(plots[step_index])
                
        step_slider = widgets.IntSlider(min=0, max=step_count, step=1, value=0, description='Step:')
        widget_ui = widgets.interactive(show_plot, step_index=step_slider)
        display(widget_ui, output)

    def _create_check_matrices(self):
        d = self.d
        data_coords = [(i,j) for i in range(d) for j in range(d)]
        self.data_coords_indices = {data_coords[i]:i for i in range(len(data_coords))}
        #print(data_coords)
        check_coords = [(1/2+coord[0],1/2+coord[1]) for coord in data_coords if ((coord[0]+coord[1]) % 2 ==0)]
        self.check_coords_indices = {check_coords[i]:i for i in range(len(check_coords))}
        #print(check_coords)
        check_matrix = []
        for coordA in check_coords:
            neighbors = [
                ((coordA[0]-1/2) % d,(coordA[1]-1/2)% d),
                ((coordA[0]+1/2)% d,(coordA[1]-1/2)% d),
                ((coordA[0]-1/2)% d,(coordA[1]+1/2)% d),
                ((coordA[0]+1/2)% d,(coordA[1]+1/2)% d)]
            supp = []
            for coordB in data_coords:
                if coordB in neighbors:
                    supp.append(1)
                else:
                    supp.append(0)
            check_matrix.append(supp)
        #print(check_matrix)
        check_matrix = np.array(check_matrix)
        self.hz = check_matrix
        
        logical_ops_coords = [{(i,0) for i in range(d)}, {(0,j) for j in range(d)}]
        logical_ops = []
        for logical_op in logical_ops_coords:
            logical_ops.append([1 if data_coord in logical_op else 0 for data_coord in data_coords])

    def _create_logicals(self):
        d = self.d
        data_coords = [(i,j) for i in range(d) for j in range(d)]
        logical_ops_coords = [{(i,0) for i in range(d)}, {(0,j) for j in range(d)}]
        logical_ops = []
        for logical_op in logical_ops_coords:
            logical_ops.append([1 if data_coord in logical_op else 0 for data_coord in data_coords])

        logical_ops = np.array(logical_ops)

        self._logical_z = PauliList(np.hstack([np.zeros((2, self.n),dtype=np.int8), np.array(logical_ops,dtype=np.int8)]))

        # center_, x_new, z_new = normalizer(self.symplectic_stabilizers.astype(np.bool_))

        # self._logical_z = PauliList(z_new)
        # self._logical_x = PauliList(x_new)

    def _idx2coord(self, idx):
        raise NotImplementedError
        # returns the relative coordinate of an idx within group (X, Z, L or R)
        if type(idx) == list:
            idx = np.array(idx)
        return (idx % self.l, idx // self.l)
    
    def _fault_idx2coord(self, idx):
        if type(idx) == list:
            idx = np.array(idx)

        return (idx // self.d, idx % self.d)
    
    def _check_index2coord(self, idx, check_type):
        if type(idx) == list:
            idx = np.array(idx)

        x_shift = (check_type=='x') * (2*idx // self.d % 2 == 0)
        z_shift = (check_type=='z') * (2*idx // self.d % 2 == 1)

        return (2*idx // self.d + 0.5, 2*idx % self.d + 0.5 + x_shift + z_shift)
    
    def _coord2idx(self, coord):
        return coord[0] + coord[1]*self.l
    
    def _coord_add(self, coord, v):
        return (coord[0] + v[0]) % self.l, (coord[1] + v[1]) % self.m
    
    def _coord_sub(self, coord, v):
        return (coord[0] - v[0]) % self.l, (coord[1] - v[1]) % self.m
    
    class CodeConverter:
        def __init__(self, code_a: "BBCodeVector", code_b: "BBCodeVector", graph_matcher: nx.isomorphism.GraphMatcher) -> None:
            self.code_a = code_a
            self.code_b = code_b
            self.graph_matcher = graph_matcher
            num_checks = code_a.s
            num_faults = code_a.n
            self._check_forward_exchange = np.empty(num_checks, dtype=int)
            self._check_backward_exchange = np.empty(num_checks, dtype=int)
            self._fault_forward_exchange = np.empty(num_faults, dtype=int)
            self._fault_backward_exchange = np.empty(num_faults, dtype=int)
            for key in graph_matcher.mapping:
                val = graph_matcher.mapping[key]
                if key < num_checks:
                    self._check_forward_exchange[key] = val
                    self._check_backward_exchange[val] = key
                else:
                    self._fault_forward_exchange[key-num_checks] = val - num_checks
                    self._fault_backward_exchange[key-num_checks] = val - num_checks

        def faults_forwards(self, indices_a: List[int]) -> List[int]:
            indices_b = self._fault_backward_exchange[indices_a]
            return indices_b

        def faults_backwards(self, indices_b: List[int]) -> List[int]:
            indices_a = self._fault_backward_exchange[indices_b]
            return indices_a
        
        def checks_forwards(self, indices_a: List[int]) -> List[int]:
            indices_b = self._check_forward_exchange[indices_a]
            return indices_b
        
        def checks_backwards(self, indices_b: List[int]) -> List[int]:
            indices_a = self._check_forward_exchange[indices_b]
            return indices_a
        
    def z_fault_graph(self):
        sparse_repr = sparse.csr_matrix(self.hz)
        return nx.algorithms.bipartite.from_biadjacency_matrix(sparse_repr)

    def check_equivalence(self, other: "BBCodeVector"):
        s_a = sparse.csr_matrix(self.hz)
        s_b = sparse.csr_matrix(other.hz)
        Ga = nx.algorithms.bipartite.from_biadjacency_matrix(s_a)
        Gb = nx.algorithms.bipartite.from_biadjacency_matrix(s_b)
        GM = nx.isomorphism.GraphMatcher(Ga, Gb)
        if not GM.is_isomorphic():
            return None
        
        return BBCodeVector.CodeConverter(code_a=self, code_b=other, graph_matcher=GM)

    @property
    def tanner_graph(self):
        return self._tanner_graph
    
    @property
    def tanner_graph_X(self):
        nodes = self.tanner_graph.filter_nodes(lambda node: node['subtype'] != 'Z')
        return self.tanner_graph.subgraph(nodes)
    
    @property
    def tanner_graph_Z(self):
        nodes = self.tanner_graph.filter_nodes(lambda node: node['subtype'] != 'X')
        return self.tanner_graph.subgraph(nodes)
    
    def create_tanner_graph(self):
        """
        Creates the tanner graph of the code. Manually creates nodes and edges to have more flexibility and additional parameters,
        instead of something like rx.from_adjacency_matrix(np.hstack([self.hx, self.hz])).

        """
        tanner = rx.PyGraph()

        offset = self.l*self.m

        l_nodes = [{'index': i,
                    'subindex': i,
                    'type': 'data',
                    'subtype': 'L',
                    'node_attr': {'color': 'blue', 'fillcolor': 'blue', 'style': 'filled', 'shape': 'circle', 'label': f'L_{i}'}}
                    for i in range(self.s)]
        r_nodes = [{'index': i + offset, 
                    'subindex': i,
                    'type': 'data',
                    'subtype': 'R',
                    'node_attr': {'color': 'orange', 'fillcolor': 'orange', 'style': 'filled', 'shape': 'circle', 'label': f'R_{i}'}}
                    for i in range(self.s)]
        x_nodes = [{'index': i + 2*offset, 
                    'subindex': i,
                    'type': 'check',
                    'subtype': 'X',
                    'node_attr': {'color': 'red', 'fillcolor': 'red', 'style': 'filled', 'shape': 'square', 'label': f'X_{i}'}}
                    for i in range(self.s)]
        z_nodes = [{'index': i + 3*offset, 
                    'subindex': i,
                    'type': 'check',
                    'subtype': 'Z',
                    'node_attr': {'color': 'green', 'fillcolor': 'green', 'style': 'filled', 'shape': 'square', 'label': f'Z_{i}'}}
                    for i in range(self.s)]

        tanner.add_nodes_from(l_nodes)
        tanner.add_nodes_from(r_nodes)
        tanner.add_nodes_from(x_nodes)
        tanner.add_nodes_from(z_nodes)

        for c,q in zip(*np.where(self.A)): # between X and L
            tanner.add_edge(c + 2*self.s, q, None)
        for c,q in zip(*np.where(self.B)): # between X and R
            tanner.add_edge(c + 2*self.s, q + self.s, None)
        for c,q in zip(*np.where(self.B.T)): # between Z and L
            tanner.add_edge(c + 3*self.s, q, None)
        for c,q in zip(*np.where(self.A.T)): # between Z and R
            tanner.add_edge(c + 3*self.s, q + self.s, None)

        self._tanner_graph = tanner

    def draw_tanner(self):
        """
        Draws the tanner graph using rustworkx.visualization.graphviz_draw. Graphviz must be installed.
        """
        return graphviz_draw(self.tanner_graph, node_attr_fn=lambda node: node['node_attr'])

