from typing import Iterable
from functools import cached_property
import numpy as np
import networkx as nx


class DegreeDistribution(object):
    def __init__(self, degree_dict, num_nodes: int) -> None:
        self.node_degrees = dict(degree_dict)

        unique_degrees, degree_counts = np.unique(list(self.node_degrees.values()), return_counts=True)
        unique_degrees, degree_counts = list(unique_degrees), list(degree_counts)

        self.degree_freq_dict = {d:0 for d in range(num_nodes)}
        self.degree_freq_dict.update({d:c for d,c in zip(unique_degrees, degree_counts)})

    def __getitem__(self, node: int) -> int:
        return self.node_degrees[node]
    
    def __setitem__(self, node: int, degree: int) -> None:
        self.node_degrees[node] = degree
        
    def add_node(self, node: int, neighbors: list[int]) -> None:
        # Add new node
        node_degree = len(neighbors)

        self.node_degrees[node] = node_degree        
        self.degree_freq_dict[node_degree] += 1

        # Change neighbor degrees
        for neighbor in neighbors:
            self.degree_freq_dict[self[neighbor]] -= 1

            self[neighbor] += 1
            
            self.degree_freq_dict[self[neighbor]] += 1

    def remove_node(self, node: int, neighbors: list[int]) -> None:
        # Add new node
        node_degree = len(neighbors)

        self.node_degrees.pop(node)     
        self.degree_freq_dict[node_degree] -= 1

        # Change neighbor degrees
        for neighbor in neighbors:
            self.degree_freq_dict[self[neighbor]] -= 1

            self[neighbor] -= 1
            
            self.degree_freq_dict[self[neighbor]] += 1

    @property
    def unique_degrees(self) -> list[int]:
        return [degree for degree, count in self.degree_freq_dict.items() if count > 0]
    
    @property
    def cdf(self) -> list[float]:
        counts = [count for count in self.degree_freq_dict.values() if count > 0]
        return np.cumsum(counts) / np.sum(counts)


class SubgraphHandler:
    def __init__(self, full_graph: nx.Graph, initial_node_set: Iterable[int]) -> None:
        self.full_graph = full_graph

        self.subgraph_nodes = list(initial_node_set)

        # Determine which nodes are not in the subgraph
        self.not_subgraph_nodes = [node for node in full_graph.nodes() if node not in initial_node_set]

        self.degree_distribution = DegreeDistribution(self.full_graph.subgraph(self.subgraph_nodes).degree(), num_nodes=len(initial_node_set))


    def add(self, node: int) -> None:
        # Add node, and increase degree of neighbors
        add_node_neighbors = list(self.full_graph.neighbors(node))
        add_node_neighbors_in_subgraph = [n for n in add_node_neighbors if n in self.subgraph_nodes]

        self.degree_distribution.add_node(node, add_node_neighbors_in_subgraph)

        self.subgraph_nodes.append(node)
        self.not_subgraph_nodes.remove(node)

    def remove(self, node: int) -> None:
        # Remove node and reduce degree of neighbors
        remove_node_neighbors = list(self.full_graph.neighbors(node))
        remove_node_neighbors_in_subgraph = [n for n in remove_node_neighbors if n in self.subgraph_nodes]

        self.degree_distribution.remove_node(node, remove_node_neighbors_in_subgraph)

        self.subgraph_nodes.remove(node)
        self.not_subgraph_nodes.append(node)

    def ks_distance(self):
        subgraph_degrees = np.sort(list(self.degree_distribution.node_degrees.values()))
        full_degrees = np.sort(self.full_graph_degree_sequence)

        all_degrees = np.concatenate([subgraph_degrees, full_degrees])

        cdf1 = np.searchsorted(subgraph_degrees, all_degrees, side='right') / len(subgraph_degrees)
        cdf2 = np.searchsorted(full_degrees, all_degrees, side='right') / len(full_degrees)

        dist = np.max(np.abs(cdf1 - cdf2))

        return dist

    @property
    def nodes(self):
        return self.subgraph_nodes
    
    @property
    def nodes_not_in_subgraph(self):
        return self.not_subgraph_nodes
    
    @cached_property
    def full_graph_degree_sequence(self):
        return list([d for n, d in self.full_graph.degree()])
    
    @cached_property
    def full_graph_unique_degrees(self):
        return list(np.unique([d for n, d in self.full_graph.degree()]))
    
    @cached_property
    def full_graph_degree_cdf(self):
        degree_sequence = [d for n, d in self.full_graph.degree()]

        _, counts = np.unique(degree_sequence, return_counts=True)
        cdf = np.cumsum(counts) / sum(counts)

        return cdf
    
    @property
    def unique_degrees(self) -> list[int]:
        return self.degree_distribution.unique_degrees
    
    @property
    def degree_cdf(self) -> list[float]:
        return self.degree_distribution.cdf