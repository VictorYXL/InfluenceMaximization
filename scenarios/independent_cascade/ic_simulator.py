import random
from typing import List 
import networkx as nx

def ic_simulator(G: nx.Graph, seed_nodes: List[int]):
    """ Independent cascade (IC) model for one step.

    Args:
        G (nx.Graph): Source graph.
        seed_nodes (int): List of seed nodes.
    """
    influenced_nodes = set(seed_nodes)
    newly_influenced_nodes = set(seed_nodes)

    # One step IC simulation.
    next_influenced_nodes = set()
    for node in newly_influenced_nodes:
        neighbors = G[node]
        for neighbor in neighbors:
            if neighbor not in influenced_nodes:
                influence_prob = G[node][neighbor]["weight"]
                if random.random() < influence_prob:
                    next_influenced_nodes.add(neighbor)
                    influenced_nodes.add(neighbor)

    # Update the sets for the next iteration
    newly_influenced_nodes = next_influenced_nodes - influenced_nodes
    influenced_nodes |= newly_influenced_nodes
    return newly_influenced_nodes
