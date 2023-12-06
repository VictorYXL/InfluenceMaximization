from typing import List 
import networkx as nx


# LT Model Simulator
def lt_simulator(G: nx.Graph, seed_nodes: List[int]):
    """ Linear threshold (LT) model for only one step.
    Args:
        G (nx.Graph): Source graph.
        seed_nodes (int): List of seed nodes.
    """
    influenced_nodes = set(seed_nodes)
    newly_influenced_nodes = set(seed_nodes)

    # One step LT simulation.
    # Use a set to avoid duplicates
    next_influenced_nodes = set()

    for node in G:
        if node not in influenced_nodes:
            neighbors = G.predecessors(node)  # Nodes that influence "node"
            # Sum the weights of the active neighbors
            influence_sum = sum(G[n][node]["weight"] for n in neighbors if n in influenced_nodes)
            # Node becomes active if the sum of weights is greater than its threshold
            if influence_sum >= G.nodes[node]["threshold"]:
                next_influenced_nodes.add(node)

        # Update the sets for the next iteration
        newly_influenced_nodes = next_influenced_nodes - influenced_nodes
        influenced_nodes |= newly_influenced_nodes

    return influenced_nodes
