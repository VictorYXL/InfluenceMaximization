import os
import random
import math
import json
from typing import Dict, List
from graph_loader import load_graph_from_tsv
from scenarios.independent_cascade.ic_simulator import ic_simulator
from scenarios.linear_threshold.lt_simulator import lt_simulator

def softmax(weights):
    max_weight = max(weights)
    e_weights = [math.exp(w - max_weight) for w in weights]  # Subtract max_weight for numerical stability
    total = sum(e_weights)
    return [w / total for w in e_weights]

def generate_edge_weight(origin_file_path: str, edge_file_path: str, lower_bound: float, upper_bound: float, smaller_than_1: bool=False):
    """Generate edge weights using softmax to ensure the sum of weights for any node is at most 1."""
    # Dictionary to keep track of the edges and their initial random weights for each destination node.
    edges_dict = {}

    # Read the origin file and store the initial random weights.
    with open(origin_file_path, "r") as origin_file:
        for line in origin_file:
            src_node, dst_node = line.strip().split(" ")
            intensity = random.uniform(lower_bound, upper_bound)
            if dst_node not in edges_dict:
                edges_dict[dst_node] = {}
            edges_dict[dst_node][src_node] = intensity

    if smaller_than_1:
        # Apply softmax to the weights of the edges for each destination node.
        for dst_node in edges_dict:
            weights = list(edges_dict[dst_node].values())
            if sum(weights) >= 1:
                s += 1
                weights = softmax(weights)
                
            for src_node, weight in zip(edges_dict[dst_node].keys(), weights):
                edges_dict[dst_node][src_node] = weight

    # Write the normalized weights to the edge file.
    with open(edge_file_path, "w") as edge_file:
        edge_file.write("src_node\tdst_node\tweight\n")
        for dst_node, src_nodes in edges_dict.items():
            for src_node, weight in src_nodes.items():
                edge_file.write(f"{src_node}\t{dst_node}\t{weight}\n")

def generate_dataset(node_file: str, edge_file: str, num_simulations: Dict[str, int], file_path: str, influence_function: callable, seed_ratio: float):
    graph = load_graph_from_tsv(node_file, edge_file)
    data = {
        "node_file": node_file,
        "edge_file": edge_file,
        "influence_function": str(influence_function.__name__),
        "seed_ratio": seed_ratio
    }
    all_nodes = list(graph.nodes())
   
    for mode, num in num_simulations.items():
        data[mode] = []
        for index in range(num):
            print(index)
            # Randomly select seed nodes
            seed_size = int(random.random() * seed_ratio * 2 * len(graph.nodes))
            seed_nodes = random.sample(all_nodes, seed_size)

            # Run the simulation
            influenced_nodes = influence_function(graph, seed_nodes)

            # Create input and label vectors
            input_vector = [1 if node in seed_nodes else 0 for node in all_nodes]
            label_vector = [1 if node in influenced_nodes else 0 for node in all_nodes]

            # Store the simulation data
            data[mode].append({
                'input': input_vector,
                'label': label_vector
            })
   
    # Save the data to a file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    graph_path = os.path.join("data", "facebook", "graph")
    node_file = os.path.join(graph_path, "node.tsv")
    edge_file = os.path.join(graph_path, "edge_norm.tsv")
    num_simulations = {"training_data": 10000, "evaluation_data": 3000}
    output_path = os.path.join("data", "facebook", "dataset", "dataset_10k_3k_0.1.json")
    simulator_function = lt_simulator
    generate_dataset(node_file, edge_file, num_simulations, output_path, simulator_function, 0.1)
