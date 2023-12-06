import os
import random
import json
from typing import Dict, List
from graph_loader import load_graph_from_tsv
from scenarios.independent_cascade.ic_simulator import ic_simulator
from scenarios.linear_threshold.lt_simulator import lt_simulator


def generate_intensity(file_path: str, edge_file_path: str, lower_bound: float, upper_bound: float):
    """Generate intensity by uniform in [lower_bound, upper_bound]"""
    edge_file = open(edge_file_path, "w")
    edge_file.write("src_node\tdst_node\tweight\n")
    for line in open(file_path).readlines():
        data = line.strip().split(" ")
        intensity = random.uniform(lower_bound, upper_bound)
        edge_file.write(f"{data[0]}\t{data[1]}\t{intensity}\n")
    edge_file.close()

def generate_threshold(file_path: str, node_file_path: str, lower_bound: float, upper_bound: float):
    """Generate threshold by uniform in [lower_bound, upper_bound]"""
    node_file = open(node_file_path, "w")
    node_file.write("node\tthreshold\n")
    nodes = set()
    for line in open(file_path).readlines():
        data = line.strip().split(" ")
        if data[0] not in nodes:
            threshold = random.uniform(lower_bound, upper_bound)
            node_file.write(f"{data[0]}\t{threshold}\n")
            nodes.add(data[0])
        if data[1] not in nodes:
            threshold = random.uniform(lower_bound, upper_bound)
            node_file.write(f"{data[1]}\t{threshold}\n")
            nodes.add(data[1])
    node_file.close()

def generate_dataset(graph_path: str, num_simulations: Dict[str, int], file_path: str, influence_function: callable, seed_ratio: float):
    graph = load_graph_from_tsv(graph_path)
    data = {
        "graph_path": graph_path,
        "influence_function": str(influence_function.__name__),
        "seed_ratio": seed_ratio
    }
    all_nodes = list(graph.nodes())
   
    for mode, num in num_simulations.items():
        data[mode] = []
        for index in range(num):
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
    num_simulations = {"training_data": 10000, "evaluation_data": 3000}
    output_path = os.path.join("data", "facebook", "dataset", "dataset_10k_3k.json")
    simulator_function = lt_simulator
    generate_dataset(graph_path, num_simulations, output_path, simulator_function, 0.05)
