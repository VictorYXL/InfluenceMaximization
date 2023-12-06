import os
import networkx as nx
import pandas as pd

def load_graph_from_tsv(graph_dir: str) -> nx.DiGraph:
    """Load a directed graph with attributes from TSV files in the specified directory.

    Args:
        graph_dir (str): The directory containing 'node.tsv' and 'edge.tsv' files.

    Returns:
        nx.DiGraph: A directed graph with nodes and edges loaded from the TSV files, including their attributes.
    """
    node_file = os.path.join(graph_dir, "node.tsv")
    edge_file = os.path.join(graph_dir, "edge.tsv")

    # Read the node file using pandas
    node_df = pd.read_csv(node_file, sep='\t')

    # Create a networkx graph and add nodes with their attributes
    graph = nx.DiGraph()
    for _, row in node_df.iterrows():
        node_id = int(row["node"])
        node_attrs = {attr: value for attr, value in row.items() if attr not in ["node"]}
        graph.add_node(node_id, **node_attrs)

    # Read the edge file using pandas
    edge_df = pd.read_csv(edge_file, sep='\t')

    # Add edges to the networkx graph
    for _, row in edge_df.iterrows():
        src = int(row["src_node"])
        dst = int(row["dst_node"])
        weight = float(row['weight'])
        graph.add_edge(src, dst, weight=weight)
    return graph


if __name__ == "__main__":
    graph = load_graph_from_tsv(os.path.join("graph", "example"))
    print(graph)