import networkx as nx
import numpy as np
from scipy.io import mmread

def load_graph(file_path):
    if file_path.endswith(".mtx"):
        adj_matrix = mmread(file_path).toarray()
        return nx.from_numpy_array(adj_matrix)
    elif file_path.endswith(".gml"):
        return nx.read_gml(file_path)
    else:
        raise ValueError("Unsupported file format. Use .mtx or .gml")

def print_result(modularity, method):
    print(f"\n✅ [{method}] Final Modularity: {modularity:.4f}")
