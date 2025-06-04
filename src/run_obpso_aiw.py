from src.obpso_aiw import OBPSO_AIW
from src.utils import load_graph, print_result

def main():
    path = input("Enter path to dataset (e.g., datasets/soc-dolphins.mtx): ")
    G = load_graph(path)
    print("Running OBPSO-AIW...")
    algo = OBPSO_AIW(G)
    result = algo.run()
    modularity = algo.calculate_modularity(result)
    print_result(modularity, "OBPSO-AIW")

if __name__ == "__main__":
    main()
