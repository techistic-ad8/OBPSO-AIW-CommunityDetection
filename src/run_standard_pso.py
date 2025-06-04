from src.standard_pso import Standard_PSO
from src.utils import load_graph, print_result

def main():
    path = input("Enter path to dataset (e.g., datasets/soc-dolphins.mtx): ")
    G = load_graph(path)
    print("Running Standard PSO...")
    algo = Standard_PSO(G)
    result = algo.run()
    modularity = algo.calculate_modularity(result)
    print_result(modularity, "Standard PSO")

if __name__ == "__main__":
    main()
