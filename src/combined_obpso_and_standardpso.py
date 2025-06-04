import numpy as np
import networkx as nx
from numpy.random import rand, randint
from sklearn.metrics import normalized_mutual_info_score as nmi
from networkx.algorithms import community

class Standard_PSO:
    def __init__(self, graph, population_size=200, max_iter=500):
        self.graph = graph
        self.n = len(graph.nodes)
        self.population_size = population_size
        self.max_iter = max_iter
        self.omega = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.array([randint(0, self.n, self.n) for _ in range(self.population_size)])

    def calculate_modularity(self, community_assignment):
        return nx.community.modularity(self.graph, self.get_communities(community_assignment))

    def get_communities(self, community_assignment):
        communities = {}
        for node, com in enumerate(community_assignment):
            communities.setdefault(com, []).append(node)
        return list(communities.values())

    def run(self):
        velocities = np.zeros((self.population_size, self.n))
        personal_best = self.population.copy()
        personal_best_scores = np.array([self.calculate_modularity(ind) for ind in self.population])
        global_best = personal_best[np.argmax(personal_best_scores)]
        global_best_score = max(personal_best_scores)

        for iteration in range(self.max_iter):
            r1, r2 = rand(self.population_size, self.n), rand(self.population_size, self.n)
            velocities = (self.omega * velocities + self.c1 * r1 * (personal_best - self.population) 
                          + self.c2 * r2 * (global_best - self.population))
            self.population = np.round(self.population + velocities).astype(int) % self.n
            self.population = np.clip(self.population, 0, self.n - 1)

            fitness = np.array([self.calculate_modularity(ind) for ind in self.population])
            update_mask = fitness > personal_best_scores
            personal_best[update_mask] = self.population[update_mask]
            personal_best_scores[update_mask] = fitness[update_mask]

            if max(fitness) > global_best_score:
                global_best_score = max(fitness)
                global_best = self.population[np.argmax(fitness)]

            if iteration % 10 == 0:
                print(f"Iteration {iteration+1}/{self.max_iter}, Best Modularity: {global_best_score}")
        
        return global_best


class OBPSO_AIW:
    def __init__(self, graph, population_size=200, max_iter=500):
        self.graph = graph
        self.n = len(graph.nodes)
        self.population_size = population_size
        self.max_iter = max_iter
        self.omega_max = 0.9
        self.omega_min = 0.5
        self.c1 = 2.8
        self.c2 = 2.8
        self.mutation_rate = 0.4
        self.crossover_rate = 0.7
        self.population = self.initialize_population()

    def initialize_population(self):
        initial_partition = community.louvain_communities(self.graph)
        community_map = {node: i for i, comm in enumerate(initial_partition) for node in comm}
        return np.array([np.array([community_map[node] for node in self.graph.nodes]) for _ in range(self.population_size)])

    def calculate_modularity(self, community_assignment):
        return nx.community.modularity(self.graph, self.get_communities(community_assignment))

    def get_communities(self, community_assignment):
        communities = {}
        for node, com in enumerate(community_assignment):
            communities.setdefault(com, []).append(node)
        return list(communities.values())

    def crossover(self, population):
        new_population = []
        for _ in range(self.population_size // 2):
            if rand() < self.crossover_rate:
                parent1, parent2 = population[randint(0, self.population_size, 2)]
                alpha = rand()
                child1 = np.round(alpha * parent1 + (1 - alpha) * parent2).astype(int) % self.n
                child2 = np.round((1 - alpha) * parent1 + alpha * parent2).astype(int) % self.n
                new_population.extend([child1, child2])
            else:
                new_population.extend(population[randint(0, self.population_size, 2)])
        return np.array(new_population)

    def mutate(self, population, iteration):
        dynamic_mutation = self.mutation_rate * (1 - iteration / self.max_iter)
        for individual in population:
            if rand() < dynamic_mutation:
                indices = randint(0, self.n, size=3)
                individual[indices[0]], individual[indices[1]], individual[indices[2]] = (
                    individual[indices[2]], individual[indices[0]], individual[indices[1]]
                )
        return population

    def merge_small_communities(self, population):
        for individual in population:
            unique_communities, counts = np.unique(individual, return_counts=True)
            small_communities = unique_communities[counts < 4]
            for com in small_communities:
                larger_com = unique_communities[np.argmax(counts)]
                individual[individual == com] = larger_com
        return population

    def run(self):
        velocities = np.zeros((self.population_size, self.n))
        personal_best = self.population.copy()
        personal_best_scores = np.array([self.calculate_modularity(ind) for ind in self.population])
        global_best = personal_best[np.argmax(personal_best_scores)]
        global_best_score = max(personal_best_scores)

        for iteration in range(self.max_iter):
            omega = max(self.omega_min, self.omega_max - (global_best_score / 2))
            r1, r2 = rand(self.population_size, self.n), rand(self.population_size, self.n)
            velocities = omega * velocities + self.c1 * r1 * (personal_best - self.population) + self.c2 * r2 * (global_best - self.population)
            self.population = np.round(self.population + velocities).astype(int) % self.n
            self.population = np.clip(self.population, 0, self.n - 1)

            self.population = self.merge_small_communities(self.population)
            self.population = self.mutate(self.population, iteration)
            self.population = self.crossover(self.population)

            fitness = np.array([self.calculate_modularity(ind) for ind in self.population])
            update_mask = fitness > personal_best_scores
            personal_best[update_mask] = self.population[update_mask]
            personal_best_scores[update_mask] = fitness[update_mask]

            if max(fitness) > global_best_score:
                global_best_score = max(fitness)
                global_best = self.population[np.argmax(fitness)]

            if iteration % 10 == 0:
                print(f"Iteration {iteration+1}/{self.max_iter}, Best Modularity: {global_best_score}")
        
        return global_best

if __name__ == "__main__":
    from scipy.io import mmread
    adj_matrix = mmread(r"Path to your dataset .mtx files only, if file source not matching kindly update the __name__ code accordingly").toarray() 
    print("Running Standard PSO...")
    G = nx.from_numpy_array(adj_matrix)
    standard_pso = Standard_PSO(G)
    print("Final Modularity:", standard_pso.calculate_modularity(standard_pso.run()))
    
    print("\nRunning OBPSO-AIW...")
    obpso = OBPSO_AIW(G)
    print("Final Modularity:", obpso.calculate_modularity(obpso.run()))
    
