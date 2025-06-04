import numpy as np
import networkx as nx
from numpy.random import rand, randint

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
                print(f"[Standard PSO] Iteration {iteration+1}/{self.max_iter}, Best Modularity: {global_best_score:.4f}")

        return global_best
