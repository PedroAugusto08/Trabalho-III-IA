import numpy as np
import random

# ===============================
# Algoritmo Genético (AG) - Base
# ===============================

class GeneticAlgorithm:
    def __init__(self, fitness_func, n_genes, pop_size=30, n_generations=50, crossover_rate=0.8, mutation_rate=0.05, elitism=True, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.fitness_func = fitness_func
        self.n_genes = n_genes
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism

    def initialize_population(self):
        # População binária: cada indivíduo é um vetor de 0/1
        return np.random.randint(2, size=(self.pop_size, self.n_genes))

    def evaluate_population(self, population):
        return np.array([self.fitness_func(ind) for ind in population])

    def select_parents(self, population, fitness):
        # Seleção por torneio
        parents = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(self.pop_size, 2, replace=False)
            winner = population[i] if fitness[i] > fitness[j] else population[j]
            parents.append(winner)
        return np.array(parents)

    def crossover(self, parent1, parent2):
        # Cruzamento de 1 ponto
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.n_genes - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        # Mutação bit flip
        for i in range(self.n_genes):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        population = self.initialize_population()
        best_fitness = -np.inf
        best_individual = None
        history = []

        for gen in range(self.n_generations):
            fitness = self.evaluate_population(population)
            gen_best_idx = np.argmax(fitness)
            gen_best_fit = fitness[gen_best_idx]
            gen_best_ind = population[gen_best_idx].copy()
            history.append(gen_best_fit)

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_individual = gen_best_ind.copy()

            # Elitismo: mantém o melhor da geração
            new_population = []
            if self.elitism:
                new_population.append(gen_best_ind)

            # Seleção de pais
            parents = self.select_parents(population, fitness)

            # Cruzamento e mutação
            for i in range(0, self.pop_size - int(self.elitism), 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < len(parents) else parents[0]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            # Ajusta tamanho da população
            population = np.array(new_population[:self.pop_size])

            print(f"Geração {gen+1:02d} | Melhor fitness: {gen_best_fit:.4f}")

        return best_individual, best_fitness, history
