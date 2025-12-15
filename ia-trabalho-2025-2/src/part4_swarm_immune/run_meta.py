from sklearn.datasets import load_breast_cancer

from fitness import make_fitness
from clonalg import clonalg_feature_selection
from pso import pso_feature_selection

# Dataset
X, y = load_breast_cancer(return_X_y=True)
d = X.shape[1]

# Fitness (MESMA para os dois)
fitness_fn = make_fitness(X, y, alpha=0.9, seed=42)

print("\n=== CLONALG ===")
best_clon, fit_clon = clonalg_feature_selection(
    fitness_fn=fitness_fn,
    d=d,
    population_size=30,
    n_generations=20,
    seed=42
)

print("\n=== PSO ===")
best_pso, fit_pso = pso_feature_selection(
    fitness_fn=fitness_fn,
    d=d,
    n_particles=30,
    n_iterations=20,
    seed=42
)

print("\n=== RESULTADOS FINAIS ===")
print(f"CLONALG -> fitness: {fit_clon:.6f} | features: {sum(best_clon)}")
print(f"PSO     -> fitness: {fit_pso:.6f} | features: {sum(best_pso)}")