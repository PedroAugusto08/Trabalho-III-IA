import random
import math
from typing import List, Callable, Tuple

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Distância euclidiana entre dois vetores."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def generate_random_vector(dim: int, lower_bounds: List[float], upper_bounds: List[float]) -> List[float]:
    """Gera um vetor aleatório dentro dos limites especificados."""
    return [
        random.uniform(lower_bounds[i], upper_bounds[i])
        for i in range(dim)
    ]

def negative_selection_train(
    self_samples: List[List[float]],
    n_detectors: int,
    radius: float,
    lower_bounds: List[float],
    upper_bounds: List[float],
    max_trials: int = 10000
) -> List[List[float]]:
    """
    Treina detectores usando o algoritmo de seleção negativa.
    """
    detectors: List[List[float]] = []
    dim = len(self_samples[0])
    trials = 0

    while len(detectors) < n_detectors and trials < max_trials:
        cand = generate_random_vector(dim, lower_bounds, upper_bounds)

        if all(euclidean_distance(cand, s) >= radius for s in self_samples):
            detectors.append(cand)

        trials += 1

    return detectors

def negative_selection_detect(sample: List[float], detectors: List[List[float]], radius: float) -> bool:
    """
    Verifica se um padrão é anômalo.
    Retorna True se for ANÔMALO, False se for NORMAL.
    """
    for d in detectors:
        if euclidean_distance(sample, d) < radius:
            return True
    return False

if __name__ == "__main__":
    random.seed(42)

    self_samples = [
        [0.18, 0.22],
        [0.21, 0.19],
        [0.19, 0.18],
        [0.22, 0.21],
        [0.20, 0.23],
    ]

    lower = [0.0, 0.0]
    upper = [1.0, 1.0]

    detectors = negative_selection_train(
        self_samples=self_samples,
        n_detectors=20,
        radius=0.15,
        lower_bounds=lower,
        upper_bounds=upper,
        max_trials=5000
    )

    print(f"Número de detectores gerados: {len(detectors)}")

    test_points = [
        [0.2, 0.2],
        [0.25, 0.15],
        [0.8, 0.8],
        [0.6, 0.1],
    ]

    for x in test_points:
        is_anomaly = negative_selection_detect(x, detectors, radius=0.15)
        status = "ANÔMALO" if is_anomaly else "normal"
        print(f"{x} -> {status}")

    def clonalg_optimize(
        f: Callable[[List[float]], float],
        dim: int,
        lower_bounds: List[float],
        upper_bounds: List[float],
        population_size: int = 20,
        n_selected: int = 5,
        beta: int = 5,
        n_random: int = 2,
        n_generations: int = 50,
        minimize: bool = True
    ) -> Tuple[List[float], float]:

        def random_antibody() -> List[float]:
            return [
                random.uniform(lower_bounds[i], upper_bounds[i])
                for i in range(dim)
            ]

        def affinity(value: float) -> float:
            return -value if minimize else value

        population: List[List[float]] = [random_antibody() for _ in range(population_size)]

        for gen in range(n_generations):
            values = [f(ab) for ab in population]
            affs = [affinity(v) for v in values]

            ranked = sorted(
                zip(population, values, affs),
                key=lambda x: x[2],
                reverse=True
            )

            selected = ranked[:n_selected]
            mutated_clones: List[List[float]] = []

            for rank, (ab, val, aff) in enumerate(selected, start=1):
                n_clones = int(beta * population_size / rank)
                if n_clones < 1:
                    n_clones = 1

                m_max = 0.3
                m_min = 0.01
                frac = (rank - 1) / max(1, n_selected - 1)
                mutation_rate = m_min + (m_max - m_min) * frac

                for _ in range(n_clones):
                    new_ab: List[float] = []
                    for i, x in enumerate(ab):
                        span = upper_bounds[i] - lower_bounds[i]
                        x_new = x + random.gauss(0, mutation_rate * span)
                        x_new = max(lower_bounds[i], min(upper_bounds[i], x_new))
                        new_ab.append(x_new)
                    mutated_clones.append(new_ab)

            clone_values = [f(ab) for ab in mutated_clones]
            clone_affs = [affinity(v) for v in clone_values]

            combined = ranked + list(zip(mutated_clones, clone_values, clone_affs))

            for _ in range(n_random):
                ab = random_antibody()
                v = f(ab)
                a = affinity(v)
                combined.append((ab, v, a))

            combined_sorted = sorted(combined, key=lambda x: x[2], reverse=True)
            population = [ab for ab, v, a in combined_sorted[:population_size]]

            best_val_gen = min([f(ab) for ab in population])
            print(f"Geração {gen+1}: melhor f(x) = {best_val_gen:.6f}")

        if minimize:
            best = min(population, key=f)
        else:
            best = max(population, key=f)

        return best, f(best)

    random.seed(123)

    def sphere(x: List[float]) -> float:
        return sum(xi ** 2 for xi in x)

    best, value = clonalg_optimize(
        f=sphere,
        dim=2,
        lower_bounds=[-5.0, -5.0],
        upper_bounds=[5.0, 5.0],
        population_size=30,
        n_selected=5,
        beta=5,
        n_random=2,
        n_generations=50,
        minimize=True
    )

    print("\nMelhor solução encontrada:", best)
    print("Valor da função na melhor solução:", value)