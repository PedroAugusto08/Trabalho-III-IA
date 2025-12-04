import random
import math
from typing import List

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

    self_samples: exemplos normais (conjunto próprio)
    n_detectors: número desejado de detectores
    radius: raio de reconhecimento
    lower_bounds, upper_bounds: limites do espaço de entrada
    max_trials: limite de tentativas de geração de detectores
    """
    detectors: List[List[float]] = []
    dim = len(self_samples[0])
    trials = 0

    while len(detectors) < n_detectors and trials < max_trials:
        cand = generate_random_vector(dim, lower_bounds, upper_bounds)

        # Verifica se o candidato é "não-próprio" (distante de todos os self)
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
            return True  # anômalo
    return False  # normal

if __name__ == "__main__":
    random.seed(42)

    # Conjunto de padrões próprios (normais), perto de (0.2, 0.2)
    self_samples = [
        [0.18, 0.22],
        [0.21, 0.19],
        [0.19, 0.18],
        [0.22, 0.21],
        [0.20, 0.23],
    ]

    lower = [0.0, 0.0]
    upper = [1.0, 1.0]

    # Treina detectores
    detectors = negative_selection_train(
        self_samples=self_samples,
        n_detectors=20,
        radius=0.15,
        lower_bounds=lower,
        upper_bounds=upper,
        max_trials=5000
    )

    print(f"Número de detectores gerados: {len(detectors)}")

    # Pontos de teste
    test_points = [
        [0.2, 0.2],   # normal
        [0.25, 0.15], # provavelmente normal
        [0.8, 0.8],   # anômalo
        [0.6, 0.1],   # anômalo
    ]

    for x in test_points:
        is_anomaly = negative_selection_detect(x, detectors, radius=0.15)
        status = "ANÔMALO" if is_anomaly else "normal"
        print(f"{x} -> {status}")