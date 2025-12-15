import random
import math
from typing import Callable, List, Tuple

Mask = List[int]

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pso_feature_selection(
    fitness_fn: Callable[[Mask], float],
    d: int,
    n_particles: int = 30,
    n_iterations: int = 50,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    p_on: float = 0.3,
    seed: int | None = None
) -> Tuple[Mask, float]:

    if seed is not None:
        random.seed(seed)

    # Inicialização
    positions: List[Mask] = []
    velocities: List[List[float]] = []

    for _ in range(n_particles):
        pos = [1 if random.random() < p_on else 0 for _ in range(d)]
        if sum(pos) == 0:
            pos[random.randrange(d)] = 1

        vel = [random.uniform(-1, 1) for _ in range(d)]

        positions.append(pos)
        velocities.append(vel)

    pbest = [p.copy() for p in positions]
    pbest_val = [fitness_fn(p) for p in positions]

    gbest_idx = min(range(n_particles), key=lambda i: pbest_val[i])
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    # Loop principal
    for it in range(n_iterations):
        for i in range(n_particles):
            for j in range(d):
                r1 = random.random()
                r2 = random.random()

                velocities[i][j] = (
                    w * velocities[i][j]
                    + c1 * r1 * (pbest[i][j] - positions[i][j])
                    + c2 * r2 * (gbest[j] - positions[i][j])
                )

                prob = sigmoid(velocities[i][j])
                positions[i][j] = 1 if random.random() < prob else 0

            if sum(positions[i]) == 0:
                positions[i][random.randrange(d)] = 1

            val = fitness_fn(positions[i])

            if val < pbest_val[i]:
                pbest[i] = positions[i].copy()
                pbest_val[i] = val

                if val < gbest_val:
                    gbest = positions[i].copy()
                    gbest_val = val

        print(f"Iteração {it+1}: melhor fitness = {gbest_val:.6f}")

    return gbest, gbest_val