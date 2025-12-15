import random
from typing import Callable, List, Tuple

#debug
print("clonalg.py carregado")

Mask = List[int]

def clonalg_feature_selection(
    fitness_fn: Callable[[Mask], float],
    d: int,
    population_size: int = 20,
    n_selected: int = 5,
    beta: int = 5,
    n_random: int = 2,
    n_generations: int = 50,
    p_on: float = 0.3,
    seed: int = 42
) -> Tuple[Mask, float]:


    # Inicialização
    def random_antibody() -> Mask:
        m = [1 if random.random() < p_on else 0 for _ in range(d)]
        if sum(m) == 0:
            m[random.randrange(d)] = 1
        return m

    def affinity(value: float) -> float:
        # Menor fitness = maior afinidade
        return -value

    population: List[Mask] = [random_antibody() for _ in range(population_size)]

    # Loop evolutivo
    for gen in range(n_generations):

        values = [fitness_fn(ab) for ab in population]
        affs = [affinity(v) for v in values]

        ranked = sorted(
            zip(population, values, affs),
            key=lambda x: x[2],
            reverse=True
        )

        selected = ranked[:n_selected]
        mutated_clones: List[Mask] = []

        # Clonagem + Hipermutação
        for rank, (ab, val, aff) in enumerate(selected, start=1):

            n_clones = max(1, int(beta * population_size / rank))

            # Taxa de mutação proporcional ao rank
            m_max = 0.3
            m_min = 0.01
            frac = (rank - 1) / max(1, n_selected - 1)
            mutation_rate = m_min + (m_max - m_min) * frac

            for _ in range(n_clones):
                clone = ab.copy()

                for i in range(d):
                    if random.random() < mutation_rate:
                        clone[i] = 1 - clone[i]  # bit-flip

                if sum(clone) == 0:
                    clone[random.randrange(d)] = 1

                mutated_clones.append(clone)

        # Avaliação dos clones
        clone_values = [fitness_fn(ab) for ab in mutated_clones]
        clone_affs = [affinity(v) for v in clone_values]

        combined = ranked + list(zip(mutated_clones, clone_values, clone_affs))


        # Diversidade (novos anticorpos)
        for _ in range(n_random):
            ab = random_antibody()
            v = fitness_fn(ab)
            a = affinity(v)
            combined.append((ab, v, a))

        # Seleção para próxima geração
        combined_sorted = sorted(combined, key=lambda x: x[2], reverse=True)
        population = [ab for ab, v, a in combined_sorted[:population_size]]

        best_val = min(v for _, v, _ in combined_sorted[:population_size])
        print(f"Geração {gen+1}: melhor fitness = {best_val:.6f}")

    best = min(population, key=fitness_fn)
    return best, fitness_fn(best)