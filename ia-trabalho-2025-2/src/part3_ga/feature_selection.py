import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ga import GeneticAlgorithm

# Carregar o dataset
df = pd.read_csv("ia-trabalho-2025-2/data/diabetes_dataset.csv")

# Seleção das features e target
features = [
    "age",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "physical_activity_minutes_per_week",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c"
]
X = df[features].values
y = df["diagnosed_diabetes"].values

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Padronização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# Fitness: Avaliação do subconjunto de features
# ===============================
def fitness(ind):
    # Indivíduo: vetor binário indicando quais features usar
    if np.sum(ind) == 0:
        return 0  # Não pode usar zero features
    selected_idx = np.where(ind == 1)[0]
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]
    # Treina KNN simples (rápido)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_sel, y_train)
    y_pred = knn.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    # Penaliza número de features (menos é melhor)
    penalty = 0.01 * np.sum(ind)  # Ajuste conforme necessário
    return acc - penalty

# ===============================
# Executa o AG para seleção de atributos
# ===============================
if __name__ == "__main__":
    n_genes = len(features)
    ag = GeneticAlgorithm(
        fitness_func=fitness,
        n_genes=n_genes,
        pop_size=30,
        n_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism=True
    )
    best_ind, best_fit, history = ag.run()
    print("\nMelhor indivíduo (seleção de features):", best_ind)
    print("Features selecionadas:", [features[i] for i in range(n_genes) if best_ind[i] == 1])
    print("Fitness do melhor indivíduo:", best_fit)
    print("Histórico de fitness por geração:", history)
