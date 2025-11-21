import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # <- correção do import
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("ia-trabalho-2025-2/data/diabetes_dataset.csv")

# ===== Menu interativo para escolher o tamanho do dataset =====
print("=" * 60)
print("SELEÇÃO DO TAMANHO DO DATASET")
print("=" * 60)
print(f"Dataset completo tem {len(df)} linhas")
print("\nEscolha quantas linhas deseja usar:")
print("1 - 10.000 linhas")
print("2 - 30.000 linhas")
print("3 - 50.000 linhas")
print("4 - 70.000 linhas")
print("5 - 100.000 linhas (completo)")
print("=" * 60)

while True:
    escolha = input("Digite sua escolha (1-5): ").strip()
    if escolha == "1":
        n_samples = 10000
        break
    elif escolha == "2":
        n_samples = 30000
        break
    elif escolha == "3":
        n_samples = 50000
        break
    elif escolha == "4":
        n_samples = 70000
        break
    elif escolha == "5":
        n_samples = 100000
        break
    else:
        print("Opção inválida! Digite um número entre 1 e 5.")

# Shuffle e seleção das linhas
df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
print(f"\n✓ Dataset preparado com {n_samples} linhas (embaralhadas aleatoriamente)\n")

# Inspecionar os dados
print(df.head())

# Pré-processar os dados (ajuste as colunas conforme necessário)
X = df[
    [
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
].values

y = df["diagnosed_diabetes"].values


# Dividir os dados em treino e teste (com estratificação e semente fixa)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===== Escolha do melhor k SEM olhar para o teste (evita vazamento) =====
print("\n" + "=" * 60)
print("BUSCA DO MELHOR K")
print("=" * 60)
start_time_cv = time.time()

k_values = list(range(1, 31))
scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for k in k_values:
    # Escalonamento dentro do CV via pipeline (evita vazamento dentro das dobras)
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, n_jobs=-1))  # n_jobs=-1 usa todos os cores
    score = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=-1).mean()
    scores.append(score)

# Curva do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_values, scores)
plt.xlabel("K Values")
plt.ylabel("Accuracy (CV)")
plt.title("KNN Classifier Accuracy for Different K Values")
plt.xticks(k_values)
plt.grid (True)
plt.show()

end_time_cv = time.time()
best_k = k_values[int(np.argmax(scores))]
print(f"Melhor k (CV no treino): {best_k}")
print(f"Tempo de busca do melhor k: {end_time_cv - start_time_cv:.2f} segundos")

# ===== Treino final no treino e avaliação no teste =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start_time_train = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
end_time_train = time.time()

# Predição e métricas
start_time_pred = time.time()
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Probabilidades para ROC-AUC
end_time_pred = time.time()

accuracy = accuracy_score(y_test, y_pred)
# 'macro' funciona bem para multi-classe; em binário pode usar average='binary'
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("MÉTRICAS DE AVALIAÇÃO - KNN")
print("=" * 60)
print(f"Acurácia:  {accuracy:.4f}")
print(f"Precisão:  {precision:.4f} (macro)")
print(f"Recall:    {recall:.4f} (macro)")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"\nTempo de treinamento: {end_time_train - start_time_train:.4f} segundos")
print(f"Tempo de predição: {end_time_pred - start_time_pred:.4f} segundos")
print(f"Tempo total (busca k + treino + predição): {(end_time_cv - start_time_cv) + (end_time_train - start_time_train) + (end_time_pred - start_time_pred):.2f} segundos")
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão - KNN')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.tight_layout()
plt.show()

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - KNN')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
