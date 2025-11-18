import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o conjunto de dados de diabetes
df = pd.read_csv("ia-trabalho-2025-2/data/processed/diabetes_filtrado.csv")

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

# Pré-processar os dados
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

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Padronizar os dados (muito importante para PCA e SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=0.95)  # Reduzindo para 2 componentes principais
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Treinar o modelo SVM
start_time_train = time.time()
svm = SVC(kernel='linear', probability=True, random_state=42)  # random_state para reprodutibilidade
svm.fit(X_train_pca, y_train)
end_time_train = time.time()

# Fazer previsões no conjunto de teste
start_time_pred = time.time()
y_pred = svm.predict(X_test_pca)
y_pred_proba = svm.predict_proba(X_test_pca)[:, 1]  # Probabilidades para ROC-AUC
end_time_pred = time.time()

# Avaliar as métricas do modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("MÉTRICAS DE AVALIAÇÃO - SVM")
print("=" * 60)
print(f"Acurácia:  {accuracy:.4f}")
print(f"Precisão:  {precision:.4f} (macro)")
print(f"Recall:    {recall:.4f} (macro)")
print(f"F1-Score:  {f1:.4f} (macro)")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"\nTempo de treinamento: {end_time_train - start_time_train:.4f} segundos")
print(f"Tempo de predição: {end_time_pred - start_time_pred:.4f} segundos")
print(f"Tempo total (treino + predição): {(end_time_train - start_time_train) + (end_time_pred - start_time_pred):.4f} segundos")
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão - SVM')
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
plt.title('Curva ROC - SVM')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import pickle
# Save the model to a file
with open('svm.model', 'wb') as file:
  pickle.dump(svm, file)

# Load the model from the file
with open('svm.model', 'rb') as file:
  svm = pickle.load(file)
  
  
# treinar o model com cross-validation
from sklearn.model_selection import cross_val_score
print("\n" + "=" * 60)
print("AVALIAÇÃO COM CROSS-VALIDATION")
print("=" * 60)
start_time_cv = time.time()
svm_cv = SVC(kernel='linear', probability=True, random_state=42)
scores = cross_val_score(svm_cv, X_train_pca, y_train, cv=5)
end_time_cv = time.time()
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores):.4f}')
print(f'Tempo de cross-validation: {end_time_cv - start_time_cv:.2f} segundos')

# fazer as predicoes com o modelo treinado com cross-validation
y_pred_cv = svm_cv.fit(X_train_pca, y_train).predict(X_test_pca)
y_pred_proba_cv = svm_cv.predict_proba(X_test_pca)[:, 1]
accuracy_cv = accuracy_score(y_test, y_pred_cv)
precision_cv = precision_score(y_test, y_pred_cv, average='macro', zero_division=0)
recall_cv = recall_score(y_test, y_pred_cv, average='macro', zero_division=0)
f1_cv = f1_score(y_test, y_pred_cv, average='macro', zero_division=0)
roc_auc_cv = roc_auc_score(y_test, y_pred_proba_cv)

print(f"\nMétricas com cross-validation:")
print(f"Acurácia:  {accuracy_cv:.4f}")
print(f"Precisão:  {precision_cv:.4f} (macro)")
print(f"Recall:    {recall_cv:.4f} (macro)")
print(f"F1-Score:  {f1_cv:.4f} (macro)")
print(f"ROC-AUC:   {roc_auc_cv:.4f}")
