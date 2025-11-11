import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Carregar dataset original
df = pd.read_csv("diabetes.csv")  # coloque o nome correto aqui

# 2. Colunas selecionadas
colunas_utilizadas = [
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
    "hba1c",
    "diagnosed_diabetes"
]

# 3. Filtrar o dataframe
df = df[colunas_utilizadas]

# 4. Separar features e target
X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

# 5. Normalizar os dados
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# 6. Transformar para DataFrame novamente
df_normalizado = pd.DataFrame(X_normalizado, columns=X.columns)
df_normalizado["diagnosed_diabetes"] = y.values

# 7. Salvar dataset final
df_normalizado.to_csv("diabetes_normalizado.csv", index=False)

print("✅ Dataset normalizado salvo como diabetes_normalizado.csv")
print("✅ Shape final:", df_normalizado.shape)
