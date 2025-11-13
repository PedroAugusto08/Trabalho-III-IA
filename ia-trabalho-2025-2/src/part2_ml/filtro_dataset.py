import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Carregar dataset original
df = pd.read_csv("ia-trabalho-2025-2/data/raw/diabetes_dataset.csv")

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

df.to_csv("ia-trabalho-2025-2/data/processed/diabetes_filtrado.csv", index=False)

print("✅ Dataset filtrado salvo como diabetes_filtrado.csv")
print("✅ Shape final:", df.shape)
