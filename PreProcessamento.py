import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset
data = pd.read_csv("qualidade_sono_dataset.csv")

# Converter a coluna 'Qualidade do Sono' para valores numéricos
qualidade_map = {"ótimo": 3, "bom": 2, "médio": 1, "ruim": 0}
data["Qualidade do Sono"] = data["Qualidade do Sono"].map(qualidade_map)

# Separar as variáveis independentes (X) e o alvo (y)
X = data.drop(columns=["Qualidade do Sono"])
y = data["Qualidade do Sono"]

# Normalizar os dados numéricos (X)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

print("Pré-processamento concluído!")