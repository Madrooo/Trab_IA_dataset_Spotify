
##############################################################################
# TRABALHO DE MACHINE LEARNING - SPOTIFY TRACKS DATASET
# Modelo utilizado: MLP (Multilayer Perceptron)
##############################################################################

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, 
    confusion_matrix, classification_report
)

##############################################################################
# 1. LEITURA E ANALISE DOS DADOS
##############################################################################

df = pd.read_csv("dataset.csv", sep=",")

print("\nPrimeiras linhas: \n", df.head())
print("\nTipos das colunas: \n", df.dtypes)
print("\nDimensões da base: ", df.shape)
print("\nEstatísticas: \n", df.describe())
print("\nQuantidade por genero: \n", df["track_genre"].value_counts())

##############################################################################
# 2. PRÉ-PROCESSAMENTO
##############################################################################

colunas = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]

# Valores de entrada:
X = df[colunas].copy() #Features
y = df["track_genre"].copy() #Coluna com saída esperada

# Copiando dados e salvando em um arquivo .csv
df_mlp = X.copy()
df_mlp["track_genre"] = y

df_mlp.to_csv("spotify_mlp_dados.csv", index=False)

print("\nVerificando se há valores ausentes em X:\n", X.isnull().sum())

##############################################################################
# 3. DIVISAO ENTRE TREINO E TESTE
##############################################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTreino:", X_train.shape)
print("Teste:", X_test.shape)

##############################################################################
# 4. TREINAMENTO USANDO MODELO MLP
##############################################################################

# Determina o numero máximo de iterações do MLPClassifier
max_iter = 100

modelo = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("escalonador", MinMaxScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15
    ))
])

modelo.fit(X_train, y_train)
mlp = modelo.named_steps["mlp"]

print("Épocas executadas:", mlp.n_iter_)
print("Perda final:", round(mlp.loss_, 4))
print("Melhor pontuação na validação:", round(mlp.best_validation_score_, 4))

print("\nTreinamento Concluido!")

##############################################################################
# 5. AVALIACAO DOS RESULTADOS
##############################################################################

print("\n5. Avaliacao dos resultados")

y_predict = modelo.predict(X_test)
print("Acurácia: ", round(accuracy_score(y_test, y_predict), 4))

print("Precisão com igualdade de pesos: ", 
    round(precision_score(
        y_test, y_predict, average="macro", zero_division=0), 4
    ))


print("Recall com igualdade de pesos", 
    round(recall_score(
        y_test, y_predict, average="macro", zero_division=0), 4
    ))

print("F1_score com igualdade de pesos:", 
    round(f1_score(
        y_test, y_predict, average="macro", zero_division=0), 4
    ))

classes = modelo.named_steps["mlp"].classes_

print("\nMatriz de confusão:\n")
print(confusion_matrix(y_test, y_predict, labels=classes))

print("\n\nRESULTADOS POR GÊNERO!")
print(classification_report(
    y_test,
    y_predict,
    labels=classes,
    zero_division=0
))

##############################################################################
# SALVANDO AS PREVISOES EM UM ARQUIVO .csv
##############################################################################

df_resultados = pd.DataFrame({
    "genero_real": y_test.to_numpy(),
    "genero_previsto": y_predict
})

df_resultados.to_csv("spotify_mlp_resultados.csv", index=False
)