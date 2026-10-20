
##############################################################################

# TRABALHO DE MACHINE LEARNING - SPOTIFY TRACKS DATASET
# Modelo utilizado: MLP (Multilayer Perceptron)
# SOLUÇÃO DO PROBLEMA: CLASSIFICAÇÃO DE GÊNEROS MUSICAIS

##############################################################################

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
print("\nQuantidade por gênero: \n", df["track_genre"].value_counts())

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


generos_selecionados = [
    "hip-hop", "classical", "jazz",
    "country", "reggae", "forro",
    "metal", "techno"
]

# Filtrar somente as músicas dos gêneros escolhidos
df_filtrado = df[
    df["track_genre"].isin(generos_selecionados)
].copy()

# Separar as características e os gêneros correspondentes
colum_genre = df_filtrado["track_genre"].copy()

# Converter os gêneros para números
label_encoder = LabelEncoder()

# Valores de entrada: Features (X) e saída esperada: Gênero (y)
X = df_filtrado[colunas].copy()  # Features
y = label_encoder.fit_transform(colum_genre)  # Coluna com saída esperada

print("\nDimensões de X:", X.shape)
print("Quantidade de gêneros:", len(label_encoder.classes_))
print("Gêneros encontrados:", list(label_encoder.classes_))


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

# Analisando se as dimensões para treino e teste foram dividadas corretamente
print("\nTreino:", X_train.shape)
print("Teste:", X_test.shape)

##############################################################################
# 4. TREINAMENTO USANDO MODELO MLP
##############################################################################

# Determina o numero máximo de iterações do MLPClassifier
max_iter = 200

# Pipelina para pré-processamento e treinamento do modelo MLP
# Organiza em uma sequência de etapas que serão executadas automaticamente,
# sendo um transformador ou um estimador

modelo = Pipeline(
    [
        # Substitui valores ausentes pela mediana de cada coluna
        ("imputador", SimpleImputer(strategy="median")),
        # Escalonamento dos dados (normalização) entre 0 e 1
        ("escalonador", MinMaxScaler()),
        ("mlp", MLPClassifier(                           # Treinamento do modelo MLP
            hidden_layer_sizes=(100, 50),                # Duas camadas ocultas de neurônios
            activation="relu",                           # Função de ativação 
            solver="adam",                               # Algoritmo de otimização (Ajusta os pesos da rede neural)
            max_iter=max_iter,                           # Máximo de iterações para o treinamento
            random_state=42,                             # Aleatoriedade para reprodutibilidade
            early_stopping=True,                         # Parada antecipada (se não houver melhora na validação)
            validation_fraction=0.1,                     # Porcentagem de dados para validação durante o treinamento
            n_iter_no_change=15                          # Número de iterações sem melhoria toleradas 
        ))
    ]
)
modelo.fit(X_train, y_train) # Treinamento do modelo MLP (dados de treino)
mlp = modelo.named_steps["mlp"] # Acessando o modelo MLP no pipeline para obter informações sobre o treinamento


# Consultando os dados do treinamento do modelo MLP
print("Épocas executadas:", mlp.n_iter_, "vezes de", max_iter)  # Passagem completa do treinamento (n°_iterações)
print("Perda final:", round(mlp.loss_, 4) * 100, "%") #Função de perda final
print("Melhor pontuação na validação:", round(mlp.best_validation_score_, 4) * 100, "%")


print("\nTREINAMENTO DO MODELO MLP CONCLUIDO!")


##############################################################################
# 5. AVALIACAO DOS RESULTADOS
##############################################################################

print("\n5. Avaliação dos resultados")


y_predict = modelo.predict(X_test)

print("Acurácia: ", round(accuracy_score(y_test, y_predict) * 100, 4), "%")

print("Precisão com igualdade de pesos: ", # Confiabilidade do modelo em classificar corretamento os gêneros
      round(precision_score(
          y_test, y_predict, average="macro", zero_division=0) * 100, 4
      ), "%")


print("Recall com igualdade de pesos", # Capacidade do modelo em identificar corretamente os gêneros
      round(recall_score(
          y_test, y_predict, average="macro", zero_division=0) * 100, 4
      ), "%")

print("F1_score com igualdade de pesos:", # Combinação da precisão e recall (faz uma média harmônica entre eles)
      round(f1_score(
          y_test, y_predict, average="macro", zero_division=0) * 100, 4
      ), "%")



classes = modelo.named_steps["mlp"].classes_ # Obtendo os gêneros previstos pelo modelo MLP
# Invertendo a transformação usando LabelEncoder 
print("\nMatriz de confusão:\n")
print(confusion_matrix(y_test, y_predict, labels=classes))

# Relatório de classificação dos resultados do modelo MLP
print("\n\nRESULTADOS POR GÊNERO!")
print(classification_report(
    y_test,
    y_predict,
    labels=classes,
    target_names=label_encoder.inverse_transform(classes),
    zero_division=0
))


##############################################################################
# SALVANDO AS PREVISOES EM UM ARQUIVO .csv
##############################################################################

# Criando um DataFrame com os resultados com o genêro real e o gênero previsto.
# Antes disto, é necessário inverter a transformação usando LabelEncoder,
# para que os gêneros voltem a ser Strings e seja possível ler os resultados,
# dentro do arquivo .csv

df_resultados = pd.DataFrame({
    "genero_real": label_encoder.inverse_transform(y_test),
    "genero_previsto": label_encoder.inverse_transform(y_predict)
})

df_resultados.to_csv("spotify_mlp_resultados.csv", index=False)