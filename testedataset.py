import pandas as pd
from sklearn.model_selection import train_test_split
from os import replace
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix


##############################################################################
# 1. TESTANDO ALGUNS COMANDOS COM O DATAFRAME dataset.csv
##############################################################################

spotify_data = pd.read_csv("dataset.csv", sep=",")

"""

# APRESENTANDO O DATASET
print(spotify_data.head())
print("\n1. Quantidade de linhas e colunas: ", spotify_data.shape)
print("\n2. Informações do dataset: ", spotify_data.info())
print("\n3. Todas as colunas do dataset Spotify: ", spotify_data.columns)
print("\n4. Valores nulos: ", spotify_data.isnull().sum())
print("\n5. Calculando métricas: \n", spotify_data.describe())

resultado = spotify_data.agg(
    media = ("popularity", "mean")
).reset_index()

print("\n6. Agregando valores: \n", resultado)

"""

# ANALISE DA COLUNA "track_genre"
generos = spotify_data["track_genre"].value_counts()

print("Quantidade de generos:\n", spotify_data["track_genre"].nunique())
print("\nQuantidade de musicas por genero:\n", generos)

spotify_data["track_genre"].value_counts().head(15).plot(
    kind="bar",
    figsize=(10, 5)
)

# ANALISE DE DADOS AUSENTES

print("\nDADOS AUSENTES:", spotify_data.isnull().sum())

print("\nTotal de valores ausentes:", spotify_data.isnull().sum().sum())

# ANALISE DE DUPLICATAS

print("\nREGISTROS REPETIDOS:")

print("Linhas duplicadas:")
print(spotify_data.duplicated().sum())

print("\nIDs de musicas repetidos:")
print(spotify_data["track_id"].duplicated().sum())

# Exibir alguns IDs repetidos
ids_repetidos = spotify_data[
    spotify_data["track_id"].duplicated(keep=False)
]

print("\nExemplos de IDs repetidos:")
print(
    ids_repetidos[
        ["track_id", "track_name", "artists", "track_genre"]
    ].head(10)
)

# ANALISE DOS GENEROS APÓS OS IDS REPETIDOS SEREM REMOVIDOS

spotify_data_remove = spotify_data.drop_duplicates(
    subset="track_id",
    keep="first"
)

print("\n10. ANALISE APOS REMOVER IDS REPETIDOS")

print("Quantidade original de linhas:", len(spotify_data))
print("Quantidade de IDs unicos:", len(spotify_data_remove))

print("\nQuantidade de musicas por genero:")
print(spotify_data_remove["track_genre"].value_counts())

print("\nQuantidade de generos:")
print(spotify_data_remove["track_genre"].nunique())