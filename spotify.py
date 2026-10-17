import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from os import replace
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

##############################################################################
# Localizando arquivo .csv dentro da pasta Spotify
##############################################################################

pasta_script = Path(__file__).resolve().parent
caminho_csv = pasta_script / "dataset.csv"

print("Arquivo .csv encontrado:", caminho_csv)

##############################################################################
# 1. Leitura dos dados de dataset.csv
##############################################################################

spotify_data = pd.read_csv(caminho_csv, sep=",")

print("\nQuantidade de linhas e colunas: ", spotify_data.shape)
print("Informações do dataset: ", spotify_data.info())

