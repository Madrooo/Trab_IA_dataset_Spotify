import pandas as pd

df = pd.read_csv('dataset.csv', index_col=0)

df.shape #linhas/colunas
df.head() #primeiras linhas
df.info() #informações do dataset

df.describe().T
print (df.describe().T) # printa estatísticas do dataset

df.isnull().sum() #verifica valores nulos
print("Valores Nulos:")
print(df.isnull().sum()) # printa valores nulos

df.duplicated().sum() #verifica valores duplicados
print("Valores Duplicados:")
print(df.duplicated().sum()) # printa valores duplicados

df["track_id"].duplicated() #verifica valores únicos
print("IDs Duplicados:")
print(df["track_id"].duplicated().sum()) # printa id duplicada