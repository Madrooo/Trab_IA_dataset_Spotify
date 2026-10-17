import pandas as pd
from biblioteca_facilitar.linha import linha

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Apenas lendo o dataset e verificando informações básicas
print("Lendo o dataset...")
df = pd.read_csv('dataset.csv', index_col=0)

#separador automatico
linha("-=", 40)

print("Informacoes do Dataset:")
df.shape #linhas/colunas
df.head() #primeiras linhas
df.info() #informações do dataset

#separador automatico
linha("-=", 40)

df.describe().T
print("Estatisticas do Dataset:")
print (df.describe().T) # printa estatísticas do dataset

#separador automatico
linha("-=", 40)

df.isnull().sum() #verifica valores nulos
print("Valores Nulos:")
print(df.isnull().sum()) # printa valores nulos

#separador automatico
linha("-=", 40)

df.duplicated().sum() #verifica valores duplicados
print("Valores Duplicados:")
print(df.duplicated().sum()) # printa valores duplicados

#separador automatico
linha("-=", 40)

df["track_id"].duplicated() #verifica valores duplicados na coluna track_id


print("IDs Duplicados:")
print(df["track_id"].duplicated().sum()) # printa todos os IDs duplicados