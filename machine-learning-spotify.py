import pandas as pd
from biblioteca_facilitar.linha import linha
import colorama
from colorama import Fore, init
from sklearn.model_selection import train_test_split

init(autoreset=True)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Apenas lendo o dataset e verificando informações básicas
print(Fore.GREEN + "Lendo o dataset...")
df = pd.read_csv('dataset.csv', index_col=0)

#separador automatico
linha("-=", 40)

print(Fore.BLUE + "Informacoes do Dataset:")
df.shape #linhas/colunas
df.head() #primeiras linhas
df.info() #informações do dataset

#separador automatico
linha("-=", 40)

df.describe().T
print(Fore.BLUE + "Estatisticas do Dataset:")
print (df.describe().T) # printa estatísticas do dataset

#separador automatico
linha("-=", 40)

df.isnull().sum() #verifica valores nulos
print(Fore.BLUE + "Valores Nulos:")
print(df.isnull().sum()) # printa valores nulos

#separador automatico
linha("-=", 40)

df.duplicated().sum() #verifica valores duplicados
print(Fore.BLUE + "Linhas totalmente identicas:")
print(df.duplicated().sum()) # printa valores duplicados

#separador automatico
linha("-=", 40)

df["track_id"].duplicated() #verifica valores duplicados na coluna track_id
print(Fore.BLUE + "IDs Duplicados:")
print(df["track_id"].duplicated().sum()) # printa todos os IDs duplicados

linha("-=", 40)

print(Fore.BLUE + "Lista dos generos de musica:")
print(sorted(df["track_genre"].unique())) # printa todos os generos de música  

linha("-=", 40)

generos = df[df["track_genre"].isin(["pop", "hip-hop", "classical", "jazz", "country", "reggae", "blues", "forro", "metal", "techno"])] #generos de musica desejados
print(Fore.BLUE + "Generos de Musica Desejados:")
print(sorted(generos["track_genre"].unique())) # printa todos os generos de música desejados
print(Fore.BLUE + "Total de generos encontrados:", generos["track_genre"].nunique()) # printa total de generos encontrados

linha("-=", 40)

#limpeza do dataset, removendo valores nulos e duplicados

df_cleaned = generos.dropna() #removendo valores nulos
df_cleaned = df_cleaned.drop_duplicates(subset=["track_id","track_genre"]) #removendo valores duplicados, considerando apenas as colunas track_id e track_genre 
df_cleaned = df_cleaned[df_cleaned["track_id"].duplicated() == False] #removendo valores duplicados na coluna track_id
df_cleaned = df_cleaned[df_cleaned["tempo"] > 0] #removendo valores com tempo de música igual a 0

print(Fore.BLUE + "Contagem de musicas apos o dataset limpo:")
print(len(df_cleaned)) # printa o tamanho do dataset limpo
print(df_cleaned["track_genre"].value_counts()) # printa a quantidade por gênero

linha("-=", 40)

#limpando as colunas inuteis

colunas_inuteis = ["track_id", "artists", "album_name", "track_name",
                   "popularity", "duration_ms", "explicit",
                   "key", "mode", "time_signature"]

df_cleaned = df_cleaned.drop(columns=colunas_inuteis) #removendo colunas inuteis

print(Fore.BLUE + "Colunas restantes apos a limpeza:")
print(df_cleaned.columns) # printa as colunas restantes

linha("-=", 40)

#separando os dados em treino e teste

X = df_cleaned.drop(columns=["track_genre"]) #recebe todas as colunas menos a coluna track_genre
Y = df_cleaned["track_genre"] #recebe apenas a coluna track_genre

print(Fore.BLUE + "Colunas (X)")
print(X.columns) # printa as colunas de X

print(Fore.BLUE + "Colunas (df_cleaned)")
print(df_cleaned.columns) # printa todas as colunas do dataset limpo

linha("-=", 40)

#dividindo em treino por volta de 80% e teste por volta de 20%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(Fore.BLUE + "Treino: ", X_train.shape) # printa o tamanho do dataset de treino
print(Fore.BLUE + "Teste: ", X_test.shape) # printa o tamanho do dataset de teste