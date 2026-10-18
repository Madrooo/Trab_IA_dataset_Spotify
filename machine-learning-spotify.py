import pandas as pd
from biblioteca_facilitar.linha import linha
import colorama
from colorama import Fore, init

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

generos = df[df["track_genre"].isin(["pop", "rock", "hip hop", "classical", "jazz", "country", "reggae", "blues", "forro", "indie", "indie-pop"])] #generos de musica desejados
print(Fore.BLUE + "Generos de Musica Desejados:")
print(sorted(generos["track_genre"].unique()))