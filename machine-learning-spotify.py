import pandas as pd

df = pd.read_csv('dataset.csv', index_col=0)

df.shape #linhas/colunas
df.head() #primeiras linhas
df.info() #informações do dataset