# Classificação de gênero musical com KNN e MLP

## 1. Ideia

Usar um dataset de faixas do Spotify para treinar modelos de machine learning que descubram o **gênero de uma música** a partir de suas características de áudio, como dançabilidade, energia, volume e andamento.

## 2. Dados

O projeto vai usar o arquivo `dataset.csv`, com cerca de 114 mil faixas musicais e 21 colunas. Cada linha é uma música, descrita por informações de identificação (nome, artista, álbum), características de áudio e o gênero a que pertence.

Na primeira etapa, o objetivo será **compreender os dados**: ver quais colunas existem, como estão distribuídas, se há valores ausentes ou repetidos e se os gêneros realmente se diferenciam pelo som.

## 3. Problema a ser resolvido

- **Tipo:** classificação multiclasse
- **Entrada:** características de áudio da faixa (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`)
- **Saída:** gênero musical (`track_genre`)

Como o dataset tem muitos gêneros, a ideia é trabalhar com um **subconjunto de poucos gêneros bem diferentes entre si** (por exemplo, clássica, metal, hip-hop, jazz e pop), para que a análise dos resultados seja clara.

## 4. Abordagem planejada

Serão usados dois modelos, comparados sobre os mesmos dados:

- **KNN** (K-Nearest Neighbors)
- **MLP** (Multi-Layer Perceptron)

Etapas previstas:

1. Ler e explorar o dataset
2. Escolher os gêneros que serão usados
3. Limpar os dados (nulos, duplicatas e valores inconsistentes)
4. Separar entrada e saída
5. Dividir em treino e teste
6. Normalizar as características, já que os dois modelos são sensíveis à escala
7. Treinar o KNN e o MLP, ajustando seus parâmetros com validação cruzada
8. Avaliar os dois modelos com acurácia, matriz de confusão, precisão, recall e F1
9. Comparar os resultados e escrever as conclusões

## 5. O que se espera descobrir

- Se as características de áudio são suficientes para distinguir os gêneros escolhidos
- Qual dos dois modelos tem melhor desempenho
- Quais gêneros são mais difíceis de separar

## 6. Próximos passos

- Definir os gêneros que serão usados
- Iniciar a exploração dos dados
- Montar o notebook com as etapas acima