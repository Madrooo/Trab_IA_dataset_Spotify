# Classificador de gêneros musicais com KNN

Projeto em Python para classificar gêneros musicais usando dados do Spotify e o algoritmo K-Nearest Neighbors (KNN).

## Visão geral

Este projeto lê um dataset de músicas, realiza limpeza e pré-processamento dos dados, seleciona um subconjunto de gêneros mais representativos e treina um modelo de classificação para prever o gênero musical a partir de atributos de áudio.

O script principal é:

- `machine-learning-spotify.py`

## Objetivo

Identificar o gênero de uma faixa musical com base em características como:

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`

## Dataset

O projeto usa o arquivo:

- `dataset.csv`

Esse dataset contém uma coleção de faixas com informações como identificação da música, artista, álbum, atributos de áudio e o gênero (`track_genre`).

## Fluxo do projeto

O script executa as seguintes etapas:

1. Carrega o dataset
2. Explora informações básicas do arquivo
3. Verifica valores nulos e duplicados
4. Analisa gêneros disponíveis no conjunto de dados
5. Filtra um subconjunto de gêneros relevantes
6. Limpa dados inconsistentes
7. Remove colunas pouco úteis para a classificação
8. Separa variáveis de entrada e alvo
9. Divide os dados em treino e teste
10. Normaliza as features
11. Testa diferentes valores de `k` com validação cruzada
12. Escolhe o melhor valor de `k`
13. Treina o modelo KNN final
14. Avalia a acurácia e o relatório de classificação
15. Exibe a matriz de confusão

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Colorama
- Biblioteca local `biblioteca_facilitar`

## Requisitos

Você precisará ter o Python instalado e as dependências do projeto disponíveis.

### Instalação

```bash
pip install pandas scikit-learn colorama
```

## Como executar

Na raiz do projeto, execute:

```bash
python machine-learning-spotify.py
```

O script imprime no terminal informações sobre o dataset, limpeza, treino e avaliação do modelo.

## Estrutura do projeto

```text
Trab_IA_dataset_Spotify/
├── README.md
├── dataset.csv
├── machine-learning-spotify.py
├── biblioteca_facilitar/
│   └── linha.py
└── state/
```

## Resultado esperado

Ao final da execução, o programa mostra:

- contagem de músicas após limpeza
- quantidade por gênero
- métricas de avaliação do modelo
- melhor valor de `k` encontrado
- acurácia no conjunto de teste
- relatório por gênero
- matriz de confusão

## Observações

O projeto foi pensado como uma análise prática de classificação multiclasse em machine learning, com foco na comparação de desempenho entre diferentes valores de vizinhos para o algoritmo KNN.

## Próximos passos possíveis

- testar outros modelos, como Random Forest ou SVM
- comparar com redes neurais
- explorar melhor seleção de gêneros
- melhorar a análise de métricas e visualização dos resultados
- exportar os resultados em gráficos e relatórios
