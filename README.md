# Classificação de gêneros musicais com dados do Spotify

Projeto de machine learning que usa atributos de áudio do dataset do Spotify para classificar gêneros musicais. O repositório contém duas abordagens: K-Nearest Neighbors (KNN) e uma rede neural Multilayer Perceptron (MLP).

## Dataset

Os scripts leem `dataset.csv` na raiz do projeto. O arquivo deve conter a coluna `track_genre` e os atributos de áudio usados pelos modelos. O dataset não é gerado pelos scripts.

## Requisitos

- Python 3
- pandas
- scikit-learn
- colorama (usado pelo script KNN)

Instale as dependências com:

```bash
pip install pandas scikit-learn colorama
```

O script KNN também importa o módulo local `biblioteca_facilitar`, que já faz parte deste repositório.

## Modelos

### KNN — `machine-learning-spotify.py`

Para executar:

```bash
python machine-learning-spotify.py
```

O script explora o dataset, seleciona os gêneros `hip-hop`, `classical`, `jazz`, `country`, `reggae`, `forro`, `metal` e `techno`, remove registros duplicados ou inválidos e treina um classificador KNN. O valor de `k` é escolhido por validação cruzada no conjunto de treino. Ao final, o programa imprime métricas, relatório por gênero e matriz de confusão no terminal.

### MLP — `spotify_mlp.py`

Para executar:

```bash
python spotify_mlp.py
```

O script usa os mesmos oito gêneros e 13 atributos de áudio: duração, dançabilidade, energia, tonalidade, volume, modo, fala, acústica, instrumentalidade, vivacidade, valência, andamento e fórmula de compasso. Os dados são divididos em treino e teste de forma estratificada. Um pipeline substitui valores ausentes pela mediana, escala os atributos entre 0 e 1 e treina uma rede MLP com duas camadas ocultas.

O script imprime métricas de avaliação, matriz de confusão e relatório de classificação por gênero. Também cria ou sobrescreve na pasta atual:

- `spotify_mlp_dados.csv`: atributos utilizados e rótulo do gênero codificado numericamente.
- `spotify_mlp_resultados.csv`: gênero real e gênero previsto para cada registro do conjunto de teste.

## Estrutura

```text
Trab_IA_dataset_Spotify/
├── README.md
├── dataset.csv
├── machine-learning-spotify.py
├── spotify_mlp.py
└── biblioteca_facilitar/
    └── linha.py
```

Os arquivos CSV de saída do MLP são criados ao executar o script e não precisam estar presentes antes da execução.
