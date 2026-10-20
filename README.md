# Classificação de gênero musical com KNN e MLP

Trabalho de Machine Learning usando o dataset de faixas do Spotify. O objetivo é prever o **gênero de uma música** a partir de suas características de áudio, comparando dois modelos: **KNN** e **MLP**.

> **Status:** análise dos dados, limpeza e KNN concluídos. MLP em ajuste para usar os mesmos dados do KNN. Os campos marcados com `TODO` ainda precisam ser preenchidos.

**Autores:** Matheus Oliveira Silva e Lucas (TODO: sobrenome)
**Instituição:** IFCE — Ciência da Computação
**Disciplina:** TODO
**Professor(a):** TODO

---

## 1. Dados

- **Arquivo:** `dataset.csv` (TODO: link da fonte)
- **Tamanho original:** 114.000 músicas e 21 colunas
- **Gêneros:** 114, com 1.000 músicas cada
- **Cada linha:** uma faixa, com informações de identificação (`track_id`, `artists`, `album_name`, `track_name`), características de áudio e o gênero (`track_genre`)

### Problemas encontrados na exploração

- 24.259 `track_id` repetidos (a mesma música aparece em mais de um gênero)
- 450 linhas totalmente idênticas
- 1 linha com valores nulos
- 157 faixas com `tempo` igual a 0 (erro de detecção)
- Versões repetidas da mesma música com `track_id` diferentes (single, álbum, relançamento)

## 2. Problema

- **Tipo:** classificação multiclasse
- **Entrada (X):** 9 características de áudio

| Coluna | Descrição |
|---|---|
| `danceability` | Quão adequada a faixa é para dançar (0 a 1) |
| `energy` | Intensidade e atividade percebidas (0 a 1) |
| `loudness` | Volume médio em dB |
| `speechiness` | Presença de fala (0 a 1) |
| `acousticness` | Confiança de que a faixa é acústica (0 a 1) |
| `instrumentalness` | Probabilidade de não ter vocais (0 a 1) |
| `liveness` | Probabilidade de ser gravação ao vivo (0 a 1) |
| `valence` | Positividade musical (0 a 1) |
| `tempo` | Andamento em BPM |

- **Saída (y):** `track_genre`

### Gêneros escolhidos (8)

`hip-hop`, `classical`, `jazz`, `country`, `reggae`, `forro`, `metal`, `techno`

Foram escolhidos gêneros com perfis de áudio distintos entre si, já que usar os 114 gêneros deixaria o problema muito difícil e a matriz de confusão ilegível. TODO: ajustar a justificativa se a equipe quiser.

## 3. Pré-processamento

1. Filtro dos 8 gêneros (8.000 músicas)
2. Remoção de valores nulos
3. Remoção de duplicatas por `track_id` + `track_genre`
4. Remoção de `track_id` repetidos (mantém a primeira ocorrência)
5. Remoção de faixas com `tempo` igual a 0
6. Remoção de versões repetidas da mesma música (mesmo `track_name` e `artists`)
7. Remoção de faixas com características de áudio idênticas
8. Remoção das colunas que não entram no modelo

Resultado: **5.915 músicas**.

| Gênero | Músicas |
|---|---|
| forro | 968 |
| techno | 954 |
| metal | 814 |
| classical | 811 |
| hip-hop | 733 |
| country | 604 |
| reggae | 562 |
| jazz | 469 |

> **Por que remover as versões repetidas:** antes dessa etapa, o melhor `k` do KNN era 1 (acurácia de validação de ~61%), sinal de que o modelo encontrava a "gêmea" da música no treino e copiava o gênero dela. Depois da remoção, o melhor `k` passou a ser 21 (~54%), um resultado mais honesto. Esses números são da configuração com 10 gêneros, testada antes da lista final.

## 4. Divisão e normalização

- **Treino/teste:** 80% / 20%, com `stratify` e `random_state=42` (treino: 4.732 músicas, teste: 1.183)
- **Normalização:** `StandardScaler`, ajustado somente com os dados de treino para não vazar informação do teste

## 5. Modelos

### KNN

- O melhor `k` foi escolhido por validação cruzada (5 partes) apenas no treino, testando `k` = 1, 3, 5, 7, 9, 11, 15, 21 e 31
- **Melhor `k`:** 11 (acurácia média na validação: 0,6513)

### MLP

- Rede com duas camadas ocultas `(100, 50)`, ativação `relu`, otimizador `adam` e parada antecipada
- Deve usar os mesmos dados, as mesmas 9 colunas, a mesma divisão e a mesma normalização do KNN
- TODO: confirmar a configuração final

## 6. Métricas de avaliação

- **Acurácia:** proporção de músicas classificadas corretamente
- **Precisão, recall e F1 (média macro):** dão o mesmo peso a todos os gêneros, inclusive os menores
- **Relatório por gênero e matriz de confusão:** mostram onde o modelo acerta e com o que ele confunde

## 7. Resultados

| Modelo | Acurácia no teste | F1 macro |
|---|---|---|
| KNN (`k = 11`) | 0,6391 | TODO |
| MLP | TODO | TODO |

- Gêneros que o modelo acerta mais: TODO
- Gêneros que o modelo acerta menos: TODO
- Principais confusões da matriz: TODO

## 8. Conclusões

- Qual modelo teve o melhor desempenho: TODO
- Quais gêneros são mais fáceis e mais difíceis de separar, e por quê: TODO
- Limitações: as características de áudio sozinhas não bastam para separar gêneros parecidos. TODO: complementar
- Possíveis melhorias: TODO

## 9. Estrutura do projeto

```
Trab_IA_dataset_Spotify/
├── dataset.csv
├── biblioteca_facilitar/        # módulo próprio (função linha)
├── machine-learning-spotify.py  # KNN
├── TODO.py                      # MLP (confirmar o nome)
└── README.md
```

## 10. Como executar

```
python -m pip install pandas scikit-learn colorama
python machine-learning-spotify.py
python spotify_mip.py
```

A pasta `biblioteca_facilitar` precisa estar na mesma pasta do script. TODO: acrescentar o comando do MLP.