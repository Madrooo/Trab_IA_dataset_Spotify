# Classificação de gênero musical com KNN e MLP

Trabalho de Machine Learning usando o dataset de faixas do Spotify. O objetivo é prever o **gênero de uma música** a partir de suas características de áudio, comparando dois modelos: **KNN** e **MLP**.

> **Status:** análise dos dados, limpeza e KNN concluídos. Os resultados do MLP são provisórios: o script ainda precisa ser alinhado com o KNN. Os campos marcados com `TODO` ainda precisam ser preenchidos.

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

### KNN (`machine-learning-spotify.py`)

- O melhor `k` foi escolhido por validação cruzada (5 partes) apenas no treino, testando `k` = 1, 3, 5, 7, 9, 11, 15, 21 e 31
- **Melhor `k`:** 11 (acurácia média na validação: 0,6513)

### MLP (`spotify_mlp.py`)

- Rede com duas camadas ocultas `(100, 50)`, ativação `relu`, otimizador `adam` e parada antecipada
- **Versão atual do script:** usa 13 atributos (os 9 acima mais `duration_ms`, `key`, `mode` e `time_signature`), substitui valores ausentes pela mediana e escala os dados entre 0 e 1 com `MinMaxScaler`. Ainda não aplica a limpeza de duplicatas do KNN
- **Objetivo:** usar os mesmos dados, as mesmas 9 colunas, a mesma divisão e a mesma normalização do KNN
- TODO: alinhar o script com o KNN e confirmar a configuração final

## 6. Métricas de avaliação

- **Acurácia:** proporção de músicas classificadas corretamente
- **Precisão, recall e F1 (média macro):** dão o mesmo peso a todos os gêneros, inclusive os menores
- **Relatório por gênero e matriz de confusão:** mostram onde o modelo acerta e com o que ele confunde

## 7. Resultados

Os dois modelos foram avaliados nas mesmas 1.183 músicas de teste.

| Modelo | Acurácia | Precisão macro | Recall macro | F1 macro |
|---|---|---|---|---|
| KNN (`k = 11`) | 0,6391 | 0,6086 | 0,5970 | 0,5964 |
| MLP (alinhado com o KNN)* | 0,6855 | 0,6505 | 0,6449 | 0,6447 |

\* TODO: confirmar rodando o `spotify_mlp.py` já ajustado (mesma limpeza, 9 colunas e `StandardScaler`). Os valores do MLP vêm de uma execução com essa configuração. A versão atual do script, sem a limpeza, usa outros dados (8.000 músicas e teste com 1.600) e não é comparável com o KNN (acurácia de 0,6538 nessa versão).

### F1 por gênero

| Gênero | KNN | MLP |
|---|---|---|
| classical | 0,81 | 0,84 |
| techno | 0,78 | 0,81 |
| forro | 0,76 | 0,79 |
| metal | 0,71 | 0,79 |
| hip-hop | 0,48 | 0,52 |
| country | 0,48 | 0,49 |
| jazz | 0,37 | 0,50 |
| reggae | 0,37 | 0,42 |

### Principais confusões do KNN

- reggae previsto como hip-hop: 40 das 112 músicas de reggae
- hip-hop previsto como reggae: 27 das 147
- jazz previsto como classical (18) e como country (18), de 94
- metal previsto como country: 20 das 163

## 8. Conclusões

- **Desempenho:** nos dados alinhados, o MLP (68,6%) ficou acima do KNN (63,9%) em acurácia e em F1 macro (0,645 contra 0,596). A diferença é de poucos pontos e o MLP varia entre execuções. TODO: confirmar rodando com mais de uma semente
- **Gêneros mais fáceis:** classical, techno, forro e metal têm perfis de áudio próprios, e os dois modelos os acertam com F1 entre 0,71 e 0,84
- **Gêneros mais difíceis:** reggae, jazz, hip-hop e country. O reggae se confunde com o hip-hop (e o contrário), e o jazz com o classical e o country, o que é coerente com a proximidade das características de áudio
- **Limitações:** as 9 características de áudio sozinhas não separam bem gêneros parecidos. A lista foi escolhida com gêneros distintos, e com gêneros mais próximos (como pop e indie) a acurácia cai. TODO: complementar
- **Possíveis melhorias:** TODO (ideias: testar atributos extras como `key`, `mode` e `duration_ms`, ajustar a arquitetura do MLP por validação cruzada e agrupar gêneros em famílias)

## 9. Estrutura do projeto

```
Trab_IA_dataset_Spotify/
├── README.md
├── dataset.csv
├── machine-learning-spotify.py  # KNN
├── spotify_mlp.py               # MLP
└── biblioteca_facilitar/        # módulo próprio
    └── linha.py                 # função linha, usada pelo KNN
```

Os arquivos `spotify_mlp_dados.csv` e `spotify_mlp_resultados.csv` são criados ao executar o MLP e não precisam existir antes.

## 10. Como executar

```
python -m pip install pandas scikit-learn colorama
python machine-learning-spotify.py
python spotify_mlp.py
```

- A pasta `biblioteca_facilitar` precisa estar na mesma pasta dos scripts (o KNN importa a função `linha` dela)
- O `dataset.csv` precisa estar na raiz do projeto
- O MLP cria ou sobrescreve, na pasta atual:
  - `spotify_mlp_dados.csv`: atributos utilizados e gênero codificado em números
  - `spotify_mlp_resultados.csv`: gênero real e gênero previsto de cada música do teste

## 11. Referências

- Documentação do scikit-learn