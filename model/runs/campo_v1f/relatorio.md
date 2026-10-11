# Variante `v1f`: campo: só fotométrica

Gerado por `python model/treinar_campo.py --triagem` em 10/10/2026 13:59. Treino: 1.180 imagens ({'bracol': 1180}); 20 épocas fixas; o `melhor.pt` é a última época (fora do git). Teste fechado.

## Validação oficial do BRACOL (252 imagens)

- **Acurácia:** 91,3% (230 de 252 imagens; intervalo de Wilson de 95%: 87,1% a 94,2%).
- **F1 macro:** 89,5% (média simples do F1 das 5 classes).

| classe | imagens | precisão | recall | F1 |
|---|---:|---:|---:|---:|
| saudavel | 41 | 100,0% | 97,6% | 98,8% |
| ferrugem | 79 | 92,5% | 93,7% | 93,1% |
| bicho_mineiro | 58 | 94,1% | 82,8% | 88,1% |
| phoma | 52 | 98,0% | 94,2% | 96,1% |
| cercosporiose | 22 | 61,3% | 86,4% | 71,7% |

Matriz de confusão (linhas: classe verdadeira; colunas: classe prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 40 | 0 | 0 | 1 | 0 |
| ferrugem | 0 | 74 | 0 | 0 | 5 |
| bicho_mineiro | 0 | 5 | 48 | 0 | 5 |
| phoma | 0 | 0 | 1 | 49 | 2 |
| cercosporiose | 0 | 1 | 2 | 0 | 19 |

**Severidade** (níveis 0 a 4; 252 imagens com alvo):

- MAE (em níveis): 0,282; kappa quadrático ponderado: 0,700.
- Acurácia exata: 76,6%; com tolerância de 1 nível: 95,2%.
- Imagens por nível verdadeiro: 0: 41, 1: 138, 2: 49, 3: 15, 4: 9.

Tempo: 4.3 min. Histórico em `historico.csv`.

