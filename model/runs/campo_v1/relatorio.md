# Variante `v1`: campo: fotométrica e geométrica

Gerado por `python model/treinar_campo.py --triagem` em 10/10/2026 13:55. Treino: 1.180 imagens ({'bracol': 1180}); 20 épocas fixas; o `melhor.pt` é a última época (fora do git). Teste fechado.

## Validação oficial do BRACOL (252 imagens)

- **Acurácia:** 89,3% (225 de 252 imagens; intervalo de Wilson de 95%: 84,9% a 92,5%).
- **F1 macro:** 87,1% (média simples do F1 das 5 classes).

| classe | imagens | precisão | recall | F1 |
|---|---:|---:|---:|---:|
| saudavel | 41 | 97,5% | 95,1% | 96,3% |
| ferrugem | 79 | 94,7% | 91,1% | 92,9% |
| bicho_mineiro | 58 | 90,6% | 82,8% | 86,5% |
| phoma | 52 | 94,1% | 92,3% | 93,2% |
| cercosporiose | 22 | 56,2% | 81,8% | 66,7% |

Matriz de confusão (linhas: classe verdadeira; colunas: classe prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 39 | 0 | 0 | 1 | 1 |
| ferrugem | 1 | 72 | 0 | 0 | 6 |
| bicho_mineiro | 0 | 3 | 48 | 1 | 6 |
| phoma | 0 | 0 | 3 | 48 | 1 |
| cercosporiose | 0 | 1 | 2 | 1 | 18 |

**Severidade** (níveis 0 a 4; 252 imagens com alvo):

- MAE (em níveis): 0,329; kappa quadrático ponderado: 0,630.
- Acurácia exata: 73,0%; com tolerância de 1 nível: 94,4%.
- Imagens por nível verdadeiro: 0: 41, 1: 138, 2: 49, 3: 15, 4: 9.

Tempo: 3.8 min. Histórico em `historico.csv`.

