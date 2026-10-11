# Variante `v2`: v1 + troca de fundo

Gerado por `python model/treinar_campo.py --triagem` em 10/10/2026 14:07. Treino: 1.180 imagens ({'bracol': 1180}); 20 épocas fixas; o `melhor.pt` é a última época (fora do git). Teste fechado.

## Validação oficial do BRACOL (252 imagens)

- **Acurácia:** 89,3% (225 de 252 imagens; intervalo de Wilson de 95%: 84,9% a 92,5%).
- **F1 macro:** 87,5% (média simples do F1 das 5 classes).

| classe | imagens | precisão | recall | F1 |
|---|---:|---:|---:|---:|
| saudavel | 41 | 100,0% | 97,6% | 98,8% |
| ferrugem | 79 | 93,4% | 89,9% | 91,6% |
| bicho_mineiro | 58 | 92,0% | 79,3% | 85,2% |
| phoma | 52 | 96,1% | 94,2% | 95,1% |
| cercosporiose | 22 | 54,3% | 86,4% | 66,7% |

Matriz de confusão (linhas: classe verdadeira; colunas: classe prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 40 | 0 | 0 | 1 | 0 |
| ferrugem | 0 | 71 | 0 | 0 | 8 |
| bicho_mineiro | 0 | 4 | 46 | 1 | 7 |
| phoma | 0 | 0 | 2 | 49 | 1 |
| cercosporiose | 0 | 1 | 2 | 0 | 19 |

**Severidade** (níveis 0 a 4; 252 imagens com alvo):

- MAE (em níveis): 0,321; kappa quadrático ponderado: 0,638.
- Acurácia exata: 73,8%; com tolerância de 1 nível: 94,4%.
- Imagens por nível verdadeiro: 0: 41, 1: 138, 2: 49, 3: 15, 4: 9.

Tempo: 4.3 min. Histórico em `historico.csv`.

