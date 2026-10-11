# Variante `v3s`: v2 + JMuBEN sem phoma e cercosporiose

Gerado por `python model/treinar_campo.py --triagem` em 10/10/2026 14:19. Treino: 1.513 imagens ({'bracol': 1180, 'jmuben': 333}); 20 épocas fixas; o `melhor.pt` é a última época (fora do git). Teste fechado.

## Validação oficial do BRACOL (252 imagens)

- **Acurácia:** 90,5% (228 de 252 imagens; intervalo de Wilson de 95%: 86,2% a 93,5%).
- **F1 macro:** 88,6% (média simples do F1 das 5 classes).

| classe | imagens | precisão | recall | F1 |
|---|---:|---:|---:|---:|
| saudavel | 41 | 100,0% | 97,6% | 98,8% |
| ferrugem | 79 | 96,1% | 93,7% | 94,9% |
| bicho_mineiro | 58 | 95,7% | 77,6% | 85,7% |
| phoma | 52 | 90,7% | 94,2% | 92,5% |
| cercosporiose | 22 | 58,8% | 90,9% | 71,4% |

Matriz de confusão (linhas: classe verdadeira; colunas: classe prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 40 | 0 | 0 | 1 | 0 |
| ferrugem | 0 | 74 | 0 | 0 | 5 |
| bicho_mineiro | 0 | 2 | 45 | 3 | 8 |
| phoma | 0 | 0 | 2 | 49 | 1 |
| cercosporiose | 0 | 1 | 0 | 1 | 20 |

**Severidade** (níveis 0 a 4; 252 imagens com alvo):

- MAE (em níveis): 0,282; kappa quadrático ponderado: 0,674.
- Acurácia exata: 77,4%; com tolerância de 1 nível: 94,8%.
- Imagens por nível verdadeiro: 0: 41, 1: 138, 2: 49, 3: 15, 4: 9.

Tempo: 5.3 min. Histórico em `historico.csv`.

