# Variante `v3`: v2 + JMuBEN

Gerado por `python model/treinar_campo.py --triagem` em 10/10/2026 14:14. Treino: 1.686 imagens ({'bracol': 1180, 'jmuben': 506}); 20 épocas fixas; o `melhor.pt` é a última época (fora do git). Teste fechado.

## Validação oficial do BRACOL (252 imagens)

- **Acurácia:** 90,1% (227 de 252 imagens; intervalo de Wilson de 95%: 85,8% a 93,2%).
- **F1 macro:** 88,0% (média simples do F1 das 5 classes).

| classe | imagens | precisão | recall | F1 |
|---|---:|---:|---:|---:|
| saudavel | 41 | 100,0% | 100,0% | 100,0% |
| ferrugem | 79 | 96,1% | 92,4% | 94,2% |
| bicho_mineiro | 58 | 93,6% | 75,9% | 83,8% |
| phoma | 52 | 92,6% | 96,2% | 94,3% |
| cercosporiose | 22 | 55,9% | 86,4% | 67,9% |

Matriz de confusão (linhas: classe verdadeira; colunas: classe prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 41 | 0 | 0 | 0 | 0 |
| ferrugem | 0 | 73 | 0 | 0 | 6 |
| bicho_mineiro | 0 | 2 | 44 | 3 | 9 |
| phoma | 0 | 0 | 2 | 50 | 0 |
| cercosporiose | 0 | 1 | 1 | 1 | 19 |

**Severidade** (níveis 0 a 4; 252 imagens com alvo):

- MAE (em níveis): 0,274; kappa quadrático ponderado: 0,679.
- Acurácia exata: 78,6%; com tolerância de 1 nível: 94,4%.
- Imagens por nível verdadeiro: 0: 41, 1: 138, 2: 49, 3: 15, 4: 9.

Tempo: 6.5 min. Histórico em `historico.csv`.

