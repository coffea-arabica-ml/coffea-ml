# Variante `v0`: augmentation atual

Gerado por `python model/treinar_campo.py --triagem` em 10/10/2026 13:51. Treino: 1.180 imagens ({'bracol': 1180}); 20 épocas fixas; o `melhor.pt` é a última época (fora do git). Teste fechado.

## Validação oficial do BRACOL (252 imagens)

- **Acurácia:** 93,3% (235 de 252 imagens; intervalo de Wilson de 95%: 89,5% a 95,7%).
- **F1 macro:** 91,4% (média simples do F1 das 5 classes).

| classe | imagens | precisão | recall | F1 |
|---|---:|---:|---:|---:|
| saudavel | 41 | 100,0% | 95,1% | 97,5% |
| ferrugem | 79 | 95,1% | 97,5% | 96,2% |
| bicho_mineiro | 58 | 94,4% | 87,9% | 91,1% |
| phoma | 52 | 98,0% | 94,2% | 96,1% |
| cercosporiose | 22 | 67,9% | 86,4% | 76,0% |

Matriz de confusão (linhas: classe verdadeira; colunas: classe prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 39 | 0 | 0 | 1 | 1 |
| ferrugem | 0 | 77 | 0 | 0 | 2 |
| bicho_mineiro | 0 | 3 | 51 | 0 | 4 |
| phoma | 0 | 0 | 1 | 49 | 2 |
| cercosporiose | 0 | 1 | 2 | 0 | 19 |

**Severidade** (níveis 0 a 4; 252 imagens com alvo):

- MAE (em níveis): 0,266; kappa quadrático ponderado: 0,729.
- Acurácia exata: 77,4%; com tolerância de 1 nível: 96,0%.
- Imagens por nível verdadeiro: 0: 41, 1: 138, 2: 49, 3: 15, 4: 9.

## Receita x validação cruzada antiga do v0

Igual à do diagnóstico por blocos: a validação cruzada do v0 é reaproveitada.
as duas usam o WeightedRandomSampler com a semente 42, o modelo da última época e AMP; a triagem lê as imagens de um cache já em 448x224 (o mesmo INTER_AREA do primeiro passo da augmentation).

Tempo: 3.4 min. Histórico em `historico.csv`.

