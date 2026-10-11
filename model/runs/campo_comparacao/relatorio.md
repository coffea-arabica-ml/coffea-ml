# Classificador robusto ao campo: comparação das variantes

Gerado por `python model/treinar_campo.py --comparar` em 11/10/2026 10:04. Testes do BRACOL e do BRACOT fechados. Cada variante tem a pasta `model/runs/campo_<v>/` (triagem, proxies e rótulos de campo, validação cruzada das finalistas) e `campo_<v>_pipeline/` (T1, T2, T3 e a curva risco-cobertura).

**Regra de adoção (decisão do gestor, 10/10/2026):** fica o que não piorar o BRACOL (queda de F1 macro de até ~2 pontos na validação cruzada) e melhorar o campo. No campo, só os rótulos decidem, e só onde os intervalos de Wilson sustentam; concordância e estabilidade servem para detectar comportamento degenerado e desempatar. Sem diferença sustentada pelos rótulos, a recomendação é ficar no v0.

**Ressalva:** os rótulos de campo escolhem as finalistas e também medem, então a medida sai otimista (120 folhas, 46 fotos de 2 sessões, 1 lavoura; rótulos feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026, não de especialista).

| variante | F1 val (252) | F1 CV (1.432) | Δ CV x v0 | modo | A×B | estab. | saudáveis | classe dominante | P(doença dado S) | P(saudável dado D) | limiar recalibrado |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|---|
| `v0` (augmentation atual) | 91,4% | 88,2% | +0,0 | B | 54,4% | 87,1% | 10,1% | 51,9% | 80,6% (65,0%–90,2%) | 0,0% (0,0%–7,7%) | 0,58 (val) |
| `v1` (campo: fotométrica e geométrica) | 87,1% | - | - | B | 53,8% | 90,6% | 6,0% | 53,8% | 91,7% (78,2%–97,1%) | 0,0% (0,0%–7,7%) | 0,53 (val) |
| `v1f` (campo: só fotométrica) | 89,5% | - | - | B | 46,5% | 90,6% | 8,2% | 48,1% | 88,9% (74,7%–95,6%) | 0,0% (0,0%–7,7%) | 0,57 (val) |
| `v1g` (campo: só geométrica (corte e erasing)) | 88,8% | - | - | B | 49,4% | 85,6% | 9,1% | 46,9% | 83,3% (68,1%–92,1%) | 0,0% (0,0%–7,7%) | 0,64 (val) |
| `v2` (v1 + troca de fundo) | 87,5% | - | - | B | 43,7% | 86,7% | 17,9% | 34,6% | 69,4% (53,1%–82,0%) | 0,0% (0,0%–7,7%) | 0,64 (val) |
| `v3` (v2 + JMuBEN) | 88,0% | - | - | B | 51,9% | 92,0% | 14,5% | 52,2% | 63,9% (47,6%–77,5%) | 0,0% (0,0%–7,7%) | 0,65 (val) |
| `v3s` (v2 + JMuBEN sem phoma e cercosporiose) | 88,6% | - | - | B | 27,4% | 88,4% | 21,7% | 35,5% | 50,0% (34,5%–65,5%) | 0,0% (0,0%–7,7%) | 0,62 (val) |

### Rótulos de campo nos dois modos

P(prever doença dado S) em 36 folhas S (intervalo de Wilson de 95%). P(prever saudável dado D) foi 0% (0 a 8%) em todas as variantes e nos dois modos.

| variante | modo A | modo B | modo B, S com brilho | modo B, S sem brilho |
|---|---|---|---|---|
| `v0` | 100% (90%–100%) | 81% (65%–90%) | 79% (52%–92%) | 82% (61%–93%) |
| `v1` | 100% (90%–100%) | 92% (78%–97%) | 93% (69%–99%) | 91% (72%–97%) |
| `v1f` | 100% (90%–100%) | 89% (75%–96%) | 93% (69%–99%) | 86% (67%–95%) |
| `v1g` | 100% (90%–100%) | 83% (68%–92%) | 93% (69%–99%) | 77% (57%–90%) |
| `v2` | 100% (90%–100%) | 69% (53%–82%) | 64% (39%–84%) | 73% (52%–87%) |
| `v3` | 100% (90%–100%) | 64% (48%–78%) | 79% (52%–92%) | 55% (35%–73%) |
| `v3s` | 100% (90%–100%) | 50% (34%–66%) | 43% (21%–67%) | 55% (35%–73%) |

F1 CV do v0: as predições fora da dobra do diagnóstico por blocos (mesma receita e mesmas dobras; conferido no `relatorio.md` de `campo_v0`). Colunas de campo no modo escolhido para a variante; "-" onde não se aplica ou não foi medido.

## Recomendação

**Recomendação (11/10/2026): ficar no v0, a receita atual; o modelo oficial do pipeline continua o `base_resnet50_bracol`.** A validação cruzada das variantes não foi rodada: fica a critério do gestor.

- **Nenhuma variante tem diferença sustentada no campo.**
  - No modo A, todas chamam de doente 100% (90–100%) das 36 folhas S.
  - No modo B, a melhor é a v3s (JMuBEN sem phoma e cercosporiose), com 50% (34–66%) de P(prever doença | S). O v0 tem 81% (65–90%): os intervalos se tocam por 1 ponto.
  - P(prever saudável | D) é 0% (0–8%) em todas as variantes e nos dois modos. O número quase não informa, porque quase tudo sai doente.
- **No BRACOL, todas as variantes perderam na validação oficial**, de −1,9 (v1f) a −4,3 pontos (v1) de F1 macro contra o v0. Pela triagem (queda de mais de ~3 pontos), v1, v2 e v3 já sairiam.
- **O que os rótulos sustentam: o modo B é melhor que o A.**
  - No modelo atual, P(prever doença | S) cai de 100% (90–100%) para 69% (53–82%), sem aumentar P(prever saudável | D), que fica em 0%.
  - O mesmo vale para v0, v2, v3 e v3s.
  - Trocar o `recorte` da cadeia para "B" é uma decisão do gestor.
  - Mesmo no B, 7 em cada 10 folhas saudáveis saem doentes: o classificador continua inutilizável no campo.
- **A hipótese do brilho forte não se sustenta.**
  - No modo A, as folhas S com "brilho" na observação (14) e sem (22) saem todas como doentes (100% nas duas).
  - No modo B, os intervalos se sobrepõem: 79% (52–92%) contra 64% (43–80%) no modelo atual.
  - O erro é mais amplo que o brilho.
- **Filtro de fragmentos: o da fração visível tem apoio dos rótulos.**
  - Com fração visível abaixo de 0,5, saem 79 das 318 folhas (75 cortadas pela borda).
  - Entre as rotuladas que ele cortaria, 51% são "?" (n=35), contra 24% entre as que ficam (n=85); os intervalos de Wilson não se tocam.
  - As folhas com "fragmento" na observação têm fração visível mediana de 45%, contra 96% das demais.
  - O preenchimento não discrimina.
  - A proposta, para decisão do gestor, é `fracao_visivel_minima = 0,5`.
- **Próximo passo possível (decisão do gestor):** treinar com folhas de campo rotuladas. Os 120 rótulos daqui são só de avaliação, então seria preciso uma fonte nova de treino. Outro caminho é um classificador saudável x doente treinado com folhas de campo, que barre o falso "doente" antes do classificador de doença.
- **Ressalvas:**
  - Os rótulos foram feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026; não são de especialista.
  - São 120 folhas, de 46 fotos de 2 sessões, numa lavoura só.
  - Os rótulos escolhem e medem ao mesmo tempo, então a medida sai otimista.
  - Nenhuma folha D recebeu classe: a classe da doença no campo não foi medida.

