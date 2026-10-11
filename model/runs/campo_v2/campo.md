## Campo: 318 folhas das fotos de validação do BRACOT

Classificador `model/runs/campo_v2`. Concordância de classe entre os modos A e B: 43,7%. A estabilidade é a fração de 8 perturbações pequenas (brilho e contraste de 10%, deslocamento de 5%) com a mesma classe. **Proxies não decidem adoção:** um modelo que diz sempre a mesma classe é perfeitamente estável.

|  | modo A | modo B |
|---|---:|---:|
| folhas saudavel | 1 | 57 |
| folhas ferrugem | 178 | 80 |
| folhas bicho_mineiro | 16 | 44 |
| folhas phoma | 119 | 110 |
| folhas cercosporiose | 4 | 27 |
| severidade saudavel | 1 | 57 |
| severidade muito_baixa | 86 | 234 |
| severidade baixa | 231 | 27 |
| severidade alta | 0 | 0 |
| severidade muito_alta | 0 | 0 |
| fração da classe mais comum | 56,0% | 34,6% |
| confiança média | 55,4% | 43,7% |
| estabilidade média | 90,6% | 86,7% |
| estáveis nas 8 perturbações | 67,9% | 60,1% |
| borda: confiança / saudáveis | 53,3% / 0,5% | 41,5% / 24,0% |
| inteiras: confiança / saudáveis | 58,7% / 0,0% | 47,0% / 8,7% |

A severidade prevista no campo não é informativa: o modelo só aprendeu severidade em folhas do BRACOL, e no campo não há rótulo de severidade.

**Rótulos de campo** (feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026; não são de especialista; 120 de 120 casados com as folhas atuais; sem os "?"):

|  | modo A | modo B |
|---|---|---|
| rotuladas (S / D / ?) | 36 / 46 / 38 | 36 / 46 / 38 |
| P(prever doença dado S) | 100,0% (90,4% a 100,0%; n=36) | 69,4% (53,1% a 82,0%; n=36) |
| P(prever saudável dado D) | 0,0% (0,0% a 7,7%; n=46) | 0,0% (0,0% a 7,7%; n=46) |
| classe certa, dado D com classe | - | - |
| P(prever doença dado S), S com "brilho" | 100,0% (78,5% a 100,0%; n=14) | 64,3% (38,8% a 83,7%; n=14) |
| P(prever doença dado S), S sem "brilho" | 100,0% (85,1% a 100,0%; n=22) | 72,7% (51,8% a 86,8%; n=22) |

Folhas com "fragmento" na observação: 35 (34 na borda); "?" entre elas 40,0% (25,6% a 56,4%; n=35) e entre as demais 28,2% (19,8% a 38,6%; n=85); fração visível mediana 44,8% x 96,4%.

**Fragmentos** (192 das 318 folhas cortadas pela borda; fração visível mediana 64,3% na borda e 100,0% nas inteiras):

| filtro | limiar | cortadas | na borda | "?" entre cortadas | "?" entre mantidas |
|---|---:|---:|---:|---:|---:|
| fracao_visivel | 0,20 | 20 | 19 | 60,0% | 29,1% |
| fracao_visivel | 0,30 | 37 | 36 | 56,2% | 27,9% |
| fracao_visivel | 0,50 | 79 | 75 | 51,4% | 23,5% |
| preenchimento | 0,50 | 19 | 15 | 62,5% | 29,5% |
| preenchimento | 0,60 | 95 | 63 | 33,3% | 31,0% |
| preenchimento | 0,70 | 254 | 142 | 30,6% | 36,4% |
