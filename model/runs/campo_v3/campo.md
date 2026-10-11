## Campo: 318 folhas das fotos de validação do BRACOT

Classificador `model/runs/campo_v3`. Concordância de classe entre os modos A e B: 51,9%. A estabilidade é a fração de 8 perturbações pequenas (brilho e contraste de 10%, deslocamento de 5%) com a mesma classe. **Proxies não decidem adoção:** um modelo que diz sempre a mesma classe é perfeitamente estável.

|  | modo A | modo B |
|---|---:|---:|
| folhas saudavel | 1 | 46 |
| folhas ferrugem | 45 | 56 |
| folhas bicho_mineiro | 76 | 26 |
| folhas phoma | 190 | 166 |
| folhas cercosporiose | 6 | 24 |
| severidade saudavel | 1 | 46 |
| severidade muito_baixa | 137 | 242 |
| severidade baixa | 180 | 30 |
| severidade alta | 0 | 0 |
| severidade muito_alta | 0 | 0 |
| fração da classe mais comum | 59,7% | 52,2% |
| confiança média | 64,2% | 53,4% |
| estabilidade média | 91,0% | 92,0% |
| estáveis nas 8 perturbações | 68,9% | 72,3% |
| borda: confiança / saudáveis | 64,1% / 0,5% | 51,6% / 18,2% |
| inteiras: confiança / saudáveis | 64,2% / 0,0% | 56,0% / 8,7% |

A severidade prevista no campo não é informativa: o modelo só aprendeu severidade em folhas do BRACOL, e no campo não há rótulo de severidade.

**Rótulos de campo** (feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026; não são de especialista; 120 de 120 casados com as folhas atuais; sem os "?"):

|  | modo A | modo B |
|---|---|---|
| rotuladas (S / D / ?) | 36 / 46 / 38 | 36 / 46 / 38 |
| P(prever doença dado S) | 100,0% (90,4% a 100,0%; n=36) | 63,9% (47,6% a 77,5%; n=36) |
| P(prever saudável dado D) | 0,0% (0,0% a 7,7%; n=46) | 0,0% (0,0% a 7,7%; n=46) |
| classe certa, dado D com classe | - | - |
| P(prever doença dado S), S com "brilho" | 100,0% (78,5% a 100,0%; n=14) | 78,6% (52,4% a 92,4%; n=14) |
| P(prever doença dado S), S sem "brilho" | 100,0% (85,1% a 100,0%; n=22) | 54,5% (34,7% a 73,1%; n=22) |

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
