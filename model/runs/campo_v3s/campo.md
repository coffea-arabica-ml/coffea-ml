## Campo: 318 folhas das fotos de validação do BRACOT

Classificador `model/runs/campo_v3s`. Concordância de classe entre os modos A e B: 27,4%. A estabilidade é a fração de 8 perturbações pequenas (brilho e contraste de 10%, deslocamento de 5%) com a mesma classe. **Proxies não decidem adoção:** um modelo que diz sempre a mesma classe é perfeitamente estável.

|  | modo A | modo B |
|---|---:|---:|
| folhas saudavel | 4 | 69 |
| folhas ferrugem | 69 | 68 |
| folhas bicho_mineiro | 212 | 40 |
| folhas phoma | 31 | 113 |
| folhas cercosporiose | 2 | 28 |
| severidade saudavel | 4 | 69 |
| severidade muito_baixa | 120 | 227 |
| severidade baixa | 194 | 22 |
| severidade alta | 0 | 0 |
| severidade muito_alta | 0 | 0 |
| fração da classe mais comum | 66,7% | 35,5% |
| confiança média | 67,2% | 49,1% |
| estabilidade média | 92,0% | 88,4% |
| estáveis nas 8 perturbações | 72,0% | 63,8% |
| borda: confiança / saudáveis | 68,5% / 2,1% | 47,8% / 24,0% |
| inteiras: confiança / saudáveis | 65,1% / 0,0% | 51,0% / 18,3% |

A severidade prevista no campo não é informativa: o modelo só aprendeu severidade em folhas do BRACOL, e no campo não há rótulo de severidade.

**Rótulos de campo** (feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026; não são de especialista; 120 de 120 casados com as folhas atuais; sem os "?"):

|  | modo A | modo B |
|---|---|---|
| rotuladas (S / D / ?) | 36 / 46 / 38 | 36 / 46 / 38 |
| P(prever doença dado S) | 100,0% (90,4% a 100,0%; n=36) | 50,0% (34,5% a 65,5%; n=36) |
| P(prever saudável dado D) | 0,0% (0,0% a 7,7%; n=46) | 0,0% (0,0% a 7,7%; n=46) |
| classe certa, dado D com classe | - | - |
| P(prever doença dado S), S com "brilho" | 100,0% (78,5% a 100,0%; n=14) | 42,9% (21,4% a 67,4%; n=14) |
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
