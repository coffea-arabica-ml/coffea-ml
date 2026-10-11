## Campo: 318 folhas das fotos de validação do BRACOT

Classificador `model/runs/campo_v1g`. Concordância de classe entre os modos A e B: 49,4%. A estabilidade é a fração de 8 perturbações pequenas (brilho e contraste de 10%, deslocamento de 5%) com a mesma classe. **Proxies não decidem adoção:** um modelo que diz sempre a mesma classe é perfeitamente estável.

|  | modo A | modo B |
|---|---:|---:|
| folhas saudavel | 1 | 29 |
| folhas ferrugem | 262 | 149 |
| folhas bicho_mineiro | 26 | 47 |
| folhas phoma | 28 | 85 |
| folhas cercosporiose | 1 | 8 |
| severidade saudavel | 1 | 29 |
| severidade muito_baixa | 95 | 224 |
| severidade baixa | 222 | 65 |
| severidade alta | 0 | 0 |
| severidade muito_alta | 0 | 0 |
| fração da classe mais comum | 82,4% | 46,9% |
| confiança média | 51,5% | 39,8% |
| estabilidade média | 89,9% | 85,6% |
| estáveis nas 8 perturbações | 70,4% | 57,5% |
| borda: confiança / saudáveis | 50,0% / 0,5% | 37,7% / 12,0% |
| inteiras: confiança / saudáveis | 53,8% / 0,0% | 42,9% / 4,8% |

A severidade prevista no campo não é informativa: o modelo só aprendeu severidade em folhas do BRACOL, e no campo não há rótulo de severidade.

**Rótulos de campo** (feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026; não são de especialista; 120 de 120 casados com as folhas atuais; sem os "?"):

|  | modo A | modo B |
|---|---|---|
| rotuladas (S / D / ?) | 36 / 46 / 38 | 36 / 46 / 38 |
| P(prever doença dado S) | 100,0% (90,4% a 100,0%; n=36) | 83,3% (68,1% a 92,1%; n=36) |
| P(prever saudável dado D) | 0,0% (0,0% a 7,7%; n=46) | 0,0% (0,0% a 7,7%; n=46) |
| classe certa, dado D com classe | - | - |
| P(prever doença dado S), S com "brilho" | 100,0% (78,5% a 100,0%; n=14) | 92,9% (68,5% a 98,7%; n=14) |
| P(prever doença dado S), S sem "brilho" | 100,0% (85,1% a 100,0%; n=22) | 77,3% (56,6% a 89,9%; n=22) |

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
