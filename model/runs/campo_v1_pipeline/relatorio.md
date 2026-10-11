# Inferência ponta a ponta: run `campo_v1_pipeline`

Gerado por `python model/avaliar_pipeline.py` em 10/10/2026 14:30, no commit `451c33a` (com alterações não commitadas). Classificador `model/runs/campo_v1`, detector `model/runs/detector_base/weights/last.pt`. **Os testes do BRACOL e do BRACOT não foram lidos nem avaliados.** Módulo: `model/inferencia.py`; interface no README (seção "Interface para o backend").

## Decisões do gestor

- **10/10/2026:** o detector está bom o bastante: AP50 de máscara de 94,7% (região) e 88,1% (conservador), limiar 0,35.
- **10/10/2026:** classificar também as folhas cortadas pela borda, com área mínima e corte por confiança do classificador; reavaliar o filtro de borda olhando os recortes.
- **10/10/2026:** folha única: a foto inteira só se houver detecção com score >= 0,05; sem nenhuma, planta_nao_identificada.
- **10/10/2026:** arquivo_muito_grande acima de ~64 megapixels, com as dimensões lidas do cabeçalho.

## Configuração da cadeia

| parâmetro | valor | origem |
|---|---|---|
| limiar_deteccao | 0,35 | decisão do gestor |
| max_folhas | 15 | RNF02 (a validação do BRACOT tem até 13 detecções) |
| area_minima | 0,5% | pedido do gestor (cortadas: T2) |
| saturacao_minima / fracao_colorida_minima | 30 / 25% | filtro de cor, medido no treino (abaixo) |
| limiar_confianca | 0,61 | proposta da curva risco-cobertura; **aguarda a decisão do gestor** |
| recorte | A | T1: só o recorte acerta 87,6% no A e 85,3% no B (abaixo) |
| margem | 17,0% | medida no treino do BRACOL (abaixo) |
| cinza_fundo | 197 | medido no treino do BRACOL (abaixo) |
| cobertura_folha_unica | 30% | entre o BRACOL e o BRACOT de treino (abaixo) |
| score_minimo_folha_unica | 0,05 | decisão do gestor |
| max_megapixels | 64 | decisão do gestor |

## Constantes medidas no treino

**Cinza do fundo.** Mediana do tom da borda de 8% de cada uma das 1.180 imagens de treino do BRACOL, já em 448x224. O tom do fundo identifica a sessão de fotos (checagem de atalho), por isso cada classe pesa igual: cinza neutro 197 (em uso: 197).

| classe | R | G | B | luminância |
|---|---:|---:|---:|---:|
| saudavel | 188 | 184 | 177 | 183 |
| ferrugem | 207 | 202 | 200 | 203 |
| bicho_mineiro | 198 | 195 | 191 | 194 |
| phoma | 211 | 199 | 194 | 201 |
| cercosporiose | 208 | 200 | 196 | 202 |

**Filtro de cor.** Nas fotos de folha sobre fundo liso do BRACOL, o detector também acha pedaços do fundo. Das 373 detecções no limiar nas 200 imagens de treino, 130 têm menos de 25% da máscara com saturação >= 30 e saem (histograma da fração colorida, de 0 a 1 em passos de 0,1: [109, 15, 10, 10, 10, 9, 14, 13, 31, 152]). Nas 450 folhas anotadas de 80 fotos de treino do BRACOT, a fração colorida tem mediana 99,0%, p1 35,3% e mínimo 14,9%; 2 ficariam abaixo do mínimo.

**Margem.** Em 200 imagens de treino do BRACOL (amostra estratificada, semente 42), a maior folha detectada tem comprimento mediano de 74,7% da largura da foto: margem de 17,0% de cada lado (em uso: 17,0%). Detecções por imagem: nenhuma em 10, uma em 148, mais de uma em 42.

**Cobertura da folha única.** A caixa da maior folha cobre de 17,3% a 78,2% da foto no BRACOL (p5 28,6%, mediana 50,1%). Nas 194 fotos de treino do BRACOT (sem a validação), a maior caixa cobre no máximo 61,4% (p95 39,0%), e 0 foto(s) têm uma só folha utilizável. Regra 1 a partir de 30%.

## Limiar de confiança (RF07)

Curva risco-cobertura: com o limiar t, a folha com probabilidade da classe prevista abaixo de t sai da resposta (todas abaixo: `baixa_confianca`). Fonte principal: as 1.432 predições fora da dobra do diagnóstico por blocos (`model/runs/diagnostico_blocos`, protocolo aleatório; acurácia 90,7% sem limiar). Conferência: as 252 imagens de validação no classificador avaliado (imagem inteira, T1).

![Curva risco-cobertura](risco_cobertura.png)

| limiar | cobertura (fora da dobra) | acurácia das mantidas (IC 95%) | cobertura saudavel | cobertura ferrugem | cobertura bicho_mineiro | cobertura phoma | cobertura cercosporiose | cobertura (val) | acurácia (val) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,50 | 95,3% | 93,0% (91,6% a 94,3%) | 98,7% | 95,3% | 92,1% | 98,0% | 91,2% | 93,3% | 93,2% |
| 0,60 | 88,8% | 94,9% (93,5% a 96,0%) | 96,5% | 87,8% | 84,2% | 96,6% | 71,2% | 84,9% | 96,3% |
| 0,61 | 88,1% | 95,1% (93,7% a 96,1%) | 96,1% | 86,9% | 83,6% | 95,9% | 70,4% | 83,7% | 96,2% |
| 0,70 | 81,8% | 97,3% (96,2% a 98,1%) | 93,9% | 80,0% | 75,1% | 92,2% | 58,4% | 77,8% | 97,4% |
| 0,80 | 71,1% | 98,5% (97,6% a 99,1%) | 88,7% | 68,1% | 60,8% | 86,8% | 39,2% | 69,8% | 98,3% |
| 0,90 | 55,3% | 99,2% (98,4% a 99,7%) | 81,4% | 50,3% | 38,9% | 76,4% | 18,4% | 49,2% | 100,0% |
| 0,95 | 38,1% | 99,8% (99,0% a 100,0%) | 64,9% | 32,8% | 21,0% | 57,1% | 7,2% | 32,9% | 100,0% |

**Proposta: 0,61**, o menor limiar em que a acurácia das mantidas fora da dobra chega a 95%. **A decisão é do gestor.**
Em uso nesta avaliação: 0,61. Arquivo: `risco_cobertura.csv`.

## T1: validação do BRACOL pela cadeia

As 252 imagens de validação do BRACOL (uma folha por foto, rótulo conhecido) pela cadeia inteira e pela imagem inteira, com o mesmo código (`inferencia.classificar`). A folha avaliada é a de maior área. "Só o recorte" desliga a regra 1, para medir o recorte numa folha com rótulo. Sem limiar de confiança.

| variante | com previsão | acurácia (IC 95%) | F1 macro | recall saudavel | recall ferrugem | recall bicho_mineiro | recall phoma | recall cercosporiose | severidade: exata / ±1 | no limiar 0,61: cobertura / acurácia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| imagem inteira (a referência) | 100,0% | 89,3% (84,9% a 92,5%) | 87,1% | 95,1% | 91,1% | 82,8% | 92,3% | 81,8% | 73,0% / 94,4% | 83,7% / 96,2% |
| cadeia, recorte A | 99,6% | 89,6% (85,3% a 92,8%) | 87,3% | 95,1% | 91,1% | 86,2% | 92,2% | 77,3% | 73,7% / 94,4% | 83,7% / 95,7% |
| cadeia, recorte B | 99,6% | 88,0% (83,5% a 91,5%) | 85,6% | 95,1% | 88,6% | 82,8% | 92,2% | 77,3% | 73,7% / 94,0% | 81,3% / 96,1% |
| só o recorte A (sem a regra 1) | 99,6% | 87,6% (83,0% a 91,2%) | 85,1% | 92,7% | 89,9% | 82,8% | 92,2% | 72,7% | 74,1% / 94,8% | 83,3% / 95,2% |
| só o recorte B (sem a regra 1) | 99,6% | 85,3% (80,3% a 89,1%) | 82,6% | 92,7% | 86,1% | 75,9% | 94,1% | 72,7% | 75,3% / 94,4% | 75,8% / 95,3% |

**Regras de folha única nas imagens do BRACOL** (cadeia):

- regra 1 (uma folha, caixa com 30% da foto ou mais): 191;
- regra 2 (nenhuma folha utilizável, alguma detecção com cor de folha e score >= 0,05): 12;
- regra 3 (`planta_nao_identificada`): 1;
- pelas detecções: 48, das quais 38 com mais de uma folha.

**Correção das regras depois do T1.** Na primeira rodada, sem o filtro de cor, o detector achava pedaços do fundo liso do BRACOL como folhas: 135 imagens saíam com mais de uma folha, e a regra 1 só pegava 105. O filtro de cor (constantes, acima) tira esses pedaços, e a regra 2 passou a exigir uma detecção com cor de folha, porque um pedaço de fundo não é sinal de folha.

Detecções utilizáveis por imagem: 0: 13, 1: 201, 2: 31, 3: 7; o filtro de cor tirou 173 detecções de fundo. Conferência da imagem inteira com o `predicoes_val.csv` do run: diferença máxima das probabilidades 4.1e-03 (o treino usou AMP), classe prevista igual em 252 de 252. Arquivo: `t1_imagens.csv`.

## T2: validação do BRACOT (sem rótulo de classe)

As 46 fotos de validação do detector (cenas inteiras do treino dos autores): 321 detecções no limiar, 1 cortadas pela área mínima, 2 pelo filtro de cor, 0 pelo número máximo; 318 folhas classificadas, 192 cortadas pela borda. Planos: {'deteccao': 46}. Classe igual nos dois modos em 53,8% das folhas.

|  | modo A | modo B |
|---|---:|---:|
| folhas saudavel | 1 | 19 |
| folhas ferrugem | 179 | 86 |
| folhas bicho_mineiro | 10 | 21 |
| folhas phoma | 123 | 171 |
| folhas cercosporiose | 5 | 21 |
| severidade saudavel | 1 | 19 |
| severidade muito_baixa | 101 | 243 |
| severidade baixa | 216 | 56 |
| severidade alta | 0 | 0 |
| severidade muito_alta | 0 | 0 |
| confiança média / mediana | 51,3% / 48,4% | 43,8% / 39,9% |
| abaixo de 0,50 | 55,0% | 71,4% |
| abaixo de 0,60 | 73,3% | 84,6% |
| abaixo de 0,61 | 73,9% | 86,2% |
| abaixo de 0,70 | 84,6% | 91,8% |
| abaixo de 0,80 | 92,5% | 98,1% |
| abaixo de 0,90 | 97,5% | 99,4% |
| cortadas pela borda: confiança média | 50,6% | 41,7% |
| inteiras: confiança média | 52,3% | 47,1% |
| cortadas pela borda abaixo de 0,61 | 75,5% | 89,1% |
| inteiras abaixo de 0,61 | 71,4% | 81,7% |
| fotos sem nenhuma folha acima de 0,50 (baixa_confianca) | 4 de 46 | 7 de 46 |
| fotos sem nenhuma folha acima de 0,60 (baixa_confianca) | 9 de 46 | 17 de 46 |
| fotos sem nenhuma folha acima de 0,61 (baixa_confianca) | 9 de 46 | 19 de 46 |
| fotos sem nenhuma folha acima de 0,70 (baixa_confianca) | 19 de 46 | 27 de 46 |
| fotos sem nenhuma folha acima de 0,80 (baixa_confianca) | 30 de 46 | 40 de 46 |
| fotos sem nenhuma folha acima de 0,90 (baixa_confianca) | 38 de 46 | 44 de 46 |

Arquivo: `t2_folhas.csv` (uma linha por folha) e `t2_fotos.csv`.

### O que vi nos recortes (sem imagens no repositório)

Ainda não descrito (falta o observacoes.md do run).

## T3: tempo em CPU (RNF02: diagnóstico em até 5 s)

Nesta máquina, em CPU (8 threads do torch), nas 46 fotos de validação do BRACOT (4032x3024), depois de uma passada de aquecimento. Leitura dos pesos dos dois modelos, uma vez ao subir o backend: 0,3 s, com o torch e o ultralytics já importados (o import leva alguns segundos a mais na primeira vez).

| etapa | mediana (s) | máximo (s) |
|---|---:|---:|
| leitura | 0,07 | 0,08 |
| detecção | 0,19 | 0,22 |
| recortes | 0,08 | 0,10 |
| classificação (lote) | 0,48 | 1,08 |
| contrato | 0,00 | 0,00 |
| **total** | 0,82 | 1,43 |

Folhas por foto: mediana 6, máximo 13; 0 fotos passaram de 5 s. Pior caso, com 15 folhas: a classificação leva 1,25 s (máximo 1,31 s), e o total estimado (maior leitura, maior detecção, recortes e classificação de 15 folhas) é 1,79 s. A CPU do backend pode ser mais lenta que esta. Arquivo: `t3_tempos.csv`.

## Acessos à rede

Nenhum: o classificador é montado sem pré-treino e lê só o `melhor.pt`; o detector roda com o modo offline do ultralytics imposto (`detector.importar_ultralytics`).

## Limites

- O limiar de confiança é calibrado em folhas do BRACOL (fundo claro, uma folha por foto); no campo (T2) a confiança é outra, e não há rótulo para medir a acurácia ali.
- As predições fora da dobra vêm de modelos de 20 épocas, mais fracos que o run base: o limiar proposto tende a ser conservador para ele.
- `especie_incorreta` não é produzido: não há dados de outras plantas. Uma foto sem café pode sair como folha de café com confiança alta.
- O T2 não tem rótulo de classe: as observações dos recortes são uma inspeção visual, não uma medida.

