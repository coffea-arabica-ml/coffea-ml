# Inferência ponta a ponta: run `pipeline_padrao_b`

Gerado por `python model/avaliar_pipeline.py` em 11/10/2026 10:47, no commit `71ca4a4` (com alterações não commitadas). Classificador `model/runs/base_resnet50_bracol`, detector `model/runs/detector_base/weights/last.pt`. **Os testes do BRACOL e do BRACOT não foram lidos nem avaliados.** Módulo: `model/inferencia.py`; interface no README (seção "Interface para o backend").

## Decisões do gestor

- **10/10/2026:** o detector está bom o bastante: AP50 de máscara de 94,7% (região) e 88,1% (conservador), limiar 0,35.
- **10/10/2026:** classificar também as folhas cortadas pela borda, com área mínima e corte por confiança do classificador; reavaliar o filtro de borda olhando os recortes.
- **10/10/2026:** folha única: a foto inteira só se houver detecção com score >= 0,05; sem nenhuma, planta_nao_identificada.
- **10/10/2026:** arquivo_muito_grande acima de ~64 megapixels, com as dimensões lidas do cabeçalho.
- **11/10/2026:** recorte B por padrão (o A continua como opção), pelos rótulos de campo do passo 4.
- **11/10/2026:** filtro de fragmentos ligado com fracao_visivel_minima = 0,5; folha sozinha inteira não é filtrada; se o filtro tirar todas as folhas, baixa_confianca pedindo para enquadrar a folha inteira.
- **11/10/2026:** sem validação cruzada das variantes do passo 4: o modelo oficial segue o base_resnet50_bracol. O limiar de confiança segue pendente.

## Configuração da cadeia

| parâmetro | valor | origem |
|---|---|---|
| limiar_deteccao | 0,35 | decisão do gestor |
| max_folhas | 15 | RNF02 (a validação do BRACOT tem até 13 detecções) |
| area_minima | 0,5% | pedido do gestor (cortadas: T2) |
| saturacao_minima / fracao_colorida_minima | 30 / 25% | filtro de cor, medido no treino (abaixo) |
| limiar_confianca | 0,61 | proposta da curva risco-cobertura; **aguarda a decisão do gestor** |
| recorte | B | decisão do gestor (11/10/2026), pelos rótulos de campo do passo 4; no T1 só o recorte acerta 92,4% no A e 88,4% no B (abaixo) |
| margem | 17,0% | medida no treino do BRACOL (abaixo) |
| cinza_fundo | 197 | medido no treino do BRACOL (abaixo) |
| cobertura_folha_unica | 30% | entre o BRACOL e o BRACOT de treino (abaixo) |
| score_minimo_folha_unica | 0,05 | decisão do gestor |
| max_megapixels | 64 | decisão do gestor |
| fracao_visivel_minima | 0,50 | filtro de fragmentos; decisão do gestor (11/10/2026) |

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
| 0,50 | 95,3% | 93,0% (91,6% a 94,3%) | 98,7% | 95,3% | 92,1% | 98,0% | 91,2% | 97,2% | 94,7% |
| 0,60 | 88,8% | 94,9% (93,5% a 96,0%) | 96,5% | 87,8% | 84,2% | 96,6% | 71,2% | 93,3% | 95,3% |
| 0,61 | 88,1% | 95,1% (93,7% a 96,1%) | 96,1% | 86,9% | 83,6% | 95,9% | 70,4% | 92,1% | 96,1% |
| 0,70 | 81,8% | 97,3% (96,2% a 98,1%) | 93,9% | 80,0% | 75,1% | 92,2% | 58,4% | 88,5% | 96,4% |
| 0,80 | 71,1% | 98,5% (97,6% a 99,1%) | 88,7% | 68,1% | 60,8% | 86,8% | 39,2% | 82,5% | 98,6% |
| 0,90 | 55,3% | 99,2% (98,4% a 99,7%) | 81,4% | 50,3% | 38,9% | 76,4% | 18,4% | 70,6% | 99,4% |
| 0,95 | 38,1% | 99,8% (99,0% a 100,0%) | 64,9% | 32,8% | 21,0% | 57,1% | 7,2% | 60,3% | 100,0% |

**Proposta: 0,61**, o menor limiar em que a acurácia das mantidas fora da dobra chega a 95%. **A decisão é do gestor.**
Em uso nesta avaliação: 0,61. Arquivo: `risco_cobertura.csv`.

## T1: validação do BRACOL pela cadeia

As 252 imagens de validação do BRACOL (uma folha por foto, rótulo conhecido) pela cadeia inteira e pela imagem inteira, com o mesmo código (`inferencia.classificar`). A folha avaliada é a de maior área. "Só o recorte" desliga a regra 1, para medir o recorte numa folha com rótulo. Sem limiar de confiança.

| variante | com previsão | acurácia (IC 95%) | F1 macro | recall saudavel | recall ferrugem | recall bicho_mineiro | recall phoma | recall cercosporiose | severidade: exata / ±1 | no limiar 0,61: cobertura / acurácia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| imagem inteira (a referência) | 100,0% | 94,8% (91,4% a 97,0%) | 93,2% | 97,6% | 97,5% | 93,1% | 96,2% | 81,8% | 77,0% / 96,0% | 92,1% / 96,1% |
| cadeia, recorte A | 99,6% | 94,4% (90,9% a 96,6%) | 93,1% | 97,6% | 97,5% | 89,7% | 94,1% | 90,9% | 76,5% / 96,0% | 92,9% / 96,2% |
| cadeia, recorte B | 99,6% | 94,0% (90,4% a 96,3%) | 92,9% | 97,6% | 94,9% | 91,4% | 94,1% | 90,9% | 76,5% / 95,6% | 91,7% / 95,2% |
| só o recorte A (sem a regra 1) | 99,6% | 92,4% (88,5% a 95,1%) | 90,0% | 97,6% | 94,9% | 89,7% | 94,1% | 77,3% | 78,5% / 95,6% | 90,1% / 96,9% |
| só o recorte B (sem a regra 1) | 99,6% | 88,4% (83,9% a 91,8%) | 86,2% | 97,6% | 88,6% | 81,0% | 94,1% | 77,3% | 77,7% / 94,8% | 85,3% / 95,3% |

**Regras de folha única nas imagens do BRACOL** (cadeia):

- regra 1 (uma folha, caixa com 30% da foto ou mais): 191;
- regra 2 (nenhuma folha utilizável, alguma detecção com cor de folha e score >= 0,05): 12;
- regra 3 (`planta_nao_identificada`): 1;
- pelas detecções: 48, das quais 38 com mais de uma folha.

**Correção das regras depois do T1.** Na primeira rodada, sem o filtro de cor, o detector achava pedaços do fundo liso do BRACOL como folhas: 135 imagens saíam com mais de uma folha, e a regra 1 só pegava 105. O filtro de cor (constantes, acima) tira esses pedaços, e a regra 2 passou a exigir uma detecção com cor de folha, porque um pedaço de fundo não é sinal de folha.

Detecções utilizáveis por imagem: 0: 13, 1: 201, 2: 31, 3: 7; o filtro de cor tirou 173 detecções de fundo. Conferência da imagem inteira com o `predicoes_val.csv` do run: diferença máxima das probabilidades 7.1e-03 (o treino usou AMP), classe prevista igual em 252 de 252. Arquivo: `t1_imagens.csv`.

## T2: validação do BRACOT (sem rótulo de classe)

As 46 fotos de validação do detector (cenas inteiras do treino dos autores): 321 detecções no limiar, 1 cortadas pela área mínima, 2 pelo filtro de cor, 79 pelo filtro de fragmentos, 0 pelo número máximo; 239 folhas classificadas, 117 cortadas pela borda. Planos: {'deteccao': 46}. Classe igual nos dois modos em 43,5% das folhas.

|  | modo A | modo B |
|---|---:|---:|
| folhas saudavel | 0 | 24 |
| folhas ferrugem | 174 | 96 |
| folhas bicho_mineiro | 26 | 56 |
| folhas phoma | 39 | 59 |
| folhas cercosporiose | 0 | 4 |
| severidade saudavel | 0 | 24 |
| severidade muito_baixa | 47 | 149 |
| severidade baixa | 192 | 66 |
| severidade alta | 0 | 0 |
| severidade muito_alta | 0 | 0 |
| confiança média / mediana | 53,3% / 50,6% | 42,1% / 35,7% |
| abaixo de 0,50 | 49,0% | 74,5% |
| abaixo de 0,60 | 65,7% | 86,2% |
| abaixo de 0,61 | 66,9% | 87,0% |
| abaixo de 0,70 | 78,2% | 91,2% |
| abaixo de 0,80 | 89,5% | 94,1% |
| abaixo de 0,90 | 97,1% | 98,3% |
| cortadas pela borda: confiança média | 50,9% | 39,6% |
| inteiras: confiança média | 55,6% | 44,5% |
| cortadas pela borda abaixo de 0,61 | 69,2% | 88,0% |
| inteiras abaixo de 0,61 | 64,8% | 86,1% |
| fotos sem nenhuma folha acima de 0,50 (baixa_confianca) | 6 de 46 | 15 de 46 |
| fotos sem nenhuma folha acima de 0,60 (baixa_confianca) | 11 de 46 | 24 de 46 |
| fotos sem nenhuma folha acima de 0,61 (baixa_confianca) | 12 de 46 | 25 de 46 |
| fotos sem nenhuma folha acima de 0,70 (baixa_confianca) | 16 de 46 | 31 de 46 |
| fotos sem nenhuma folha acima de 0,80 (baixa_confianca) | 30 de 46 | 37 de 46 |
| fotos sem nenhuma folha acima de 0,90 (baixa_confianca) | 40 de 46 | 42 de 46 |

Arquivo: `t2_folhas.csv` (uma linha por folha) e `t2_fotos.csv`.

### Observações escritas à mão (sem imagens no repositório)

**Efeito dos padrões de 11/10/2026 (recorte B e filtro de fragmentos em 0,5), contra o
`pipeline_base` (recorte A, sem filtro).** Mesmo classificador, mesmo detector e mesmas fotos;
nenhum recorte foi olhado neste run.

- **T1 (BRACOL):**
  - **Cadeia:** a que está em uso, a B, acerta 94,0%, contra 94,4% da A: queda de 0,4 ponto.
    Os intervalos se sobrepõem.
  - **Só o recorte (sem a regra 1):** cai de 92,4% (A) para 88,4% (B), uma queda de 4,0
    pontos. A regra 1 (foto inteira) cobre quase toda a diferença no BRACOL. O A rende mais lá
    porque o fundo das fotos carrega a sessão: o T1 não serve de evidência para o modo de
    recorte.
  - **Filtro de fragmentos:** tirou 1 folha de 1 imagem (id 1427, de 3 para 2 folhas), sem
    mudar a classe. Nas outras 251 imagens, as folhas, os planos, as regras e as classes são
    os mesmos do `pipeline_base`.
  - **Limiar de 0,61:** cobertura e acurácia iguais às do `pipeline_base`. Na cadeia B,
    cobertura de 91,7% e acurácia de 95,2%; na A, 92,9% e 96,2%.
- **T2 (campo, sem rótulo de classe):**
  - **Folhas classificadas:** caem de 318 para 239. O filtro de fragmentos tira 79, e as
    cortadas pela borda caem de 192 para 117.
  - **Os "?" dos rótulos de campo:** 51% entre as folhas que o filtro tira, contra 24% entre as
    que ficam (passo 4).
  - **Classes no modo B:**
    - saudável cai de 42 para 24 (de 13% para 10% das folhas);
    - ferrugem, de 126 para 96;
    - bicho-mineiro, de 70 para 56;
    - phoma, de 75 para 59;
    - cercosporiose, de 5 para 4.
  - **Fotos em `baixa_confianca` no limiar de 0,61:** no modo B, sobem de 20 para 25 de 46; no
    A, de 8 para 12. Nas 5 fotos a mais (4 no A), as únicas folhas acima do limiar eram as que
    o filtro tira.
  - **O filtro nunca tirou todas as folhas de uma foto.** A fração visível é relativa à
    mediana da foto, então pelo menos metade das folhas fica, e uma folha sozinha tem fração 1.
    Só com `preenchimento_minimo` ligado (hoje em 0) aparece o erro de fragmentos.
- **T3 (CPU):** continua abaixo dos 5 s do RNF02 (tabela acima).
  - **Contra o `pipeline_base`:** a mediana vai de 0,82 para 0,90 s, e o pior caso estimado
    (15 folhas), de 1,75 para 2,44 s.
  - **Parte disso é variação de medida nesta máquina.** A classificação de 15 folhas usa a
    mesma rede, com a mesma entrada nos dois modos, e mesmo assim foi de 1,25 para 1,40 s.
  - **Uma rodada anterior com os mesmos padrões** deu 0,88 e 2,38 s.
  - O modo B só acrescenta o preenchimento do fundo em cinza no recorte.

## T3: tempo em CPU (RNF02: diagnóstico em até 5 s)

Nesta máquina, em CPU (8 threads do torch), nas 46 fotos de validação do BRACOT (4032x3024), depois de uma passada de aquecimento. Leitura dos pesos dos dois modelos, uma vez ao subir o backend: 0,5 s, com o torch e o ultralytics já importados (o import leva alguns segundos a mais na primeira vez).

| etapa | mediana (s) | máximo (s) |
|---|---:|---:|
| leitura | 0,08 | 0,10 |
| detecção | 0,21 | 0,27 |
| recortes | 0,18 | 0,38 |
| classificação (lote) | 0,42 | 1,06 |
| contrato | 0,00 | 0,00 |
| **total** | 0,90 | 1,64 |

Folhas por foto: mediana 5, máximo 11; 0 fotos passaram de 5 s. Pior caso, com 15 folhas: a classificação leva 1,40 s (máximo 1,52 s), e o total estimado (maior leitura, maior detecção, recortes e classificação de 15 folhas) é 2,44 s. A CPU do backend pode ser mais lenta que esta. Arquivo: `t3_tempos.csv`.

## Acessos à rede

Nenhum: o classificador é montado sem pré-treino e lê só o `melhor.pt`; o detector roda com o modo offline do ultralytics imposto (`detector.importar_ultralytics`).

## Limites

- O limiar de confiança é calibrado em folhas do BRACOL (fundo claro, uma folha por foto); no campo (T2) a confiança é outra, e não há rótulo para medir a acurácia ali.
- As predições fora da dobra vêm de modelos de 20 épocas, mais fracos que o run base: o limiar proposto tende a ser conservador para ele.
- `especie_incorreta` não é produzido: não há dados de outras plantas. Uma foto sem café pode sair como folha de café com confiança alta.
- O T2 não tem rótulo de classe: as observações dos recortes são uma inspeção visual, não uma medida.

