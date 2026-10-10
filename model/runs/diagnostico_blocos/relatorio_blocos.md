# Diagnóstico por blocos de ids

Gerado por `python evaluation/diagnostico_blocos.py`. Plano criado em 09/10/2026 23:16, no commit `d66c49a` (com alterações não commitadas). Pool: treino + validação do BRACOL (1.432 imagens); o teste não foi lido. Os modelos deste diagnóstico não foram salvos.

**Decisão do gestor (09/10/2026):** K=5 porque K=4 deixava a classe saudável com 47% das imagens no treino da dobra 1.

## Como ler (regra definida antes dos números)

- **D = F1 macro (aleatório) - F1 macro (blocos)**, em pontos, sobre as predições de fora da dobra juntas: as mesmas imagens nos dois protocolos.
- **D até 3 pontos:** pouco atalho de sessão detectado. **De 3 a 8:** atenção. **Acima de 8:** atalho relevante.
- A queda só é chamada de **clara** se os intervalos de Wilson de 95% da acurácia dos dois protocolos não se tocarem.

## Protocolos e dobras

- **Blocos:** blocos de 100 ids consecutivos (1-100, 101-200, ...); o bloco i fica de fora na dobra i mod 5. O treino da dobra tira as imagens a até 20 ids de um bloco de fora (purga por distância) e as que têm o grupo (folha repetida) no conjunto de fora (purga por grupo).
- **Aleatório:** `StratifiedGroupKFold` em 5 dobras, estratificado pela classe, com os grupos do manifest. O treino de cada dobra é sorteado (estratificado, semente 42) até o mesmo número de imagens do treino de blocos da mesma dobra.
- Cada imagem do pool fica de fora exatamente uma vez em cada protocolo.

| protocolo | dobra | treino | fora | purga (distância) | purga (grupo) | descartadas no sorteio | menor fração de uma classe no treino |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocos | 0 | 989 | 327 | 115 | 1 | 0 | 58,4% (bicho_mineiro) |
| blocos | 1 | 976 | 328 | 128 | 0 | 0 | 54,4% (cercosporiose) |
| blocos | 2 | 1.035 | 281 | 116 | 0 | 0 | 67,5% (saudavel) |
| blocos | 3 | 1.075 | 254 | 102 | 1 | 0 | 58,8% (phoma) |
| blocos | 4 | 1.088 | 242 | 102 | 0 | 0 | 62,8% (phoma) |
| aleatório | 0 | 989 | 287 | 0 | 0 | 156 | 68,8% (cercosporiose) |
| aleatório | 1 | 976 | 286 | 0 | 0 | 170 | 68,0% (saudavel) |
| aleatório | 2 | 1.035 | 287 | 0 | 0 | 110 | 72,0% (cercosporiose) |
| aleatório | 3 | 1.075 | 286 | 0 | 0 | 71 | 74,9% (saudavel) |
| aleatório | 4 | 1.088 | 286 | 0 | 0 | 58 | 75,8% (saudavel) |

Imagens por classe no treino e fora (saudavel / ferrugem / bicho_mineiro / phoma / cercosporiose):

| protocolo | dobra | treino | fora |
|---|---:|---:|---:|
| blocos | 0 | 167 / 316 / 192 / 228 / 86 | 36 / 89 / 123 / 50 / 29 |
| blocos | 1 | 140 / 276 / 219 / 273 / 68 | 68 / 132 / 68 / 21 / 39 |
| blocos | 2 | 156 / 333 / 241 / 218 / 87 | 41 / 85 / 65 / 59 / 31 |
| blocos | 3 | 200 / 326 / 264 / 174 / 111 | 22 / 87 / 47 / 89 / 9 |
| blocos | 4 | 162 / 361 / 278 / 186 / 101 | 64 / 58 / 26 / 77 / 17 |
| aleatório | 0 | 160 / 312 / 227 / 204 / 86 | 46 / 90 / 66 / 60 / 25 |
| aleatório | 1 | 157 / 307 / 225 / 202 / 85 | 47 / 90 / 65 / 59 / 25 |
| aleatório | 2 | 167 / 326 / 238 / 214 / 90 | 46 / 91 / 66 / 59 / 25 |
| aleatório | 3 | 173 / 339 / 247 / 222 / 94 | 46 / 90 / 66 / 59 / 25 |
| aleatório | 4 | 175 / 343 / 250 / 225 / 95 | 46 / 90 / 66 / 59 / 25 |

Arquivos: `plano_dobras.csv` (o papel de cada imagem em cada dobra) e `config.json`.

**Treino:** a configuração do run `base_resnet50_bracol` (`resnet50` pré-treinado, 448x224, AdamW com lr 0,0001 e weight decay 0,0001, batch 32, AMP, RD02, semente 42), mas com 20 épocas fixas, cosseno de T_max 20, sem parada antecipada e com as métricas da última época. O F1 macro por época, em `historico_<protocolo>_<dobra>.csv`, é só informativo. Tempo: 61 min nos 10 treinos; pico de 3,8 GB de memória da GPU.

## Resultado (fora da dobra, 1.432 imagens por protocolo)

| protocolo | acurácia | IC 95% (Wilson) | F1 macro | MAE sev (informativo) | kappa sev (informativo) |
|---|---:|---:|---:|---:|---:|
| aleatório | 90,7% | 89,1% a 92,1% | 88,2% | 0,286 | 0,699 |
| blocos | 89,9% | 88,3% a 91,4% | 87,2% | 0,273 | 0,708 |

**D = 1,0 ponto de F1 macro: pouco atalho de sessão detectado.**

Recall por classe:

| protocolo | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| aleatório | 99,1% | 90,5% | 87,2% | 95,3% | 74,4% |
| blocos | 98,7% | 89,6% | 86,6% | 95,6% | 70,4% |
| diferença (pontos) | 0,4 | 0,9 | 0,6 | -0,3 | 4,0 |

Matriz de confusão, aleatório (linhas: verdadeira; colunas: prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 229 | 0 | 0 | 2 | 0 |
| ferrugem | 3 | 408 | 9 | 0 | 31 |
| bicho_mineiro | 7 | 6 | 287 | 4 | 25 |
| phoma | 1 | 0 | 8 | 282 | 5 |
| cercosporiose | 2 | 13 | 14 | 3 | 93 |

Matriz de confusão, blocos (linhas: verdadeira; colunas: prevista):

| classe verdadeira | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| saudavel | 228 | 1 | 0 | 1 | 1 |
| ferrugem | 11 | 404 | 10 | 1 | 25 |
| bicho_mineiro | 6 | 6 | 285 | 4 | 28 |
| phoma | 0 | 0 | 6 | 283 | 7 |
| cercosporiose | 1 | 16 | 17 | 3 | 88 |

### Por dobra

A última época vale; o maior F1 macro ao longo das épocas é só informativo (escolher a época pelo conjunto de fora seria olhar a resposta).

| protocolo | dobra | fora | acurácia | F1 macro | maior F1 macro nas épocas |
|---|---:|---:|---:|---:|---:|
| aleatório | 0 | 287 | 90,2% | 88,3% | 88,8% (época 13) |
| aleatório | 1 | 286 | 90,6% | 87,9% | 89,3% (época 18) |
| aleatório | 2 | 287 | 89,9% | 86,9% | 88,0% (época 12) |
| aleatório | 3 | 286 | 93,4% | 91,5% | 92,0% (época 13) |
| aleatório | 4 | 286 | 89,5% | 86,8% | 86,8% (época 16) |
| blocos | 0 | 327 | 89,3% | 86,9% | 87,0% (época 17) |
| blocos | 1 | 328 | 83,8% | 80,3% | 81,8% (época 14) |
| blocos | 2 | 281 | 93,2% | 92,0% | 92,4% (época 18) |
| blocos | 3 | 254 | 95,3% | 88,7% | 90,7% (época 17) |
| blocos | 4 | 242 | 89,7% | 81,5% | 82,9% (época 12) |

### Comparação pareada (imagem a imagem)

| certas nos dois | certas só no aleatório | certas só em blocos | erradas nos dois |
|---|---:|---:|---:|
| 1.250 | 49 | 38 | 95 |

### Por pureza local (a dos passos 1c, pela classe do manifest)

| pureza local | imagens | acurácia aleatório | IC 95% | acurácia blocos | IC 95% |
|---|---:|---:|---:|---:|---:|
| 1,0 | 407 | 99,0% | 97,5% a 99,6% | 95,8% | 93,4% a 97,4% |
| de 0,6 a menos de 1,0 | 455 | 92,1% | 89,2% a 94,2% | 92,3% | 89,5% a 94,4% |
| abaixo de 0,6 | 570 | 83,7% | 80,4% a 86,5% | 83,9% | 80,6% a 86,7% |

Arquivos: `predicoes_<protocolo>_<dobra>.csv` e `metricas_blocos.json`.

## Controle de cor nos dois protocolos

A regressão logística só com a cor do passo 1c, ajustada no treino de cada dobra e medida no conjunto de fora, com as predições juntas.

| variante | acurácia aleatório | F1 macro aleatório | acurácia blocos | F1 macro blocos | diferença de F1 (pontos) |
|---|---:|---:|---:|---:|---:|
| (a) imagem inteira | 65,9% (63,4% a 68,3%) | 54,6% | 57,3% (54,8% a 59,9%) | 47,0% | 7,6 |
| (c) borda de 8% | 62,6% (60,1% a 65,1%) | 48,1% | 54,3% (51,7% a 56,9%) | 41,6% | 6,5 |

Arquivo: `controle_cor_blocos.json`.

## O que isto mostra e o que NÃO mostra

**Mostra (fatos medidos):**

- D = 1,0 ponto de F1 macro (88,2% no aleatório e 87,2% em blocos): pela regra, **pouco atalho de sessão detectado**.
- Acurácia: 90,7% (89,1% a 92,1%) no aleatório e 89,9% (88,3% a 91,4%) em blocos. Os intervalos se tocam: a diferença não é clara com estes n.
- Por dobra, o F1 macro vai de 86,8% a 91,5% no aleatório e de 80,3% a 92,0% em blocos.
- A maior queda de recall do aleatório para blocos é em cercosporiose (4,0 pontos).
- Imagem a imagem: 49 certas só no aleatório e 38 certas só em blocos, de 1.432.
- Pureza local 1,0 (407 imagens): 99,0% no aleatório e 95,8% em blocos; **os intervalos não se tocam.**
- Pureza local de 0,6 a menos de 1,0 (455 imagens): 92,1% no aleatório e 92,3% em blocos; os intervalos se tocam.
- Pureza local abaixo de 0,6 (570 imagens): 83,7% no aleatório e 83,9% em blocos; os intervalos se tocam.
- Controle de cor (a) imagem inteira: 65,9% no aleatório e 57,3% em blocos (F1 macro 54,6% e 47,0%, diferença de 7,6 pontos); **os intervalos de acurácia não se tocam: a queda é clara.**
- Controle de cor (c) borda de 8%: 62,6% no aleatório e 54,3% em blocos (F1 macro 48,1% e 41,6%, diferença de 6,5 pontos); **os intervalos de acurácia não se tocam: a queda é clara.**
- Run base `base_resnet50_bracol` intacto (SHA-256 de todos os arquivos antes e depois): sim.

**Não mostra:**

- Um D pequeno não prova que não há atalho: só mede o quanto o desempenho depende de ver imagens vizinhas por id no treino.
- As sessões de foto não são conhecidas: os blocos de ids são um proxy. Uma sessão pode atravessar blocos, e um bloco pode juntar sessões.
- A purga de 20 ids é arbitrária.
- K=5, com uma semente só.
- A GPU não é determinística: repetir os treinos muda os números em cerca de 1 ponto.
- Os protocolos também diferem na mistura de classes do treino (tabela das dobras): em blocos, uma classe concentrada num bloco de fora fica com menos imagens no treino daquela dobra.

