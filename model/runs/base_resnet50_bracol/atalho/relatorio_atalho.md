# Checagem de atalho: run `base_resnet50_bracol`

Gerado por `python evaluation/atalho.py --run base_resnet50_bracol` em 09/10/2026 21:14, no commit `2bb5855` (com alterações não commitadas). O modelo é o `melhor.pt` do run (época 14), sem retreino. Usa só treino e validação do BRACOL: nenhuma imagem de teste foi lida, e o teste não foi avaliado.

**Por que checar.** As classes do BRACOL vêm em blocos de ids seguidos, e o brilho e o tom do fundo mudam por bloco (`data/reports/eda_bracol.md`, seções 2 e 8). Se cada bloco for uma sessão de fotos, o modelo pode ter aprendido a sessão (luz, fundo, câmera) em vez da lesão. A divisão treino/val/teste é por imagem, então as mesmas sessões aparecem nos três splits.

## A. Acurácia por posição na sequência de ids

### Faixas de 200 ids (validação)

| faixa de ids | imagens | acertos | acurácia | IC 95% (Wilson) | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-200 | 31 | 31 | 100,0% | 89,0% a 100,0% | 9 | 13 | 9 | 0 | 0 |
| 201-400 | 31 | 31 | 100,0% | 89,0% a 100,0% | 2 | 0 | 2 | 27 | 0 |
| 401-600 | 30 | 29 | 96,7% | 83,3% a 99,4% | 3 | 2 | 0 | 22 | 3 |
| 601-800 | 23 | 22 | 95,7% | 79,0% a 99,2% | 7 | 6 | 7 | 1 | 2 |
| 801-1000 | 32 | 29 | 90,6% | 75,8% a 96,8% | 17 | 3 | 10 | 1 | 1 |
| 1001-1200 | 28 | 27 | 96,4% | 82,3% a 99,4% | 2 | 4 | 21 | 0 | 1 |
| 1201-1400 | 33 | 32 | 97,0% | 84,7% a 99,5% | 1 | 25 | 6 | 0 | 1 |
| 1401-1600 | 25 | 22 | 88,0% | 70,0% a 95,8% | 0 | 16 | 3 | 0 | 6 |
| 1601-1747 | 19 | 16 | 84,2% | 62,4% a 94,5% | 0 | 10 | 0 | 1 | 8 |

Arquivo: `faixas_id.csv`. As colunas das classes contam as imagens da validação na faixa.

### Pureza local

Para cada imagem da validação, a fração dos 10 vizinhos por id (5 antes e 5 depois, entre as 1.685 elegíveis de qualquer split, pela classe do manifest) que têm a mesma classe dela. Nas pontas da sequência, contam só os vizinhos que existem. Pureza 1,0: a imagem está no meio de um bloco da própria classe.

| pureza local | imagens | acertos | acurácia | IC 95% (Wilson) | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,0 | 76 | 76 | 100,0% | 95,2% a 100,0% | 20 | 15 | 8 | 33 | 0 |
| de 0,6 a menos de 1,0 | 81 | 79 | 97,5% | 91,4% a 99,3% | 2 | 43 | 19 | 14 | 3 |
| abaixo de 0,6 | 95 | 84 | 88,4% | 80,4% a 93,4% | 19 | 21 | 31 | 5 | 19 |

Arquivo: `pureza_local.csv`. As colunas das classes contam as imagens da validação no grupo.

### Os 13 erros da validação

| id | verdadeira | prevista | prob. da prevista | prob. da verdadeira | faixa de ids | pureza local |
|---|---:|---:|---:|---:|---:|---:|
| 578 | saudavel | phoma | 0,73 | 0,21 | 401-600 | 0,40 |
| 783 | bicho_mineiro | ferrugem | 0,55 | 0,27 | 601-800 | 0,40 |
| 811 | phoma | bicho_mineiro | 0,77 | 0,16 | 801-1000 | 0,00 |
| 817 | bicho_mineiro | ferrugem | 0,89 | 0,06 | 801-1000 | 0,30 |
| 922 | bicho_mineiro | cercosporiose | 0,74 | 0,11 | 801-1000 | 0,30 |
| 1182 | bicho_mineiro | cercosporiose | 0,73 | 0,17 | 1001-1200 | 0,30 |
| 1215 | cercosporiose | bicho_mineiro | 0,61 | 0,10 | 1201-1400 | 0,20 |
| 1410 | cercosporiose | bicho_mineiro | 0,57 | 0,25 | 1401-1600 | 0,10 |
| 1488 | ferrugem | cercosporiose | 0,93 | 0,06 | 1401-1600 | 0,70 |
| 1583 | ferrugem | cercosporiose | 0,63 | 0,36 | 1401-1600 | 0,50 |
| 1642 | cercosporiose | ferrugem | 0,71 | 0,18 | 1601-1747 | 0,20 |
| 1658 | cercosporiose | ferrugem | 0,60 | 0,38 | 1601-1747 | 0,70 |
| 1736 | phoma | cercosporiose | 0,87 | 0,01 | 1601-1747 | 0,00 |

Arquivo: `erros_val.csv`.

## B. Controle de fundo (o modelo "burro")

Uma regressão logística só com a média e o desvio de R, G, B e de H, S, V (12 números por imagem), medidos na imagem já em 448x224, antes da normalização. Padronização com a média e o desvio do treino; ajuste nas 1.180 imagens de treino; medida nas 252 da validação. A borda é a faixa a até 15% (ou 8%) da largura das laterais e da altura do topo e da base.

| variante | área usada | acurácia val | IC 95% (Wilson) | F1 macro val | acurácia treino |
|---|---:|---:|---:|---:|---:|
| (a) imagem inteira | 100,0% | 66,7% | 60,6% a 72,2% | 55,0% | 66,9% |
| (b) borda de 15% | 51,2% | 65,9% | 59,8% a 71,4% | 52,3% | 65,5% |
| (c) borda de 8% | 29,6% | 67,1% | 61,0% a 72,6% | 52,0% | 63,0% |

Referências na validação: acaso 20,0%; sempre a classe majoritária (ferrugem) 31,3%; o modelo do run (época 14) 94,8%, com F1 macro 93,2%.

> **Destaque: só a cor já acerta mais de 50,0% da validação:** (a) imagem inteira 66,7%, (b) borda de 15% 65,9% e (c) borda de 8% 67,1%.

Recall do controle por classe (validação):

| variante | saudavel | ferrugem | bicho_mineiro | phoma | cercosporiose |
|---|---:|---:|---:|---:|---:|
| (a) imagem inteira | 51,2% | 78,5% | 60,3% | 96,2% | 0,0% |
| (b) borda de 15% | 31,7% | 79,7% | 69,0% | 96,2% | 0,0% |
| (c) borda de 8% | 24,4% | 86,1% | 70,7% | 96,2% | 0,0% |

Arquivo: `controle_fundo.json`, com as matrizes de confusão.

## C. Grad-CAM

Amostra da validação: os 13 erros e 4 acertos por classe sorteados com a semente 42 (20 acertos). O mapa é o Grad-CAM da cabeça de classe, no último bloco do layer4 da ResNet50, em float32 e sem AMP. Nos erros há dois mapas: o da classe prevista e o da verdadeira. Arquivos: `gradcam_amostra.csv` e `gradcam_borda.csv`.

**Energia do mapa na borda de 15%.** É a fração da soma do mapa que cai na faixa a até 15% de cada lado, que ocupa 51,2% da área: um mapa uniforme daria 51,2%.

Mapa da classe prevista, por classe verdadeira:

| classe verdadeira | imagens | média na borda | mín. a máx. |
|---|---:|---:|---:|
| saudavel | 5 | 5,7% | 2,1% a 16,1% |
| ferrugem | 6 | 4,4% | 3,5% a 5,7% |
| bicho_mineiro | 8 | 4,1% | 1,5% a 9,3% |
| phoma | 6 | 7,5% | 0,6% a 16,4% |
| cercosporiose | 8 | 5,2% | 0,9% a 13,3% |

Erros x acertos:

| grupo | mapas | média na borda | mín. a máx. |
|---|---:|---:|---:|
| acertos (mapa da prevista) | 20 | 5,3% | 0,9% a 16,4% |
| erros (mapa da prevista) | 13 | 5,4% | 0,6% a 16,1% |
| erros (mapa da verdadeira) | 13 | 16,6% | 0,9% a 67,4% |
| todos (mapa da prevista) | 33 | 5,3% | 0,6% a 16,4% |

**Conferência.** As probabilidades da amostra, recalculadas em float32 (o run avaliou com AMP), diferem das do `predicoes_val.csv` em no máximo 0,0057; a classe prevista é a mesma nas 33 imagens.

**Pranchas.** Cada linha mostra a imagem e o mapa sobreposto; o título traz o id, a classe verdadeira, a prevista e a probabilidade da prevista.

- [gradcam_erros_1.jpg](gradcam_erros_1.jpg)
- [gradcam_erros_2.jpg](gradcam_erros_2.jpg)
- [gradcam_erros_3.jpg](gradcam_erros_3.jpg)
- [gradcam_erros_4.jpg](gradcam_erros_4.jpg)
- [gradcam_acertos_1.jpg](gradcam_acertos_1.jpg)
- [gradcam_acertos_2.jpg](gradcam_acertos_2.jpg)
- [gradcam_acertos_3.jpg](gradcam_acertos_3.jpg)
- [gradcam_acertos_4.jpg](gradcam_acertos_4.jpg)
- [gradcam_acertos_5.jpg](gradcam_acertos_5.jpg)

**É só um proxy.** A folha pode encostar na borda, então energia na borda não é sinônimo de fundo. O mapa sai de uma grade de 7x14 células (o layer4 em 448x224), reescalada para a imagem, e é normalizado por imagem: mostra onde o modelo olhou, de forma grosseira, e não quanto cada região pesou na decisão.

## O que isto mostra e o que NÃO mostra

**Mostra (fatos medidos):**

- Por faixa de 200 ids, a acurácia vai de 84,2% (ids 1601-1747, 19 imagens) a 100,0% (ids 1-200, 31 imagens). Com cerca de 30 imagens por faixa, cada erro vale uns 3,3 pontos.
- **A acurácia cai de forma clara nas imagens de vizinhança misturada:** 88,4% (84 de 95; IC 80,4% a 93,4%) com pureza local abaixo de 0,6, contra 100,0% (76 de 76; IC 95,2% a 100,0%) com pureza 1,0. Os intervalos de Wilson não se tocam.
- Os grupos de pureza não têm a mesma mistura de classes: cercosporiose, a classe de menor recall do modelo (81,8%), tem 19 das 95 imagens do grupo abaixo de 0,6 e 0 das 76 do grupo 1,0.
- **Só a cor já passa de 50,0% de acurácia:** o controle de fundo acerta (a) imagem inteira 66,7%, (b) borda de 15% 65,9% e (c) borda de 8% 67,1% da validação, contra 31,3% de prever sempre a classe majoritária (ferrugem) e 20,0% do acaso; o intervalo de Wilson de cada variante fica inteiro acima da majoritária.
- A borda de 8%, que ocupa 29,6% da área, acerta 67,1%; a imagem inteira, 66,7%.
- Só com a cor da borda de 8%, o recall vai de 0,0% (cercosporiose) a 96,2% (phoma). Na EDA (seção 8), phoma é a classe de fundo mais amarelado.
- Em média, 5,3% da energia do Grad-CAM (mapa da classe prevista) cai na borda de 15%, menos que os 51,2% de um mapa uniforme: os mapas se concentram no interior da foto.
- 0 dos 33 mapas da classe prevista põem na borda mais que 51,2% da energia.
- Nos erros, o mapa da classe verdadeira põe em média 16,6% da energia na borda, contra 5,4% do mapa da prevista; o maior valor é 67,4% (id 1215, mapa de cercosporiose).

**Não mostra:**

- Um resultado bom aqui não prova que não há atalho. A validação tem as mesmas sessões de foto do treino (a divisão é por imagem, não por sessão), e um modelo que usa a sessão também acerta nela.
- A pureza local olha só a posição na sequência de ids. Ids vizinhos da mesma classe não são, com certeza, da mesma sessão: o BRACOL não traz data, câmera nem planta.
- A queda com a pureza baixa pode ter outra causa além da sessão: os grupos não têm a mesma mistura de classes, e uma classe difícil concentrada nas regiões misturadas também baixaria a acurácia delas.
- O controle de fundo mostra que a cor da borda carrega informação da classe, não que o modelo a use: o Grad-CAM, ao mesmo tempo, aponta para o interior da foto.
- O controle de fundo usa 12 estatísticas de cor por região. Uma rede pode usar pistas mais finas (textura do fundo, sombras, luz), então um controle baixo não exclui atalho de fundo. E a imagem inteira inclui a cor da folha e da lesão: o controle (a) alto, sozinho, não indica uso do fundo.
- A borda de 15% ocupa 51,2% da área, e a de 8%, 29,6%. Nas fotos do BRACOL, a folha pode entrar nessas faixas, então a borda não é só fundo.
- O Grad-CAM é grosso (grade de 7x14) e normalizado por imagem. A amostra tem só 33 imagens, e a validação, 252: os intervalos são largos.
- Uma evidência mais forte exigiria validar com blocos de ids inteiros fora do treino, ou com fotos de outra sessão ou do campo. É uma decisão do gestor.

