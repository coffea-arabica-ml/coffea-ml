# Detector de folhas: run `detector_base`

Gerado por `python model/treinar_detector.py` em 10/10/2026 02:28, no commit `bd6bcf8` (com alterações não commitadas). YOLO11s-seg (ultralytics 8.4.175), ajustado nas fotos de treino do BRACOT. **O teste dos autores não foi lido nem avaliado.**

## Decisões do gestor

- **09/10/2026:** o BRACOT é a fonte do detector, com a divisão oficial dos autores (240 treino / 60 teste); o teste fica fechado até o modelo final.
- **10/10/2026:** anotação parcial: no treino, cinza fora do casco das folhas anotadas (margem de 2% do lado maior); validação e inferência na foto inteira; métrica principal ignorando a previsão com mais da metade da máscara fora da região, mais a conservadora; contar e olhar as detecções da periferia.
- **10/10/2026:** ferramenta: YOLO11s-seg (ultralytics), com a licença AGPL-3.0 aceita para este projeto acadêmico de repositórios públicos.
- **10/10/2026:** instalação: OpenCV só headless 5.0.0.93; ultralytics 8.4.175 com --no-deps e as 5 dependências que faltam, fixadas; o pip check acusa o opencv-python, o que é esperado.

## Dados

- Treino: 194 fotos, 61 cenas, 1.070 folhas, com cinza (114) fora da região anotada.
- Validação: 46 fotos, 24 cenas, 248 folhas, na foto inteira (bracol.dividir, estratificado por dia, grupo = cena, semente 42, 20% das fotos; lista em `fotos_validacao.csv`).

## Treino

- Peso inicial: `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt` (SHA-256 `1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3`), licença AGPL-3.0.
- 640 px, batch 16, semente 42, determinístico, sem AMP (a checagem de AMP do ultralytics baixaria outro peso), sem gráficos (não grava imagens), workers 0. O modelo do run é o `weights/last.pt`, depois das épocas fixas. Os demais argumentos do ultralytics estão no `config.json`.
- Inferência (`detector.detectar_folhas`, a mesma da avaliação e do RNF02): a foto inteira reduzida a 640 px no lado maior (INTER_AREA), com moldura de 16 px de cinza 114, como o validador do ultralytics monta a entrada. Sem a moldura, a borda da foto coincide com a borda da entrada, o que o treino quase não mostra; a comparação está no docstring de `model/detector.py`.
- 60 épocas em 32,1 min (32,1 s por época, com a validação do ultralytics); histórico em `historico.csv`.

## Validação

**Como ler.** A métrica principal (região) ignora a previsão sem folha anotada correspondente que tem mais da metade da máscara fora da região anotada; a conservadora conta toda previsão sem par como erro, mesmo que seja uma folha real que os autores não contornaram.

| modo | AP50 máscara | AP50-95 máscara | AP50 caixa | AP50-95 caixa |
|---|---:|---:|---:|---:|
| região (principal) | 94,7% | 88,5% | 94,8% | 86,9% |
| conservador | 88,1% | 82,9% | 88,3% | 81,3% |

Conferência com o validador do ultralytics (foto inteira, comparável ao conservador): máscara AP50 88,5% e AP50-95 82,0%; caixa AP50 88,5% e AP50-95 81,3%. Pequenas diferenças são esperadas: o ultralytics interpola a curva de precisão até o recall 1, casa previsões e folhas de outro jeito (cada previsão só concorre pela folha de maior IoU), mede a máscara em 1/4 da entrada e reduz a foto preparada de 1.280 px com INTER_LINEAR.

### Curva por limiar de score (máscara, IoU 0,5)

| limiar | previstas | fora da região | precisão (região) | recall | F1 (região) | precisão (conservador) | F1 (conservador) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0,05 | 407 | 116 | 80,4% | 94,4% | 86,8% | 57,5% | 71,5% |
| 0,10 | 366 | 96 | 83,3% | 90,7% | 86,9% | 61,5% | 73,3% |
| 0,15 | 351 | 89 | 84,7% | 89,5% | 87,1% | 63,2% | 74,1% |
| 0,20 | 340 | 84 | 86,3% | 89,1% | 87,7% | 65,0% | 75,2% |
| 0,25 | 335 | 83 | 87,3% | 88,7% | 88,0% | 65,7% | 75,5% |
| 0,30 | 326 | 77 | 88,4% | 88,7% | 88,5% | 67,5% | 76,7% |
| 0,35 | 321 | 75 | 89,0% | 88,3% | 88,7% | 68,2% | 77,0% |
| 0,40 | 318 | 72 | 89,0% | 88,3% | 88,7% | 68,9% | 77,4% |
| 0,45 | 310 | 66 | 89,3% | 87,9% | 88,6% | 70,3% | 78,1% |
| 0,50 | 301 | 61 | 90,0% | 87,1% | 88,5% | 71,8% | 78,7% |
| 0,55 | 292 | 59 | 91,4% | 85,9% | 88,6% | 72,9% | 78,9% |
| 0,60 | 278 | 52 | 91,6% | 83,5% | 87,3% | 74,5% | 78,7% |
| 0,65 | 275 | 49 | 91,6% | 83,5% | 87,3% | 75,3% | 79,2% |
| 0,70 | 267 | 45 | 91,9% | 82,3% | 86,8% | 76,4% | 79,2% |
| 0,75 | 251 | 37 | 94,4% | 81,5% | 87,4% | 80,5% | 81,0% |
| 0,80 | 240 | 33 | 95,7% | 79,8% | 87,0% | 82,5% | 81,1% |
| 0,85 | 213 | 25 | 97,3% | 73,8% | 83,9% | 85,9% | 79,4% |
| 0,90 | 168 | 13 | 98,1% | 61,3% | 75,4% | 90,5% | 73,1% |
| 0,95 | 7 | 0 | 100,0% | 2,8% | 5,5% | 100,0% | 5,5% |

Arquivos: `curva_limiar_val.csv` e `contagens_val.csv` (por foto, no limiar escolhido).

### Limiar

- **0,35, escolhido (maior F1 da métrica principal):** 321 folhas previstas para 248 anotadas; 75 previsões fora da região (em 38 fotos), que não contam na métrica principal. Região: precisão 89,0%, recall 88,3%, F1 88,7%. Conservador: precisão 68,2%, F1 77,0%.

### Detecções da periferia (olhadas nas fotos, sem imagens no repositório)

No limiar escolhido (0,35), 75 previsões ficam fora da região anotada (mais da metade da máscara
fora), em 38 das 46 fotos de validação. Nenhuma se sobrepõe a uma folha anotada (IoU acima de 0,1:
nenhuma), 73 tocam a borda da foto e a área mediana é 2,2% da foto (p10 0,8%, p90 4,7%; a folha
anotada mediana do treino tem 5,0%). O score mediano é 0,75, e 33 têm 0,8 ou mais.

Em 10/10/2026 olhei 9 fotos com as detecções da periferia desenhadas, cobrindo 28 das 75
previsões: 20190831_163138, 20190831_164144, 20190831_164159, 20190831_165255, 20191208_141947,
20191208_142146, 20191208_142532, 20191208_142536 e 20191208_143600. As sobreposições ficaram só
no rascunho da sessão, fora do repositório, e foram apagadas no fim do passo.

- **Folhas de verdade: 26 das 28.** São folhas de café inteiras ou, quase sempre, cortadas pelo
  enquadramento, em foco e com o contorno seguindo a borda da folha. Os autores não as
  contornaram: o BRACOT anota cerca de 1/3 das folhas de cada foto, as do centro.
- **Erros: 2 das 28.** Em 20191208_142146 (score 0,58), a máscara cobre um objeto laranja
  desfocado no canto da foto (parece um dedo na frente da lente) junto com parte de uma folha. Em
  20191208_142532 (score 0,61), é um fragmento de folha com um pedaço de galho.
- **Polígono:** as sobreposições mostraram traços retos ligando pedaços da mesma máscara (o
  `masks.xy` do ultralytics emenda todos os pedaços num polígono só). O `detectar_folhas` passou
  a devolver o contorno do maior pedaço (ver `model/detector.py`).

**Conclusão:** a periferia não é lixo. A métrica principal, que não conta essas previsões, é a
mais justa, e a conservadora subestima a precisão. Por isso não há um segundo limiar mais alto:
ele também não separaria a periferia, que tem scores tão altos quanto os das folhas anotadas.
As folhas cortadas pela borda da foto são uma questão de produto. Se o backend não deve
classificar folha incompleta, o caminho é um filtro de borda ou de área, não o limiar; fica para
decisão do gestor.

## RNF02 (diagnóstico inteiro em até 5 s, em CPU)

Medido nesta máquina, em CPU (8 threads do torch), nas 46 fotos de validação, depois de uma passada de aquecimento: leitura da foto, detecção (redução, moldura e YOLO em PyTorch, 640 px, no limiar escolhido) e classificação das folhas detectadas (o classificador `base_resnet50_bracol`, recorte pela caixa).

| etapa | mediana (s) | máximo (s) |
|---|---:|---:|
| leitura | 0,06 | 0,07 |
| detecção | 0,15 | 0,18 |
| classificação | 0,52 | 1,15 |
| **total** | 0,73 | 1,35 |

Mediana de 6 folhas por foto; 0 fotos passaram de 5 s. A CPU do backend pode ser mais lenta que esta.

## Licença

O ultralytics e os pesos YOLO são AGPL-3.0 (decisão do gestor, 10/10/2026): aceita para este projeto acadêmico de repositórios públicos. Se o backend servir o modelo pela rede, o código-fonte do serviço tem de ficar disponível a quem o usa.

## Acessos à rede neste passo

- metadados: https://pypi.org/pypi/ultralytics/json
- metadados: https://pypi.org/pypi/polars/json
- metadados: https://pypi.org/pypi/polars-runtime-32/json
- metadados: https://pypi.org/pypi/nvidia-ml-py/json
- metadados: https://pypi.org/pypi/ultralytics-thop/json
- metadados: https://pypi.org/pypi/ultralytics-platform/json
- metadados: https://api.github.com/repos/ultralytics/assets/releases/tags/v8.3.0
- cabeçalho: https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
- cabeçalho: https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth
- pip: índice do PyPI para opencv-python-headless 5.0.0.93 (o wheel veio do cache local)
- pip: wheels do PyPI: ultralytics 8.4.175, polars 2.0.0, polars-runtime-32 2.0.0, nvidia-ml-py 13.615.71, ultralytics-thop 2.2.2, ultralytics-platform 0.1.83
- peso: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
- download não autorizado: https://ultralytics.com/assets/Arial.ttf (fonte de 755 KB, baixada pela checagem do dataset do ultralytics no treino de 3 épocas que mediu o tempo; o arquivo foi apagado e a pré-carga, desligada em detector.importar_ultralytics)

## Limites

- A anotação é parcial: mesmo na região, folhas não contornadas viram erro.
- O casco convexo é uma aproximação da área que os autores olharam.
- A validação tem 46 fotos das duas sessões de fotos do BRACOT (um dia cada). O teste dos autores vem das mesmas sessões, e 27 cenas misturam treino e teste (55 das 60 fotos de teste; `data/README.md`): o teste deve sair otimista para fotos de outras lavouras.
- O limiar e a moldura da entrada foram escolhidos na própria validação, o que a deixa um pouco otimista; o teste, fechado, mede sem essa escolha.
- Uma semente só; a GPU não é determinística, e o ultralytics avisa que o cache em RAM também pode mudar o resultado entre execuções.

