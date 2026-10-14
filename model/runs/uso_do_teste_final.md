# Uso único dos testes fechados (passo 7)

**Autorização:** "pode" do gestor em 14/10/2026 (11:39). **Política de descarte fechada:** t_d = 0,65 mantido (decisão do gestor, 14/10/2026). **Esta é a única abertura dos testes.** Nada é retreinado, nenhum limiar ou padrão muda, e os testes não servem a nenhuma escolha. Este log substitui, neste passo, os registros `uso_do_teste.md` e `uso_do_teste_detector.md`, que não são criados.

## Abertura, antes de ler qualquer teste

- **Data e hora:** 14/10/2026 12:11:34.
- **Commit:** `3a863b326826fce8e0dcbea6444fd26932b8287b`. Nenhuma alteração não commitada em `model/` (fora de `model/runs/`), `evaluation/` ou `tests/`.
- **git status --porcelain:** vazio.

**Pesos finais**, conferidos com `model/pesos/MANIFESTO.md`:

| peso | arquivo | bytes | SHA-256 |
|---|---|---:|---|
| classificador | `model/runs/base_resnet50_bracol/melhor.pt` | 94.438.195 | `0d7c043ab2907e186b5c4f99fa09a6a62120687f862c91404b5bafee9b66066a` |
| detector | `model/runs/detector_base/weights/last.pt` | 20.504.100 | `0532766bf0d2c011a79ead110b217a2a79a8f252cc17a69c6383260de6c4dc21` |

**Configuração efetiva**, lida do código (`asdict(inferencia.Configuracao())`) e conferida com a declarada pelo gestor:

| parâmetro | valor |
|---|---|
| limiar_deteccao | `0.35` |
| max_folhas | `15` |
| area_minima | `0.005` |
| saturacao_minima | `30` |
| fracao_colorida_minima | `0.25` |
| limiar_confianca | `0.61` |
| recorte | `'B'` |
| margem | `0.17` |
| cinza_fundo | `197` |
| cobertura_folha_unica | `0.3` |
| score_minimo_folha_unica | `0.05` |
| fracao_visivel_minima | `0.5` |
| preenchimento_minimo | `0.0` |
| max_megapixels | `64.0` |
| decisao | `'por_caminho'` |
| limiar_saudavel | `0.083` |
| limiar_doenca | `0.65` |

- Pesos que a cadeia carrega: `model/runs/base_resnet50_bracol/melhor.pt` e `model/runs/detector_base/weights/last.pt`.
- Limiar de detecção escolhido na validação do detector: 0.35.
- Semente do sorteio das folhas do uso 3: 42.

## O que será medido (só isto)

1. **Uso 1, teste do BRACOL** (as 253 imagens de `dados.montar_teste()`), em `model/runs/teste_bracol/`:
   - **Cadeia final inteira:** `avaliar_pipeline.t1` só no modo B, como no T1 do `pipeline_final`.
     - Por caminho: a foto inteira (argmax com corte 0,61), a detecção (duas etapas) e a regra 3.
     - Métricas: acurácia com Wilson, F1 macro, recall por classe, matriz de confusão, severidade (exata e ±1), cobertura e acurácia das mantidas e doentes chamadas saudáveis.
   - **O que o usuário recebe:** `inferencia.diagnosticar` em cada imagem (sucesso, `baixa_confianca`, `planta_nao_identificada` e exceções), conferido com o T1 nas imagens da foto inteira.
   - **Classificador sozinho, sem corte (só informação):** pelo caminho do `metricas_val.json` (`train.avaliar`, com AMP), com as mesmas métricas, MAE e kappa da severidade.
   - **Comparação, sem ajuste:** com a validação e com os protocolos por blocos e aleatório do `diagnostico_blocos`, lidos dos arquivos.
2. **Uso 2, detecção no teste do BRACOT** (as 60 fotos), em `model/runs/teste_bracot_deteccao/`:
   - `treinar_detector.avaliar_teste("detector_base")`, com o limiar da validação (0,35).
   - Métricas: AP50 e AP50-95 de máscara e de caixa, nos modos região e conservador, e precisão, recall e F1 no limiar.
   - Com as 60 fotos e sem as 2 de `TESTE_SOBREPOSTO` (20190831_164920, 20190831_165013), mais a contagem de folhas por foto, anotadas contra previstas.
   - Sem o validador do ultralytics, que exigiria copiar as fotos de teste.
3. **Uso 3A, folhas de campo do teste do BRACOT, parte A** (as 58 fotos, sem as de `TESTE_SOBREPOSTO`), em `model/runs/teste_bracot_folhas/parte_a/`:
   - Detector, área, cor, limite e filtro de fragmentos (0,5); as anotações de teste não são lidas.
   - Sorteio de até 2 folhas por foto, com a semente 42 (`campo.sortear_rotulagem`), sem o classificador carregado.
   - Pranchas cegas em `data/processed/campo_rotulagem_teste/` (fora do git) e o CSV vazio `data/manifests/campo_rotulado_teste.csv`. Só contagens: nenhuma predição.
4. **Uso 3B, folhas de campo do teste do BRACOT, parte B** (só com o CSV rotulado e o "pode avaliar" do gestor), em `model/runs/teste_bracot_folhas/parte_b/`. A estrutura é a do passo 6b, e a cadeia congelada é o resultado principal:
   - destino de S, D e "?": saudável, doente mantida (com as classes), descartada por fragmento, por t_d ou por outro motivo, com Wilson;
   - D evidentes que chegam ao usuário como doente, como saudável ou somem;
   - classes e confiança das doentes mantidas;
   - saudável dado S e dado D, com Wilson;
   - AUC de P(saudável) com bootstrap por foto (2.000 reamostras, semente 42);
   - os "?" como D e como S.
   - Os números com o corte t_d desligado entram só como informação, declarada aqui.
   - Comparação, sem ajuste, com as antigas e as novas do passo 6b.

## Regras de execução

- **Saídas:** cada uso roda uma vez e grava numa pasta nova, sem sobrescrever.
- **Erro de código:** a pasta temporária é apagada e o erro vai para este log. Uma nova tentativa só roda com o motivo registrado.
- **Conferência interna que falha sem exceção:** fica registrada e relatada como saiu. Outra rodada só com decisão do gestor, registrada aqui.

## Registro

| data e hora | uso | evento | detalhes |
|---|---|---|---|
| 14/10/2026 12:11:45 | 1 | início | teste do BRACOL; commit 3a863b3; código não commitado: nenhum; pasta model/runs/teste_bracol |
| 14/10/2026 12:11:45 | 1 | teste lido | 253 imagens do teste do BRACOL (dados.montar_teste) |
| 14/10/2026 12:12:58 | 1 | fim | cadeia final: acurácia sem corte 88,1%, mantidas 225 de 253, acurácia das mantidas 91,1%; classificador sozinho: acurácia 90,5%, F1 macro 88,5% |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/contrato.csv, SHA-256 52feb2e848e579304c4148700777588354d1f9d25b014c3e3e4cf96c3bff4a63 |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/matriz_confusao_cadeia_teste.png, SHA-256 87ccc4f9802bdf46faa99bc8a408aa8557e8bd23552c7d5429fa807575d8cb84 |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/matriz_confusao_teste.png, SHA-256 9a85e4d6e1f3b0927ff9a810a87ba843f588a771de7f730759c12db5f8de358e |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/metricas_teste.json, SHA-256 e6c3a51f62db1359658a2c4e980850924f7882eba4259fe67ba5e7b9ce3c918f |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/predicoes_teste.csv, SHA-256 57cfdfbc044b691685de3233b0831c9647fb144ffcad124dae07d004b1298f21 |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/relatorio.md, SHA-256 89eb1114a638296311a849761edb7dc1c52704f14d396db82df292ab178bf954 |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/resultado.json, SHA-256 5e8bb60d1526c460d5a15f9bd0f5bc557a899d6982b529aefdef22e1f4f5be21 |
| 14/10/2026 12:12:58 | 1 | arquivo | model/runs/teste_bracol/t1_imagens.csv, SHA-256 e6157d185c983cd2e45f1e05b0f2e0893ccdebf10863fc754352f4ee97247675 |
| 14/10/2026 12:13:13 | 2 | início | detecção no teste do BRACOT; commit 3a863b3; código não commitado: nenhum; pasta model/runs/teste_bracot_deteccao |
| 14/10/2026 12:13:29 | 2 | teste lido | fotos e anotações de teste do BRACOT lidas e avaliadas (detector_base) |
| 14/10/2026 12:13:29 | 2 | fim | AP50 de máscara (região), 60 fotos: 92,0%; sem as sobrepostas: 91,6% |
| 14/10/2026 12:13:29 | 2 | arquivo | model/runs/teste_bracot_deteccao/contagens_teste.csv, SHA-256 f7415cae530430cc4fe222c9840356da679fc23a5353c79a5901dbef4a03f466 |
| 14/10/2026 12:13:29 | 2 | arquivo | model/runs/teste_bracot_deteccao/metricas_teste.json, SHA-256 1f704d588325a55d2bad7855fa1ddb4450caf577df500d477caa2bb70673cfc5 |
| 14/10/2026 12:13:29 | 2 | arquivo | model/runs/teste_bracot_deteccao/relatorio.md, SHA-256 fcdbe543a2662f94193c4da4c2f6a6e196f93a010ec956070f46594863320be9 |
| 14/10/2026 12:13:29 | 2 | arquivo | model/runs/teste_bracot_deteccao/resultado.json, SHA-256 2ed42124f1981a4b32f15c39d9cdb68bf643f1379dd7dc8b7e901571c9de12f9 |
| 14/10/2026 12:13:40 | 3a | início | folhas de campo do teste do BRACOT, parte A; commit 3a863b3; código não commitado: nenhum; pasta model/runs/teste_bracot_folhas/parte_a |
| 14/10/2026 12:13:40 | 3a | teste lido | 58 fotos de teste do BRACOT (só as imagens) |
| 14/10/2026 12:13:57 | 3a | fim | 115 folhas sorteadas de 58 fotos em 6 pranchas; o filtro de fragmentos tirou 76 folhas |
| 14/10/2026 12:13:57 | 3a | arquivo | model/runs/teste_bracot_folhas/parte_a/contagens.csv, SHA-256 bc8f7d516a7b14b7646e06bd6726c41248ca35e84c502d1c6353f6549004cb1d |
| 14/10/2026 12:13:57 | 3a | arquivo | model/runs/teste_bracot_folhas/parte_a/relatorio.md, SHA-256 d649e698da0a7b5dae409cd14e1c76b38311e09c332b53500ad1ce0edfb37f72 |
| 14/10/2026 12:13:57 | 3a | arquivo | model/runs/teste_bracot_folhas/parte_a/resultado.json, SHA-256 52ca941a3fc727965e9f0c1b25fb1e9d8a54a586aba0d8643f4091774884cd8c |
| 14/10/2026 12:13:57 | 3a | arquivo | data/processed/campo_rotulagem_teste/prancha_1.jpg, SHA-256 72e5fc0e4662e54f97eaf447cd5db96dd88fc12cead1bb98be19f08d05aa7113 |
| 14/10/2026 12:13:57 | 3a | arquivo | data/processed/campo_rotulagem_teste/prancha_2.jpg, SHA-256 4ac10ee93baf25a658a5420e3ba5ddfd57a5ee88596055e75116eb757d2b4dfe |
| 14/10/2026 12:13:57 | 3a | arquivo | data/processed/campo_rotulagem_teste/prancha_3.jpg, SHA-256 11ab1cf4d036298a25aad285f3d51de6f0f6eea35e2dc093d27c7d73f4c751d6 |
| 14/10/2026 12:13:57 | 3a | arquivo | data/processed/campo_rotulagem_teste/prancha_4.jpg, SHA-256 c6c77844d5b7494c2d9d650b040020177cf126c31ded45d35f8c8589c887a20a |
| 14/10/2026 12:13:57 | 3a | arquivo | data/processed/campo_rotulagem_teste/prancha_5.jpg, SHA-256 1286dcf0a9473c46fcee6892ac079b4d8e39940e711350d320fdfd0046b74605 |
| 14/10/2026 12:13:57 | 3a | arquivo | data/processed/campo_rotulagem_teste/prancha_6.jpg, SHA-256 ae99277233c54ce195e5934a752e226841c6f9727d8a854d120db4e6c72dab2d |
| 14/10/2026 12:13:57 | 3a | arquivo | data/manifests/campo_rotulado_teste.csv, SHA-256 a65aa4654710edac44f229007532caaf56db5aea148d58826aeaa0f25298202d |
| 14/10/2026 12:15:12 | 1 | relatório refeito | model/runs/teste_bracol/relatorio.md, SHA-256 5cc4f326a7d84c77693953fd89958c60a9702caa3d4367af1b9ff8a620a32909 |
| 14/10/2026 12:15:17 | 2 | relatório refeito | model/runs/teste_bracot_deteccao/relatorio.md, SHA-256 56a366ba09dc355f95785e91b9addf7ec0881369b4cb5f7515e89d6c3a9c6d7b |
