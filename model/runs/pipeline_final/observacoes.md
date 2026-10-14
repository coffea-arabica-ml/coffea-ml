**Cadeia final (regra por caminho) contra os runs anteriores.** Os modelos, a cadeia (recorte B
e filtro de fragmentos) e as fotos são os mesmos; só a regra de decisão muda. Nenhum recorte foi
olhado neste run, e nenhum limiar foi ajustado.

- **Conferência:** aplicados às probabilidades deste run, o argmax e as duas etapas dão os mesmos
  números gravados no `pipeline_padrao_b` e no `pipeline_duas_etapas`, no T1 e no T2.
- **BRACOL, foto inteira (203 das 252 imagens):** a cadeia final é igual ao argmax.
  - Das mantidas, 95,3% (91,2% a 97,5%) estão certas, e nenhuma das 166 doentes é chamada
    saudável.
  - Nesse caminho, as duas etapas teriam chamado saudáveis 3 das 166 doentes.
- **BRACOL, recortes por detecção (48 imagens):** a cadeia final é igual às duas etapas.
  - Das mantidas, 90,2% (77,5% a 96,1%) estão certas, contra 95,1% (83,9% a 98,7%) do argmax.
    A cobertura é a mesma, 41 de 48.
  - Viram saudáveis 3 das 44 doentes (6,8%, de 2,3% a 18,2%), contra 0 no argmax.
  - Os intervalos se sobrepõem: a diferença não é sustentada.
- **BRACOL, todas as imagens:**
  - Das mantidas, 94,4% (90,6% a 96,7%) estão certas: entre o argmax (95,2%) e as duas etapas em
    todo caminho (93,9%). A cobertura é a mesma nas três regras, 231 de 252.
  - Viram saudáveis 3 das 211 doentes (1,4%), contra 0 no argmax e 6 nas duas etapas.
  - Nenhuma dessas diferenças é sustentada pelos intervalos.
- **Campo (T2, sem rótulo, modo B):** as 46 fotos vão pela detecção, então a cadeia final é
  igual às duas etapas.
  - Das 239 folhas, 122 saem saudáveis, 30 doentes (24 ferrugem, 4 bicho-mineiro e 2 phoma) e 87
    são descartadas.
  - Uma foto fica em `baixa_confianca`, contra 25 no argmax. 20 fotos têm alguma doença na
    resposta, contra 21.
  - O T2 não diz se as 122 saudáveis estão certas. Isso só os rótulos de campo medem: nas 60 D
    novas, a regra chama saudável 6,7% (2,6% a 15,9%) (limitações, abaixo).
- **T3:** mediana de 0,80 s, máximo de 1,40 s e pior caso estimado de 2,15 s, abaixo dos 5 s do
  RNF02. A diferença para os tempos dos outros runs vem da máquina, não da regra.
