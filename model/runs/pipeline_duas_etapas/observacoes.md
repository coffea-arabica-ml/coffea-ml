**Regra em duas etapas (limiar de saudável 0,083; limiar de doença proposto 0,65) contra o
argmax com 0,61 do `pipeline_padrao_b`.** Mesmos modelos, mesma cadeia (recorte B e filtro de
fragmentos) e mesmas fotos. Nenhum recorte foi olhado neste run, e os rótulos de campo novos
ainda não foram usados.

- **BRACOL, fora da dobra (1.432):** a regra completa fica abaixo do argmax.
  - Acurácia das mantidas: 91,7% (90,1% a 93,1%) contra 95,1% (93,7% a 96,1%), com
    cobertura parecida (89,4% e 88,1%). Os intervalos não se tocam.
  - A causa é a 1ª etapa: P(saudável) >= 0,083 marca 61 das 1.201 doentes como saudáveis
    (5,1%), contra 13 (1,1%) do argmax, e essas folhas não passam por nenhum corte.
  - Com o limiar de saudável fixo, nenhum limiar de doença avaliado (de 0,30 a 0,90) leva a
    regra completa a 95% fora da dobra. O máximo entre eles é 93,4%, com limiar de doença 0,80.
  - Esses modelos de 20 épocas são mais fracos que o run base.
- **BRACOL, validação no modelo congelado:**
  - **Imagem inteira:** 4 de 211 doentes viram saudáveis. No corte, 95,3% contra 96,1% do
    argmax; os intervalos se sobrepõem.
  - **Cadeia B (T1):** sem corte, 92,4% contra 94,0%. Viram saudáveis 6 de 210 doentes, contra
    0 do argmax. No corte, a cobertura fica igual (91,7%) e a acurácia vai de 95,2% para 93,9%.
- **Custo por classe do limiar de doença de 0,65 (fora da dobra):**
  - a cercosporiose mantém só 67,2% das folhas;
  - o bicho-mineiro, 82,4%;
  - a ferrugem, 85,4%;
  - a phoma, 95,6%.
- **Campo (T2, sem rótulo de classe; modo B):**
  - **Saudáveis:** 122 das 239 folhas, contra 24 do argmax.
  - **Doenças:** das 117 folhas que vão para a 2ª etapa, só 30 passam do limiar de doença (24
    ferrugem, 4 bicho-mineiro e 2 phoma); 87 saem.
  - **Fotos em `baixa_confianca`:** 1 de 46, contra 25 com o argmax.
  - Se as 122 saudáveis estão certas, só os rótulos novos dizem.
- **T3:** mediana de 0,89 s e pior caso estimado de 2,39 s, abaixo dos 5 s do RNF02.
