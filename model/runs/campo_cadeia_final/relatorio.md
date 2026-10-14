# Campo pela cadeia final (passo 6b)

Gerado por `python evaluation/campo_cadeia_final.py` em 14/10/2026 11:32, no commit `975b484` (com alterações não commitadas). Classificador `base_resnet50_bracol`, detector `detector_base`, dispositivo cuda. **Nenhum limiar nem padrão mudou; nada do teste do BRACOL nem do BRACOT foi lido.**

A cadeia final inteira (`inferencia.Configuracao()`) roda em cada foto rotulada: o detector, os filtros (fragmentos em 0,5), as regras de folha única e a regra do caminho. Nos recortes por detecção (modo B), as duas etapas: saudável com P(saudável) >= 0,083; senão, a doença com P renormalizada, cortada em t_d = 0,65. Na foto inteira, o argmax com corte 0,61. A `confianca` das doentes é a P renormalizada entre as 4 doenças (a do contrato).

**As folhas** são só as rotuladas que já existem: as 120 antigas (rotuladas com o filtro de fragmentos desligado, então a cadeia final pode tirar algumas) e as 160 novas (rotuladas já com o filtro ligado). Cada rótulo casa com a sua folha pela caixa (IoU >= 0,5), e a folha é procurada na cadeia final, na mesma detecção. Rótulos sem par: antigas 0, novas 0.

**Planos das fotos na cadeia final:** antigas: deteccao 46; novas: deteccao 160.

**Conferência.** Nas 245 folhas classificadas pela detecção, contra `campo_validacao_duas_etapas` (mesmos recortes B, t_s e t_d): diferença máxima das probabilidades 6.7e-04; desfechos diferentes: 0; folhas sem par: 0.

## a) S, D e "?": o que a cadeia final faz com cada folha

Proporções sobre todas as folhas do rótulo, com Wilson de 95%. Descartada: a folha não chega ao usuário (se nenhuma folha da foto chega, a foto vira `baixa_confianca`).

| conjunto | rótulo | folhas | chamada saudável | chamada doente e mantida | classes das doentes mantidas | descartada | descartada: filtro de fragmentos | descartada: corte t_d | descartada: outro motivo |
|---|---|---:|---|---|---|---|---|---|---|
| antigas (fotos de validação do detector) | S | 36 | 61,1% (44,9% a 75,2%; 22 de 36) | 2,8% (0,5% a 14,2%; 1 de 36) | phoma 1 | 36,1% (22,5% a 52,4%; 13 de 36) | 22,2% (11,7% a 38,1%; 8 de 36) | 13,9% (6,1% a 28,7%; 5 de 36) | 0,0% (0,0% a 9,6%; 0 de 36) |
| antigas (fotos de validação do detector) | D | 46 | 2,2% (0,4% a 11,3%; 1 de 46) | 19,6% (10,7% a 33,2%; 9 de 46) | ferrugem 8, bicho_mineiro 1 | 78,3% (64,4% a 87,7%; 36 de 46) | 19,6% (10,7% a 33,2%; 9 de 46) | 58,7% (44,3% a 71,7%; 27 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) |
| antigas (fotos de validação do detector) | ? | 38 | 34,2% (21,2% a 50,1%; 13 de 38) | 2,6% (0,5% a 13,5%; 1 de 38) | ferrugem 1 | 63,2% (47,3% a 76,6%; 24 de 38) | 47,4% (32,5% a 62,7%; 18 de 38) | 15,8% (7,4% a 30,4%; 6 de 38) | 0,0% (0,0% a 9,2%; 0 de 38) |
| novas (fotos de treino do detector) | S | 41 | 80,5% (66,0% a 89,8%; 33 de 41) | 4,9% (1,3% a 16,1%; 2 de 41) | ferrugem 1, phoma 1 | 14,6% (6,9% a 28,4%; 6 de 41) | 0,0% (0,0% a 8,6%; 0 de 41) | 14,6% (6,9% a 28,4%; 6 de 41) | 0,0% (0,0% a 8,6%; 0 de 41) |
| novas (fotos de treino do detector) | D | 60 | 6,7% (2,6% a 15,9%; 4 de 60) | 36,7% (25,6% a 49,3%; 22 de 60) | ferrugem 18, bicho_mineiro 2, phoma 1, cercosporiose 1 | 56,7% (44,1% a 68,4%; 34 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) | 56,7% (44,1% a 68,4%; 34 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) |
| novas (fotos de treino do detector) | ? | 59 | 54,2% (41,7% a 66,3%; 32 de 59) | 11,9% (5,9% a 22,5%; 7 de 59) | ferrugem 6, phoma 1 | 33,9% (23,1% a 46,6%; 20 de 59) | 0,0% (0,0% a 6,1%; 0 de 59) | 33,9% (23,1% a 46,6%; 20 de 59) | 0,0% (0,0% a 6,1%; 0 de 59) |
| juntas | S | 77 | 71,4% (60,5% a 80,3%; 55 de 77) | 3,9% (1,3% a 10,8%; 3 de 77) | ferrugem 1, phoma 2 | 24,7% (16,4% a 35,4%; 19 de 77) | 10,4% (5,4% a 19,2%; 8 de 77) | 14,3% (8,2% a 23,8%; 11 de 77) | 0,0% (0,0% a 4,8%; 0 de 77) |
| juntas | D | 106 | 4,7% (2,0% a 10,6%; 5 de 106) | 29,2% (21,4% a 38,5%; 31 de 106) | ferrugem 26, bicho_mineiro 3, phoma 1, cercosporiose 1 | 66,0% (56,6% a 74,4%; 70 de 106) | 8,5% (4,5% a 15,4%; 9 de 106) | 57,5% (48,0% a 66,5%; 61 de 106) | 0,0% (0,0% a 3,5%; 0 de 106) |
| juntas | ? | 97 | 46,4% (36,8% a 56,3%; 45 de 97) | 8,2% (4,2% a 15,4%; 8 de 97) | ferrugem 7, phoma 1 | 45,4% (35,8% a 55,3%; 44 de 97) | 18,6% (12,1% a 27,4%; 18 de 97) | 26,8% (19,0% a 36,4%; 26 de 97) | 0,0% (0,0% a 3,8%; 0 de 97) |

"Outro motivo": nenhuma folha.

## b) As D evidentes: o que chega ao usuário

Com o corte t_d desligado no caminho de detecção (limiar de doença em 0) só para informação: nenhum padrão mudou. O filtro de fragmentos e a 1ª etapa não mudam.

| conjunto | cenário | D | chegam como doente | chegam como saudável | somem | somem: filtro de fragmentos | somem: corte t_d | somem: outro motivo |
|---|---|---:|---|---|---|---|---|---|
| antigas (fotos de validação do detector) | cadeia final (t_d = 0,65) | 46 | 19,6% (10,7% a 33,2%; 9 de 46) | 2,2% (0,4% a 11,3%; 1 de 46) | 78,3% (64,4% a 87,7%; 36 de 46) | 19,6% (10,7% a 33,2%; 9 de 46) | 58,7% (44,3% a 71,7%; 27 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) |
| antigas (fotos de validação do detector) | t_d desligado na detecção (só informação) | 46 | 78,3% (64,4% a 87,7%; 36 de 46) | 2,2% (0,4% a 11,3%; 1 de 46) | 19,6% (10,7% a 33,2%; 9 de 46) | 19,6% (10,7% a 33,2%; 9 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) |
| novas (fotos de treino do detector) | cadeia final (t_d = 0,65) | 60 | 36,7% (25,6% a 49,3%; 22 de 60) | 6,7% (2,6% a 15,9%; 4 de 60) | 56,7% (44,1% a 68,4%; 34 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) | 56,7% (44,1% a 68,4%; 34 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) |
| novas (fotos de treino do detector) | t_d desligado na detecção (só informação) | 60 | 93,3% (84,1% a 97,4%; 56 de 60) | 6,7% (2,6% a 15,9%; 4 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) | 0,0% (0,0% a 6,0%; 0 de 60) |
| juntas | cadeia final (t_d = 0,65) | 106 | 29,2% (21,4% a 38,5%; 31 de 106) | 4,7% (2,0% a 10,6%; 5 de 106) | 66,0% (56,6% a 74,4%; 70 de 106) | 8,5% (4,5% a 15,4%; 9 de 106) | 57,5% (48,0% a 66,5%; 61 de 106) | 0,0% (0,0% a 3,5%; 0 de 106) |
| juntas | t_d desligado na detecção (só informação) | 106 | 86,8% (79,0% a 92,0%; 92 de 106) | 4,7% (2,0% a 10,6%; 5 de 106) | 8,5% (4,5% a 15,4%; 9 de 106) | 8,5% (4,5% a 15,4%; 9 de 106) | 0,0% (0,0% a 3,5%; 0 de 106) | 0,0% (0,0% a 3,5%; 0 de 106) |

## c) Folhas doentes mantidas: classes e confiança

As classes são as previstas: **nenhuma D tem classe nos rótulos, então a classe não é verificada.** Confiança: mediana (p25 a p75; mínimo e máximo).

**Cadeia final (t_d = 0,65).**

| conjunto | rótulo | doentes mantidas | ferrugem | bicho_mineiro | phoma | cercosporiose | confiança |
|---|---|---:|---|---|---|---|---|
| antigas (fotos de validação do detector) | S | 1 | 0,0% (0,0% a 79,3%; 0 de 1) | 0,0% (0,0% a 79,3%; 0 de 1) | 100,0% (20,7% a 100,0%; 1 de 1) | 0,0% (0,0% a 79,3%; 0 de 1) | 0,82 (0,82 a 0,82; mín. 0,82, máx. 0,82) |
| antigas (fotos de validação do detector) | D | 9 | 88,9% (56,5% a 98,0%; 8 de 9) | 11,1% (2,0% a 43,5%; 1 de 9) | 0,0% (0,0% a 29,9%; 0 de 9) | 0,0% (0,0% a 29,9%; 0 de 9) | 0,81 (0,71 a 0,82; mín. 0,66, máx. 0,93) |
| antigas (fotos de validação do detector) | ? | 1 | 100,0% (20,7% a 100,0%; 1 de 1) | 0,0% (0,0% a 79,3%; 0 de 1) | 0,0% (0,0% a 79,3%; 0 de 1) | 0,0% (0,0% a 79,3%; 0 de 1) | 0,66 (0,66 a 0,66; mín. 0,66, máx. 0,66) |
| antigas (fotos de validação do detector) | todas | 11 | 81,8% (52,3% a 94,9%; 9 de 11) | 9,1% (1,6% a 37,7%; 1 de 11) | 9,1% (1,6% a 37,7%; 1 de 11) | 0,0% (0,0% a 25,9%; 0 de 11) | 0,81 (0,69 a 0,82; mín. 0,66, máx. 0,93) |
| novas (fotos de treino do detector) | S | 2 | 50,0% (9,5% a 90,5%; 1 de 2) | 0,0% (0,0% a 65,8%; 0 de 2) | 50,0% (9,5% a 90,5%; 1 de 2) | 0,0% (0,0% a 65,8%; 0 de 2) | 0,75 (0,70 a 0,80; mín. 0,65, máx. 0,85) |
| novas (fotos de treino do detector) | D | 22 | 81,8% (61,5% a 92,7%; 18 de 22) | 9,1% (2,5% a 27,8%; 2 de 22) | 4,5% (0,8% a 21,8%; 1 de 22) | 4,5% (0,8% a 21,8%; 1 de 22) | 0,82 (0,72 a 0,92; mín. 0,65, máx. 0,97) |
| novas (fotos de treino do detector) | ? | 7 | 85,7% (48,7% a 97,4%; 6 de 7) | 0,0% (0,0% a 35,4%; 0 de 7) | 14,3% (2,6% a 51,3%; 1 de 7) | 0,0% (0,0% a 35,4%; 0 de 7) | 0,81 (0,78 a 0,87; mín. 0,71, máx. 0,88) |
| novas (fotos de treino do detector) | todas | 31 | 80,6% (63,7% a 90,8%; 25 de 31) | 6,5% (1,8% a 20,7%; 2 de 31) | 9,7% (3,3% a 24,9%; 3 de 31) | 3,2% (0,6% a 16,2%; 1 de 31) | 0,82 (0,73 a 0,88; mín. 0,65, máx. 0,97) |
| juntas | S | 3 | 33,3% (6,1% a 79,2%; 1 de 3) | 0,0% (0,0% a 56,1%; 0 de 3) | 66,7% (20,8% a 93,9%; 2 de 3) | 0,0% (0,0% a 56,1%; 0 de 3) | 0,82 (0,73 a 0,84; mín. 0,65, máx. 0,85) |
| juntas | D | 31 | 83,9% (67,4% a 92,9%; 26 de 31) | 9,7% (3,3% a 24,9%; 3 de 31) | 3,2% (0,6% a 16,2%; 1 de 31) | 3,2% (0,6% a 16,2%; 1 de 31) | 0,81 (0,71 a 0,88; mín. 0,65, máx. 0,97) |
| juntas | ? | 8 | 87,5% (52,9% a 97,8%; 7 de 8) | 0,0% (0,0% a 32,4%; 0 de 8) | 12,5% (2,2% a 47,1%; 1 de 8) | 0,0% (0,0% a 32,4%; 0 de 8) | 0,80 (0,76 a 0,87; mín. 0,66, máx. 0,88) |
| juntas | todas | 42 | 81,0% (66,7% a 90,0%; 34 de 42) | 7,1% (2,5% a 19,0%; 3 de 42) | 9,5% (3,8% a 22,1%; 4 de 42) | 2,4% (0,4% a 12,3%; 1 de 42) | 0,81 (0,71 a 0,86; mín. 0,65, máx. 0,97) |

**Com o corte t_d desligado na detecção (só informação).**

| conjunto | rótulo | doentes mantidas | ferrugem | bicho_mineiro | phoma | cercosporiose | confiança |
|---|---|---:|---|---|---|---|---|
| antigas (fotos de validação do detector) | S | 6 | 16,7% (3,0% a 56,4%; 1 de 6) | 0,0% (0,0% a 39,0%; 0 de 6) | 83,3% (43,6% a 97,0%; 5 de 6) | 0,0% (0,0% a 39,0%; 0 de 6) | 0,42 (0,36 a 0,45; mín. 0,31, máx. 0,82) |
| antigas (fotos de validação do detector) | D | 36 | 72,2% (56,0% a 84,2%; 26 de 36) | 25,0% (13,8% a 41,1%; 9 de 36) | 0,0% (0,0% a 9,6%; 0 de 36) | 2,8% (0,5% a 14,2%; 1 de 36) | 0,50 (0,37 a 0,63; mín. 0,31, máx. 0,93) |
| antigas (fotos de validação do detector) | ? | 7 | 71,4% (35,9% a 91,8%; 5 de 7) | 14,3% (2,6% a 51,3%; 1 de 7) | 14,3% (2,6% a 51,3%; 1 de 7) | 0,0% (0,0% a 35,4%; 0 de 7) | 0,58 (0,54 a 0,62; mín. 0,43, máx. 0,66) |
| antigas (fotos de validação do detector) | todas | 49 | 65,3% (51,3% a 77,1%; 32 de 49) | 20,4% (11,5% a 33,6%; 10 de 49) | 12,2% (5,7% a 24,2%; 6 de 49) | 2,0% (0,4% a 10,7%; 1 de 49) | 0,50 (0,37 a 0,62; mín. 0,31, máx. 0,93) |
| novas (fotos de treino do detector) | S | 8 | 25,0% (7,1% a 59,1%; 2 de 8) | 12,5% (2,2% a 47,1%; 1 de 8) | 62,5% (30,6% a 86,3%; 5 de 8) | 0,0% (0,0% a 32,4%; 0 de 8) | 0,52 (0,42 a 0,62; mín. 0,35, máx. 0,85) |
| novas (fotos de treino do detector) | D | 56 | 55,4% (42,4% a 67,6%; 31 de 56) | 16,1% (8,7% a 27,8%; 9 de 56) | 23,2% (14,1% a 35,8%; 13 de 56) | 5,4% (1,8% a 14,6%; 3 de 56) | 0,56 (0,44 a 0,76; mín. 0,31, máx. 0,97) |
| novas (fotos de treino do detector) | ? | 27 | 55,6% (37,3% a 72,4%; 15 de 27) | 7,4% (2,1% a 23,4%; 2 de 27) | 33,3% (18,6% a 52,2%; 9 de 27) | 3,7% (0,7% a 18,3%; 1 de 27) | 0,45 (0,37 a 0,66; mín. 0,29, máx. 0,88) |
| novas (fotos de treino do detector) | todas | 91 | 52,7% (42,6% a 62,7%; 48 de 91) | 13,2% (7,7% a 21,6%; 12 de 91) | 29,7% (21,3% a 39,7%; 27 de 91) | 4,4% (1,7% a 10,8%; 4 de 91) | 0,54 (0,40 a 0,73; mín. 0,29, máx. 0,97) |
| juntas | S | 14 | 21,4% (7,6% a 47,6%; 3 de 14) | 7,1% (1,3% a 31,5%; 1 de 14) | 71,4% (45,4% a 88,3%; 10 de 14) | 0,0% (0,0% a 21,5%; 0 de 14) | 0,44 (0,39 a 0,59; mín. 0,31, máx. 0,85) |
| juntas | D | 92 | 62,0% (51,7% a 71,2%; 57 de 92) | 19,6% (12,7% a 28,8%; 18 de 92) | 14,1% (8,4% a 22,7%; 13 de 92) | 4,3% (1,7% a 10,7%; 4 de 92) | 0,55 (0,40 a 0,71; mín. 0,31, máx. 0,97) |
| juntas | ? | 34 | 58,8% (42,2% a 73,6%; 20 de 34) | 8,8% (3,0% a 23,0%; 3 de 34) | 29,4% (16,8% a 46,2%; 10 de 34) | 2,9% (0,5% a 14,9%; 1 de 34) | 0,48 (0,38 a 0,63; mín. 0,29, máx. 0,88) |
| juntas | todas | 140 | 57,1% (48,9% a 65,0%; 80 de 140) | 15,7% (10,6% a 22,6%; 22 de 140) | 23,6% (17,3% a 31,2%; 33 de 140) | 3,6% (1,5% a 8,1%; 5 de 140) | 0,53 (0,40 a 0,69; mín. 0,29, máx. 0,97) |

Confiança por classe prevista (juntas, cadeia final, todas as folhas mantidas como doentes):

| classe | folhas | confiança |
|---|---:|---|
| ferrugem | 34 | 0,82 (0,75 a 0,88; mín. 0,66, máx. 0,97) |
| bicho_mineiro | 3 | 0,78 (0,71 a 0,82; mín. 0,65, máx. 0,86) |
| phoma | 4 | 0,69 (0,66 a 0,74; mín. 0,65, máx. 0,82) |
| cercosporiose | 1 | 0,77 (0,77 a 0,77; mín. 0,77, máx. 0,77) |

## Leitura

**Todas as 206 fotos rotuladas vão pela detecção.** Nenhuma cai na foto inteira, então estes
números são só da regra em duas etapas; o corte de 0,61 não aparece aqui.

**D evidentes (106, juntas), na cadeia final:**
- **Chegam ao usuário como doente:** 29,2% (21,4% a 38,5%; 31 de 106).
- **Chegam como saudável:** 4,7% (2,0% a 10,6%; 5 de 106).
- **Somem:** 66,0% (56,6% a 74,4%; 70 de 106).
  - 61 pelo corte t_d (57,5% das D);
  - 9 pelo filtro de fragmentos, todas nas antigas, que foram rotuladas com o filtro
    desligado;
  - nenhuma por outro motivo.
- **Antigas contra novas:** chegam como doente 19,6% (10,7% a 33,2%) contra 36,7% (25,6% a
  49,3%). Os intervalos se sobrepõem, e parte da diferença vem do filtro, que só tira folhas das
  antigas.
- **Falso saudável:** 4,7% juntas, contra 5,7% na validação (6 de 106). Uma das 2 D antigas
  chamadas saudáveis é um fragmento, e a cadeia final a tira. Nas novas, fica igual: 6,7%.
- **Leitura:** no campo, o que mais pesa nas D evidentes é o corte t_d, não o falso saudável. A
  maioria delas não chega ao usuário.

**Com o t_d desligado na detecção (só informação; nenhum padrão mudou):**
- **D que chegam como doente:** 86,8% (79,0% a 92,0%). Só somem os 9 fragmentos.
- **S chamadas doentes:** sobem de 3,9% (1,3% a 10,8%; 3 de 77) para 18,2% (11,2% a 28,2%; 14 de
  77). Os intervalos não se tocam.
- **"?" chamadas doentes:** sobem de 8,2% para 35,1%.
- **Confiança das doentes mantidas:** a mediana cai de 0,81 para 0,53.
- **Phoma:** sobe de 9,5% para 23,6% das doentes mantidas. Das 14 S chamadas doentes, 10 seriam
  phoma.
- **O t_d corta tanto D quanto S:** sem ele, cerca de 6 em 10 D evidentes a mais chegam como
  doente, e cerca de 1 em 7 S passa a ser chamada doente.
- **A escolha do t_d (provisório) é do gestor:** estes números não ajustaram nada.

**Classes das doentes mantidas na cadeia final (não verificadas: nenhuma D tem classe):**
- **As 42:** 34 ferrugem (81,0%; 66,7% a 90,0%), 4 phoma, 3 bicho-mineiro e 1 cercosporiose.
- **Só as 31 D:** 26 ferrugem, 3 bicho-mineiro, 1 phoma e 1 cercosporiose.
- **Confiança (P renormalizada entre as doenças):** mediana 0,81 (0,71 a 0,86), de 0,65 a 0,97.
  A ferrugem tem a mais alta (mediana 0,82), e a phoma, a mais baixa (0,69).
- **Sem classe nos rótulos,** não dá para dizer se o predomínio da ferrugem é a doença da lavoura
  ou um viés do classificador.

**S (77, juntas):**
- **Chamadas saudáveis:** 71,4% (60,5% a 80,3%).
- **Chamadas doentes:** 3,9%.
- **Descartadas:** 24,7%, sendo 8 pelo filtro de fragmentos (só nas antigas) e 11 pelo t_d.
- **Nas novas,** 80,5% chamadas saudáveis, igual à validação.

**"?" (97, juntas):**
- **Chamadas saudáveis:** 46,4%.
- **Chamadas doentes:** 8,2%.
- **Descartadas:** 45,4%, sendo 18 pelo filtro de fragmentos e 26 pelo t_d.

## Limites

- Rótulos feitos por Claude por leitura visual, aceitos pelo gestor em 11/10/2026, sem especialista; 1 lavoura e 2 sessões.
- Nenhuma D tem classe: a distribuição das classes descreve o que a cadeia responde, não se ela acerta. A severidade não é informativa no campo.
- As novas vêm de fotos em que o detector treinou: as máscaras saem mais limpas que as de fotos novas.
- As folhas foram sorteadas da cadeia da rotulagem de cada conjunto, não da cadeia final: nas antigas, entram fragmentos que a cadeia final não classificaria (o filtro estava desligado na rotulagem).
- O cenário sem t_d é só informação: nenhum limiar foi ajustado com estes números.

