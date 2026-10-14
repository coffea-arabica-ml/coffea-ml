# Uso 3b: folhas de campo do teste do BRACOT, parte B (passo 7)

Gerado por `python model/teste_final.py --uso 3b` em 14/10/2026 12:27. **Uso único e declarado dos testes**, com a cadeia congelada; nada foi ajustado nem escolhido por estes números. Início, fim, arquivos e SHA-256 em `model/runs/uso_do_teste_final.md`.

As folhas rotuladas das fotos de teste do BRACOT pela cadeia final congelada (estrutura do passo 6b, `evaluation/campo_cadeia_final.py`). Os números com o corte t_d desligado são só informação, declarada antes da abertura.

**Contagens:** {'S': 37, 'D': 46, '?': 32}; destinos na cadeia: {'classificada': 115}; planos das fotos: {'deteccao': 58}; rótulos sem par: 0.

## Destino de S, D e "?" (cadeia congelada)

| rótulo | folhas | chamada saudável | chamada doente e mantida | classes | descartada | descartada: filtro de fragmentos | descartada: corte t_d | descartada: outro motivo |
|---|---:|---|---|---|---|---|---|---|
| S | 37 | 89,2% (75,3% a 95,7%; 33 de 37) | 5,4% (1,5% a 17,7%; 2 de 37) | phoma 2 | 5,4% (1,5% a 17,7%; 2 de 37) | 0,0% (0,0% a 9,4%; 0 de 37) | 5,4% (1,5% a 17,7%; 2 de 37) | 0,0% (0,0% a 9,4%; 0 de 37) |
| D | 46 | 10,9% (4,7% a 23,0%; 5 de 46) | 26,1% (15,6% a 40,3%; 12 de 46) | ferrugem 9, bicho_mineiro 2, phoma 1 | 63,0% (48,6% a 75,5%; 29 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) | 63,0% (48,6% a 75,5%; 29 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) |
| ? | 32 | 59,4% (42,3% a 74,5%; 19 de 32) | 6,2% (1,7% a 20,1%; 2 de 32) | ferrugem 1, bicho_mineiro 1 | 34,4% (20,4% a 51,7%; 11 de 32) | 0,0% (0,0% a 10,7%; 0 de 32) | 34,4% (20,4% a 51,7%; 11 de 32) | 0,0% (0,0% a 10,7%; 0 de 32) |

## D evidentes: o que chega ao usuário

| cenário | D | como doente | como saudável | somem | somem: filtro de fragmentos | somem: corte t_d | somem: outro motivo |
|---|---:|---|---|---|---|---|---|
| cadeia final (t_d = 0,65) | 46 | 26,1% (15,6% a 40,3%; 12 de 46) | 10,9% (4,7% a 23,0%; 5 de 46) | 63,0% (48,6% a 75,5%; 29 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) | 63,0% (48,6% a 75,5%; 29 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) |
| t_d desligado na detecção (só informação) | 46 | 89,1% (77,0% a 95,3%; 41 de 46) | 10,9% (4,7% a 23,0%; 5 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) | 0,0% (0,0% a 7,7%; 0 de 46) |

## Doentes mantidas: classes e confiança

As classes são as previstas; os rótulos de campo não têm classe.

| cenário | rótulo | mantidas | ferrugem | bicho_mineiro | phoma | cercosporiose | confiança |
|---|---|---:|---|---|---|---|---|
| cadeia final (t_d = 0,65) | S | 2 | 0,0% (0,0% a 65,8%; 0 de 2) | 0,0% (0,0% a 65,8%; 0 de 2) | 100,0% (34,2% a 100,0%; 2 de 2) | 0,0% (0,0% a 65,8%; 0 de 2) | 0,71 (0,68 a 0,74; mín. 0,65, máx. 0,78) |
| cadeia final (t_d = 0,65) | D | 12 | 75,0% (46,8% a 91,1%; 9 de 12) | 16,7% (4,7% a 44,8%; 2 de 12) | 8,3% (1,5% a 35,4%; 1 de 12) | 0,0% (0,0% a 24,2%; 0 de 12) | 0,74 (0,70 a 0,90; mín. 0,66, máx. 0,96) |
| cadeia final (t_d = 0,65) | ? | 2 | 50,0% (9,5% a 90,5%; 1 de 2) | 50,0% (9,5% a 90,5%; 1 de 2) | 0,0% (0,0% a 65,8%; 0 de 2) | 0,0% (0,0% a 65,8%; 0 de 2) | 0,81 (0,80 a 0,82; mín. 0,78, máx. 0,84) |
| cadeia final (t_d = 0,65) | todas | 16 | 62,5% (38,6% a 81,5%; 10 de 16) | 18,8% (6,6% a 43,0%; 3 de 16) | 18,8% (6,6% a 43,0%; 3 de 16) | 0,0% (0,0% a 19,4%; 0 de 16) | 0,76 (0,70 a 0,85; mín. 0,65, máx. 0,96) |
| t_d desligado na detecção (só informação) | S | 4 | 0,0% (0,0% a 49,0%; 0 de 4) | 25,0% (4,6% a 69,9%; 1 de 4) | 75,0% (30,1% a 95,4%; 3 de 4) | 0,0% (0,0% a 49,0%; 0 de 4) | 0,59 (0,49 a 0,68; mín. 0,35, máx. 0,78) |
| t_d desligado na detecção (só informação) | D | 41 | 56,1% (41,0% a 70,1%; 23 de 41) | 26,8% (15,7% a 41,9%; 11 de 41) | 14,6% (6,9% a 28,4%; 6 de 41) | 2,4% (0,4% a 12,6%; 1 de 41) | 0,50 (0,41 a 0,68; mín. 0,33, máx. 0,96) |
| t_d desligado na detecção (só informação) | ? | 13 | 38,5% (17,7% a 64,5%; 5 de 13) | 7,7% (1,4% a 33,3%; 1 de 13) | 53,8% (29,1% a 76,8%; 7 de 13) | 0,0% (0,0% a 22,8%; 0 de 13) | 0,44 (0,38 a 0,57; mín. 0,36, máx. 0,84) |
| t_d desligado na detecção (só informação) | todas | 58 | 48,3% (35,9% a 60,8%; 28 de 58) | 22,4% (13,6% a 34,7%; 13 de 58) | 27,6% (17,8% a 40,2%; 16 de 58) | 1,7% (0,3% a 9,1%; 1 de 58) | 0,49 (0,39 a 0,66; mín. 0,33, máx. 0,96) |

## Separação S x D nas folhas classificadas

| cenário dos "?" | S / D | AUC de P(saudável) (IC 95%, bootstrap por foto) | saudável dado S (final) | saudável dado D (final) | saudável dado D (sem t_d) |
|---|---|---|---|---|---|
| sem os "?" | 37 / 46 | 0,98 (0,95 a 0,99) | 89,2% (75,3% a 95,7%; 33 de 37) | 10,9% (4,7% a 23,0%; 5 de 46) | 10,9% (4,7% a 23,0%; 5 de 46) |
| "?" como S | 69 / 46 | 0,91 (0,85 a 0,97) | 75,4% (64,0% a 84,0%; 52 de 69) | 10,9% (4,7% a 23,0%; 5 de 46) | 10,9% (4,7% a 23,0%; 5 de 46) |
| "?" como D | 37 / 78 | 0,88 (0,81 a 0,93) | 89,2% (75,3% a 95,7%; 33 de 37) | 30,8% (21,6% a 41,7%; 24 de 78) | 30,8% (21,6% a 41,7%; 24 de 78) |

## Comparação com o passo 6b, sem ajuste

Fontes: `model/runs/campo_cadeia_final/campo.json` e `model/runs/campo_validacao_duas_etapas/validacao.json`.

| conjunto | S chamada saudável | D chamada saudável | D chamada doente e mantida | D descartada | AUC de P(saudável) |
|---|---|---|---|---|---|
| teste | 89,2% (75,3% a 95,7%; 33 de 37) | 10,9% (4,7% a 23,0%; 5 de 46) | 26,1% (15,6% a 40,3%; 12 de 46) | 63,0% (48,6% a 75,5%; 29 de 46) | 0,98 (0,95 a 0,99) |
| antigas | 61,1% (44,9% a 75,2%; 22 de 36) | 2,2% (0,4% a 11,3%; 1 de 46) | 19,6% (10,7% a 33,2%; 9 de 46) | 78,3% (64,4% a 87,7%; 36 de 46) | 0,96 (0,92 a 0,99) |
| novas | 80,5% (66,0% a 89,8%; 33 de 41) | 6,7% (2,6% a 15,9%; 4 de 60) | 36,7% (25,6% a 49,3%; 22 de 60) | 56,7% (44,1% a 68,4%; 34 de 60) | 0,95 (0,89 a 0,99) |
| juntas | 71,4% (60,5% a 80,3%; 55 de 77) | 4,7% (2,0% a 10,6%; 5 de 106) | 29,2% (21,4% a 38,5%; 31 de 106) | 66,0% (56,6% a 74,4%; 70 de 106) | 0,95 (0,92 a 0,98) |

## Leitura

**O resultado principal é a cadeia congelada, em 115 folhas de 58 fotos de teste** (37 S, 46 D e 32 "?").
- **S:** 89,2% (75,3% a 95,7%; 33 de 37) chamadas saudáveis. Duas são chamadas doentes (phoma) e
  duas são descartadas pelo corte t_d.
- **D evidentes:**
  - chegam ao usuário como doente: 26,1% (15,6% a 40,3%; 12 de 46);
  - chegam como saudável: 10,9% (4,7% a 23,0%; 5 de 46);
  - somem: 63,0% (48,6% a 75,5%; 29 de 46), todas pelo corte t_d. O filtro de fragmentos não tira
    nenhuma, porque o sorteio já foi feito com ele ligado.
- **AUC de P(saudável):** 0,98 (0,95 a 0,99), com bootstrap por foto.

**O falso saudável no teste é o maior dos conjuntos rotulados, sem diferença sustentada.**
- No teste, 10,9% (4,7% a 23,0%), contra 6,7% (2,6% a 15,9%) nas novas e 4,7% (2,0% a 10,6%)
  nas antigas e novas juntas. Os intervalos se sobrepõem.
- O limite superior (23,0%) mostra, de novo, que os dados não sustentam o teto de 5% usado para
  fixar o limiar de saudável. Isso vai para as limitações.

**S chamadas saudáveis:** 89,2% no teste, contra 80,5% nas novas, com intervalos que se
sobrepõem. As antigas (61,1%) não são comparáveis diretamente, porque nelas o filtro de
fragmentos tira 8 das 36 S.

**"?":**
- **Chamadas saudáveis:** 59,4% (42,3% a 74,5%; 19 de 32), contra 54,2% nas novas.
- **Com os "?" como D:** o falso saudável vai a 30,8% (21,6% a 41,7%).
- **Com os "?" como S:** o recall de S cai para 75,4%.

**Com o t_d desligado (só informação, declarada antes da abertura):**
- **D que chegam como doente:** 89,1% (77,0% a 95,3%).
- **S chamadas doentes:** sobem de 2 para 4 (de 5,4% para 10,8%).
- **Confiança mediana das doentes mantidas:** cai de 0,76 para 0,49.

**Classes das 16 doentes mantidas (não verificadas: nenhuma D tem classe):**
- **Todas:** 10 ferrugem, 3 bicho-mineiro e 3 phoma.
- **Só as 12 D:** 9 ferrugem, 2 bicho-mineiro e 1 phoma.
- **Confiança:** mediana 0,76 (0,65 a 0,96).
- **As 2 S chamadas doentes saem como phoma,** como no passo 6b.

**Nada foi ajustado** por estes números: t_s 0,083, t_d 0,65 e corte 0,61 continuam os
congelados.

## Limites

- Folhas (S, D ou "?") rotuladas pelo gestor a partir das pranchas cegas, antes de qualquer predição (commit `1c59792`); nenhuma D tem classe.
- Mesma lavoura das outras rotulagens, em fotos e cenas do teste dos autores.
- As folhas de teste foram sorteadas já com o filtro de fragmentos ligado; as antigas do passo 6b, sem ele (por isso o filtro tira folhas só nas antigas).

