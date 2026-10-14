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
