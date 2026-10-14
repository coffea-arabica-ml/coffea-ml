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
