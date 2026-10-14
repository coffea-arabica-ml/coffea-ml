**O teste fica abaixo da validação, sem diferença sustentada, e perto da validação cruzada.**

- **Classificador sozinho (informação):**
  - Teste: 90,5% (86,3% a 93,5%), F1 macro 88,5%. Validação: 94,8% (91,4% a 97,0%), F1 macro
    93,2%. Os intervalos se sobrepõem (de 91,4% a 93,5%).
  - O teste fica perto da validação cruzada nas 1.432 imagens de treino e validação: 89,9% por
    blocos (F1 87,2%) e 90,7% pelo protocolo aleatório (F1 88,2%).
  - A validação de 252 imagens também escolheu a época do checkpoint (a de melhor F1 macro), o
    que tende a deixá-la otimista.
- **Cadeia final (o resultado principal):**
  - **Sem corte:** 88,1% (83,5% a 91,5%), contra 92,8% (89,0% a 95,4%) na validação. Os
    intervalos se sobrepõem.
  - **No corte:** ficam 225 das 253 imagens (88,9%), e 91,1% (86,7% a 94,2%) delas estão certas,
    contra 94,4% na validação.
  - **Contra o classificador sozinho:** a cadeia perde 2,4 pontos sem corte no teste (88,1% contra
    90,5%), perto dos 2,0 pontos da validação.
- **Por caminho:**
  - **Foto inteira (192 imagens):** 93,1% (88,4% a 96,0%) das mantidas estão certas.
  - **Detecção (60 imagens, contra 48 na validação):** 84,0% (71,5% a 91,7%), contra 90,2% na
    validação. Os intervalos se sobrepõem.
  - Em 45 das 60 imagens da detecção, o detector achou mais de uma folha numa foto de folha
    única (38 de 48 na validação).
- **Doentes chamadas saudáveis:** 5 de 212 (2,4%; 1,0% a 5,4%), contra 3 de 211 na validação.
  - 3 vêm da foto inteira (argmax) e 2 da detecção (duas etapas).
  - As 5 são ferrugem. No classificador sozinho, 4 ferrugens viram saudáveis.
- **O que o usuário recebe:**
  - 231 respostas de sucesso, 21 `baixa_confianca`, 1 `planta_nao_identificada` e nenhuma
    exceção.
  - 33 respostas trazem mais de uma folha, por causa das detecções extras na mesma foto.
  - **Conferência com o T1:** 0 divergências nas 192 imagens da foto inteira, e o plano foi igual
    em todas.
- **Por classe:**
  - A cercosporiose tem 22 imagens. O recall dela na cadeia (77,3%) oscila 4,5 pontos por
    imagem.
  - Os erros mais comuns da cadeia são bicho-mineiro previsto como cercosporiose (6) ou phoma
    (4), e ferrugem prevista como saudável (5).
- **Severidade:** 80,6% exata e 96,0% com ±1 no classificador sozinho; 78,6% e 96,0% na cadeia.
- **Nada foi ajustado** por estes números.
