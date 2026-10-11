**O que os intervalos sustentam.**

- No modo B, P(saudável) ordena bem S acima de D em todos os modelos: AUC de 0,93 a 0,98.
  Os intervalos dos cinco modelos se sobrepõem (atual: 0,92 a 0,99; v3s: 0,95 a 1,00), então
  nenhum modelo é melhor que outro aqui. Isso confirma a decisão de não fazer a validação
  cruzada das variantes.
- No modo A, a AUC fica entre 0,56 e 0,67, com limites inferiores de 0,42 a 0,55, perto do
  acaso. Em cada modelo, os intervalos de A e B não se tocam (atual: até 0,77 no A, desde 0,92
  no B). O recorte B por padrão tem apoio também por este lado.
- A sonda linear chega a 0,99 (0,98 a 1,00) nos embeddings do atual e a 1,00 (0,99 a 1,00)
  nos do v0, sem depender de C (0,01 a 1). Os embeddings congelados já separam S de D quase
  linearmente, com dobras agrupadas por cena.
  - Atual: o intervalo da sonda encosta no de P(saudável).
  - v0: não encosta (0,99 a 1,00 contra 0,89 a 0,98).
  - De P(saudável) até a AUC máxima, a folga é pequena: 4 pontos no atual e 6 no v0.
- Controles:
  - A sessão de fotos sozinha dá AUC de 0,55, e dentro de cada sessão as AUCs se mantêm. O
    sinal não vem da sessão.
  - Nas 65 folhas que ficam no filtro de fragmentos, a AUC também se mantém (0,96).

**O que isso quer dizer.** O classificador atual não está cego no campo. O sinal de folha
saudável existe e aparece na ordem de P(saudável). O que falha é a regra de decisão:

- No campo, P(saudável) fica comprimida. Os limiares do ponto de operação ficam entre 0,04 e
  0,09.
- Por isso o argmax manda 69% das S para doença (passo 4).
- Com o limiar de 0,083, o recall de S do modelo atual iria de 31% (argmax) para 83% (68% a
  92%), com 2 das 46 D marcadas saudáveis.
- Esse número é otimista por três motivos:
  - o limiar sai das mesmas 82 folhas;
  - os 38 "?" (os casos difíceis) ficaram de fora;
  - é uma lavoura só, em duas sessões, com rótulos que não são de especialista.

**Auditoria.**

- As 12 S previstas como doença com mais confiança: 9 vão para phoma, 2 para bicho-mineiro e 1
  para ferrugem.
- 11 das 12 têm confiança entre 31% e 43%, abaixo do limiar proposto de 0,61. Só a primeira
  passa (80%): uma folha grande cortada pelo quadro, que o filtro de fragmentos mantém.
- Grad-CAM das 8 primeiras: 6 olham para a ponta ou a borda da folha, 1 para uma dobra e 1
  para um reflexo.
- Com o limiar de 0,083, 7 das 12 passam a saudáveis. Das 5 que ficam como doença:
  - 2 têm motivo à vista: uma folha cortada e uma rasgada;
  - 3 mostram, na resolução nativa, uma marca pequena perto da ponta: ponta pardacenta em
    142146_02, ponto avermelhado em 142156_03 e área esbranquiçada em 142829_05.
  - Essas marcas podem ser sintoma inicial ou sujeira. Não mudei nenhum rótulo: revisar é
    decisão do gestor.

**Implicações para o caminho (a decisão é do gestor).**

1. **Recalibrar o limiar de saudável no campo.** É o passo mais barato e o que os dados
   apoiam primeiro: não treina nada, e o sinal está em P(saudável).
   - No BRACOL, a mesma regra mudaria pouco. Na validação, imagem inteira, 4 de 211 D viram
     saudáveis (1,9%, de 0,7% a 4,8%) contra 0 com o argmax.
   - Antes de virar padrão da cadeia, faltam três coisas:
     - **Validação fora da amostra.** O limiar fica fixado antes e é medido em folhas de campo
       rotuladas novas, de outras fotos e, de preferência, de outra lavoura. Com o limiar tão
       baixo, uma pequena mudança de calibração muda muito o resultado.
     - **Conferência do T1 com a cadeia B.**
     - **Decisão conjunta com o limiar de confiança, que segue pendente.** Uma folha que vira
       saudável por P(saudável) de 0,1 tem confiança de 0,1 e seria cortada pelo limiar de
       0,61. A regra nova exige redefinir a confiança dessas folhas.
   - A recalibração só resolve S x D. Ela não corrige qual doença é prevista: entre os erros,
     a phoma domina.
2. **Treinar um classificador S x D com folhas de campo rotuladas.** A sonda mostra que uma
   cabeça linear sobre os embeddings congelados bastaria, sem retreinar a rede. Mas:
   - 82 folhas de uma lavoura levariam a um ajuste fino a este conjunto;
   - o `campo_rotulado.csv` é, por decisão do gestor (10/10/2026), só de avaliação, e usá-lo
     no treino exigiria uma decisão nova e um conjunto de avaliação separado;
   - o ganho sobre P(saudável) recalibrada é pequeno no modelo atual (os intervalos se
     encostam).
   - Fica como alternativa caso o limiar recalibrado não se sustente em folhas novas.
3. **Nenhum dos dois.** Não é o que os dados indicam: o sinal existe, e o que falha é a regra
   de decisão. Sem folhas de campo rotuladas novas, porém, nenhuma das duas opções pode ser
   validada honestamente. O próximo dado que mais vale é esse.
