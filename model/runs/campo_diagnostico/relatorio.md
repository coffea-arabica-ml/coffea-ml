# Diagnóstico de campo sem treinar rede (passo 5a)

Gerado por `python evaluation/campo_diagnostico.py`. 82 folhas rotuladas do campo (36 S e 46 D, sem os "?"), de 42 fotos de validação do BRACOT em 22 cenas. Rótulos feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026; não são de especialista. Testes do BRACOL e do BRACOT fechados; checkpoints existentes, nenhum treino.

## a) P(saudável) separa S de D?

AUC de P(saudável), com S positiva; intervalo de 95% por bootstrap agrupado por foto (2.000 reamostras). Ponto de operação: o limiar em P(saudável) com no máximo 5% das D marcadas como saudáveis, e o recall de S nele (Wilson). **O limiar é escolhido nas mesmas 82 folhas em que é medido: o recall sai otimista.**

| modelo | modo | AUC (IC 95%) | limiar em P(saudável) | D marcadas saudáveis | recall de S (IC 95%) |
|---|---|---|---|---|---|
| atual | A | 0,64 (0,50 a 0,77) | 0,072 | 2 de 46 | 22,2% (11,7% a 38,1%; 8 de 36) |
| atual | B | 0,96 (0,92 a 0,99) | 0,083 | 2 de 46 | 83,3% (68,1% a 92,1%; 30 de 36) |
| v0 | A | 0,56 (0,42 a 0,70) | 0,073 | 2 de 46 | 19,4% (9,8% a 35,0%; 7 de 36) |
| v0 | B | 0,94 (0,89 a 0,98) | 0,086 | 2 de 46 | 80,6% (65,0% a 90,2%; 29 de 36) |
| v1 | A | 0,56 (0,43 a 0,70) | 0,071 | 2 de 46 | 8,3% (2,9% a 21,8%; 3 de 36) |
| v1 | B | 0,93 (0,88 a 0,98) | 0,077 | 2 de 46 | 72,2% (56,0% a 84,2%; 26 de 36) |
| v3 | A | 0,64 (0,50 a 0,76) | 0,012 | 2 de 46 | 19,4% (9,8% a 35,0%; 7 de 36) |
| v3 | B | 0,98 (0,94 a 1,00) | 0,039 | 2 de 46 | 91,7% (78,2% a 97,1%; 33 de 36) |
| v3s | A | 0,67 (0,55 a 0,79) | 0,024 | 2 de 46 | 33,3% (20,2% a 49,7%; 12 de 36) |
| v3s | B | 0,98 (0,95 a 1,00) | 0,053 | 2 de 46 | 94,4% (81,9% a 98,5%; 34 de 36) |

Efeito colateral no BRACOL: a regra "saudável se P(saudável) > 0,083" (o limiar do modelo atual no modo B) na validação do BRACOL (41 saudáveis e 211 doentes; predições da imagem inteira em `base_resnet50_bracol/predicoes_val.csv`, não a cadeia; nunca o teste), contra o argmax de hoje:

| regra | P(prever saudável dado D) | P(prever doença dado S) |
|---|---|---|
| argmax (hoje) | 0,0% (0,0% a 1,8%; 0 de 211) | 2,4% (0,4% a 12,6%; 1 de 41) |
| P(saudável) > 0,083 | 1,9% (0,7% a 4,8%; 4 de 211) | 0,0% (0,0% a 8,6%; 0 de 41) |

## b) Sonda linear nos embeddings congelados (modo B)

Regressão logística L2 (C = 0,1, fixado antes; dados padronizados no treino de cada dobra) sobre a saída do backbone (2.048 números por folha), S x D, validação cruzada de 5 dobras agrupada por cena. AUC dos escores fora da dobra, com intervalo por bootstrap agrupado por foto. Só mede se há sinal linear nos embeddings; não gera modelo.

| embeddings | AUC da sonda (IC 95%) | AUC com C = 0,01 | AUC com C = 0,10 | AUC com C = 1,00 | AUC de P(saudável), modo B (mesmo modelo) |
|---|---|---|---|---|---|
| atual | 0,99 (0,98 a 1,00) | 0,99 | 0,99 | 0,99 | 0,96 (0,92 a 0,99) |
| v0 | 1,00 (0,99 a 1,00) | 1,00 | 1,00 | 1,00 | 0,94 (0,89 a 0,98) |

### Controles

Se S e D viessem de sessões de fotos diferentes, a AUC poderia medir a sessão e não a folha. A sessão sozinha, como escore, dá AUC de 0,55. A última linha refaz a AUC só nas folhas que ficam no filtro de fragmentos (fração visível de pelo menos 0,5), que é o que a cadeia classifica com os padrões de 11/10/2026 (sem intervalo: subconjuntos pequenos).

| folhas | S | D | AUC: P(saudável) do atual, B | AUC: sonda do atual | AUC: sonda do v0 |
|---|---:|---:|---|---|---|
| sessão 20190831 | 12 | 20 | 0,98 | 0,99 | 1,00 |
| sessão 20191208 | 24 | 26 | 0,96 | 1,00 | 1,00 |
| ficam no filtro de fragmentos | 28 | 37 | 0,96 | 0,99 | 1,00 |

## c) Auditoria das S previstas como doença (modelo atual, modo B)

12 folhas S que o modelo atual previu como doença com maior confiança. As pranchas ficam fora do git: `data/processed/campo_diagnostico/auditoria_s_nativa.jpg` (resolução nativa) e `auditoria_s_gradcam.jpg` (Grad-CAM de 8).

| folha | prevista | confiança | P(saudável) | observação |
|---|---|---|---|---|
| 20191208_143600_02 | phoma | 79,8% | 2,6% | fragmento; escura |
| 20191208_142420_06 | ferrugem | 43,3% | 4,9% | rasgada; sem manchas |
| 20190831_164144_01 | phoma | 40,8% | 13,5% | - |
| 20191208_142829_05 | phoma | 40,4% | 5,7% | - |
| 20191208_142146_02 | phoma | 38,3% | 5,2% | escura |
| 20191208_143600_04 | phoma | 36,6% | 12,8% | clara; brilho |
| 20190831_163458_04 | bicho_mineiro | 35,2% | 17,2% | dobrada |
| 20191208_142829_07 | phoma | 34,6% | 13,5% | fragmento |
| 20190831_163458_05 | bicho_mineiro | 32,9% | 12,2% | brilho; clara |
| 20191208_142156_03 | phoma | 32,4% | 7,9% | brilho |
| 20191208_141947_05 | phoma | 31,9% | 18,5% | clara; brilho |
| 20191208_142536_08 | phoma | 30,9% | 21,4% | fragmento |

## d) O que isso implica

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

