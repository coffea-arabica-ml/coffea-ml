No limiar escolhido (0,35), 75 previsões ficam fora da região anotada (mais da metade da máscara
fora), em 38 das 46 fotos de validação. Nenhuma se sobrepõe a uma folha anotada (IoU acima de 0,1:
nenhuma), 73 tocam a borda da foto e a área mediana é 2,2% da foto (p10 0,8%, p90 4,7%; a folha
anotada mediana do treino tem 5,0%). O score mediano é 0,75, e 33 têm 0,8 ou mais.

Em 10/10/2026 olhei 9 fotos com as detecções da periferia desenhadas, cobrindo 28 das 75
previsões: 20190831_163138, 20190831_164144, 20190831_164159, 20190831_165255, 20191208_141947,
20191208_142146, 20191208_142532, 20191208_142536 e 20191208_143600. As sobreposições ficaram só
no rascunho da sessão, fora do repositório, e foram apagadas no fim do passo.

- **Folhas de verdade: 26 das 28.** São folhas de café inteiras ou, quase sempre, cortadas pelo
  enquadramento, em foco e com o contorno seguindo a borda da folha. Os autores não as
  contornaram: o BRACOT anota cerca de 1/3 das folhas de cada foto, as do centro.
- **Erros: 2 das 28.** Em 20191208_142146 (score 0,58), a máscara cobre um objeto laranja
  desfocado no canto da foto (parece um dedo na frente da lente) junto com parte de uma folha. Em
  20191208_142532 (score 0,61), é um fragmento de folha com um pedaço de galho.
- **Polígono:** as sobreposições mostraram traços retos ligando pedaços da mesma máscara (o
  `masks.xy` do ultralytics emenda todos os pedaços num polígono só). O `detectar_folhas` passou
  a devolver o contorno do maior pedaço (ver `model/detector.py`).

**Conclusão:** a periferia não é lixo. A métrica principal, que não conta essas previsões, é a
mais justa, e a conservadora subestima a precisão. Por isso não há um segundo limiar mais alto:
ele também não separaria a periferia, que tem scores tão altos quanto os das folhas anotadas.
As folhas cortadas pela borda da foto são uma questão de produto. Se o backend não deve
classificar folha incompleta, o caminho é um filtro de borda ou de área, não o limiar; fica para
decisão do gestor.
