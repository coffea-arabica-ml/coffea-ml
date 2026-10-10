Em 10/10/2026 olhei os 24 recortes das duas pranchas desta rodada, com o modo A (foto em volta) e o B (fundo cinza) lado a lado e a previsão de cada um. Numa rodada anterior, com a mesma geometria de recorte e antes do filtro de cor, olhei outros 24. As pranchas e os recortes ficaram só no rascunho da sessão, fora do repositório.

- **Geometria:** funciona. A folha sai deitada, centrada e com a margem do BRACOL, em qualquer ângulo. A máscara do modo B quase sempre segue a borda da folha. Às vezes leva junto um pedaço da folha vizinha (20190831_163458, folha 8; 20190831_164207, folha 5).
- **Acertos óbvios:** lesões nítidas viram doença.
  - Uma mancha alaranjada isolada vira ferrugem, com 81% (A) e 93% (B) (20190831_163627, folha 2).
  - Uma mancha marrom com halo amarelo vira cercosporiose (20191208_142439, folha 6).
  - Necroses grandes com halo amarelo viram phoma no A e ferrugem no B (20191208_142532, folha 6). A classe é incerta, mas a folha é doente.
- **Erros óbvios:** a maioria das folhas sem sintoma visível recebe uma doença, com confiança entre 30% e 60%.
  - No modo A, isso acontece quase sempre: só 1 das 318 folhas saiu saudável.
  - No modo B, folhas sem sintoma às vezes saem saudáveis (20191208_142532, folha 4; 20190831_163138, folha 1; 20191208_143010, folha 3).
  - Com o limiar proposto (0,61), ficam 107 folhas no modo A, e **102 delas são "ferrugem"**. Várias são folhas lisas, sem lesão, como a folha escura na sombra de 20191208_143712, folha 1 (ferrugem, 71%).
- **Recortes ruins:**
  - pedaços de folha cortados pela borda da foto, às vezes só a ponta ou metade da folha (20191208_142551, folha 3; 20191208_143723, folha 1; 20190831_163138, folha 4);
  - folhas escuras na sombra (20191208_142539, folha 9).
- **Borda:** a confiança das folhas cortadas pela borda é parecida com a das inteiras (52% contra 55% no modo A; abaixo de 0,61: 67% contra 65%). Os recortes delas mostram folhas incompletas. Mesmo assim, não há sinal de que errem mais que as inteiras, porque as inteiras também erram. Mantive a decisão de classificá-las e não propus filtro de borda. Vale reavaliar com um classificador que funcione no campo.

**Conclusão:** a cadeia funciona como mecânica: detecção, recorte, região, regras e contrato. O classificador, porém, foi treinado só com folhas do BRACOL, sobre fundo claro e com o atalho do tom do fundo, e não transfere para as folhas da planta no campo: puxa quase tudo para doença, sobretudo ferrugem.

- **Modo de recorte:** fiquei com o modo A pelo único teste com rótulo (T1). No BRACOL, só o recorte A acerta 92,4%, contra 88,4% do B. O campo não decide entre os dois, e nenhum é confiável lá.
- **Próximo passo, que é decisão do gestor:** retreinar o classificador para o campo. Um caminho é treinar com o fundo trocado por cinza a partir de máscaras, o que combina com o modo B e tira o atalho. Outro é treinar com fundos de campo ou com outra fonte.
