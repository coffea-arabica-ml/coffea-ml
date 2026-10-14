**A detecção no teste fica abaixo da validação nas métricas de AP, e a diferença no recall não é sustentada.**

- **AP50 de máscara, modo região:** 92,0% nas 60 fotos e 91,6% sem as 2 sobrepostas, contra
  94,7% na validação. No modo conservador, 81,6% e 81,3%, contra 88,1%. A AP não tem intervalo
  calculado aqui, então essas diferenças não são testadas.
- **No limiar de 0,35 (modo região, 60 fotos):**
  - Precisão: 89,2% (289 de 324), igual à da validação (89,0%).
  - Recall: 84,0% (79,8% a 87,5%; 289 de 344), contra 88,3% (83,7% a 91,7%; 219 de 248) na
    validação. Os intervalos se sobrepõem.
- **Sem as 2 fotos de `TESTE_SOBREPOSTO`,** os números quase não mudam: o recall fica em 83,4%
  (79,1% a 87,1%). Não há sinal de que as fotos que repetem folhas do treino inflem o teste.
- **Folhas por foto:**
  - O detector prevê mais folhas que as anotadas em 39 das 60 fotos (1,4 a mais por foto, em
    média). Como a anotação é parcial, isso não é necessariamente erro.
  - Prevê menos que as anotadas em 14 fotos, contra 2 das 46 na validação. Isso combina com o
    recall menor.
- **Nada foi ajustado:** o limiar (0,35) e os pesos continuam os mesmos.
