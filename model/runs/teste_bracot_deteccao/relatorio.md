# Uso 2: detecção no teste do BRACOT (passo 7)

Gerado por `python model/teste_final.py --uso 2` em 14/10/2026 12:13. **Uso único e declarado dos testes**, com a cadeia congelada; nada foi ajustado nem escolhido por estes números. Início, fim, arquivos e SHA-256 em `model/runs/uso_do_teste_final.md`.

As fotos de teste do BRACOT pelo `detector_base` (`weights/last.pt`), com as métricas da validação (`evaluation/metricas_deteccao.py`) e o limiar escolhido nela (0,35). Modo região: a previsão com mais da metade da máscara fora da região anotada é ignorada (a anotação é parcial); modo conservador: ela conta como erro. Sem o validador do ultralytics, que exigiria copiar as fotos de teste. Validação: `model/runs/detector_base/metricas_val.json`.

| conjunto | fotos | folhas anotadas | modo | AP50 máscara | AP50-95 máscara | AP50 caixa | AP50-95 caixa | precisão no limiar | recall no limiar | F1 no limiar |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| teste, 60 fotos | 60 | 344 | região (principal) | 92,0% | 83,5% | 92,0% | 81,3% | 89,2% | 84,0% | 86,5% |
| teste, 60 fotos | 60 | 344 | conservador | 81,6% | 74,7% | 81,8% | 72,8% | 67,7% | 84,0% | 75,0% |
| teste, sem as 2 sobrepostas | 58 | 332 | região (principal) | 91,6% | 83,0% | 91,7% | 80,8% | 88,8% | 83,4% | 86,0% |
| teste, sem as 2 sobrepostas | 58 | 332 | conservador | 81,3% | 74,3% | 81,6% | 72,4% | 67,6% | 83,4% | 74,7% |
| validação | 46 | 248 | região (principal) | 94,7% | 88,5% | 94,8% | 86,9% | 89,0% | 88,3% | 88,7% |
| validação | 46 | 248 | conservador | 88,1% | 82,9% | 88,3% | 81,3% | 68,2% | 88,3% | 77,0% |

**Folhas por foto, anotadas contra previstas no limiar** (`contagens_teste.csv`; validação: `model/runs/detector_base/contagens_val.csv`):

| conjunto | fotos | anotadas | previstas | acertos | fora da região | anotadas por foto (mediana; mín. a máx.) | previstas por foto (mediana; mín. a máx.) | previstas - anotadas por foto (média; mediana; mín. a máx.) | fotos com mais / menos / iguais |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| teste, 60 fotos | 60 | 344 | 427 | 289 | 103 | 6,0; 2 a 9 | 7,0; 2 a 14 | 1,4; 1,0; -2 a 6 | 39 / 14 / 7 |
| teste, sem as 2 sobrepostas | 58 | 332 | 410 | 277 | 98 | 6,0; 2 a 9 | 7,0; 2 a 14 | 1,3; 1,0; -2 a 6 | 37 / 14 / 7 |
| validação | 46 | 248 | 321 | 219 | 75 | 5,0; 2 a 11 | 6,5; 3 a 13 | 1,6; 1,0; -1 a 6 | 34 / 2 / 10 |

Como a anotação é parcial, previstas a mais não são necessariamente erros (o modo região as separa).

## Leitura

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

