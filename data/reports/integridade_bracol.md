# Relatório de integridade do BRACOL

Gerado por `python data/organize_dataset.py`. Não editar à mão: rode o script de novo.

> **Cópia completa do BRACOL:** 1.747 imagens.
>
> **Ressalva:** A correspondência das colunas 'phoma' e 'cercospora' do dataset.csv (códigos 3 e 4 de predominant_stress) com as classes 'brown leaf spot' e 'cercospora leaf spot' de Esgario et al. (2020) ainda não foi confirmada. Aqui elas viram as classes 'phoma' e 'cercosporiose' do projeto.

## Resultado

**OK:** nenhum erro; o manifest corresponde aos dados.

Avisos: nenhum.

## Parâmetros

| item | valor |
|---|---|
| entrada | `data/raw/bracol/bracol_completo/coffee-datasets/leaf` |
| dataset.csv | 1.747 linhas, SHA-256 `e74bb83e681812c225c9b733720b88134819710c9b8acf213c0b25ea904d4a93` |
| manifest | `data/manifests/bracol.csv` |
| proporções | treino 70%, val 15%, teste 15% |
| seed | 42 |
| divisão | estável: imagens que já tinham split no manifest anterior ficam onde estavam |
| pHash do quadro | 256 bits; agrupa a até 32 bits |
| pHash da folha | 256 bits, recorte da folha em 512x256; agrupa a até 46 bits |
| folhas repetidas | 8 pares conferidos visualmente em 07/10/2026 (`PARES_MESMA_FOLHA`), sempre agrupados |

## CSV x arquivos

|  | quantidade |
|---|---:|
| linhas no dataset.csv | 1.747 |
| imagens `<id>.jpg` lidas | 1.747 |
| imagens sem linha no csv | 0 |
| ids sem imagem | 0 |
| ids sem imagem, esperados | 0 |
| ids sem imagem, inesperados | 0 |

| classe | no csv | com imagem | sem imagem | perda |
|---|---:|---:|---:|---:|
| saudavel | 272 | 272 | 0 | 0,0% |
| ferrugem | 531 | 531 | 0 | 0,0% |
| bicho_mineiro | 387 | 387 | 0 | 0,0% |
| phoma | 348 | 348 | 0 | 0,0% |
| cercosporiose | 147 | 147 | 0 | 0,0% |
| (classe 5) | 62 | 62 | 0 | 0,0% |
| **total** | 1.747 | 1.747 | 0 | 0,0% |

## Imagens

| formato | modo | resolução | imagens |
|---|---|---|---:|
| JPEG | RGB | 2048x1024 | 1.747 |

As imagens desta tabela abriram e decodificaram por completo; as ilegíveis aparecem nos erros.

## Duplicatas e folhas repetidas

Uma imagem entra no grupo de outra (de forma transitiva) se o SHA-256 for igual, se o pHash do quadro ficar a até 32 bits, se o pHash da folha ficar a até 46 bits, ou se o par estiver na lista de folhas repetidas conferidas visualmente em 07/10/2026 (`PARES_MESMA_FOLHA`). Um grupo nunca se divide entre splits.

- Duplicatas exatas (mesmo SHA-256): 1 grupo(s).
- Grupos com mais de uma imagem: 8.

| grupo | ids | classe e severidade | split | quadro (bits) | folha (bits) | critério |
|---|---|---|---|---:|---:|---|
| `bracol-469` | 469 / 471 | phoma, severidade 1 | treino | 104 | 50 | lista |
| `bracol-758` | 758 / 759 | saudavel, severidade 0 | treino | 28 | 32 | quadro + folha + lista |
| `bracol-760` | 760 / 764 | ferrugem, severidade 1 | treino | 68 | 32 | folha + lista |
| `bracol-813` | 813 / 1022 | ferrugem, severidade 1 | treino | 0 | 0 | SHA-256 + quadro + folha + lista |
| `bracol-1295` | 1295 / 1296 | ferrugem, severidade 1 | treino | 60 | 42 | folha + lista |
| `bracol-1352` | 1352 / 1356 | ferrugem, severidade 2 | treino | 82 | 36 | folha + lista |
| `bracol-1692` | 1692 / 1694 | ferrugem, severidade 1 | treino | 62 | 30 | folha + lista |
| `bracol-1715` | 1715 / 1722 | (classe 5), severidade 1 | (excluída) | 84 | 18 | folha + lista |

Os 10 pares mais próximos NÃO agrupados, pelo pHash do quadro:

| id A | id B | distância (bits) | classe A | classe B |
|---:|---:|---:|---|---|
| 131 | 140 | 48 | ferrugem | ferrugem |
| 942 | 1633 | 48 | saudavel | ferrugem |
| 1115 | 1190 | 48 | saudavel | bicho_mineiro |
| 810 | 957 | 50 | saudavel | saudavel |
| 3 | 1070 | 52 | saudavel | bicho_mineiro |
| 113 | 116 | 52 | ferrugem | ferrugem |
| 125 | 144 | 52 | ferrugem | ferrugem |
| 125 | 810 | 52 | ferrugem | saudavel |
| 575 | 634 | 52 | bicho_mineiro | cercosporiose |
| 762 | 1511 | 52 | bicho_mineiro | (classe 5) |

Os 10 pares mais próximos NÃO agrupados, pelo pHash da folha:

| id A | id B | distância (bits) | classe A | classe B |
|---:|---:|---:|---|---|
| 134 | 160 | 50 | ferrugem | ferrugem |
| 665 | 1275 | 54 | ferrugem | ferrugem |
| 954 | 971 | 54 | saudavel | saudavel |
| 986 | 987 | 54 | saudavel | saudavel |
| 1494 | 1684 | 54 | ferrugem | ferrugem |
| 946 | 949 | 56 | saudavel | saudavel |
| 947 | 1174 | 56 | saudavel | ferrugem |
| 34 | 164 | 58 | bicho_mineiro | ferrugem |
| 140 | 879 | 58 | ferrugem | saudavel |
| 626 | 941 | 58 | saudavel | saudavel |

Pares para conferir (folhas escuras parecidas, sem veredito):

| ids | classe e severidade | split | quadro (bits) | folha (bits) | critério |
|---|---|---|---:|---:|---|
| 953 / 959 | saudavel, severidade 0 | teste / treino | 70 | 62 | nenhum |
| 986 / 987 | saudavel, severidade 0 | treino | 106 | 54 | nenhum |

## Exclusões

| motivo | linhas |
|---|---:|
| `predominant_stress_5` | 62 |
| **total excluídas** | 62 |
| **elegíveis** | 1.685 |

## Classe 5 (predominant_stress = 5)

Indeterminada (`5 - undetermined` no leaf/legend.txt dos autores): fica fora de classificação, severidade e multirrótulo, como os autores também fizeram no dataset.csv dos experimentos deles. Não atribuir classe por palpite.

Linhas: 62, 62 com imagem (sem imagem: nenhuma).

| estresses marcados (com imagem) | imagens |
|---|---:|
| rust+cercospora | 30 |
| miner+rust | 20 |
| miner+cercospora | 3 |
| miner+rust+cercospora | 3 |
| (nenhum) | 2 |
| cercospora | 1 |
| phoma | 1 |
| phoma+cercospora | 1 |
| rust | 1 |

## Divisão

| classe | treino | val | teste | total |
|---|---:|---:|---:|---:|
| saudavel | 190 | 41 | 41 | 272 |
| ferrugem | 372 | 79 | 80 | 531 |
| bicho_mineiro | 271 | 58 | 58 | 387 |
| phoma | 244 | 52 | 52 | 348 |
| cercosporiose | 103 | 22 | 22 | 147 |
| **total** | 1.180 | 252 | 253 | 1.685 |

| severidade | treino | val | teste | % no teste |
|---|---:|---:|---:|---:|
| 0: saudável (<0,1%) | 190 | 41 | 41 | 15,1% |
| 1: muito baixa (0,1-5%) | 647 | 138 | 139 | 15,0% |
| 2: baixa (5-10%) | 233 | 49 | 50 | 15,1% |
| 3: alta (10-15%) | 71 | 15 | 15 | 14,9% |
| 4: muito alta (>15%) | 39 | 9 | 8 | 14,3% |

Histórico da divisão:

- 07/10/2026: divisão refeita do zero (--refazer-divisao) ao adotar a cópia completa do BRACOL: nenhum modelo tinha sido treinado, e manter a divisão anterior deixaria partido o par 469/471, a mesma folha fotografada duas vezes. Daqui em diante a divisão é estável.
