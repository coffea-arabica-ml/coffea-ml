# Relatório de integridade do BRACOL

Gerado por `python data/organize_dataset.py`. Não editar à mão: rode o script de novo.

> **Cópia PARCIAL do BRACOL:** 1.401 de 1.747 imagens. Resultados com ela não são comparáveis com Esgario et al. (2020).
>
> **Ressalva:** A correspondência das colunas 'phoma' e 'cercospora' do dataset.csv (códigos 3 e 4 de predominant_stress) com as classes 'brown leaf spot' e 'cercospora leaf spot' de Esgario et al. (2020) ainda não foi confirmada. Aqui elas viram as classes 'phoma' e 'cercosporiose' do projeto.

## Resultado

**OK:** nenhum erro; manifest gravado.

Avisos: nenhum.

## Parâmetros

| item | valor |
|---|---|
| entrada | `data/raw/bracol/bracol_recuperado/coffee-datasets/coffee-datasets/leaf` |
| dataset.csv | 1.747 linhas, SHA-256 `e74bb83e681812c225c9b733720b88134819710c9b8acf213c0b25ea904d4a93` |
| manifest | `data/manifests/bracol.csv` |
| proporções | treino 70%, val 15%, teste 15% |
| seed | 42 |
| divisão | estável: imagens que já tinham split no manifest anterior ficam onde estavam |
| pHash | 256 bits (`hash_size=16`); quase-duplicata a até 24 bits |

## CSV x arquivos

|  | quantidade |
|---|---:|
| linhas no dataset.csv | 1.747 |
| imagens `<id>.jpg` lidas | 1.401 |
| imagens sem linha no csv | 0 |
| ids sem imagem | 346 |
| ids sem imagem, esperados (truncamento do zip) | 346 |
| ids sem imagem, inesperados | 0 |

Ids sem imagem: 7-9, 69-99, 688-999.

| classe | no csv | com imagem | sem imagem | perda |
|---|---:|---:|---:|---:|
| saudavel | 272 | 142 | 130 | 47,8% |
| ferrugem | 531 | 465 | 66 | 12,4% |
| bicho_mineiro | 387 | 253 | 134 | 34,6% |
| phoma | 348 | 346 | 2 | 0,6% |
| cercosporiose | 147 | 136 | 11 | 7,5% |
| (classe 5) | 62 | 59 | 3 | 4,8% |
| **total** | 1.747 | 1.401 | 346 | 19,8% |

## Imagens

| formato | modo | resolução | imagens |
|---|---|---|---:|
| JPEG | RGB | 2048x1024 | 1.401 |

As imagens desta tabela abriram e decodificaram por completo; as ilegíveis aparecem nos erros.

## Duplicatas

- Duplicatas exatas (mesmo SHA-256): 0 grupo(s).
- Grupos com mais de uma imagem (SHA-256 igual ou pHash a até 24 bits): 0.

Pares mais próximos pelo pHash (para conferir o limiar):

| id A | id B | distância (bits) | classe A | classe B |
|---:|---:|---:|---|---|
| 131 | 140 | 48 | ferrugem | ferrugem |
| 1115 | 1190 | 48 | saudavel | bicho_mineiro |
| 3 | 1070 | 52 | saudavel | bicho_mineiro |
| 113 | 116 | 52 | ferrugem | ferrugem |
| 125 | 144 | 52 | ferrugem | ferrugem |
| 575 | 634 | 52 | bicho_mineiro | cercosporiose |
| 22 | 113 | 54 | bicho_mineiro | ferrugem |
| 101 | 138 | 54 | ferrugem | ferrugem |
| 114 | 1504 | 54 | ferrugem | bicho_mineiro |
| 131 | 159 | 54 | ferrugem | ferrugem |

## Exclusões

| motivo | linhas |
|---|---:|
| `imagem_ausente` | 343 |
| `imagem_ausente;predominant_stress_5` | 3 |
| `predominant_stress_5` | 59 |
| **total excluídas** | 405 |
| **elegíveis** | 1.342 |

## Classe 5 (predominant_stress = 5)

Significado desconhecido (os autores foram consultados): fica fora de classificação, severidade e multirrótulo até a resposta. Não atribuir classe por palpite.

Linhas: 62, 59 com imagem (sem imagem: 710, 743, 925).

| estresses marcados (com imagem) | imagens |
|---|---:|
| rust+cercospora | 29 |
| miner+rust | 18 |
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
| saudavel | 100 | 21 | 21 | 142 |
| ferrugem | 325 | 70 | 70 | 465 |
| bicho_mineiro | 177 | 38 | 38 | 253 |
| phoma | 242 | 52 | 52 | 346 |
| cercosporiose | 95 | 20 | 21 | 136 |
| **total** | 939 | 201 | 202 | 1.342 |

| severidade | treino | val | teste | % no teste |
|---|---:|---:|---:|---:|
| 0: saudável (<0,1%) | 100 | 21 | 21 | 14,8% |
| 1: muito baixa (0,1-5%) | 555 | 118 | 120 | 15,1% |
| 2: baixa (5-10%) | 192 | 42 | 41 | 14,9% |
| 3: alta (10-15%) | 58 | 12 | 13 | 15,7% |
| 4: muito alta (>15%) | 34 | 8 | 7 | 14,3% |
