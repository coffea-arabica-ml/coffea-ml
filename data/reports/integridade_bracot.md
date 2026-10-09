# Relatório de integridade do BRACOT

Gerado por `python data/bracot.py`. Não editar à mão: rode o script de novo.

> **Detecção e segmentação de folhas** (RF09; decisão do gestor, 09/10/2026). A divisão oficial é a dos autores (240 treino / 60 teste). Não há classe de estresse por folha, e só parte das folhas de cada foto está contornada.

## Resultado

**OK:** nenhum erro; o manifest corresponde aos dados.

Avisos: 2.

- 28 polígonos passam até 4 px da borda (treino 22, teste 6): quem lê recorta na borda
- o campo area do COCO é a área do bbox em 1.662 de 1.662 anotações: não usar; a área do manifest sai do polígono

## Parâmetros

| item | valor |
|---|---|
| entrada | `data/raw/bracot/bracot-data` |
| manifest | `data/manifests/bracot.csv`, uma linha por foto |
| divisão | a dos autores (train/ = treino, test/ = teste), oficial |
| cena | fotos a até 10 s da anterior, em ordem de hora, sem olhar o split |
| borda | polígono que passa até 4 px da borda é aviso; acima, erro |
| pHash | 256 bits, foto inteira |

## Arquivos

| pasta | split | fotos | MB | COCO | SHA-256 do COCO | SHA-256 do VIA |
|---|---|---:|---:|---|---|---|
| `train/` | treino | 240 | 980,6 | `train/train_annotation_coco.json` | `de186cdb5c291590…` | `deefec16348fa516…` |
| `test/` | teste | 60 | 255,0 | `test/test_annotation_coco.json` | `65a6a8d73cb6611f…` | `87c3d80dc3f11354…` |

| tamanho | orientação EXIF | fotos |
|---|---|---:|
| 4032x3024 | 1 | 300 |

Duplicatas exatas: 0 (SHA-256 distintos: 300 de 300).

## Validação das anotações

| COCO | treino | teste |
|---|---:|---:|
| anotações (folhas) | 1.318 | 344 |
| anotações com mais de um polígono | 0 | 0 |
| anotações com iscrowd diferente de 0 | 0 | 0 |
| polígonos que passam até 4 px da borda (aviso) | 22 | 6 |
| maior excesso na borda (px) | 4 | 4 |
| polígonos com o primeiro ponto repetido no fim (válido no COCO) | 1 | 0 |
| bbox incoerente com o polígono (mais de 1 px) | 0 | 0 |
| campo area igual à área do bbox (não usar) | 1.318 | 344 |

Ids únicos, `file_name` batendo com as fotos, nenhuma anotação sem foto, nenhuma foto sem anotação e uma única categoria (`leaf`, id 0): tudo isso é conferido, e qualquer falha aparece nos erros. A área de cada folha sai do polígono (fórmula do laço), nunca do campo `area`, que o export do VIA preencheu com a área do bbox.

| VIA (`via_export_json.json`) | treino | teste |
|---|---:|---:|
| regiões | 1.318 | 344 |
| regiões com region_attributes | 0 | 0 |
| fotos com file_attributes | 0 | 0 |

O VIA tem as mesmas fotos e o mesmo número de regiões do COCO, e o tamanho gravado em cada entrada é o do arquivo. Sem atributos preenchidos, não há classe por folha.

## Folhas por foto e área coberta

| split | fotos | folhas | folhas por foto (mín / mediana / máx) | área coberta, % (mín / mediana / máx) |
|---|---:|---:|---|---|
| treino | 240 | 1.318 | 2 / 5 / 12 | 9,7 / 34,5 / 67,3 |
| teste | 60 | 344 | 2 / 6 / 9 | 15,0 / 31,9 / 57,7 |

Fotos por número de folhas: 2: 11; 3: 32; 4: 53; 5: 60; 6: 46; 7: 53; 8: 29; 9: 10; 10: 4; 11: 1; 12: 1.

As folhas contornadas cobrem só parte de cada foto, que é quase toda folhagem: a limitação está descrita no data/README.md.

## Datas e cenas

| dia | treino | teste | total |
|---|---:|---:|---:|
| 2019-08-31 | 109 | 15 | 124 |
| 2019-12-08 | 131 | 45 | 176 |

Intervalo entre fotos seguidas no mesmo dia: mediana 7 s, p90 19 s, máximo 48 s; 88 intervalos passam de 10 s.

Cenas (fotos a até 10 s da anterior): 90, com mediana de 2 fotos e no máximo 22. Cenas com fotos de treino e de teste: 27; fotos de teste nessas cenas: 55 de 60.

## Vazamento treino→teste

Tempo de cada foto de teste até a foto de treino mais próxima: mediana 6 s, máximo 21 s. Fotos de teste até 10 s: 47; até 20 s: 59; até 30 s: 60; até 60 s: 60 (de 60).

Por tempo, todo o teste fica perto do treino; mas fotos seguidas costumam mostrar folhas diferentes. Por pontos casados (ORB + RANSAC, medidos em 09/10/2026), só estas fotos de teste repetem folhas de uma foto de treino (`TESTE_SOBREPOSTO`). Reporte o detector no teste inteiro e sem elas:

| teste | treino | segundos | pHash (bits) | mesma cena |
|---|---|---:|---:|---|
| 20190831_164920 | 20190831_164921 | 1 | 16 | sim |
| 20190831_165013 | 20190831_164937 | 36 | 116 | sim |

Os 10 pares mais próximos pelo pHash da foto inteira:

| foto A | foto B | distância (bits) | split A | split B | intervalo |
|---|---|---:|---|---|---:|
| 20190831_164920 | 20190831_164921 | 16 | teste | treino | 1 s |
| 20190831_165447 | 20191208_143307 | 94 | treino | teste | outro dia |
| 20191208_142311 | 20191208_142633 | 94 | treino | treino | 202 s |
| 20191208_143206 | 20191208_143916 | 96 | treino | treino | 430 s |
| 20190831_163410 | 20191208_142908 | 98 | treino | treino | outro dia |
| 20190831_164341 | 20191208_143421 | 98 | treino | treino | outro dia |
| 20190831_164529 | 20191208_142645 | 98 | treino | treino | outro dia |
| 20190831_164529 | 20191208_142846 | 98 | treino | treino | outro dia |
| 20190831_164649 | 20191208_142045 | 98 | treino | treino | outro dia |
| 20190831_164841 | 20190831_165410 | 98 | treino | treino | 329 s |

Hash da divisão dos autores ("id:split" em ordem de id, unidos por ";"): `890c3fb2c5065fedd3a5b3807cbcd082070dcfd24d75b579b263a352853c0317`. A divisão por cena foi simulada e ficou só documentada no data/README.md (decisão do gestor, 09/10/2026).

