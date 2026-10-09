# Relatório de integridade do JMuBEN

Gerado por `python data/jmuben.py`. Não editar à mão: rode o script de novo.

> **Fonte auxiliar** (decisão do gestor, 09/10/2026): só treino de classificação, nunca validação nem teste. São recortes de folhas fotografadas em campo, com augmentation dos autores; os rótulos não foram verificados.
>
> **Ressalva:** A correspondência das colunas 'phoma' e 'cercospora' do dataset.csv (códigos 3 e 4 de predominant_stress) com as classes 'brown leaf spot' e 'cercospora leaf spot' de Esgario et al. (2020) ainda não foi confirmada. Aqui elas viram as classes 'phoma' e 'cercosporiose' do projeto. O mesmo vale para as pastas 'Phoma' e 'Cerscospora' do JMuBEN: viram 'phoma' e 'cercosporiose' sem confirmação de que nomeiam as mesmas doenças do BRACOL.

## Resultado

**OK:** nenhum erro; o manifest corresponde aos dados.

Avisos: nenhum.

## Parâmetros

| item | valor |
|---|---|
| entrada | `data/raw/jmuben` |
| manifest | `data/manifests/jmuben.csv`, uma linha por conteúdo distinto (SHA-256) |
| hash canônico | pHash de 256 bits: o menor entre as 8 transformações de giro e espelhamento do cinza equalizado, sem aplicar a orientação EXIF |
| grupo | SHA-256 igual ou hash canônico a até 64 bits, de forma transitiva |
| teto por classe | saudavel 95, ferrugem 186, bicho_mineiro 135, phoma 122, cercosporiose 51 |
| seleção | um representante por grupo; grupos e representantes por sorteio determinístico (seed 42) |
| ilegíveis esperados | `Healthy/2 (691).jpg` |

## Arquivos por pasta

| pasta | classe | subconjunto | arquivos | extensões | MB | conteúdos distintos | ilegíveis |
|---|---|---|---:|---|---:|---:|---:|
| `Cerscospora` | cercosporiose | jmuben | 7.681 | .jpg 7.681 | 251,9 | 322 | 0 |
| `Healthy` | saudavel | jmuben2 | 18.984 | .jpg 18.984 | 566,2 | 63 | 1 |
| `Leaf rust` | ferrugem | jmuben | 8.336 | .jpeg 144, .jpg 8.192 | 68,0 | 1.042 | 0 |
| `Miner` | bicho_mineiro | jmuben2 | 16.978 | .jpg 16.978 | 725,8 | 1.639 | 0 |
| `Phoma` | phoma | jmuben | 6.571 | .jpg 6.571 | 227,2 | 691 | 0 |
| **total** |  |  | 58.550 |  | 1.839,1 | 3.757 | 1 |

Arquivos por subconjunto: jmuben 22.588; jmuben2 35.962.

Arquivos ilegíveis (excluídos com motivo `arquivo_ilegivel`):

- `Healthy/2 (691).jpg`: UnidentifiedImageError: cannot identify image file

## Imagens

Contagens em arquivos (cada cópia conta). Tamanho e orientação como gravados.

| pasta | 128x128 | outros tamanhos | mais comuns fora de 128x128 |
|---|---:|---:|---|
| `Cerscospora` | 6.336 | 1.345 | 122x256 (192), 104x256 (96), 108x256 (96) |
| `Healthy` | 18.983 | 0 | - |
| `Leaf rust` | 8.336 | 0 | - |
| `Miner` | 15.370 | 1.608 | 111x256 (88), 102x256 (64), 103x256 (64) |
| `Phoma` | 6.571 | 0 | - |

| formato | modo | arquivos |
|---|---|---:|
| JPEG | RGB | 58.549 |

Orientação EXIF (tag 274). Varia entre as cópias: parte da "rotação" do augmentation dos autores está só na etiqueta. O Pillow ignora a etiqueta; o `cv2.imread` a aplica. O hash canônico não depende dela.

| pasta | sem | 0 | 1 | 3 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| `Cerscospora` | 0 | 0 | 1.920 | 1.921 | 1.920 | 1.920 |
| `Healthy` | 0 | 0 | 3.796 | 3.796 | 3.796 | 7.595 |
| `Leaf rust` | 3.368 | 0 | 800 | 0 | 4.168 | 0 |
| `Miner` | 0 | 32 | 4.212 | 0 | 4.246 | 8.488 |
| `Phoma` | 0 | 23 | 1.598 | 1.643 | 1.651 | 1.656 |

Conteúdos legíveis: 3.756; matrizes de pixels distintas: 1.357 (o resto difere só nos metadados, como a orientação EXIF).

## Nomes dos arquivos

| pasta | no padrão `N (k).jpg` | fora do padrão | arquivos por N |
|---|---:|---:|---|
| `Cerscospora` | 7.680 | 1 | 4: 960; 6: 960; 7: 960; 8: 960; 9: 3.840 |
| `Healthy` | 18.980 | 4 | 1: 4.745; 2: 4.745; 10: 9.490 |
| `Leaf rust` | 7.680 | 656 | 1: 521; 2: 1.042; 3: 2.084; 5: 4.168 |
| `Miner` | 16.978 | 0 | 1: 16.978 |
| `Phoma` | 6.571 | 0 | 1: 819; 2: 841; 3: 825; 4: 807; 6: 3.279 |

O prefixo N não identifica a foto de origem: cópias e variantes do mesmo recorte aparecem com N diferentes. Ele não é usado como grupo. Grupos com arquivos de N diferentes: Cerscospora 79 de 80; Healthy 12 de 12; Leaf rust 406 de 406; Miner 0 de 309; Phoma 173 de 177.

Nomes fora do padrão (dígitos trocados por `#`), incluídos como os outros:

| pasta | forma do nome | arquivos |
|---|---|---:|
| `Cerscospora` | `# (#) (Custom).jpg` | 1 |
| `Healthy` | `# (#).bak.jpg` | 3 |
| `Healthy` | `# (#) (Custom).jpg` | 1 |
| `Leaf rust` | `#_#.jpg.jpg` | 232 |
| `Leaf rust` | `# (#).jpeg` | 135 |
| `Leaf rust` | `#_#(#).jpg.jpg` | 123 |
| `Leaf rust` | `IMG_#_#.jpg` | 36 |
| `Leaf rust` | `#_#.jpg` | 33 |
| `Leaf rust` | `IMG_#_#_#.jpg.jpg` | 31 |
| `Leaf rust` | `IMG_#_#.jpg.jpg` | 26 |
| `Leaf rust` | `#_#(#).jpg` | 18 |
| `Leaf rust` | `IMG_#_#_#.jpg` | 13 |
| `Leaf rust` | `<hex>.jpeg` | 9 |

Grupos só com nomes fora do padrão (recortes que não aparecem como `N (k).jpg`): 9.

| grupo | pasta | arquivos | extensões | primeiro arquivo | selecionado |
|---|---|---:|---|---|---|
| `jmuben-386` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/059eaf863cca.jpeg` | não |
| `jmuben-387` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (1).jpeg` | não |
| `jmuben-499` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (2).jpeg` | não |
| `jmuben-723` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (4).jpeg` | sim |
| `jmuben-835` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (5).jpeg` | não |
| `jmuben-860` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (6).jpeg` | não |
| `jmuben-872` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (7).jpeg` | sim |
| `jmuben-884` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (8).jpeg` | sim |
| `jmuben-896` | `Leaf rust` | 16 | .jpeg 16 | `Leaf rust/1 (9).jpeg` | sim |

## Duplicatas e grupos

Duplicatas exatas: 3.747 conteúdos têm mais de uma cópia, cobrindo 58.540 arquivos. Cópias por conteúdo, as mais comuns: 8 cópias em 1.978 conteúdos; 10 cópias em 623 conteúdos; 16 cópias em 466 conteúdos; 24 cópias em 319 conteúdos; 6 cópias em 153 conteúdos; 12 cópias em 64 conteúdos; 4 cópias em 37 conteúdos; 2 cópias em 24 conteúdos; no máximo 520.

Um conteúdo entra no grupo de outro, de forma transitiva, se o SHA-256 for igual ou se o hash canônico ficar a até 64 bits.

| classe | grupos | conteúdos | arquivos | maior grupo (arquivos) | mediana (arquivos por grupo) |
|---|---:|---:|---:|---:|---:|
| saudavel | 12 | 62 | 18.983 | 1.800 | 1.580,5 |
| ferrugem | 406 | 1.042 | 8.336 | 128 | 16 |
| bicho_mineiro | 309 | 1.639 | 16.978 | 272 | 64 |
| phoma | 177 | 691 | 6.571 | 80 | 40 |
| cercosporiose | 80 | 322 | 7.681 | 192 | 96 |

Grupos com um arquivo só: 2.

Os 5 maiores grupos:

| grupo | classe | conteúdos | arquivos |
|---|---|---:|---:|
| `jmuben-335` | saudavel | 5 | 1.800 |
| `jmuben-336` | saudavel | 5 | 1.800 |
| `jmuben-337` | saudavel | 5 | 1.780 |
| `jmuben-338` | saudavel | 5 | 1.780 |
| `jmuben-339` | saudavel | 5 | 1.760 |

Os 10 pares mais próximos NÃO agrupados, pelo hash canônico:

| id A | id B | distância (bits) | classe A | classe B | arquivo A | arquivo B |
|---:|---:|---:|---|---|---|---|
| 431 | 432 | 66 | ferrugem | ferrugem | `Leaf rust/1 (138).jpg` | `Leaf rust/1 (139).jpg` |
| 431 | 1079 | 66 | ferrugem | ferrugem | `Leaf rust/1 (138).jpg` | `Leaf rust/2 (651).jpg` |
| 432 | 1078 | 66 | ferrugem | ferrugem | `Leaf rust/1 (139).jpg` | `Leaf rust/2 (650).jpg` |
| 578 | 600 | 66 | ferrugem | ferrugem | `Leaf rust/1 (27).jpg` | `Leaf rust/1 (29).jpg` |
| 578 | 969 | 66 | ferrugem | ferrugem | `Leaf rust/1 (27).jpg` | `Leaf rust/2 (541).jpg` |
| 600 | 967 | 66 | ferrugem | ferrugem | `Leaf rust/1 (29).jpg` | `Leaf rust/2 (539).jpg` |
| 831 | 846 | 66 | ferrugem | ferrugem | `Leaf rust/1 (496).jpg` | `Leaf rust/1 (508).jpg` |
| 831 | 929 | 66 | ferrugem | ferrugem | `Leaf rust/1 (496).jpg` | `Leaf rust/2 (1020).jpg` |
| 846 | 917 | 66 | ferrugem | ferrugem | `Leaf rust/1 (508).jpg` | `Leaf rust/2 (1008).jpg` |
| 891 | 895 | 66 | ferrugem | ferrugem | `Leaf rust/1 (85).jpg` | `Leaf rust/1 (89).jpg` |

Par de classes diferentes mais próximo: 4 (cercosporiose, `Cerscospora/4 (1001).jpg`) e 1965 (bicho_mineiro, `Miner/1 (11579).jpg`), a 86 bits; o limiar é 64.

## Seleção para o treino auxiliar

| classe | grupos | teto | selecionadas |
|---|---:|---:|---:|
| saudavel | 12 | 95 | 12 |
| ferrugem | 406 | 186 | 186 |
| bicho_mineiro | 309 | 135 | 135 |
| phoma | 177 | 122 | 122 |
| cercosporiose | 80 | 51 | 51 |
| **total** | 984 |  | 506 |

Hash da seleção (ids selecionados em ordem, unidos por ";"): `6b206f932ee7c561ac1ac82749e20e0d612c81bab882a948c82f001ed4ddb1a5`.

| uso | motivo | conteúdos | arquivos |
|---|---|---:|---:|
| excluida | `arquivo_ilegivel` | 1 | 1 |
| treino_auxiliar | (selecionada) | 506 | 9.219 |
| treino_auxiliar | `acima_do_limite_da_classe` | 1.793 | 18.967 |
| treino_auxiliar | `grupo_ja_representado` | 1.457 | 30.363 |

