# Dados do coffea-ml (Frente 8)

Imagens de folhas de café de três fontes:
- **BRACOL:** classificação, severidade e teste;
- **JMuBEN/JMuBEN2:** treino auxiliar de classificação, com folhas de campo;
- **BRACOT:** detecção e segmentação das folhas.

Cada fonte tem um manifest, que diz o que cada imagem é e como ela é usada, e um relatório de
integridade; o BRACOL tem também a EDA. As imagens não entram no git; os manifests, os
relatórios e as figuras, sim.

## Decisões do gestor (09/10/2026)

- **Mudança de requisito:** o app vai receber a foto de uma planta inteira, de um galho ou de uma
  única folha, em qualquer situação (luz, fundo, ângulo). O sistema acha cada folha na foto e
  diagnostica cada uma (classe e severidade).
- **Desenho da Frente 8, atualizado:**
  - o BRACOL (folhas isoladas, 2048x1024) continua sustentando a classificação, a severidade e o
    teste do RNF01, e é a única fonte de teste de classificação e de severidade;
  - o JMuBEN/JMuBEN2 entra só no treino de classificação, para trazer folhas fotografadas em
    campo. Nunca entra em validação nem em teste. Fica sem duplicatas, com teto por classe e com
    a fonte marcada no manifest;
  - o BRACOT é usado para achar e contornar as folhas na foto da planta. A divisão dos autores
    (240 treino / 60 teste) é a oficial;
  - um teste de campo com rótulo de classe ainda não existe. Quando vier, entra como fonte nova,
    com manifest próprio.
- **RD02 (desbalanceamento): compensar.**
  - Amostragem balanceada por classe no treino e augmentation moderada (Albumentations) só no
    treino, em memória.
  - Validação e teste intactos.
  - Métricas por classe (recall, F1 macro), além da acurácia.
  - Os parâmetros finos ficam com a Frente 9, que os compara na prática.
- **Entrada do JMuBEN no modelo:** só se melhorar a validação do BRACOL sem piorar o recall de
  saudável. A Frente 9 compara o treino com e sem JMuBEN, e com tetos menores, que o leitor
  aceita como parâmetro (`jmuben.ler_manifest(teto=...)`).
- **Validação do detector:** separa cenas inteiras do treino dos autores, nunca imagens soltas.
  O teste dos autores fica intacto.

## Fontes

| fonte | papel | situação |
|---|---|---|
| BRACOL, cópia completa (`data/raw/bracol/bracol_completo/`) | classificação e severidade: treino, validação e teste (RNF01) | em uso desde 07/10/2026 |
| BRACOL, cópia parcial (`data/raw/bracol/bracol_recuperado/`) | nenhum | obsoleta: mantida, sem uso |
| JMuBEN e JMuBEN2 (`data/raw/jmuben/`) | só treino auxiliar de classificação, nunca validação nem teste | em uso desde 09/10/2026 |
| BRACOT (`data/raw/bracot/bracot-data/`) | detecção e segmentação de folhas (RF09), com a divisão dos autores | em uso desde 09/10/2026 |

## Procedência do BRACOL

- **Dados originais:** Krohling, Esgario e Ventura (2019), Mendeley Data, DOI
  [10.17632/yy2k5y8mxg.1](https://doi.org/10.17632/yy2k5y8mxg.1), versão 1. A licença é a do
  registro no Mendeley: CC BY 4.0, que exige citar os autores.
- **Cópia em uso:** `coffee-datasets.zip`, do Google Drive linkado no README de
  [github.com/esgario/lara2018](https://github.com/esgario/lara2018), baixado em 07/10/2026.
  - 278.741.231 bytes, SHA-256 `e6fc32d504ae33e667475dec4c4aef710aac65ad12f16a401c706445c46c9f89`.
  - 4.959 arquivos, todos dentro da pasta `coffee-datasets/`.
  - Este README não atribui licença própria a essa cópia: a referência é o registro do Mendeley.
- **Cópia parcial (histórico):** o zip publicado no Mendeley
  (`BRACOL_coffee_leaf_ images_datasets.zip`, 164.516.964 bytes, SHA-256
  `25a2fc973137f92caa6700b95e645a0eceeb041c3cfc1eaba853f3e1b5fdfc1f`, baixado em 07/10/2026)
  está truncado.
  - Não tem índice central, guarda as entradas em ordem alfabética do nome e termina no meio de
    `688.jpg`.
  - Dele saíram 1.401 das 1.747 imagens de folha; faltavam os ids 7-9, 69-99 e 688-999. Elas
    foram recuperadas com `data/tools/recuperar_zip.py` em `data/raw/bracol/bracol_recuperado/`.
  - Essa pasta fica obsoleta, mantida e sem uso. Usada como entrada, hoje dá erro (346 ausentes
    fora da lista esperada, que agora é vazia).
- **Conferência da cópia completa (07/10/2026):**
  - o `leaf/dataset.csv` é idêntico ao da cópia parcial (SHA-256 `e74bb83e681812c225c9b733720b88134819710c9b8acf213c0b25ea904d4a93`);
  - as 1.401 imagens em comum têm o mesmo SHA-256;
  - as 1.747 imagens de folha abrem e decodificam por completo.

## O que há na cópia completa

```
data/raw/bracol/
  bracol_completo/coffee-datasets/            <- entrada padrão
    leaf/                  usado: dataset.csv (1.747 linhas), legend.txt, images/ (1.747 JPEG 2048x1024)
    segmentation/          sem uso: images/ e annotations/ (train 400, val 50, test 50), annotations-info.txt
    symptom/               sem uso: train/, val/ e test/, com 1_health ... 5_cercospora (2.209 JPEG)
  bracol_recuperado/coffee-datasets/coffee-datasets/leaf/   obsoleta (cópia parcial)
```

- **`leaf/`:** é o que o projeto usa. O `legend.txt` define `predominant_stress` como 0 healthy,
  1 miner, 2 rust, 3 phoma, 4 cercospora e 5 undetermined. As colunas `miner`, `rust`, `phoma` e
  `cercospora` dizem se a folha tem cada estresse, e `severity` vai de 0 a 4.
- **`segmentation/`:** 500 imagens e 500 anotações. O `annotations-info.txt` define as cores
  background `[0,0,0]`, leaf `[0,176,0]` e symptom `[255,0,0]`. Só documentado, sem uso.
- **`symptom/`:** 2.209 recortes de sintomas. Só documentado, sem uso. Cautela: segundo o README
  do repositório dos autores, consultado em 07/10/2026, esse conjunto combina imagens do BRACOL
  com imagens do Embrapa Digipathos. Antes de usar, é preciso checar a licença.
- **Versão do GitHub dos autores:** segundo o repositório dos autores, consultado em 07/10/2026, a
  pasta `leaf` de github.com/esgario/lara2018 tem 1.747 imagens de 512x256, recortadas. Não são os
  originais de 2048x1024 usados aqui, e as duas versões não devem ser misturadas.

## Classes e severidade

| `predominant_stress` | legend.txt | classe do projeto | índice em `CLASSES` |
|---:|---|---|---:|
| 0 | healthy | `saudavel` | 0 |
| 1 | miner | `bicho_mineiro` | 2 |
| 2 | rust | `ferrugem` | 1 |
| 3 | phoma | `phoma` | 3 |
| 4 | cercospora | `cercosporiose` | 4 |
| 5 | undetermined | excluída | - |

- O índice de uma classe para o modelo é `CLASSES.index(classe)`, nunca o código
  `predominant_stress`: com o código, bicho-mineiro e ferrugem se trocariam.
- **Classe 5** ("undetermined", 62 linhas): fica fora de classificação, severidade e
  multirrótulo. Segundo o repositório dos autores, consultado em 07/10/2026, o `dataset.csv` dos
  experimentos deles também não tem essas linhas.
- A correspondência das colunas `phoma` e `cercospora` com as classes do artigo ainda não está
  confirmada, e o mesmo vale para as pastas `Phoma` e `Cerscospora` do JMuBEN. A ressalva fica
  só em `RESSALVA_PHOMA_CERCOSPORA`, em `data/bracol.py`.
- **Severidade (`severity`):** 0 saudável (<0,1% da área), 1 muito baixa (0,1-5%), 2 baixa
  (5-10%), 3 alta (10-15%) e 4 muito alta (>15%).

## Comparabilidade com Esgario et al. (2020)

Segundo o repositório dos autores, consultado em 07/10/2026, o `dataset.csv` de lá tem 1.685
linhas, sem a classe 5: são as mesmas imagens elegíveis daqui. A divisão deles, segundo o mesmo
repositório, é:
- aleatória, com seed 150;
- 70/15/15, com rotação em 5 dobras, sem estratificar;
- com amostragem balanceada por classe no treino;
- com entrada de 224x224.

A nossa é estratificada por classe (seed 42) e mantém juntas as folhas repetidas. Os números do
artigo e os deste projeto não são diretamente comparáveis: compare sempre com essa ressalva.

## Manifest: `data/manifests/bracol.csv`

É a fonte única da verdade dos dados, com uma linha por id do `dataset.csv` (1.747 linhas). Não
editar à mão: ele é gerado por `python data/organize_dataset.py`.

| coluna | conteúdo |
|---|---|
| `fonte` | `bracol` |
| `id` | id do `dataset.csv`; a imagem é `<id>.jpg` |
| `caminho` | caminho POSIX relativo à raiz do repositório |
| `presente` | 1 se a imagem existe |
| `classe` | classe do projeto (vazia na classe 5) |
| `predominant_stress`, `miner`, `rust`, `phoma`, `cercospora`, `severity` | valores originais do `dataset.csv` |
| `largura`, `altura` | tamanho em pixels |
| `sha256` | hash exato do arquivo |
| `phash`, `phash_folha` | pHash de 256 bits do quadro inteiro e do recorte da folha |
| `grupo` | `bracol-<menor id do grupo>`; o grupo nunca se divide entre splits |
| `split` | `treino`, `val` ou `teste`, ou vazio se a linha estiver excluída |
| `excluida`, `motivo_exclusao` | 1 e o motivo (`predominant_stress_5` ou `imagem_ausente`) |

**Leitura (Frente 9):**
```python
import sys
sys.path.insert(0, "data")  # a partir da raiz do repositório
import bracol

treino = bracol.ler_manifest(split="treino")       # só linhas elegíveis
rotulos = treino["classe"].map(bracol.CLASSES.index)
```

**Folhas repetidas e duplicatas.** Uma imagem entra no grupo de outra, de forma transitiva, se:
- o SHA-256 for igual;
- o pHash do quadro ficar a até 32 bits;
- o pHash do recorte da folha ficar a até 46 bits;
- ou o par estiver em `PARES_MESMA_FOLHA` (`data/bracol.py`).

Os 8 grupos atuais são:
- 813 = 1022 (arquivos idênticos);
- 758/759, 1352/1356, 1692/1694, 760/764, 469/471 e 1295/1296;
- 1715/1722 (classe 5).

Todos foram conferidos visualmente em 07/10/2026. Os pares 953/959 e 986/987 (folhas escuras parecidas) foram conferidos visualmente em 07/10/2026, em cópias reduzidas: pareceram folhas diferentes e ficam fora dos grupos; continuam listados no relatório para nova conferência nos originais. A calibração dos dois limiares está comentada em `data/bracol.py`.

**Divisão.** Proporção 70/15/15, estratificada por classe e espalhada pelos níveis de severidade,
com seed 42.

| classe | treino | val | teste |
|---|---:|---:|---:|
| saudavel | 190 | 41 | 41 |
| ferrugem | 372 | 79 | 80 |
| bicho_mineiro | 271 | 58 | 58 |
| phoma | 244 | 52 | 52 |
| cercosporiose | 103 | 22 | 22 |
| **total** | 1.180 | 252 | 253 |

- **Refeita do zero em 07/10/2026**, ao adotar a cópia completa: nenhum modelo tinha sido
  treinado, e manter a divisão anterior deixaria partido o par 469/471.
- **Estável daí em diante:** ao rodar de novo, quem já tem split fica onde está, e só imagens
  novas são distribuídas.
- **Para recomeçar:** só com `--refazer-divisao` e decisão explícita, registrada em
  `HISTORICO_DIVISAO` (`data/bracol.py`).
- **Teste:** só BRACOL, nunca aumentado. Augmentation (Albumentations) só no treino, em memória.

**Decisão: sem cópia das imagens em `data/processed/`.**
- O texto original da Frente 8 previa copiar as imagens para
  `data/processed/{train,val,test}/{classe}/`.
- Isso foi substituído pelo manifest e pela leitura direta de `data/raw/` (decisão de
  07/10/2026). Assim não se duplicam cerca de 210 MB de imagens, e a divisão fica num arquivo
  pequeno e versionado.
- `data/processed/` continua ignorada pelo git, reservada para derivados pesados (por exemplo, um
  cache de imagens reduzidas), se a Frente 9 precisar.

## JMuBEN e JMuBEN2 (treino auxiliar)

**Citação obrigatória.** Quem usar estas imagens precisa citar os autores:
- **JMuBEN:** Jepkoech, Kenduiywo, Mugo e Chebet (2021), Mendeley Data, DOI
  [10.17632/t2r6rszp5c.1](https://doi.org/10.17632/t2r6rszp5c.1), versão 1, publicado em
  26/03/2021. Pastas `Cerscospora`, `Leaf rust` e `Phoma`.
- **JMuBEN2:** [primeiro autor sem nome na página; provavelmente Jepkoech, a confirmar], Mugo,
  Kenduiywo e Chebet (2021), Mendeley Data, DOI
  [10.17632/tgv3zb82nd.1](https://doi.org/10.17632/tgv3zb82nd.1), versão 1, publicado em
  26/03/2021. Pastas `Healthy` e `Miner`.
- **Artigo:** Jepkoech, Mugo, Kenduiywo e Too, *Data in Brief* 36 (2021) 107142.

**Licença:** não exibida na página consultada em 09/10/2026; confirmar.

**Cópia local:**
- O material foi recebido pela equipe em cinco pastas exportadas (`<Classe>-20210326T…Z-001`) e
  copiado em 09/10/2026 para `data/raw/jmuben/<Pasta>/`.
- `Cerscospora` é a grafia da origem.
- O JMuBEN publicado tem 22.591 imagens; a cópia local tem 22.588. A diferença de 3 arquivos está
  registrada, sem investigar.

| pasta | classe | subconjunto | arquivos | conteúdos distintos | grupos | teto | selecionadas |
|---|---|---|---:|---:|---:|---:|---:|
| `Cerscospora` | cercosporiose | jmuben | 7.681 | 322 | 80 | 51 | 51 |
| `Healthy` | saudavel | jmuben2 | 18.984 | 63 | 12 | 95 | 12 |
| `Leaf rust` | ferrugem | jmuben | 8.336 | 1.042 | 406 | 186 | 186 |
| `Miner` | bicho_mineiro | jmuben2 | 16.978 | 1.639 | 309 | 135 | 135 |
| `Phoma` | phoma | jmuben | 6.571 | 691 | 177 | 122 | 122 |
| **total** | | | 58.550 | 3.757 | 984 | | 506 |

**Limitações** (detalhes em `data/reports/integridade_jmuben.md`):
- **Recortes, não folhas inteiras:**
  - quase todos têm 128x128;
  - 2.953 são retangulares, com lado maior de 256, em `Cerscospora` e `Miner`.

  Segundo a descrição dos autores, são café arábica fotografado em campo, recortado para a região
  de interesse, com augmentation em parte das imagens. As imagens são do Quênia.
- **Augmentation e cópias:**
  - 58.550 arquivos, mas só 3.757 conteúdos distintos (1.357 matrizes de pixels distintas) e 984
    recortes (grupos);
  - parte da "rotação" está só na etiqueta EXIF de orientação. O Pillow ignora a etiqueta e o
    `cv2.imread` a aplica.
- **Saudável quase sem diversidade:**
  - os 18.984 arquivos de `Healthy` viram 12 grupos, que a olho são 2 regiões de folha;
  - o treino auxiliar traz 494 recortes de doença e só 12 de folha saudável;
  - o risco é o modelo associar "cara de recorte de campo" a doença. É para isso a regra de
    entrada no modelo (seção "Decisões do gestor").
- **Rótulos não verificados:**
  - `Leaf rust` é desbotado: saturação mediana de 40 (escala 0–255), contra 102 a 145 nas outras
    pastas;
  - `Phoma` tem imagens com manchas escuras grandes;
  - a correspondência de `Phoma` e `Cerscospora` com as classes do projeto não está confirmada.
- **Os 144 `.jpeg` de `Leaf rust`:** formam 9 recortes próprios, de 16 arquivos cada, com o mesmo
  aspecto desbotado do resto da pasta. Entram como os outros.
- **Prefixo N dos nomes (`N (k).jpg`):** não identifica a foto de origem (cópias do mesmo recorte
  aparecem com N diferentes) e não é usado.
- **Arquivo ilegível:** `Healthy/2 (691).jpg` fica excluído com motivo. O Pillow não o
  reconhece. Ele tem o mesmo tamanho de `2 (691).bak.jpg`, que abre e tem 293 cópias idênticas.

**Manifest: `data/manifests/jmuben.csv`.** Tem uma linha por conteúdo distinto (SHA-256), ou seja,
3.757 linhas. As 54.793 cópias idênticas só aparecem na contagem `copias`. Não editar à mão: ele é
gerado por `python data/jmuben.py`. **Não tem coluna de split.**

| coluna | conteúdo |
|---|---|
| `fonte` | `jmuben` |
| `subconjunto` | `jmuben` (`Cerscospora`, `Leaf rust`, `Phoma`) ou `jmuben2` (`Healthy`, `Miner`) |
| `id` | 1 a 3.757, na ordem do caminho |
| `caminho` | a primeira cópia do conteúdo, em ordem de nome |
| `copias` | número de arquivos idênticos (mesmo SHA-256) |
| `classe` | classe do projeto |
| `largura`, `altura` | como gravadas, sem aplicar a orientação EXIF |
| `sha256`, `phash_canonico` | hash exato e hash canônico (abaixo) |
| `grupo` | `jmuben-<menor id do grupo>` (vazio no ilegível) |
| `uso` | `treino_auxiliar` ou `excluida` |
| `selecionada`, `motivo` | 1 e motivo vazio; ou 0 com `grupo_ja_representado`, `acima_do_limite_da_classe` ou `arquivo_ilegivel` |

**Hash canônico e grupos.**
- **Hash canônico:** o menor pHash de 256 bits entre as 8 transformações de giro e espelhamento
  do cinza equalizado. Por isso não muda com giro, espelhamento ou brilho.
- **Grupo:** um conteúdo entra no grupo de outro, de forma transitiva, se o SHA-256 for igual ou
  se o hash canônico ficar a até 64 bits.
- **Calibração** (09/10/2026; detalhes em `data/jmuben.py`):
  - variantes do mesmo recorte foram conferidas a olho de 2 a 88 bits;
  - recortes diferentes aparecem a partir de 94 bits, e o par de classes diferentes mais próximo
    fica a 86;
  - não há lacuna limpa. Com 64 bits saem 984 grupos, nenhum com duas classes.
- **Por que o erro não é grave:** errar aqui só muda a diversidade do treino auxiliar, nunca vaza
  para validação ou teste.

**Seleção.**
- **Teto por classe** (`CAP_POR_CLASSE`): metade do treino do BRACOL na classe (190, 372, 271,
  244 e 103 imagens).
- **Um representante por grupo, sem repetir.** Os membros de um grupo são giros, espelhos e
  mudanças de brilho do mesmo recorte, que o augmentation já produz.
- **Sorteio determinístico (seed 42):** a ordem dos grupos sai do SHA-256 de `"42:grupo"` e o
  representante, do de `"42:id"`.
- **Resultado:** 506 imagens, o que dá 1/3 do treino de cada classe doente e 6% do de saudável.
- **Hash da seleção:** `6b206f932ee7c561ac1ac82749e20e0d612c81bab882a948c82f001ed4ddb1a5`.

**Leitura (Frente 9):**
```python
import jmuben                             # com data/ no sys.path, como no exemplo do BRACOL

auxiliar = jmuben.ler_manifest()          # só as 506 selecionadas; não há parâmetro de split
menor = jmuben.ler_manifest(teto=50)      # no máximo 50 por classe (subconjunto do anterior)
sem_saudavel = jmuben.ler_manifest(teto={"saudavel": 0})
```
O teto do leitor só reduz a seleção, e um teto menor dá sempre um subconjunto do maior. Para
passar de `CAP_POR_CLASSE` é preciso regerar o manifest, por decisão do gestor.

## BRACOT (detecção e segmentação de folhas)

**Citação obrigatória.** Krohling, Tozzi de Souza e Tassis (2021), Mendeley Data, DOI
[10.17632/pmkbyjpf6k.1](https://doi.org/10.17632/pmkbyjpf6k.1), versão 1, publicado em
08/01/2021, material complementar ao BRACOL. Citar o dataset.

**Licença:** não exibida na página consultada em 09/10/2026; confirmar.

**Cópia local:** `data/raw/bracot/bracot-data/`, copiada em 09/10/2026.

```
data/raw/bracot/bracot-data/
  train/   240 JPEG 4032x3024, train_annotation_coco.json (1.318 folhas), via_export_json.json
  test/     60 JPEG 4032x3024, test_annotation_coco.json (344 folhas), via_export_json.json
```

| split | fotos | folhas | folhas por foto (mín / mediana / máx) | área coberta (mediana; mín–máx) | 31/08/2019 | 08/12/2019 |
|---|---:|---:|---|---|---:|---:|
| treino | 240 | 1.318 | 2 / 5 / 12 | 34,5% (9,7–67,3%) | 109 | 131 |
| teste | 60 | 344 | 2 / 6 / 9 | 31,9% (15,0–57,7%) | 15 | 45 |

**Anotações** (detalhes em `data/reports/integridade_bracot.md`):
- **Formato:** polígonos COCO com uma única categoria, `leaf` (id 0). O export do VIA tem as
  mesmas regiões, sem atributos: não há classe de estresse por folha.
- **Campo `area`:** é a área do bbox nas 1.662 anotações, efeito do export do VIA. Não usar.
- **Borda:** 28 polígonos passam 3 ou 4 px da borda da foto. Quem lê recorta na borda.

**Limitações:**
- **Só parte das folhas de cada foto está contornada.**
  - As folhas anotadas cobrem cerca de 1/3 da foto, que é quase toda folhagem.
  - Elas são mais claras que o resto da foto em 287 de 300 fotos (brilho V mediano +42, em escala
    0–255, medido em 09/10/2026).
  - Folha não contornada conta como fundo. Um detector treinado aqui aprende esse tipo de folha e
    pode ignorar as outras; nas métricas, acertar uma folha não anotada conta como erro.
- **Não há classe de estresse por folha.**
- **São só duas sessões de fotos,** de um dia cada (31/08/2019 e 08/12/2019).

**Manifest: `data/manifests/bracot.csv`.** Tem uma linha por foto (300 linhas). Não editar à mão:
ele é gerado por `python data/bracot.py`. As anotações ficam em `data/raw/` e não são copiadas nem
convertidas.

| coluna | conteúdo |
|---|---|
| `fonte` | `bracot` |
| `id` | o nome da foto sem extensão (`AAAAMMDD_HHMMSS`, a hora do celular) |
| `caminho` | caminho POSIX relativo à raiz do repositório |
| `largura`, `altura` | tamanho em pixels |
| `split` | `treino` ou `teste`: a divisão dos autores, oficial |
| `n_folhas` | folhas anotadas na foto |
| `area_coberta` | a soma das áreas dos polígonos (fórmula do laço) dividida pela área da foto. Folhas que se tocam contam duas vezes; a diferença em relação à união é de no máximo 0,09 ponto percentual |
| `data_hora` | data e hora tiradas do nome |
| `cena` | o id da primeira foto da cena (abaixo) |
| `sha256`, `phash` | hash exato e pHash de 256 bits da foto inteira |
| `anotacao` | o arquivo COCO de origem, em `data/raw/` |

**Cenas e vazamento treino→teste.**
- **Cena:** junta as fotos tiradas em sequência, a até 10 s da anterior, em ordem de hora e sem
  olhar o split. São 90 cenas, com mediana de 2 fotos e no máximo 22.
  - O intervalo mediano entre fotos é 7 s.
  - 10 s é o menor intervalo inteiro que deixa numa cena só cada par de fotos com as mesmas folhas
    (abaixo).
- **Pelo tempo:**
  - as 60 fotos de teste ficam a no máximo 21 s de uma foto de treino;
  - 27 cenas misturam treino e teste, com 55 das 60 fotos de teste.
- **Pelo conteúdo:** medido por pontos casados (ORB + RANSAC, nos 8.540 pares a até 300 s) e
  conferido a olho em 09/10/2026.
  - Fotos seguidas, mesmo a 2 s de distância, costumam mostrar folhas diferentes.
  - Só 2 fotos de teste repetem folhas de uma foto de treino, listadas em
    `bracot.TESTE_SOBREPOSTO`:
    - `20190831_164920` é quase a mesma foto que a `164921` de treino;
    - `20190831_165013` mostra o mesmo galho que a `164937` de treino, de outro ângulo.
  - Reporte o detector no teste inteiro e sem essas 2 fotos.
- **Divisão:** a oficial é a dos autores, com hash
  `890c3fb2c5065fedd3a5b3807cbcd082070dcfd24d75b579b263a352853c0317`.
- **Divisão por cena, só simulada** (cenas de 10 s, estratificada por dia, 80/20, seed 42, com
  `bracol.dividir`). Ficou só documentada, por decisão do gestor:
  - treino 241 e teste 59 (31/08: 99/25; 08/12: 142/34), com 29 cenas no teste e 1.339/323 folhas;
  - 87 fotos mudariam de split;
  - hash `fc3220b592bfa234c05d0c9d5f04fead6b1ef0a3cb967253f731cbd82f2718af`.
- **Validação do detector:** para ajustar hiperparâmetros ou parar o treino, separe cenas
  inteiras do treino dos autores, nunca fotos soltas. O grupo é a coluna `cena`, por exemplo com
  `bracol.dividir` estratificando por dia, ou com `GroupShuffleSplit` do scikit-learn.

**Leitura (detector):**
```python
import bracot                             # com data/ no sys.path, como no exemplo do BRACOL

treino = bracot.ler_manifest(split="treino")       # 240 fotos, com a coluna cena
coco_treino = bracot.caminho_coco("treino")        # data/raw/.../train_annotation_coco.json
sobrepostas = [teste for teste, _ in bracot.TESTE_SOBREPOSTO]
```

## Interface para a Frente 9

- **Classificação e severidade:**
  - o treino junta o BRACOL e o JMuBEN selecionado;
  - validação e teste são só do BRACOL;
  - o rótulo é `CLASSES.index(classe)`.

  ```python
  import sys
  import pandas as pd
  sys.path.insert(0, "data")  # a partir da raiz do repositório
  import bracol, jmuben

  colunas = ["fonte", "id", "caminho", "classe"]
  treino = pd.concat([bracol.ler_manifest(split="treino")[colunas],
                      jmuben.ler_manifest()[colunas]], ignore_index=True)
  val = bracol.ler_manifest(split="val")
  teste = bracol.ler_manifest(split="teste")
  ```

  - **Severidade:** vem só das linhas do BRACOL. As do JMuBEN não têm alvo de severidade, e a
    perda dessa saída é mascarada.
  - **Amostragem balanceada do RD02:** os pesos são calculados sobre o treino combinado.
  - **Regra de entrada:** o JMuBEN só fica no modelo se melhorar a validação do BRACOL sem piorar
    o recall de saudável.
- **Detector:**
  - lê `bracot.ler_manifest()` e os polígonos COCO de `bracot.caminho_coco(split)`;
  - valida com cenas inteiras do treino;
  - reporta o teste com e sem `TESTE_SOBREPOSTO`.

## Comandos (na raiz do repositório, com o venv ativo)

| comando | o que faz |
|---|---|
| `python data/organize_dataset.py` | gera o manifest e `data/reports/integridade_bracol.md` |
| `python data/organize_dataset.py --verificar` | confere dados x manifest sem alterar o manifest (código 0 se tudo bate) |
| `python data/organize_dataset.py --entrada PASTA` | usa outra cópia do BRACOL (dentro do repositório) |
| `python data/eda_bracol.py` | gera `data/reports/eda_bracol.md` e as figuras em `data/reports/figures/` |
| `python data/jmuben.py` | gera o manifest do JMuBEN e `data/reports/integridade_jmuben.md` (cerca de 1 min) |
| `python data/jmuben.py --verificar` | confere dados x manifest do JMuBEN sem alterar o manifest |
| `python data/bracot.py` | gera o manifest do BRACOT e `data/reports/integridade_bracot.md` (cerca de 30 s) |
| `python data/bracot.py --verificar` | confere dados x manifest do BRACOT sem alterar o manifest |
| `jupyter nbconvert --clear-output --inplace notebooks/01_eda.ipynb` | limpa as saídas do notebook antes de commitar |
| `python -m pytest` | roda os testes; os do manifest versionado não precisam das imagens |

**Figuras.** Elas podem diferir byte a byte entre sistemas, versões de bibliotecas e fontes
instaladas, mesmo com os mesmos dados. Só recommite uma figura quando os números mudarem, ou seja,
quando o manifest ou o `eda_bracol.md` mudarem.

## Ferramentas (`data/tools/`)

Usadas para recuperar a cópia parcial. Elas ficam como registro e para qualquer zip danificado.
Só usam a biblioteca padrão do Python.

- **Diagnóstico** (só leitura): mostra o tamanho, o SHA-256, a estrutura e os arquivos com erro.
  ```
  python data/tools/diagnosticar_zip.py "C:\caminho\arquivo.zip"
  ```
- **Recuperação:** lê o zip entrada por entrada, sem depender do índice, confere o CRC-32 de cada
  arquivo e grava só os íntegros. A pasta de destino precisa ser nova ou estar vazia.
  ```
  python data/tools/recuperar_zip.py "C:\caminho\arquivo.zip" --extrair data/raw/bracol/PASTA_NOVA
  ```
- As duas gravam um relatório `relatorio_*.txt` ao lado do script, que o git ignora, e esperam um
  Enter no final.
- O caminho do zip vai entre aspas: o nome do arquivo do Mendeley tem um espaço.

## Reproduzir do zero

No PowerShell, a partir da pasta onde o repositório vai ficar:

```
git clone https://github.com/coffea-arabica-ml/coffea-ml.git
cd coffea-ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip list
```

Confira no `pip list` que aparece `opencv-python-headless` e **não** aparece `opencv-python`.
Depois:
1. Baixe o `coffee-datasets.zip` pelo link do README de github.com/esgario/lara2018 e salve na
   raiz do repositório (o `.gitignore` já ignora `*.zip`).
2. Confira o SHA-256:
   ```
   Get-FileHash .\coffee-datasets.zip -Algorithm SHA256
   ```
   O resultado tem de ser `e6fc32d504ae33e667475dec4c4aef710aac65ad12f16a401c706445c46c9f89`.
3. Extraia o zip e rode a verificação, a EDA e os testes:

```
Expand-Archive -Path .\coffee-datasets.zip -DestinationPath data\raw\bracol\bracol_completo
python data/organize_dataset.py --verificar
python data/eda_bracol.py
python -m pytest
```

O `--verificar` tem de terminar com "Manifest confere com os dados" e código 0. Não regenere o
manifest para "reproduzir": a divisão versionada é a que vale.

**JMuBEN e BRACOT.**
1. Copie as fontes para as mesmas pastas da cópia local:
   - JMuBEN: as cinco pastas de classe em `data/raw/jmuben/<Pasta>/`;
   - BRACOT: `train/` e `test/` em `data/raw/bracot/bracot-data/`.
2. Rode:

```
python data/jmuben.py --verificar
python data/bracot.py --verificar
```

Cuidado com o JMuBEN: a cópia local tem 3 arquivos a menos que a publicada no Mendeley. Uma cópia
baixada de lá pode não bater com o manifest versionado; nesse caso, o `--verificar` aponta as
diferenças. O manifest versionado vale como registro da cópia usada.
