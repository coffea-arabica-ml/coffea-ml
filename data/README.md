# Dados do coffea-ml (Frente 8)

Imagens de folhas de café do BRACOL, o manifest que diz o que cada imagem é e em que split
está, o relatório de integridade e a EDA. As imagens não entram no git; o manifest, os
relatórios e as figuras, sim.

## Fontes

| fonte | papel | situação |
|---|---|---|
| BRACOL, cópia completa (`data/raw/bracol/bracol_completo/`) | treino, validação e teste | em uso desde 07/10/2026 |
| BRACOL, cópia parcial (`data/raw/bracol/bracol_recuperado/`) | nenhum | obsoleta: mantida, sem uso |
| JMuBEN e JMuBEN2 | só treino auxiliar, nunca teste | prevista, não baixada |
| BRACOT | detecção de folhas na foto do pé (RF09) | prevista, não baixada |

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
  confirmada. A ressalva fica só em `RESSALVA_PHOMA_CERCOSPORA`, em `data/bracol.py`.
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

## Comandos (na raiz do repositório, com o venv ativo)

| comando | o que faz |
|---|---|
| `python data/organize_dataset.py` | gera o manifest e `data/reports/integridade_bracol.md` |
| `python data/organize_dataset.py --verificar` | confere dados x manifest sem alterar o manifest (código 0 se tudo bate) |
| `python data/organize_dataset.py --entrada PASTA` | usa outra cópia do BRACOL (dentro do repositório) |
| `python data/eda_bracol.py` | gera `data/reports/eda_bracol.md` e as figuras em `data/reports/figures/` |
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
