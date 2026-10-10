# coffea-ml (Cafélens)

Parte de dados e ML do projeto acadêmico Cafélens (UNIFRAN, prazo 06/11/2026): classificar o
estresse biótico de folhas de café (saudavel, ferrugem, bicho_mineiro, phoma, cercosporiose) e
a severidade, por aprendizado por transferência. Pastas: `data/` (Frente 8), `model/` e
`evaluation/` (Frente 9), `notebooks/`, `tests/`.

## Regras de trabalho

- Não fazer commit nem push: o usuário commita pelo GitHub Desktop.
- Trabalhar um passo por vez. Ao fim de cada passo, mostrar a saída das validações, sugerir a
  mensagem de commit (português, sem acento, verbo no presente: "Adiciona ...", "Corrige ...")
  e parar até o usuário mandar seguir.
- Nunca apagar nem modificar `data/raw/`. Imagens do dataset nunca entram no git, exceto o
  mosaico derivado da EDA em `data/reports/figures/` e as pranchas do Grad-CAM em
  `model/runs/<run>/atalho/` (decisão do gestor, 09/10/2026), em JPEG e com crédito CC BY 4.0.
- Português do Brasil em textos, comentários e mensagens. A saída de terminal dos scripts fica
  sem acento, setas, emojis ou outros símbolos especiais: no Windows, a saída redirecionada sai
  em cp1252 e os acentos viram lixo. Arquivos gravados (relatórios, README) são UTF-8 com acento.
- Python 3.12+ e `pathlib`; caminhos resolvidos a partir de `__file__`, para os scripts rodarem
  de qualquer pasta. Nada que dependa de um shell específico.
- Sem DVC, Hydra ou frameworks de pipeline. Dependência nova: perguntar antes.
- Notebooks são finos (chamam funções dos scripts, sem duplicar lógica) e entram no git com as
  saídas limpas.
- O usuário é o gestor e o único tomador de decisões do projeto (o professor não acompanha).
  Mudança de escopo ou de requisito: sinalizar e esperar a decisão do usuário, registrada como
  "decisão do gestor", com a data. Não pedir nem sugerir aprovação de terceiros.
- Todo o histórico passa pelo git (RI03): manifest, relatórios e figuras são versionados.
- Não baixar dados da internet sem pedido explícito.

## Dados (detalhes em `data/README.md`)

- O BRACOL é a cópia completa (`data/raw/bracol/bracol_completo/`, recebida dos autores em
  07/10/2026): 1.747 imagens. A cópia parcial em `data/raw/bracol/bracol_recuperado/` é
  obsoleta: não apagar e não usar.
- Sem a classe 5 são as mesmas 1.685 imagens que os autores usaram, mas a divisão é outra
  (seed 42): comparar com Esgario et al. (2020) só com essa ressalva.
- `data/manifests/bracol.csv` é a fonte única da verdade (uma linha por id do csv original).
  Não editar à mão: regenerar com `data/organize_dataset.py`.
- `predominant_stress = 5` é "undetermined" (leaf/legend.txt): fica excluída de classificação,
  severidade e multirrótulo. Nunca atribuir classe por palpite.
- Duplicatas e folhas fotografadas de novo ficam no mesmo grupo, e um grupo nunca se divide
  entre splits: SHA-256 igual, pHash do quadro a até 32 bits, pHash da folha a até 46 bits ou
  `PARES_MESMA_FOLHA` (8 pares conferidos visualmente em 07/10/2026).
- A correspondência de phoma/cercosporiose com as fontes (colunas do BRACOL e o artigo, pastas
  `Phoma`/`Cerscospora` do JMuBEN) não está confirmada; a ressalva fica só em
  `RESSALVA_PHOMA_CERCOSPORA` (`data/bracol.py`).
- A ordem de `CLASSES` segue o RF01. O índice de uma classe é `CLASSES.index(nome)`, nunca o
  código `predominant_stress`.
- A divisão treino/val/teste é estável: nenhuma imagem muda de split sem `--refazer-divisao` e
  decisão explícita do usuário. Ela foi refeita do zero uma vez, em 07/10/2026, ao adotar a
  cópia completa (motivo em `HISTORICO_DIVISAO`, `data/bracol.py`).
- Teste: só BRACOL, nunca aumentado. Augmentation (Albumentations) só no treino, em memória.
- Fontes de campo (decisão do gestor, 09/10/2026):
  - JMuBEN/JMuBEN2, e qualquer fonte auxiliar: só treino, nunca validação nem teste. Sem
    duplicatas (um representante por grupo) e com limite de imagens por classe. A fonte fica
    marcada em toda linha do manifest, que não tem coluna de split. JMuBEN:
    `data/manifests/jmuben.csv`, gerado por `data/jmuben.py`, uma linha por conteúdo distinto;
    grupos pelo hash canônico a até 64 bits; teto em `CAP_POR_CLASSE`.
  - BRACOT: detecção e segmentação de folhas (RF09). A divisão dos autores (240 treino / 60
    teste) é a oficial. Não há classe de estresse por folha. Manifest
    `data/manifests/bracot.csv`, gerado por `data/bracot.py`, uma linha por foto; cenas de
    10 s. A validação do detector separa cenas inteiras do treino, nunca imagens soltas, e as
    métricas saem com e sem as 2 fotos de `TESTE_SOBREPOSTO`.
  - Teste de classificação e de severidade (RNF01): só BRACOL. Um teste de campo com rótulo de
    classe, se vier, entra como fonte nova, com manifest próprio.

## Detector de folhas (RF09)

- YOLO11s-seg (ultralytics 8.4.175) ajustado no BRACOT: `model/treinar_detector.py` treina e
  avalia; `model/detector.py` tem `detectar_folhas`, a interface do backend. Licença AGPL-3.0
  aceita para este projeto acadêmico de repositórios públicos (decisão do gestor, 10/10/2026).
- Instalação (decisão do gestor, 10/10/2026): OpenCV só headless; o ultralytics entra com
  `--no-deps`, e as dependências que faltavam ficam fixadas no `requirements.txt`. O `pip check`
  acusa o `opencv-python` que o ultralytics pede: é esperado (ver README.md).
- Rede: o único download autorizado é o peso `yolo11s-seg.pt` da release oficial (SHA-256 em
  `detector.py`, conferido ao treinar). `detector.importar_ultralytics()` impõe o modo offline e
  troca a pré-carga da fonte (que baixava a Arial.ttf); o treino roda sem AMP e sem gráficos.
- O teste do BRACOT (60 fotos) fica fechado até o gestor mandar: só `--avaliar-teste` lê as
  anotações dele, e cada uso fica em `model/runs/uso_do_teste_detector.md`.
- Anotação parcial (decisão do gestor, 10/10/2026): no treino, cinza (114) fora do casco das
  folhas anotadas, com margem de 2%; as métricas saem nos modos região (principal) e conservador.

## Inferência para o backend (passo 3)

- `model/inferencia.py`: `carregar_modelos()` e `diagnosticar_arquivo(bytes ou caminho)` devolvem
  o contrato do coffea-web (`src/api/types.ts`); interface no README. Os parâmetros ficam em
  `inferencia.Configuracao`. O limiar de confiança 0,61 é uma proposta e aguarda a decisão do
  gestor.
- Regras de folha única (decisão do gestor, 10/10/2026): a foto inteira é classificada se houver
  uma só folha utilizável cobrindo a foto (regra 1) ou só detecção fraca com cor de folha, score
  >= 0,05 (regra 2); sem nenhuma das duas, `planta_nao_identificada`. Mais de 64 MP (cabeçalho,
  Pillow): `arquivo_muito_grande`. `especie_incorreta` não é produzido.
- O classificador atual só funciona em folha de BRACOL (fundo claro). No campo (T2 do
  `pipeline_base`), ele puxa quase tudo para ferrugem. Não usar em campo sem retreinar.

## Comandos (na raiz do repositório, com o venv ativo)

```
pip install -r requirements.txt              # CPU; no Linux (Colab) o torch do PyPI já vem com CUDA
pip install "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130
                                             # GPU no Windows: só troca o build (cu126 se o driver
                                             # for CUDA 12; ver README.md, seção Ambiente)
pip install --no-deps ultralytics==8.4.175   # detector: depois do requirements.txt (ver README.md)
python data/organize_dataset.py              # gera o manifest e o relatório de integridade
python data/organize_dataset.py --verificar  # confere dados x manifest sem mudar o manifest
python data/eda_bracol.py                    # figuras em data/reports/figures/
python data/jmuben.py                        # manifest e relatório do JMuBEN (fonte auxiliar)
python data/jmuben.py --verificar            # confere dados x manifest do JMuBEN
python data/bracot.py                        # manifest e relatório do BRACOT (detecção)
python data/bracot.py --verificar            # confere dados x manifest do BRACOT
python model/train.py --rapido               # teste de fumaça do treino (model/runs/_rapido/, fora do git)
python model/train.py --nome NOME            # treino; grava model/runs/NOME/ e não usa o teste
python model/train.py --avaliar-teste --run NOME   # só no fim: avalia no teste e registra o uso
python evaluation/atalho.py --run NOME       # checagem de atalho (só treino e val); model/runs/NOME/atalho/
python evaluation/diagnostico_blocos.py      # validação cruzada por blocos de ids x aleatória (só treino e
                                             # val, retomável); model/runs/diagnostico_blocos/
python model/treinar_detector.py --nome NOME --epocas 60   # detector: prepara os dados (fora do git),
                                             # treina e avalia na validação; model/runs/NOME/
python model/treinar_detector.py --avaliar --run NOME      # refaz a avaliação e o relatório do detector
python model/treinar_detector.py --avaliar-teste --run NOME   # só por ordem do gestor: teste do BRACOT
python model/avaliar_pipeline.py             # cadeia inteira: limiar de confiança (RF07), T1, T2 e T3;
                                             # model/runs/pipeline_base/ (--classificador, --detector e
                                             # --recortes PASTA_FORA_DO_REPO para recortes e pranchas)
jupyter nbconvert --clear-output --inplace notebooks/01_eda.ipynb   # antes de commitar
python -m pytest
```
