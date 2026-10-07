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
  mosaico derivado da EDA em `data/reports/figures/`, com crédito CC BY 4.0.
- Português do Brasil em textos, comentários e mensagens. A saída de terminal dos scripts fica
  sem acento, setas, emojis ou outros símbolos especiais: no Windows, a saída redirecionada sai
  em cp1252 e os acentos viram lixo. Arquivos gravados (relatórios, README) são UTF-8 com acento.
- Python 3.12+ e `pathlib`; caminhos resolvidos a partir de `__file__`, para os scripts rodarem
  de qualquer pasta. Nada que dependa de um shell específico.
- Sem DVC, Hydra ou frameworks de pipeline. Dependência nova: perguntar antes.
- Notebooks são finos (chamam funções dos scripts, sem duplicar lógica) e entram no git com as
  saídas limpas.
- Mudança de escopo exige aprovação do professor (RI01): sinalizar, não decidir.
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
- A correspondência das colunas phoma/cercospora com o artigo não está confirmada; a ressalva
  fica só em `RESSALVA_PHOMA_CERCOSPORA` (`data/bracol.py`).
- A ordem de `CLASSES` segue o RF01. O índice de uma classe é `CLASSES.index(nome)`, nunca o
  código `predominant_stress`.
- A divisão treino/val/teste é estável: nenhuma imagem muda de split sem `--refazer-divisao` e
  decisão explícita do usuário. Ela foi refeita do zero uma vez, em 07/10/2026, ao adotar a
  cópia completa (motivo em `HISTORICO_DIVISAO`, `data/bracol.py`).
- Teste: só BRACOL, nunca aumentado. Augmentation (Albumentations) só no treino, em memória.
- JMuBEN/JMuBEN2: só treino auxiliar, nunca teste. BRACOT: futuro (detecção por folha, RF09).

## Comandos (na raiz do repositório, com o venv ativo)

```
pip install -r requirements.txt
python data/organize_dataset.py              # gera o manifest e o relatório de integridade
python data/organize_dataset.py --verificar  # confere dados x manifest sem mudar o manifest
python data/eda_bracol.py                    # figuras em data/reports/figures/
jupyter nbconvert --clear-output --inplace notebooks/01_eda.ipynb   # antes de commitar
python -m pytest
```
