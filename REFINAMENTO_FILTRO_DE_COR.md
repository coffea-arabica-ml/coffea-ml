# Refinamento do modelo: filtro de cor de folha (passo 8)

Resumo para aprovação do líder do projeto. Trabalho feito em 30/10/2026, nos repositórios
coffea-ml (a mudança) e coffea-backend (documentação). Números completos em
[`model/runs/cor_de_folha/relatorio.md`](model/runs/cor_de_folha/relatorio.md), gerados por
`evaluation/cor_de_folha.py`.

## Resumo

- **Problema:** foto sem planta voltava como diagnóstico de folha. A `teste-sem-planta.jpg` do
  coffea-web (caneca e notebook numa mesa de madeira) voltava com 3 folhas saudáveis.
- **Solução:** um filtro de **matiz** nas máscaras do detector. Uma "folha" só entra na cadeia se
  pelo menos 33% da máscara tiver cor de folha (matiz de 24 a 110 no HSV do OpenCV, do amarelo ao
  azul claro). **Os pesos não mudam:** nem retreino, nem troca de modelo.
- **Resultado** (cadeia inteira, a resposta que chega ao usuário):

  | conjunto | diagnóstico com sucesso, antes | depois |
  |---|---:|---:|
  | 38 fotos sem planta | 23 | **9** |
  | 30 fotos reais do BRACOT (treino) | 28 | **28** |

- **Custo nas folhas reais:** 0 das 61 folhas doentes e 0 das 42 saudáveis do campo rotulado são
  cortadas. No total, 11 de 1.727 detecções (0,6%), e nenhuma foto fica sem folha.
- **O que não resolve:** objetos verdes, amarelos ou azuis ainda podem passar, e a própria
  `teste-sem-planta.jpg` continua passando (o notebook azul-marinho). A causa está explicada na
  seção "Brilho mínimo: hipótese testada e descartada".
- **Pedido de aprovação:** adotar o filtro como padrão da cadeia (`inferencia.Configuracao`) e
  levá-lo à produção.

## Contexto e decisões

| data | decisão do gestor |
|---|---|
| 11/10/2026 | Pesos congelados (`base_resnet50_bracol` e `detector_base`), sem retreino. |
| 30/10/2026 | Congelamento aberto só para a cadeia: filtro de cor sem troca de pesos (caminho A). O retreino do detector com imagens negativas (caminho B) ficou de fora pelo prazo (06/11) e pela necessidade de GPU. |
| 30/10/2026 | Autorizados os downloads dos pesos (Release `pesos-v1`), do BRACOT e de fotos sem planta. |
| 30/10/2026 | Adotar o filtro sem brilho mínimo (opção 1). |

A decisão está registrada no `CLAUDE.md` do coffea-ml (seção "Filtro de cor de folha") e na
exceção ao congelamento do `CLAUDE.md` e do `AGENTS.md` do coffea-backend.

## Por que o modelo via folha onde não havia

1. **O detector nunca viu um exemplo negativo.** O YOLO11s-seg foi ajustado só no BRACOT, que é
   100% foto de cafeeiro. Qualquer coisa com formato parecido com folha recebia score alto (0,80
   a 0,94).
2. **O filtro de cor existente só media saturação.** Ele foi feito para tirar o fundo branco do
   BRACOL; uma mesa de madeira, uma caneca colorida ou a pele passavam, porque são "coloridas".

Por isso o README do backend já registrava que nenhum limiar de detecção separava os dois casos.

## Como foi feito

### 1. Preparação
- **Pesos** baixados da Release `pesos-v1`. O tamanho e o SHA-256 dos dois conferem com
  `model/pesos/MANIFESTO.md`.
- **BRACOT:** só o `train/` (240 fotos, 1.318 folhas) foi extraído do zip do Mendeley. O `test/`,
  fechado, não foi extraído, e o zip foi apagado. O `data/bracot.py --verificar` confere as 240
  fotos e as 1.318 folhas com o manifest versionado.
- Nenhum teste fechado (BRACOL de teste, `test/` do BRACOT, folhas de campo de teste) foi lido.

### 2. Medição da cor das folhas (só treino do BRACOT)
- **Pixels medidos:** em cada uma das 1.318 folhas anotadas, o matiz dos pixels com saturação
  ≥ 30, na cópia da foto reduzida a 1.024 px, exatamente como a cadeia mede.
- **Faixa:** o menor arco do círculo de matiz que cobre 99% desses pixels, com cada folha pesando
  igual. Deu **H de 24 a 110** (48° a 220°).
- **Corte:** o p1 da fração da máscara com cor de folha, arredondado para baixo. Deu **33%**.
- **Conferência do método:** recalculado do mesmo jeito, o p1 do filtro de saturação deu 34,0%,
  contra os 35% medidos em 10/10/2026.
- **Detecções reais:** nas mesmas fotos, as detecções foram casadas pela caixa com as folhas S, D
  e "?" do campo rotulado de treino, para medir o custo em folhas doentes.

### 3. Fotos sem planta, só para avaliação
- **Fonte:** 55 fotos do Wikimedia Commons, escolhidas por categoria, com ênfase no caso difícil
  (objetos verdes e amarelos: canecas, copos, garrafas, sacolas, camisetas, bolas de tênis) e em
  cenas comuns (mesas de madeira, notebooks, mãos, sapatos, caixas).
- **Curadoria visual:** 18 foram descartadas, 12 por terem vegetação (gramado, árvores, vasos,
  buquê, palha) e 6 por serem preto e branco ou sépia. Ficaram **37**, mais a
  `teste-sem-planta.jpg`.
- **Licenças:** todas livres (CC0, domínio público, CC BY e CC BY-SA). Autor, licença e link de
  cada foto estão em
  [`data/manifests/negativas_creditos.csv`](data/manifests/negativas_creditos.csv).
- **Fora do git:** as fotos ficam em `data/negativas/`, incluída no `.gitignore`.
- **Uso:** só avaliação. As fotos nunca escolheram a faixa nem o corte.

### 4. Primeira avaliação
O filtro reduziu as fotos sem planta com "folha" utilizável de **32 para 13** de 38. Madeira,
pele, papelão e objetos vermelhos ou laranja passaram a ser cortados. A `teste-sem-planta.jpg`
continuou passando: a mesa de madeira (matiz ≈ 12) foi cortada, mas o notebook (matiz ≈ 107)
não.

### 5. Brilho mínimo: hipótese testada e descartada
- **Hipótese:** o azul da faixa vinha das sombras, porque o matiz de pixel escuro é ruído. A
  ideia era descartar os pixels escuros.
- **Teste:** a faixa e o corte foram recalculados nas folhas para brilho mínimo de 0 a 96. Pelo
  rigor do projeto, o valor não foi escolhido olhando as negativas.
- **Resultado:** a faixa não encolhe (continua indo até 110–112) e as negativas não melhoram.

  | brilho mínimo | faixa (H) | corte | detecções cortadas | S cortadas | negativas com folha |
  |---:|---|---:|---:|---:|---:|
  | 0 | 24 a 110 | 33% | 11 de 1.727 | 0 de 42 | 13 de 38 |
  | 32 | 24 a 110 | 33% | 11 de 1.727 | 0 de 42 | 13 de 38 |
  | 48 | 24 a 110 | 33% | 11 de 1.727 | 0 de 42 | 13 de 38 |
  | 64 | 25 a 111 | 30% | 14 de 1.727 | 0 de 42 | 13 de 38 |
  | 80 | 24 a 111 | 25% | 15 de 1.727 | 0 de 42 | 13 de 38 |
  | 96 | 24 a 112 | 17% | 35 de 1.727 | 2 de 42 | 15 de 38 |

- **Causa real:** o azul é **reflexo do céu** na folha de café, que é brilhante (cutícula
  cerosa):
  - 8,6% dos pixels coloridos das folhas estão no azul (H 90 a 112), e são pixels **claros** (V
    de 160 a 240);
  - 207 das 1.318 folhas têm mais de 10% da máscara nesse azul;
  - cerca de 1% das folhas é predominantemente azulada.

  Pela cor, um objeto azul é indistinguível dessas folhas. Tirar o azul da faixa cortaria folhas
  reais.

### 6. Implementação na cadeia
Em `model/inferencia.py`:
- **Parâmetros novos** em `Configuracao`: `faixa_matiz_folha = (24, 110)` e
  `fracao_cor_de_folha_minima = 0.33`. Com o valor 0, o filtro fica desligado.
- **Onde vale:** o filtro vale para as folhas detectadas **e para a regra 2** (foto inteira).
  Sem isso, um objeto cortado como folha voltaria por ela.
- **Cadeias de rotulagem:** em `campo.CONFIG_ROTULAGEM`, `CONFIG_ROTULAGEM_TREINO` e
  `teste_final.CONFIG_ROTULAGEM_TESTE`, o filtro fica desligado, para os ids das folhas rotuladas
  não mudarem.

### 7. Validação
- **coffea-ml:** 373 testes passaram. Há 12 testes novos do filtro (10 em
  `tests/test_cor_de_folha.py` e 2 em `tests/test_inferencia.py`) e 1 fixture atualizado. Os 2
  testes do Grad-CAM foram desmarcados porque o `grad-cam` não estava instalado na máquina; eles
  não têm relação com a mudança.
- **coffea-backend:** 118 testes passaram, incluindo o `test_modelo_real` com os pesos reais e o
  coffea-ml alterado.
- **Cadeia inteira** (`diagnosticar_arquivo`), sem e com o filtro:
  - nas negativas, o sucesso caiu de 23 para 9. Várias respostas que eram `baixa_confianca`
    viraram `planta_nao_identificada`, que é a mensagem certa;
  - no BRACOT, 28 de 30 fotos com sucesso, antes e depois. Duas mudaram em ±1 folha
    (`20190831_163057`: 9 → 8; `20190831_164946`: 2 → 3). A segunda ganhou uma folha porque o
    filtro de fragmentos compara cada folha com a mediana das folhas da foto, e essa mediana muda
    quando um objeto sai da conta.

## Limitações e ressalvas

- **Números otimistas:** as 38 fotos sem planta motivaram o teste do brilho mínimo, então viraram
  conjunto de desenvolvimento. O número honesto viria de 15 a 20 fotos novas da equipe, que não
  guiaram nenhuma escolha.
- **O que ainda passa:** objetos verdes, amarelos e azuis. Das 9 fotos sem planta que ainda dão
  sucesso, quase todas têm esses objetos, e uma delas é a `teste-sem-planta.jpg`.
- **Folhas do BRACOL não medidas:** o BRACOL (folha sobre fundo claro, caminho de foto inteira)
  não estava na máquina. O risco é uma folha com ferrugem extensa ser cortada.
- **Corte medido nas próprias fotos de treino:** o detector treinou nessas fotos, então as
  máscaras saem mais limpas que no campo.
- **Testes fechados:** os números do passo 7 mediram a cadeia **sem** este filtro, e os testes
  não são reabertos.

## Arquivos alterados

**coffea-ml**

| arquivo | mudança |
|---|---|
| `model/inferencia.py` | o filtro de matiz (`fracao_cor_de_folha`, `com_cor_de_folha`), também na regra 2 |
| `evaluation/campo.py`, `model/teste_final.py` | filtro desligado nas cadeias de rotulagem |
| `evaluation/cor_de_folha.py` | novo: medição, sensibilidade, avaliação e comparação da cadeia |
| `model/runs/cor_de_folha/` | novo: `medicao.json` e `relatorio.md` |
| `tests/test_cor_de_folha.py`, `tests/test_inferencia.py`, `tests/test_campo_cadeia_final.py` | testes |
| `data/manifests/negativas_creditos.csv` | novo: créditos das fotos sem planta |
| `.gitignore` | `data/negativas/` fora do git |
| `README.md`, `CLAUDE.md` | passo 8, tabela de parâmetros, limitações |

**coffea-backend:** `README.md` ("Limitações conhecidas"), `CLAUDE.md` e `AGENTS.md` (a exceção ao
congelamento). Nenhum código do backend mudou.

## Como reproduzir

Na raiz do coffea-ml, com os pesos (`pesos-v1`) e o treino do BRACOT em
`data/raw/bracot/bracot-data/train/`, e as fotos sem planta em `data/negativas/`:

```bash
python evaluation/cor_de_folha.py              # mede tudo (cerca de 5 min em CPU)
python evaluation/cor_de_folha.py --negativas  # só as fotos sem planta e a cadeia inteira
python -m pytest
```

## Próximos passos

1. **Aprovação** deste refinamento.
2. **Produção:** atualizar o commit fixado do coffea-ml em `/opt/cafelens/coffea-ml/` e
   reconstruir a imagem da API (`deploy/DEPLOY.md` do backend). Os pesos não mudam.
3. **Número honesto:** opcionalmente, medir com 15 a 20 fotos sem planta tiradas pela equipe.
4. **Depois da entrega:** retreinar o detector com imagens negativas (caminho B), a correção de
   raiz para "o que é folha".

## Registro de incidentes

- **Git no backend:** foram feitos `git checkout main` e `git pull --ff-only` antes de ler o
  `CLAUDE.md`, que reserva essas operações ao gestor. Foi um avanço simples, sem perda de nada.
- **Relatório sobrescrito:** o `data/bracot.py --verificar` reescreveu
  `data/reports/integridade_bracot.md` com a versão sem o teste. O arquivo foi restaurado do git.
