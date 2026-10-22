# coffea-ml
Dados, notebooks e treinamento do modelo de classificação de estresses bióticos em Coffea arabica

Dados (fontes BRACOL, JMuBEN e BRACOT, decisões do gestor, manifests, divisão e como reproduzir): ver [data/README.md](data/README.md).

## Ambiente

Python 3.12 ou mais novo, num venv na raiz do repositório (passo a passo em
[data/README.md](data/README.md), seção "Reproduzir do zero"). O `requirements.txt` fixa as
versões. Com GPU, só muda o build do torch e do torchvision, nunca a versão.

- **CPU, em qualquer sistema:**
  ```
  pip install -r requirements.txt
  ```
  No Windows, o torch do PyPI é só de CPU.
- **GPU NVIDIA no Windows:** depois do comando acima, troque o build:
  ```
  pip install "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130
  ```
  - **Qual build:** o do CUDA não pode passar da "CUDA Version" que o `nvidia-smi` mostra, que é
    a do driver. O cu130 pede CUDA 13.0 ou mais. Com um driver de CUDA 12, use o cu126: troque
    `cu130` por `cu126` nos três lugares do comando.
  - **O que muda:** o download tem cerca de 1,9 GB, e só o torch e o torchvision trocam. As
    dependências são as mesmas do build de CPU.
  - **Conferência:** `python -c "import torch; print(torch.cuda.is_available())"` tem de mostrar
    `True`.
- **Google Colab (Linux, com GPU):** `pip install -r requirements.txt`.
  - No Linux, o torch do PyPI já vem com CUDA 13.0 e substitui o torch que o Colab traz.
  - Se o `nvidia-smi` do Colab mostrar CUDA 12, troque o build pelo cu126, como no Windows:
    ```
    pip install "torch==2.13.0+cu126" "torchvision==0.28.0+cu126" --index-url https://download.pytorch.org/whl/cu126
    ```
  - Se der conflito, o `requirements.txt` sugere apagar as linhas do torch e do torchvision e
    ficar com o torch do Colab, em outra versão. As versões usadas ficam no `config.json` de
    cada run.
- **Detector de folhas (ultralytics), em qualquer sistema:** depois do `requirements.txt`, sem
  dependências:
  ```
  pip install --no-deps ultralytics==8.4.175
  ```
  O ultralytics pede o `opencv-python`, que instalaria um segundo módulo `cv2` por cima do
  `opencv-python-headless` (os dois pacotes gravam o mesmo `cv2`). As outras dependências dele já
  estão no `requirements.txt`, fixadas. Por isso o `pip check` acusa "ultralytics 8.4.175
  requires opencv-python, which is not installed": é esperado, e o headless atende o ultralytics.

## Treino

`python model/train.py` treina o classificador (estresse e severidade) e grava cada execução em
`model/runs/<nome>/`. O uso está no docstring de `model/train.py`, e os comandos, no
`CLAUDE.md`.

## Detector de folhas

`python model/treinar_detector.py` ajusta o YOLO11s-seg (ultralytics) nas fotos de treino do
BRACOT e avalia na validação (cenas inteiras do treino). `model/detector.py` tem
`detectar_folhas(imagem_rgb, limiar)`, que devolve o polígono, a caixa e o score de cada folha,
para o backend. O run de base, com as métricas e a medida do RNF02, está em
`model/runs/detector_base/relatorio.md`. O teste do BRACOT foi aberto uma única vez, no passo 7
(seção "Testes fechados", no fim).

- **Licença (decisão do gestor, 10/10/2026):** o ultralytics e os pesos YOLO são AGPL-3.0,
  aceita para este projeto acadêmico de repositórios públicos. Se o backend servir o modelo pela
  rede, o código-fonte do serviço tem de ficar disponível a quem o usa.
- **Peso pré-treinado:** o `yolo11s-seg.pt` da release oficial v8.3.0 do ultralytics, em
  `model/pesos/` (fora do git). O treino confere o SHA-256 (`detector.SHA256_PESO_PRETREINADO`).
  Para baixá-lo:
  ```
  curl -L -o model/pesos/yolo11s-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
  ```
- **Rede:** `detector.importar_ultralytics()` impõe o modo offline do ultralytics e desliga a
  instalação automática de pacotes, a telemetria (`sync`) e a pré-carga da fonte dos gráficos. O
  treino roda sem AMP (a checagem de AMP baixaria outro peso) e sem gráficos.

## Interface para o backend

`model/inferencia.py` é o módulo que o backend chama. Ele recebe a foto (de uma planta, de um
galho ou de uma folha) e devolve o diagnóstico no contrato dos frontends (coffea-web:
`docs/frontend-reference/03-contrato-api-mock.md` e `src/api/types.ts`). A cadeia é:

1. o detector acha as folhas;
2. saem as detecções pequenas, os pedaços de fundo liso (filtro de cor) e os pedaços de folha
   cortados pela borda (filtro de fragmentos);
3. as regras de folha única decidem se a foto inteira vira uma folha;
4. cada folha é recortada deitada, com margem, como nas fotos do BRACOL;
5. o classificador avalia todas as folhas num lote só;
6. a regra de decisão do caminho dá a classe, e as folhas abaixo do corte saem: na foto inteira,
   o argmax com o limiar de confiança; nos recortes por detecção, a regra em duas etapas (ver
   "Regra de decisão por caminho", abaixo).

A cadeia está congelada desde o passo 6 (decisões do gestor, 11/10/2026). A avaliação completa
dela, com o tempo em CPU e a comparação com os padrões anteriores, está em
`model/runs/pipeline_final/relatorio.md`. Os runs anteriores ficam como registro:
- `pipeline_padrao_b`: argmax em todo caminho;
- `pipeline_duas_etapas`: duas etapas em todo caminho;
- `pipeline_base`: o passo 3, com recorte A e sem filtro.

```python
import sys
sys.path.insert(0, "coffea-ml/model")  # a pasta model/ do repositório
import inferencia

inferencia.carregar_modelos()                        # uma vez, ao subir o backend (CPU)
resposta = inferencia.diagnosticar_arquivo(conteudo)  # bytes do upload ou caminho da foto
```

**Pesos.** Ficam fora do git e são lidos de arquivos locais, sem rede. Os nomes, tamanhos e
SHA-256 estão em [`model/pesos/MANIFESTO.md`](model/pesos/MANIFESTO.md):
- o `melhor.pt` do classificador, em `model/runs/base_resnet50_bracol/`;
- o `last.pt` do detector, em `model/runs/detector_base/weights/`.

Quem não tem os arquivos pode baixá-los na
[Release `pesos-v1`](https://github.com/coffea-arabica-ml/coffea-ml/releases/tag/pesos-v1), que
traz o destino de cada um. Renomeie para o caminho indicado acima e confira o SHA-256 com o
manifesto antes de usar (`Get-FileHash -Algorithm SHA256 <arquivo>` no PowerShell,
`sha256sum <arquivo>` no Linux).

Para carregar:
- **Nos caminhos padrão:** `carregar_modelos()`.
- **Em outro lugar:** `carregar_modelos(pasta_do_classificador=PASTA_DO_melhor.pt,
  pesos_do_detector=CAMINHO_DO_last.pt, dispositivo="cpu")`.
- **Cache:** os modelos ficam em cache por caminho e dispositivo.
- **Antes de subir o backend:** confira o SHA-256 dos dois arquivos com o manifesto, que traz o
  comando.

**Versões das dependências.** As do `requirements.txt`, mais o ultralytics sem dependências (seção
"Ambiente"). Os pesos foram gerados com estas versões, e a cadeia final foi avaliada com elas:

| pacote | versão |
|---|---|
| Python | 3.12 ou mais novo (avaliado no 3.13.14) |
| torch / torchvision | 2.13.0 / 0.28.0 (no CPU, o build do PyPI serve) |
| timm | 1.0.28 |
| albumentations | 2.0.8 |
| numpy / pandas | 2.5.2 / 3.0.5 |
| pillow | 12.3.0 |
| opencv-python-headless | 5.0.0.93 |
| ultralytics | 8.4.175 (`pip install --no-deps`; AGPL-3.0) |

**Sucesso** (exemplo do formato):

```json
{
  "status": "sucesso",
  "folhas": [
    {"id": "folha-1", "categoria": "ferrugem", "severidade": "baixa",
     "regiao": {"x": 0.412, "y": 0.377, "raio": 0.142},
     "confianca": 0.81, "scoreDeteccao": 0.93, "caixa": [0.28, 0.21, 0.55, 0.52],
     "cortadaNaBorda": false},
    {"id": "folha-2", "categoria": "saudavel", "severidade": "saudavel",
     "regiao": {"x": 0.731, "y": 0.664, "raio": 0.118},
     "confianca": 0.21, "scoreDeteccao": 0.88, "caixa": [0.63, 0.52, 0.86, 0.81],
     "cortadaNaBorda": true}
  ],
  "detalhes": {"plano": "deteccao", "regra": null, "deteccoes": 7, "abaixo_do_limiar": 4,
               "cortadas_por_area": 0, "cortadas_por_cor": 0, "cortadas_por_fragmento": 0,
               "cortadas_por_limite": 0,
               "abaixo_da_confianca": 5, "recorte": "B", "decisao": "duas_etapas",
               "decisao_configurada": "por_caminho",
               "limiar_saudavel": 0.083, "limiar_doenca": 0.65,
               "classificador": "base_resnet50_bracol", "detector": "detector_base"}
}
```

Na foto inteira (`plano` "foto_inteira"), `detalhes` traz `decisao` "argmax" e
`limiar_confianca` 0,61 no lugar de `limiar_saudavel` e `limiar_doenca`.

- **Categoria e severidade:**
  - `categoria` vem de `CLASSES` (saudavel, ferrugem, bicho_mineiro, phoma, cercosporiose).
  - `severidade` vem dos níveis 0 a 4 do modelo: saudavel, muito_baixa, baixa, alta e muito_alta.
  - A coerência é a do normalizador do frontend: folha saudável tem severidade "saudavel", e folha
    doente com nível 0 vira "muito_baixa".
- **Região:**
  - `regiao` é o centroide da máscara, relativo à largura e à altura da foto.
  - O raio é o do círculo de mesma área, relativo à menor dimensão, limitado a [0,01; 0,5].
  - Quando a foto inteira vira uma folha, a região é a da detecção na regra 1 e o meio da foto, com
    raio 0,5, na regra 2.
- **Extras:** `confianca`, `scoreDeteccao` (nulo sem detecção), `caixa` (relativa, [x0, y0, x1,
  y1]), `cortadaNaBorda` e `detalhes`. São campos extras, e o normalizador do frontend os ignora.
- **`confianca`:** depende da regra do caminho.
  - Na foto inteira, é a probabilidade da classe prevista.
  - Nos recortes por detecção, a folha saudável traz P(saudável), que pode ser baixa (0,1, por
    exemplo), e a doente traz a probabilidade renormalizada entre as 4 doenças.
  - Como o frontend mostra a confiança das folhas saudáveis ainda está pendente: P(saudável) não
    é a chance de o diagnóstico estar certo.
- **imagemUrl:** fica com o backend, que guarda a foto. Sem ela, o frontend usa a foto local.

**Erros** (`{"status": "erro", "tipo": ..., "mensagem": ...}`, às vezes com `detalhes`):

| tipo | quando | mensagem |
|---|---|---|
| `planta_nao_identificada` | nenhuma folha utilizável, e nenhuma detecção com cor de folha e score >= 0,05 | "Não encontramos folhas de café na foto. Fotografe a planta ou a folha mais de perto, com boa luz." |
| `baixa_confianca` | todas as folhas abaixo do corte da regra do caminho (RF07): na foto inteira, confiança < 0,61; nos recortes por detecção, doença com probabilidade renormalizada < 0,65 (as saudáveis não têm corte) | "Encontramos folhas, mas o diagnóstico não ficou confiável. Tente outra foto, mais nítida e com boa luz." |
| `baixa_confianca` | o filtro de fragmentos tirou todas as folhas (`detalhes.plano` = "fragmentos") | "Só aparecem pedaços de folha na foto. Enquadre a folha inteira e tente de novo." |
| `formato_invalido` | o arquivo não decodifica (JPG ou PNG) | "Não foi possível ler a imagem. Envie uma foto em JPG ou PNG." |
| `arquivo_muito_grande` | mais de 64 megapixels, lidos do cabeçalho com Pillow, sem decodificar | "A imagem tem 9000 x 8000 pixels (72,0 megapixels); o limite é 64,0 megapixels." |

- **`especie_incorreta`:** não é produzido. Não há dados de outras plantas para treinar essa recusa,
  então uma foto de outra planta pode sair como folha de café.
- **Limite de 10 MB:** o limite de tamanho do arquivo é do backend.

**Parâmetros congelados** (`inferencia.Configuracao`, uma dataclass congelada; `diagnosticar(imagem,
config)`). Os padrões são os da cadeia final, e nenhum limiar se reajusta com dados de campo
(decisão do gestor, 11/10/2026):

| parâmetro | padrão | o que faz |
|---|---|---|
| `limiar_deteccao` | 0,35 | score mínimo do detector (decisão do gestor, 10/10/2026) |
| `max_folhas` | 15 | folhas classificadas por foto, as de maior score (RNF02) |
| `area_minima` | 0,5% | área mínima da máscara, em fração da foto |
| `saturacao_minima`, `fracao_colorida_minima` | 30, 25% | filtro de cor: tira os pedaços de fundo liso |
| `limiar_confianca` | 0,61 | RF07: corte do argmax, na foto inteira e no modo A (decisão do gestor, 11/10/2026) |
| `recorte` | "B" | "B": fundo fora da máscara em cinza; "A": a foto em volta da folha. B por padrão (decisão do gestor, 11/10/2026) |
| `margem`, `cinza_fundo` | 0,17, 197 | medidos no treino do BRACOL |
| `cobertura_folha_unica` | 30% | regra 1: uma folha cuja caixa cobre isso da foto → a foto inteira |
| `score_minimo_folha_unica` | 0,05 | regra 2 (decisão do gestor, 10/10/2026) |
| `max_megapixels` | 64 | acima disso, `arquivo_muito_grande` |
| `fracao_visivel_minima` | 0,5 | filtro de fragmentos: sai a folha com área menor que metade da mediana das folhas inteiras da foto (decisão do gestor, 11/10/2026) |
| `preenchimento_minimo` | 0 | filtro de fragmentos pela área sobre o retângulo mínimo; desligado (não discriminou no campo) |
| `decisao` | "por_caminho" | regra de decisão (abaixo); "argmax" e "duas_etapas" aplicam uma regra só em todo caminho |
| `limiar_saudavel` | 0,083 | 1ª etapa da regra em duas etapas; fixado pelo gestor em 11/10/2026, sem reajuste |
| `limiar_doenca` | 0,65 | 2ª etapa; **provisório** (decisão do gestor, 11/10/2026), calibrado só no BRACOL (`pipeline_duas_etapas`) |

**Regra de decisão por caminho** (`decisao="por_caminho"`, o padrão desde o passo 6):
- **Foto inteira** (regras de folha única 1 e 2, como as fotos do BRACOL): argmax, com corte em
  `limiar_confianca`.
- **Recortes por detecção, no modo B:** a regra em duas etapas.
  - **1ª etapa:** com P(saudável) >= `limiar_saudavel`, a folha é "saudavel", e o corte de
    confiança não se aplica a ela.
  - **2ª etapa:** senão, a classe é a doença de maior probabilidade, com o softmax renormalizado
    entre as 4 doenças. Abaixo de `limiar_doenca` (sobre a probabilidade renormalizada), a
    folha sai.
- **Modo A:** fica no argmax, porque o limiar de saudável só foi fixado e validado no modo B.
- **Padrões anteriores, como opção:**
  - `Configuracao(decisao="argmax")` é a cadeia do `pipeline_padrao_b`;
  - `Configuracao(decisao="duas_etapas")` é a do `pipeline_duas_etapas`.
- **`detalhes`:** `decisao` é a regra aplicada às folhas da foto, e `decisao_configurada`, a da
  configuração.

**Filtro de fragmentos.** A fração visível é a área da folha sobre a mediana da área das folhas
inteiras da mesma foto (ou de todas, se nenhuma for inteira). Por isso:
- uma folha sozinha tem fração 1 e nunca sai;
- pelo menos metade das folhas de uma foto sempre fica;
- o erro de fragmentos só aparece com `preenchimento_minimo` ligado.

**Efeito dos padrões de 11/10/2026** (`pipeline_padrao_b` contra `pipeline_base`; detalhes nas
observações do relatório):
- **T1 (BRACOL):** a cadeia B acerta 94,0%, contra 94,4% da A. Só o recorte, sem a regra 1, cai de
  92,4% (A) para 88,4% (B). O filtro tira 1 folha de 1 imagem, sem mudar a classe.
- **T2 (campo):** as folhas classificadas caem de 318 para 239 (79 fragmentos). As fotos em
  `baixa_confianca` no limiar de 0,61 sobem de 20 para 25 de 46.

**Efeito da cadeia final** (`pipeline_final` contra `pipeline_padrao_b` e `pipeline_duas_etapas`).
São os mesmos modelos e as mesmas probabilidades; só a regra muda. Recalculadas aqui, as regras
dos runs anteriores dão os números gravados neles.
- **T1, foto inteira (203 das 252 imagens):** igual ao argmax. Das mantidas, 95,3% estão certas,
  e nenhuma das 166 doentes é chamada saudável.
- **T1, recortes por detecção (48):** igual às duas etapas.
  - Das mantidas, 90,2% (77,5% a 96,1%) estão certas, contra 95,1% (83,9% a 98,7%) do argmax.
  - 3 das 44 doentes viram saudáveis.
  - Os intervalos se sobrepõem.
- **T1, todas:** das mantidas, 94,4% estão certas, contra 95,2% do argmax e 93,9% das duas etapas
  em todo caminho. A cobertura é a mesma nas três (231 de 252), e nenhuma diferença é sustentada.
- **T2 (campo, sem rótulo):** 122 das 239 folhas saem saudáveis, contra 0 no argmax com 0,61. As
  fotos em `baixa_confianca` caem de 25 para 1 de 46.
- **T3:** mediana de 0,80 s, pior caso estimado de 2,15 s.

**Limitações** (decisão do gestor, 11/10/2026; números em `model/runs/pipeline_final/relatorio.md`):
- **Treino:** o classificador foi treinado só com folhas do BRACOL (fundo claro). No campo, a regra
  por caminho corrige a decisão saudável x doente, não o classificador.
- **Rótulos de campo:** as 120 folhas antigas e as 160 novas foram rotuladas por Claude por
  leitura visual das pranchas cegas (só S e D evidentes; sem especialista; uma fazenda), aceitas
  pelo gestor em 11/10/2026. As 115 do teste também, aceitas pelo gestor em 14/10/2026. São
  rótulos feitos por IA (leitura visual), não por especialista nem pelo gestor.
- **Amostra:** 183 folhas evidentes (77 S e 106 D), de 1 lavoura e 2 sessões.
- **Falso saudável nas D evidentes:** 5,7% (2,6% a 11,8%) nas 106 D. Nas 60 D novas, 6,7%, com IC
  até 15,9%. Os dados não mostram que a regra respeita o teto de 5%.
- **Folhas ambíguas ("?"):** a regra chama saudável 54% das "?" novas (32 de 59) e 62% das duas
  rotulagens juntas (60 de 97). Se elas forem doentes, o falso saudável sobe; nas novas, com os
  "?" como D, vai a 30%.
- **Recall das saudáveis:** a regra chama saudável 81,8% das S (71,8% a 88,8%), contra 0% do argmax
  com 0,61.
- **Classe da doença no campo:** não foi verificada, porque os rótulos de campo só dizem S ou D.
  O limiar de doença (0,65, provisório) só foi calibrado no BRACOL.
- **D evidentes que chegam ao usuário** (passo 6b, `model/runs/campo_cadeia_final`, com a cadeia
  final inteira, incluindo o filtro de fragmentos):
  - como doente: 29,2% (21,4% a 38,5%) das 106;
  - como saudável: 4,7%;
  - somem: 66,0%, sendo 57,5% pelo corte t_d e 8,5% pelo filtro de fragmentos.
  - Com o t_d desligado (só informação), chegariam como doente 86,8%, mas as S chamadas doentes
    subiriam de 3,9% para 18,2%.
  - Das 42 doentes mantidas, 81% saem como ferrugem; a classe não foi verificada.
- **Severidade:** não é informativa no campo.
- **Teste de campo** (passo 7; seção "Testes fechados"): 115 folhas das fotos de teste,
  rotuladas por Claude por leitura visual das pranchas cegas (só S e D evidentes; sem
  especialista; uma fazenda), aceitas pelo gestor em 14/10/2026.
  - D chamadas saudáveis: 10,9% (4,7% a 23,0%; 5 de 46), o maior dos conjuntos rotulados, sem
    diferença sustentada;
  - D que chegam como doente: 26,1%, e 63,0% somem pelo t_d.

## Classificador para o campo

O classificador do run base acerta no BRACOL, mas no campo (fotos de validação do BRACOT) manda
quase tudo para doença. `model/treinar_campo.py` compara variantes de treino:
- **v0:** a augmentation atual;
- **v1:** augmentation de campo (v1f, só fotométrica; v1g, só corte e erasing);
- **v2:** v1 com troca do fundo pela máscara da folha;
- **v3 e v3s:** v2 com o JMuBEN (v3s sem phoma e cercosporiose).

Todas usam 20 épocas fixas e a última época. A comparação usa a validação oficial, a validação
cruzada nas dobras do diagnóstico por blocos e o campo. `evaluation/campo.py` gera as pranchas
cegas de rotulagem e mede o campo:
- **proxies:** concordância entre os modos A e B, estabilidade sob perturbações e distribuição
  das classes;
- **rótulos de campo**, quando o CSV está preenchido.

O relatório comparativo, com a regra de adoção aplicada, fica em
`model/runs/campo_comparacao/relatorio.md`. O modelo oficial do pipeline só muda por decisão do
gestor. As 120 folhas do campo rotulado foram rotuladas por Claude por leitura visual das
pranchas cegas (só S e D evidentes; sem especialista; uma fazenda), aceitas pelo gestor em
11/10/2026 (2 sessões; ver `data/README.md`).

**Decisões do gestor (11/10/2026).** Não há validação cruzada das variantes do passo 4, e o
modelo oficial segue o `base_resnet50_bracol`. O limiar de confiança ficou em 0,61 na foto
inteira da cadeia final (passo 6).

**Diagnóstico sem treino (passo 5a).** `evaluation/campo_diagnostico.py` usa só os checkpoints
existentes, nas 82 folhas S ou D do campo rotulado (sem os "?"):
- **AUC de P(saudável):** separa S de D, nos modos A e B, com bootstrap agrupado por foto. Inclui o
  ponto de operação com no máximo 5% das D marcadas saudáveis, escolhido nas mesmas folhas
  (otimista).
- **Sonda linear:** regressão logística sobre os embeddings congelados, com dobras agrupadas por
  cena. Não gera modelo.
- **Controles:** a sessão das fotos e as folhas que ficam no filtro de fragmentos.
- **Efeito no BRACOL:** o que o limiar de campo faria na validação do BRACOL.
- **Auditoria das S previstas como doença:** pranchas em resolução nativa e Grad-CAM, em
  `data/processed/campo_diagnostico/`, fora do git.

O relatório, com a leitura, fica em `model/runs/campo_diagnostico/relatorio.md`.

**Regra saudável x doente (passo 5b; decisões do gestor, 11/10/2026).**
- **Modelos finais:** o `base_resnet50_bracol` e o `detector_base` ficam congelados, sem retreino
  final. Os testes fechados medem estes modelos uma única vez, no fim.
- **Regra em duas etapas:** desde o passo 6, ligada nos recortes por detecção (ver "Regra de
  decisão por caminho"). O limiar de saudável, 0,083, foi fixado antes de dados novos e não se
  reajusta.
- **Limiar de doença:** 0,65, calibrado só no BRACOL. É o menor limiar com 95% de acerto nas
  doentes fora da dobra. Na cadeia final, é provisório.
- **Rótulos:** os das 82 folhas ficam congelados.
- **Run `model/runs/pipeline_duas_etapas`:** roda T1 a T3 com a regra ligada.
- **Resultado no BRACOL:** fora da dobra, a regra completa fica abaixo do argmax com 0,61 (91,7%
  contra 95,1% das mantidas, com intervalos que não se tocam). P(saudável) >= 0,083 marca 5,1%
  das doentes como saudáveis.
- **Resultado no campo (T2, sem rótulos):** 122 das 239 folhas viram saudáveis.
- **Validação no campo** (`evaluation/campo_validacao.py`, em
  `model/runs/campo_validacao_duas_etapas`): usa as 160 folhas novas das fotos de treino do
  detector (`data/manifests/campo_rotulado_treino.csv`), com a regra fixada e sem reajuste.
  - **Saudável dado S:** a regra chama 80,5% (66,0% a 89,8%), contra 0% do argmax com 0,61.
  - **Saudável dado D:** 6,7% (2,6% a 15,9%), contra 0%; os intervalos se tocam.
  - **Novas contra as 82 antigas:** as novas saem um pouco piores, sem diferença sustentada.
  - **"?":** são 37% das folhas e a maior incerteza.
  - **Decisão do gestor (11/10/2026, passo 6):** a regra fica ligada só nos recortes por
    detecção, com o limiar de doença de 0,65 provisório; a foto inteira fica no argmax com 0,61.
    A cadeia final está em `model/runs/pipeline_final`.

## Testes fechados (passo 7, uso único)

Os testes foram abertos uma única vez, em 14/10/2026, com o "pode" do gestor e a cadeia
congelada. Nada foi ajustado nem escolhido com estes números. O script é `model/teste_final.py`.
O log único, `model/runs/uso_do_teste_final.md`, registra:
- a abertura, antes da leitura, com o commit, os SHA dos pesos e a configuração conferida;
- o que seria medido, declarado antes;
- cada evento, com o SHA-256 dos arquivos gerados.

| uso | teste | validação (ou validação cruzada) |
|---|---|---|
| **BRACOL, cadeia final:** acurácia sem corte | 88,1% (83,5% a 91,5%), F1 macro 86,0% | 92,8% (89,0% a 95,4%), F1 macro 91,6% |
| BRACOL, cadeia final: mantidas no corte | 225 de 253, com 91,1% certas (86,7% a 94,2%) | 231 de 252, com 94,4% certas |
| BRACOL, cadeia final: doentes chamadas saudáveis | 2,4% (5 de 212) | 1,4% (3 de 211) |
| BRACOL, classificador sozinho (informação) | 90,5% (86,3% a 93,5%), F1 macro 88,5% | 94,8%, F1 macro 93,2%; validação cruzada por blocos: 89,9%, F1 macro 87,2% |
| **BRACOT, detecção:** AP50 de máscara, região / conservador | 92,0% / 81,6% (sem as 2 fotos sobrepostas: 91,6% / 81,3%) | 94,7% / 88,1% |
| BRACOT, detecção: recall no limiar de 0,35 | 84,0% (79,8% a 87,5%) | 88,3% (83,7% a 91,7%) |
| **Campo (fotos de teste):** S chamadas saudáveis | 89,2% (75,3% a 95,7%; 33 de 37) | novas: 80,5% |
| Campo: D chamadas saudáveis | 10,9% (4,7% a 23,0%; 5 de 46) | novas: 6,7% (2,6% a 15,9%) |
| Campo: D que chegam como doente / somem pelo t_d | 26,1% / 63,0% | novas: 36,7% / 56,7% |
| Campo: AUC de P(saudável) | 0,98 (0,95 a 0,99) | novas: 0,95 (0,89 a 0,99) |

- **Teste contra validação:** o teste fica abaixo da validação em quase tudo, mas os intervalos
  se sobrepõem.
- **Classificador sozinho:** fica perto da validação cruzada (89,9% por blocos e 90,7% pelo
  protocolo aleatório), que é a estimativa mais realista.
- **Falso saudável no campo:** o do teste é o maior dos conjuntos rotulados. De novo, os dados
  não sustentam o teto de 5% usado para fixar o limiar de saudável.
- **Relatórios e leituras:**
  - `model/runs/teste_bracol/`;
  - `model/runs/teste_bracot_deteccao/`;
  - `model/runs/teste_bracot_folhas/parte_a/` e `parte_b/`.
- **Rótulos de campo do teste:** as folhas de `data/manifests/campo_rotulado_teste.csv` foram
  rotuladas por Claude por leitura visual das pranchas cegas (só S e D evidentes; sem
  especialista; uma fazenda), aceitas pelo gestor em 14/10/2026, sem ver nenhuma predição. Ver
  `data/README.md`.
  - **Errata:** a mensagem do commit `1c59792` diz "rotuladas pelo gestor" por engano. O
    histórico não foi reescrito, e a errata está no log.
- **Limitação:** os rótulos de campo dos três conjuntos são de IA (leitura visual), não de
  especialista nem do gestor.
