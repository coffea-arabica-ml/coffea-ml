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
para o backend. O teste do BRACOT fica fechado até o modelo final. O run de base, com as métricas
e a medida do RNF02, está em `model/runs/detector_base/relatorio.md`.

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
2. saem as detecções pequenas e os pedaços de fundo liso (filtro de cor);
3. as regras de folha única decidem se a foto inteira vira uma folha;
4. cada folha é recortada deitada, com margem, como nas fotos do BRACOL;
5. o classificador avalia todas as folhas num lote só;
6. as folhas abaixo do limiar de confiança saem.

A avaliação completa, com o tempo em CPU, está em `model/runs/pipeline_base/relatorio.md`.

```python
import sys
sys.path.insert(0, "coffea-ml/model")  # a pasta model/ do repositório
import inferencia

inferencia.carregar_modelos()                        # uma vez, ao subir o backend (CPU)
resposta = inferencia.diagnosticar_arquivo(conteudo)  # bytes do upload ou caminho da foto
```

**Pesos.** Ficam fora do git e são lidos de arquivos locais, sem rede:
- o `melhor.pt` do classificador, em `model/runs/base_resnet50_bracol/`;
- o `last.pt` do detector, em `model/runs/detector_base/weights/`.

Para usar outros: `carregar_modelos(pasta_do_classificador=..., pesos_do_detector=...,
dispositivo="cpu")`. Os modelos ficam em cache por caminho e dispositivo.

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
     "confianca": 0.74, "scoreDeteccao": 0.88, "caixa": [0.63, 0.52, 0.86, 0.81],
     "cortadaNaBorda": true}
  ],
  "detalhes": {"plano": "deteccao", "regra": null, "deteccoes": 7, "abaixo_do_limiar": 4,
               "cortadas_por_area": 0, "cortadas_por_cor": 0, "cortadas_por_limite": 0,
               "abaixo_da_confianca": 5, "recorte": "A", "limiar_confianca": 0.61,
               "classificador": "base_resnet50_bracol", "detector": "detector_base"}
}
```

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
- **Extras:** `confianca` (probabilidade da classe prevista), `scoreDeteccao` (nulo sem detecção),
  `caixa` (relativa, [x0, y0, x1, y1]), `cortadaNaBorda` e `detalhes`. São campos extras, e o
  normalizador do frontend os ignora.
- **imagemUrl:** fica com o backend, que guarda a foto. Sem ela, o frontend usa a foto local.

**Erros** (`{"status": "erro", "tipo": ..., "mensagem": ...}`, às vezes com `detalhes`):

| tipo | quando | mensagem |
|---|---|---|
| `planta_nao_identificada` | nenhuma folha utilizável, e nenhuma detecção com cor de folha e score >= 0,05 | "Não encontramos folhas de café na foto. Fotografe a planta ou a folha mais de perto, com boa luz." |
| `baixa_confianca` | todas as folhas abaixo do limiar de confiança (RF07) | "Encontramos folhas, mas o diagnóstico não ficou confiável. Tente outra foto, mais nítida e com boa luz." |
| `formato_invalido` | o arquivo não decodifica (JPG ou PNG) | "Não foi possível ler a imagem. Envie uma foto em JPG ou PNG." |
| `arquivo_muito_grande` | mais de 64 megapixels, lidos do cabeçalho com Pillow, sem decodificar | "A imagem tem 9000 x 8000 pixels (72,0 megapixels); o limite é 64,0 megapixels." |

- **`especie_incorreta`:** não é produzido. Não há dados de outras plantas para treinar essa recusa,
  então uma foto de outra planta pode sair como folha de café.
- **Limite de 10 MB:** o limite de tamanho do arquivo é do backend.

**Parâmetros** (`inferencia.Configuracao`, uma dataclass congelada; `diagnosticar(imagem, config)`):

| parâmetro | padrão | o que faz |
|---|---|---|
| `limiar_deteccao` | 0,35 | score mínimo do detector (decisão do gestor, 10/10/2026) |
| `max_folhas` | 15 | folhas classificadas por foto, as de maior score (RNF02) |
| `area_minima` | 0,5% | área mínima da máscara, em fração da foto |
| `saturacao_minima`, `fracao_colorida_minima` | 30, 25% | filtro de cor: tira os pedaços de fundo liso |
| `limiar_confianca` | 0,61 | RF07; **proposta, aguarda a decisão do gestor** |
| `recorte` | "A" | "A": a foto em volta da folha; "B": fundo fora da máscara em cinza |
| `margem`, `cinza_fundo` | 0,17, 197 | medidos no treino do BRACOL |
| `cobertura_folha_unica` | 30% | regra 1: uma folha cuja caixa cobre isso da foto → a foto inteira |
| `score_minimo_folha_unica` | 0,05 | regra 2 (decisão do gestor, 10/10/2026) |
| `max_megapixels` | 64 | acima disso, `arquivo_muito_grande` |

**Limite atual:**
- O classificador foi treinado só com folhas do BRACOL (fundo claro). No BRACOL, a cadeia acerta
  tanto quanto a imagem inteira: 94,4% contra 94,8%.
- Nas folhas do campo, porém, ele puxa quase tudo para doença, sobretudo ferrugem. O T2 está no
  relatório do `pipeline_base`.
- Antes de usar em campo, o classificador precisa ser retreinado.
