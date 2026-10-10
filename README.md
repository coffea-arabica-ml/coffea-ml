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
