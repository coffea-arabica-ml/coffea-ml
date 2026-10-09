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

## Treino

`python model/train.py` treina o classificador (estresse e severidade) e grava cada execução em
`model/runs/<nome>/`. O uso está no docstring de `model/train.py`, e os comandos, no
`CLAUDE.md`.
