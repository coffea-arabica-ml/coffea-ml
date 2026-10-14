# Pesos finais da cadeia (passo 6)

Os pesos ficaram congelados por decisão do gestor (11/10/2026): o classificador
`base_resnet50_bracol` e o detector `detector_base`, sem retreino final. Os testes fechados (BRACOL
e BRACOT) medem estes pesos uma única vez, no fim. Os arquivos ficam fora do git (`.gitignore`:
`model/runs/**/*.pt` e `model/pesos/*`); só este manifesto é versionado.

| arquivo | bytes | SHA-256 | o que é |
|---|---:|---|---|
| `model/runs/base_resnet50_bracol/melhor.pt` | 94.438.195 | `0d7c043ab2907e186b5c4f99fa09a6a62120687f862c91404b5bafee9b66066a` | classificador: ResNet-50 multitarefa (5 classes e 5 níveis de severidade), entrada 448 x 224; época 14, a de melhor F1 macro na validação do BRACOL (run de 09/10/2026) |
| `model/runs/detector_base/weights/last.pt` | 20.504.100 | `0532766bf0d2c011a79ead110b217a2a79a8f252cc17a69c6383260de6c4dc21` | detector: YOLO11s-seg (ultralytics 8.4.175) ajustado no BRACOT, 60 épocas fixas; a última época é o modelo oficial do run (10/10/2026) |

## Como carregar

`inferencia.carregar_modelos()` usa esses dois caminhos por padrão (`inferencia.PASTA_CLASSIFICADOR` e
`detector.PESOS_PADRAO`), só de arquivos locais e sem rede. Com os pesos em outro lugar:

```python
inferencia.carregar_modelos(pasta_do_classificador="PASTA_COM_O_melhor.pt",
                            pesos_do_detector="CAMINHO_DO_last.pt", dispositivo="cpu")
```

O classificador é montado sem pré-treino e lê o `melhor.pt` com `torch.load(weights_only=True)`.
O `carregar_modelo` confere as classes e o pré-processamento gravados no checkpoint.

## Conferência

Antes de subir o backend, confira o SHA-256 de cada arquivo com a tabela. O mesmo comando vale no
Windows, no Linux e no Colab:

```
python -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" model/runs/base_resnet50_bracol/melhor.pt
```

O `tests/test_cadeia_final.py` faz a mesma conferência quando os pesos estão na máquina, e pula
quando não estão.

## Arquivos que não são pesos da cadeia

- `model/runs/detector_base/weights/best.pt` (20.504.100 bytes, SHA-256
  `5ab006e9f0192192bbc62b599ac0d2e2a83b3225f14a0e31968c39e768a406f2`): a época de melhor métrica
  do ultralytics. O modelo oficial do run é o `last.pt`.
- `model/pesos/yolo11s-seg.pt` (20.669.228 bytes): o pré-treino COCO do detector, só para treinar.
  O SHA-256 dele está em `detector.SHA256_PESO_PRETREINADO`.
- O pré-treino ImageNet da ResNet-50 (cache do Hugging Face, pelo timm) só entra no treino.

## Versões

Os dois runs foram gerados com Python 3.13.14, torch 2.13.0+cu130, timm 1.0.28, albumentations
2.0.8, OpenCV 5.0.0, numpy 2.5.2, pandas 3.0.5 e ultralytics 8.4.175 (`config.json` de cada run).
Para carregar, instale o `requirements.txt` e depois `pip install --no-deps ultralytics==8.4.175`
(README, seção "Ambiente"). Em CPU, o build do torch do PyPI serve; os pesos não dependem do CUDA.

Licença do detector: AGPL-3.0 (ultralytics e pesos YOLO; aceita pelo gestor em 10/10/2026).
