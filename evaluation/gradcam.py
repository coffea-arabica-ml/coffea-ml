"""
Mapa de explicabilidade Grad-CAM (RF04) da cabeça de classe do modelo multitarefa
(model/train.py), com o pytorch-grad-cam. Usado pela checagem de atalho
(evaluation/atalho.py); a Frente 5 (frontend) e a Frente 6 (backend) poderão consumi-lo.
"""
import numpy as np
import torch
from torch import nn


class _SoClasse(nn.Module):
    """Expõe só os logits de classe do modelo multitarefa: o Grad-CAM espera uma saída só."""

    def __init__(self, modelo: nn.Module):
        super().__init__()
        self.modelo = modelo

    def forward(self, x):
        return self.modelo(x)[0]


def gerar_gradcam(modelo, imagem: torch.Tensor, alvo: int | None = None) -> np.ndarray:
    """Mapa Grad-CAM de uma imagem para a cabeça de CLASSE.

    modelo: ModeloMultitarefa (model/train.py) com backbone ResNet do timm. A camada-alvo é o
        último bloco do layer4. O modelo fica em eval, e o cálculo é em float32, sem AMP.
    imagem: o tensor normalizado (3, H, W) ou (1, 3, H, W), como sai de
        dados.transformacao_avaliacao; vai para o dispositivo do modelo.
    alvo: índice da classe em CLASSES. None (padrão) usa a classe prevista pelo modelo.

    Devolve o mapa (H, W) em float32, do tamanho da entrada, com valores de 0 a 1. Ele é
    normalizado por imagem (o maior valor vira 1): mostra onde o modelo olhou, não quanto.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if imagem.dim() == 3:
        imagem = imagem.unsqueeze(0)
    if imagem.dim() != 4 or imagem.shape[0] != 1:
        raise ValueError(f"esperada uma imagem (3, H, W) ou (1, 3, H, W): {tuple(imagem.shape)}")
    modelo.eval()
    alvos = None if alvo is None else [ClassifierOutputTarget(int(alvo))]
    with torch.enable_grad(), GradCAM(model=_SoClasse(modelo),
                                      target_layers=[modelo.backbone.layer4[-1]]) as cam:
        mapa = cam(input_tensor=imagem.float(), targets=alvos)
    return mapa[0].astype(np.float32)
