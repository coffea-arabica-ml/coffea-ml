"""
Script de treinamento por aprendizado por transferência.

TODO (Frente 9): implementar o loop de treino real, usando um backbone
pré-treinado carregado via timm (ex.: resnet50, efficientnet_b0).
"""

import timm


def criar_modelo(nome_backbone: str = "resnet50", num_classes: int = 5):
    return timm.create_model(nome_backbone, pretrained=True, num_classes=num_classes)


def treinar():
    raise NotImplementedError("Implementar o loop de treino na Frente 9.")


if __name__ == "__main__":
    treinar()