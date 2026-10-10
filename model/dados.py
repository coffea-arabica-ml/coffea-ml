"""
Dados da Frente 9: monta treino, validação e teste a partir dos manifests da Frente 8 e entrega
o Dataset PyTorch, as transformações (Albumentations) e a amostragem balanceada do RD02.

- Treino: o split treino do BRACOL e, só quando pedido (auxiliar="jmuben"), as imagens
  selecionadas do JMuBEN, que não têm alvo de severidade (-1).
- Validação e teste: só BRACOL, intactos: sem augmentation e sem amostragem (decisões do
  gestor, 09/10/2026).
- Rótulo de classe: bracol.CLASSES.index(classe), nunca o código predominant_stress.
- Augmentation só no treino e em memória: nada é gravado em disco.

Importado por model/train.py e pelos testes.
"""
import os
import sys
from collections import Counter
from pathlib import Path

# O import do Albumentations consultaria o PyPI atrás de versão nova, e o projeto não acessa a
# internet sem pedido explícito. Por isso a variável vem antes dos imports.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset, WeightedRandomSampler, get_worker_info

RAIZ_REPO = Path(__file__).resolve().parents[1]
if str(RAIZ_REPO / "data") not in sys.path:  # bracol e jmuben ficam em data/
    sys.path.insert(0, str(RAIZ_REPO / "data"))

import bracol

COLUNAS = ["fonte", "id", "caminho", "classe", "severity"]
SEM_SEVERIDADE = -1  # linha sem alvo de severidade (JMuBEN): fica fora da perda e das métricas
AUXILIARES = ("jmuben",)

# Entrada padrão 448x224 (largura x altura): a proporção 2:1 das fotos do BRACOL (2048x1024).
LARGURA_PADRAO = 448
ALTURA_PADRAO = 224
# Redução forte (2048 -> 448): INTER_AREA faz a média dos pixels e não serrilha. É a mesma em
# treino, validação e teste, e o backend precisa usar a mesma (vai no checkpoint).
INTERPOLACAO = cv2.INTER_AREA
NOME_INTERPOLACAO = "INTER_AREA"
MEDIA = tuple(IMAGENET_DEFAULT_MEAN)  # normalização do ImageNet, a do pré-treino
DESVIO = tuple(IMAGENET_DEFAULT_STD)
# Augmentation moderada do RD02 (só no treino).
ROTACAO_MAX = 15  # graus, para os dois lados
BRILHO_CONTRASTE_MAX = 0.1  # variação máxima de brilho e de contraste (10%)


# --------------------------------------------------------------------------- montagem
def rotulo(classe: str) -> int:
    """Índice da classe para o modelo: CLASSES.index(classe), nunca predominant_stress."""
    return bracol.CLASSES.index(classe)


def montar_treino(auxiliar: str | None = None, teto=None) -> pd.DataFrame:
    """Treino: o split treino do BRACOL e, com auxiliar="jmuben", as imagens selecionadas do
    JMuBEN (teto vai para jmuben.ler_manifest; None usa a seleção inteira).

    Colunas COLUNAS; severity = -1 nas linhas do JMuBEN, que não têm alvo de severidade.
    """
    if auxiliar is not None and auxiliar not in AUXILIARES:
        raise ValueError(f"fonte auxiliar desconhecida: {auxiliar!r} (esperado None ou um de "
                         f"{AUXILIARES})")
    if teto is not None and auxiliar is None:
        raise ValueError("teto só vale com uma fonte auxiliar")
    partes = [_padronizar(bracol.ler_manifest(split="treino"))]
    if auxiliar == "jmuben":
        import jmuben  # só quem usa a fonte auxiliar precisa dela

        auxiliares = jmuben.ler_manifest(teto=teto).assign(severity=SEM_SEVERIDADE)
        partes.append(_padronizar(auxiliares))
    return pd.concat(partes, ignore_index=True)


def montar_val() -> pd.DataFrame:
    """Validação: o split val do BRACOL, intacto."""
    return _padronizar(bracol.ler_manifest(split="val"))


def montar_teste() -> pd.DataFrame:
    """Teste: o split teste do BRACOL, intacto. Só para a avaliação final (--avaliar-teste)."""
    return _padronizar(bracol.ler_manifest(split="teste"))


def _padronizar(df) -> pd.DataFrame:
    """Só as COLUNAS, com id e severity inteiros; confere as classes e os níveis."""
    df = df[COLUNAS].astype({"id": "int64", "severity": "int64"}).reset_index(drop=True)
    desconhecidas = sorted(set(df["classe"]) - set(bracol.CLASSES))
    if desconhecidas:
        raise ValueError(f"classes fora de CLASSES: {desconhecidas}")
    invalidas = sorted(set(df["severity"]) - {SEM_SEVERIDADE, *bracol.SEVERIDADES})
    if invalidas:
        raise ValueError(f"níveis de severidade inválidos: {invalidas}")
    return df


def amostra_estratificada(df, n: int, seed: int = bracol.SEED_PADRAO) -> pd.DataFrame:
    """n linhas de df com a mesma proporção de classes (modo --rapido)."""
    from sklearn.model_selection import train_test_split

    amostra, _ = train_test_split(df, train_size=n, stratify=df["classe"], random_state=seed)
    return amostra.sort_values(["fonte", "id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- leitura
def ler_imagem(caminho: Path):
    """Imagem RGB (altura x largura x 3, uint8).

    Os bytes são lidos pelo numpy e decodificados com cv2.imdecode, porque no Windows o
    cv2.imread não abre caminho com acento. O resultado é o mesmo do cv2.imread, inclusive a
    orientação EXIF aplicada, que parte do JMuBEN usa (ver data/README.md). O backend também
    decodifica assim a foto que recebe em bytes.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"imagem não encontrada: {caminho}")
    imagem = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"não foi possível decodificar a imagem {caminho}: arquivo corrompido "
                         "ou formato desconhecido")
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


class FolhasDataset(Dataset):
    """Um único carregador para treino, validação e teste.

    df: colunas COLUNAS (de montar_treino, montar_val ou montar_teste); o caminho é relativo à
    raiz do repositório. Cada item é (imagem, classe, severidade): a imagem transformada, o
    índice CLASSES.index(classe) e o nível de severidade (-1 sem alvo).
    """

    def __init__(self, df, transform, raiz: Path = RAIZ_REPO):
        self.caminhos = [raiz / caminho for caminho in df["caminho"]]
        self.classes = torch.tensor([rotulo(c) for c in df["classe"]], dtype=torch.long)
        self.severidades = torch.tensor(df["severity"].to_numpy(dtype="int64"), dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.caminhos)

    def __getitem__(self, i):
        imagem = self.transform(image=ler_imagem(self.caminhos[i]))["image"]
        return imagem, self.classes[i], self.severidades[i]


# ---------------------------------------------------------------------- transformações
def transformacao_treino(largura: int = LARGURA_PADRAO, altura: int = ALTURA_PADRAO,
                         seed: int = bracol.SEED_PADRAO):
    """Treino, com a augmentation moderada do RD02: redimensiona, espelha na horizontal e na
    vertical, gira até ROTACAO_MAX graus e muda um pouco (BRILHO_CONTRASTE_MAX) o brilho e o
    contraste. Sem mudança forte de cor: o tom do fundo do BRACOL muda por bloco de ids
    (data/reports/eda_bracol.md, seção 8).

    O Albumentations 2 usa um gerador próprio, que não segue as sementes globais: a semente
    vai no Compose.
    """
    return A.Compose([
        _redimensionar(largura, altura),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Os cantos que a rotação descobre recebem o reflexo da imagem: o preto padrão não
        # aparece na validação.
        A.Rotate(limit=ROTACAO_MAX, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=BRILHO_CONTRASTE_MAX,
                                   contrast_limit=BRILHO_CONTRASTE_MAX, p=0.5),
        A.Normalize(mean=MEDIA, std=DESVIO),
        ToTensorV2(),
    ], seed=seed)


def transformacao_avaliacao(largura: int = LARGURA_PADRAO, altura: int = ALTURA_PADRAO):
    """Validação e teste: só redimensiona e normaliza."""
    return A.Compose([
        _redimensionar(largura, altura),
        A.Normalize(mean=MEDIA, std=DESVIO),
        ToTensorV2(),
    ])


def transformacao_sem_normalizar(largura: int = LARGURA_PADRAO, altura: int = ALTURA_PADRAO):
    """Só o redimensionamento da avaliação: devolve a imagem RGB uint8 (altura x largura x 3)
    que o modelo vê antes da normalização. Serve para medir cor e para mostrar figuras."""
    return A.Compose([_redimensionar(largura, altura)])


def _redimensionar(largura: int, altura: int):
    """O redimensionamento comum a todas as transformações (INTERPOLACAO)."""
    return A.Resize(altura, largura, interpolation=INTERPOLACAO)


def descrever_transformacao(transform) -> dict:
    """A transformação como dicionário (vai para o config.json do run)."""
    return A.to_dict(transform)


def semear_worker(_worker_id: int) -> None:
    """worker_init_fn do DataLoader, com workers > 0.

    Cada worker recebe uma cópia do Dataset com o mesmo gerador do Albumentations e repetiria
    os sorteios dos outros. Aqui ele passa a usar a semente do worker, que o PyTorch tira da
    semente global: difere por worker e por época e se repete entre execuções.
    """
    info = get_worker_info()
    info.dataset.transform.set_random_seed(info.seed % 2**32)


# ------------------------------------------------------------------ amostragem (RD02)
def pesos_por_amostra(classes) -> torch.Tensor:
    """Peso de cada imagem: 1 / (imagens da classe dela). Toda classe soma 1."""
    classes = list(classes)
    contagem = Counter(classes)
    return torch.tensor([1.0 / contagem[c] for c in classes], dtype=torch.double)


def criar_sampler(df, seed: int = bracol.SEED_PADRAO) -> WeightedRandomSampler:
    """Amostragem balanceada por classe do RD02, com os pesos calculados sobre o treino montado
    (com a fonte auxiliar, se houver). Sorteia len(df) imagens por época, com reposição."""
    return WeightedRandomSampler(
        pesos_por_amostra(df["classe"]),
        num_samples=len(df),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
