"""
Classificador robusto ao campo (Frente 9, passo 4): máscaras das folhas do BRACOL, banco de
fundos, variantes de augmentation, triagem na validação oficial e validação cruzada nas dobras
do diagnóstico por blocos (protocolo aleatório).

Variantes (todas com a receita do run base: resnet50 pré-treinado, AdamW, lr 1e-4, cosseno, batch
32, AMP, amostragem balanceada, semente 42, 20 épocas fixas e o modelo da última época):
- v0: a augmentation atual (dados.transformacao_treino).
- v1: augmentation de campo, fotométrica (cor, brilho, sombra, desfoque) e geométrica (corte da
  folha pela borda, limitado a ~15% da folha, e erasing pequeno e raro). Nas imagens cortadas ou
  apagadas, a severidade vira -1 (fora da perda). v1f: só a fotométrica; v1g: só a geométrica.
- v2: v1 com troca do fundo pela máscara da folha em parte das imagens.
- v3: v2 com o JMuBEN (teto do README); v3s: sem phoma e cercosporiose do JMuBEN
  (RESSALVA_PHOMA_CERCOSPORA).

Decisões do gestor (10/10/2026): o JMuBEN entra na comparação, só no treino; regra de adoção:
não piorar o BRACOL (F1 macro na validação cruzada, queda de até ~2 pontos) e melhorar o campo;
uso novo do BRACOT como fundo: só as fotos de treino, nunca as de validação do detector nem o
teste, trechos fora dos polígonos anotados, borrão forte, sorteio independente da classe.

Os testes do BRACOL e do BRACOT ficam fechados. Máscaras, fundos e pranchas ficam em
data/processed/ (fora do git); os pesos, fora do git.

Uso (de qualquer pasta, com o venv ativo):
    python model/treinar_campo.py --mascaras           máscaras do BRACOL e prancha de exemplos
    python model/treinar_campo.py --fundos             banco de fundos e prancha
    python model/treinar_campo.py --triagem v0 v1      treino 1.180 -> validação 252 (campo_<v>)
    python model/treinar_campo.py --cv v1              5 dobras (campo_<v>/cv/)
    python model/treinar_campo.py --comparar           model/runs/campo_comparacao/relatorio.md
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import albumentations as A  # noqa: E402  (dados.py já definiu NO_ALBUMENTATIONS_UPDATE)
from albumentations.pytorch import ToTensorV2  # noqa: E402

import bracol  # noqa: E402
import bracot  # noqa: E402
import dados  # noqa: E402
import formatacao as fmt  # noqa: E402
import inferencia as inf  # noqa: E402
import metrics  # noqa: E402
import train  # noqa: E402
import treinar_detector as td  # noqa: E402

PASTA_MASCARAS = RAIZ_REPO / "data" / "processed" / "campo_mascaras"  # fora do git
PASTA_FUNDOS = RAIZ_REPO / "data" / "processed" / "campo_fundos"  # fora do git
PASTA_DIAGNOSTICO = train.PASTA_RUNS / "diagnostico_blocos"
PASTA_COMPARACAO = train.PASTA_RUNS / "campo_comparacao"
RUN_BASE = "base_resnet50_bracol"
LARGURA, ALTURA = dados.LARGURA_PADRAO, dados.ALTURA_PADRAO
EPOCAS = 20
SEED = bracol.SEED_PADRAO
K = 5
LADO_SEGMENTACAO = 896  # lado maior da imagem em que se segmenta a folha do BRACOL
IOU_CONCORDANCIA = 0.8  # máscara do detector usada se concordar com a segmentação simples
AREA_PLAUSIVEL = (0.10, 0.80)  # fração da foto ocupada por uma folha do BRACOL
DILATACAO_FUNDO = 0.01  # da largura: a folha não perde a borda na troca de fundo
FUNDOS_POR_FOTO = 3
SIGMA_BORRAO = 0.04  # da largura do trecho: borrão forte, sem forma de folha
DILATACAO_ANOTADA = 0.05  # do lado maior da foto: margem em volta dos polígonos anotados
P_FUNDO = 0.6  # fração das imagens com fundo trocado (v2, v3, v3s)
P_CORTE = 0.3
CORTE_MAXIMO = 0.15  # fração da folha que sai do quadro, no máximo
P_APAGAR = 0.2
APAGAR_AREA_MAXIMA = 0.04  # de cada buraco, em fração da imagem
APAGAR_BURACOS = 2
CINZA = inf.Configuracao().cinza_fundo  # o cinza do que fica fora da foto na inferência
VARIANTES = {
    "v0": {"fotometrica": False, "corte": False, "apagar": False, "fundo": False,
           "jmuben": None},
    "v1": {"fotometrica": True, "corte": True, "apagar": True, "fundo": False, "jmuben": None},
    "v1f": {"fotometrica": True, "corte": False, "apagar": False, "fundo": False,
            "jmuben": None},
    "v1g": {"fotometrica": False, "corte": True, "apagar": True, "fundo": False,
            "jmuben": None},
    "v2": {"fotometrica": True, "corte": True, "apagar": True, "fundo": True, "jmuben": None},
    "v3": {"fotometrica": True, "corte": True, "apagar": True, "fundo": True,
           "jmuben": "todas"},
    "v3s": {"fotometrica": True, "corte": True, "apagar": True, "fundo": True,
            "jmuben": "sem_phoma_cercosporiose"},
}
DESCRICAO = {
    "v0": "augmentation atual",
    "v1": "campo: fotométrica e geométrica",
    "v1f": "campo: só fotométrica",
    "v1g": "campo: só geométrica (corte e erasing)",
    "v2": "v1 + troca de fundo",
    "v3": "v2 + JMuBEN",
    "v3s": "v2 + JMuBEN sem phoma e cercosporiose",
}
TETO_SEM_PHOMA_CERCOSPORIOSE = {"phoma": 0, "cercosporiose": 0}
CREDITO_BRACOL = ("Imagens: BRACOL (Krohling, Esgario e Ventura, 2019), CC BY 4.0, "
                  "DOI 10.17632/yy2k5y8mxg.1")
CREDITO_BRACOT = ("Fotos: BRACOT (Krohling, Tozzi de Souza e Tassis, 2021), CC BY 4.0, "
                  "DOI 10.17632/pmkbyjpf6k.1")


# --------------------------------------------------------------------------- máscaras
def segmentar_fundo_liso(rgb) -> np.ndarray:
    """Máscara (bool) da folha numa foto de folha sobre fundo liso (as do BRACOL): Otsu no canal
    de saturação, fechamento, maior componente e furos preenchidos."""
    saturacao = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1], (5, 5), 0)
    _, binaria = cv2.threshold(saturacao, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raio = max(1, round(0.01 * max(rgb.shape[:2])))
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, nucleo)
    n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if n <= 1:
        return np.zeros(rgb.shape[:2], dtype=bool)
    maior = 1 + int(np.argmax(estatisticas[1:, cv2.CC_STAT_AREA]))
    mascara = (rotulos == maior).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cheia = np.zeros_like(mascara)
    cv2.drawContours(cheia, contornos, -1, 1, thickness=cv2.FILLED)
    return cheia.astype(bool)


def iou(a, b) -> float:
    uniao = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / uniao) if uniao else 0.0


def escolher_mascara(do_detector, simples) -> tuple[np.ndarray | None, str, float]:
    """A máscara de treino: se a do detector concordar com a segmentação simples (IoU >=
    IOU_CONCORDANCIA), a união das duas, porque a do detector corta a folha com retas (a caixa)
    e às vezes perde a ponta ou a borda com lesão, e a troca de fundo apagaria a lesão; senão a
    simples, se plausível (área em AREA_PLAUSIVEL); senão nenhuma. Devolve (máscara, regra,
    IoU)."""
    concordancia = iou(do_detector, simples) if do_detector is not None else 0.0
    if do_detector is not None and concordancia >= IOU_CONCORDANCIA:
        return do_detector | simples, "uniao", concordancia
    if AREA_PLAUSIVEL[0] <= simples.mean() <= AREA_PLAUSIVEL[1]:
        return simples, "simples", concordancia
    return None, "nenhuma", concordancia


def mascara_do_detector(imagem, modelos, config, altura: int, largura: int):
    """União, na resolução (altura, largura), das detecções utilizáveis da cadeia (limiar,
    área e cor); None sem nenhuma."""
    escala = largura / imagem.shape[1]
    uteis = [d for d in inf.detectar(imagem, config, modelos) if inf.utilizavel(d, config)]
    if not uteis:
        return None
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    for d in uteis:
        cv2.fillPoly(mascara, [np.round(np.asarray(d["poligono"]) * escala).astype(np.int32)], 1)
    return mascara.astype(bool)


def gerar_mascaras(modelos) -> pd.DataFrame:
    """Máscaras (PNG 448x224) das imagens de treino e validação do BRACOL, nunca do teste, em
    PASTA_MASCARAS, com mascaras.csv e uma prancha de exemplos."""
    PASTA_MASCARAS.mkdir(parents=True, exist_ok=True)
    config = inf.Configuracao()
    pool = pd.concat([dados.montar_treino(), dados.montar_val()], ignore_index=True)
    sorteados = set(pool["id"].sample(80, random_state=SEED))  # candidatos da prancha
    linhas, exemplos = [], []
    for n, linha in enumerate(pool.itertuples(), start=1):
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        trabalho, _ = inf.reduzir(imagem, LADO_SEGMENTACAO)
        altura, largura = trabalho.shape[:2]
        simples = segmentar_fundo_liso(trabalho)
        do_detector = mascara_do_detector(imagem, modelos, config, altura, largura)
        mascara, regra, concordancia = escolher_mascara(do_detector, simples)
        if mascara is not None:
            pequena = cv2.resize(mascara.astype(np.uint8) * 255, (LARGURA, ALTURA),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imencode(".png", pequena)[1].tofile(PASTA_MASCARAS / f"bracol-{linha.id}.png")
        linhas.append({"id": linha.id, "classe": linha.classe, "regra": regra,
                       "iou_detector_simples": round(concordancia, 4),
                       "area": round(float(mascara.mean()), 4) if mascara is not None else None})
        if linha.id in sorteados:
            exemplos.append((linha.id, regra, concordancia, trabalho, do_detector, simples,
                             mascara))
        if n % 200 == 0:
            fmt.dizer(f"  mascaras: {n}/{len(pool)}")
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(PASTA_MASCARAS / "mascaras.csv", index=False, lineterminator="\n")
    _prancha_de_mascaras(exemplos, PASTA_MASCARAS / "prancha_mascaras.jpg")
    return tabela


def _prancha_de_mascaras(exemplos, caminho: Path) -> None:
    """12 exemplos (4 por regra, sorteados): a foto, a máscara do detector (vermelho), a simples
    (azul) e a escolhida (contorno verde)."""
    gerador = np.random.default_rng(SEED)
    escolhidos = []
    for regra in ("uniao", "simples", "nenhuma"):
        da_regra = [e for e in exemplos if e[1] == regra]
        for i in gerador.permutation(len(da_regra))[:6]:
            escolhidos.append(da_regra[i])
    cartoes = []
    for ident, regra, concordancia, trabalho, do_detector, simples, mascara in escolhidos:
        img = cv2.resize(trabalho, (448, 224), interpolation=cv2.INTER_AREA).copy()
        camada = img.copy()
        for m, cor in ((do_detector, (230, 40, 40)), (simples, (40, 90, 230))):
            if m is not None:
                pequena = cv2.resize(m.astype(np.uint8), (448, 224),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                camada[pequena] = (0.5 * camada[pequena] + 0.5 * np.array(cor)).astype(np.uint8)
        if mascara is not None:
            pequena = cv2.resize(mascara.astype(np.uint8), (448, 224),
                                 interpolation=cv2.INTER_NEAREST)
            contornos, _ = cv2.findContours(pequena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(camada, contornos, -1, (40, 200, 40), 2)
        cartao = np.full((254, 904, 3), 255, np.uint8)
        cartao[30:254, 0:448] = img
        cartao[30:254, 456:904] = camada
        cv2.putText(cartao, f"bracol {ident}: regra {regra}, IoU detector x simples "
                            f"{concordancia:.2f}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cartoes.append(cartao)
    legenda = ("Mascaras: vermelho = detector (deteccoes com cor), azul = segmentacao simples, "
               "verde = a usada no treino (uniao se concordam)")
    _gravar_prancha(cartoes, legenda, CREDITO_BRACOL, caminho)


def ler_mascaras() -> dict[int, np.ndarray]:
    """{id do BRACOL: máscara bool 224x448} das imagens que têm máscara."""
    tabela = pd.read_csv(PASTA_MASCARAS / "mascaras.csv")
    mascaras = {}
    for i in tabela.loc[tabela["regra"] != "nenhuma", "id"]:
        png = cv2.imdecode(np.fromfile(PASTA_MASCARAS / f"bracol-{i}.png", np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        mascaras[int(i)] = png > 127
    return mascaras


# ----------------------------------------------------------------------------- fundos
def fotos_de_fundo() -> pd.DataFrame:
    """As fotos de treino do BRACOT fora das cenas de validação do detector (nunca o teste)."""
    manifest = bracot.ler_manifest(split="treino")
    validacao = set(td.dividir_validacao(manifest))
    treino = manifest[~manifest["id"].isin(validacao)]
    cenas_val = set(manifest.loc[manifest["id"].isin(validacao), "cena"])
    if set(treino["cena"]) & cenas_val:
        raise train.ErroFatal("uma cena de treino do BRACOT aparece na validação do detector")
    return treino.reset_index(drop=True)


def gerar_fundos() -> pd.DataFrame:
    """FUNDOS_POR_FOTO trechos 2:1 de cada foto de treino do BRACOT, os que menos tocam os
    polígonos anotados (dilatados), com borrão forte, em PASTA_FUNDOS (448x224)."""
    PASTA_FUNDOS.mkdir(parents=True, exist_ok=True)
    gerador = np.random.default_rng(SEED)
    poligonos = td.ler_poligonos("treino")
    linhas = []
    for linha in fotos_de_fundo().itertuples():
        imagem, escala = inf.reduzir(dados.ler_imagem(RAIZ_REPO / linha.caminho), 1024)
        altura, largura = imagem.shape[:2]
        anotada = np.zeros((altura, largura), np.uint8)
        for p in poligonos[linha.id]:
            cv2.fillPoly(anotada, [np.round(np.asarray(p) * escala).astype(np.int32)], 1)
        raio = round(DILATACAO_ANOTADA * max(altura, largura))
        anotada = cv2.dilate(anotada, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                (2 * raio + 1, 2 * raio + 1)))
        candidatos = []
        for _ in range(40):
            w = int(gerador.uniform(0.3, 0.6) * largura)
            h = w // 2
            x, y = int(gerador.integers(0, largura - w)), int(gerador.integers(0, altura - h))
            candidatos.append((float(anotada[y:y + h, x:x + w].mean()), x, y, w, h))
        for k, (tocada, x, y, w, h) in enumerate(sorted(candidatos)[:FUNDOS_POR_FOTO]):
            trecho = cv2.resize(imagem[y:y + h, x:x + w], (LARGURA, ALTURA),
                                interpolation=cv2.INTER_AREA)
            trecho = cv2.GaussianBlur(trecho, (0, 0), SIGMA_BORRAO * LARGURA)
            nome = f"{linha.id}_{k}.jpg"
            _gravar_jpeg(trecho, PASTA_FUNDOS / nome)
            linhas.append({"arquivo": nome, "foto": linha.id, "cena": linha.cena,
                           "fracao_anotada": round(tocada, 4)})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(PASTA_FUNDOS / "fundos.csv", index=False, lineterminator="\n")
    amostra = tabela.sample(20, random_state=SEED)
    cartoes = []
    for r in amostra.itertuples():
        img = _ler_rgb(PASTA_FUNDOS / r.arquivo)
        cartao = np.full((254, 448, 3), 255, np.uint8)
        cartao[30:, :] = img
        cv2.putText(cartao, f"{r.foto} (anotado no trecho: {100 * r.fracao_anotada:.0f}%)",
                    (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cartoes.append(cartao)
    _gravar_prancha(cartoes, "Fundos borrados das fotos de treino do BRACOT (20 sorteados)",
                    CREDITO_BRACOT, PASTA_FUNDOS / "prancha_fundos.jpg", colunas=4)
    return tabela


def ler_fundos() -> list[np.ndarray]:
    tabela = pd.read_csv(PASTA_FUNDOS / "fundos.csv")
    return [_ler_rgb(PASTA_FUNDOS / a) for a in tabela["arquivo"]]


def fundo_aleatorio(gerador, banco) -> np.ndarray:
    """Um fundo 224x448, sorteado sem olhar a imagem nem a classe: do banco do BRACOT (40%),
    cinza neutro (30%, metade no cinza do modo B), cor lisa (15%) ou ruído colorido (15%)."""
    sorteio = gerador.random()
    if banco and sorteio < 0.4:
        fundo = banco[int(gerador.integers(len(banco)))]
        return fundo[:, ::-1].copy() if gerador.random() < 0.5 else fundo.copy()
    if sorteio < 0.7:
        nivel = CINZA if gerador.random() < 0.5 else int(gerador.integers(110, 236))
        return np.full((ALTURA, LARGURA, 3), nivel, np.uint8)
    cor = gerador.integers(30, 226, size=3)
    if sorteio < 0.85:
        return np.broadcast_to(cor, (ALTURA, LARGURA, 3)).astype(np.uint8).copy()
    ruido = gerador.normal(0, 25, (ALTURA // 4, LARGURA // 4, 3)) + cor
    ruido = cv2.resize(np.clip(ruido, 0, 255).astype(np.uint8), (LARGURA, ALTURA),
                       interpolation=cv2.INTER_LINEAR)
    return ruido


def trocar_fundo(imagem, mascara, fundo) -> np.ndarray:
    """A imagem com o fundo trocado fora da máscara (dilatada em DILATACAO_FUNDO)."""
    raio = max(1, round(DILATACAO_FUNDO * imagem.shape[1]))
    dilatada = cv2.dilate(mascara.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1))).astype(bool)
    saida = imagem.copy()
    saida[~dilatada] = fundo[~dilatada]
    return saida


def completar_2_por_1(quadrado, fundo) -> np.ndarray:
    """Um recorte (do JMuBEN) com a altura levada a ALTURA, sem distorcer, centrado sobre o
    fundo 2:1."""
    h, w = quadrado.shape[:2]
    escala = ALTURA / h
    novo_w = min(LARGURA, max(1, round(w * escala)))
    reduzido = cv2.resize(quadrado, (novo_w, ALTURA), interpolation=cv2.INTER_AREA
                          if escala < 1 else cv2.INTER_LINEAR)
    saida = fundo.copy()
    x = (LARGURA - novo_w) // 2
    saida[:, x:x + novo_w] = reduzido
    return saida


def cortar_pela_borda(imagem, mascara, gerador) -> np.ndarray | None:
    """Corte reto da folha, como a de uma foto que corta a folha: tudo além de uma linha
    vertical ou horizontal vira CINZA (o que fica fora da foto na inferência). A linha deixa de
    fora de 5% a CORTE_MAXIMO dos pixels da folha (sem máscara, da imagem). None se não cortar."""
    tem_mascara = mascara is not None and mascara.any()
    ys, xs = np.nonzero(mascara if tem_mascara else np.ones(imagem.shape[:2], dtype=bool))
    fracao = gerador.uniform(0.05, CORTE_MAXIMO)
    eixo, lado = int(gerador.integers(2)), int(gerador.integers(2))
    valores = xs if eixo == 0 else ys
    linha = int(np.quantile(valores, 1 - fracao if lado == 1 else fracao))
    saida = imagem.copy()
    if eixo == 0:
        if lado == 1:
            saida[:, linha:] = CINZA
        else:
            saida[:, :linha] = CINZA
    elif lado == 1:
        saida[linha:, :] = CINZA
    else:
        saida[:linha, :] = CINZA
    return saida


def apagar(imagem, gerador) -> np.ndarray:
    """Erasing pequeno: 1 a APAGAR_BURACOS retângulos de até APAGAR_AREA_MAXIMA da imagem cada,
    com cinza ou ruído."""
    saida = imagem.copy()
    altura, largura = imagem.shape[:2]
    for _ in range(int(gerador.integers(1, APAGAR_BURACOS + 1))):
        area = gerador.uniform(0.01, APAGAR_AREA_MAXIMA) * altura * largura
        proporcao = gerador.uniform(0.5, 2.0)
        h = int(min(altura, math.sqrt(area / proporcao)))
        w = int(min(largura, area / max(h, 1)))
        y, x = int(gerador.integers(0, altura - h + 1)), int(gerador.integers(0, largura - w + 1))
        saida[y:y + h, x:x + w] = (gerador.integers(0, 256, (h, w, 3)) if gerador.random() < 0.5
                                   else int(gerador.integers(60, 230)))
    return saida


# ----------------------------------------------------------------------------- dataset
def montar_dados(variante: str) -> pd.DataFrame:
    """Treino + validação do BRACOL (o pool da validação cruzada) e, conforme a variante, o
    JMuBEN (severity -1), com a coluna "fonte"."""
    pool = pd.concat([dados.montar_treino(), dados.montar_val()], ignore_index=True)
    jmuben = VARIANTES[variante]["jmuben"]
    if jmuben:
        import jmuben as fonte_jmuben

        teto = TETO_SEM_PHOMA_CERCOSPORIOSE if jmuben == "sem_phoma_cercosporiose" else None
        auxiliar = fonte_jmuben.ler_manifest(teto=teto).assign(severity=dados.SEM_SEVERIDADE)
        pool = pd.concat([pool, dados._padronizar(auxiliar)], ignore_index=True)
    return pool


class Cache:
    """As imagens já em 448x224 (INTER_AREA, como o resize do treino), uma vez por processo; as
    do JMuBEN, com a altura em 224, sem distorcer."""

    def __init__(self):
        self.imagens = {}

    def obter(self, fonte: str, ident: int, caminho: str) -> np.ndarray:
        chave = (fonte, int(ident))
        if chave not in self.imagens:
            rgb = dados.ler_imagem(RAIZ_REPO / caminho)
            if fonte == "jmuben":
                escala = ALTURA / rgb.shape[0]
                w = min(LARGURA, max(1, round(rgb.shape[1] * escala)))
                self.imagens[chave] = cv2.resize(rgb, (w, ALTURA), interpolation=cv2.INTER_AREA
                                                 if escala < 1 else cv2.INTER_LINEAR)
            else:
                self.imagens[chave] = cv2.resize(rgb, (LARGURA, ALTURA),
                                                 interpolation=dados.INTERPOLACAO)
        return self.imagens[chave]


def transformacao_campo(variante: str) -> A.Compose:
    """A parte do Albumentations, sem normalizar: a do v0 é a atual (dados.transformacao_treino,
    sem Normalize e ToTensor); as de campo têm flips, ±15 graus e, se fotométrica, cor, brilho,
    sombra e desfoque mais fortes."""
    if not VARIANTES[variante]["fotometrica"]:
        atual = dados.transformacao_treino(LARGURA, ALTURA, SEED)
        return A.Compose(list(atual.transforms)[:-2], seed=SEED)
    return A.Compose([
        A.Resize(ALTURA, LARGURA, interpolation=dados.INTERPOLACAO),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=dados.ROTACAO_MAX, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=25, val_shift_limit=0, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2),
                       shadow_intensity_range=(0.3, 0.7), p=0.3),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 7))], p=0.3),
    ], seed=SEED)


FINAL = A.Compose([A.Normalize(mean=dados.MEDIA, std=dados.DESVIO), ToTensorV2()])


class DatasetCampo(Dataset):
    """(imagem, classe, severidade) com a augmentation da variante; treino=False só normaliza.
    Os passos próprios (fundo, corte, apagar) usam um gerador com a semente; nas imagens
    cortadas ou apagadas, a severidade vira -1."""

    def __init__(self, df, variante: str, cache: Cache, mascaras=None, banco=None,
                 treino: bool = True):
        self.df = df.reset_index(drop=True)
        self.cfg = VARIANTES[variante]
        self.cache, self.mascaras, self.banco = cache, mascaras or {}, banco or []
        self.treino = treino
        self.aug = transformacao_campo(variante) if treino else None
        self.gerador = np.random.default_rng(SEED)
        self.classes = [dados.rotulo(c) for c in self.df["classe"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        linha = self.df.iloc[i]
        imagem = self.cache.obter(linha["fonte"], linha["id"], linha["caminho"])
        severidade = int(linha["severity"])
        mascara = self.mascaras.get(int(linha["id"])) if linha["fonte"] == "bracol" else None
        if linha["fonte"] == "jmuben":
            imagem = completar_2_por_1(imagem, fundo_aleatorio(self.gerador, self.banco)
                                       if self.treino else
                                       np.full((ALTURA, LARGURA, 3), CINZA, np.uint8))
        if self.treino:
            g = self.gerador
            if self.cfg["fundo"] and mascara is not None and g.random() < P_FUNDO:
                imagem = trocar_fundo(imagem, mascara, fundo_aleatorio(g, self.banco))
            if self.cfg["corte"] and linha["fonte"] == "bracol" and g.random() < P_CORTE:
                imagem = cortar_pela_borda(imagem, mascara, g)
                severidade = dados.SEM_SEVERIDADE
            imagem = self.aug(image=imagem)["image"]
            if self.cfg["apagar"] and g.random() < P_APAGAR:
                imagem = apagar(imagem, g)
                severidade = dados.SEM_SEVERIDADE
        return FINAL(image=imagem)["image"], self.classes[i], severidade


# ----------------------------------------------------------------------------- treino
def treinar(variante: str, treino_df, fora_df, cache: Cache, mascaras, banco, dispositivo,
            avaliar_cada_epoca: bool = True):
    """EPOCAS épocas fixas com a receita do run base; devolve (modelo, avaliação final no fora,
    histórico)."""
    train.semear(SEED)
    amp = dispositivo.type == "cuda"
    loader = DataLoader(DatasetCampo(treino_df, variante, cache, mascaras, banco), batch_size=32,
                        sampler=dados.criar_sampler(treino_df, SEED), num_workers=0,
                        pin_memory=amp)
    loader_fora = DataLoader(DatasetCampo(fora_df, variante, cache, treino=False), batch_size=32,
                             shuffle=False, num_workers=0, pin_memory=amp)
    modelo = train.criar_modelo("resnet50", pretrained=True).to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-4)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=EPOCAS)
    escalador = torch.amp.GradScaler("cuda", enabled=amp)
    historico, aval = [], None
    for epoca in range(1, EPOCAS + 1):
        inicio = time.perf_counter()
        lr = otimizador.param_groups[0]["lr"]
        perdas = train.treinar_epoca(modelo, loader, otimizador, escalador, dispositivo, 1.0,
                                     amp)
        agendador.step()
        linha = {"epoca": epoca, "lr": lr, "perda_treino": perdas[0],
                 "perda_classe_treino": perdas[1], "perda_sev_treino": perdas[2]}
        if avaliar_cada_epoca or epoca == EPOCAS:
            aval = train.avaliar(modelo, loader_fora, dispositivo, 1.0, amp)
            m = metrics.calcular_metricas(aval["classe"], aval["classe_prevista"])
            linha.update({"perda_fora": aval["perdas"][0], "acuracia_fora": m["acuracia"],
                          "f1_macro_fora": m["f1_macro"]})
        linha["tempo_s"] = round(time.perf_counter() - inicio, 1)
        historico.append(train._arredondar(linha))
        fmt.dizer(f"  {variante} | epoca {epoca}/{EPOCAS} | perda {perdas[0]:.3f}"
                  + (f" | F1 fora {100 * linha['f1_macro_fora']:.1f}%" if "f1_macro_fora" in linha
                     else "") + f" | {linha['tempo_s']:.1f} s")
    return modelo, aval, historico


def config_da_variante(variante: str, treino_df, extra: dict) -> dict:
    commit, alterado = train.estado_git()
    return {
        "variante": variante, "descricao": DESCRICAO[variante], "criado_em":
            datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "commit_com_alteracoes": alterado,
        "backbone": "resnet50", "pretreino": True, "largura": LARGURA, "altura": ALTURA,
        "batch": 32, "lr": 1e-4, "weight_decay": 1e-4, "peso_sev": 1.0, "epocas": EPOCAS,
        "t_max": EPOCAS, "parada_antecipada": False, "modelo": "última época", "seed": SEED,
        "amostragem": "WeightedRandomSampler (1/frequência da classe), como no run base",
        "augmentation": VARIANTES[variante],
        "albumentations": A.to_dict(transformacao_campo(variante)),
        "parametros": {"p_fundo": P_FUNDO, "p_corte": P_CORTE, "corte_maximo": CORTE_MAXIMO,
                       "p_apagar": P_APAGAR, "apagar_area_maxima": APAGAR_AREA_MAXIMA,
                       "apagar_buracos": APAGAR_BURACOS, "cinza": CINZA},
        "treino": {"n": len(treino_df),
                   "por_fonte": treino_df["fonte"].value_counts().to_dict(),
                   "por_classe": treino_df["classe"].value_counts().to_dict()},
        **extra, "versoes": {k: str(v) for k, v in train.versoes().items()},
    }


def sanear_checkpoint(caminho: Path) -> None:
    """Regrava um melhor.pt desta triagem com o config só em tipos do JSON. Os primeiros saíram
    com torch.__version__ (TorchVersion) no config, que o torch.load(weights_only=True) do
    train.carregar_modelo recusa. Só para checkpoints gerados aqui (fonte conhecida)."""
    checkpoint = torch.load(caminho, map_location="cpu", weights_only=False)
    checkpoint["config"] = json.loads(json.dumps(checkpoint["config"], default=str))
    torch.save(checkpoint, caminho)


def conferir_receita_v0() -> dict:
    """A receita da validação cruzada antiga do v0 (diagnostico_blocos) x a desta triagem."""
    antiga = json.loads((PASTA_DIAGNOSTICO / "config.json").read_text(encoding="utf-8"))["treino"]
    # no mesmo formato do config.json gravado (tuplas viram listas; floats arredondados)
    atual = json.loads(json.dumps(train._arredondar(A.to_dict(
        dados.transformacao_treino(LARGURA, ALTURA, SEED)))))
    comparacoes = {
        "epocas": (antiga["epocas"], EPOCAS), "t_max": (antiga["t_max"], EPOCAS),
        "parada_antecipada": (antiga["parada_antecipada"], False),
        "lr": (antiga["lr"], 1e-4), "weight_decay": (antiga["weight_decay"], 1e-4),
        "batch": (antiga["batch"], 32), "peso_sev": (antiga["peso_sev"], 1.0),
        "largura": (antiga["largura"], LARGURA), "altura": (antiga["altura"], ALTURA),
        "backbone": (antiga["backbone"], "resnet50"), "workers": (antiga["workers"], 0),
        "augmentation": (antiga["augmentation"], atual),
    }
    diferentes = [k for k, (a, b) in comparacoes.items() if a != b]
    return {"igual": not diferentes, "diferentes": diferentes,
            "observacao": "as duas usam o WeightedRandomSampler com a semente 42, o modelo da "
                          "última época e AMP; a triagem lê as imagens de um cache já em 448x224 "
                          "(o mesmo INTER_AREA do primeiro passo da augmentation)."}


def triagem(variantes, dispositivo) -> None:
    """Treino (1.180) -> validação oficial (252); grava model/runs/campo_<v>/."""
    cache = Cache()
    mascaras = ler_mascaras() if any(VARIANTES[v]["fundo"] or VARIANTES[v]["corte"]
                                     for v in variantes) else {}
    banco = ler_fundos() if any(VARIANTES[v]["fundo"] for v in variantes) else []
    for variante in variantes:
        pasta = train.PASTA_RUNS / f"campo_{variante}"
        if (pasta / "melhor.pt").is_file():
            fmt.dizer(f"{variante}: ja treinado em {train._exibir(pasta)}; pulei")
            continue
        pool = montar_dados(variante)
        val_ids = set(dados.montar_val()["id"])
        fora = pool[(pool["fonte"] == "bracol") & pool["id"].isin(val_ids)]
        treino_df = pool[~((pool["fonte"] == "bracol") & pool["id"].isin(val_ids))]
        fmt.dizer(f"Triagem {variante}: treino {len(treino_df)}, validacao {len(fora)}")
        inicio = time.perf_counter()
        modelo, aval, historico = treinar(variante, treino_df, fora, cache, mascaras, banco,
                                          dispositivo)
        pasta.mkdir(parents=True, exist_ok=True)
        config = config_da_variante(variante, treino_df, {
            "conjunto": "triagem: treino oficial do BRACOL (+ auxiliar) -> validação oficial",
            "minutos": round((time.perf_counter() - inicio) / 60, 1),
            "receita_v0": conferir_receita_v0() if variante == "v0" else None})
        m = metrics.calcular_metricas(aval["classe"], aval["classe_prevista"])
        ms = metrics.calcular_metricas_severidade(aval["severidade"], aval["severidade_prevista"])
        torch.save(train.montar_checkpoint(modelo, config, EPOCAS, m["f1_macro"]),
                   pasta / "melhor.pt")
        train.gravar_json(config, pasta / "config.json")
        train.gravar_historico(historico, pasta / "historico.csv")
        train.gravar_resultados(pasta, "val", fora, aval, m, ms, EPOCAS)
        (pasta / "relatorio.md").write_text(relatorio_variante(config, m, ms, historico),
                                            encoding="utf-8", newline="\n")
        fmt.dizer(f"{variante}: F1 macro val {100 * m['f1_macro']:.1f}%, acuracia "
                  f"{100 * m['acuracia']:.1f}%, {config['minutos']} min")
        del modelo
        torch.cuda.empty_cache()


def validacao_cruzada(variantes, dispositivo) -> None:
    """As 5 dobras do protocolo aleatório do diagnóstico (plano_dobras.csv), 20 épocas fixas,
    com o JMuBEN somado ao treino de cada dobra quando a variante o usa. Retomável."""
    plano = pd.read_csv(PASTA_DIAGNOSTICO / "plano_dobras.csv")
    cache = Cache()
    mascaras = ler_mascaras()
    banco = ler_fundos()
    for variante in variantes:
        pasta = train.PASTA_RUNS / f"campo_{variante}" / "cv"
        pasta.mkdir(parents=True, exist_ok=True)
        pool = montar_dados(variante)
        bracol_pool = pool[pool["fonte"] == "bracol"].set_index("id")
        auxiliar = pool[pool["fonte"] != "bracol"]
        for d in range(K):
            arquivo = pasta / f"predicoes_aleatorio_{d}.csv"
            if arquivo.is_file():
                continue
            papel = plano.set_index("id")[f"aleatorio_d{d}"]
            treino_ids = papel[papel == "treino"].index
            fora_ids = papel[papel == "fora"].index
            treino_df = pd.concat([bracol_pool.loc[treino_ids].reset_index(), auxiliar],
                                  ignore_index=True)
            fora = bracol_pool.loc[fora_ids].reset_index()
            fmt.dizer(f"CV {variante} dobra {d}: treino {len(treino_df)}, fora {len(fora)}")
            inicio = time.perf_counter()
            modelo, aval, historico = treinar(variante, treino_df, fora, cache, mascaras, banco,
                                              dispositivo, avaliar_cada_epoca=False)
            predicoes = train.tabela_de_predicoes(fora, aval).assign(protocolo="aleatorio",
                                                                      dobra=d)
            temporario = arquivo.with_name(arquivo.name + ".tmp")
            predicoes.to_csv(temporario, index=False, float_format="%.6f", lineterminator="\n")
            temporario.replace(arquivo)
            train.gravar_historico(historico, pasta / f"historico_aleatorio_{d}.csv")
            fmt.dizer(f"  dobra {d}: {(time.perf_counter() - inicio) / 60:.1f} min")
            del modelo
            torch.cuda.empty_cache()


def f1_das_dobras(pasta: Path) -> dict | None:
    """F1 macro das predições fora da dobra: das 1.432 juntas e de cada dobra."""
    arquivos = sorted(pasta.glob("predicoes_aleatorio_*.csv"))
    if len(arquivos) < K:
        return None
    todas = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    indice = {c: i for i, c in enumerate(bracol.CLASSES)}

    def f1(df):
        return metrics.calcular_metricas(df["classe_verdadeira"].map(indice),
                                         df["classe_prevista"].map(indice))
    juntas = f1(todas)
    return {"n": len(todas), "f1_macro": juntas["f1_macro"], "acuracia": juntas["acuracia"],
            "acuracia_ic95": juntas["acuracia_ic95"], "recall": juntas["recall"],
            "por_dobra": [f1(todas[todas["dobra"] == d])["f1_macro"] for d in range(K)]}


# ------------------------------------------------------------------------- relatórios
def relatorio_variante(config: dict, m: dict, ms: dict, historico) -> str:
    linhas = [f"# Variante `{config['variante']}`: {config['descricao']}", "",
              f"Gerado por `python model/treinar_campo.py --triagem` em "
              f"{datetime.fromisoformat(config['criado_em']):%d/%m/%Y %H:%M}. Treino: "
              f"{fmt.n(config['treino']['n'])} imagens ({config['treino']['por_fonte']}); "
              f"{EPOCAS} épocas fixas; o `melhor.pt` é a última época (fora do git). Teste "
              "fechado.", "", "## Validação oficial do BRACOL (252 imagens)", ""]
    linhas += metrics.trecho_relatorio(m, ms)
    if config.get("receita_v0"):
        r = config["receita_v0"]
        linhas += ["", "## Receita x validação cruzada antiga do v0", "",
                   ("Igual à do diagnóstico por blocos: a validação cruzada do v0 é reaproveitada."
                    if r["igual"] else f"**Diferente** em {r['diferentes']}: refazer a "
                                       "validação cruzada do v0."), r["observacao"]]
    linhas += ["", f"Tempo: {config['minutos']} min. Histórico em `historico.csv`.", ""]
    return "\n".join(linhas) + "\n"


def _ler_json(caminho: Path):
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.is_file() else None


def comparar() -> int:
    """A tabela única das variantes em PASTA_COMPARACAO/relatorio.md (com recomendacao.md)."""
    PASTA_COMPARACAO.mkdir(parents=True, exist_ok=True)
    v0_cv = f1_das_dobras(PASTA_DIAGNOSTICO)
    linhas_tabela, dados_json = [], {"v0_cv_diagnostico": v0_cv, "variantes": {}}
    for variante in VARIANTES:
        pasta = train.PASTA_RUNS / f"campo_{variante}"
        val = _ler_json(pasta / "metricas_val.json")
        if val is None:
            continue
        cv = f1_das_dobras(pasta / "cv") if variante != "v0" else v0_cv
        campo = _ler_json(pasta / "campo.json")
        pipeline = _ler_json(train.PASTA_RUNS / f"campo_{variante}_pipeline" / "metricas.json")
        dados_json["variantes"][variante] = {"val": val["classificacao"], "cv": cv,
                                             "campo": campo, "pipeline": pipeline}
        linhas_tabela.append(_linha_da_comparacao(variante, val, cv, v0_cv, campo, pipeline))
    train.gravar_json(json.loads(json.dumps(dados_json, default=float)),
                      PASTA_COMPARACAO / "comparacao.json")
    recomendacao = PASTA_COMPARACAO / "recomendacao.md"
    texto = [
        "# Classificador robusto ao campo: comparação das variantes", "",
        f"Gerado por `python model/treinar_campo.py --comparar` em "
        f"{datetime.now():%d/%m/%Y %H:%M}. Testes do BRACOL e do BRACOT fechados. Cada variante "
        "tem a pasta `model/runs/campo_<v>/` (triagem, proxies e rótulos de campo, validação "
        "cruzada das finalistas) e `campo_<v>_pipeline/` (T1, T2, T3 e a curva risco-cobertura).",
        "",
        "**Regra de adoção (decisão do gestor, 10/10/2026):** fica o que não piorar o BRACOL "
        "(queda de F1 macro de até ~2 pontos na validação cruzada) e melhorar o campo. No campo, "
        "só os rótulos decidem, e só onde os intervalos de Wilson sustentam; concordância e "
        "estabilidade servem para detectar comportamento degenerado e desempatar. Sem diferença "
        "sustentada pelos rótulos, a recomendação é ficar no v0.", "",
        "**Ressalva:** os rótulos de campo escolhem as finalistas e também medem, então a medida "
        "sai otimista (120 folhas, 46 fotos de 2 sessões, 1 lavoura; rótulos feitos por Claude "
        "por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026, não de "
        "especialista).", ""]
    texto += fmt.tabela(
        ["variante", "F1 val (252)", "F1 CV (1.432)", "Δ CV x v0", "modo", "A×B", "estab.",
         "saudáveis", "classe dominante", "P(doença dado S)", "P(saudável dado D)",
         "limiar recalibrado"], linhas_tabela,
        "llrrlrrrrlll")
    texto += ["", "### Rótulos de campo nos dois modos", "",
              "P(prever doença dado S) em 36 folhas S (intervalo de Wilson de 95%). P(prever "
              "saudável dado D) foi 0% (0 a 8%) em todas as variantes e nos dois modos.", ""]
    texto += fmt.tabela(["variante", "modo A", "modo B", "modo B, S com brilho",
                         "modo B, S sem brilho"], _linhas_dos_modos(dados_json), "lllll")
    texto += ["", "F1 CV do v0: as predições fora da dobra do diagnóstico por blocos (mesma "
                  "receita e mesmas dobras; conferido no `relatorio.md` de `campo_v0`). Colunas "
                  "de campo "
                  "no modo escolhido para a variante; \"-\" onde não se aplica ou não foi medido.",
              "", "## Recomendação", "",
              recomendacao.read_text(encoding="utf-8").strip() if recomendacao.is_file()
              else "Ainda não escrita.", ""]
    (PASTA_COMPARACAO / "relatorio.md").write_text("\n".join(texto) + "\n", encoding="utf-8",
                                                   newline="\n")
    fmt.dizer(f"Comparacao em {train._exibir(PASTA_COMPARACAO / 'relatorio.md')}")
    return 0


def _linhas_dos_modos(dados_json) -> list:
    def wilson(p):
        if not p or not p["n"]:
            return "-"
        return f"{_pct(p['valor'], 0)} ({_pct(p['ic95'][0], 0)}–{_pct(p['ic95'][1], 0)})"
    linhas = []
    for variante, d in dados_json["variantes"].items():
        rot = d["campo"]["rotulos"] if d["campo"] else None
        if not rot:
            continue
        linhas.append([f"`{variante}`", wilson(rot["A"]["doenca_dado_S"]),
                       wilson(rot["B"]["doenca_dado_S"]),
                       wilson(rot["B"].get("doenca_dado_S_com_brilho")),
                       wilson(rot["B"].get("doenca_dado_S_sem_brilho"))])
    return linhas


def ler_escolhas() -> dict:
    """escolhas.json da comparação, escrito à mão depois dos rótulos de campo: {"modos":
    {variante: "A" ou "B"}, "finalistas": [...]}; vazio se ainda não existe."""
    return _ler_json(PASTA_COMPARACAO / "escolhas.json") or {}


def _pct(valor, casas: int = 1) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "-"
    return fmt.pct(valor, 1, casas)


def _linha_da_comparacao(variante, val, cv, v0_cv, campo, pipeline) -> list:
    modo = ler_escolhas().get("modos", {}).get(variante, "A")
    px = campo["proxies"] if campo else None
    rot = campo["rotulos"][modo] if campo and campo.get("rotulos") else None

    def proporcao(chave):
        if not rot or not rot[chave]["n"]:
            return "-"
        p = rot[chave]
        return f"{_pct(p['valor'])} ({_pct(p['ic95'][0])}–{_pct(p['ic95'][1])})"
    limiar = None
    if pipeline:  # com a validação cruzada da variante, a proposta dela; senão, a da validação
        calibracao = pipeline["calibracao"]
        if calibracao["fora_da_dobra"]["pasta"].endswith(f"campo_{variante}/cv"):
            limiar = fmt.decimal(calibracao["proposta"], 2) if calibracao["proposta"] else "-"
        else:
            ponto = next((p for p in calibracao["validacao"]["curva"]
                          if p["mantidas"] and p["acuracia"] >= calibracao["alvo"]), None)
            limiar = f"{fmt.decimal(ponto['limiar'], 2)} (val)" if ponto else "-"
    return [
        f"`{variante}` ({DESCRICAO[variante]})", _pct(val["classificacao"]["f1_macro"]),
        _pct(cv["f1_macro"]) if cv else "-",
        f"{100 * (cv['f1_macro'] - v0_cv['f1_macro']):+.1f}".replace(".", ",")
        if cv and v0_cv else "-",
        modo, _pct(px["concordancia_A_B"]) if px else "-",
        _pct(px[modo]["estabilidade_media"]) if px else "-",
        _pct(px[modo]["fracao_saudavel"]) if px else "-",
        _pct(px[modo]["classe_mais_comum"]) if px else "-",
        proporcao("doenca_dado_S"), proporcao("saudavel_dado_D"), limiar or "-"]


# --------------------------------------------------------------------------- utilitários
def _ler_rgb(caminho: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.fromfile(caminho, np.uint8), cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB)


def _gravar_jpeg(rgb, caminho: Path) -> None:
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise train.ErroFatal(f"não foi possível gravar {caminho}")
    caminho.write_bytes(jpeg.tobytes())


def _gravar_prancha(cartoes, titulo: str, credito: str, caminho: Path, colunas: int = 2) -> None:
    altura, largura = cartoes[0].shape[:2]
    linhas = math.ceil(len(cartoes) / colunas)
    prancha = np.full((60 + linhas * (altura + 10) + 30, colunas * (largura + 10) + 10, 3), 255,
                      np.uint8)
    for i, cartao in enumerate(cartoes):
        y, x = 60 + (i // colunas) * (altura + 10), 10 + (i % colunas) * (largura + 10)
        prancha[y:y + altura, x:x + largura] = cartao
    cv2.putText(prancha, titulo, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(prancha, credito, (10, prancha.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 90, 90), 1, cv2.LINE_AA)
    _gravar_jpeg(prancha, caminho)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classificador robusto ao campo: mascaras, fundos, "
                                             "triagem, validacao cruzada e comparacao.")
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--mascaras", action="store_true")
    modo.add_argument("--fundos", action="store_true")
    modo.add_argument("--triagem", nargs="+", choices=list(VARIANTES))
    modo.add_argument("--cv", nargs="+", choices=list(VARIANTES))
    modo.add_argument("--comparar", action="store_true")
    args = ap.parse_args(argv)
    try:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.mascaras:
            tabela = gerar_mascaras(inf.carregar_modelos(dispositivo=str(dispositivo)))
            fmt.dizer(f"Mascaras: {tabela['regra'].value_counts().to_dict()}; prancha em "
                      f"{PASTA_MASCARAS / 'prancha_mascaras.jpg'}")
        elif args.fundos:
            tabela = gerar_fundos()
            fmt.dizer(f"Fundos: {len(tabela)} de {tabela['foto'].nunique()} fotos; prancha em "
                      f"{PASTA_FUNDOS / 'prancha_fundos.jpg'}")
        elif args.triagem:
            triagem(args.triagem, dispositivo)
        elif args.cv:
            validacao_cruzada(args.cv, dispositivo)
        else:
            return comparar()
        return 0
    except (train.ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
