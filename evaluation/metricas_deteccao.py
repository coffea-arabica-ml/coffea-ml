"""
Métricas do detector de folhas (RF09): AP de máscara e de caixa no estilo COCO (101 pontos de
recall, IoU de 0,50 a 0,95), precisão e recall por limiar de score e a contagem das previsões
fora da região anotada.

Anotação parcial do BRACOT (decisão do gestor, 10/10/2026): só cerca de 1/3 das folhas de cada
foto está contornado, e as que faltam ficam sobretudo na periferia. A região anotada é o casco
convexo das folhas anotadas, dilatado por MARGEM_REGIAO do lado maior; a mesma função
(regiao_anotada) pinta de cinza o resto da foto no treino. Dois modos de medir:
- "regiao" (principal): a previsão sem folha anotada correspondente que tem mais de
  FRACAO_FORA_MAX da máscara fora da região é ignorada (nem acerto nem erro);
- "conservador": foto inteira; toda previsão sem folha anotada correspondente é erro.

Tudo em numpy (e cv2 para preencher polígonos): testável sem modelo.
"""
import cv2
import numpy as np

LIMIARES_IOU = np.round(np.arange(0.50, 0.96, 0.05), 2)  # 0,50, 0,55, ..., 0,95
PONTOS_RECALL = np.linspace(0, 1, 101)
LIMIARES_SCORE = np.round(np.arange(0.05, 0.96, 0.05), 2)
MARGEM_REGIAO = 0.02  # fração do lado maior da imagem
FRACAO_FORA_MAX = 0.5
MODOS = ("regiao", "conservador")
TIPOS = ("mascara", "caixa")


# --------------------------------------------------------------------- geometria
def rasterizar(poligonos, altura: int, largura: int, escala: float = 1.0) -> np.ndarray:
    """(n, altura, largura) bool: cada polígono ([[x, y], ...] ou [x1, y1, x2, y2, ...], em
    pixels da foto original) multiplicado por `escala` e preenchido."""
    mascaras = np.zeros((len(poligonos), altura, largura), dtype=np.uint8)
    for k, poligono in enumerate(poligonos):
        pontos = np.asarray(poligono, dtype=np.float64).reshape(-1, 2) * escala
        if len(pontos) >= 3:
            cv2.fillPoly(mascaras[k], [np.round(pontos).astype(np.int32)], 1)
    return mascaras.astype(bool)


def caixas(poligonos, escala: float = 1.0) -> np.ndarray:
    """(n, 4) xyxy: a caixa de cada polígono, multiplicada por `escala`."""
    if not len(poligonos):
        return np.zeros((0, 4))
    pontos = [np.asarray(p, dtype=np.float64).reshape(-1, 2) * escala for p in poligonos]
    return np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                     for p in pontos])


def regiao_anotada(poligonos, altura: int, largura: int, escala: float = 1.0,
                   margem: float = MARGEM_REGIAO) -> np.ndarray:
    """Máscara (altura x largura) da região anotada: o casco convexo de todos os pontos dos
    polígonos (multiplicados por `escala`), dilatado por `margem` vezes o lado maior."""
    pontos = np.vstack([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in poligonos])
    casco = cv2.convexHull(np.round(pontos * escala).astype(np.int32))
    regiao = np.zeros((altura, largura), dtype=np.uint8)
    cv2.fillPoly(regiao, [casco], 1)
    raio = int(round(margem * max(altura, largura)))
    if raio > 0:
        nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1))
        regiao = cv2.dilate(regiao, nucleo)
    return regiao.astype(bool)


def fracao_fora(mascaras: np.ndarray, regiao: np.ndarray) -> np.ndarray:
    """Fração da área de cada máscara fora da região (1 para máscara vazia)."""
    if not len(mascaras):
        return np.zeros(0)
    area = mascaras.reshape(len(mascaras), -1).sum(axis=1)
    dentro = (mascaras & regiao).reshape(len(mascaras), -1).sum(axis=1)
    return np.where(area > 0, 1 - dentro / np.maximum(area, 1), 1.0)


def iou_mascaras(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cada máscara de a (n, H, W) e de b (m, H, W): matriz n x m."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    fa = a.reshape(len(a), -1).astype(np.float32)
    fb = b.reshape(len(b), -1).astype(np.float32)
    intersecao = fa @ fb.T
    uniao = fa.sum(axis=1)[:, None] + fb.sum(axis=1)[None, :] - intersecao
    return np.where(uniao > 0, intersecao / np.maximum(uniao, 1), 0.0)


def iou_caixas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre caixas xyxy: matriz n x m."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersecao = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    uniao = area_a[:, None] + area_b[None, :] - intersecao
    return np.where(uniao > 0, intersecao / np.maximum(uniao, 1e-12), 0.0)


# ---------------------------------------------------------------------- casamento
def casar(iou: np.ndarray, ignorar: np.ndarray, limiar: float) -> np.ndarray:
    """Casamento no estilo COCO, com as previsões em ordem de score decrescente (colunas de iou).

    Cada previsão pega a folha anotada livre de maior IoU, se esse IoU for >= limiar. Devolve,
    por previsão, 1 (acerto), 0 (erro) ou -1 (sem par e marcada em `ignorar`: fica de fora).
    """
    n_anotadas, n_previstas = iou.shape
    livre = np.ones(n_anotadas, dtype=bool)
    estado = np.zeros(n_previstas, dtype=np.int64)
    for j in range(n_previstas):
        candidatas = livre & (iou[:, j] >= limiar)
        if candidatas.any():
            k = int(np.argmax(np.where(candidatas, iou[:, j], -1.0)))
            livre[k] = False
            estado[j] = 1
        elif ignorar[j]:
            estado[j] = -1
    return estado


def precisao_media(estados: np.ndarray, scores: np.ndarray, n_anotadas: int) -> float:
    """AP com 101 pontos de recall (como o COCO), juntando as previsões de todas as fotos:
    estados (1, 0 ou -1) e scores na mesma ordem. As ignoradas (-1) ficam de fora."""
    if n_anotadas == 0:
        return float("nan")
    validas = estados >= 0
    ordem = np.argsort(-scores[validas], kind="mergesort")
    acertos = estados[validas][ordem] == 1
    tp, fp = np.cumsum(acertos), np.cumsum(~acertos)
    if not len(tp):
        return 0.0
    recall = tp / n_anotadas
    precisao = tp / np.maximum(tp + fp, 1)
    precisao = np.maximum.accumulate(precisao[::-1])[::-1]  # envelope: precisão máxima à direita
    posicao = np.searchsorted(recall, PONTOS_RECALL, side="left")
    pontos = np.where(posicao < len(precisao), precisao[np.minimum(posicao, len(precisao) - 1)], 0)
    return float(np.mean(pontos))


# ---------------------------------------------------------------------- avaliação
def resumir_foto(poligonos_anotados, deteccoes, altura: int, largura: int,
                 lado: int = 1024) -> dict:
    """O que avaliar() precisa de uma foto, calculado na resolução de trabalho (lado maior =
    `lado`) e sem guardar as máscaras: as matrizes de IoU (máscara e caixa), quais previsões
    ficam fora da região anotada, os scores e o número de folhas anotadas.

    poligonos_anotados: polígonos das folhas anotadas, em pixels da foto original.
    deteccoes: dicts com "poligono", "caixa" (xyxy) e "score", em pixels da foto original (o que
    model/detector.detectar_folhas devolve).
    """
    escala = lado / max(altura, largura)
    alt, larg = int(round(altura * escala)), int(round(largura * escala))
    deteccoes = sorted(deteccoes, key=lambda d: -d["score"])
    anotadas = rasterizar(poligonos_anotados, alt, larg, escala)
    previstas = rasterizar([d["poligono"] for d in deteccoes], alt, larg, escala)
    caixas_previstas = np.asarray([d["caixa"] for d in deteccoes],
                                  dtype=np.float64).reshape(-1, 4) * escala
    return {
        "iou_mascara": iou_mascaras(anotadas, previstas),
        "iou_caixa": iou_caixas(caixas(poligonos_anotados, escala), caixas_previstas),
        "fora": fracao_fora(previstas, regiao_anotada(poligonos_anotados, alt, larg, escala))
        > FRACAO_FORA_MAX,
        "scores": np.asarray([d["score"] for d in deteccoes], dtype=np.float64),
        "n_anotadas": len(poligonos_anotados),
    }


def avaliar(resumos: list[dict], limiares_score=LIMIARES_SCORE) -> dict:
    """AP de máscara e de caixa (IoU 0,50 e 0,50 a 0,95) e a curva de precisão, recall e F1 por
    limiar de score (máscara, IoU 0,5), nos dois modos. resumos: saídas de resumir_foto."""
    n_anotadas = sum(r["n_anotadas"] for r in resumos)
    scores = np.concatenate([r["scores"] for r in resumos]) if resumos else np.zeros(0)
    fora = [r["fora"] for r in resumos]
    resultado = {"n_fotos": len(resumos), "n_anotadas": n_anotadas, "n_previstas": len(scores)}
    for modo in MODOS:
        ignorar = fora if modo == "regiao" else [np.zeros(len(x), dtype=bool) for x in fora]
        resultado[modo] = {}
        for tipo in TIPOS:
            aps = []
            for limiar in LIMIARES_IOU:
                estados = _estados(resumos, f"iou_{tipo}", ignorar, limiar)
                aps.append(precisao_media(estados, scores, n_anotadas))
            resultado[modo][tipo] = {"ap50": aps[0], "ap50_95": float(np.mean(aps))}
        estados = _estados(resumos, "iou_mascara", ignorar, 0.5)
        resultado[modo]["curva"] = curva_por_limiar(
            estados, scores, np.concatenate(fora) if fora else np.zeros(0, dtype=bool),
            n_anotadas, limiares_score)
    return resultado


def contagens(resumos: list[dict], limiar: float) -> list[dict]:
    """Por foto, com score >= limiar: folhas anotadas, previstas, acertos (máscara, IoU 0,5) e
    previsões fora da região anotada."""
    linhas = []
    for r in resumos:
        mantidas = r["scores"] >= limiar
        estados = casar(r["iou_mascara"], r["fora"], 0.5)
        linhas.append({"anotadas": r["n_anotadas"], "previstas": int(mantidas.sum()),
                       "acertos": int(np.sum(mantidas & (estados == 1))),
                       "fora_da_regiao": int(np.sum(mantidas & r["fora"]))})
    return linhas


def _estados(resumos, chave: str, ignorar, limiar: float) -> np.ndarray:
    if not resumos:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([casar(r[chave], ig, limiar) for r, ig in zip(resumos, ignorar)])


def curva_por_limiar(estados, scores, fora, n_anotadas: int, limiares_score) -> list[dict]:
    """Por limiar de score: acertos, erros, ignoradas, previsões fora da região, precisão,
    recall e F1 (IoU 0,5)."""
    curva = []
    for limiar in limiares_score:
        mantidas = scores >= limiar
        acertos = int(np.sum(mantidas & (estados == 1)))
        erros = int(np.sum(mantidas & (estados == 0)))
        precisao = acertos / (acertos + erros) if acertos + erros else float("nan")
        recall = acertos / n_anotadas if n_anotadas else float("nan")
        f1 = (2 * precisao * recall / (precisao + recall)
              if acertos and precisao + recall > 0 else 0.0)
        curva.append({"limiar": float(limiar), "acertos": acertos, "erros": erros,
                      "ignoradas": int(np.sum(mantidas & (estados == -1))),
                      "fora_da_regiao": int(np.sum(mantidas & fora)),
                      "previstas": int(mantidas.sum()), "precisao": precisao, "recall": recall,
                      "f1": f1})
    return curva


def melhor_limiar(curva: list[dict]) -> dict:
    """O ponto da curva com maior F1 (no empate, o menor limiar)."""
    return max(curva, key=lambda ponto: (ponto["f1"], -ponto["limiar"]))
