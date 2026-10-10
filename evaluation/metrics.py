"""
Métricas da Frente 9 (validação do RNF01).

- Classificação do estresse: acurácia com intervalo de Wilson de 95%, precisão, recall e F1 por
  classe (na ordem de CLASSES), F1 macro, matriz de confusão e suporte.
- Severidade (níveis 0 a 4): MAE em níveis, kappa quadrático ponderado e acurácia exata e com
  tolerância de 1 nível, só nas linhas com alvo (severidade >= 0).

Também desenha a matriz de confusão, no estilo das figuras da EDA, e escreve o trecho de
métricas dos relatórios de model/train.py.
"""
import math
import sys
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

RAIZ_REPO = Path(__file__).resolve().parents[1]
if str(RAIZ_REPO / "data") not in sys.path:  # bracol, formatacao e eda_bracol ficam em data/
    sys.path.insert(0, str(RAIZ_REPO / "data"))

import bracol
import formatacao as fmt

NIVEIS = list(bracol.SEVERIDADES)  # severidade: níveis 0 a 4


# ---------------------------------------------------------------------- classificação
def intervalo_wilson(proporcao: float, n: int, confianca: float = 0.95) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporção medida em n casos (por exemplo, a acurácia)."""
    if n <= 0:
        raise ValueError(f"n precisa ser positivo: {n}")
    if not 0 <= proporcao <= 1:
        raise ValueError(f"proporção fora de [0, 1]: {proporcao}")
    z = NormalDist().inv_cdf(0.5 + confianca / 2)
    z2n = z * z / n
    centro = (proporcao + z2n / 2) / (1 + z2n)
    meia = z * math.sqrt(proporcao * (1 - proporcao) / n + z2n / (4 * n)) / (1 + z2n)
    return max(0.0, centro - meia), min(1.0, centro + meia)


def calcular_metricas(y_true, y_pred, classes=bracol.CLASSES) -> dict:
    """Métricas de classificação. y_true e y_pred são índices em classes (CLASSES.index).

    Precisão, recall, F1 e suporte vêm em listas na ordem de classes. A matriz de confusão é de
    inteiros (linhas: classe verdadeira; colunas: classe prevista). O F1 macro é a média simples
    dos F1 das classes; classe sem nenhuma previsão tem precisão 0.
    """
    y_true, y_pred = _vetores(y_true, y_pred)
    rotulos = list(range(len(classes)))
    fora = sorted(set(np.concatenate([y_true, y_pred]).tolist()) - set(rotulos))
    if fora:
        raise ValueError(f"índices de classe fora de 0 a {len(classes) - 1}: {fora}")
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_true, y_pred, labels=rotulos, zero_division=0
    )
    acertos = int(np.sum(y_true == y_pred))
    acuracia = acertos / len(y_true)
    return {
        "classes": list(classes),
        "n": len(y_true),
        "acertos": acertos,
        "acuracia": acuracia,
        "acuracia_ic95": list(intervalo_wilson(acuracia, len(y_true))),
        "precisao": precisao.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "suporte": suporte.tolist(),
        "f1_macro": float(np.mean(f1)),
        "matriz_confusao": confusion_matrix(y_true, y_pred, labels=rotulos).tolist(),
    }


# ------------------------------------------------------------------------- severidade
def calcular_metricas_severidade(sev_true, sev_pred) -> dict:
    """Métricas de severidade só nas linhas com alvo (severidade verdadeira >= 0).

    mae: erro médio absoluto, em níveis. kappa_quadratico: kappa de Cohen com peso quadrático;
    None quando é indefinido (por exemplo, um só nível nas duas listas). acuracia_exata e
    acuracia_tolerancia_1: fração com erro 0 e com erro de até 1 nível. suporte: linhas por
    nível verdadeiro. Sem nenhuma linha com alvo, as métricas ficam None.
    """
    sev_true, sev_pred = _vetores(sev_true, sev_pred, vazio_ok=True)
    com_alvo = sev_true >= 0
    alvo, previsto = sev_true[com_alvo], sev_pred[com_alvo]
    fora = sorted(set(np.concatenate([alvo, previsto]).tolist()) - set(NIVEIS))
    if fora:
        raise ValueError(f"níveis de severidade fora de {NIVEIS}: {fora}")
    resultado = {"n": int(com_alvo.sum()), "mae": None, "kappa_quadratico": None,
                 "acuracia_exata": None, "acuracia_tolerancia_1": None,
                 "suporte": [int(np.sum(alvo == k)) for k in NIVEIS]}
    if not len(alvo):
        return resultado
    erro = np.abs(alvo - previsto)
    with warnings.catch_warnings():  # kappa indefinido vira None, sem aviso no terminal
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        kappa = cohen_kappa_score(alvo, previsto, labels=NIVEIS, weights="quadratic")
    resultado.update({
        "mae": float(erro.mean()),
        "kappa_quadratico": None if np.isnan(kappa) else float(kappa),
        "acuracia_exata": float(np.mean(erro == 0)),
        "acuracia_tolerancia_1": float(np.mean(erro <= 1)),
    })
    return resultado


# --------------------------------------------------------------------- confiança (RF07)
def curva_risco_cobertura(confiancas, acertos, limiares, classes_verdadeiras=None,
                          classes=bracol.CLASSES) -> list[dict]:
    """Curva risco-cobertura do corte por confiança (RF07): em cada limiar, as predições com
    confiança >= limiar ficam e as outras viram "baixa confiança". Por limiar: as mantidas, a
    cobertura (fração mantida), a acurácia das mantidas com o intervalo de Wilson e o risco
    (1 - acurácia). Com as classes verdadeiras (índices em classes), também a cobertura de cada
    classe. Sem nenhuma mantida, acurácia e risco ficam NaN."""
    confiancas = np.asarray(confiancas, dtype=np.float64).ravel()
    acertos = np.asarray(acertos, dtype=bool).ravel()
    if not len(confiancas) or confiancas.shape != acertos.shape:
        raise ValueError("confiancas e acertos precisam ter o mesmo tamanho, maior que zero")
    if classes_verdadeiras is not None:
        classes_verdadeiras = np.asarray(classes_verdadeiras).ravel()
    curva = []
    for limiar in limiares:
        mantidas = confiancas >= limiar
        n = int(mantidas.sum())
        acuracia = float(acertos[mantidas].mean()) if n else float("nan")
        ponto = {"limiar": float(limiar), "mantidas": n, "cobertura": n / len(confiancas),
                 "acuracia": acuracia,
                 "acuracia_ic95": list(intervalo_wilson(acuracia, n)) if n else [float("nan")] * 2,
                 "risco": 1 - acuracia}
        if classes_verdadeiras is not None:
            ponto["cobertura_por_classe"] = {
                classe: float(mantidas[classes_verdadeiras == k].mean())
                if np.any(classes_verdadeiras == k) else float("nan")
                for k, classe in enumerate(classes)}
        curva.append(ponto)
    return curva


def _vetores(a, b, vazio_ok: bool = False):
    """Duas listas de inteiros do mesmo tamanho, como vetores numpy."""
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    if len(a) != len(b):
        raise ValueError(f"tamanhos diferentes: {len(a)} e {len(b)}")
    if not len(a) and not vazio_ok:
        raise ValueError("nenhuma linha para medir")
    return a, b


# ---------------------------------------------------------------------- figura e texto
def desenhar_matriz_confusao(matriz, titulo: str = "Matriz de confusão",
                             classes=bracol.CLASSES):
    """Mapa de calor da matriz de confusão, no estilo das figuras da EDA: a cor é a fração da
    linha (na diagonal, o recall da classe) e o número, a quantidade de imagens. Linhas: classe
    verdadeira; colunas: classe prevista. Devolve a Figure (gravar com salvar_figura)."""
    import eda_bracol as eda  # estilo comum das figuras do projeto

    m = np.asarray(matriz, dtype=np.int64)
    por_linha = m.sum(axis=1, keepdims=True)
    fracao = np.divide(m, por_linha, out=np.zeros(m.shape), where=por_linha > 0)
    with eda._estilo():
        fig = eda._figura(7.6, 4.4, titulo)
        ax = fig.add_subplot()
        ax.imshow(np.ma.masked_where(m == 0, fracao), cmap=eda.RAMPA_AZUL, vmin=0, vmax=1,
                  aspect="auto")
        for i, j in np.ndindex(m.shape):
            q = int(m[i, j])
            cor = eda._tinta_sobre(eda.RAMPA_AZUL(fracao[i, j])) if q else eda.APAGADO
            ax.text(j, i, fmt.n(q), ha="center", va="center", color=cor,
                    fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)),
                      [f"{c} ({fmt.n(int(t))})" for c, t in zip(classes, por_linha.ravel())])
        ax.set_xlabel("classe prevista")
        ax.set_ylabel("classe verdadeira (imagens)")
        ax.set_xticks(np.arange(-0.5, len(classes)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(classes)), minor=True)
        ax.grid(which="minor", color=eda.SUPERFICIE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title("cor: fração da linha (na diagonal, o recall); número: imagens", loc="left")
    return fig


def salvar_figura(fig, caminho: Path) -> None:
    """PNG sem o metadado de versão do matplotlib, como as figuras da EDA."""
    import eda_bracol as eda

    eda._salvar(fig, caminho)


def trecho_relatorio(m: dict, ms: dict) -> list[str]:
    """Linhas markdown com as métricas de calcular_metricas (m) e de
    calcular_metricas_severidade (ms), para os relatórios dos runs."""
    baixo, alto = m["acuracia_ic95"]
    linhas = [
        f"- **Acurácia:** {_pct(m['acuracia'])} ({fmt.n(m['acertos'])} de {fmt.n(m['n'])} "
        f"imagens; intervalo de Wilson de 95%: {_pct(baixo)} a {_pct(alto)}).",
        f"- **F1 macro:** {_pct(m['f1_macro'])} (média simples do F1 das "
        f"{len(m['classes'])} classes).",
        "",
    ]
    linhas += fmt.tabela(
        ["classe", "imagens", "precisão", "recall", "F1"],
        [[c, fmt.n(s), _pct(p), _pct(r), _pct(f)] for c, s, p, r, f in
         zip(m["classes"], m["suporte"], m["precisao"], m["recall"], m["f1"])],
    )
    linhas += ["", "Matriz de confusão (linhas: classe verdadeira; colunas: classe prevista):",
               ""]
    linhas += fmt.tabela(
        ["classe verdadeira", *m["classes"]],
        [[c, *map(fmt.n, linha)] for c, linha in zip(m["classes"], m["matriz_confusao"])],
    )
    linhas += ["", f"**Severidade** (níveis 0 a 4; {fmt.n(ms['n'])} imagens com alvo):", ""]
    if not ms["n"]:
        return [*linhas, "- Nenhuma imagem com alvo de severidade."]
    kappa = ("indefinido" if ms["kappa_quadratico"] is None
             else fmt.decimal(ms["kappa_quadratico"], 3))
    por_nivel = ", ".join(f"{k}: {fmt.n(q)}" for k, q in zip(NIVEIS, ms["suporte"]))
    return linhas + [
        f"- MAE (em níveis): {fmt.decimal(ms['mae'], 3)}; kappa quadrático ponderado: {kappa}.",
        f"- Acurácia exata: {_pct(ms['acuracia_exata'])}; com tolerância de 1 nível: "
        f"{_pct(ms['acuracia_tolerancia_1'])}.",
        f"- Imagens por nível verdadeiro: {por_nivel}.",
    ]


def _pct(fracao: float) -> str:
    """Fração como percentual com vírgula: 0.8125 -> 81,3%."""
    return fmt.pct(fracao, 1)
