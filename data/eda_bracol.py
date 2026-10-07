"""
EDA do BRACOL: gera as figuras em data/reports/figures/ e o resumo legível
data/reports/eda_bracol.md, com figuras, tabelas e comentários.

Uso (de qualquer pasta):
    python data/eda_bracol.py

carregar(), medir_brilho_e_cor(), distancias_ao_vizinho() e as funções tabela_*() e figura_*()
não gravam nada: o notebook notebooks/01_eda.ipynb chama as mesmas funções e só mostra o
resultado. Quem grava é gerar(), chamado pelo main().

Cores: paleta de referência da skill de visualização, validada com o script dela (as cinco
classes e os três splits passam nos testes de daltonismo). A classe 5 fica em cinza porque não
é classe do projeto. Rodar de novo com os mesmos dados produz os mesmos arquivos.
"""
import argparse
import hashlib
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

import bracol
import formatacao as fmt

CREDITO = ("Imagens: BRACOL (Krohling, Esgario e Ventura, 2019), CC BY 4.0, "
           "DOI 10.17632/yy2k5y8mxg.1")
ROTULOS = [*bracol.CLASSES, "classe 5"]  # ordem das classes nas tabelas e figuras

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
APAGADO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
FUNDO_AUSENTE = "#f0efec"  # faixas de ids sem imagem
CINZA_AUSENTE = "#d6d5ce"  # barras e marcas de imagens ausentes
AZUL = "#2a78d6"
COR_CLASSE = {
    "saudavel": "#2a78d6",
    "ferrugem": "#eb6834",
    "bicho_mineiro": "#1baf7a",
    "phoma": "#eda100",
    "cercosporiose": "#e87ba4",
    "classe 5": APAGADO,
}
COR_SPLIT = {"treino": "#2a78d6", "val": "#eb6834", "teste": "#1baf7a"}
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
ESTILO = {
    "font.size": 9,
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "axes.edgecolor": EIXO,
    "axes.linewidth": 0.8,
    "axes.labelcolor": TINTA_2,
    "axes.titlecolor": TINTA_2,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": EIXO,
    "ytick.color": EIXO,
    "xtick.labelcolor": TINTA_2,
    "ytick.labelcolor": TINTA_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.titlesize": 12,
}


# ------------------------------------------------------------------------------ dados
def carregar(manifest=bracol.MANIFEST_BRACOL) -> pd.DataFrame:
    """Manifest inteiro (inclusive as linhas excluídas), com a coluna `rotulo`: a classe do
    projeto, ou "classe 5"."""
    df = bracol.ler_manifest(manifest, incluir_excluidas=True)
    df["rotulo"] = df["classe"].where(df["classe"] != "", "classe 5")
    return df


def medir_brilho_e_cor(df, raiz=bracol.RAIZ_REPO, largura: int = 256) -> pd.DataFrame:
    """Brilho e cor média de cada imagem presente, numa versão reduzida a `largura` px.

    luminancia: média da versão em tons de cinza (Pillow "L", pesos ITU-R 601), de 0 a 255.
    r, g, b: cor média da imagem inteira.
    fundo_luminancia, fundo_tom: luminância e tom (azul menos vermelho) das faixas de cima e
    de baixo, 1/8 da altura cada, que nas fotos do BRACOL costumam ter só o fundo.
    """
    pesos = np.array([0.299, 0.587, 0.114])
    medidas = []
    for linha in df[df["presente"] == 1].itertuples():
        with Image.open(Path(raiz) / linha.caminho) as im:
            im.draft("RGB", (largura, largura // 2))  # decodifica o JPEG já reduzido
            reduzida = im.convert("RGB")
        altura = max(1, round(reduzida.height * largura / reduzida.width))
        reduzida = reduzida.resize((largura, altura), Image.Resampling.BILINEAR)
        rgb = np.asarray(reduzida, dtype=np.float64)
        faixa = max(1, altura // 8)
        fundo = np.concatenate([rgb[:faixa], rgb[-faixa:]]).reshape(-1, 3).mean(axis=0)
        r, g, b = rgb.reshape(-1, 3).mean(axis=0)
        medidas.append({
            "id": linha.id,
            "rotulo": linha.rotulo,
            "luminancia": np.asarray(reduzida.convert("L"), dtype=np.float64).mean(),
            "r": r,
            "g": g,
            "b": b,
            "fundo_luminancia": float(fundo @ pesos),
            "fundo_tom": fundo[2] - fundo[0],
        })
    return pd.DataFrame(medidas)


def distancias_ao_vizinho(df) -> pd.DataFrame:
    """Distância (bits) de cada imagem presente até a mais parecida, pelos dois pHash: colunas
    "quadro" e "folha", indexadas por id."""
    presentes = df[df["presente"] == 1][["id", "phash", "phash_folha"]].to_dict("records")
    return pd.DataFrame({
        "quadro": bracol.distancias_ao_vizinho(presentes, chave="phash"),
        "folha": bracol.distancias_ao_vizinho(presentes, chave="phash_folha"),
    }).sort_index()


# ---------------------------------------------------------------------------- tabelas
def tabela_classes(df) -> pd.DataFrame:
    """Por classe: linhas no dataset.csv, com imagem, sem imagem e perda (%)."""
    t = df.groupby("rotulo").agg(no_csv=("id", "size"), com_imagem=("presente", "sum"))
    t = t.reindex(ROTULOS, fill_value=0)
    t["sem_imagem"] = t["no_csv"] - t["com_imagem"]
    t["perda_pct"] = 100 * t["sem_imagem"] / t["no_csv"].where(t["no_csv"] > 0)
    return t


def tabela_severidade(df) -> pd.DataFrame:
    """Imagens elegíveis por classe (linhas) e nível de severidade (colunas, 0 a 4)."""
    eleg = df[df["excluida"] == 0]
    t = pd.crosstab(eleg["rotulo"], eleg["severity"])
    return t.reindex(index=bracol.CLASSES, columns=list(bracol.SEVERIDADES), fill_value=0)


def tabela_estresses(df) -> pd.DataFrame:
    """Combinações de estresses marcados nas imagens elegíveis, da mais comum à menos comum."""
    eleg = df[df["excluida"] == 0]
    colunas = list(bracol.PS_PARA_COLUNA.values())
    combinacao = eleg[colunas].apply(
        lambda linha: " + ".join(c for c in colunas if linha[c]) or "nenhum (saudável)", axis=1
    )
    t = combinacao.value_counts().rename_axis("estresses").reset_index(name="imagens")
    return t.sort_values(["imagens", "estresses"], ascending=[False, True], ignore_index=True)


def tabela_divisao(df) -> pd.DataFrame:
    """Imagens elegíveis por classe (linhas) e split (colunas)."""
    eleg = df[df["excluida"] == 0]
    t = pd.crosstab(eleg["rotulo"], eleg["split"])
    return t.reindex(index=bracol.CLASSES, columns=list(bracol.SPLITS), fill_value=0)


def tabela_divisao_por_severidade(df) -> pd.DataFrame:
    """Imagens elegíveis por nível de severidade (linhas) e split (colunas)."""
    eleg = df[df["excluida"] == 0]
    t = pd.crosstab(eleg["severity"], eleg["split"])
    return t.reindex(index=list(bracol.SEVERIDADES), columns=list(bracol.SPLITS), fill_value=0)


def tabela_brilho_por_faixa(medidas, tamanho: int = 100) -> pd.DataFrame:
    """Medianas por faixa de ids (só faixas com imagem): luminância da imagem, luminância e
    tom do fundo, a dispersão do tom (distância interquartil) e as duas classes mais comuns."""
    ultimo = int(medidas["id"].max())
    linhas = []
    for k, grupo in medidas.groupby((medidas["id"] - 1) // tamanho):
        contagem = grupo["rotulo"].value_counts().sort_index().sort_values(
            ascending=False, kind="stable"
        )
        linhas.append({
            "ids": f"{k * tamanho + 1}-{min((k + 1) * tamanho, ultimo)}",
            "imagens": len(grupo),
            "luminancia": grupo["luminancia"].median(),
            "fundo_luminancia": grupo["fundo_luminancia"].median(),
            "fundo_tom": grupo["fundo_tom"].median(),
            "fundo_tom_iqr": grupo["fundo_tom"].quantile(0.75) - grupo["fundo_tom"].quantile(0.25),
            "classes": ", ".join(f"{c} {q}" for c, q in contagem.head(2).items()),
        })
    return pd.DataFrame(linhas)


def tabela_brilho_por_classe(medidas) -> pd.DataFrame:
    """Medianas por classe: luminância da imagem, luminância e tom do fundo."""
    grupos = medidas.groupby("rotulo")
    t = pd.DataFrame({
        "imagens": grupos.size(),
        "luminancia": grupos["luminancia"].median(),
        "fundo_luminancia": grupos["fundo_luminancia"].median(),
        "fundo_tom": grupos["fundo_tom"].median(),
    })
    return t.reindex([r for r in ROTULOS if r in t.index])


# ---------------------------------------------------------------------------- figuras
@contextmanager
def _estilo():
    with matplotlib.rc_context(ESTILO):
        yield


def _figura(largura, altura, titulo) -> Figure:
    fig = Figure(figsize=(largura, altura), layout="constrained")
    fig.suptitle(titulo, x=0.01, ha="left", color=TINTA)
    return fig


def _tinta_sobre(cor) -> str:
    """Branco ou tinta escura, conforme a cor de fundo de uma célula."""
    r, g, b = matplotlib.colors.to_rgb(cor)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else TINTA


def figura_classes(df) -> Figure:
    """Barras por classe. Se faltam imagens, mostra também as que faltam e a perda."""
    t = tabela_classes(df)
    faltam = t["sem_imagem"].sum() > 0
    titulo = ("Classes no dataset.csv completo e na cópia recuperada" if faltam
              else "Imagens por classe")
    with _estilo():
        fig = _figura(9, 3.4, titulo)
        ax = fig.add_subplot()
        y = np.arange(len(t))[::-1]
        ax.barh(y, t["com_imagem"], height=0.6, color=AZUL, edgecolor=SUPERFICIE,
                linewidth=1.5, label="com imagem")
        if faltam:
            ax.barh(y, t["sem_imagem"], left=t["com_imagem"], height=0.6, color=CINZA_AUSENTE,
                    edgecolor=SUPERFICIE, linewidth=1.5, label="sem imagem")
        total = t["no_csv"].sum()
        for yi, linha in zip(y, t.itertuples()):
            if faltam:
                texto = (f"{fmt.n(linha.com_imagem)} de {fmt.n(linha.no_csv)} "
                         f"(perda de {fmt.decimal(linha.perda_pct)}%)")
            else:
                texto = f"{fmt.n(linha.no_csv)} ({fmt.pct(linha.no_csv, total)})"
            ax.text(linha.no_csv + t["no_csv"].max() * 0.015, yi, texto, va="center",
                    color=TINTA_2)
        ax.set_yticks(y, t.index)
        ax.set_xlim(0, t["no_csv"].max() * (1.5 if faltam else 1.25))
        ax.set_xlabel("imagens")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        if faltam:
            ax.legend(loc="lower right")
    return fig


def figura_ausentes_por_id(df) -> Figure:
    """Uma faixa por classe: cada id do csv é um traço na cor da classe (cinza claro se a
    imagem falta). Mostra que as classes aparecem em blocos de ids."""
    faltam = (df["presente"] == 0).any()
    titulo = ("Ids do dataset.csv por classe, com e sem imagem" if faltam
              else "Classes ao longo dos ids do dataset.csv")
    with _estilo():
        fig = _figura(11, 3.3, titulo)
        ax = fig.add_subplot()
        for a, b in fmt.intervalos(df.loc[df["presente"] == 0, "id"]):
            ax.axvspan(a - 0.5, b + 0.5, color=FUNDO_AUSENTE, linewidth=0, zorder=0)
        for k, rotulo in enumerate(ROTULOS):
            y = len(ROTULOS) - 1 - k
            da_classe = df[df["rotulo"] == rotulo]
            for presente, cor in ((0, CINZA_AUSENTE), (1, COR_CLASSE[rotulo])):
                ids = da_classe.loc[da_classe["presente"] == presente, "id"]
                ax.vlines(ids, y - 0.32, y + 0.32, color=cor, linewidth=0.6)
        ax.set_yticks(range(len(ROTULOS)), ROTULOS[::-1])
        ax.set_xlim(0, int(df["id"].max()) + 1)
        ax.set_xlabel("id no dataset.csv")
        ax.spines["left"].set_visible(False)
        if faltam:
            fig.legend(handles=[
                Line2D([], [], color=TINTA_2, linewidth=3, label="com imagem (na cor da classe)"),
                Line2D([], [], color=CINZA_AUSENTE, linewidth=3, label="sem imagem"),
            ], loc="outside upper right", ncols=2)
    return fig


def figura_severidade(df) -> Figure:
    """Mapa de calor classe x severidade: cor = fração da classe, número = imagens."""
    t = tabela_severidade(df)
    fracao = t.div(t.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    with _estilo():
        fig = _figura(9, 3.5, "Severidade por classe (imagens elegíveis)")
        ax = fig.add_subplot()
        ax.imshow(np.ma.masked_where(t.to_numpy() == 0, fracao.to_numpy()), cmap=RAMPA_AZUL,
                  vmin=0, vmax=1, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                q = int(t.iat[i, j])
                cor = _tinta_sobre(RAMPA_AZUL(fracao.iat[i, j])) if q else APAGADO
                ax.text(j, i, fmt.n(q), ha="center", va="center", color=cor)
        ax.set_xticks(range(t.shape[1]),
                      [f"{k}: {d}".replace(" (", "\n(") for k, d in bracol.SEVERIDADES.items()])
        ax.set_yticks(range(t.shape[0]), t.index)
        ax.set_xticks(np.arange(-0.5, t.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, t.shape[0]), minor=True)
        ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title("cor: fração da classe em cada nível; número: imagens", loc="left")
    return fig


def figura_estresses(df) -> Figure:
    """Barras: combinações de estresses marcados na mesma folha."""
    t = tabela_estresses(df)
    with _estilo():
        fig = _figura(8.5, 0.3 * len(t) + 1.3, "Estresses marcados por folha (imagens elegíveis)")
        ax = fig.add_subplot()
        y = np.arange(len(t))[::-1]
        ax.barh(y, t["imagens"], height=0.62, color=AZUL)
        for yi, q in zip(y, t["imagens"]):
            ax.text(q + t["imagens"].max() * 0.01, yi, fmt.n(q), va="center", color=TINTA_2)
        ax.set_yticks(y, t["estresses"])
        ax.set_xlim(0, t["imagens"].max() * 1.12)
        ax.set_xlabel("imagens")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    return fig


def figura_exemplos(df, raiz=bracol.RAIZ_REPO, por_classe: int = 4) -> Figure:
    """Mosaico: imagens de treino de cada classe e, na última linha, da classe 5 (excluída)."""
    with _estilo():
        fig = Figure(figsize=(13.4, 11))
        fig.suptitle("Exemplos por classe (imagens de treino; a classe 5 está excluída)",
                     x=0.01, y=0.985, ha="left", color=TINTA)
        grade = fig.add_gridspec(len(ROTULOS), por_classe, left=0.12, right=0.995, top=0.94,
                                 bottom=0.035, wspace=0.03, hspace=0.2)
        for k, rotulo in enumerate(ROTULOS):
            for c, i in enumerate(_exemplos(df, rotulo, por_classe)):
                ax = fig.add_subplot(grade[k, c])
                ax.imshow(_miniatura(Path(raiz) / df.loc[df["id"] == i, "caminho"].item()))
                ax.set_axis_off()
                ax.set_title(f"id {i}", loc="left", fontsize=8, pad=2)
                if c == 0:
                    nome = "classe 5\n(excluída)" if rotulo == "classe 5" else rotulo
                    ax.text(-0.04, 0.5, nome, transform=ax.transAxes, ha="right", va="center",
                            fontsize=10, color=TINTA)
        fig.text(0.01, 0.01, CREDITO, fontsize=8, color=TINTA_2)
    return fig


def _exemplos(df, rotulo, quantos):
    """Ids sorteados de forma determinística (seed do projeto): de treino para as classes e,
    para a classe 5, entre as linhas com imagem."""
    if rotulo == "classe 5":
        candidatos = df[(df["rotulo"] == rotulo) & (df["presente"] == 1)]
    else:
        candidatos = df[(df["rotulo"] == rotulo) & (df["split"] == "treino")]
    sorteio = sorted(candidatos["id"], key=lambda i: hashlib.sha256(
        f"{bracol.SEED_PADRAO}:{i}".encode()).hexdigest())
    return sorted(sorteio[:quantos])


def _miniatura(caminho, largura: int = 300):
    with Image.open(caminho) as im:
        im.draft("RGB", (2 * largura, largura))
        im = im.convert("RGB")
    altura = max(1, round(im.height * largura / im.width))
    return np.asarray(im.resize((largura, altura), Image.Resampling.LANCZOS))


def figura_vizinho_mais_proximo(distancias) -> Figure:
    """Histogramas da distância de cada imagem até a mais parecida, um por pHash (quadro
    inteiro e recorte da folha), cada um com o seu limiar de agrupamento."""
    paineis = (("quadro", "pHash do quadro inteiro", bracol.LIMIAR_QUASE_DUPLICATA),
               ("folha", "pHash do recorte da folha", bracol.LIMIAR_QUASE_DUPLICATA_FOLHA))
    topo = int(distancias.to_numpy().max()) + 4 if len(distancias) else 64
    with _estilo():
        fig = _figura(12, 3.6, "Distância de cada imagem até a mais parecida (pHash de 256 bits)")
        eixos = fig.subplots(1, 2, sharey=True)
        for ax, (coluna, nome, limiar) in zip(eixos, paineis):
            valores = distancias[coluna]
            ax.hist(valores, bins=np.arange(-1, topo + 2, 2), color=AZUL, edgecolor=SUPERFICIE,
                    linewidth=1)
            ax.axvline(limiar, color=TINTA, linewidth=1.2)
            ax.text(limiar + 1.5, 0.96, f"limiar: {limiar} bits",
                    transform=ax.get_xaxis_transform(), va="top", color=TINTA)
            ax.set_xlim(0, topo + 1)
            ax.set_xlabel(f"bits diferentes (de {bracol.PHASH_TAMANHO ** 2})")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            if len(valores):
                ax.set_title(f"{nome}: mais próxima a {int(valores.min())} bits, mediana "
                             f"{fmt.decimal(valores.median(), 0)}", loc="left")
        eixos[0].set_ylabel("imagens")
    return fig


def figura_divisao(df) -> Figure:
    """Barras empilhadas: splits por classe (imagens) e por nível de severidade (%)."""
    por_classe = tabela_divisao(df)
    por_nivel = tabela_divisao_por_severidade(df)
    pct_nivel = 100 * por_nivel.div(por_nivel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    with _estilo():
        fig = _figura(12, 3.6, f"Divisão treino/val/teste (imagens elegíveis, seed "
                               f"{bracol.SEED_PADRAO})")
        ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.2, 1])
        _empilhar(ax1, por_classe, list(por_classe.index))
        for yi, total in zip(np.arange(len(por_classe))[::-1], por_classe.sum(axis=1)):
            ax1.text(total + por_classe.sum(axis=1).max() * 0.015, yi, fmt.n(total), va="center",
                     color=TINTA_2)
        ax1.set_xlim(0, por_classe.sum(axis=1).max() * 1.12)
        ax1.set_title("por classe (imagens)", loc="left")
        rotulos = [f"{k}: {d}" for k, d in bracol.SEVERIDADES.items()]
        _empilhar(ax2, pct_nivel, rotulos)
        for yi, (nivel, linha) in zip(np.arange(len(pct_nivel))[::-1], pct_nivel.iterrows()):
            ax2.text(101.5, yi, f"teste {fmt.decimal(linha['teste'])}%", va="center", color=TINTA_2)
        ax2.set_xlim(0, 122)
        ax2.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax2.set_title("por nível de severidade (% de cada nível)", loc="left")
        fig.legend(handles=[Patch(color=COR_SPLIT[s], label=s) for s in bracol.SPLITS],
                   loc="outside upper right", ncols=3)
    return fig


def _empilhar(ax, tabela, rotulos):
    y = np.arange(len(tabela))[::-1]
    esquerda = np.zeros(len(tabela))
    for split in bracol.SPLITS:
        valores = tabela[split].to_numpy(dtype=float)
        ax.barh(y, valores, left=esquerda, height=0.6, color=COR_SPLIT[split],
                edgecolor=SUPERFICIE, linewidth=1.5)
        esquerda += valores
    ax.set_yticks(y, rotulos)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def figura_brilho_e_cor(medidas, df) -> Figure:
    """Por id: luminância (uma faixa por classe), cor média da imagem e tom do fundo."""
    ausentes = fmt.intervalos(df.loc[df["presente"] == 0, "id"])
    alturas = [1] * len(ROTULOS) + [0.45, 1.5]
    with _estilo():
        fig = Figure(figsize=(12, 8.8), layout="constrained")
        fig.suptitle("Brilho e cor média por id (imagens reduzidas a 256 px de largura)",
                     x=0.01, ha="left", color=TINTA)
        eixos = fig.subplots(len(alturas), 1, sharex=True, height_ratios=alturas)
        for ax in eixos:
            for a, b in ausentes:
                ax.axvspan(a - 0.5, b + 0.5, color=FUNDO_AUSENTE, linewidth=0, zorder=0)
        limites = (medidas["luminancia"].min() - 5, medidas["luminancia"].max() + 5)
        for ax, rotulo in zip(eixos, ROTULOS):
            da_classe = medidas[medidas["rotulo"] == rotulo]
            ax.scatter(da_classe["id"], da_classe["luminancia"], s=5, color=COR_CLASSE[rotulo],
                       linewidths=0)
            ax.set_ylim(*limites)
            ax.set_ylabel(rotulo, rotation=0, ha="right", va="center", color=TINTA)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        eixos[0].set_title("luminância média da imagem (0 = preto, 255 = branco), uma faixa por "
                           "classe", loc="left")
        cor = eixos[len(ROTULOS)]
        cor.vlines(medidas["id"], 0, 1, colors=medidas[["r", "g", "b"]].to_numpy() / 255,
                   linewidth=0.9)
        cor.set_ylim(0, 1)
        cor.set_yticks([])
        cor.spines["left"].set_visible(False)
        cor.set_ylabel("cor média\nda imagem", rotation=0, ha="right", va="center", color=TINTA)
        tom = eixos[-1]
        tom.scatter(medidas["id"], medidas["fundo_tom"], s=5, color=TINTA_2, linewidths=0)
        tom.axhline(0, color=EIXO, linewidth=0.8)
        tom.set_ylabel("tom do fundo\n(azul - vermelho)", rotation=0, ha="right", va="center",
                       color=TINTA)
        tom.grid(axis="y")
        tom.set_axisbelow(True)
        tom.set_xlabel("id no dataset.csv (faixas cinza: ids sem imagem)" if ausentes
                       else "id no dataset.csv")
        tom.set_xlim(0, int(df["id"].max()) + 1)
    return fig


# ----------------------------------------------------------------------------- resumo
def resumo(df, medidas, distancias) -> str:
    """O eda_bracol.md: figuras, tabelas e comentários, todos calculados dos dados."""
    presentes = df[df["presente"] == 1]
    elegiveis = df[df["excluida"] == 0]
    classe5 = df[df["rotulo"] == "classe 5"]
    r = ["# EDA do BRACOL", ""]
    r += ["Gerado por `python data/eda_bracol.py` a partir de `data/manifests/bracol.csv`. Não "
          "editar à mão: rode o script de novo. O notebook `notebooks/01_eda.ipynb` mostra as "
          "mesmas tabelas e figuras; as figuras ficam em `data/reports/figures/` para a "
          "documentação (Frente 12).", ""]
    if len(presentes) < len(df):
        r += [f"> **Cópia PARCIAL do BRACOL:** {fmt.n(len(presentes))} de {fmt.n(len(df))} "
              "imagens. Resultados com ela não são comparáveis com Esgario et al. (2020).", ">"]
    else:
        r += [f"> **Cópia completa do BRACOL:** {fmt.n(len(presentes))} imagens. Sem a classe 5 "
              f"são {fmt.n(len(elegiveis))}, as mesmas imagens que os autores usaram, mas a "
              f"divisão é outra (seed {bracol.SEED_PADRAO}): comparar com Esgario et al. (2020) "
              "só com essa ressalva.", ">"]
    r += [f"> **Ressalva:** {bracol.RESSALVA_PHOMA_CERCOSPORA}", ">",
          f"> **Classe 5** (`predominant_stress = 5`, \"undetermined\" no leaf/legend.txt): "
          f"{fmt.n(len(classe5))} linhas, {fmt.n(int(classe5['presente'].sum()))} com imagem. "
          "Fica fora de classificação, severidade e multirrótulo.", ""]
    resolucoes = presentes.groupby(["largura", "altura"]).size()
    if len(resolucoes) == 1:
        (largura, altura), _ = next(iter(resolucoes.items()))
        r += [f"Todas as {fmt.n(len(presentes))} imagens têm {largura}x{altura} pixels, por isso "
              "não há figura de resolução.", ""]
    else:
        lista = "; ".join(f"{w}x{h}: {fmt.n(q)}" for (w, h), q in resolucoes.items())
        r += [f"Resoluções: {lista}.", ""]

    r += _secao_classes(df)
    r += _secao_ids(df)
    r += _secao_severidade(df, elegiveis)
    r += _secao_estresses(df, elegiveis)
    r += ["## 5. Exemplos por classe", "",
          "![Exemplos por classe](figures/05_exemplos_por_classe.jpg)", "",
          f"Quatro imagens de treino por classe, sorteadas de forma determinística (seed "
          f"{bracol.SEED_PADRAO}); a última linha mostra a classe 5, que está excluída. "
          f"{CREDITO}.", ""]
    r += _secao_duplicatas(presentes, distancias)
    r += _secao_divisao(df)
    r += _secao_brilho(medidas)
    return "\n".join(r).rstrip("\n") + "\n"


def _secao_classes(df):
    t = tabela_classes(df)
    r = ["## 1. Classes", "", "![Imagens por classe](figures/01_classes.png)", ""]
    if t["sem_imagem"].sum():
        r += fmt.tabela(["classe", "no csv", "com imagem", "sem imagem", "perda"], [
            [l.Index, fmt.n(l.no_csv), fmt.n(l.com_imagem), fmt.n(l.sem_imagem),
             f"{fmt.decimal(l.perda_pct)}%"] for l in t.itertuples()
        ]) + [""]
        classes = t.loc[bracol.CLASSES].sort_values("perda_pct", ascending=False, kind="stable")
        mais, segunda, menos = classes.index[0], classes.index[1], classes.index[-1]
        perda = classes["perda_pct"].map(fmt.decimal)
        antes = fmt.pct(t.at[mais, "no_csv"], t["no_csv"].sum())
        depois = fmt.pct(t.at[mais, "com_imagem"], t["com_imagem"].sum())
        r += [f"A perda não é uniforme: {mais} perdeu {perda[mais]}% e {segunda}, "
              f"{perda[segunda]}%; {menos} perdeu só {perda[menos]}%. Com isso, {mais} passa de "
              f"{antes} das linhas do csv para {depois} das imagens presentes.", ""]
        return r
    total = int(t["no_csv"].sum())
    r += fmt.tabela(["classe", "imagens", "% do total"],
                    [[l.Index, fmt.n(l.no_csv), fmt.pct(l.no_csv, total)] for l in t.itertuples()])
    classes = t.loc[bracol.CLASSES].sort_values("no_csv", ascending=False, kind="stable")
    maior, menor = classes.index[0], classes.index[-1]
    razao = classes.at[maior, "no_csv"] / classes.at[menor, "no_csv"]
    r += ["", f"A maior classe é {maior} ({fmt.n(classes.at[maior, 'no_csv'])} imagens) e a menor, "
          f"{menor} ({fmt.n(classes.at[menor, 'no_csv'])}): razão de {fmt.decimal(razao)} para 1. "
          f"A classe 5 ({fmt.n(t.at['classe 5', 'no_csv'])} imagens) fica fora.", ""]
    return r


def _secao_ids(df):
    r = ["## 2. Classes ao longo dos ids", "",
         "![Classes ao longo dos ids do dataset.csv](figures/02_ausentes_por_id.png)", ""]
    ausentes = df.loc[df["presente"] == 0, "id"]
    if len(ausentes):
        r += [f"Os ids sem imagem formam as faixas {fmt.faixas(ausentes)}. A causa é o truncamento "
              "do zip publicado, que guarda as imagens em ordem alfabética do nome (ver "
              "`data/README.md`). Como as classes aparecem em blocos de ids no csv, perder faixas "
              "de ids vira perder classes de forma desigual.", ""]
    blocos = []  # [primeiro id, último id, rótulo] de cada sequência de ids da mesma classe
    for i, rotulo in df.sort_values("id")[["id", "rotulo"]].itertuples(index=False):
        if blocos and blocos[-1][2] == rotulo and blocos[-1][1] == i - 1:
            blocos[-1][1] = i
        else:
            blocos.append([i, i, rotulo])
    maiores = sorted(blocos, key=lambda b: (b[0] - b[1], b[0]))[:3]
    lista = "; ".join(f"{a}-{b} ({rotulo}, {fmt.n(b - a + 1)} ids)" for a, b, rotulo in maiores)
    r += [f"As classes aparecem em blocos de ids seguidos: as três sequências mais longas da mesma "
          f"classe são {lista}. Se cada bloco corresponder a uma sessão de foto, classe e sessão "
          "se confundem (ver a seção 8).", ""]
    return r


def _secao_severidade(df, elegiveis):
    t = tabela_severidade(df)
    r = ["## 3. Severidade por classe", "",
         "![Severidade por classe](figures/03_severidade.png)", ""]
    r += fmt.tabela(["classe", *(str(k) for k in t.columns)],
                    [[classe, *(fmt.n(q) for q in linha)] for classe, linha in t.iterrows()])
    r += ["", "Níveis: " + "; ".join(f"{k} = {d}" for k, d in bracol.SEVERIDADES.items())
          + " da área da folha.", ""]
    por_nivel = t.sum(axis=0)
    nivel = int(por_nivel.idxmax())
    raras = [f"{classe}/{k}: {fmt.n(int(t.at[classe, k]))}" for classe in t.index
             if classe != "saudavel" for k in t.columns if k > 0 and t.at[classe, k] < 5]
    r += [f"O nível {nivel}, {bracol.SEVERIDADES[nivel]}, é o mais comum: "
          f"{fmt.n(int(por_nivel[nivel]))} de {fmt.n(len(elegiveis))} imagens elegíveis "
          f"({fmt.pct(por_nivel[nivel], len(elegiveis))}). Combinações de classe e nível com menos "
          f"de 5 imagens: {', '.join(raras) or 'nenhuma'}. Métricas de severidade nessas "
          "combinações ficam muito incertas.", ""]
    return r


def _secao_estresses(df, elegiveis):
    t = tabela_estresses(df)
    r = ["## 4. Mais de um estresse na mesma folha", "",
         "![Estresses marcados por folha](figures/04_estresses_por_folha.png)", ""]
    r += fmt.tabela(["estresses marcados", "imagens"],
                    [[linha.estresses, fmt.n(linha.imagens)] for linha in t.itertuples()])
    multiplos = t[t["estresses"].str.contains("+", regex=False)]
    if len(multiplos):
        total = int(multiplos["imagens"].sum())
        primeira = multiplos.iloc[0]
        r += ["", f"{fmt.n(total)} das {fmt.n(len(elegiveis))} imagens elegíveis "
              f"({fmt.pct(total, len(elegiveis))}) têm mais de um estresse marcado; a combinação "
              f"mais comum é {primeira['estresses']} ({fmt.n(int(primeira['imagens']))}). A classe "
              "do projeto usa só o estresse predominante; as colunas `miner`, `rust`, `phoma` e "
              "`cercospora` guardam o multirrótulo."]
    return r + [""]


def _secao_duplicatas(presentes, distancias):
    exatas = int(presentes["sha256"].duplicated(keep=False).sum())
    grupos = sorted(ids for ids in presentes.groupby("grupo")["id"].apply(sorted) if len(ids) > 1)
    r = ["## 6. Duplicatas e folhas repetidas", "",
         "![Distância de cada imagem até a mais parecida](figures/06_vizinho_mais_proximo.png)",
         ""]
    if len(distancias):
        r += [f"Para cada imagem, a distância até a mais parecida por dois pHash de "
              f"{bracol.PHASH_TAMANHO ** 2} bits: o do quadro inteiro e o do recorte da folha. A "
              f"mais próxima fica a {int(distancias['quadro'].min())} bits no quadro (mediana "
              f"{fmt.decimal(distancias['quadro'].median(), 0)}) e a "
              f"{int(distancias['folha'].min())} bits na folha (mediana "
              f"{fmt.decimal(distancias['folha'].median(), 0)}). Imagens com SHA-256 repetido: "
              f"{fmt.n(exatas)}.", ""]
    descricao = "; ".join("/".join(str(i) for i in ids) for ids in grupos)
    r += [f"Grupos com mais de uma imagem: {fmt.n(len(grupos))}"
          + (f" ({descricao})" if grupos else "") + ". Uma imagem entra no grupo de outra se o "
          f"SHA-256 for igual, se o pHash do quadro ficar a até {bracol.LIMIAR_QUASE_DUPLICATA} "
          f"bits, se o da folha ficar a até {bracol.LIMIAR_QUASE_DUPLICATA_FOLHA} bits, ou se o "
          "par estiver na lista de folhas repetidas conferidas visualmente em 07/10/2026; um grupo "
          "nunca se divide entre splits. A tabela de cada par, com os rótulos e as duas "
          "distâncias, está em `data/reports/integridade_bracol.md`.", ""]
    if len(presentes) > 1:
        itens = presentes[["id", "phash", "phash_folha"]].to_dict("records")
        grupo_de = dict(zip(presentes["id"], presentes["grupo"]))
        perto = {chave: bracol.pares_mais_proximos(itens, k=1, chave=chave, grupos=grupo_de)
                 for chave in ("phash", "phash_folha")}
        if perto["phash"] and perto["phash_folha"]:
            (dq, aq, bq), (dfo, af, bf) = perto["phash"][0], perto["phash_folha"][0]
            r += [f"O pHash do quadro pega a mesma foto re-salva, redimensionada ou com outro "
                  "brilho, mas o fundo e a luz pesam muito nele; o da folha pega boa parte das "
                  "fotos repetidas da mesma folha. Os pares não agrupados mais próximos ficam a "
                  f"{dq} bits no quadro ({aq}/{bq}) e a {dfo} bits na folha ({af}/{bf}).", ""]
    return r


def _secao_divisao(df):
    t = tabela_divisao(df)
    por_nivel = tabela_divisao_por_severidade(df)
    r = ["## 7. Divisão treino/val/teste", "",
         "![Divisão treino/val/teste](figures/07_divisao.png)", ""]
    linhas = [[classe, *(fmt.n(q) for q in linha), fmt.n(int(linha.sum()))]
              for classe, linha in t.iterrows()]
    linhas.append(["**total**", *(fmt.n(int(t[s].sum())) for s in bracol.SPLITS),
                   fmt.n(int(t.to_numpy().sum()))])
    r += fmt.tabela(["classe", *bracol.SPLITS, "total"], linhas) + [""]
    com_imagens = por_nivel[por_nivel.sum(axis=1) > 0]
    fracao_teste = 100 * com_imagens["teste"] / com_imagens.sum(axis=1)
    n_teste = int(t["teste"].sum())
    proporcoes = "/".join(str(p) for p in bracol.PROPORCOES.values())
    texto = (f"Estratificada por classe ({proporcoes}, seed {bracol.SEED_PADRAO}), com os níveis "
             f"de severidade espalhados: a fração de teste fica entre "
             f"{fmt.decimal(fracao_teste.min())}% e {fmt.decimal(fracao_teste.max())}% em todos os "
             f"níveis.")
    if n_teste:
        baixo, alto = _wilson(0.9, n_teste)
        texto += (f" Com {fmt.n(n_teste)} imagens de teste, uma acurácia observada de 90% tem "
                  f"intervalo de confiança de 95% (Wilson) de {fmt.decimal(baixo)}% a "
                  f"{fmt.decimal(alto)}%.")
    return r + [texto, ""]


def _wilson(p, n, z=1.96):
    """Intervalo de Wilson (em %) para uma proporção p observada em n casos."""
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centro - meia), 100 * (centro + meia)


def _secao_brilho(medidas):
    r = ["## 8. Brilho e cor média por id (indícios de sessões de foto)", "",
         "![Brilho e cor média por id](figures/08_brilho_e_cor_por_id.png)", "",
         "Medidas numa versão reduzida de cada imagem (256 px de largura):", "",
         "- **luminância**: média da imagem em tons de cinza, de 0 (preto) a 255 (branco);",
         "- **cor média**: média de R, G e B da imagem inteira (a faixa colorida da figura);",
         "- **fundo**: faixas de cima e de baixo, 1/8 da altura cada, que nessas fotos em geral "
         "só têm fundo. O **tom** é azul menos vermelho: quanto mais negativo, mais amarelado.",
         ""]
    if medidas.empty:
        return r
    faixas = tabela_brilho_por_faixa(medidas)
    r += ["Medianas por faixa de 100 ids (faixas sem imagem não aparecem); a dispersão do tom é a "
          "distância interquartil dentro da faixa:", ""]
    r += fmt.tabela(["ids", "imagens", "luminância", "luminância do fundo", "tom do fundo",
                     "dispersão do tom", "classes mais comuns"],
                    [[l.ids, fmt.n(l.imagens), fmt.decimal(l.luminancia),
                      fmt.decimal(l.fundo_luminancia), fmt.decimal(l.fundo_tom),
                      fmt.decimal(l.fundo_tom_iqr), l.classes]
                     for l in faixas.itertuples()], alinhar="lrrrrrl") + [""]
    classes = tabela_brilho_por_classe(medidas)
    r += ["Medianas por classe:", ""]
    r += fmt.tabela(["classe", "imagens", "luminância", "luminância do fundo", "tom do fundo"],
                    [[l.Index, fmt.n(l.imagens), fmt.decimal(l.luminancia),
                      fmt.decimal(l.fundo_luminancia), fmt.decimal(l.fundo_tom)]
                     for l in classes.itertuples()]) + [""]

    cheias = faixas[faixas["imagens"] >= 20]
    so_classes = classes.drop(index="classe 5", errors="ignore")
    if len(cheias) >= 2 and len(so_classes) >= 2:
        frio = cheias.loc[cheias["fundo_tom"].idxmax()]
        quente = cheias.loc[cheias["fundo_tom"].idxmin()]
        amarela = so_classes["fundo_tom"].idxmin()
        outras = so_classes.drop(index=amarela)["fundo_tom"]
        ids_amarela = medidas.loc[medidas["rotulo"] == amarela, "id"]
        maior = so_classes["imagens"].idxmax()
        da_maior = medidas[medidas["rotulo"] == maior]
        por_faixa = da_maior.groupby((da_maior["id"] - 1) // 100)["fundo_tom"]
        por_faixa = por_faixa.agg(["size", "median"])
        por_faixa = por_faixa[por_faixa["size"] >= 20]["median"]
        tom_amarela = fmt.decimal(so_classes.at[amarela, "fundo_tom"])
        texto = (f"**O que a figura mostra.** Nesta cópia, brilho e cor mudam por blocos de ids: "
                 f"faixas vizinhas têm valores parecidos e há saltos entre blocos. Entre as faixas "
                 f"de 100 ids com pelo menos 20 imagens, a mediana do tom do fundo vai de "
                 f"{fmt.decimal(quente['fundo_tom'])} (ids {quente['ids']}: {quente['classes']}) a "
                 f"{fmt.decimal(frio['fundo_tom'])} (ids {frio['ids']}: {frio['classes']}), e a da "
                 f"luminância do fundo, de {fmt.decimal(cheias['fundo_luminancia'].min())} a "
                 f"{fmt.decimal(cheias['fundo_luminancia'].max())}. Os blocos coincidem em parte "
                 f"com os blocos de classe do csv: {amarela}, com 90% das imagens entre os ids "
                 f"{int(ids_amarela.quantile(0.05))} e {int(ids_amarela.quantile(0.95))}, tem o "
                 f"fundo mais amarelado (mediana {tom_amarela}) "
                 f"do que as outras classes ({fmt.decimal(outras.min())} a "
                 f"{fmt.decimal(outras.max())}).")
        if len(por_faixa) >= 2:
            texto += (f" Dentro de uma mesma classe o tom também varia entre faixas: em {maior}, "
                      f"a mediana por faixa com pelo menos 20 imagens dessa classe vai de "
                      f"{fmt.decimal(por_faixa.min())} a {fmt.decimal(por_faixa.max())}.")
        homogeneas = cheias[cheias["fundo_tom_iqr"] <= 2]
        dispersas = cheias[cheias["fundo_tom_iqr"] > 2]
        if len(dispersas):
            texto += (f" Em {len(homogeneas)} das {len(cheias)} faixas o tom do fundo é homogêneo "
                      f"(dispersão de no máximo 2); nas outras ({', '.join(dispersas['ids'])}), a "
                      "figura mostra um bloco que termina no meio da faixa ou dois níveis que se "
                      "alternam.")
        r += [texto, ""]
    r += ["**O que a figura não mostra.** Não prova que houve sessões de foto diferentes: o "
          "BRACOL não traz data, câmera nem planta, e a média da imagem inteira também depende do "
          "tamanho da folha e da cor das lesões. O padrão é compatível com sessões diferentes "
          "(luz, câmera ou balanço de branco), algumas concentradas em uma classe. Se for isso, um "
          "modelo pode aprender a cor do fundo junto com a lesão, e a divisão aleatória por "
          "imagem põe fotos da mesma sessão em treino e teste. Vale checar na Frente 9: Grad-CAM "
          "nas imagens de teste e acurácia por faixa de ids.", ""]
    return r


# ---------------------------------------------------------------------------- execução
def gerar(manifest=bracol.MANIFEST_BRACOL, saida=bracol.PASTA_REPORTS,
          raiz=bracol.RAIZ_REPO) -> list[Path]:
    """Gera as figuras em <saida>/figures/ e o <saida>/eda_bracol.md. Devolve os arquivos."""
    df = carregar(manifest)
    medidas = medir_brilho_e_cor(df, raiz)
    distancias = distancias_ao_vizinho(df)
    pasta = Path(saida) / "figures"
    pasta.mkdir(parents=True, exist_ok=True)
    figuras = {
        "01_classes.png": figura_classes(df),
        "02_ausentes_por_id.png": figura_ausentes_por_id(df),
        "03_severidade.png": figura_severidade(df),
        "04_estresses_por_folha.png": figura_estresses(df),
        "05_exemplos_por_classe.jpg": figura_exemplos(df, raiz),
        "06_vizinho_mais_proximo.png": figura_vizinho_mais_proximo(distancias),
        "07_divisao.png": figura_divisao(df),
        "08_brilho_e_cor_por_id.png": figura_brilho_e_cor(medidas, df),
    }
    gravados = []
    for nome, fig in figuras.items():
        _salvar(fig, pasta / nome)
        gravados.append(pasta / nome)
    md = Path(saida) / "eda_bracol.md"
    md.write_text(resumo(df, medidas, distancias), encoding="utf-8", newline="\n")
    return [*gravados, md]


def _salvar(fig, caminho: Path) -> None:
    """PNG sem o metadado "Software" (versão do matplotlib) e JPEG com qualidade fixa: os
    bytes só mudam se a figura mudar."""
    if caminho.suffix == ".png":
        fig.savefig(caminho, dpi=110, facecolor=SUPERFICIE, metadata={"Software": None})
    else:
        fig.savefig(caminho, dpi=100, facecolor=SUPERFICIE,
                    pil_kwargs={"quality": 85, "optimize": True})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gera as figuras e o resumo da EDA do BRACOL (ver data/README.md)."
    )
    ap.add_argument("--manifest", type=Path, default=bracol.MANIFEST_BRACOL,
                    help="manifest (padrao: data/manifests/bracol.csv)")
    ap.add_argument("--saida", type=Path, default=bracol.PASTA_REPORTS,
                    help="pasta de saida (padrao: data/reports)")
    args = ap.parse_args(argv)
    fmt.dizer("Medindo brilho e cor (imagens reduzidas a 256 px) e gerando as figuras...")
    for caminho in gerar(args.manifest.resolve(), args.saida.resolve()):
        mostrado = (caminho.relative_to(bracol.RAIZ_REPO).as_posix()
                    if caminho.is_relative_to(bracol.RAIZ_REPO) else str(caminho))
        fmt.dizer(f"  {mostrado}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
