"""
Avaliação ponta a ponta da inferência (Frente 9, passo 3), gravada em model/runs/<nome>/:

- Constantes medidas no TREINO do BRACOL: o cinza do fundo (borda das fotos), a margem do
  recorte (comprimento da folha sobre a largura da foto) e a cobertura da caixa nas fotos de
  folha única; e, no treino do BRACOT (sem a validação), a cobertura da maior caixa.
- Calibração do limiar de confiança (RF07): curva risco-cobertura com as predições fora da dobra
  do diagnóstico por blocos (protocolo aleatório) e com a validação do classificador avaliado.
- T1: as imagens de validação do BRACOL pela cadeia (recortes A e B) x a imagem inteira.
- T2: as fotos de validação do BRACOT (cenas inteiras do treino dos autores): classes,
  severidades e confiança por folha, nos recortes A e B; folhas cortadas pela borda x inteiras.
- T3: tempo da cadeia em CPU (RNF02).

Os testes do BRACOL e do BRACOT ficam fechados: nada aqui lê o split de teste do BRACOL nem
as anotações de teste do BRACOT.

Uso (de qualquer pasta, com o venv ativo):
    python model/avaliar_pipeline.py                    tudo, em model/runs/pipeline_base/
    python model/avaliar_pipeline.py --so-constantes    só mede e mostra as constantes
    python model/avaliar_pipeline.py --classificador PASTA --detector PESOS --nome NOME
                                                        outros modelos (recalibrar o limiar)
    python model/avaliar_pipeline.py --recortes PASTA   grava também os recortes do T2 e duas
                                                        pranchas em PASTA, fora do repositório

O código de saída é 0 sem erros e 1 com erros.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import atalho
import bracol
import bracot
import dados
import detector
import formatacao as fmt
import inferencia as inf
import metrics
import train
import treinar_detector as td

NOME_PADRAO = "pipeline_base"
PASTA_FORA_DA_DOBRA = train.PASTA_RUNS / "diagnostico_blocos"
ALVO_ACURACIA = 0.95  # acurácia das predições mantidas que a proposta de limiar busca
LIMIARES_CURVA = np.round(np.arange(0.30, 1.0, 0.01), 2)
LIMIARES_T2 = (0.5, 0.6, 0.7, 0.8, 0.9)
LIMIARES_DESTAQUE = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
AMOSTRA_FOLHA_UNICA = 200  # imagens de treino do BRACOL para a margem e a cobertura
BORDA_FUNDO = 0.08  # faixa de borda do tom do fundo (a da checagem de atalho)
RECORTES_POR_PRANCHA = 12
CREDITO_BRACOT = ("Fotos: BRACOT (Krohling, Tozzi de Souza e Tassis, 2021), CC BY 4.0, "
                  "DOI 10.17632/pmkbyjpf6k.1")
SEED = bracol.SEED_PADRAO
MODOS = ("A", "B")
VARIANTES_T1 = ("inteira", "cadeia_A", "cadeia_B", "recorte_A", "recorte_B")
DESCRICAO_T1 = {
    "inteira": "imagem inteira (a referência)",
    "cadeia_A": "cadeia, recorte A",
    "cadeia_B": "cadeia, recorte B",
    "recorte_A": "só o recorte A (sem a regra 1)",
    "recorte_B": "só o recorte B (sem a regra 1)",
}
DECISOES = (
    ("10/10/2026", "o detector está bom o bastante: AP50 de máscara de 94,7% (região) e 88,1% "
                   "(conservador), limiar 0,35."),
    ("10/10/2026", "classificar também as folhas cortadas pela borda, com área mínima e corte "
                   "por confiança do classificador; reavaliar o filtro de borda olhando os "
                   "recortes."),
    ("10/10/2026", "folha única: a foto inteira só se houver detecção com score >= 0,05; sem "
                   "nenhuma, planta_nao_identificada."),
    ("10/10/2026", "arquivo_muito_grande acima de ~64 megapixels, com as dimensões lidas do "
                   "cabeçalho."),
)


# ------------------------------------------------------------------------ constantes
def medir_constantes(modelos, config) -> dict:
    """As constantes da cadeia medidas em dados de treino (nada é escolhido na validação)."""
    treino = dados.montar_treino()
    fmt.dizer(f"Constantes: tom do fundo em {len(treino)} imagens de treino do BRACOL...")
    fundo = medir_fundo(treino)
    amostra = dados.amostra_estratificada(treino, AMOSTRA_FOLHA_UNICA, SEED)
    fmt.dizer(f"Constantes: folhas de {len(amostra)} imagens de treino do BRACOL...")
    folha = medir_folha_unica(amostra, modelos, config)
    fmt.dizer("Constantes: fotos de treino do BRACOT (sem a validacao)...")
    plantas = medir_plantas(modelos, config)
    fmt.dizer("Constantes: cor das folhas anotadas no treino do BRACOT...")
    cor = medir_cor_das_folhas(config)
    return {"fundo": fundo, "folha_unica_bracol": folha, "plantas_bracot": plantas,
            "cor_folhas_bracot": cor}


def medir_fundo(treino) -> dict:
    """Mediana do tom da borda (BORDA_FUNDO) de cada imagem, já em 448x224, por classe; o cinza
    é a média das medianas de luminância das classes (cada classe com o mesmo peso)."""
    ver = dados.transformacao_sem_normalizar()
    linhas, borda = [], None
    for linha in treino.itertuples():
        rgb = ver(image=dados.ler_imagem(RAIZ_REPO / linha.caminho))["image"]
        if borda is None:
            borda = atalho.mascara_borda(rgb.shape[0], rgb.shape[1], BORDA_FUNDO)
        pixels = rgb[borda].astype(np.float64)
        r, g, b = np.median(pixels, axis=0)
        linhas.append({"classe": linha.classe, "r": r, "g": g, "b": b,
                       "luminancia": float(np.median(pixels.mean(axis=1)))})
    tabela = pd.DataFrame(linhas).groupby("classe")[["r", "g", "b", "luminancia"]].median()
    tabela = tabela.reindex(bracol.CLASSES)
    return {"imagens": len(linhas), "borda": BORDA_FUNDO,
            "por_classe": tabela.round(1).to_dict("index"),
            "cinza": int(round(tabela["luminancia"].mean()))}


def medir_folha_unica(amostra, modelos, config) -> dict:
    """Nas fotos de folha única do BRACOL: detecções utilizáveis, comprimento e espessura da
    maior folha (relativos à largura e à altura da foto) e cobertura da caixa dela."""
    linhas, cores = [], []
    for linha in amostra.itertuples():
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        altura, largura = imagem.shape[:2]
        deteccoes = inf.detectar(imagem, config, modelos)
        cores += [d["colorida"] for d in deteccoes
                  if d["score"] >= config.limiar_deteccao and d["area"] >= config.area_minima]
        uteis = _utilizaveis(deteccoes, config)
        if not uteis:
            linhas.append({"deteccoes": 0})
            continue
        maior = max(uteis, key=lambda d: d["area"])
        _, _, comprimento, espessura = inf.eixo_da_folha(maior["poligono"])
        linhas.append({"deteccoes": len(uteis), "comprimento": comprimento / largura,
                       "espessura": espessura / altura,
                       "cobertura": inf.cobertura_da_caixa(maior["caixa"], altura, largura)})
    t = pd.DataFrame(linhas)
    com = t[t["deteccoes"] > 0]
    comprimento = float(com["comprimento"].median())
    return {"imagens": len(t), "sem_deteccao": int((t["deteccoes"] == 0).sum()),
            "uma_deteccao": int((t["deteccoes"] == 1).sum()),
            "mais_de_uma": int((t["deteccoes"] > 1).sum()),
            "comprimento_mediano": comprimento,
            "espessura_mediana": float(com["espessura"].median()),
            "margem": (1 / comprimento - 1) / 2,
            "cobertura": _percentis(com["cobertura"]),
            "deteccoes_antes_da_cor": len(cores),
            "cortadas_pela_cor": int(sum(c < config.fracao_colorida_minima for c in cores)),
            "colorida_histograma": np.histogram(cores, bins=np.linspace(0, 1, 11))[0].tolist()}


def medir_plantas(modelos, config) -> dict:
    """Nas fotos de treino do BRACOT (sem as de validação): cobertura da maior caixa e quantas
    fotos têm uma só folha utilizável."""
    manifest = bracot.ler_manifest(split="treino")
    validacao = set(td.dividir_validacao(manifest))
    linhas = []
    for linha in manifest[~manifest["id"].isin(validacao)].itertuples():
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        altura, largura = imagem.shape[:2]
        uteis = _utilizaveis(inf.detectar(imagem, config, modelos), config)
        linhas.append({"deteccoes": len(uteis), "cobertura": max(
            (inf.cobertura_da_caixa(d["caixa"], altura, largura) for d in uteis), default=0.0)})
    t = pd.DataFrame(linhas)
    return {"fotos": len(t), "uma_deteccao": int((t["deteccoes"] == 1).sum()),
            "sem_deteccao": int((t["deteccoes"] == 0).sum()),
            "cobertura_maior": _percentis(t["cobertura"])}


def medir_cor_das_folhas(config, fotos: int = 80) -> dict:
    """Nas folhas anotadas de fotos de treino do BRACOT (sem a validação, sorteadas com a
    semente): a fração da máscara com saturação >= saturacao_minima, que o filtro de cor usa."""
    manifest = bracot.ler_manifest(split="treino")
    validacao = set(td.dividir_validacao(manifest))
    poligonos = td.ler_poligonos("treino")
    amostra = manifest[~manifest["id"].isin(validacao)].sample(fotos, random_state=SEED)
    fracoes = []
    for linha in amostra.itertuples():
        reduzida, escala = inf.reduzir(dados.ler_imagem(RAIZ_REPO / linha.caminho), inf.LADO_COR)
        saturacao = cv2.cvtColor(reduzida, cv2.COLOR_RGB2HSV)[..., 1]
        fracoes += [inf.fracao_colorida(saturacao, p * escala, config.saturacao_minima)
                    for p in poligonos[linha.id]]
    return {"fotos": fotos, "folhas": len(fracoes), **_percentis(fracoes),
            "p1": float(np.quantile(fracoes, 0.01)),
            "abaixo_do_minimo": int(sum(f < config.fracao_colorida_minima for f in fracoes))}


def _utilizaveis(deteccoes, config) -> list[dict]:
    return [d for d in deteccoes if inf.utilizavel(d, config)]


def _percentis(serie) -> dict:
    s = pd.Series(serie, dtype=float)
    return {"min": float(s.min()), "p5": float(s.quantile(0.05)), "p25": float(s.quantile(0.25)),
            "mediana": float(s.median()), "p95": float(s.quantile(0.95)), "max": float(s.max())}


# ----------------------------------------------------------------------- calibração
def ler_fora_da_dobra(pasta: Path) -> pd.DataFrame:
    """As predições fora da dobra do protocolo aleatório do diagnóstico por blocos."""
    arquivos = sorted(Path(pasta).glob("predicoes_aleatorio_*.csv"))
    if not arquivos:
        raise train.ErroFatal(f"sem predicoes_aleatorio_*.csv em {train._exibir(Path(pasta))}")
    return pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)


def curva_das_probabilidades(probabilidades, verdadeiras) -> list[dict]:
    probabilidades = np.asarray(probabilidades, dtype=np.float64)
    verdadeiras = np.asarray(verdadeiras)
    return metrics.curva_risco_cobertura(
        probabilidades.max(axis=1), probabilidades.argmax(axis=1) == verdadeiras,
        LIMIARES_CURVA, verdadeiras)


def propor_limiar(curva, alvo: float = ALVO_ACURACIA) -> dict | None:
    """O menor limiar em que a acurácia das mantidas chega ao alvo."""
    return next((p for p in curva if p["mantidas"] and p["acuracia"] >= alvo), None)


def calibrar(pasta_fora_da_dobra: Path, inteira_val: pd.DataFrame) -> dict:
    fora = ler_fora_da_dobra(pasta_fora_da_dobra)
    colunas = [f"prob_{c}" for c in bracol.CLASSES]
    verdadeiras = fora["classe_verdadeira"].map(bracol.CLASSES.index)
    curva_fora = curva_das_probabilidades(fora[colunas].to_numpy(), verdadeiras)
    curva_val = curva_das_probabilidades(np.stack(inteira_val["probs"]),
                                         inteira_val["classe"].to_numpy())
    proposta = propor_limiar(curva_fora)
    return {"fora_da_dobra": {"pasta": train._exibir(Path(pasta_fora_da_dobra)),
                              "n": len(fora),
                              "acuracia": float(np.mean(
                                  fora[colunas].to_numpy().argmax(axis=1) == verdadeiras)),
                              "curva": curva_fora},
            "validacao": {"n": len(inteira_val), "curva": curva_val},
            "alvo": ALVO_ACURACIA,
            "proposta": None if proposta is None else proposta["limiar"]}


def figura_risco_cobertura(calibracao: dict, limiar_em_uso: float):
    """Acurácia das mantidas x cobertura nas duas fontes, com os limiares em destaque."""
    import eda_bracol as eda

    series = (("fora_da_dobra", "fora da dobra (1.432, protocolo aleatório)", eda.AZUL),
              ("validacao", "validação do classificador (252)", eda.COR_CLASSE["ferrugem"]))
    with eda._estilo():
        fig = eda._figura(7.6, 4.4, "Curva risco-cobertura do limiar de confiança (RF07)")
        ax = fig.add_subplot()
        menor = min(p["cobertura"] for chave, _, _ in series
                    for p in calibracao[chave]["curva"])
        inicio = max(0.0, math.floor((menor - 0.02) * 20) / 20)
        ax.axhline(ALVO_ACURACIA, color=eda.EIXO, linewidth=0.8, linestyle="--")
        ax.text(inicio + 0.005, ALVO_ACURACIA + 0.002, f"alvo {fmt.pct(ALVO_ACURACIA, 1, 0)}",
                color=eda.TINTA_2, va="bottom")
        for chave, rotulo, cor in series:
            curva = calibracao[chave]["curva"]
            x = [p["cobertura"] for p in curva]
            y = [p["acuracia"] for p in curva]
            ax.plot(x, y, color=cor, linewidth=2, label=rotulo)
            for p in curva:
                if any(abs(p["limiar"] - d) < 1e-9 for d in LIMIARES_DESTAQUE):
                    ax.plot(p["cobertura"], p["acuracia"], "o", color=cor, markersize=5,
                            markeredgecolor=eda.SUPERFICIE, markeredgewidth=1)
                    if chave == "fora_da_dobra":
                        ax.annotate(fmt.decimal(p["limiar"], 2), (p["cobertura"], p["acuracia"]),
                                    textcoords="offset points", xytext=(4, -11),
                                    color=eda.TINTA_2, fontsize=8)
        proposta = calibracao["proposta"]
        if proposta is not None:
            p = next(q for q in calibracao["fora_da_dobra"]["curva"]
                     if abs(q["limiar"] - proposta) < 1e-9)
            ax.plot(p["cobertura"], p["acuracia"], "o", color=eda.TINTA, markersize=9,
                    markerfacecolor="none", markeredgewidth=1.5)
            ax.annotate(f"proposta {fmt.decimal(proposta, 2)}", (p["cobertura"], p["acuracia"]),
                        textcoords="offset points", xytext=(8, 10), ha="left", color=eda.TINTA)
        ax.set_xlim(inicio, 1.005)
        ax.set_xlabel("cobertura (fração das imagens mantidas)")
        ax.set_ylabel("acurácia das mantidas")
        ax.xaxis.set_major_formatter(lambda v, _: fmt.pct(v, 1, 0))
        ax.yaxis.set_major_formatter(lambda v, _: fmt.pct(v, 1, 0))
        ax.grid(axis="y")
        ax.legend(loc="lower left")
        ax.set_title(f"pontos: limiares {', '.join(fmt.decimal(d, 2) for d in LIMIARES_DESTAQUE)};"
                     f" em uso na cadeia: {fmt.decimal(limiar_em_uso, 2)}", loc="left")
    return fig


# --------------------------------------------------------------------------- T1
def t1(modelos, config) -> pd.DataFrame:
    """Cada imagem de validação do BRACOL pela imagem inteira, pela cadeia (regras completas) e
    só pelo recorte da maior folha (sem a regra 1), nos modos A e B, sem limiar de confiança."""
    val = dados.montar_val()
    cadeia = replace(config, limiar_confianca=0.0)
    so_recorte = replace(cadeia, cobertura_folha_unica=math.inf)
    linhas = []
    for n, linha in enumerate(val.itertuples(), start=1):
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        altura, largura = imagem.shape[:2]
        deteccoes = inf.detectar(imagem, cadeia, modelos)
        preparo = inf.decidir(deteccoes, altura, largura, cadeia)
        preparo_recorte = inf.decidir(deteccoes, altura, largura, so_recorte)
        uteis = _utilizaveis(deteccoes, cadeia)
        registro = {"id": linha.id, "classe": bracol.CLASSES.index(linha.classe),
                    "severidade": int(linha.severity), "deteccoes": len(uteis),
                    "deteccoes_baixas": sum(d["score"] < cadeia.limiar_deteccao
                                            for d in deteccoes),
                    "cobertura_maior": max((inf.cobertura_da_caixa(d["caixa"], altura, largura)
                                            for d in uteis), default=0.0),
                    "cortadas_por_cor": preparo["contagens"]["cortadas_por_cor"],
                    "plano": preparo["plano"], "regra": preparo["regra"],
                    "folhas": len(preparo["folhas"])}
        inteira = _prever([inf.recorte_da_foto_inteira(imagem, config.cinza_fundo)], modelos)[0]
        _registrar(registro, "inteira", inteira)
        for modo in MODOS:
            recorte = _folha_principal(imagem, preparo_recorte, cadeia, modo, modelos, inteira)
            _registrar(registro, f"recorte_{modo}", recorte)
            principal = {"foto_inteira": inteira, "deteccao": recorte}.get(preparo["plano"])
            _registrar(registro, f"cadeia_{modo}", principal)
        linhas.append(registro)
        if n % 50 == 0:
            fmt.dizer(f"  T1: {n}/{len(val)}")
    return pd.DataFrame(linhas)


def _folha_principal(imagem, preparo, config, modo, modelos, inteira):
    """A previsão da maior folha do preparo (a foto inteira, se for esse o plano)."""
    if preparo["plano"] == "foto_inteira":
        return inteira
    if not preparo["folhas"]:
        return None
    previsoes = _prever(inf.recortar(imagem, preparo, config, modo), modelos)
    return previsoes[int(np.argmax([f["area"] for f in preparo["folhas"]]))]


def _prever(recortes, modelos) -> list[dict]:
    probabilidades, niveis = inf.classificar(recortes, modelos)
    return [{"probs": p, "classe": int(np.argmax(p)), "confianca": float(np.max(p)),
             "nivel": int(nivel)} for p, nivel in zip(probabilidades, niveis)]


def _registrar(registro: dict, prefixo: str, previsao) -> None:
    registro[f"{prefixo}_classe"] = None if previsao is None else previsao["classe"]
    registro[f"{prefixo}_confianca"] = None if previsao is None else previsao["confianca"]
    registro[f"{prefixo}_nivel"] = None if previsao is None else previsao["nivel"]
    registro[f"{prefixo}_probs"] = None if previsao is None else previsao["probs"]


def metricas_t1(tabela: pd.DataFrame, limiar: float) -> dict:
    resultado = {}
    for variante in VARIANTES_T1:
        com = tabela[tabela[f"{variante}_classe"].notna()]
        previstas = com[f"{variante}_classe"].astype(int).to_numpy()
        m = metrics.calcular_metricas(com["classe"].to_numpy(), previstas)
        ms = metrics.calcular_metricas_severidade(
            com["severidade"].to_numpy(),
            np.where(previstas == 0, 0, np.maximum(com[f"{variante}_nivel"].astype(int), 1)))
        mantidas = com[f"{variante}_confianca"] >= limiar
        resultado[variante] = {
            "cobertura_sem_limiar": len(com) / len(tabela), "metricas": m, "severidade": ms,
            "limiar": limiar, "mantidas": int(mantidas.sum()),
            "cobertura_no_limiar": float(mantidas.sum() / len(tabela)),
            "acuracia_no_limiar": float(np.mean(previstas[mantidas.to_numpy()]
                                                == com["classe"].to_numpy()[mantidas.to_numpy()]))
            if mantidas.any() else None}
    return resultado


# --------------------------------------------------------------------------- T2
def t2(modelos, config, pasta_recortes: Path | None = None):
    """Cada folha das fotos de validação do BRACOT, recortada nos modos A e B."""
    manifest = bracot.ler_manifest(split="treino")
    caminhos = manifest.set_index("id")["caminho"]
    cadeia = replace(config, limiar_confianca=0.0)
    folhas, fotos = [], []
    if pasta_recortes:
        (pasta_recortes / "recortes").mkdir(parents=True, exist_ok=True)
    for foto in td.dividir_validacao(manifest):
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        preparo = inf.decidir(inf.detectar(imagem, cadeia, modelos), imagem.shape[0],
                              imagem.shape[1], cadeia)
        fotos.append({"foto": foto, "plano": preparo["plano"], "regra": preparo["regra"],
                      "folhas": len(preparo["folhas"]), **preparo["contagens"]})
        if not preparo["folhas"]:
            continue
        recortes = {modo: inf.recortar(imagem, preparo, cadeia, modo) for modo in MODOS}
        previsoes = {modo: _prever(recortes[modo], modelos) for modo in MODOS}
        for k, folha in enumerate(preparo["folhas"]):
            linha = {"foto": foto, "folha": k + 1, "plano": preparo["plano"],
                     "score": folha["score"], "area": folha["area"], "borda": folha["borda"]}
            for modo in MODOS:
                p = previsoes[modo][k]
                categoria = inf.CATEGORIAS[p["classe"]]
                linha.update({f"classe_{modo}": categoria, f"confianca_{modo}": p["confianca"],
                              f"severidade_{modo}": inf.severidade_do_contrato(categoria,
                                                                              p["nivel"])})
            folhas.append(linha)
            if pasta_recortes:
                _gravar_recorte(pasta_recortes / "recortes" / f"{foto}_{k + 1:02d}.jpg",
                                [recortes[m][k] for m in MODOS], linha, modelos)
    return pd.DataFrame(folhas), pd.DataFrame(fotos)


def _entrada_do_modelo(recorte, modelos) -> np.ndarray:
    """O recorte como o classificador o vê, antes da normalização (448x224, INTER_AREA)."""
    redimensionar = modelos.transformacao.transforms[0]
    altura, largura = redimensionar.height, redimensionar.width
    return cv2.resize(recorte, (largura, altura), interpolation=dados.INTERPOLACAO)


def _gravar_recorte(caminho: Path, recortes, linha: dict, modelos) -> None:
    """A | B lado a lado, como o classificador vê, com a previsão (texto ASCII)."""
    partes = []
    for modo, recorte in zip(MODOS, recortes):
        img = cv2.copyMakeBorder(_entrada_do_modelo(recorte, modelos), 22, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        texto = (f"{modo}: {linha[f'classe_{modo}']} / {linha[f'severidade_{modo}']} "
                 f"{linha[f'confianca_{modo}']:.2f}")
        cv2.putText(img, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        partes.append(img)
    junto = cv2.hconcat([partes[0], np.full((partes[0].shape[0], 8, 3), 255, np.uint8),
                         partes[1]])
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(junto, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        caminho.write_bytes(jpeg.tobytes())


def escolher_para_pranchas(folhas: pd.DataFrame, n: int) -> pd.DataFrame:
    """n folhas variadas: um sorteio (semente) por classe prevista no modo A, em rodízio, com
    metade das vagas, quando houver, para folhas cortadas pela borda."""
    gerador = np.random.default_rng(SEED)
    embaralhadas = folhas.iloc[gerador.permutation(len(folhas))]
    filas = {(c, b): list(g.index) for (c, b), g in embaralhadas.groupby(["classe_A", "borda"])}
    escolhidas = []
    while len(escolhidas) < min(n, len(folhas)):
        for chave in sorted(filas, key=lambda k: (k[1], k[0])):
            if filas[chave] and len(escolhidas) < n:
                escolhidas.append(filas[chave].pop(0))
    return folhas.loc[escolhidas]


def gravar_pranchas(folhas_escolhidas: pd.DataFrame, pasta: Path) -> list[Path]:
    """Pranchas JPEG de RECORTES_POR_PRANCHA folhas cada (A | B), com a previsão."""
    import eda_bracol as eda

    caminhos = []
    blocos = [folhas_escolhidas.iloc[i:i + RECORTES_POR_PRANCHA]
              for i in range(0, len(folhas_escolhidas), RECORTES_POR_PRANCHA)]
    for numero, bloco in enumerate(blocos, start=1):
        with eda._estilo():
            fig = eda._figura(16, 2.05 * math.ceil(len(bloco) / 2) + 0.6,
                              f"Recortes do T2 (fotos de validação do BRACOT), prancha {numero}: "
                              "modo A (foto em volta) | modo B (fundo cinza)")
            eixos = fig.subplots(math.ceil(len(bloco) / 2), 2, squeeze=False)
            for ax in eixos.ravel():
                ax.axis("off")
            for ax, linha in zip(eixos.ravel(), bloco.itertuples()):
                imagem = cv2.imdecode(np.fromfile(
                    pasta / "recortes" / f"{linha.foto}_{linha.folha:02d}.jpg", np.uint8),
                    cv2.IMREAD_COLOR)[22:, :, ::-1]
                ax.imshow(imagem)
                ax.set_title(
                    f"{linha.foto}, folha {linha.folha}"
                    f"{' (cortada na borda)' if linha.borda else ''}"
                    f"\nA: {linha.classe_A} / {linha.severidade_A}, "
                    f"{fmt.pct(linha.confianca_A, 1, 0)}    B: {linha.classe_B} / "
                    f"{linha.severidade_B}, {fmt.pct(linha.confianca_B, 1, 0)}",
                    loc="left", fontsize=8)
            fig.text(0.01, 0.002, CREDITO_BRACOT, fontsize=7, color=eda.TINTA_2)
        caminho = pasta / f"prancha_{numero}.jpg"
        eda._salvar(fig, caminho)
        caminhos.append(caminho)
    return caminhos


def resumo_t2(folhas: pd.DataFrame, fotos: pd.DataFrame, limiar: float) -> dict:
    resultado = {"fotos": len(fotos), "folhas": len(folhas),
                 "planos": fotos["plano"].value_counts().to_dict(),
                 "cortadas_por_area": int(fotos["cortadas_por_area"].sum()),
                 "cortadas_por_limite": int(fotos["cortadas_por_limite"].sum()),
                 "deteccoes": int(fotos["deteccoes"].sum()),
                 "na_borda": int(folhas["borda"].sum())}
    for modo in MODOS:
        conf = folhas[f"confianca_{modo}"]
        borda = folhas["borda"].astype(bool)
        resultado[modo] = {
            "classes": folhas[f"classe_{modo}"].value_counts().reindex(
                bracol.CLASSES, fill_value=0).to_dict(),
            "severidades": folhas[f"severidade_{modo}"].value_counts().reindex(
                inf.NIVEIS_CONTRATO, fill_value=0).to_dict(),
            "confianca_media": float(conf.mean()), "confianca_mediana": float(conf.median()),
            "abaixo": {f"{t:g}": float((conf < t).mean())
                       for t in sorted({*LIMIARES_T2, limiar})},
            "borda": {"folhas": int(borda.sum()), "confianca_media": float(conf[borda].mean()),
                      "abaixo_do_limiar": float((conf[borda] < limiar).mean())},
            "inteiras": {"folhas": int((~borda).sum()),
                         "confianca_media": float(conf[~borda].mean()),
                         "abaixo_do_limiar": float((conf[~borda] < limiar).mean())},
        }
    resultado["concordancia_A_B"] = float((folhas["classe_A"] == folhas["classe_B"]).mean())
    resultado["cortadas_por_cor"] = int(fotos["cortadas_por_cor"].sum())
    maxima = folhas.groupby("foto")[[f"confianca_{m}" for m in MODOS]].max()
    resultado["fotos_baixa_confianca"] = {
        modo: {f"{t:g}": int((maxima[f"confianca_{modo}"] < t).sum())
               for t in sorted({*LIMIARES_T2, limiar})} for modo in MODOS}
    return resultado


# --------------------------------------------------------------------------- T3
def t3(pasta_classificador: Path, pesos_detector: Path, config) -> tuple[pd.DataFrame, dict]:
    """Tempo da cadeia em CPU, por foto de validação do BRACOT: leitura, detecção, recortes,
    classificação (um lote) e contrato. Os modelos de CPU são carregados à parte (o detector numa
    instância nova, para não herdar a da GPU). A primeira foto aquece e fica de fora. Pior caso:
    a classificação de max_folhas recortes."""
    import torch

    inicio = time.perf_counter()
    modelos = inf.carregar_modelos(pasta_classificador, pesos_detector, "cpu")
    modelos = replace(modelos, detector=detector.importar_ultralytics()(str(pesos_detector)))
    carga = time.perf_counter() - inicio
    manifest = bracot.ler_manifest(split="treino")
    caminhos = manifest.set_index("id")["caminho"]
    ids = td.dividir_validacao(manifest)
    linhas = []
    for k, foto in enumerate([ids[0], *ids]):
        t0 = time.perf_counter()
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        t1_ = time.perf_counter()
        deteccoes = inf.detectar(imagem, config, modelos)
        t2_ = time.perf_counter()
        preparo = inf.decidir(deteccoes, imagem.shape[0], imagem.shape[1], config)
        recortes = inf.recortar(imagem, preparo, config) if preparo["folhas"] else []
        t3_ = time.perf_counter()
        if recortes:
            probabilidades, niveis = inf.classificar(recortes, modelos)
            t4_ = time.perf_counter()
            resposta = inf.montar_resposta(preparo, probabilidades, niveis, config, modelos)
        else:
            t4_ = time.perf_counter()
            resposta = inf.resposta_de_erro("planta_nao_identificada")
        t5_ = time.perf_counter()
        if k:
            linhas.append({"foto": foto, "folhas": len(recortes), "leitura_s": t1_ - t0,
                           "deteccao_s": t2_ - t1_, "recortes_s": t3_ - t2_,
                           "classificacao_s": t4_ - t3_, "contrato_s": t5_ - t4_,
                           "total_s": t5_ - t0, "status": resposta["status"]})
    tabela = pd.DataFrame(linhas)
    # Pior caso: max_folhas recortes (os da foto repetidos) num lote só.
    pior = []
    for foto in ids[:10]:
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        preparo = inf.decidir(inf.detectar(imagem, config, modelos), imagem.shape[0],
                              imagem.shape[1], config)
        recortes = inf.recortar(imagem, preparo, config) if preparo["folhas"] else []
        if not recortes:
            continue
        recortes = (recortes * config.max_folhas)[:config.max_folhas]
        t0 = time.perf_counter()
        inf.classificar(recortes, modelos)
        pior.append(time.perf_counter() - t0)
    etapas = [c for c in tabela.columns if c.endswith("_s")]
    por_folha = (tabela["recortes_s"] / tabela["folhas"].clip(lower=1)).median()
    resumo = {
        "threads_torch": torch.get_num_threads(), "fotos": len(tabela),
        "carga_dos_modelos_s": carga,
        "mediana": {c: float(tabela[c].median()) for c in etapas},
        "maximo": {c: float(tabela[c].max()) for c in etapas},
        "folhas_mediana": float(tabela["folhas"].median()),
        "folhas_maximo": int(tabela["folhas"].max()),
        "pior_caso": {
            "folhas": config.max_folhas,
            "classificacao_mediana_s": float(np.median(pior)),
            "classificacao_maximo_s": float(np.max(pior)),
            "total_estimado_s": float(tabela["leitura_s"].max() + tabela["deteccao_s"].max()
                                      + por_folha * config.max_folhas + np.max(pior)),
        },
        "acima_de_5s": int((tabela["total_s"] > 5).sum()),
    }
    return tabela, resumo


# ------------------------------------------------------------------------- relatório
def montar_config(nome, config, pasta_classificador, pesos_detector, pasta_fora, dispositivo,
                  constantes) -> dict:
    commit, alterado = train.estado_git()
    return {
        "run": nome, "criado_em": datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "commit_com_alteracoes": alterado,
        "decisoes_do_gestor": [{"data": d, "texto": t} for d, t in DECISOES],
        "configuracao": asdict(config),
        "modelos": {"classificador": train._exibir(Path(pasta_classificador).resolve()),
                    "detector": train._exibir(Path(pesos_detector).resolve()),
                    "fora_da_dobra": train._exibir(Path(pasta_fora).resolve())},
        "dispositivo_t1_t2": str(dispositivo), "dispositivo_t3": "cpu",
        "constantes_medidas": constantes,
        "acessos_a_rede": [],
        "versoes": train.versoes(),
    }


def _pct(valor, casas: int = 1) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "-"
    return fmt.pct(valor, 1, casas)


def _ic(m: dict) -> str:
    baixo, alto = m["acuracia_ic95"]
    return f"{_pct(baixo)} a {_pct(alto)}"


def relatorio(config_run: dict, metricas: dict, observacoes: str | None) -> str:
    c = config_run["configuracao"]
    cal = metricas["calibracao"]
    t1m, t2m, t3m = metricas["t1"], metricas["t2"], metricas["t3"]
    const = config_run["constantes_medidas"]
    linhas = [
        f"# Inferência ponta a ponta: run `{config_run['run']}`", "",
        f"Gerado por `python model/avaliar_pipeline.py` em {_data(config_run['criado_em'])}, "
        f"{_origem(config_run)}. Classificador `{config_run['modelos']['classificador']}`, "
        f"detector `{config_run['modelos']['detector']}`. **Os testes do BRACOL e do BRACOT não "
        "foram lidos nem avaliados.** Módulo: `model/inferencia.py`; interface no README "
        "(seção \"Interface para o backend\").", "",
        "## Decisões do gestor", "",
        *(f"- **{x['data']}:** {x['texto']}" for x in config_run["decisoes_do_gestor"]), "",
        "## Configuração da cadeia", "",
    ]
    linhas += fmt.tabela(["parâmetro", "valor", "origem"], [
        ["limiar_deteccao", fmt.decimal(c["limiar_deteccao"], 2), "decisão do gestor"],
        ["max_folhas", c["max_folhas"], "RNF02 (a validação do BRACOT tem até 13 detecções)"],
        ["area_minima", _pct(c["area_minima"]), "pedido do gestor (cortadas: T2)"],
        ["saturacao_minima / fracao_colorida_minima",
         f"{c['saturacao_minima']} / {_pct(c['fracao_colorida_minima'], 0)}",
         "filtro de cor, medido no treino (abaixo)"],
        ["limiar_confianca", fmt.decimal(c["limiar_confianca"], 2),
         "proposta da curva risco-cobertura; **aguarda a decisão do gestor**"],
        ["recorte", c["recorte"],
         "T1: só o recorte acerta "
         f"{_pct(t1m['variantes']['recorte_A']['metricas']['acuracia'])} no A e "
         f"{_pct(t1m['variantes']['recorte_B']['metricas']['acuracia'])} no B (abaixo)"],
        ["margem", _pct(c["margem"]), "medida no treino do BRACOL (abaixo)"],
        ["cinza_fundo", c["cinza_fundo"], "medido no treino do BRACOL (abaixo)"],
        ["cobertura_folha_unica", _pct(c["cobertura_folha_unica"], 0),
         "entre o BRACOL e o BRACOT de treino (abaixo)"],
        ["score_minimo_folha_unica", fmt.decimal(c["score_minimo_folha_unica"], 2),
         "decisão do gestor"],
        ["max_megapixels", fmt.decimal(c["max_megapixels"], 0), "decisão do gestor"],
    ], "lll")
    linhas += _secao_constantes(const, c)
    linhas += _secao_calibracao(cal, c)
    linhas += _secao_t1(t1m, c)
    linhas += _secao_t2(t2m, c, observacoes)
    linhas += _secao_t3(t3m, c)
    linhas += [
        "## Acessos à rede", "",
        "Nenhum: o classificador é montado sem pré-treino e lê só o `melhor.pt`; o detector roda "
        "com o modo offline do ultralytics imposto (`detector.importar_ultralytics`).", "",
        "## Limites", "",
        "- O limiar de confiança é calibrado em folhas do BRACOL (fundo claro, uma folha por "
        "foto); no campo (T2) a confiança é outra, e não há rótulo para medir a acurácia ali.",
        "- As predições fora da dobra vêm de modelos de 20 épocas, mais fracos que o run base: "
        "o limiar proposto tende a ser conservador para ele.",
        "- `especie_incorreta` não é produzido: não há dados de outras plantas. Uma foto sem "
        "café pode sair como folha de café com confiança alta.",
        "- O T2 não tem rótulo de classe: as observações dos recortes são uma inspeção visual, "
        "não uma medida.", ""]
    return "\n".join(linhas) + "\n"


def _secao_constantes(const: dict, c: dict) -> list[str]:
    fundo, folha, plantas = const["fundo"], const["folha_unica_bracol"], const["plantas_bracot"]
    linhas = ["", "## Constantes medidas no treino", "",
              f"**Cinza do fundo.** Mediana do tom da borda de {_pct(fundo['borda'], 0)} de cada "
              f"uma das {fmt.n(fundo['imagens'])} imagens de treino do BRACOL, já em 448x224. "
              "O tom do fundo identifica a sessão de fotos (checagem de atalho), por isso cada "
              f"classe pesa igual: cinza neutro {fundo['cinza']} (em uso: {c['cinza_fundo']}).",
              ""]
    linhas += fmt.tabela(["classe", "R", "G", "B", "luminância"], [
        [classe, *(fmt.decimal(v[canal], 0) for canal in ("r", "g", "b")),
         fmt.decimal(v["luminancia"], 0)]
        for classe, v in fundo["por_classe"].items()])
    cor = const["cor_folhas_bracot"]
    linhas += ["", f"**Filtro de cor.** Nas fotos de folha sobre fundo liso do BRACOL, o detector "
                   f"também acha pedaços do fundo. Das {folha['deteccoes_antes_da_cor']} detecções "
                   f"no limiar nas {folha['imagens']} imagens de treino, "
                   f"{folha['cortadas_pela_cor']} têm menos de "
                   f"{_pct(c['fracao_colorida_minima'], 0)} da máscara com saturação >= "
                   f"{c['saturacao_minima']} e saem (histograma da fração colorida, de 0 a 1 em "
                   f"passos de 0,1: {folha['colorida_histograma']}). Nas {cor['folhas']} folhas "
                   f"anotadas de {cor['fotos']} fotos de treino do BRACOT, a fração colorida tem "
                   f"mediana {_pct(cor['mediana'])}, p1 {_pct(cor['p1'])} e mínimo "
                   f"{_pct(cor['min'])}; {cor['abaixo_do_minimo']} ficariam abaixo do mínimo."]
    cob = folha["cobertura"]
    linhas += ["", f"**Margem.** Em {fmt.n(folha['imagens'])} imagens de treino do BRACOL "
                   f"(amostra estratificada, semente {SEED}), a maior folha detectada tem "
                   f"comprimento mediano de {_pct(folha['comprimento_mediano'])} da largura da "
                   f"foto: margem de {_pct(folha['margem'])} de cada lado (em uso: "
                   f"{_pct(c['margem'])}). Detecções por imagem: nenhuma em "
                   f"{folha['sem_deteccao']}, uma em {folha['uma_deteccao']}, mais de uma em "
                   f"{folha['mais_de_uma']}.", "",
               f"**Cobertura da folha única.** A caixa da maior folha cobre de "
               f"{_pct(cob['min'])} a {_pct(cob['max'])} da foto no BRACOL (p5 {_pct(cob['p5'])},"
               f" mediana {_pct(cob['mediana'])}). Nas {plantas['fotos']} fotos de treino do "
               f"BRACOT (sem a validação), a maior caixa cobre no máximo "
               f"{_pct(plantas['cobertura_maior']['max'])} (p95 "
               f"{_pct(plantas['cobertura_maior']['p95'])}), e {plantas['uma_deteccao']} foto(s) "
               f"têm uma só folha utilizável. Regra 1 a partir de "
               f"{_pct(c['cobertura_folha_unica'], 0)}.", ""]
    return linhas


def _secao_calibracao(cal: dict, c: dict) -> list[str]:
    fora, val = cal["fora_da_dobra"], cal["validacao"]
    linhas = ["## Limiar de confiança (RF07)", "",
              f"Curva risco-cobertura: com o limiar t, a folha com probabilidade da classe "
              f"prevista abaixo de t sai da resposta (todas abaixo: `baixa_confianca`). Fonte "
              f"principal: as {fmt.n(fora['n'])} predições fora da dobra do diagnóstico por "
              f"blocos (`{fora['pasta']}`, protocolo aleatório; acurácia "
              f"{_pct(fora['acuracia'])} sem limiar). Conferência: as {val['n']} imagens de "
              "validação no classificador avaliado (imagem inteira, T1).", "",
              "![Curva risco-cobertura](risco_cobertura.png)", ""]
    pontos_fora = {round(p["limiar"], 2): p for p in fora["curva"]}
    pontos_val = {round(p["limiar"], 2): p for p in val["curva"]}
    destaque = sorted({*LIMIARES_DESTAQUE, *(x for x in (cal["proposta"],) if x is not None)})
    linhas += fmt.tabela(
        ["limiar", "cobertura (fora da dobra)", "acurácia das mantidas (IC 95%)",
         *(f"cobertura {cl}" for cl in bracol.CLASSES), "cobertura (val)", "acurácia (val)"],
        [[fmt.decimal(t, 2), _pct(pontos_fora[round(t, 2)]["cobertura"]),
          f"{_pct(pontos_fora[round(t, 2)]['acuracia'])} "
          f"({_pct(pontos_fora[round(t, 2)]['acuracia_ic95'][0])} a "
          f"{_pct(pontos_fora[round(t, 2)]['acuracia_ic95'][1])})",
          *(_pct(pontos_fora[round(t, 2)]["cobertura_por_classe"][cl]) for cl in bracol.CLASSES),
          _pct(pontos_val[round(t, 2)]["cobertura"]), _pct(pontos_val[round(t, 2)]["acuracia"])]
         for t in destaque])
    proposta = cal["proposta"]
    linhas += ["", (f"**Proposta: {fmt.decimal(proposta, 2)}**, o menor limiar em que a "
                    f"acurácia das mantidas fora da dobra chega a {_pct(cal['alvo'], 0)}. "
                    "**A decisão é do gestor.**" if proposta is not None else
                    f"Nenhum limiar chega a {_pct(cal['alvo'], 0)} de acurácia."),
               f"Em uso nesta avaliação: {fmt.decimal(c['limiar_confianca'], 2)}. Arquivo: "
               "`risco_cobertura.csv`.", ""]
    return linhas


def _secao_t1(t1m: dict, c: dict) -> list[str]:
    tabela = t1m["variantes"]
    linhas = ["## T1: validação do BRACOL pela cadeia", "",
              f"As {t1m['imagens']} imagens de validação do BRACOL (uma folha por foto, rótulo "
              "conhecido) pela cadeia inteira e pela imagem inteira, com o mesmo código "
              "(`inferencia.classificar`). A folha avaliada é a de maior área. \"Só o recorte\" "
              "desliga a regra 1, para medir o recorte numa folha com rótulo. Sem limiar de "
              "confiança.", ""]
    linhas += fmt.tabela(
        ["variante", "com previsão", "acurácia (IC 95%)", "F1 macro",
         *(f"recall {cl}" for cl in bracol.CLASSES), "severidade: exata / ±1",
         f"no limiar {fmt.decimal(c['limiar_confianca'], 2)}: cobertura / acurácia"],
        [[DESCRICAO_T1[v], _pct(r["cobertura_sem_limiar"]),
          f"{_pct(r['metricas']['acuracia'])} ({_ic(r['metricas'])})",
          _pct(r["metricas"]["f1_macro"]), *(_pct(x) for x in r["metricas"]["recall"]),
          f"{_pct(r['severidade']['acuracia_exata'])} / "
          f"{_pct(r['severidade']['acuracia_tolerancia_1'])}",
          f"{_pct(r['cobertura_no_limiar'])} / {_pct(r['acuracia_no_limiar'])}"]
         for v, r in tabela.items()])
    regras = t1m["regras"]
    linhas += ["", "**Regras de folha única nas imagens do BRACOL** (cadeia):", "",
               f"- regra 1 (uma folha, caixa com {_pct(c['cobertura_folha_unica'], 0)} da foto "
               f"ou mais): {regras['regra_1']};",
               f"- regra 2 (nenhuma folha utilizável, alguma detecção com cor de folha e score "
               f">= {fmt.decimal(c['score_minimo_folha_unica'], 2)}): {regras['regra_2']};",
               f"- regra 3 (`planta_nao_identificada`): {regras['regra_3']};",
               f"- pelas detecções: {regras['deteccao']}, das quais {regras['mais_de_uma']} com "
               "mais de uma folha.", "",
               "**Correção das regras depois do T1.** Na primeira rodada, sem o filtro de cor, o "
               "detector achava pedaços do fundo liso do BRACOL como folhas: 135 imagens saíam "
               "com mais de uma folha, e a regra 1 só pegava 105. O filtro de cor (constantes, "
               "acima) tira esses pedaços, e a regra 2 passou a exigir uma detecção com cor de "
               "folha, porque um pedaço de fundo não é sinal de folha.", "",
               f"Detecções utilizáveis por imagem: {t1m['deteccoes_por_imagem']}; o filtro de cor "
               f"tirou {t1m['cortadas_por_cor']} detecções de fundo. Conferência da "
               f"imagem inteira com o `predicoes_val.csv` do run: {t1m['conferencia']}. "
               "Arquivo: `t1_imagens.csv`.", ""]
    return linhas


def _secao_t2(t2m: dict, c: dict, observacoes: str | None) -> list[str]:
    linhas = ["## T2: validação do BRACOT (sem rótulo de classe)", "",
              f"As {t2m['fotos']} fotos de validação do detector (cenas inteiras do treino dos "
              f"autores): {t2m['deteccoes']} detecções no limiar, {t2m['cortadas_por_area']} "
              f"cortadas pela área mínima, {t2m['cortadas_por_cor']} pelo filtro de cor, "
              f"{t2m['cortadas_por_limite']} pelo número máximo; "
              f"{t2m['folhas']} folhas classificadas, {t2m['na_borda']} cortadas pela borda. "
              f"Planos: {t2m['planos']}. Classe igual nos dois modos em "
              f"{_pct(t2m['concordancia_A_B'])} das folhas.", ""]
    linhas += fmt.tabela(["", *(f"modo {m}" for m in MODOS)], [
        *([f"folhas {cl}", *(str(t2m[m]["classes"][cl]) for m in MODOS)]
          for cl in bracol.CLASSES),
        *([f"severidade {s}", *(str(t2m[m]["severidades"][s]) for m in MODOS)]
          for s in inf.NIVEIS_CONTRATO),
        ["confiança média / mediana", *(f"{_pct(t2m[m]['confianca_media'])} / "
                                        f"{_pct(t2m[m]['confianca_mediana'])}" for m in MODOS)],
        *([f"abaixo de {fmt.decimal(float(t), 2)}", *(_pct(t2m[m]["abaixo"][t]) for m in MODOS)]
          for t in t2m["A"]["abaixo"]),
        ["cortadas pela borda: confiança média", *(_pct(t2m[m]["borda"]["confianca_media"])
                                                   for m in MODOS)],
        ["inteiras: confiança média", *(_pct(t2m[m]["inteiras"]["confianca_media"])
                                        for m in MODOS)],
        [f"cortadas pela borda abaixo de {fmt.decimal(c['limiar_confianca'], 2)}",
         *(_pct(t2m[m]["borda"]["abaixo_do_limiar"]) for m in MODOS)],
        [f"inteiras abaixo de {fmt.decimal(c['limiar_confianca'], 2)}",
         *(_pct(t2m[m]["inteiras"]["abaixo_do_limiar"]) for m in MODOS)],
        *([f"fotos sem nenhuma folha acima de {fmt.decimal(float(t), 2)} (baixa_confianca)",
           *(f"{t2m['fotos_baixa_confianca'][m][t]} de {t2m['fotos']}" for m in MODOS)]
          for t in t2m["fotos_baixa_confianca"]["A"]),
    ], "lrr")
    linhas += ["", "Arquivo: `t2_folhas.csv` (uma linha por folha) e `t2_fotos.csv`.", "",
               "### O que vi nos recortes (sem imagens no repositório)", "",
               observacoes or "Ainda não descrito (falta o observacoes.md do run).", ""]
    return linhas


def _secao_t3(t3m: dict, c: dict) -> list[str]:
    pior = t3m["pior_caso"]
    linhas = ["## T3: tempo em CPU (RNF02: diagnóstico em até 5 s)", "",
              f"Nesta máquina, em CPU ({t3m['threads_torch']} threads do torch), nas "
              f"{t3m['fotos']} fotos de validação do BRACOT (4032x3024), depois de uma passada de "
              f"aquecimento. Leitura dos pesos dos dois modelos, uma vez ao subir o backend: "
              f"{fmt.decimal(t3m['carga_dos_modelos_s'], 1)} s, com o torch e o ultralytics já "
              "importados (o import leva alguns segundos a mais na primeira vez).", ""]
    linhas += fmt.tabela(["etapa", "mediana (s)", "máximo (s)"], [
        [nome, fmt.decimal(t3m["mediana"][chave], 2), fmt.decimal(t3m["maximo"][chave], 2)]
        for nome, chave in (("leitura", "leitura_s"), ("detecção", "deteccao_s"),
                            ("recortes", "recortes_s"), ("classificação (lote)", "classificacao_s"),
                            ("contrato", "contrato_s"), ("**total**", "total_s"))])
    linhas += ["", f"Folhas por foto: mediana {fmt.decimal(t3m['folhas_mediana'], 0)}, máximo "
                   f"{t3m['folhas_maximo']}; {t3m['acima_de_5s']} fotos passaram de 5 s. Pior "
                   f"caso, com {pior['folhas']} folhas: a classificação leva "
                   f"{fmt.decimal(pior['classificacao_mediana_s'], 2)} s (máximo "
                   f"{fmt.decimal(pior['classificacao_maximo_s'], 2)} s), e o total estimado "
                   f"(maior leitura, maior detecção, recortes e classificação de {pior['folhas']}"
                   f" folhas) é {fmt.decimal(pior['total_estimado_s'], 2)} s. A CPU do backend "
                   "pode ser mais lenta que esta. Arquivo: `t3_tempos.csv`.", ""]
    return linhas


def _data(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")


def _origem(config: dict) -> str:
    if not config["commit"]:
        return "fora de um repositório git"
    return (f"no commit `{config['commit'][:7]}`"
            + (" (com alterações não commitadas)" if config["commit_com_alteracoes"] else ""))


# ------------------------------------------------------------------------- execução
def conferir_inteira(tabela: pd.DataFrame, pasta_classificador: Path) -> str:
    """A imagem inteira da cadeia contra as predições gravadas pelo treino do run."""
    arquivo = Path(pasta_classificador) / "predicoes_val.csv"
    if not arquivo.is_file():
        return "sem predicoes_val.csv no run"
    gravado = pd.read_csv(arquivo).set_index("id")
    colunas = [f"prob_{c}" for c in bracol.CLASSES]
    probs = np.stack(tabela["inteira_probs"])
    referencia = gravado.loc[tabela["id"], colunas].to_numpy()
    iguais = int((probs.argmax(axis=1) == referencia.argmax(axis=1)).sum())
    return (f"diferença máxima das probabilidades {np.abs(probs - referencia).max():.1e} "
            f"(o treino usou AMP), classe prevista igual em {iguais} de {len(tabela)}")


def resumir_t1(tabela: pd.DataFrame, config, pasta_classificador) -> dict:
    regras = tabela["regra"].fillna(0).astype(int)
    contagem = tabela["deteccoes"].value_counts().sort_index()
    return {"imagens": len(tabela), "variantes": metricas_t1(tabela, config.limiar_confianca),
            "regras": {"regra_1": int((regras == 1).sum()), "regra_2": int((regras == 2).sum()),
                       "regra_3": int((regras == 3).sum()),
                       "deteccao": int((tabela["plano"] == "deteccao").sum()),
                       "mais_de_uma": int(((tabela["plano"] == "deteccao")
                                           & (tabela["folhas"] > 1)).sum())},
            "deteccoes_por_imagem": ", ".join(f"{k}: {v}" for k, v in contagem.items()),
            "cortadas_por_cor": int(tabela["cortadas_por_cor"].sum()),
            "conferencia": conferir_inteira(tabela, pasta_classificador)}


def nativo(objeto):
    """Tipos do numpy como tipos do Python, em qualquer profundidade (para o json)."""
    if isinstance(objeto, dict):
        return {str(k): nativo(v) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple, np.ndarray)):
        return [nativo(v) for v in objeto]
    if isinstance(objeto, np.generic):
        return objeto.item()
    return objeto


def gravar_csv(tabela: pd.DataFrame, caminho: Path) -> None:
    tabela.to_csv(caminho, index=False, float_format="%.6f", lineterminator="\n")


def executar(args) -> int:
    import torch

    config = inf.Configuracao()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelos = inf.carregar_modelos(args.classificador, args.detector, dispositivo)
    constantes = medir_constantes(modelos, config)
    fmt.dizer(json.dumps({"cinza": constantes["fundo"]["cinza"],
                          "margem": round(constantes["folha_unica_bracol"]["margem"], 4),
                          "cobertura_bracol": constantes["folha_unica_bracol"]["cobertura"],
                          "cobertura_bracot": constantes["plantas_bracot"]["cobertura_maior"],
                          "bracot_uma_folha": constantes["plantas_bracot"]["uma_deteccao"]}))
    if args.so_constantes:
        return 0
    pasta = train.PASTA_RUNS / args.nome
    pasta.mkdir(parents=True, exist_ok=True)
    fmt.dizer("T1: validacao do BRACOL...")
    tabela_t1 = t1(modelos, config)
    inteira = tabela_t1[["classe", "inteira_probs"]].rename(columns={"inteira_probs": "probs"})
    calibracao = calibrar(args.fora_da_dobra, inteira)
    fmt.dizer(f"Limiar proposto: {calibracao['proposta']}")
    fmt.dizer("T2: validacao do BRACOT...")
    folhas_t2, fotos_t2 = t2(modelos, config, args.recortes)
    pranchas = []
    if args.recortes:
        pranchas = gravar_pranchas(escolher_para_pranchas(folhas_t2, 2 * RECORTES_POR_PRANCHA),
                                   args.recortes)
    fmt.dizer("T3: tempo em CPU...")
    tabela_t3, resumo_t3 = t3(args.classificador, args.detector, config)
    metricas = {"calibracao": calibracao, "t1": resumir_t1(tabela_t1, config, args.classificador),
                "t2": resumo_t2(folhas_t2, fotos_t2, config.limiar_confianca), "t3": resumo_t3}
    curva = pd.DataFrame([{"fonte": fonte, **{k: v for k, v in p.items()
                                              if k not in ("acuracia_ic95",
                                                           "cobertura_por_classe")},
                           "ic95_baixo": p["acuracia_ic95"][0], "ic95_alto": p["acuracia_ic95"][1],
                           **{f"cobertura_{c}": v for c, v in p["cobertura_por_classe"].items()}}
                          for fonte in ("fora_da_dobra", "validacao")
                          for p in calibracao[fonte]["curva"]])
    gravar_csv(curva, pasta / "risco_cobertura.csv")
    metrics.salvar_figura(figura_risco_cobertura(calibracao, config.limiar_confianca),
                          pasta / "risco_cobertura.png")
    gravar_csv(tabela_t1.drop(columns=[c for c in tabela_t1.columns if c.endswith("_probs")])
               .assign(classe=lambda t: t["classe"].map(lambda k: bracol.CLASSES[k])),
               pasta / "t1_imagens.csv")
    gravar_csv(folhas_t2, pasta / "t2_folhas.csv")
    gravar_csv(fotos_t2, pasta / "t2_fotos.csv")
    gravar_csv(tabela_t3, pasta / "t3_tempos.csv")
    config_run = montar_config(args.nome, config, args.classificador, args.detector,
                               args.fora_da_dobra, dispositivo, constantes)
    config_run, metricas = nativo(config_run), nativo(metricas)
    train.gravar_json(config_run, pasta / "config.json")
    train.gravar_json(metricas, pasta / "metricas.json")
    nota = pasta / "observacoes.md"
    (pasta / "relatorio.md").write_text(
        relatorio(config_run, metricas,
                  nota.read_text(encoding="utf-8").strip() if nota.is_file() else None),
        encoding="utf-8", newline="\n")
    for v in VARIANTES_T1:
        m = metricas["t1"]["variantes"][v]["metricas"]
        fmt.dizer(f"T1 {v}: acuracia {_pct(m['acuracia'])}, F1 macro {_pct(m['f1_macro'])}")
    fmt.dizer(f"T3: total mediano {fmt.decimal(resumo_t3['mediana']['total_s'], 2)} s, maximo "
              f"{fmt.decimal(resumo_t3['maximo']['total_s'], 2)} s")
    for caminho in pranchas:
        fmt.dizer(f"Prancha: {caminho}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Avaliacao ponta a ponta da inferencia (detector + recorte + classificador). "
                    "Detalhes no docstring de model/avaliar_pipeline.py.")
    ap.add_argument("--nome", default=NOME_PADRAO, help=f"pasta em model/runs (padrao: "
                                                        f"{NOME_PADRAO})")
    ap.add_argument("--classificador", type=Path, default=inf.PASTA_CLASSIFICADOR,
                    help="pasta do run do classificador (com o melhor.pt)")
    ap.add_argument("--detector", type=Path, default=detector.PESOS_PADRAO,
                    help="pesos do detector (last.pt)")
    ap.add_argument("--fora-da-dobra", type=Path, default=PASTA_FORA_DA_DOBRA,
                    help="pasta com as predicoes_aleatorio_*.csv do diagnostico por blocos")
    ap.add_argument("--recortes", type=Path,
                    help="pasta FORA do repositorio para os recortes do T2 e as pranchas")
    ap.add_argument("--so-constantes", action="store_true",
                    help="so mede e mostra as constantes (nao grava nada)")
    args = ap.parse_args(argv)
    if args.recortes and args.recortes.resolve().is_relative_to(RAIZ_REPO):
        ap.error("--recortes precisa ficar fora do repositorio (imagens do dataset)")
    try:
        return executar(args)
    except (train.ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
