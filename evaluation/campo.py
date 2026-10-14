"""
Campo (Frente 9, passo 4): o classificador nas folhas das 46 fotos de validação do BRACOT, que
não têm rótulo de classe, e a rotulagem cega dessas folhas.

- --gerar-rotulagem: sorteia 120 folhas (semente 42, espalhadas pelas fotos), grava pranchas
  numeradas de 20 recortes em data/processed/campo_rotulagem/ (fora do git), com o recorte B
  (folha isolada) grande e o A pequeno, SEM a previsão, e o CSV-modelo
  data/manifests/campo_rotulado.csv (não sobrescreve um CSV com rótulos).
- --avaliar: com um classificador, nas folhas da cadeia (modos A e B), os proxies de campo
  (concordância A x B, estabilidade sob perturbações, distribuição das classes, da confiança e
  da severidade, folhas na borda x inteiras), o filtro de fragmentos e, só se o CSV estiver
  preenchido, as métricas com os rótulos de campo (com intervalo de Wilson, sem os "?").

- --gerar-rotulagem --fotos treino (passo 5b): a rotulagem de validação, 160 folhas das 194 fotos
  de TREINO do detector no BRACOT (nunca as 46 de validação, nunca o teste), na cadeia padrão de
  11/10/2026 (recorte B, filtro de fragmentos em 0,5), em 8 pranchas cegas em
  data/processed/campo_rotulagem_treino/ e no CSV-modelo data/manifests/campo_rotulado_treino.csv.
  O detector treinou nessas fotos: as máscaras delas saem mais limpas que as de fotos novas.

O campo_rotulado é uma fonte nova, só de avaliação, nunca de treino. Os rótulos foram feitos
por Claude por leitura visual das pranchas e aceitos pelo gestor em 11/10/2026 (não são de
especialista): S saudável evidente, D doente evidente, ? não sei ou recorte ruim; a classe
é opcional. O teste do BRACOT fica fechado: só fotos de validação do detector (cenas inteiras
do treino dos autores).

Uso (de qualquer pasta, com o venv ativo):
    python evaluation/campo.py --gerar-rotulagem
    python evaluation/campo.py --gerar-rotulagem --fotos treino
    python evaluation/campo.py --avaliar --classificador model/runs/campo_v1 --saida campo_v1
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import bracot
import dados
import formatacao as fmt
import inferencia as inf
import metrics
import train
import treinar_detector as td

PASTA_ROTULAGEM = RAIZ_REPO / "data" / "processed" / "campo_rotulagem"  # fora do git
MANIFEST_CAMPO = RAIZ_REPO / "data" / "manifests" / "campo_rotulado.csv"
N_ROTULAGEM = 120
POR_PRANCHA = 20
SEED = 42
COLUNAS_CAMPO = ["id", "foto", "folha", "prancha", "numero", "score", "caixa_x0", "caixa_y0",
                 "caixa_x1", "caixa_y1", "rotulo", "classe", "observacao"]
ROTULOS = ("S", "D", "?")
CLASSES_DOENCA = tuple(c for c in inf.CATEGORIAS if c != "saudavel")
MODOS = ("A", "B")
# Perturbações pequenas: brilho e contraste de 10% e deslocamento do recorte de 5%.
PERTURBACOES = (("brilho+10", "brilho", 0.10), ("brilho-10", "brilho", -0.10),
                ("contraste+10", "contraste", 0.10), ("contraste-10", "contraste", -0.10),
                ("desloc_x+5", "x", 0.05), ("desloc_x-5", "x", -0.05),
                ("desloc_y+5", "y", 0.05), ("desloc_y-5", "y", -0.05))
LIMIARES_FRACAO_VISIVEL = (0.2, 0.3, 0.5)
LIMIARES_PREENCHIMENTO = (0.5, 0.6, 0.7)
IOU_CASAMENTO = 0.5
LOTE = 32
# Quem rotulou (registrado a pedido do gestor em 11/10/2026).
ORIGEM_ROTULOS = ("feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em "
                  "11/10/2026; não são de especialista")
# A cadeia da rotulagem (passo 4): o filtro de fragmentos desligado, para que a numeração das
# 318 folhas, que casa com os rótulos, não mude quando os padrões da cadeia mudarem.
CONFIG_ROTULAGEM = inf.Configuracao(fracao_visivel_minima=0.0)
# A rotulagem de validação (passo 5b, decisão do gestor de 11/10/2026): fotos de treino do
# detector, na cadeia padrão de 11/10/2026, fixada aqui para que os ids não mudem com os padrões.
PASTA_ROTULAGEM_TREINO = RAIZ_REPO / "data" / "processed" / "campo_rotulagem_treino"  # fora do git
MANIFEST_CAMPO_TREINO = RAIZ_REPO / "data" / "manifests" / "campo_rotulado_treino.csv"
N_ROTULAGEM_TREINO = 160
CONFIG_ROTULAGEM_TREINO = inf.Configuracao(recorte="B", fracao_visivel_minima=0.5)
CREDITO_BRACOT = ("Fotos: BRACOT (Krohling, Tozzi de Souza e Tassis, 2021), CC BY 4.0, "
                  "DOI 10.17632/pmkbyjpf6k.1")


# --------------------------------------------------------------------------- folhas
def fotos_de_validacao() -> tuple[list[str], pd.Series]:
    """As 46 fotos de validação do detector (cenas inteiras do treino dos autores)."""
    manifest = bracot.ler_manifest(split="treino")
    return td.dividir_validacao(manifest), manifest.set_index("id")["caminho"]


def fotos_de_treino() -> tuple[list[str], pd.Series]:
    """As 194 fotos de treino do detector: o treino dos autores sem as cenas de validação (nunca
    o teste). Falha se uma cena aparecer nos dois lados."""
    manifest = bracot.ler_manifest(split="treino")
    validacao = set(td.dividir_validacao(manifest))
    treino = manifest[~manifest["id"].isin(validacao)]
    if set(treino["cena"]) & set(manifest.loc[manifest["id"].isin(validacao), "cena"]):
        raise train.ErroFatal("uma cena de treino do BRACOT aparece na validação do detector")
    return sorted(treino["id"]), manifest.set_index("id")["caminho"]


def percorrer_folhas(modelos, config=None, fotos=None, contagens: dict | None = None):
    """Para cada foto (as de validação, ou fotos = (ids, caminhos)): (foto, imagem reduzida,
    escala, folhas). As folhas são as da cadeia (inferencia.decidir), numeradas de 1 em ordem de
    score, com a fração visível e o preenchimento. Uma foto por vez (as fotos têm 4032x3024).
    Com contagens (um dict), guarda nele, por foto, o plano, a regra, as folhas que ficam e as
    contagens de inferencia.decidir (inclusive das fotos que não vão pela detecção)."""
    config = config or CONFIG_ROTULAGEM
    ids, caminhos = fotos or fotos_de_validacao()
    for foto in ids:
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        altura, largura = imagem.shape[:2]
        preparo = inf.decidir(inf.detectar(imagem, config, modelos), altura, largura, config)
        if contagens is not None:
            contagens[foto] = {"plano": preparo["plano"], "regra": preparo["regra"],
                               "folhas": len(preparo["folhas"]), **preparo["contagens"]}
        if preparo["plano"] != "deteccao":
            continue
        reduzida, escala = inf.reduzir(imagem, inf.LADO_TRABALHO)
        folhas = []
        for k, folha in enumerate(preparo["folhas"], start=1):
            folhas.append({
                "foto": foto, "folha": k, "id": f"{foto}_{k:02d}", "score": folha["score"],
                "area": folha["area"], "borda": folha["borda"], "poligono": folha["poligono"],
                "caixa": inf.caixa_relativa(folha["caixa"], altura, largura),
                "fracao_visivel": inf.fracao_visivel(folha, preparo["folhas"]),
                "preenchimento": inf.preenchimento(folha["poligono"])})
        yield foto, reduzida, escala, folhas


def recorte(reduzida, escala, folha, modo, config, deslocamento=(0.0, 0.0)):
    return inf.recortar_folha(reduzida, np.asarray(folha["poligono"]) * escala, modo,
                              config.margem, config.cinza_fundo, deslocamento)


# ------------------------------------------------------------------------ rotulagem
def sortear_rotulagem(folhas_por_foto: dict[str, int], n: int = N_ROTULAGEM,
                      seed: int = SEED, max_por_foto: int | None = None) -> list[tuple[str, int]]:
    """n folhas (foto, índice), espalhadas: as fotos em ordem sorteada e, em cada uma, as folhas
    em ordem sorteada; pega-se uma folha de cada foto por rodada, até n. Com max_por_foto, cada
    foto dá no máximo esse número de folhas (o sorteio das outras não muda). Determinístico."""
    gerador = np.random.default_rng(seed)
    fotos = sorted(folhas_por_foto)
    ordem_fotos = [fotos[i] for i in gerador.permutation(len(fotos))]
    filas = {f: [int(k) + 1 for k in gerador.permutation(folhas_por_foto[f])][:max_por_foto]
             for f in ordem_fotos}
    escolhidas, total = [], sum(len(fila) for fila in filas.values())
    while len(escolhidas) < min(n, total):
        for foto in ordem_fotos:
            if filas[foto] and len(escolhidas) < n:
                escolhidas.append((foto, filas[foto].pop(0)))
    return escolhidas


def gerar_rotulagem(modelos, pasta: Path = PASTA_ROTULAGEM, manifest: Path = MANIFEST_CAMPO,
                    config=None, fotos=None, n: int = N_ROTULAGEM, max_por_foto: int | None = None,
                    contagens: dict | None = None) -> dict:
    """Pranchas cegas e CSV-modelo (ver o docstring do módulo). Por padrão, a rotulagem do passo
    4 (fotos de validação); a de validação do passo 5b passa as fotos de treino, a pasta, o CSV,
    a configuração e o n dela. max_por_foto limita as folhas sorteadas de cada foto, e contagens
    recebe as contagens da cadeia por foto (percorrer_folhas). Só o detector é usado: nada aqui
    chama o classificador."""
    if manifest.is_file() and ler_rotulos(manifest) is not None:
        raise train.ErroFatal(f"{train._exibir(manifest)} já tem rótulos: não sobrescrevo")
    config = config or CONFIG_ROTULAGEM
    todas = {foto: (reduzida, escala, folhas)
             for foto, reduzida, escala, folhas in percorrer_folhas(modelos, config, fotos,
                                                                    contagens)}
    escolhidas = sortear_rotulagem({f: len(v[2]) for f, v in todas.items()}, n=n,
                                   max_por_foto=max_por_foto)
    pasta.mkdir(parents=True, exist_ok=True)
    linhas, cartoes = [], []
    for posicao, (foto, k) in enumerate(escolhidas):
        reduzida, escala, folhas = todas[foto]
        folha = folhas[k - 1]
        prancha, numero = posicao // POR_PRANCHA + 1, posicao % POR_PRANCHA + 1
        linhas.append({"id": folha["id"], "foto": foto, "folha": k, "prancha": prancha,
                       "numero": numero, "score": round(folha["score"], 4),
                       **dict(zip(("caixa_x0", "caixa_y0", "caixa_x1", "caixa_y1"),
                                  folha["caixa"])),
                       "rotulo": "", "classe": "", "observacao": ""})
        cartoes.append(_cartao(numero, folha["id"], recorte(reduzida, escala, folha, "B", config),
                               recorte(reduzida, escala, folha, "A", config)))
    pranchas = []
    for p in range(0, len(cartoes), POR_PRANCHA):
        caminho = pasta / f"prancha_{p // POR_PRANCHA + 1}.jpg"
        _gravar_jpeg(_prancha(cartoes[p:p + POR_PRANCHA], p // POR_PRANCHA + 1), caminho)
        pranchas.append(caminho)
    pd.DataFrame(linhas, columns=COLUNAS_CAMPO).to_csv(manifest, index=False,
                                                       lineterminator="\n", encoding="utf-8")
    return {"folhas": len(linhas), "fotos": len({l["foto"] for l in linhas}),
            "pranchas": pranchas, "manifest": manifest,
            "total_de_folhas": sum(len(v[2]) for v in todas.values())}


def _cartao(numero: int, ident: str, recorte_b, recorte_a) -> np.ndarray:
    """Um cartão da prancha: o recorte B grande, o A pequeno ao lado, o número e o id."""
    grande = cv2.resize(recorte_b, (560, 280), interpolation=cv2.INTER_AREA)
    pequeno = cv2.resize(recorte_a, (280, 140), interpolation=cv2.INTER_AREA)
    cartao = np.full((330, 860, 3), 255, dtype=np.uint8)
    cartao[40:320, 0:560] = grande
    cartao[40:180, 576:856] = pequeno
    cv2.putText(cartao, str(numero), (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(cartao, ident, (64, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1,
                cv2.LINE_AA)
    cv2.putText(cartao, "B (folha isolada)", (4, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(cartao, "A (foto em volta)", (576, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 90, 90), 1, cv2.LINE_AA)
    return cartao


def _prancha(cartoes, numero: int) -> np.ndarray:
    """Os cartões em 2 colunas, com título e crédito."""
    linhas = math.ceil(len(cartoes) / 2)
    altura, largura = cartoes[0].shape[:2]
    prancha = np.full((70 + linhas * (altura + 10) + 40, 2 * largura + 30, 3), 255, np.uint8)
    for i, cartao in enumerate(cartoes):
        y, x = 70 + (i // 2) * (altura + 10), 10 + (i % 2) * (largura + 10)
        prancha[y:y + altura, x:x + largura] = cartao
    titulo = (f"Prancha {numero}: rotule cada numero com S (saudavel evidente), D (doente "
              "evidente) ou ? (nao sei / recorte ruim)")
    cv2.putText(prancha, titulo, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(prancha, CREDITO_BRACOT, (10, prancha.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return prancha


def _gravar_jpeg(rgb, caminho: Path) -> None:
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise train.ErroFatal(f"não foi possível gravar {caminho}")
    caminho.write_bytes(jpeg.tobytes())


# ------------------------------------------------------------------------- rótulos
def ler_rotulos(caminho: Path = MANIFEST_CAMPO) -> pd.DataFrame | None:
    """Os rótulos de campo, ou None se o CSV não existe ou não tem nenhum rótulo. Aceita UTF-8
    (com ou sem BOM) ou cp1252 e vírgula ou ponto e vírgula (CSV salvo pelo Excel)."""
    if not Path(caminho).is_file():
        return None
    bruto = Path(caminho).read_bytes()
    for codificacao in ("utf-8-sig", "cp1252"):
        try:
            texto = bruto.decode(codificacao)
            break
        except UnicodeDecodeError:
            continue
    cabecalho = texto.splitlines()[0]
    separador = ";" if cabecalho.count(";") > cabecalho.count(",") else ","
    df = pd.read_csv(io.StringIO(texto), sep=separador, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    faltam = {"id", "rotulo", "classe"} - set(df.columns)
    if faltam:
        raise ValueError(f"colunas faltando em {caminho}: {sorted(faltam)}")
    df["rotulo"] = df["rotulo"].str.strip().str.upper()
    df["classe"] = df["classe"].str.strip().str.lower()
    if not (df["rotulo"] != "").any():
        return None
    invalidos = sorted(set(df["rotulo"]) - {"", *ROTULOS})
    if invalidos:
        raise ValueError(f"rótulos fora de S, D e ?: {invalidos}")
    classes = sorted(set(df["classe"]) - {"", *CLASSES_DOENCA})
    if classes:
        raise ValueError(f"classes desconhecidas: {classes}")
    return df[df["rotulo"] != ""].reset_index(drop=True)


def proporcao(sucessos: int, n: int) -> dict:
    """Proporção com o intervalo de Wilson de 95% (n = 0: sem valor)."""
    if not n:
        return {"n": 0, "valor": None, "ic95": [None, None]}
    p = sucessos / n
    return {"n": n, "valor": p, "ic95": list(metrics.intervalo_wilson(p, n))}


def metricas_rotuladas(tabela: pd.DataFrame) -> dict:
    """Com os rótulos de campo (sem os "?") e a classe prevista (coluna "prevista"):
    P(prever doença | S), P(prever saudável | D) e a acurácia de classe nas folhas D com classe,
    cada uma com o intervalo de Wilson de 95%. Com a coluna "observacao", também P(prever doença |
    S) separado nas folhas S com e sem "brilho" na observação (hipótese do brilho forte)."""
    saudaveis = tabela[tabela["rotulo"] == "S"]
    doentes = tabela[tabela["rotulo"] == "D"]
    com_classe = doentes[doentes["classe"] != ""]
    return {
        "rotuladas": len(tabela), "S": len(saudaveis), "D": len(doentes),
        "nao_sei": int((tabela["rotulo"] == "?").sum()),
        "doenca_dado_S": proporcao(int((saudaveis["prevista"] != "saudavel").sum()),
                                   len(saudaveis)),
        "saudavel_dado_D": proporcao(int((doentes["prevista"] == "saudavel").sum()), len(doentes)),
        "classe_certa_dado_D": proporcao(int((com_classe["prevista"] == com_classe["classe"])
                                             .sum()), len(com_classe)),
        **_por_observacao(saudaveis),
    }


def _por_observacao(saudaveis: pd.DataFrame) -> dict:
    if "observacao" not in saudaveis:
        return {}
    brilho = saudaveis["observacao"].str.lower().str.contains("brilho")
    return {"doenca_dado_S_com_brilho": proporcao(
                int((saudaveis.loc[brilho, "prevista"] != "saudavel").sum()), int(brilho.sum())),
            "doenca_dado_S_sem_brilho": proporcao(
                int((saudaveis.loc[~brilho, "prevista"] != "saudavel").sum()),
                int((~brilho).sum()))}


def iou_caixas(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def casar_rotulos(rotulos: pd.DataFrame, folhas: pd.DataFrame) -> pd.DataFrame:
    """Os rótulos com as folhas atuais, pelo id, conferido pela caixa (IoU >= IOU_CASAMENTO):
    se a cadeia mudar a numeração, o rótulo não vai para a folha errada."""
    atuais = folhas.set_index("id")
    linhas = []
    for r in rotulos.itertuples():
        if r.id not in atuais.index:
            continue
        colunas = ("caixa_x0", "caixa_y0", "caixa_x1", "caixa_y1")
        caixa_r = [float(getattr(r, c)) for c in colunas]
        caixa_f = [float(atuais.loc[r.id, c]) for c in colunas]
        if iou_caixas(caixa_r, caixa_f) >= IOU_CASAMENTO:
            linhas.append({"id": r.id, "rotulo": r.rotulo, "classe": r.classe,
                           "observacao": str(getattr(r, "observacao", "") or "")})
    return pd.DataFrame(linhas, columns=["id", "rotulo", "classe", "observacao"])


# --------------------------------------------------------------------------- proxies
def perturbar(recorte_rgb, tipo: str, valor: float) -> np.ndarray:
    """Brilho (multiplica) ou contraste (afasta da média) em `valor`, cortado em 0..255."""
    x = recorte_rgb.astype(np.float32)
    x = x * (1 + valor) if tipo == "brilho" else (x - x.mean()) * (1 + valor) + x.mean()
    return np.clip(x, 0, 255).astype(np.uint8)


def resumir_proxies(folhas: pd.DataFrame) -> dict:
    """Proxies de campo a partir da tabela por folha (colunas classe_<m>, confianca_<m>,
    severidade_<m> e estavel_<m>, a fração das perturbações com a mesma classe)."""
    resultado = {"folhas": len(folhas),
                 "concordancia_A_B": float((folhas["classe_A"] == folhas["classe_B"]).mean())}
    borda = folhas["borda"].astype(bool)
    for modo in MODOS:
        classe = folhas[f"classe_{modo}"]
        resultado[modo] = {
            "classes": classe.value_counts().reindex(inf.CATEGORIAS, fill_value=0).to_dict(),
            "severidades": folhas[f"severidade_{modo}"].value_counts().reindex(
                inf.NIVEIS_CONTRATO, fill_value=0).to_dict(),
            "fracao_saudavel": float((classe == "saudavel").mean()),
            "classe_mais_comum": float(classe.value_counts().iloc[0] / len(classe)),
            "confianca_media": float(folhas[f"confianca_{modo}"].mean()),
            "estabilidade_media": float(folhas[f"estavel_{modo}"].mean()),
            "estaveis_em_todas": float((folhas[f"estavel_{modo}"] == 1).mean()),
            "borda": {"folhas": int(borda.sum()),
                      "confianca_media": float(folhas.loc[borda, f"confianca_{modo}"].mean()),
                      "fracao_saudavel": float((classe[borda] == "saudavel").mean())},
            "inteiras": {"folhas": int((~borda).sum()),
                         "confianca_media": float(folhas.loc[~borda, f"confianca_{modo}"].mean()),
                         "fracao_saudavel": float((classe[~borda] == "saudavel").mean())},
        }
    return resultado


def resumir_fragmentos(folhas: pd.DataFrame, rotulos: pd.DataFrame | None = None) -> dict:
    """Quantas folhas cada corte de fragmento tiraria, quantas delas estão na borda e, com
    rótulos, a fração de "?" entre as cortadas e as mantidas."""
    resultado = {"folhas": len(folhas), "na_borda": int(folhas["borda"].sum()),
                 "fracao_visivel": {"mediana_borda": float(folhas.loc[folhas["borda"],
                                                                      "fracao_visivel"].median()),
                                    "mediana_inteiras": float(folhas.loc[~folhas["borda"],
                                                                         "fracao_visivel"]
                                                              .median())},
                 "cortes": []}
    rotulado = None if rotulos is None else folhas.merge(rotulos, on="id", how="inner")
    for nome, coluna, limiares in (("fracao_visivel", "fracao_visivel", LIMIARES_FRACAO_VISIVEL),
                                   ("preenchimento", "preenchimento", LIMIARES_PREENCHIMENTO)):
        for limiar in limiares:
            corta = folhas[coluna] < limiar
            linha = {"filtro": nome, "limiar": limiar, "cortadas": int(corta.sum()),
                     "cortadas_na_borda": int((corta & folhas["borda"]).sum())}
            if rotulado is not None and len(rotulado):
                cortadas = rotulado[rotulado[coluna] < limiar]
                mantidas = rotulado[rotulado[coluna] >= limiar]
                linha["nao_sei_entre_cortadas"] = (float((cortadas["rotulo"] == "?").mean())
                                                   if len(cortadas) else None)
                linha["nao_sei_entre_mantidas"] = (float((mantidas["rotulo"] == "?").mean())
                                                   if len(mantidas) else None)
                linha["rotuladas_cortadas"] = len(cortadas)
            resultado["cortes"].append(linha)
    return resultado


# --------------------------------------------------------------------------- avaliar
def avaliar(pasta_classificador: Path, saida: Path, dispositivo: str = "cuda") -> dict:
    """Proxies, fragmentos e (com o CSV preenchido) métricas rotuladas de um classificador nas
    folhas de validação do BRACOT. Grava campo.json, campo_folhas.csv e campo.md em `saida`."""
    modelos = inf.carregar_modelos(pasta_classificador, dispositivo=dispositivo)
    config = CONFIG_ROTULAGEM
    linhas = []
    for foto, reduzida, escala, folhas in percorrer_folhas(modelos, config):
        for folha in folhas:
            linha = {k: folha[k] for k in ("id", "foto", "folha", "score", "area", "borda",
                                           "fracao_visivel", "preenchimento")}
            linha.update(dict(zip(("caixa_x0", "caixa_y0", "caixa_x1", "caixa_y1"),
                                  folha["caixa"])))
            for modo in MODOS:
                base = recorte(reduzida, escala, folha, modo, config)
                versoes = [base]
                for _, tipo, valor in PERTURBACOES:
                    if tipo in ("brilho", "contraste"):
                        versoes.append(perturbar(base, tipo, valor))
                    else:
                        desloc = (valor, 0.0) if tipo == "x" else (0.0, valor)
                        versoes.append(recorte(reduzida, escala, folha, modo, config, desloc))
                probabilidades, niveis = _classificar_em_lotes(versoes, modelos)
                classes = probabilidades.argmax(axis=1)
                categoria = inf.CATEGORIAS[classes[0]]
                linha.update({f"classe_{modo}": categoria,
                              f"confianca_{modo}": float(probabilidades[0].max()),
                              f"severidade_{modo}": inf.severidade_do_contrato(categoria,
                                                                              int(niveis[0])),
                              f"estavel_{modo}": float(np.mean(classes[1:] == classes[0]))})
            linhas.append(linha)
    folhas = pd.DataFrame(linhas)
    resultado = {"classificador": train._exibir(Path(pasta_classificador).resolve()),
                 "proxies": resumir_proxies(folhas)}
    saida.mkdir(parents=True, exist_ok=True)
    folhas.to_csv(saida / "campo_folhas.csv", index=False, float_format="%.6f",
                  lineterminator="\n")
    return gravar_resultado(resultado, folhas, saida)


def gravar_resultado(resultado: dict, folhas: pd.DataFrame, saida: Path) -> dict:
    """Completa o resultado com os fragmentos e os rótulos (se o CSV tiver rótulos) e grava
    campo.json e campo.md em `saida`."""
    rotulos = ler_rotulos()
    casados = None if rotulos is None else casar_rotulos(rotulos, folhas)
    resultado["fragmentos"] = resumir_fragmentos(folhas, casados)
    resultado["rotulos"] = None
    if casados is not None:
        resultado["rotulos"] = {"casados": len(casados), "no_csv": len(rotulos),
                                "origem": ORIGEM_ROTULOS,
                                "fragmento": _fragmento_nos_rotulos(casados, folhas)}
        for modo in MODOS:
            tabela = casados.merge(folhas[["id", f"classe_{modo}"]], on="id").rename(
                columns={f"classe_{modo}": "prevista"})
            resultado["rotulos"][modo] = metricas_rotuladas(tabela)
    train.gravar_json(_nativo(resultado), saida / "campo.json")
    (saida / "campo.md").write_text(trecho_relatorio(resultado), encoding="utf-8", newline="\n")
    return resultado


def reavaliar_rotulos(saida: Path) -> dict:
    """Refaz só a parte dos rótulos de um run já avaliado (campo_folhas.csv e campo.json), sem
    rodar os modelos: as previsões por folha já estão gravadas."""
    folhas = pd.read_csv(saida / "campo_folhas.csv", dtype={"id": str, "foto": str})
    anterior = json.loads((saida / "campo.json").read_text(encoding="utf-8"))
    resultado = {"classificador": anterior["classificador"], "proxies": anterior["proxies"]}
    return gravar_resultado(resultado, folhas, saida)


def _fragmento_nos_rotulos(casados: pd.DataFrame, folhas: pd.DataFrame) -> dict:
    """As folhas com "fragmento" na observação: quantas são "?" e a fração visível delas x a
    das outras rotuladas."""
    tabela = casados.merge(folhas[["id", "fracao_visivel", "borda"]], on="id")
    marcado = tabela["observacao"].str.lower().str.contains("fragmento")
    return {"marcadas": int(marcado.sum()),
            "nao_sei_entre_marcadas": proporcao(int((tabela.loc[marcado, "rotulo"] == "?").sum()),
                                                int(marcado.sum())),
            "nao_sei_entre_demais": proporcao(int((tabela.loc[~marcado, "rotulo"] == "?").sum()),
                                              int((~marcado).sum())),
            "fracao_visivel_mediana_marcadas": float(tabela.loc[marcado, "fracao_visivel"]
                                                     .median()),
            "fracao_visivel_mediana_demais": float(tabela.loc[~marcado, "fracao_visivel"]
                                                   .median()),
            "na_borda_entre_marcadas": int(tabela.loc[marcado, "borda"].astype(bool).sum())}


def _classificar_em_lotes(recortes, modelos):
    probabilidades, niveis = [], []
    for i in range(0, len(recortes), LOTE):
        p, n = inf.classificar(recortes[i:i + LOTE], modelos)
        probabilidades.append(p)
        niveis.append(n)
    return np.concatenate(probabilidades), np.concatenate(niveis)


def _nativo(objeto):
    if isinstance(objeto, dict):
        return {str(k): _nativo(v) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple, np.ndarray)):
        return [_nativo(v) for v in objeto]
    if isinstance(objeto, np.generic):
        return objeto.item()
    return objeto


def _pct(valor, casas: int = 1) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "-"
    return fmt.pct(valor, 1, casas)


def _proporcao(p: dict) -> str:
    if not p["n"]:
        return "-"
    return f"{_pct(p['valor'])} ({_pct(p['ic95'][0])} a {_pct(p['ic95'][1])}; n={p['n']})"


def trecho_relatorio(resultado: dict) -> str:
    """O trecho de campo do relatório de uma variante (UTF-8)."""
    px, fr = resultado["proxies"], resultado["fragmentos"]
    linhas = [f"## Campo: {px['folhas']} folhas das fotos de validação do BRACOT", "",
              f"Classificador `{resultado['classificador']}`. Concordância de classe entre os "
              f"modos A e B: {_pct(px['concordancia_A_B'])}. A estabilidade é a fração de 8 "
              "perturbações pequenas (brilho e contraste de 10%, deslocamento de 5%) com a mesma "
              "classe. **Proxies não decidem adoção:** um modelo que diz sempre a mesma classe é "
              "perfeitamente estável.", ""]
    linhas += fmt.tabela(["", "modo A", "modo B"], [
        *([f"folhas {c}", *(str(px[m]["classes"][c]) for m in MODOS)] for c in inf.CATEGORIAS),
        *([f"severidade {s}", *(str(px[m]["severidades"][s]) for m in MODOS)]
          for s in inf.NIVEIS_CONTRATO),
        ["fração da classe mais comum", *(_pct(px[m]["classe_mais_comum"]) for m in MODOS)],
        ["confiança média", *(_pct(px[m]["confianca_media"]) for m in MODOS)],
        ["estabilidade média", *(_pct(px[m]["estabilidade_media"]) for m in MODOS)],
        ["estáveis nas 8 perturbações", *(_pct(px[m]["estaveis_em_todas"]) for m in MODOS)],
        ["borda: confiança / saudáveis", *(f"{_pct(px[m]['borda']['confianca_media'])} / "
                                           f"{_pct(px[m]['borda']['fracao_saudavel'])}"
                                           for m in MODOS)],
        ["inteiras: confiança / saudáveis", *(f"{_pct(px[m]['inteiras']['confianca_media'])} / "
                                              f"{_pct(px[m]['inteiras']['fracao_saudavel'])}"
                                              for m in MODOS)],
    ], "lrr")
    linhas += ["", "A severidade prevista no campo não é informativa: o modelo só aprendeu "
                   "severidade em folhas do BRACOL, e no campo não há rótulo de severidade.", ""]
    rot = resultado["rotulos"]
    if rot is None:
        linhas += ["**Rótulos de campo:** o CSV `data/manifests/campo_rotulado.csv` ainda não "
                   "tem rótulos.", ""]
    else:
        linhas += [f"**Rótulos de campo** ({rot['origem']}; {rot['casados']} de "
                   f"{rot['no_csv']} casados com as folhas atuais; sem os \"?\"):", ""]
        linhas += fmt.tabela(["", "modo A", "modo B"], [
            ["rotuladas (S / D / ?)", *(f"{rot[m]['S']} / {rot[m]['D']} / {rot[m]['nao_sei']}"
                                       for m in MODOS)],
            ["P(prever doença dado S)", *(_proporcao(rot[m]["doenca_dado_S"]) for m in MODOS)],
            ["P(prever saudável dado D)", *(_proporcao(rot[m]["saudavel_dado_D"]) for m in MODOS)],
            ["classe certa, dado D com classe", *(_proporcao(rot[m]["classe_certa_dado_D"])
                                                  for m in MODOS)],
            ["P(prever doença dado S), S com \"brilho\"",
             *(_proporcao(rot[m]["doenca_dado_S_com_brilho"]) for m in MODOS)],
            ["P(prever doença dado S), S sem \"brilho\"",
             *(_proporcao(rot[m]["doenca_dado_S_sem_brilho"]) for m in MODOS)],
        ], "lll")
        fr_rot = rot["fragmento"]
        linhas += ["", f"Folhas com \"fragmento\" na observação: {fr_rot['marcadas']} "
                       f"({fr_rot['na_borda_entre_marcadas']} na borda); \"?\" entre elas "
                       f"{_proporcao(fr_rot['nao_sei_entre_marcadas'])} e entre as demais "
                       f"{_proporcao(fr_rot['nao_sei_entre_demais'])}; fração visível mediana "
                       f"{_pct(fr_rot['fracao_visivel_mediana_marcadas'])} x "
                       f"{_pct(fr_rot['fracao_visivel_mediana_demais'])}.", ""]
    linhas += [f"**Fragmentos** ({fr['na_borda']} das {fr['folhas']} folhas cortadas pela borda;"
               f" fração visível mediana {_pct(fr['fracao_visivel']['mediana_borda'])} na borda e"
               f" {_pct(fr['fracao_visivel']['mediana_inteiras'])} nas inteiras):", ""]
    linhas += fmt.tabela(["filtro", "limiar", "cortadas", "na borda", "\"?\" entre cortadas",
                          "\"?\" entre mantidas"],
                         [[c["filtro"], fmt.decimal(c["limiar"], 2), c["cortadas"],
                           c["cortadas_na_borda"], _pct(c.get("nao_sei_entre_cortadas")),
                           _pct(c.get("nao_sei_entre_mantidas"))] for c in fr["cortes"]])
    return "\n".join(linhas) + "\n"


# -------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Campo: rotulagem cega e avaliacao do "
                                             "classificador nas folhas de validacao do BRACOT.")
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--gerar-rotulagem", action="store_true",
                      help="pranchas cegas e CSV-modelo de rotulagem")
    modo.add_argument("--avaliar", action="store_true", help="proxies e rotulos de um "
                                                             "classificador")
    modo.add_argument("--so-rotulos", action="store_true",
                      help="refaz so os rotulos de um run ja avaliado (--saida), sem modelos")
    ap.add_argument("--fotos", choices=("validacao", "treino"), default="validacao",
                    help="com --gerar-rotulagem: as 46 fotos de validacao do detector (passo 4) "
                         "ou as 194 de treino (rotulagem de validacao do passo 5b)")
    ap.add_argument("--classificador", type=Path, default=inf.PASTA_CLASSIFICADOR,
                    help="pasta do run do classificador (com o melhor.pt)")
    ap.add_argument("--saida", help="com --avaliar: pasta em model/runs para os resultados")
    args = ap.parse_args(argv)
    try:
        import torch

        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        if args.gerar_rotulagem:
            modelos = inf.carregar_modelos(dispositivo=dispositivo)
            if args.fotos == "treino":
                pasta = PASTA_ROTULAGEM_TREINO
                r = gerar_rotulagem(modelos, pasta, MANIFEST_CAMPO_TREINO,
                                    CONFIG_ROTULAGEM_TREINO, fotos_de_treino(),
                                    N_ROTULAGEM_TREINO)
            else:
                pasta = PASTA_ROTULAGEM
                r = gerar_rotulagem(modelos)
            fmt.dizer(f"{r['folhas']} folhas de {r['fotos']} fotos (de {r['total_de_folhas']}) em"
                      f" {len(r['pranchas'])} pranchas em {pasta}; CSV-modelo em "
                      f"{train._exibir(r['manifest'])}")
            return 0
        if not args.saida:
            ap.error("--avaliar e --so-rotulos precisam de --saida")
        if args.so_rotulos:
            r = reavaliar_rotulos(train.PASTA_RUNS / args.saida)
            fmt.dizer(f"{args.saida}: rotulos casados "
                      f"{r['rotulos']['casados'] if r['rotulos'] else 0}")
            return 0
        saida = train.PASTA_RUNS / args.saida
        if saida.exists() and not saida.name.startswith("campo_"):
            ap.error("--saida precisa ser uma pasta nova ou campo_*: os runs antigos ficam "
                     "intactos")
        r = avaliar(args.classificador, saida, dispositivo)
        px = r["proxies"]
        fmt.dizer(f"A x B: {_pct(px['concordancia_A_B'])} | saudaveis A "
                  f"{_pct(px['A']['fracao_saudavel'])} B {_pct(px['B']['fracao_saudavel'])} | "
                  "estabilidade A "
                  f"{_pct(px['A']['estabilidade_media'])} B {_pct(px['B']['estabilidade_media'])}")
        return 0
    except (train.ErroFatal, FileNotFoundError, ValueError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
