"""
Filtro de cor de folha (Frente 9, passo 8; decisão do gestor, 30/10/2026): mede o matiz das
máscaras para recusar objetos que não são folha. O filtro de saturação da cadeia
(inf.Configuracao.fracao_colorida_minima) deixa passar qualquer objeto colorido; este passo mede
a faixa de matiz das folhas e a fração da máscara dentro dela.

a) Folhas anotadas do treino do BRACOT (1.318): histograma conjunto do matiz (H do HSV do
   OpenCV, 0 a 179) e do brilho (V, em classes de BRILHO_PASSO) dos pixels com saturação >=
   inf.Configuracao.saturacao_minima, na cópia reduzida a inf.LADO_COR, como a cadeia mede. Para
   cada brilho mínimo de BRILHOS_MINIMOS (o matiz de pixel escuro é ruído: as sombras das folhas
   puxam para o azul), a faixa é o menor arco de matiz que cobre FRACAO_PIXELS dos pixels
   coloridos e claros das folhas (cada folha pesa igual), e o corte é o p1 da fração da máscara
   dentro da faixa, como no filtro de saturação (medido no treino em 10/10/2026: p1 de 35%, que
   este script recalcula como conferência).
b) Detecções do detector nas mesmas fotos (score >= o menor limiar da cadeia), casadas pela caixa
   com as folhas S, D e "?" do campo rotulado de treino: quantas o corte tiraria. O detector
   treinou nessas fotos: as máscaras saem mais limpas que no campo.
c) Fotos sem planta (data/negativas/, fora do git; créditos em
   data/manifests/negativas_creditos.csv): nunca escolhem a faixa nem o corte. Elas motivaram o
   brilho mínimo (30/10/2026): daqui em diante são conjunto de desenvolvimento, e o número nelas
   sai otimista.
d) A cadeia inteira (diagnosticar_arquivo), sem e com o filtro adotado em inf.Configuracao, nas
   fotos sem planta e numa amostra das fotos de treino do BRACOT: a resposta ao usuário.

Os testes fechados (BRACOL e BRACOT) não são lidos. Saída: model/runs/cor_de_folha/ (medicao.json
e relatorio.md); os histogramas ficam em data/processed/cor_de_folha/ (fora do git).

Uso (de qualquer pasta, com o venv ativo):
    python evaluation/cor_de_folha.py               # mede a), b) e c) (alguns minutos na CPU)
    python evaluation/cor_de_folha.py --negativas   # só remede c), com os histogramas de a) e b)
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import bracot  # noqa: E402
import campo  # noqa: E402
import dados  # noqa: E402
import detector  # noqa: E402
import formatacao as fmt  # noqa: E402
import inferencia as inf  # noqa: E402
import train  # noqa: E402

PASTA_SAIDA = train.PASTA_RUNS / "cor_de_folha"
PASTA_HISTOGRAMAS = RAIZ_REPO / "data" / "processed" / "cor_de_folha"  # fora do git
PASTA_NEGATIVAS = RAIZ_REPO / "data" / "negativas"  # fora do git
EXTENSOES = (".jpg", ".jpeg", ".png")
MATIZES = 180  # H do HSV do OpenCV em uint8: 0 a 179
BRILHO_PASSO = 16  # classes de V de 16 em 16
BRILHOS = 256 // BRILHO_PASSO
BRILHOS_MINIMOS = (0, 32, 48, 64, 80, 96)  # múltiplos de BRILHO_PASSO
FRACAO_PIXELS = 0.99
PERCENTIL_CORTE = 1
IOU_CASAMENTO = 0.5
AMOSTRA_BRACOT = 8  # na comparação da cadeia inteira: uma foto de treino a cada 8 (30 fotos)
CONFIG = inf.Configuracao()


# --------------------------------------------------------------------------- matiz
def hsv_reduzido(imagem) -> tuple[np.ndarray, float]:
    """O HSV da cópia reduzida a inf.LADO_COR e a escala, como inf.detectar mede a cor."""
    reduzida, escala = inf.reduzir(imagem, inf.LADO_COR)
    return cv2.cvtColor(reduzida, cv2.COLOR_RGB2HSV), escala


def histograma(hsv, poligono) -> tuple[np.ndarray, int]:
    """Histograma conjunto (MATIZES x BRILHOS) do matiz e do brilho dos pixels coloridos da
    máscara do polígono (na escala do hsv) e o total de pixels da máscara."""
    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [np.round(np.asarray(poligono, dtype=np.float64).reshape(-1, 2))
                           .astype(np.int32)], 1)
    pixels = hsv[mascara.astype(bool)]
    coloridos = pixels[pixels[:, 1] >= CONFIG.saturacao_minima]
    indice = coloridos[:, 0].astype(np.int64) * BRILHOS + coloridos[:, 2] // BRILHO_PASSO
    return np.bincount(indice, minlength=MATIZES * BRILHOS).reshape(MATIZES, BRILHOS), len(pixels)


def matiz(histogramas: np.ndarray, brilho_minimo: int) -> np.ndarray:
    """O histograma do matiz só dos pixels com brilho >= brilho_minimo (múltiplo de
    BRILHO_PASSO)."""
    return histogramas[..., brilho_minimo // BRILHO_PASSO:].sum(axis=-1)


def faixa_minima(histogramas: np.ndarray, fracao: float) -> tuple[int, int]:
    """O menor arco de matiz (início, fim, inclusive; passa do 179 ao 0 se início > fim) que
    cobre `fracao` dos pixels de histogramas de matiz, com cada histograma pesando igual."""
    somas = histogramas.sum(axis=1, keepdims=True)
    proporcao = (histogramas[somas[:, 0] > 0] / somas[somas[:, 0] > 0]).mean(axis=0)
    melhor = None
    for inicio in range(MATIZES):
        acumulado = np.cumsum(np.roll(proporcao, -inicio))
        comprimento = int(np.searchsorted(acumulado, fracao - 1e-12)) + 1
        if melhor is None or comprimento < melhor[1]:
            melhor = (inicio, comprimento)
    inicio, comprimento = melhor
    return inicio, (inicio + comprimento - 1) % MATIZES


def na_faixa(histogramas: np.ndarray, faixa: tuple[int, int]) -> np.ndarray:
    """Pixels de cada histograma de matiz dentro da faixa."""
    inicio, fim = faixa
    if inicio <= fim:
        return histogramas[..., inicio:fim + 1].sum(axis=-1)
    return histogramas[..., inicio:].sum(axis=-1) + histogramas[..., :fim + 1].sum(axis=-1)


def fracoes(histogramas: np.ndarray, totais: np.ndarray, faixa,
            brilho_minimo: int) -> tuple[np.ndarray, np.ndarray]:
    """(fração colorida, como o filtro atual; fração com cor de folha e brilho >= brilho_minimo)
    de cada máscara; máscara vazia: 0."""
    totais = np.maximum(np.asarray(totais), 1)
    colorida = histogramas.sum(axis=(-2, -1)) / totais
    return colorida, na_faixa(matiz(histogramas, brilho_minimo), faixa) / totais


# --------------------------------------------------------------------------- dados
def folhas_anotadas():
    """Fotos de treino do BRACOT com os polígonos anotados (COCO dos autores)."""
    coco = json.loads(bracot.caminho_coco("treino").read_text(encoding="utf-8"))
    poligonos = {}
    for anotacao in coco["annotations"]:
        for segmento in anotacao["segmentation"]:
            poligonos.setdefault(anotacao["image_id"], []).append(
                np.asarray(segmento, dtype=np.float64).reshape(-1, 2))
    pasta = bracot.caminho_coco("treino").parent
    return [(pasta / imagem["file_name"], poligonos.get(imagem["id"], []))
            for imagem in sorted(coco["images"], key=lambda i: i["file_name"])]


def rotulos_de_campo() -> dict[str, list[dict]]:
    """As folhas do campo rotulado de treino, por foto, com a caixa normalizada (0 a 1)."""
    por_foto = {}
    with open(campo.MANIFEST_CAMPO_TREINO, encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            caixa = [float(linha[f"caixa_{c}"]) for c in ("x0", "y0", "x1", "y1")]
            por_foto.setdefault(linha["foto"], []).append({"rotulo": linha["rotulo"],
                                                           "caixa": caixa})
    return por_foto


def iou(a, b) -> float:
    largura = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    altura = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersecao = largura * altura
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


def rotulo_da_deteccao(caixa_normalizada, rotulados) -> str:
    """O rótulo da folha rotulada de maior IoU com a caixa (>= IOU_CASAMENTO), ou "" (sem
    rótulo)."""
    melhor = max(rotulados, key=lambda r: iou(caixa_normalizada, r["caixa"]), default=None)
    if melhor is None or iou(caixa_normalizada, melhor["caixa"]) < IOU_CASAMENTO:
        return ""
    return melhor["rotulo"]


def deteccoes(imagem, modelo_detector) -> list[dict]:
    """As detecções da cadeia (até o menor dos dois limiares), com histograma, total, área e a
    caixa normalizada."""
    limiar = min(CONFIG.limiar_deteccao, CONFIG.score_minimo_folha_unica)
    folhas = detector.detectar_folhas(imagem, limiar, detector=modelo_detector, dispositivo="cpu")
    altura, largura = imagem.shape[:2]
    hsv, escala = hsv_reduzido(imagem)
    saida = []
    for folha in folhas:
        hist, total = histograma(hsv, np.asarray(folha["poligono"]) * escala)
        x0, y0, x1, y1 = folha["caixa"]
        saida.append({"score": float(folha["score"]), "hist": hist, "total": total,
                      "area": inf.area_do_poligono(folha["poligono"]) / (altura * largura),
                      "caixa": [x0 / largura, y0 / altura, x1 / largura, y1 / altura]})
    return saida


def _gravar(nome: str, campos: dict) -> None:
    PASTA_HISTOGRAMAS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PASTA_HISTOGRAMAS / nome, **{k: np.asarray(v) for k, v in campos.items()})


# --------------------------------------------------------------------------- medição
def medir_bracot(modelo_detector) -> None:
    """Histogramas das folhas anotadas e das detecções nas fotos de treino do BRACOT."""
    rotulados = rotulos_de_campo()
    campos = {k: [] for k in ("anotadas_hist", "anotadas_total", "detectadas_hist",
                              "detectadas_total", "detectadas_score", "detectadas_area",
                              "detectadas_rotulo", "detectadas_foto")}
    fotos = folhas_anotadas()
    for i, (caminho, poligonos) in enumerate(fotos, 1):
        imagem = dados.ler_imagem(caminho)
        hsv, escala = hsv_reduzido(imagem)
        for poligono in poligonos:
            hist, total = histograma(hsv, poligono * escala)
            campos["anotadas_hist"].append(hist)
            campos["anotadas_total"].append(total)
        for d in deteccoes(imagem, modelo_detector):
            for chave in ("hist", "total", "score", "area"):
                campos[f"detectadas_{chave}"].append(d[chave])
            campos["detectadas_rotulo"].append(rotulo_da_deteccao(
                d["caixa"], rotulados.get(caminho.stem, [])))
            campos["detectadas_foto"].append(caminho.stem)
        if i % 20 == 0 or i == len(fotos):
            fmt.dizer(f"  {i}/{len(fotos)} fotos")
    _gravar("bracot_treino.npz", campos)


def medir_negativas(modelo_detector) -> bool:
    """Histogramas das detecções nas fotos sem planta; False se não houver fotos."""
    fotos = sorted(p for p in PASTA_NEGATIVAS.glob("*") if p.suffix.lower() in EXTENSOES)
    if not fotos:
        return False
    campos = {k: [] for k in ("hist", "total", "score", "area", "foto")}
    for caminho in fotos:
        imagem = dados.decodificar_imagem(np.fromfile(caminho, dtype=np.uint8), caminho.name)
        for d in deteccoes(imagem, modelo_detector):
            for chave in ("hist", "total", "score", "area"):
                campos[chave].append(d[chave])
            campos["foto"].append(caminho.name)
    campos["fotos"] = [p.name for p in fotos]
    _gravar("negativas.npz", campos)
    return True


# --------------------------------------------------------------------------- análise
def resumo(valores: np.ndarray) -> dict:
    return {"n": int(len(valores)), "min": float(valores.min()),
            "p1": float(np.percentile(valores, 1)), "p5": float(np.percentile(valores, 5)),
            "mediana": float(np.median(valores))}


def utilizaveis_hoje(h, prefixo: str = "") -> np.ndarray:
    """As detecções que a cadeia usa hoje (limiar, área mínima e o filtro de saturação)."""
    colorida = h[f"{prefixo}hist"].sum(axis=(-2, -1)) / np.maximum(h[f"{prefixo}total"], 1)
    return ((h[f"{prefixo}score"] >= CONFIG.limiar_deteccao)
            & (h[f"{prefixo}area"] >= CONFIG.area_minima)
            & (colorida >= CONFIG.fracao_colorida_minima))


def variante(h, neg, brilho_minimo: int) -> dict:
    """Faixa, corte e efeito de um brilho mínimo nas folhas, nas detecções e nas negativas."""
    faixa = faixa_minima(matiz(h["anotadas_hist"], brilho_minimo), FRACAO_PIXELS)
    colorida, folha = fracoes(h["anotadas_hist"], h["anotadas_total"], faixa, brilho_minimo)
    corte = math.floor(np.percentile(folha, PERCENTIL_CORTE) * 100) / 100
    _, det_folha = fracoes(h["detectadas_hist"], h["detectadas_total"], faixa, brilho_minimo)
    utilizaveis = utilizaveis_hoje(h, "detectadas_")
    cortadas = utilizaveis & (det_folha < corte)
    por_rotulo = {}
    for rotulo, nome in (("S", "S"), ("D", "D"), ("?", "?"), ("", "sem_rotulo")):
        sel = utilizaveis & (h["detectadas_rotulo"] == rotulo)
        por_rotulo[nome] = {"utilizaveis": int(sel.sum()), "cortadas": int((sel & cortadas).sum())}
    fotos = h["detectadas_foto"]
    sem_folha = sum(1 for f in set(fotos[utilizaveis])
                    if not (utilizaveis & ~cortadas & (fotos == f)).any())
    saida = {"brilho_minimo": brilho_minimo, "faixa_matiz": list(faixa),
             "faixa_graus": [faixa[0] * 2, faixa[1] * 2], "corte": corte,
             "anotadas": {"colorida": resumo(colorida), "cor_de_folha": resumo(folha),
                          "abaixo_do_corte": int((folha < corte).sum())},
             "detectadas": {"total": int(len(det_folha)), "utilizaveis": int(utilizaveis.sum()),
                            "cortadas": int(cortadas.sum()), "fotos_que_perdem_todas": sem_folha,
                            "por_rotulo": por_rotulo}}
    if neg is not None:
        saida["negativas"] = avaliar_negativas(neg, faixa, corte, brilho_minimo)
    return saida


def avaliar_negativas(neg, faixa, corte: float, brilho_minimo: int) -> dict:
    """Por foto sem planta: as detecções utilizáveis hoje e as que sobram com o corte."""
    usa_hoje = (utilizaveis_hoje(neg) if len(neg["score"])
                else np.zeros(0, dtype=bool))
    _, folha = (fracoes(neg["hist"], neg["total"], faixa, brilho_minimo) if len(neg["score"])
                else (None, np.zeros(0)))
    por_foto = {}
    for nome in neg["fotos"]:
        sel = neg["foto"] == nome
        por_foto[str(nome)] = {"utilizaveis_hoje": int((sel & usa_hoje).sum()),
                               "com_corte": int((sel & usa_hoje & (folha >= corte)).sum()),
                               "maior_cor_de_folha": float(folha[sel].max()) if sel.any() else 0.0}
    return {"fotos": len(por_foto),
            "com_folha_hoje": sum(v["utilizaveis_hoje"] > 0 for v in por_foto.values()),
            "com_folha_com_corte": sum(v["com_corte"] > 0 for v in por_foto.values()),
            "por_foto": por_foto}


def comparar_cadeia() -> dict:
    """A cadeia inteira (inf.diagnosticar_arquivo: detector, classificador e regras) sem e com o
    filtro de matiz, nas fotos sem planta e numa amostra das fotos de treino do BRACOT (uma a
    cada AMOSTRA_BRACOT, na ordem do nome): a resposta que chega ao usuário."""
    modelos = inf.carregar_modelos(dispositivo="cpu")
    configs = {"sem_filtro": inf.Configuracao(fracao_cor_de_folha_minima=0.0),
               "com_filtro": inf.Configuracao()}
    grupos = {"negativas": sorted(p for p in PASTA_NEGATIVAS.glob("*")
                                  if p.suffix.lower() in EXTENSOES),
              "bracot_treino": [caminho for caminho, _ in folhas_anotadas()][::AMOSTRA_BRACOT]}
    saida = {}
    for grupo, fotos in grupos.items():
        linhas = []
        for caminho in fotos:
            conteudo = caminho.read_bytes()
            linha = {"foto": caminho.name}
            for nome, config in configs.items():
                r = inf.diagnosticar_arquivo(conteudo, config, modelos)
                linha[nome] = {"status": r["status"], "tipo": r.get("tipo"),
                               "folhas": len(r.get("folhas", []))}
            linhas.append(linha)
        saida[grupo] = {"fotos": len(linhas),
                        **{f"sucesso_{nome}": sum(l[nome]["status"] == "sucesso" for l in linhas)
                           for nome in configs},
                        "mudaram": [l for l in linhas if l["sem_filtro"] != l["com_filtro"]]}
    return saida


def analisar() -> dict:
    h = np.load(PASTA_HISTOGRAMAS / "bracot_treino.npz")
    caminho_neg = PASTA_HISTOGRAMAS / "negativas.npz"
    neg = np.load(caminho_neg) if caminho_neg.is_file() else None
    return {"fracao_pixels": FRACAO_PIXELS, "saturacao_minima": CONFIG.saturacao_minima,
            "variantes": [variante(h, neg, b) for b in BRILHOS_MINIMOS]}


# --------------------------------------------------------------------------- relatório
def _pct(valor: float) -> str:
    return f"{valor * 100:.1f}%".replace(".", ",")


def gravar_relatorio(medicao: dict) -> None:
    vs = medicao["variantes"]
    base = vs[0]
    linhas = [
        "# Filtro de cor de folha (passo 8)", "",
        "Decisão do gestor, 30/10/2026. Gerado por `evaluation/cor_de_folha.py`; números em "
        "`medicao.json`. Só o treino do BRACOT e as fotos sem planta; os testes fechados não são "
        "lidos.", "",
        "## Método", "",
        f"- Pixel colorido: saturação >= {medicao['saturacao_minima']} (HSV do OpenCV, na cópia "
        f"reduzida a {inf.LADO_COR} px, como a cadeia). Pixel com cor de folha: colorido, com "
        "brilho (V) >= o brilho mínimo e matiz dentro da faixa.",
        f"- Para cada brilho mínimo, a faixa é o menor arco de matiz que cobre "
        f"{_pct(medicao['fracao_pixels'])} dos pixels coloridos e claros das "
        f"{base['anotadas']['cor_de_folha']['n']} folhas anotadas do treino (cada folha pesa "
        "igual), e o corte é o p1 da fração da máscara com cor de folha, arredondado para baixo.",
        "- O brilho mínimo existe porque o matiz de pixel escuro é ruído: com brilho mínimo 0, as "
        "sombras das folhas levam a faixa até o azul, e um notebook azul-marinho passou.", "",
        f"Conferência do método: o p1 da fração colorida deu "
        f"{_pct(base['anotadas']['colorida']['p1'])}; em 10/10/2026, 35%.", "",
        "## Sensibilidade ao brilho mínimo", "",
        "| brilho mínimo | faixa de matiz (H) | graus | corte | folhas anotadas abaixo | "
        "detecções cortadas | S cortadas | D cortadas | fotos sem planta com folha |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in vs:
        d = v["detectadas"]
        neg = v.get("negativas")
        neg_txt = (f"{neg['com_folha_com_corte']} de {neg['fotos']}" if neg else "-")
        linhas.append(
            f"| {v['brilho_minimo']} | {v['faixa_matiz'][0]} a {v['faixa_matiz'][1]} | "
            f"{v['faixa_graus'][0]}° a {v['faixa_graus'][1]}° | {_pct(v['corte'])} | "
            f"{v['anotadas']['abaixo_do_corte']} | {d['cortadas']} de {d['utilizaveis']} | "
            f"{d['por_rotulo']['S']['cortadas']} de {d['por_rotulo']['S']['utilizaveis']} | "
            f"{d['por_rotulo']['D']['cortadas']} de {d['por_rotulo']['D']['utilizaveis']} | "
            f"{neg_txt} |")
    neg0 = base.get("negativas")
    if neg0:
        linhas += ["", f"Sem o filtro de matiz (só o de saturação, hoje): "
                   f"{neg0['com_folha_hoje']} de {neg0['fotos']} fotos sem planta com folha.", "",
                   "## Fotos sem planta, por foto (folhas utilizáveis que sobram)", "",
                   "Conjunto de desenvolvimento desde 30/10/2026 (motivou o brilho mínimo): o "
                   "número aqui sai otimista.", "",
                   "| foto | hoje | " + " | ".join(f"V >= {v['brilho_minimo']}" for v in vs)
                   + " |", "|---|---:|" + "---:|" * len(vs)]
        for nome, info in neg0["por_foto"].items():
            linhas.append(f"| {nome} | {info['utilizaveis_hoje']} | " + " | ".join(
                str(v["negativas"]["por_foto"][nome]["com_corte"]) for v in vs) + " |")
    cadeia = medicao.get("cadeia")
    if cadeia:
        adotado = next(v for v in vs if v["brilho_minimo"] == 0)
        linhas += ["", "## A cadeia inteira, sem e com o filtro adotado", "",
                   f"Adotado (decisão do gestor, 30/10/2026): brilho mínimo 0 (não adotado), faixa "
                   f"{adotado['faixa_matiz'][0]} a {adotado['faixa_matiz'][1]} e corte "
                   f"{_pct(adotado['corte'])}, em `inferencia.Configuracao`. Resposta de "
                   "`inferencia.diagnosticar_arquivo` (detector, classificador e regras), que é a "
                   "que chega ao usuário.", "",
                   "| conjunto | fotos | sucesso sem o filtro | sucesso com o filtro |",
                   "|---|---:|---:|---:|"]
        nomes = {"negativas": "sem planta (desenvolvimento)",
                 "bracot_treino": f"treino do BRACOT (1 a cada {AMOSTRA_BRACOT})"}
        for grupo, r in cadeia.items():
            linhas.append(f"| {nomes[grupo]} | {r['fotos']} | {r['sucesso_sem_filtro']} | "
                          f"{r['sucesso_com_filtro']} |")
        texto = lambda r: (f"sucesso, {r['folhas']} folha(s)" if r["status"] == "sucesso"
                           else r["tipo"])
        for grupo, r in cadeia.items():
            linhas += ["", f"Mudaram ({nomes[grupo]}):", "", "| foto | sem o filtro | com o filtro |",
                       "|---|---|---|"]
            linhas += [f"| {l['foto']} | {texto(l['sem_filtro'])} | {texto(l['com_filtro'])} |"
                       for l in r["mudaram"]] or ["| nenhuma | | |"]
    linhas += ["", "## Limitações", "",
               "- O corte sai das folhas anotadas do treino, nas fotos em que o detector treinou.",
               "- Só folhas de campo do BRACOT: as folhas do BRACOL (foto inteira, fundo claro) "
               "não estão nesta máquina e não foram medidas.",
               "- Objeto com cor de folha (verde, amarelo) passa pelo filtro.",
               "- As fotos sem planta do Commons motivaram o brilho mínimo: o número honesto sai "
               "de um conjunto novo, que não guiou nenhuma escolha."]
    (PASTA_SAIDA / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mede o filtro de cor de folha (passo 8).")
    parser.add_argument("--negativas", action="store_true",
                        help="só remede as fotos sem planta, com os histogramas do BRACOT gravados")
    args = parser.parse_args()
    try:
        if args.negativas and not (PASTA_HISTOGRAMAS / "bracot_treino.npz").is_file():
            raise train.ErroFatal("histogramas do BRACOT nao encontrados: rode sem --negativas")
        modelo_detector = detector.carregar_detector(detector.PESOS_PADRAO)
        if not args.negativas:
            fmt.dizer("Medindo as folhas anotadas e as deteccoes do treino do BRACOT...")
            medir_bracot(modelo_detector)
        fmt.dizer("Medindo as fotos sem planta (data/negativas/)...")
        if not medir_negativas(modelo_detector):
            (PASTA_HISTOGRAMAS / "negativas.npz").unlink(missing_ok=True)
        medicao = analisar()
        fmt.dizer("Comparando a cadeia inteira sem e com o filtro...")
        medicao["cadeia"] = comparar_cadeia()
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        (PASTA_SAIDA / "medicao.json").write_text(
            json.dumps(medicao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        gravar_relatorio(medicao)
        for v in medicao["variantes"]:
            neg = v.get("negativas")
            fmt.dizer(f"V>={v['brilho_minimo']:3d} faixa {v['faixa_matiz']} corte {v['corte']:.2f}"
                      f" | deteccoes cortadas {v['detectadas']['cortadas']}"
                      + (f" | negativas com folha {neg['com_folha_com_corte']}/{neg['fotos']}"
                         if neg else ""))
        fmt.dizer(f"Relatorio: {PASTA_SAIDA / 'relatorio.md'}")
        return 0
    except train.ErroFatal as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
