"""
BRACOT: contrato, manifest (data/manifests/bracot.csv) e relatório de integridade
(data/reports/integridade_bracot.md).

Fonte de detecção e segmentação de folhas (RF09; decisão do gestor, 09/10/2026): fotos de campo
de pés de café (4032x3024) com folhas contornadas por polígonos (COCO, uma categoria "leaf"),
sem classe de estresse por folha. A divisão oficial é a dos autores: 240 fotos em train/ e 60 em
test/. As anotações ficam em data/raw/ e não são copiadas nem convertidas.

O manifest tem uma linha por foto: tamanho, split dos autores, número de folhas, área coberta,
data e hora (do nome do arquivo), cena, hashes e o arquivo COCO de origem. A área coberta é a
soma das áreas dos polígonos dividida pela área da foto: folhas que se tocam contam duas vezes,
e a diferença em relação à união é de no máximo 0,09 ponto percentual (medido em 09/10/2026).

Uso (de qualquer pasta):
    python data/bracot.py               gera o manifest e o relatório
    python data/bracot.py --verificar   confere dados x manifest, sem alterá-lo

Leitura (detector, Frente 9): ler_manifest(split=...) e caminho_coco(split).
O código de saída é 0 sem erros e 1 com erros; com erros, o manifest não é gravado.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

import bracol
import formatacao as fmt
import organize_dataset as od

# ------------------------------------------------------------------------ caminhos
PASTA_BRACOT = bracol.RAIZ_REPO / "data" / "raw" / "bracot" / "bracot-data"
MANIFEST_BRACOT = bracol.RAIZ_REPO / "data" / "manifests" / "bracot.csv"
RELATORIO_BRACOT = bracol.PASTA_REPORTS / "integridade_bracot.md"

# -------------------------------------------------------------------------- contrato
FONTE = "bracot"
# Pasta dos autores -> (split do projeto, arquivo COCO). O via_export_json.json fica ao lado.
SPLITS_DOS_AUTORES = {
    "train": ("treino", "train_annotation_coco.json"),
    "test": ("teste", "test_annotation_coco.json"),
}
SPLITS = ("treino", "teste")
ARQUIVO_VIA = "via_export_json.json"
CATEGORIA_FOLHA = (0, "leaf")  # (id, nome): a única categoria do COCO
NOME_IMAGEM = re.compile(r"^(\d{8}_\d{6})\.jpg$")  # AAAAMMDD_HHMMSS.jpg, a hora do celular

# Polígonos que passam da borda até TOLERANCIA_BORDA px viram aviso (28 passam 3 ou 4 px; quem
# lê recorta na borda); acima disso, erro.
TOLERANCIA_BORDA = 4

# Cena: fotos tiradas em sequência, a até INTERVALO_CENA segundos da anterior (as 300 em ordem de
# hora, sem olhar o split). Calibração (09/10/2026): a mediana do intervalo entre fotos é 7 s.
# Por pontos casados (ORB + RANSAC a 1/4 da resolução, nos 8.540 pares a até 300 s), só 2 pares
# mostram as mesmas folhas (TESTE_SOBREPOSTO), e 10 s é o menor intervalo inteiro que deixa cada
# par numa cena só (com 9 s, 164937 e 165013 se separam). Dá 90 cenas. A divisão oficial continua
# a dos autores; a cena mede o vazamento e guia a validação do detector, que separa cenas
# inteiras do treino, nunca imagens soltas.
INTERVALO_CENA = 10

# Imagens de teste que repetem folhas de uma imagem de treino: (teste, treino). Medidas por
# pontos casados (ORB + RANSAC) e conferidas visualmente em 09/10/2026; todos os outros pares
# ficaram com 13 pontos ou menos. A Frente 9 reporta o detector no teste inteiro e sem elas.
TESTE_SOBREPOSTO = (
    ("20190831_164920", "20190831_164921"),  # 1 s depois, 1.923 pontos: quase a mesma foto
    ("20190831_165013", "20190831_164937"),  # 36 s antes, 85 pontos: o mesmo galho, outro ângulo
)

COLUNAS_MANIFEST = [
    "fonte", "id", "caminho", "largura", "altura", "split", "n_folhas", "area_coberta",
    "data_hora", "cena", "sha256", "phash", "anotacao",
]


# ------------------------------------------------------------------- funções puras
def data_hora_do_nome(nome: str) -> datetime | None:
    """'20190831_163057' -> datetime(2019, 8, 31, 16, 30, 57); None se não for uma data válida."""
    try:
        return datetime.strptime(nome, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def area_do_poligono(coordenadas) -> float:
    """Área (fórmula do laço) de um polígono COCO [x1, y1, x2, y2, ...], fechado ou não."""
    xs = np.asarray(coordenadas[0::2], dtype=float)
    ys = np.asarray(coordenadas[1::2], dtype=float)
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def cenas(datas: dict, intervalo: int = INTERVALO_CENA) -> dict:
    """Agrupa as fotos por proximidade de tempo: em ordem de hora, uma foto a até `intervalo`
    segundos da anterior fica na cena dela. datas: {id: datetime}. Devolve {id: id da primeira
    foto da cena}."""
    resultado, primeira, anterior = {}, None, None
    for i in sorted(datas, key=lambda i: (datas[i], i)):
        if anterior is None or (datas[i] - datas[anterior]).total_seconds() > intervalo:
            primeira = i
        resultado[i] = primeira
        anterior = i
    return resultado


def hash_da_divisao(linhas) -> str:
    """SHA-256 de "id:split" em ordem de id, unidos por ";" (fixado nos testes)."""
    texto = ";".join(f"{linha['id']}:{linha['split']}"
                     for linha in sorted(linhas, key=lambda linha: linha["id"]))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------------- leitura
def ler_manifest(caminho=MANIFEST_BRACOT, split: str | None = None):
    """Manifest do BRACOT como pandas.DataFrame, uma linha por foto, com tipos fixos.

    split: "treino" ou "teste" (a divisão oficial, dos autores). Para validar o detector,
    separe cenas inteiras do treino pela coluna `cena`, nunca imagens soltas.
    """
    import pandas as pd  # só quem lê o manifest precisa do pandas

    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    if list(df.columns) != COLUNAS_MANIFEST:
        raise ValueError(f"colunas inesperadas em {caminho}: {list(df.columns)}")
    for coluna in ["largura", "altura", "n_folhas"]:
        df[coluna] = df[coluna].astype("int64")
    df["area_coberta"] = df["area_coberta"].astype("float64")
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split desconhecido: {split!r} (esperado um de {SPLITS})")
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


def caminho_coco(split: str, entrada: Path = PASTA_BRACOT) -> Path:
    """Arquivo COCO dos autores para o split ("treino" ou "teste"), dentro de data/raw/."""
    for pasta, (nome_split, arquivo) in SPLITS_DOS_AUTORES.items():
        if nome_split == split:
            return entrada / pasta / arquivo
    raise ValueError(f"split desconhecido: {split!r} (esperado um de {SPLITS})")


# ------------------------------------------------------------------ imagens e anotações
def inspecionar_imagens(entrada: Path, oc: od.Ocorrencias) -> dict[str, dict]:
    """Abre e decodifica por completo cada foto e calcula tamanho, orientação EXIF, SHA-256 e
    pHash. Devolve {id: dados}, com id = nome sem extensão."""
    arquivos = []
    for pasta, (split, _) in SPLITS_DOS_AUTORES.items():
        if not (entrada / pasta).is_dir():
            oc.erros.append(f"pasta não encontrada: {pasta}/")
            continue
        for p in sorted((entrada / pasta).iterdir()):
            if p.suffix.lower() == ".json":
                continue  # anotações: lidas à parte
            if p.is_file() and NOME_IMAGEM.match(p.name):
                arquivos.append((pasta, split, p))
            else:
                oc.avisos.append(f"ignorado em {pasta}/: {p.name} (fora do padrão "
                                 "AAAAMMDD_HHMMSS.jpg)")
    fmt.dizer(f"Abrindo {len(arquivos)} fotos (decodificacao completa, SHA-256 e pHash)...")
    imagens = {}
    for n, (pasta, split, p) in enumerate(arquivos, start=1):
        i = NOME_IMAGEM.match(p.name).group(1)
        if i in imagens:
            oc.erros.append(f"a mesma foto em mais de uma pasta: {p.name}")
            continue
        data_hora = data_hora_do_nome(i)
        if data_hora is None:
            oc.erros.append(f"nome sem data e hora válidas: {pasta}/{p.name}")
            continue
        dados = p.read_bytes()
        try:
            with Image.open(io.BytesIO(dados)) as im:
                im.load()  # decodifica tudo: é aqui que um arquivo incompleto falha
                imagens[i] = {
                    "id": i,
                    "pasta": pasta,
                    "split": split,
                    "arquivo": p,
                    "largura": im.width,
                    "altura": im.height,
                    "orientacao": im.getexif().get(0x0112),
                    "bytes": len(dados),
                    "sha256": hashlib.sha256(dados).hexdigest(),
                    "phash": str(imagehash.phash(im, hash_size=bracol.PHASH_TAMANHO)),
                    "data_hora": data_hora,
                }
        except Exception as e:  # incompleto, corrompido ou não é imagem: o Pillow usa vários tipos
            mensagem = re.sub(r"\s*<[^>]*>", "", str(e))
            oc.erros.append(f"imagem ilegível: {pasta}/{p.name} ({type(e).__name__}: {mensagem})")
        if n % 50 == 0:
            fmt.dizer(f"  ...{n}/{len(arquivos)}")
    return imagens


def validar_coco(coco: dict, tamanhos: dict, rotulo: str, oc: od.Ocorrencias):
    """Confere um arquivo COCO dos autores contra as fotos da pasta.

    tamanhos: {file_name: (largura, altura)} das fotos lidas. O que impede usar a anotação vira
    erro; o que é conhecido e inofensivo só é contado (os avisos saem somados no relatório).
    Devolve {file_name: [polígonos]} e um Counter com as contagens.
    """
    conta = Counter()
    faltando = [chave for chave in ("images", "annotations", "categories") if chave not in coco]
    if faltando:
        oc.erros.append(f"{rotulo}: faltam as chaves {faltando}")
        return {}, conta
    imagens, anotacoes = coco["images"], coco["annotations"]
    categorias = [(c.get("id"), c.get("name")) for c in coco["categories"]]
    if categorias != [CATEGORIA_FOLHA]:
        oc.erros.append(f"{rotulo}: categorias inesperadas: {categorias}")
    for campo, valores in (("id de imagem", [im["id"] for im in imagens]),
                           ("file_name", [im["file_name"] for im in imagens]),
                           ("id de anotação", [a["id"] for a in anotacoes])):
        repetidos = sorted(str(v) for v, q in Counter(valores).items() if q > 1)
        if repetidos:
            oc.erros.append(f"{rotulo}: {campo} repetido: {', '.join(repetidos)}")
    por_id = {im["id"]: im for im in imagens}
    nomes = {im["file_name"] for im in imagens}
    for nome in sorted(nomes - set(tamanhos)):
        oc.erros.append(f"{rotulo}: imagem do COCO sem arquivo: {nome}")
    for nome in sorted(set(tamanhos) - nomes):
        oc.erros.append(f"{rotulo}: arquivo sem entrada no COCO: {nome}")
    for im in imagens:
        real = tamanhos.get(im["file_name"])
        if real and real != (im["width"], im["height"]):
            oc.erros.append(f"{rotulo}: {im['file_name']} tem {im['width']}x{im['height']} no "
                            f"COCO e {real[0]}x{real[1]} no arquivo")

    poligonos, orfas = defaultdict(list), []
    for a in anotacoes:
        im = por_id.get(a["image_id"])
        if im is None:
            orfas.append(str(a["id"]))
            continue
        conta["anotacoes"] += 1
        if a.get("category_id") != CATEGORIA_FOLHA[0]:
            oc.erros.append(f"{rotulo}: anotação {a['id']} com category_id {a.get('category_id')}")
        conta["iscrowd"] += a.get("iscrowd", 0) != 0
        segmentos = a.get("segmentation")
        if not isinstance(segmentos, list) or not segmentos:
            oc.erros.append(f"{rotulo}: anotação {a['id']} sem polígono")
            continue
        conta["varios_poligonos"] += len(segmentos) > 1
        validos = []
        for seg in segmentos:
            if len(seg) % 2 or len(seg) < 6:
                oc.erros.append(f"{rotulo}: anotação {a['id']} tem um polígono com {len(seg)} "
                                "coordenadas (precisa de 3 pontos ou mais, em pares x, y)")
                continue
            xs, ys = seg[0::2], seg[1::2]
            excesso = max(0, -min(xs), -min(ys), max(xs) - im["width"], max(ys) - im["height"])
            if excesso > TOLERANCIA_BORDA:
                oc.erros.append(f"{rotulo}: anotação {a['id']} passa {excesso:g} px da borda de "
                                f"{im['file_name']}")
            elif excesso > 0:
                conta["fora_da_borda"] += 1
                conta["excesso_maximo"] = max(conta["excesso_maximo"], excesso)
            conta["primeiro_igual_ao_ultimo"] += (xs[0], ys[0]) == (xs[-1], ys[-1])
            validos.append(seg)
        poligonos[im["file_name"]].extend(validos)
        if validos and "bbox" in a:
            xs = [x for seg in validos for x in seg[0::2]]
            ys = [y for seg in validos for y in seg[1::2]]
            bx, by, bw, bh = a["bbox"]
            conta["bbox_incoerente"] += max(abs(bx - min(xs)), abs(by - min(ys)),
                                            abs(bx + bw - max(xs)), abs(by + bh - max(ys))) > 1
            conta["area_igual_ao_bbox"] += a.get("area") == bw * bh
    if orfas:
        oc.erros.append(f"{rotulo}: anotações sem imagem: ids {', '.join(orfas)}")
    sem_anotacao = sorted(nomes - set(poligonos))
    if sem_anotacao:
        oc.erros.append(f"{rotulo}: imagens sem anotação: {', '.join(sem_anotacao)}")
    return dict(poligonos), conta


def conferir_via(via: dict, regioes_coco: dict, bytes_por_arquivo: dict, rotulo: str,
                 oc: od.Ocorrencias) -> Counter:
    """O export do VIA precisa ter as mesmas fotos e o mesmo número de regiões do COCO, e o
    tamanho gravado em cada entrada tem de ser o do arquivo (a anotação é desta versão da foto).
    Conta os atributos preenchidos: nenhum quer dizer nenhuma classe por folha."""
    conta = Counter()
    por_nome = {entrada.get("filename"): entrada for entrada in via.values()}
    for nome in sorted(set(regioes_coco) - set(por_nome)):
        oc.erros.append(f"{rotulo}: foto do COCO sem entrada no VIA: {nome}")
    for nome in sorted(set(por_nome) - set(regioes_coco)):
        oc.erros.append(f"{rotulo}: entrada do VIA sem foto no COCO: {nome}")
    for nome in sorted(set(por_nome) & set(regioes_coco)):
        entrada = por_nome[nome]
        regioes = entrada.get("regions", [])
        if len(regioes) != regioes_coco[nome]:
            oc.erros.append(f"{rotulo}: {nome} tem {len(regioes)} regiões no VIA e "
                            f"{regioes_coco[nome]} anotações no COCO")
        if nome in bytes_por_arquivo and entrada.get("size") != bytes_por_arquivo[nome]:
            oc.erros.append(f"{rotulo}: {nome} tem {entrada.get('size')} bytes no VIA e "
                            f"{bytes_por_arquivo[nome]} no arquivo")
        conta["regioes"] += len(regioes)
        conta["regioes_com_atributos"] += sum(1 for r in regioes if r.get("region_attributes"))
        conta["fotos_com_atributos"] += bool(entrada.get("file_attributes"))
    return conta


def ler_anotacoes(entrada: Path, imagens: dict, oc: od.Ocorrencias) -> dict:
    """Valida o COCO e o VIA de cada split. Devolve {split: {"coco", "via", "poligonos",
    "conta_coco", "conta_via", "sha256_coco", "sha256_via"}}."""
    resultado = {}
    for pasta, (split, arquivo_coco) in SPLITS_DOS_AUTORES.items():
        das_fotos = {d["arquivo"].name: d for d in imagens.values() if d["pasta"] == pasta}
        caminho, via_caminho = entrada / pasta / arquivo_coco, entrada / pasta / ARQUIVO_VIA
        if not caminho.is_file():
            oc.erros.append(f"anotações não encontradas: {pasta}/{arquivo_coco}")
            continue
        rotulo = f"{pasta}/{arquivo_coco}"
        coco = json.loads(caminho.read_text(encoding="utf-8"))
        poligonos, conta_coco = validar_coco(
            coco, {n: (d["largura"], d["altura"]) for n, d in das_fotos.items()}, rotulo, oc)
        dados = {"coco": rotulo, "poligonos": poligonos, "conta_coco": conta_coco,
                 "sha256_coco": hashlib.sha256(caminho.read_bytes()).hexdigest(),
                 "via": None, "conta_via": Counter(), "sha256_via": ""}
        if via_caminho.is_file():
            via = json.loads(via_caminho.read_text(encoding="utf-8"))
            regioes = Counter({im["file_name"]: 0 for im in coco.get("images", [])})
            por_id = {im["id"]: im["file_name"] for im in coco.get("images", [])}
            for a in coco.get("annotations", []):
                if a["image_id"] in por_id:
                    regioes[por_id[a["image_id"]]] += 1
            dados.update(via=f"{pasta}/{ARQUIVO_VIA}",
                         sha256_via=hashlib.sha256(via_caminho.read_bytes()).hexdigest(),
                         conta_via=conferir_via(via, regioes, {n: d["bytes"]
                                                for n, d in das_fotos.items()},
                                                f"{pasta}/{ARQUIVO_VIA}", oc))
        else:
            oc.avisos.append(f"export do VIA não encontrado: {pasta}/{ARQUIVO_VIA}")
        resultado[split] = dados
    return resultado


# -------------------------------------------------------------------------- manifest
def montar_manifest(imagens, anotacoes, prefixo: str, intervalo: int) -> list[dict]:
    """Uma linha por foto, em ordem de id (o nome, que é a hora). prefixo: a pasta de entrada
    relativa à raiz do repositório."""
    cena = cenas({i: d["data_hora"] for i, d in imagens.items()}, intervalo)
    linhas = []
    for i in sorted(imagens):
        d = imagens[i]
        dados = anotacoes.get(d["split"], {})
        poligonos = dados.get("poligonos", {}).get(d["arquivo"].name, [])
        area = sum(area_do_poligono(seg) for seg in poligonos) / (d["largura"] * d["altura"])
        linhas.append({
            "fonte": FONTE,
            "id": i,
            "caminho": f"{prefixo}/{d['pasta']}/{d['arquivo'].name}",
            "largura": d["largura"],
            "altura": d["altura"],
            "split": d["split"],
            "n_folhas": len(poligonos),
            "area_coberta": f"{area:.4f}",
            "data_hora": d["data_hora"].isoformat(),
            "cena": cena[i],
            "sha256": d["sha256"],
            "phash": d["phash"],
            "anotacao": f"{prefixo}/{dados['coco']}" if dados else "",
        })
    return linhas


def conferir_sobrepostos(linhas, sobrepostos, oc: od.Ocorrencias) -> None:
    """Os pares de TESTE_SOBREPOSTO precisam existir, com a de teste em teste e a de treino em
    treino; se não, a constante está desatualizada."""
    split = {linha["id"]: linha["split"] for linha in linhas}
    for teste, treino in sobrepostos:
        if (split.get(teste), split.get(treino)) != ("teste", "treino"):
            oc.erros.append(f"TESTE_SOBREPOSTO desatualizado: {teste} ({split.get(teste)}) / "
                            f"{treino} ({split.get(treino)})")


# -------------------------------------------------------------------------- relatório
def gerar_relatorio(linhas, imagens, anotacoes, ctx, oc: od.Ocorrencias) -> str:
    """Relatório de integridade em markdown. Sem data nem caminho absoluto: rodar de novo com
    os mesmos dados produz o mesmo arquivo."""
    r = ["# Relatório de integridade do BRACOT", ""]
    r += ["Gerado por `python data/bracot.py`. Não editar à mão: rode o script de novo.", ""]
    r += ["> **Detecção e segmentação de folhas** (RF09; decisão do gestor, 09/10/2026). A "
          "divisão oficial é a dos autores (240 treino / 60 teste). Não há classe de estresse "
          "por folha, e só parte das folhas de cada foto está contornada.", ""]

    r += ["## Resultado", ""]
    if oc.erros:
        r += [f"**ERRO:** {len(oc.erros)} problema(s); o manifest **não** foi alterado.", ""]
        r += [f"- {erro}" for erro in oc.erros] + [""]
    else:
        r += ["**OK:** nenhum erro; o manifest corresponde aos dados.", ""]
    r += [f"Avisos: {len(oc.avisos) or 'nenhum'}.", ""]
    if oc.avisos:
        r += [f"- {aviso}" for aviso in oc.avisos] + [""]

    r += ["## Parâmetros", ""]
    r += fmt.tabela(["item", "valor"], [
        ["entrada", f"`{ctx['entrada']}`"],
        ["manifest", f"`{ctx['manifest']}`, uma linha por foto"],
        ["divisão", "a dos autores (train/ = treino, test/ = teste), oficial"],
        ["cena", f"fotos a até {ctx['intervalo']} s da anterior, em ordem de hora, sem olhar o "
                 "split"],
        ["borda", f"polígono que passa até {TOLERANCIA_BORDA} px da borda é aviso; acima, erro"],
        ["pHash", f"{bracol.PHASH_TAMANHO ** 2} bits, foto inteira"],
    ], alinhar="ll") + [""]

    r += _secao_arquivos(imagens, anotacoes)
    r += _secao_coco(anotacoes)
    r += _secao_folhas(linhas)
    r += _secao_cenas(linhas, ctx)
    r += _secao_vazamento(linhas, ctx)
    return "\n".join(r) + "\n"


def _secao_arquivos(imagens, anotacoes) -> list[str]:
    r = ["## Arquivos", ""]
    tabela = []
    for pasta, (split, _) in SPLITS_DOS_AUTORES.items():
        fotos = [d for d in imagens.values() if d["pasta"] == pasta]
        dados = anotacoes.get(split, {})
        tabela.append([f"`{pasta}/`", split, fmt.n(len(fotos)),
                       _decimal(sum(d["bytes"] for d in fotos) / 1e6),
                       f"`{dados['coco']}`" if dados else "-",
                       f"`{dados['sha256_coco'][:16]}…`" if dados else "-",
                       f"`{dados['sha256_via'][:16]}…`" if dados.get("sha256_via") else "-"])
    r += fmt.tabela(["pasta", "split", "fotos", "MB", "COCO", "SHA-256 do COCO",
                     "SHA-256 do VIA"], tabela, alinhar="llrrlll") + [""]
    tipos = Counter((f"{d['largura']}x{d['altura']}", d["orientacao"]) for d in imagens.values())
    r += fmt.tabela(["tamanho", "orientação EXIF", "fotos"],
                    [[t, "sem" if o is None else str(o), fmt.n(q)]
                     for (t, o), q in sorted(tipos.items(), key=lambda x: (x[0][0], str(x[0][1])))],
                    alinhar="llr") + [""]
    distintos = len({d["sha256"] for d in imagens.values()})
    r += [f"Duplicatas exatas: {fmt.n(len(imagens) - distintos)} (SHA-256 distintos: "
          f"{fmt.n(distintos)} de {fmt.n(len(imagens))}).", ""]
    return r


def _secao_coco(anotacoes) -> list[str]:
    r = ["## Validação das anotações", ""]
    itens = [
        ("anotações (folhas)", "anotacoes"),
        ("anotações com mais de um polígono", "varios_poligonos"),
        ("anotações com iscrowd diferente de 0", "iscrowd"),
        (f"polígonos que passam até {TOLERANCIA_BORDA} px da borda (aviso)", "fora_da_borda"),
        ("maior excesso na borda (px)", "excesso_maximo"),
        ("polígonos com o primeiro ponto repetido no fim (válido no COCO)",
         "primeiro_igual_ao_ultimo"),
        ("bbox incoerente com o polígono (mais de 1 px)", "bbox_incoerente"),
        ("campo area igual à área do bbox (não usar)", "area_igual_ao_bbox"),
    ]
    splits = [s for s in SPLITS if s in anotacoes]
    tabela = [[nome, *(_numero(anotacoes[s]["conta_coco"][chave]) for s in splits)]
              for nome, chave in itens]
    r += fmt.tabela(["COCO", *splits], tabela) + [""]
    r += ["Ids únicos, `file_name` batendo com as fotos, nenhuma anotação sem foto, nenhuma foto "
          "sem anotação e uma única categoria (`leaf`, id 0): tudo isso é conferido, e qualquer "
          "falha aparece nos erros. A área de cada folha sai do polígono (fórmula do laço), "
          "nunca do campo `area`, que o export do VIA preencheu com a área do bbox.", ""]
    itens_via = [("regiões", "regioes"), ("regiões com region_attributes", "regioes_com_atributos"),
                 ("fotos com file_attributes", "fotos_com_atributos")]
    tabela = [[nome, *(fmt.n(anotacoes[s]["conta_via"][chave]) for s in splits)]
              for nome, chave in itens_via]
    r += fmt.tabela(["VIA (`via_export_json.json`)", *splits], tabela) + [""]
    r += ["O VIA tem as mesmas fotos e o mesmo número de regiões do COCO, e o tamanho gravado em "
          "cada entrada é o do arquivo. Sem atributos preenchidos, não há classe por folha.", ""]
    return r


def _secao_folhas(linhas) -> list[str]:
    r = ["## Folhas por foto e área coberta", ""]
    tabela = []
    for split in SPLITS:
        do_split = [linha for linha in linhas if linha["split"] == split]
        if not do_split:
            continue
        folhas = [linha["n_folhas"] for linha in do_split]
        area = [100 * float(linha["area_coberta"]) for linha in do_split]
        tabela.append([split, fmt.n(len(do_split)), fmt.n(sum(folhas)),
                       f"{min(folhas)} / {_mediana(folhas)} / {max(folhas)}",
                       f"{_decimal(min(area))} / {_decimal(float(np.median(area)))} / "
                       f"{_decimal(max(area))}"])
    r += fmt.tabela(["split", "fotos", "folhas", "folhas por foto (mín / mediana / máx)",
                     "área coberta, % (mín / mediana / máx)"], tabela, alinhar="lrrll") + [""]
    distribuicao = Counter(linha["n_folhas"] for linha in linhas)
    r += ["Fotos por número de folhas: "
          + "; ".join(f"{k}: {fmt.n(q)}" for k, q in sorted(distribuicao.items())) + ".", ""]
    r += ["As folhas contornadas cobrem só parte de cada foto, que é quase toda folhagem: a "
          "limitação está descrita no data/README.md.", ""]
    return r


def _secao_cenas(linhas, ctx) -> list[str]:
    r = ["## Datas e cenas", ""]
    por_dia = Counter((linha["data_hora"][:10], linha["split"]) for linha in linhas)
    dias = sorted({dia for dia, _ in por_dia})
    r += fmt.tabela(["dia", *SPLITS, "total"],
                    [[dia, *(fmt.n(por_dia[dia, s]) for s in SPLITS),
                      fmt.n(sum(por_dia[dia, s] for s in SPLITS))] for dia in dias]) + [""]
    horas = sorted(datetime.fromisoformat(linha["data_hora"]) for linha in linhas)
    mesmo_dia = [(b - a).total_seconds() for a, b in zip(horas, horas[1:]) if a.date() == b.date()]
    if mesmo_dia:
        r += [f"Intervalo entre fotos seguidas no mesmo dia: mediana "
              f"{_mediana(mesmo_dia)} s, p90 {_numero(float(np.percentile(mesmo_dia, 90)))} s, "
              f"máximo {_numero(max(mesmo_dia))} s; "
              f"{fmt.n(sum(1 for g in mesmo_dia if g > ctx['intervalo']))} intervalos passam "
              f"de {ctx['intervalo']} s.", ""]
    membros = defaultdict(list)
    for linha in linhas:
        membros[linha["cena"]].append(linha)
    tamanhos = sorted(len(m) for m in membros.values())
    mistas = [c for c, m in membros.items() if len({linha["split"] for linha in m}) > 1]
    teste_em_mistas = sum(1 for c in mistas for linha in membros[c] if linha["split"] == "teste")
    total_teste = sum(1 for linha in linhas if linha["split"] == "teste")
    r += [f"Cenas (fotos a até {ctx['intervalo']} s da anterior): {fmt.n(len(membros))}, com "
          f"mediana de {_mediana(tamanhos)} fotos e no máximo {fmt.n(tamanhos[-1])}. Cenas com "
          f"fotos de treino e de teste: {fmt.n(len(mistas))}; fotos de teste nessas cenas: "
          f"{fmt.n(teste_em_mistas)} de {fmt.n(total_teste)}.", ""]
    return r


def _tempo_ate_o_treino(linhas) -> dict:
    """{id de teste: segundos até a foto de treino mais próxima}."""
    hora = {linha["id"]: datetime.fromisoformat(linha["data_hora"]) for linha in linhas}
    treino = [hora[linha["id"]] for linha in linhas if linha["split"] == "treino"]
    if not treino:
        return {}
    return {linha["id"]: min(abs((hora[linha["id"]] - t).total_seconds()) for t in treino)
            for linha in linhas if linha["split"] == "teste"}


def _secao_vazamento(linhas, ctx) -> list[str]:
    r = ["## Vazamento treino→teste", ""]
    hora = {linha["id"]: datetime.fromisoformat(linha["data_hora"]) for linha in linhas}
    split = {linha["id"]: linha["split"] for linha in linhas}
    distancia = _tempo_ate_o_treino(linhas)
    if distancia:
        valores = sorted(distancia.values())
        faixas = [10, 20, 30, 60]
        r += [f"Tempo de cada foto de teste até a foto de treino mais próxima: mediana "
              f"{_mediana(valores)} s, máximo {_numero(valores[-1])} s. Fotos de teste "
              + "; ".join(f"até {f} s: {fmt.n(sum(1 for v in valores if v <= f))}" for f in faixas)
              + f" (de {fmt.n(len(valores))}).", ""]
    r += ["Por tempo, todo o teste fica perto do treino; mas fotos seguidas costumam mostrar "
          "folhas diferentes. Por pontos casados (ORB + RANSAC, medidos em 09/10/2026), só estas "
          "fotos de teste repetem folhas de uma foto de treino (`TESTE_SOBREPOSTO`). Reporte o "
          "detector no teste inteiro e sem elas:", ""]
    phash = {linha["id"]: linha["phash"] for linha in linhas}
    cena = {linha["id"]: linha["cena"] for linha in linhas}
    tabela = []
    for teste, par in ctx["sobrepostos"]:
        if teste in hora and par in hora:
            tabela.append([teste, par, _numero(abs((hora[teste] - hora[par]).total_seconds())),
                           bracol.distancia_hamming(phash[teste], phash[par]),
                           "sim" if cena[teste] == cena[par] else "não"])
    r += fmt.tabela(["teste", "treino", "segundos", "pHash (bits)", "mesma cena"], tabela,
                    alinhar="llrrl") + [""]
    def intervalo(a, b):
        if hora[a].date() != hora[b].date():
            return "outro dia"
        return f"{_numero(abs((hora[a] - hora[b]).total_seconds()))} s"

    r += ["Os 10 pares mais próximos pelo pHash da foto inteira:", ""]
    r += fmt.tabela(["foto A", "foto B", "distância (bits)", "split A", "split B", "intervalo"],
                    [[a, b, d, split[a], split[b], intervalo(a, b)] for d, a, b in ctx["pares"]],
                    alinhar="llrllr") + [""]
    r += [f"Hash da divisão dos autores (\"id:split\" em ordem de id, unidos por \";\"): "
          f"`{hash_da_divisao(linhas)}`. A divisão por cena foi simulada e ficou só documentada "
          "no data/README.md (decisão do gestor, 09/10/2026).", ""]
    return r


def _mediana(valores) -> str:
    return _numero(float(np.median(valores)))


def _numero(valor: float) -> str:
    """Inteiro sem casas (21), ou uma casa decimal (6,5), com ponto de milhar."""
    valor = float(valor)
    return fmt.n(int(valor)) if valor.is_integer() else _decimal(valor)


def _decimal(valor: float) -> str:
    """Uma casa, com ponto de milhar e vírgula decimal: 1234.5 -> 1.234,5."""
    inteiro, fracao = f"{valor:.1f}".split(".")
    return f"{fmt.n(int(inteiro))},{fracao}"


# ----------------------------------------------------------------------------- execução
def gerar(entrada: Path = PASTA_BRACOT, manifest: Path = MANIFEST_BRACOT,
          relatorio: Path = RELATORIO_BRACOT, verificar: bool = False,
          raiz: Path = bracol.RAIZ_REPO, sobrepostos=TESTE_SOBREPOSTO,
          intervalo: int = INTERVALO_CENA) -> int:
    """Gera o manifest (ou, com verificar=True, só o confere) e grava o relatório.

    Devolve o código de saída: 0 sem erros, 1 com erros. Com erros, e sempre no modo
    verificar, o manifest não é alterado: só o relatório é gravado.
    """
    raiz, entrada = raiz.resolve(), entrada.resolve()
    manifest, relatorio = manifest.resolve(), relatorio.resolve()
    if not entrada.is_dir():
        raise od.ErroFatal(f"pasta de entrada não encontrada: {entrada}")
    if not entrada.is_relative_to(raiz):
        raise od.ErroFatal(f"a entrada precisa ficar dentro do repositório ({raiz}): {entrada}")
    anterior = od.ler_manifest_anterior(manifest, chave=str)
    if verificar and not anterior:
        raise od.ErroFatal(f"manifest não encontrado: {manifest}; rode sem --verificar para gerá-lo")
    fmt.dizer("Modo: " + ("verificação (o manifest não é alterado)" if verificar else "gerar"))
    fmt.dizer(f"Entrada: {od._exibir(entrada, raiz)}")

    oc = od.Ocorrencias()
    imagens = inspecionar_imagens(entrada, oc)
    anotacoes = ler_anotacoes(entrada, imagens, oc)
    fora = sum(dados["conta_coco"]["fora_da_borda"] for dados in anotacoes.values())
    if fora:
        excesso = max(dados["conta_coco"]["excesso_maximo"] for dados in anotacoes.values())
        oc.avisos.append(
            f"{fmt.n(fora)} polígonos passam até {_numero(excesso)} px da borda ("
            + ", ".join(f"{s} {fmt.n(anotacoes[s]['conta_coco']['fora_da_borda'])}"
                        for s in SPLITS if s in anotacoes)
            + "): quem lê recorta na borda")
    area_bbox = sum(dados["conta_coco"]["area_igual_ao_bbox"] for dados in anotacoes.values())
    if area_bbox:
        total = sum(dados["conta_coco"]["anotacoes"] for dados in anotacoes.values())
        oc.avisos.append(f"o campo area do COCO é a área do bbox em {fmt.n(area_bbox)} de "
                         f"{fmt.n(total)} anotações: não usar; a área do manifest sai do polígono")
    if any(dados["conta_via"]["regioes_com_atributos"] or dados["conta_via"]["fotos_com_atributos"]
           for dados in anotacoes.values()):
        oc.avisos.append("o export do VIA tem atributos preenchidos, que este pipeline ignora")

    linhas = montar_manifest(imagens, anotacoes, od._exibir(entrada, raiz), intervalo)
    conferir_sobrepostos(linhas, sobrepostos, oc)
    if verificar:
        od.conferir_com_versionado(linhas, anterior, oc, colunas_esperadas=COLUNAS_MANIFEST,
                                   colunas_hash=("phash",))
    ctx = {
        "entrada": od._exibir(entrada, raiz),
        "manifest": od._exibir(manifest, raiz),
        "intervalo": intervalo,
        "sobrepostos": sobrepostos,
        "pares": bracol.pares_mais_proximos(
            [{"id": linha["id"], "phash": linha["phash"]} for linha in linhas], chave="phash"),
    }
    relatorio.parent.mkdir(parents=True, exist_ok=True)
    relatorio.write_text(gerar_relatorio(linhas, imagens, anotacoes, ctx, oc), encoding="utf-8",
                         newline="\n")
    if not oc.erros and not verificar:
        od.escrever_manifest(linhas, manifest, colunas=COLUNAS_MANIFEST)

    por_split = Counter(linha["split"] for linha in linhas)
    folhas = Counter()
    for linha in linhas:
        folhas[linha["split"]] += linha["n_folhas"]
    distancia = _tempo_ate_o_treino(linhas)
    fmt.dizer("")
    fmt.dizer(f"Imagens: {len(linhas)} (treino {por_split['treino']}, teste {por_split['teste']}) "
              f"| folhas: {sum(folhas.values())} (treino {folhas['treino']}, teste "
              f"{folhas['teste']})")
    fmt.dizer(f"Cenas: {len({linha['cena'] for linha in linhas})} (intervalo {intervalo} s) | "
              f"teste ate a foto de treino mais proxima: no maximo "
              f"{_numero(max(distancia.values())) if distancia else '-'} s | "
              f"sobrepostas ao treino: {len(sobrepostos)}")
    fmt.dizer(f"Hash da divisao dos autores: {hash_da_divisao(linhas)}")
    fmt.dizer(f"Erros: {len(oc.erros)} | avisos: {len(oc.avisos)}")
    for erro in oc.erros[:10]:
        fmt.dizer(f"  ERRO: {erro}")
    if len(oc.erros) > 10:
        fmt.dizer(f"  ... e mais {len(oc.erros) - 10} (ver o relatório)")
    for aviso in oc.avisos[:10]:
        fmt.dizer(f"  aviso: {aviso}")
    fmt.dizer(f"Relatório: {od._exibir(relatorio, raiz)}")
    if oc.erros and verificar:
        fmt.dizer("Manifest NÃO confere com os dados: veja os erros acima e no relatório.")
    elif oc.erros:
        fmt.dizer("Manifest NÃO gravado: corrija os erros e rode de novo.")
    elif verificar:
        fmt.dizer(f"Manifest confere com os dados: {od._exibir(manifest, raiz)}")
    else:
        fmt.dizer(f"Manifest: {od._exibir(manifest, raiz)} ({len(linhas)} linhas)")
    return 1 if oc.erros else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gera o manifest do BRACOT e o relatorio de integridade (ver data/README.md)."
    )
    ap.add_argument("--entrada", type=Path, default=PASTA_BRACOT,
                    help="pasta com train/ e test/ (padrao: data/raw/bracot/bracot-data)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_BRACOT,
                    help="arquivo do manifest (padrao: data/manifests/bracot.csv)")
    ap.add_argument("--relatorio", type=Path, default=RELATORIO_BRACOT,
                    help="relatorio de integridade (padrao: data/reports/integridade_bracot.md)")
    ap.add_argument("--verificar", action="store_true",
                    help="so confere dados x manifest: grava o relatorio, nunca o manifest")
    args = ap.parse_args(argv)
    try:
        return gerar(args.entrada, args.manifest, args.relatorio, verificar=args.verificar)
    except od.ErroFatal as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
