"""
JMuBEN e JMuBEN2: contrato, manifest (data/manifests/jmuben.csv) e relatório de integridade
(data/reports/integridade_jmuben.md).

Fonte auxiliar (decisão do gestor, 09/10/2026): só treino de classificação, nunca validação nem
teste; por isso o manifest não tem coluna de split. São recortes de folhas fotografadas em campo,
com augmentation dos autores (giros, espelhos, brilho e cópias): 58.550 arquivos em
data/raw/jmuben/<Pasta>/, mas só 3.757 conteúdos distintos.

O manifest tem uma linha por conteúdo distinto (SHA-256), com o número de cópias idênticas e o
caminho da primeira. Cópias e variantes do mesmo recorte ficam no mesmo grupo; a seleção pega no
máximo um representante por grupo e CAP_POR_CLASSE imagens por classe.

Uso (de qualquer pasta):
    python data/jmuben.py               gera o manifest e o relatório
    python data/jmuben.py --verificar   confere dados x manifest, sem alterá-lo

Leitura (Frente 9): ler_manifest(teto=None) devolve só as imagens selecionadas.
O código de saída é 0 sem erros e 1 com erros; com erros, o manifest não é gravado.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import hashlib
import io
import numbers
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

import bracol
import formatacao as fmt
import organize_dataset as od

# ------------------------------------------------------------------------ caminhos
PASTA_JMUBEN = bracol.RAIZ_REPO / "data" / "raw" / "jmuben"
MANIFEST_JMUBEN = bracol.RAIZ_REPO / "data" / "manifests" / "jmuben.csv"
RELATORIO_JMUBEN = bracol.PASTA_REPORTS / "integridade_jmuben.md"

# ------------------------------------------------------------------------- classes
FONTE = "jmuben"
# Pasta original -> (classe do projeto, subconjunto publicado). "Cerscospora" é a grafia da
# origem. JMuBEN (doi:10.17632/t2r6rszp5c.1): Cerscospora, Leaf rust e Phoma. JMuBEN2
# (doi:10.17632/tgv3zb82nd.1): Healthy e Miner. A correspondência de Phoma e Cerscospora com as
# classes do projeto tem a ressalva de bracol.RESSALVA_PHOMA_CERCOSPORA.
PASTAS = {
    "Cerscospora": ("cercosporiose", "jmuben"),
    "Healthy": ("saudavel", "jmuben2"),
    "Leaf rust": ("ferrugem", "jmuben"),
    "Miner": ("bicho_mineiro", "jmuben2"),
    "Phoma": ("phoma", "jmuben"),
}

# O Pillow não reconhece este arquivo (começa como JPEG, mas não termina com o marcador de
# fim), e ele fica excluído, com motivo. Tem o mesmo tamanho de "Healthy/2 (691).bak.jpg", que
# abre e tem 293 cópias idênticas: nada se perde. Qualquer outro arquivo ilegível vira erro.
ARQUIVOS_ILEGIVEIS_ESPERADOS = frozenset({"Healthy/2 (691).jpg"})

# ---------------------------------------------------------- hash canônico e grupos
# Hash canônico: o menor pHash de 256 bits (bracol.PHASH_TAMANHO) entre as 8 transformações de
# giro e espelhamento do cinza equalizado, calculado nos pixels gravados, sem aplicar a
# orientação EXIF (que varia entre as cópias). Calibração (09/10/2026), em bits: variantes do
# mesmo recorte (giradas, espelhadas, deslocadas ou com outra cor), conferidas visualmente, a
# 2, 10, 22, 32, 36, 46, 48, 54, 58, 62, 68, 72, 78, 84 e 88; recortes diferentes a 94; o par
# entre classes mais próximo a 86. Sem equalizar, recortes desbotados quase iguais ficam a
# 120-136 bits. Com 64 bits saem 984 grupos, nenhum com duas classes (margem de 22 bits para o
# par entre classes). Errar aqui só muda a diversidade do treino auxiliar; nunca vaza para
# validação ou teste, que são só do BRACOL.
LIMIAR_HASH_CANONICO = 64

# ------------------------------------------------------------------------- seleção
# Teto por classe (decisão do gestor, 09/10/2026): metade do treino do BRACOL na classe (190,
# 372, 271, 244 e 103 imagens). Um representante por grupo, sem repetir: os membros de um grupo
# são giros, espelhos e mudanças de brilho do mesmo recorte, que o augmentation já produz.
CAP_POR_CLASSE = {
    "saudavel": 95,
    "ferrugem": 186,
    "bicho_mineiro": 135,
    "phoma": 122,
    "cercosporiose": 51,
}
SEED = bracol.SEED_PADRAO

USO_TREINO = "treino_auxiliar"
USO_EXCLUIDA = "excluida"
MOTIVO_GRUPO_REPRESENTADO = "grupo_ja_representado"
MOTIVO_ACIMA_DO_TETO = "acima_do_limite_da_classe"
MOTIVO_ILEGIVEL = "arquivo_ilegivel"

# ------------------------------------------------------------------------ manifest
COLUNAS_MANIFEST = [
    "fonte", "subconjunto", "id", "caminho", "copias", "classe", "largura", "altura",
    "sha256", "phash_canonico", "grupo", "uso", "selecionada", "motivo",
]

NOME_PADRAO = re.compile(r"^\d+ \(\d+\)\.jpg$")  # "N (k).jpg", a forma da maioria
NOME_COM_N = re.compile(r"^(\d+) \(\d+\)\.jpe?g$")
_TRANSFORMACOES = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)


# ------------------------------------------------------------------ hash e seleção
def hash_canonico(im) -> str:
    """pHash de 256 bits que não muda com giro nem espelhamento: o menor entre as 8
    transformações do cinza equalizado. A equalização tira o efeito de brilho e contraste."""
    cinza = ImageOps.equalize(im.convert("RGB").convert("L"))
    return min(
        str(imagehash.phash(cinza if t is None else cinza.transpose(t),
                            hash_size=bracol.PHASH_TAMANHO))
        for t in _TRANSFORMACOES
    )


def selecionar(linhas, teto=None, seed: int = SEED) -> set[int]:
    """Ids escolhidos para o treino auxiliar.

    linhas: dicts com "id", "classe", "grupo" e "uso"; só entram as de uso treino_auxiliar.
    teto: {classe: máximo}; None usa CAP_POR_CLASSE, e classe fora do dicionário fica sem
        nenhuma.
    Em cada classe, os grupos seguem a ordem de um sorteio determinístico (sha256 de
    "seed:grupo") até o teto da classe, e de cada grupo sai um representante só, o de menor
    sha256 de "seed:id". O resultado não depende da ordem da entrada nem das outras classes.
    """
    teto = CAP_POR_CLASSE if teto is None else teto
    grupos = defaultdict(lambda: defaultdict(list))
    for linha in linhas:
        if linha["uso"] == USO_TREINO:
            grupos[linha["classe"]][linha["grupo"]].append(linha["id"])
    escolhidos = set()
    for classe, da_classe in grupos.items():
        ordem = sorted(da_classe, key=lambda grupo: bracol._sorteio(seed, grupo))
        for grupo in ordem[:teto.get(classe, 0)]:
            escolhidos.add(min(da_classe[grupo], key=lambda i: bracol._sorteio(seed, i)))
    return escolhidos


def hash_da_selecao(ids) -> str:
    """SHA-256 dos ids selecionados em ordem crescente, unidos por ";" (fixado nos testes)."""
    return hashlib.sha256(";".join(str(i) for i in sorted(ids)).encode("utf-8")).hexdigest()


# -------------------------------------------------------------------------- leitura
def ler_manifest(caminho=MANIFEST_JMUBEN, teto=None):
    """Imagens do JMuBEN selecionadas para o treino auxiliar, como pandas.DataFrame.

    Só treino: não há parâmetro de split, porque o JMuBEN nunca entra em validação nem teste.
    teto: None (todas as selecionadas), um inteiro (o mesmo máximo para todas as classes) ou
        {classe: máximo} (classes fora do dicionário ficam com a seleção inteira). O corte
        segue a ordem do sorteio da seleção, então um teto menor dá sempre um subconjunto do
        maior. O teto só reduz: para passar de CAP_POR_CLASSE é preciso regerar o manifest.
    O rótulo para o modelo é bracol.CLASSES.index(classe), como no BRACOL.
    """
    import pandas as pd  # só quem lê o manifest precisa do pandas

    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    if list(df.columns) != COLUNAS_MANIFEST:
        raise ValueError(f"colunas inesperadas em {caminho}: {list(df.columns)}")
    for coluna in ["id", "copias", "selecionada"]:
        df[coluna] = df[coluna].astype("int64")
    for coluna in ["largura", "altura"]:  # vazias no arquivo ilegível
        df[coluna] = pd.to_numeric(df[coluna].mask(df[coluna] == "")).astype("Int64")
    df = df[(df["uso"] == USO_TREINO) & (df["selecionada"] == 1)]
    if teto is not None:
        limites = _limites(teto)
        ordem = df["grupo"].map(lambda grupo: bracol._sorteio(SEED, grupo))
        df = df.assign(_ordem=ordem).sort_values(["classe", "_ordem"])
        posicao = df.groupby("classe").cumcount()
        limite = df["classe"].map(lambda classe: limites.get(classe, len(df)))
        df = df[posicao < limite].drop(columns="_ordem")
    return df.sort_values("id").reset_index(drop=True)


def _limites(teto) -> dict[str, int]:
    """Teto do leitor -> {classe: máximo}; levanta ValueError se for inválido."""
    def inteiro(valor):
        return isinstance(valor, numbers.Integral) and not isinstance(valor, bool)

    if inteiro(teto):
        limites = dict.fromkeys(bracol.CLASSES, int(teto))
    elif isinstance(teto, dict):
        limites = dict(teto)
    else:
        raise ValueError(f"teto precisa ser inteiro, dicionário {{classe: máximo}} ou None: {teto!r}")
    desconhecidas = sorted(set(limites) - set(bracol.CLASSES))
    if desconhecidas:
        raise ValueError(f"classes desconhecidas no teto: {desconhecidas}")
    invalidos = {c: v for c, v in limites.items() if not inteiro(v) or v < 0}
    if invalidos:
        raise ValueError(f"o teto de cada classe precisa ser um inteiro >= 0: {invalidos}")
    return {c: int(v) for c, v in limites.items()}


# ---------------------------------------------------------------- arquivos e imagens
def inventariar(entrada: Path, oc: od.Ocorrencias) -> dict[str, dict]:
    """Lê todos os arquivos das pastas de classe e junta os idênticos (mesmo SHA-256).

    Devolve {sha256: {"sha256", "pasta", "arquivos" (["<Pasta>/<nome>", ...] em ordem),
    "bytes"}}.
    """
    existentes = sorted(p.name for p in entrada.iterdir() if p.is_dir())
    for nome in sorted(set(existentes) - set(PASTAS)):
        oc.avisos.append(f"pasta ignorada (fora das pastas de classe): {nome}")
    for nome in sorted(set(PASTAS) - set(existentes)):
        oc.erros.append(f"pasta de classe não encontrada: {nome}")
    soltos = sorted(p.name for p in entrada.iterdir() if p.is_file())
    if soltos:
        oc.avisos.append(f"arquivos fora das pastas de classe, ignorados: {', '.join(soltos)}")
    arquivos = []
    for pasta in sorted(set(PASTAS) & set(existentes)):
        for p in sorted((entrada / pasta).iterdir()):
            if p.is_file():
                arquivos.append((f"{pasta}/{p.name}", p))
            else:
                oc.avisos.append(f"subpasta ignorada: {pasta}/{p.name}")
    arquivos.sort()
    fmt.dizer(f"Lendo {len(arquivos)} arquivos (SHA-256)...")
    conteudos = {}
    for n, (relativo, p) in enumerate(arquivos, start=1):
        dados = p.read_bytes()
        sha = hashlib.sha256(dados).hexdigest()
        conteudo = conteudos.setdefault(sha, {
            "sha256": sha, "pasta": relativo.split("/")[0], "arquivos": [], "bytes": len(dados),
        })
        conteudo["arquivos"].append(relativo)
        if n % 10000 == 0:
            fmt.dizer(f"  ...{n}/{len(arquivos)}")
    for conteudo in conteudos.values():
        if len({a.split("/")[0] for a in conteudo["arquivos"]}) > 1:
            oc.erros.append("o mesmo arquivo em pastas de classe diferentes: "
                            + ", ".join(conteudo["arquivos"][:4]))
    return conteudos


def inspecionar(conteudos, entrada: Path, ilegiveis_esperados, oc: od.Ocorrencias) -> None:
    """Decodifica por completo a primeira cópia de cada conteúdo e acrescenta ao dicionário dele
    tamanho, formato, orientação EXIF, hash dos pixels e hash canônico."""
    fmt.dizer(f"Abrindo {len(conteudos)} conteudos distintos (decodificacao completa e hash "
              "canonico)...")
    esperados_vistos = set()
    ordem = sorted(conteudos.values(), key=lambda c: c["arquivos"][0])
    for n, conteudo in enumerate(ordem, start=1):
        primeiro = conteudo["arquivos"][0]
        try:
            with Image.open(io.BytesIO((entrada / primeiro).read_bytes())) as im:
                im.load()  # decodifica tudo: é aqui que um arquivo incompleto falha
                rgb = im.convert("RGB")
                conteudo.update(
                    largura=im.width,
                    altura=im.height,
                    formato=im.format,
                    modo=im.mode,
                    orientacao=im.getexif().get(0x0112),
                    pixels=hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest(),
                    phash_canonico=hash_canonico(rgb),
                )
        except Exception as e:  # incompleto, corrompido ou não é imagem: o Pillow usa vários tipos
            # Sem o objeto (com endereço de memória) que o Pillow põe na mensagem: o relatório
            # não muda de uma execução para outra.
            conteudo["ilegivel"] = f"{type(e).__name__}: " + re.sub(r"\s*<[^>]*>", "", str(e))
            esperados = [a for a in conteudo["arquivos"] if a in ilegiveis_esperados]
            if esperados:
                esperados_vistos.update(esperados)
            else:
                oc.erros.append(f"arquivo ilegível: {primeiro} ({conteudo['ilegivel']})")
        if n % 500 == 0:
            fmt.dizer(f"  ...{n}/{len(conteudos)}")
    nao_vistos = sorted(set(ilegiveis_esperados) - esperados_vistos)
    if nao_vistos:
        oc.avisos.append("listados como ilegíveis esperados, mas abriram ou não existem: "
                         + ", ".join(nao_vistos))


# ------------------------------------------------------------------------ manifest
def montar_manifest(conteudos, prefixo: str, teto, oc: od.Ocorrencias) -> list[dict]:
    """Uma linha por conteúdo distinto, em ordem de caminho (ids 1, 2, ...), com grupo, uso e
    seleção. prefixo: a pasta de entrada relativa à raiz do repositório."""
    linhas = []
    ordem = sorted(conteudos.values(), key=lambda c: c["arquivos"][0])
    for i, conteudo in enumerate(ordem, start=1):
        classe, subconjunto = PASTAS[conteudo["pasta"]]
        legivel = "ilegivel" not in conteudo
        linhas.append({
            "fonte": FONTE,
            "subconjunto": subconjunto,
            "id": i,
            "caminho": f"{prefixo}/{conteudo['arquivos'][0]}",
            "copias": len(conteudo["arquivos"]),
            "classe": classe,
            "largura": conteudo.get("largura", ""),
            "altura": conteudo.get("altura", ""),
            "sha256": conteudo["sha256"],
            "phash_canonico": conteudo.get("phash_canonico", ""),
            "grupo": "",
            "uso": USO_TREINO if legivel else USO_EXCLUIDA,
            "selecionada": 0,
            "motivo": "" if legivel else MOTIVO_ILEGIVEL,
        })

    legiveis = [linha for linha in linhas if linha["uso"] == USO_TREINO]
    grupos = bracol.agrupar_por_hash(
        [{"id": linha["id"], "sha256": linha["sha256"], "phash": linha["phash_canonico"]}
         for linha in legiveis],
        limiar=LIMIAR_HASH_CANONICO, pares_manuais=(), fonte=FONTE,
    )
    classes_do_grupo = defaultdict(set)
    for linha in legiveis:
        linha["grupo"] = grupos[linha["id"]]
        classes_do_grupo[linha["grupo"]].add(linha["classe"])
    mistos = sorted((g for g, c in classes_do_grupo.items() if len(c) > 1), key=_numero_do_grupo)
    if mistos:
        oc.erros.append("grupos com mais de uma classe (o mesmo recorte com rótulos diferentes): "
                        + ", ".join(mistos))

    escolhidos = selecionar(legiveis, teto)
    representados = {linha["grupo"] for linha in legiveis if linha["id"] in escolhidos}
    for linha in legiveis:
        if linha["id"] in escolhidos:
            linha["selecionada"] = 1
        elif linha["grupo"] in representados:
            linha["motivo"] = MOTIVO_GRUPO_REPRESENTADO
        else:
            linha["motivo"] = MOTIVO_ACIMA_DO_TETO
    return linhas


def _numero_do_grupo(grupo: str) -> int:
    return int(grupo.rsplit("-", 1)[1])


def par_entre_classes_mais_proximo(linhas):
    """(distância, id_a, id_b) do par de classes diferentes com hashes canônicos mais
    próximos, ou None. Confere a margem do limiar no relatório."""
    itens = sorted((linha for linha in linhas if linha["phash_canonico"]), key=lambda l: l["id"])
    classes = np.array([linha["classe"] for linha in itens])
    if len(set(classes)) < 2:
        return None
    matriz = bracol._matriz_de_bits([linha["phash_canonico"] for linha in itens])
    melhor = None
    for inicio, dist in bracol._blocos_de_distancia(matriz):
        bloco = classes[inicio:inicio + len(dist)]
        dist = np.where(bloco[:, None] != classes[None, :], dist, np.iinfo(dist.dtype).max)
        r, j = np.unravel_index(np.argmin(dist), dist.shape)
        d = int(dist[r, j])
        if melhor is None or d < melhor[0]:
            a, b = sorted((itens[inicio + r]["id"], itens[j]["id"]))
            melhor = (d, a, b)
    return melhor


# -------------------------------------------------------------------------- relatório
def gerar_relatorio(linhas, conteudos, ctx, oc: od.Ocorrencias) -> str:
    """Relatório de integridade em markdown. Sem data nem caminho absoluto: rodar de novo com
    os mesmos dados produz o mesmo arquivo."""
    r = ["# Relatório de integridade do JMuBEN", ""]
    r += ["Gerado por `python data/jmuben.py`. Não editar à mão: rode o script de novo.", ""]
    r += ["> **Fonte auxiliar** (decisão do gestor, 09/10/2026): só treino de classificação, "
          "nunca validação nem teste. São recortes de folhas fotografadas em campo, com "
          "augmentation dos autores; os rótulos não foram verificados.", ">",
          f"> **Ressalva:** {bracol.RESSALVA_PHOMA_CERCOSPORA}", ""]

    r += ["## Resultado", ""]
    if oc.erros:
        r += [f"**ERRO:** {len(oc.erros)} problema(s); o manifest **não** foi alterado.", ""]
        r += [f"- {erro}" for erro in oc.erros] + [""]
    else:
        r += ["**OK:** nenhum erro; o manifest corresponde aos dados.", ""]
    r += [f"Avisos: {len(oc.avisos) or 'nenhum'}.", ""]
    if oc.avisos:
        r += [f"- {aviso}" for aviso in oc.avisos] + [""]

    teto = ctx["teto"]
    r += ["## Parâmetros", ""]
    r += fmt.tabela(["item", "valor"], [
        ["entrada", f"`{ctx['entrada']}`"],
        ["manifest", f"`{ctx['manifest']}`, uma linha por conteúdo distinto (SHA-256)"],
        ["hash canônico", f"pHash de {bracol.PHASH_TAMANHO ** 2} bits: o menor entre as 8 "
                          "transformações de giro e espelhamento do cinza equalizado, sem "
                          "aplicar a orientação EXIF"],
        ["grupo", f"SHA-256 igual ou hash canônico a até {LIMIAR_HASH_CANONICO} bits, de forma "
                  "transitiva"],
        ["teto por classe", ", ".join(f"{c} {teto.get(c, 0)}" for c in bracol.CLASSES)],
        ["seleção", "um representante por grupo; grupos e representantes por sorteio "
                    f"determinístico (seed {SEED})"],
        ["ilegíveis esperados",
         ", ".join(f"`{a}`" for a in sorted(ctx["ilegiveis_esperados"])) or "nenhum"],
    ], alinhar="ll") + [""]

    r += _secao_arquivos(conteudos)
    r += _secao_imagens(conteudos)
    r += _secao_nomes(linhas, conteudos)
    r += _secao_grupos(linhas, conteudos, ctx)
    r += _secao_selecao(linhas, teto)
    return "\n".join(r) + "\n"


def _da_pasta(conteudos, pasta):
    return [c for c in conteudos.values() if c["pasta"] == pasta]


def _secao_arquivos(conteudos) -> list[str]:
    r = ["## Arquivos por pasta", ""]
    tabela, total = [], Counter()
    for pasta, (classe, subconjunto) in PASTAS.items():
        cs = _da_pasta(conteudos, pasta)
        arquivos = [a for c in cs for a in c["arquivos"]]
        extensoes = Counter(Path(a).suffix.lower() for a in arquivos)
        mb = sum(c["bytes"] * len(c["arquivos"]) for c in cs) / 1e6
        ilegiveis = sum(1 for c in cs if "ilegivel" in c)
        total.update(arquivos=len(arquivos), mb=mb, conteudos=len(cs), ilegiveis=ilegiveis)
        tabela.append([f"`{pasta}`", classe, subconjunto, fmt.n(len(arquivos)),
                       ", ".join(f"{e} {fmt.n(q)}" for e, q in sorted(extensoes.items())),
                       _decimal(mb), fmt.n(len(cs)), fmt.n(ilegiveis)])
    tabela.append(["**total**", "", "", fmt.n(total["arquivos"]), "", _decimal(total["mb"]),
                   fmt.n(total["conteudos"]), fmt.n(total["ilegiveis"])])
    r += fmt.tabela(["pasta", "classe", "subconjunto", "arquivos", "extensões", "MB",
                     "conteúdos distintos", "ilegíveis"], tabela, alinhar="lllrlrrr") + [""]
    por_sub = Counter()
    for c in conteudos.values():
        por_sub[PASTAS[c["pasta"]][1]] += len(c["arquivos"])
    r += ["Arquivos por subconjunto: "
          + "; ".join(f"{s} {fmt.n(q)}" for s, q in sorted(por_sub.items())) + ".", ""]
    ilegiveis = sorted((c for c in conteudos.values() if "ilegivel" in c),
                       key=lambda c: c["arquivos"][0])
    if ilegiveis:
        r += ["Arquivos ilegíveis (excluídos com motivo `arquivo_ilegivel`):", ""]
        r += [f"- `{c['arquivos'][0]}`: {c['ilegivel']}" for c in ilegiveis] + [""]
    return r


def _secao_imagens(conteudos) -> list[str]:
    legiveis = [c for c in conteudos.values() if "ilegivel" not in c]
    r = ["## Imagens", ""]
    r += ["Contagens em arquivos (cada cópia conta). Tamanho e orientação como gravados."]
    r += [""]
    tabela = []
    for pasta in PASTAS:
        tamanhos = Counter()
        for c in legiveis:
            if c["pasta"] == pasta:
                tamanhos[f"{c['largura']}x{c['altura']}"] += len(c["arquivos"])
        quadrados = tamanhos.pop("128x128", 0)
        comuns = sorted(tamanhos.items(), key=lambda x: (-x[1], x[0]))[:3]
        tabela.append([f"`{pasta}`", fmt.n(quadrados), fmt.n(sum(tamanhos.values())),
                       ", ".join(f"{t} ({fmt.n(q)})" for t, q in comuns) or "-"])
    r += fmt.tabela(["pasta", "128x128", "outros tamanhos", "mais comuns fora de 128x128"],
                    tabela, alinhar="lrrl") + [""]
    tipos = Counter()
    for c in legiveis:
        tipos[c["formato"], c["modo"]] += len(c["arquivos"])
    r += fmt.tabela(["formato", "modo", "arquivos"],
                    [[f, m, fmt.n(q)] for (f, m), q in sorted(tipos.items())], alinhar="llr")
    r += [""]
    orientacoes = sorted({c["orientacao"] for c in legiveis}, key=lambda o: (o is not None, o))
    tabela = []
    for pasta in PASTAS:
        contagem = Counter()
        for c in legiveis:
            if c["pasta"] == pasta:
                contagem[c["orientacao"]] += len(c["arquivos"])
        tabela.append([f"`{pasta}`", *(fmt.n(contagem[o]) for o in orientacoes)])
    r += ["Orientação EXIF (tag 274). Varia entre as cópias: parte da \"rotação\" do augmentation "
          "dos autores está só na etiqueta. O Pillow ignora a etiqueta; o `cv2.imread` a aplica. "
          "O hash canônico não depende dela.", ""]
    r += fmt.tabela(["pasta", *("sem" if o is None else str(o) for o in orientacoes)], tabela)
    r += [""]
    pixels = len({c["pixels"] for c in legiveis})
    r += [f"Conteúdos legíveis: {fmt.n(len(legiveis))}; matrizes de pixels distintas: "
          f"{fmt.n(pixels)} (o resto difere só nos metadados, como a orientação EXIF).", ""]
    return r


def _secao_nomes(linhas, conteudos) -> list[str]:
    r = ["## Nomes dos arquivos", ""]
    tabela = []
    for pasta in PASTAS:
        nomes = [a.split("/", 1)[1] for c in _da_pasta(conteudos, pasta) for a in c["arquivos"]]
        no_padrao = sum(1 for nome in nomes if NOME_PADRAO.match(nome))
        valores_n = Counter(int(m.group(1)) for m in map(NOME_COM_N.match, nomes) if m)
        tabela.append([f"`{pasta}`", fmt.n(no_padrao), fmt.n(len(nomes) - no_padrao),
                       "; ".join(f"{v}: {fmt.n(q)}" for v, q in sorted(valores_n.items()))])
    r += fmt.tabela(["pasta", "no padrão `N (k).jpg`", "fora do padrão", "arquivos por N"],
                    tabela, alinhar="lrrl") + [""]

    por_sha = {c["sha256"]: c for c in conteudos.values()}
    arquivos_do_grupo = defaultdict(list)
    for linha in linhas:
        if linha["grupo"]:
            arquivos_do_grupo[linha["grupo"]].extend(por_sha[linha["sha256"]]["arquivos"])
    cruzam, so_fora = Counter(), []
    for grupo, arquivos in arquivos_do_grupo.items():
        nomes = [a.split("/", 1)[1] for a in arquivos]
        valores_n = {m.group(1) for m in map(NOME_COM_N.match, nomes) if m}
        if len(valores_n) > 1:
            cruzam[arquivos[0].split("/")[0]] += 1
        if not any(NOME_PADRAO.match(nome) for nome in nomes):
            so_fora.append(grupo)
    grupos_da_pasta = Counter(arquivos[0].split("/")[0] for arquivos in arquivos_do_grupo.values())
    r += ["O prefixo N não identifica a foto de origem: cópias e variantes do mesmo recorte "
          "aparecem com N diferentes. Ele não é usado como grupo. Grupos com arquivos de N "
          "diferentes: "
          + "; ".join(f"{pasta} {fmt.n(cruzam[pasta])} de {fmt.n(grupos_da_pasta[pasta])}"
                      for pasta in PASTAS) + ".", ""]

    formas = Counter()
    for c in conteudos.values():
        for a in c["arquivos"]:
            pasta, nome = a.split("/", 1)
            if not NOME_PADRAO.match(nome):
                formas[pasta, _forma_do_nome(nome)] += 1
    if formas:
        ordem = sorted(formas.items(), key=lambda x: (x[0][0], -x[1], x[0][1]))
        r += ["Nomes fora do padrão (dígitos trocados por `#`), incluídos como os outros:", ""]
        r += fmt.tabela(["pasta", "forma do nome", "arquivos"],
                        [[f"`{p}`", f"`{f}`", fmt.n(q)] for (p, f), q in ordem], alinhar="llr")
        r += [""]
    r += [f"Grupos só com nomes fora do padrão (recortes que não aparecem como `N (k).jpg`): "
          f"{fmt.n(len(so_fora))}.", ""]
    if so_fora:
        selecionados = {linha["grupo"] for linha in linhas if linha["selecionada"]}
        tabela = []
        for grupo in sorted(so_fora, key=_numero_do_grupo):
            arquivos = sorted(arquivos_do_grupo[grupo])
            extensoes = Counter(Path(a).suffix.lower() for a in arquivos)
            tabela.append([f"`{grupo}`", f"`{arquivos[0].split('/')[0]}`", fmt.n(len(arquivos)),
                           ", ".join(f"{e} {fmt.n(q)}" for e, q in sorted(extensoes.items())),
                           f"`{arquivos[0]}`", "sim" if grupo in selecionados else "não"])
        r += fmt.tabela(["grupo", "pasta", "arquivos", "extensões", "primeiro arquivo",
                         "selecionado"], tabela, alinhar="llrlll") + [""]
    return r


def _forma_do_nome(nome: str) -> str:
    """'059eaf863cca.jpeg' -> '<hex>.jpeg'; 'IMG_20200101_1234.jpg' -> 'IMG_#_#.jpg'."""
    base, _, extensao = nome.partition(".")
    if re.fullmatch(r"[0-9a-f]{8,}", base):
        return f"<hex>.{extensao}"
    return re.sub(r"\d+", "#", nome)


def _secao_grupos(linhas, conteudos, ctx) -> list[str]:
    por_sha = {c["sha256"]: c for c in conteudos.values()}
    legiveis = [linha for linha in linhas if linha["uso"] == USO_TREINO]
    por_id = {linha["id"]: linha for linha in linhas}
    r = ["## Duplicatas e grupos", ""]
    copias = Counter(linha["copias"] for linha in linhas)
    comuns = sorted(copias.items(), key=lambda x: (-x[1], x[0]))[:8]
    r += [f"Duplicatas exatas: {fmt.n(sum(q for c, q in copias.items() if c > 1))} conteúdos têm "
          "mais de uma cópia, cobrindo "
          f"{fmt.n(sum(c * q for c, q in copias.items() if c > 1))} arquivos. Cópias por "
          "conteúdo, as mais comuns: "
          + "; ".join(f"{c} {'cópias' if c > 1 else 'cópia'} em {fmt.n(q)} conteúdos"
                      for c, q in comuns)
          + f"; no máximo {fmt.n(max(copias))}.", ""]
    r += [f"Um conteúdo entra no grupo de outro, de forma transitiva, se o SHA-256 for igual ou se "
          f"o hash canônico ficar a até {LIMIAR_HASH_CANONICO} bits.", ""]

    membros = defaultdict(list)
    for linha in legiveis:
        membros[linha["grupo"]].append(linha)
    tabela = []
    for classe in bracol.CLASSES:
        grupos = [m for m in membros.values() if m[0]["classe"] == classe]
        arquivos = sorted(sum(l["copias"] for l in m) for m in grupos)
        tabela.append([classe, fmt.n(len(grupos)), fmt.n(sum(len(m) for m in grupos)),
                       fmt.n(sum(arquivos)), fmt.n(arquivos[-1]) if arquivos else "-",
                       _mediana(arquivos)])
    r += fmt.tabela(["classe", "grupos", "conteúdos", "arquivos", "maior grupo (arquivos)",
                     "mediana (arquivos por grupo)"], tabela) + [""]
    sozinhos = sum(1 for m in membros.values() if sum(l["copias"] for l in m) == 1)
    r += [f"Grupos com um arquivo só: {fmt.n(sozinhos)}.", ""]

    maiores = sorted(membros.items(),
                     key=lambda x: (-sum(l["copias"] for l in x[1]), _numero_do_grupo(x[0])))[:5]
    r += ["Os 5 maiores grupos:", ""]
    r += fmt.tabela(["grupo", "classe", "conteúdos", "arquivos"],
                    [[f"`{g}`", m[0]["classe"], fmt.n(len(m)), fmt.n(sum(l["copias"] for l in m))]
                     for g, m in maiores], alinhar="llrr") + [""]

    def arquivo(i):
        return f"`{por_sha[por_id[i]['sha256']]['arquivos'][0]}`"

    r += ["Os 10 pares mais próximos NÃO agrupados, pelo hash canônico:", ""]
    r += fmt.tabela(["id A", "id B", "distância (bits)", "classe A", "classe B", "arquivo A",
                     "arquivo B"],
                    [[a, b, d, por_id[a]["classe"], por_id[b]["classe"], arquivo(a), arquivo(b)]
                     for d, a, b in ctx["pares"]], alinhar="rrrllll") + [""]
    if ctx["entre_classes"]:
        d, a, b = ctx["entre_classes"]
        r += [f"Par de classes diferentes mais próximo: {a} ({por_id[a]['classe']}, {arquivo(a)}) "
              f"e {b} ({por_id[b]['classe']}, {arquivo(b)}), a {d} bits; o limiar é "
              f"{LIMIAR_HASH_CANONICO}.", ""]
    return r


def _mediana(valores) -> str:
    if not valores:
        return "-"
    mediana = float(np.median(valores))
    return fmt.n(int(mediana)) if mediana.is_integer() else _decimal(mediana)


def _decimal(valor: float) -> str:
    """Uma casa, com ponto de milhar e vírgula decimal: 1839.1 -> 1.839,1."""
    inteiro, fracao = f"{valor:.1f}".split(".")
    return f"{fmt.n(int(inteiro))},{fracao}"


def _secao_selecao(linhas, teto) -> list[str]:
    legiveis = [linha for linha in linhas if linha["uso"] == USO_TREINO]
    r = ["## Seleção para o treino auxiliar", ""]
    tabela = []
    for classe in bracol.CLASSES:
        da_classe = [linha for linha in legiveis if linha["classe"] == classe]
        selecionadas = sum(linha["selecionada"] for linha in da_classe)
        tabela.append([classe, fmt.n(len({l["grupo"] for l in da_classe})),
                       fmt.n(teto.get(classe, 0)), fmt.n(selecionadas)])
    total = sum(linha["selecionada"] for linha in legiveis)
    tabela.append(["**total**", fmt.n(len({l["grupo"] for l in legiveis})), "", fmt.n(total)])
    r += fmt.tabela(["classe", "grupos", "teto", "selecionadas"], tabela) + [""]
    escolhidos = [linha["id"] for linha in linhas if linha["selecionada"]]
    r += [f"Hash da seleção (ids selecionados em ordem, unidos por \";\"): "
          f"`{hash_da_selecao(escolhidos)}`.", ""]
    motivos = Counter((linha["uso"], linha["motivo"]) for linha in linhas)
    copias = Counter()
    for linha in linhas:
        copias[linha["uso"], linha["motivo"]] += linha["copias"]
    r += fmt.tabela(["uso", "motivo", "conteúdos", "arquivos"],
                    [[u, f"`{m}`" if m else "(selecionada)", fmt.n(q), fmt.n(copias[u, m])]
                     for (u, m), q in sorted(motivos.items())], alinhar="llrr") + [""]
    return r


# ----------------------------------------------------------------------------- execução
def gerar(entrada: Path = PASTA_JMUBEN, manifest: Path = MANIFEST_JMUBEN,
          relatorio: Path = RELATORIO_JMUBEN, verificar: bool = False,
          raiz: Path = bracol.RAIZ_REPO, ilegiveis_esperados=ARQUIVOS_ILEGIVEIS_ESPERADOS,
          teto=None) -> int:
    """Gera o manifest (ou, com verificar=True, só o confere) e grava o relatório.

    teto: {classe: máximo}; None usa CAP_POR_CLASSE.
    Devolve o código de saída: 0 sem erros, 1 com erros. Com erros, e sempre no modo
    verificar, o manifest não é alterado: só o relatório é gravado.
    """
    teto = dict(CAP_POR_CLASSE if teto is None else teto)
    raiz, entrada = raiz.resolve(), entrada.resolve()
    manifest, relatorio = manifest.resolve(), relatorio.resolve()
    if not entrada.is_dir():
        raise od.ErroFatal(f"pasta de entrada não encontrada: {entrada}")
    if not entrada.is_relative_to(raiz):
        raise od.ErroFatal(f"a entrada precisa ficar dentro do repositório ({raiz}): {entrada}")
    anterior = od.ler_manifest_anterior(manifest)
    if verificar and not anterior:
        raise od.ErroFatal(f"manifest não encontrado: {manifest}; rode sem --verificar para gerá-lo")
    fmt.dizer("Modo: " + ("verificação (o manifest não é alterado)" if verificar else "gerar"))
    fmt.dizer(f"Entrada: {od._exibir(entrada, raiz)}")

    oc = od.Ocorrencias()
    conteudos = inventariar(entrada, oc)
    inspecionar(conteudos, entrada, ilegiveis_esperados, oc)
    linhas = montar_manifest(conteudos, od._exibir(entrada, raiz), teto, oc)
    escolhidos = {linha["id"] for linha in linhas if linha["selecionada"]}
    if verificar:
        od.conferir_com_versionado(linhas, anterior, oc, colunas_esperadas=COLUNAS_MANIFEST,
                                   colunas_hash=("phash_canonico",))
    elif anterior:
        antes = {i for i, linha in anterior.items() if linha.get("selecionada") == "1"}
        if antes != escolhidos:
            oc.avisos.append(f"seleção diferente do manifest anterior: {len(escolhidos - antes)} "
                             f"entram e {len(antes - escolhidos)} saem")
    legiveis = [{"id": l["id"], "phash": l["phash_canonico"]} for l in linhas if l["grupo"]]
    ctx = {
        "entrada": od._exibir(entrada, raiz),
        "manifest": od._exibir(manifest, raiz),
        "teto": teto,
        "ilegiveis_esperados": set(ilegiveis_esperados),
        "pares": bracol.pares_mais_proximos(
            legiveis, chave="phash", grupos={l["id"]: l["grupo"] for l in linhas if l["grupo"]}),
        "entre_classes": par_entre_classes_mais_proximo(linhas),
    }
    relatorio.parent.mkdir(parents=True, exist_ok=True)
    relatorio.write_text(gerar_relatorio(linhas, conteudos, ctx, oc), encoding="utf-8",
                         newline="\n")
    if not oc.erros and not verificar:
        od.escrever_manifest(linhas, manifest, colunas=COLUNAS_MANIFEST)

    grupos = defaultdict(set)
    for linha in linhas:
        if linha["grupo"]:
            grupos[linha["classe"]].add(linha["grupo"])
    selecionadas = Counter(linha["classe"] for linha in linhas if linha["selecionada"])
    ilegiveis = [linha for linha in linhas if linha["uso"] == USO_EXCLUIDA]
    fmt.dizer("")
    fmt.dizer(f"Arquivos: {sum(linha['copias'] for linha in linhas)} | conteudos distintos: "
              f"{len(linhas)} | ilegiveis: {len(ilegiveis)}")
    fmt.dizer(f"Grupos: {sum(len(g) for g in grupos.values())} ("
              + ", ".join(f"{c} {len(grupos[c])}" for c in bracol.CLASSES) + ")")
    fmt.dizer(f"Selecionadas: {len(escolhidos)} ("
              + ", ".join(f"{c} {selecionadas[c]}" for c in bracol.CLASSES) + ")")
    fmt.dizer(f"Hash da selecao: {hash_da_selecao(escolhidos)}")
    fmt.dizer(f"Erros: {len(oc.erros)} | avisos: {len(oc.avisos)}")
    for erro in oc.erros[:10]:
        fmt.dizer(f"  ERRO: {erro}")
    if len(oc.erros) > 10:
        fmt.dizer(f"  ... e mais {len(oc.erros) - 10} (ver o relatório)")
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
        description="Gera o manifest do JMuBEN e o relatorio de integridade (ver data/README.md)."
    )
    ap.add_argument("--entrada", type=Path, default=PASTA_JMUBEN,
                    help="pasta com as 5 pastas de classe (padrao: data/raw/jmuben)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_JMUBEN,
                    help="arquivo do manifest (padrao: data/manifests/jmuben.csv)")
    ap.add_argument("--relatorio", type=Path, default=RELATORIO_JMUBEN,
                    help="relatorio de integridade (padrao: data/reports/integridade_jmuben.md)")
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
