"""
Gera o manifest do BRACOL (data/manifests/bracol.csv) e o relatório de integridade
(data/reports/integridade_bracol.md).

O manifest é a fonte única da verdade dos dados: uma linha por id do dataset.csv original,
com caminho da imagem, rótulos originais, classe do projeto, tamanho, hashes, grupo de
duplicatas, split e exclusão. As imagens não são copiadas: quem treina lê direto de data/raw
pelo manifest. Isso substitui as pastas data/processed/{train,val,test}/{classe}/ previstas
no documento da Frente 8.

Uso (de qualquer pasta):
    python data/organize_dataset.py                     gera o manifest e o relatório
    python data/organize_dataset.py --verificar         confere dados x manifest, sem alterá-lo
    python data/organize_dataset.py --entrada PASTA     usa outra cópia do BRACOL
    python data/organize_dataset.py --refazer-divisao   ignora a divisão do manifest anterior

Ao rodar de novo, a divisão já gravada no manifest é mantida e só as imagens novas são
distribuídas (divisão estável). Com qualquer erro, o manifest não é gravado, só o relatório.
O código de saída é 0 sem erros e 1 com erros.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import csv
import hashlib
import io
import re
import sys
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

import bracol
import formatacao as fmt

RELATORIO_PADRAO = bracol.PASTA_REPORTS / "integridade_bracol.md"
NOME_IMAGEM = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)


class ErroFatal(Exception):
    """Problema que impede continuar: entrada não encontrada ou csv com colunas erradas."""


class Ocorrencias:
    """Erros impedem gravar o manifest; avisos só vão para o relatório."""

    def __init__(self):
        self.erros = []
        self.avisos = []


# ---------------------------------------------------------------------------- leitura
def localizar_entrada(entrada: Path) -> tuple[Path, Path]:
    """Acha, dentro de `entrada`, exatamente um dataset.csv com a pasta images/ ao lado."""
    if not entrada.is_dir():
        raise ErroFatal(f"pasta de entrada não encontrada: {entrada}")
    achados = sorted(p for p in entrada.rglob("dataset.csv") if (p.parent / "images").is_dir())
    if len(achados) != 1:
        lista = "".join(f"\n  {p}" for p in achados) or " nenhum"
        raise ErroFatal(
            f"esperado exatamente 1 dataset.csv com images/ ao lado em {entrada}; achados:{lista}\n"
            "Use --entrada para apontar a cópia certa."
        )
    return achados[0], achados[0].parent / "images"


def ler_csv(caminho: Path, oc: Ocorrencias) -> tuple[list[dict], set[int]]:
    """Lê e valida o dataset.csv original.

    Devolve as linhas válidas (valores inteiros) e os ids das linhas com erro.
    """
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        leitor = csv.DictReader(f)
        if leitor.fieldnames != bracol.COLUNAS_CSV:
            raise ErroFatal(
                f"colunas do dataset.csv: {leitor.fieldnames}; esperado: {bracol.COLUNAS_CSV}"
            )
        brutas = list(leitor)
    linhas, vistos, invalidos = [], set(), set()
    for numero, bruta in enumerate(brutas, start=2):  # a linha 1 é o cabeçalho
        try:
            linha = {coluna: int(bruta[coluna]) for coluna in bracol.COLUNAS_CSV}
        except (TypeError, ValueError):
            oc.erros.append(f"dataset.csv, linha {numero}: valor que não é inteiro: {dict(bruta)}")
            if str(bruta.get("id") or "").strip().isdigit():
                invalidos.add(int(bruta["id"]))
            continue
        problema = _problema_da_linha(linha, vistos)
        if problema:
            oc.erros.append(f"dataset.csv, linha {numero} (id {linha['id']}): {problema}")
            invalidos.add(linha["id"])
            continue
        vistos.add(linha["id"])
        linhas.append(linha)
    return linhas, invalidos


def _problema_da_linha(linha: dict, vistos: set) -> str | None:
    """Descrição do problema de uma linha do csv, ou None se ela estiver coerente."""
    ps, severidade = linha["predominant_stress"], linha["severity"]
    marcadas = {coluna: linha[coluna] for coluna in bracol.PS_PARA_COLUNA.values()}
    if linha["id"] in vistos:
        return "id repetido"
    if linha["id"] < 1:
        return "id precisa ser positivo"
    if not 0 <= ps <= 5:
        return f"predominant_stress fora de 0 a 5: {ps}"
    if not 0 <= severidade <= 4:
        return f"severity fora de 0 a 4: {severidade}"
    if any(valor not in (0, 1) for valor in marcadas.values()):
        return f"coluna binária fora de 0/1: {marcadas}"
    if ps == 0 and (any(marcadas.values()) or severidade):
        return "saudável (0) com estresse marcado ou severidade maior que 0"
    if ps in bracol.PS_PARA_COLUNA and not linha[bracol.PS_PARA_COLUNA[ps]]:
        return f"predominant_stress {ps} sem a coluna {bracol.PS_PARA_COLUNA[ps]} marcada"
    return None


def inspecionar_imagens(pasta: Path, oc: Ocorrencias) -> tuple[dict[int, dict], set[int]]:
    """Abre e decodifica cada <id>.jpg por completo e calcula tamanho, SHA-256 e pHash.

    Devolve {id: dados da imagem} e os ids das imagens ilegíveis.
    """
    arquivos = {}
    for p in sorted(pasta.iterdir()):
        nome = NOME_IMAGEM.match(p.name)
        if not (p.is_file() and nome):
            oc.avisos.append(f"ignorado em images/: {p.name} (fora do padrão <id>.jpg)")
        elif int(nome.group(1)) in arquivos:
            oc.erros.append(f"mais de um arquivo para o id {int(nome.group(1))}: {p.name}")
        else:
            arquivos[int(nome.group(1))] = p
    fmt.dizer(f"Abrindo {len(arquivos)} imagens (decodificação completa, SHA-256 e dois pHash)...")
    imagens, ilegiveis = {}, set()
    for n, (i, p) in enumerate(sorted(arquivos.items()), start=1):
        dados = p.read_bytes()
        try:
            with Image.open(io.BytesIO(dados)) as im:
                im.load()  # decodifica tudo: é aqui que um arquivo truncado falha
                imagens[i] = {
                    "arquivo": p,
                    "largura": im.width,
                    "altura": im.height,
                    "sha256": hashlib.sha256(dados).hexdigest(),
                    "phash": str(imagehash.phash(im, hash_size=bracol.PHASH_TAMANHO)),
                    "phash_folha": phash_da_folha(im),
                    "formato": im.format,
                    "modo": im.mode,
                }
        except Exception as e:  # truncado, corrompido ou não é imagem: o Pillow usa vários tipos
            oc.erros.append(f"imagem ilegível: {p.name} ({type(e).__name__}: {e})")
            ilegiveis.add(i)
        if n % 250 == 0:
            fmt.dizer(f"  ...{n}/{len(arquivos)}")
    return imagens, ilegiveis


def phash_da_folha(im) -> str:
    """pHash de 256 bits do recorte da folha (caixa envolvente do que não é fundo).

    O fundo do BRACOL é claro e pouco saturado. Numa cópia reduzida a 1/4, o limiar de Otsu
    sobre a saturação (HSV) separa a folha; a caixa usa os quantis de 0,2% e 99,8% dos pixels
    da folha, para ignorar sujeirinhas. A região da imagem original vai para 512x256. Sem pixel
    acima do limiar (imagem lisa), usa a imagem inteira.
    """
    original = im.convert("RGB")
    reduzida = original.reduce(4)
    saturacao = np.asarray(reduzida.convert("HSV"))[:, :, 1]
    ys, xs = np.nonzero(saturacao > _otsu(saturacao))
    caixa = None
    if len(xs):
        k = original.width / reduzida.width
        x0, x1 = np.quantile(xs, [0.002, 0.998])
        y0, y1 = np.quantile(ys, [0.002, 0.998])
        caixa = (int(x0 * k), int(y0 * k), int(np.ceil((x1 + 1) * k)), int(np.ceil((y1 + 1) * k)))
    folha = original.crop(caixa) if caixa else original
    folha = folha.resize((512, 256), Image.Resampling.LANCZOS)
    return str(imagehash.phash(folha, hash_size=bracol.PHASH_TAMANHO))


def _otsu(valores) -> int:
    """Limiar de Otsu (0 a 255) de uma matriz uint8: maximiza a variância entre as classes."""
    histograma = np.bincount(valores.ravel(), minlength=256).astype(float)
    niveis = np.arange(256)
    peso_fundo = np.cumsum(histograma)
    peso_frente = histograma.sum() - peso_fundo
    soma_fundo = np.cumsum(niveis * histograma)
    media_fundo = soma_fundo / np.maximum(peso_fundo, 1)
    media_frente = (soma_fundo[-1] - soma_fundo) / np.maximum(peso_frente, 1)
    return int(np.argmax(peso_fundo * peso_frente * (media_fundo - media_frente) ** 2))


def ler_manifest_anterior(caminho: Path) -> dict[int, dict]:
    """Linhas do manifest já gravado, por id ({} se ele ainda não existe)."""
    if not caminho.is_file():
        return {}
    with open(caminho, newline="", encoding="utf-8") as f:
        return {int(linha["id"]): linha for linha in csv.DictReader(f)}


# --------------------------------------------------------------------------- manifest
def montar_manifest(linhas, imagens, raiz, seed, fixos, ausentes_esperados, com_erro, oc):
    """Linhas do manifest (uma por id do csv, em ordem de id), com grupos e divisão.

    fixos: {id: split} de uma divisão anterior, que é mantida (divisão estável).
    Devolve as linhas e {"mantidas": n, "novas": n}, as atribuições de split mantidas e as
    feitas agora.
    """
    ids_csv = {linha["id"] for linha in linhas}
    for i in sorted(set(imagens) - ids_csv - com_erro):
        oc.erros.append(f"imagem sem linha no dataset.csv: {imagens[i]['arquivo'].name}")
    inesperados = sorted(ids_csv - set(imagens) - com_erro - set(ausentes_esperados))
    if inesperados:
        oc.erros.append(f"imagens ausentes fora da lista esperada: ids {fmt.faixas(inesperados)}")

    presentes = [{"id": i, **imagens[i]} for i in sorted(ids_csv & set(imagens))]
    grupos = bracol.agrupar_por_hash(presentes)
    manifest = []
    for linha in sorted(linhas, key=lambda linha: linha["id"]):
        i, ps = linha["id"], linha["predominant_stress"]
        imagem = imagens.get(i)
        motivos = bracol.motivos_exclusao(ps, imagem is not None)
        manifest.append({
            "fonte": "bracol",
            "id": i,
            "caminho": imagem["arquivo"].relative_to(raiz).as_posix() if imagem else "",
            "presente": int(imagem is not None),
            "classe": bracol.classe_do_projeto(ps) or "",
            **{coluna: linha[coluna] for coluna in bracol.COLUNAS_CSV[1:]},
            **{c: imagem[c] if imagem else ""
               for c in ("largura", "altura", "sha256", "phash", "phash_folha")},
            "grupo": grupos.get(i, ""),
            "split": "",
            "excluida": int(bool(motivos)),
            "motivo_exclusao": ";".join(motivos),
        })

    elegiveis = [linha for linha in manifest if not linha["excluida"]]
    try:
        divisao = bracol.dividir(elegiveis, seed=seed, fixos=fixos)
    except ValueError as e:
        oc.erros.append(f"divisão: {e}")
        divisao = {}
    for linha in manifest:
        linha["split"] = divisao.get(linha["id"], "")

    por_grupo = {}
    for linha in manifest:
        if linha["grupo"]:
            por_grupo.setdefault(linha["grupo"], set()).add(linha["predominant_stress"])
    mistos = sorted(g for g, codigos in por_grupo.items() if len(codigos) > 1)
    if mistos:
        oc.avisos.append(f"grupos de duplicatas com rótulos diferentes: {', '.join(mistos)}")

    mantidas = sum(1 for i, split in divisao.items() if fixos.get(i) == split)
    return manifest, {"mantidas": mantidas, "novas": len(divisao) - mantidas}


def conferir_com_versionado(manifest, versionado, oc, colunas_esperadas=bracol.COLUNAS_MANIFEST,
                            colunas_hash=("phash", "phash_folha")):
    """Modo --verificar: o manifest recalculado precisa bater com o versionado. O SHA-256 e as
    outras colunas são comparados exatamente; as colunas de pHash (`colunas_hash`), com
    tolerância de TOLERANCIA_PHASH bits (outro decodificador JPEG pode mudar alguns bits).

    Os padrões são os do BRACOL; o jmuben.py passa as colunas do manifest dele."""
    colunas = list(next(iter(versionado.values())))
    if colunas != colunas_esperadas:
        oc.erros.append(
            f"colunas do manifest versionado: {colunas}; esperado: {colunas_esperadas}"
        )
        return
    atuais = {linha["id"]: linha for linha in manifest}
    so_no_versionado = sorted(set(versionado) - set(atuais))
    so_nos_dados = sorted(set(atuais) - set(versionado))
    if so_no_versionado:
        ids = fmt.faixas(so_no_versionado)
        oc.erros.append(f"ids no manifest versionado que não estão nos dados: {ids}")
    if so_nos_dados:
        ids = fmt.faixas(so_nos_dados)
        oc.erros.append(f"ids nos dados que faltam no manifest versionado: {ids}")
    divergentes = {}
    for i in sorted(set(atuais) & set(versionado)):
        for coluna in colunas_esperadas:
            atual, gravado = str(atuais[i][coluna]), versionado[i][coluna]
            if coluna in colunas_hash and atual and gravado:
                confere = bracol.distancia_hamming(atual, gravado) <= bracol.TOLERANCIA_PHASH
            else:
                confere = atual == gravado
            if not confere:
                divergentes.setdefault(coluna, []).append(i)
    for coluna in colunas_esperadas:
        if coluna in divergentes:
            ids = fmt.faixas(divergentes[coluna])
            oc.erros.append(f"coluna {coluna} diverge do manifest versionado: ids {ids}")


def _comparar_com_anterior(manifest, anterior, fixos, oc):
    """Modo gerar: avisos do que mudou em relação ao manifest anterior (rótulos, imagens,
    splits descartados)."""
    rotulo_mudou, imagem_mudou = [], []
    for linha in manifest:
        antes = anterior.get(linha["id"])
        if antes is None:
            continue
        if any(str(linha[c]) != antes.get(c) for c in bracol.COLUNAS_CSV[1:]):
            rotulo_mudou.append(linha["id"])
        if linha["sha256"] and antes.get("sha256") and linha["sha256"] != antes["sha256"]:
            imagem_mudou.append(linha["id"])
    if rotulo_mudou:
        oc.avisos.append(f"rótulos diferentes do manifest anterior: ids {fmt.faixas(rotulo_mudou)}")
    if imagem_mudou:
        ids = fmt.faixas(imagem_mudou)
        oc.avisos.append(f"imagens diferentes do manifest anterior (SHA-256 mudou): ids {ids}")
    com_split = {linha["id"] for linha in manifest if linha["split"]}
    descartados = sorted(set(fixos) - com_split)
    if descartados:
        oc.avisos.append(
            "atribuições de split descartadas (deixaram de ser elegíveis): "
            f"ids {fmt.faixas(descartados)}"
        )


def escrever_manifest(linhas: list[dict], caminho: Path, colunas=bracol.COLUNAS_MANIFEST) -> None:
    """Grava o manifest em UTF-8 sem BOM e com fim de linha LF (mesmos bytes em qualquer SO).

    As colunas padrão são as do BRACOL; o jmuben.py passa as do manifest dele."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=colunas, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(linhas)


# -------------------------------------------------------------------------- relatório
def gerar_relatorio(linhas: list[dict], ctx: dict, oc: Ocorrencias) -> str:
    """Relatório de integridade em markdown. Sem data nem caminho absoluto: rodar de novo com
    os mesmos dados produz o mesmo arquivo."""
    presentes = [linha for linha in linhas if linha["presente"]]
    ausentes = [linha["id"] for linha in linhas if not linha["presente"]]
    elegiveis = [linha for linha in linhas if not linha["excluida"]]
    inesperados = [i for i in ausentes if i not in ctx["ausentes_esperados"]]
    r = ["# Relatório de integridade do BRACOL", ""]
    r += ["Gerado por `python data/organize_dataset.py`. Não editar à mão: rode o script de novo.",
          ""]
    if ausentes:
        r += [
            f"> **Cópia PARCIAL do BRACOL:** {fmt.n(len(presentes))} de {fmt.n(len(linhas))} "
            "imagens. Resultados com ela não são comparáveis com Esgario et al. (2020).",
            ">",
        ]
    else:
        r += [f"> **Cópia completa do BRACOL:** {fmt.n(len(presentes))} imagens.", ">"]
    r += [f"> **Ressalva:** {bracol.RESSALVA_PHOMA_CERCOSPORA}", ""]

    r += ["## Resultado", ""]
    if oc.erros:
        r += [f"**ERRO:** {len(oc.erros)} problema(s); o manifest **não** foi alterado.", ""]
        r += [f"- {erro}" for erro in oc.erros] + [""]
    else:
        r += ["**OK:** nenhum erro; o manifest corresponde aos dados.", ""]
    r += [f"Avisos: {len(oc.avisos) or 'nenhum'}.", ""]
    if oc.avisos:
        r += [f"- {aviso}" for aviso in oc.avisos] + [""]

    if ctx["refazer_divisao"]:
        divisao = "refeita do zero (`--refazer-divisao`)"
    else:
        divisao = "estável: imagens que já tinham split no manifest anterior ficam onde estavam"
    r += ["## Parâmetros", ""]
    r += fmt.tabela(["item", "valor"], [
        ["entrada", f"`{ctx['entrada']}`"],
        ["dataset.csv", f"{fmt.n(len(linhas))} linhas, SHA-256 `{ctx['csv_sha256']}`"],
        ["manifest", f"`{ctx['manifest']}`"],
        ["proporções", ", ".join(f"{s} {p}%" for s, p in bracol.PROPORCOES.items())],
        ["seed", ctx["seed"]],
        ["divisão", divisao],
        ["pHash do quadro", f"{bracol.PHASH_TAMANHO ** 2} bits; agrupa a até "
                            f"{bracol.LIMIAR_QUASE_DUPLICATA} bits"],
        ["pHash da folha", f"{bracol.PHASH_TAMANHO ** 2} bits, recorte da folha em 512x256; "
                           f"agrupa a até {bracol.LIMIAR_QUASE_DUPLICATA_FOLHA} bits"],
        ["folhas repetidas", f"{len(bracol.PARES_MESMA_FOLHA)} pares conferidos visualmente em "
                             "07/10/2026 (`PARES_MESMA_FOLHA`), sempre agrupados"],
    ], alinhar="ll") + [""]

    r += ["## CSV x arquivos", ""]
    r += fmt.tabela(["", "quantidade"], [
        ["linhas no dataset.csv", fmt.n(len(linhas))],
        ["imagens `<id>.jpg` lidas", fmt.n(len(ctx["imagens"]))],
        ["imagens sem linha no csv", fmt.n(len(ctx["imagens"]) - len(presentes))],
        ["ids sem imagem", fmt.n(len(ausentes))],
        ["ids sem imagem, esperados", fmt.n(len(ausentes) - len(inesperados))],
        ["ids sem imagem, inesperados", fmt.n(len(inesperados))],
    ]) + [""]
    if ausentes:
        r += [f"Ids sem imagem: {fmt.faixas(ausentes)}.", ""]
    tabela = []
    for classe in [*bracol.CLASSES, ""]:
        da_classe = [linha for linha in linhas if linha["classe"] == classe]
        sem = sum(1 for linha in da_classe if not linha["presente"])
        tabela.append([classe or "(classe 5)", fmt.n(len(da_classe)), fmt.n(len(da_classe) - sem),
                       fmt.n(sem), fmt.pct(sem, len(da_classe))])
    tabela.append(["**total**", fmt.n(len(linhas)), fmt.n(len(presentes)), fmt.n(len(ausentes)),
                   fmt.pct(len(ausentes), len(linhas))])
    r += fmt.tabela(["classe", "no csv", "com imagem", "sem imagem", "perda"], tabela) + [""]

    r += ["## Imagens", ""]
    tipos = Counter(
        (d["formato"], d["modo"], f"{d['largura']}x{d['altura']}") for d in ctx["imagens"].values()
    )
    linhas_tipos = [[*tipo, fmt.n(q)] for tipo, q in sorted(tipos.items())]
    r += fmt.tabela(["formato", "modo", "resolução", "imagens"], linhas_tipos, alinhar="lllr")
    r += [""]
    r += ["As imagens desta tabela abriram e decodificaram por completo; as ilegíveis aparecem "
          "nos erros.", ""]

    r += _secao_duplicatas(linhas, ctx)

    r += ["## Exclusões", ""]
    motivos = Counter(linha["motivo_exclusao"] for linha in linhas if linha["excluida"])
    r += fmt.tabela(["motivo", "linhas"], [
        *([f"`{m}`", fmt.n(q)] for m, q in sorted(motivos.items())),
        ["**total excluídas**", fmt.n(sum(motivos.values()))],
        ["**elegíveis**", fmt.n(len(elegiveis))],
    ]) + [""]

    classe5 = [linha for linha in linhas if linha["predominant_stress"] == bracol.PS_INDETERMINADO]
    com_imagem = [linha for linha in classe5 if linha["presente"]]
    sem_imagem = [str(linha["id"]) for linha in classe5 if not linha["presente"]]
    r += ["## Classe 5 (predominant_stress = 5)", ""]
    r += ["Indeterminada (`5 - undetermined` no leaf/legend.txt dos autores): fica fora de "
          "classificação, severidade e multirrótulo, como os autores também fizeram no dataset.csv "
          "dos experimentos deles. Não atribuir classe por palpite.", ""]
    r += [f"Linhas: {fmt.n(len(classe5))}, {fmt.n(len(com_imagem))} com imagem "
          f"(sem imagem: {', '.join(sem_imagem) or 'nenhuma'}).", ""]
    combinacoes = Counter(
        "+".join(c for c in bracol.PS_PARA_COLUNA.values() if linha[c]) or "(nenhum)"
        for linha in com_imagem
    )
    ordem = sorted(combinacoes.items(), key=lambda x: (-x[1], x[0]))
    r += fmt.tabela(["estresses marcados (com imagem)", "imagens"],
                    [[c, fmt.n(q)] for c, q in ordem])
    r += [""]

    r += ["## Divisão", ""]
    por_classe = Counter((linha["classe"], linha["split"]) for linha in elegiveis)
    tabela = []
    for classe in bracol.CLASSES:
        valores = [por_classe[classe, s] for s in bracol.SPLITS]
        tabela.append([classe, *map(fmt.n, valores), fmt.n(sum(valores))])
    totais = Counter(linha["split"] for linha in elegiveis)
    tabela.append(["**total**", *(fmt.n(totais[s]) for s in bracol.SPLITS), fmt.n(len(elegiveis))])
    r += fmt.tabela(["classe", *bracol.SPLITS, "total"], tabela) + [""]
    por_severidade = Counter((linha["severity"], linha["split"]) for linha in elegiveis)
    tabela = []
    for nivel, descricao in bracol.SEVERIDADES.items():
        valores = [por_severidade[nivel, s] for s in bracol.SPLITS]
        tabela.append([f"{nivel}: {descricao}", *map(fmt.n, valores),
                       fmt.pct(por_severidade[nivel, "teste"], sum(valores))])
    r += fmt.tabela(["severidade", *bracol.SPLITS, "% no teste"], tabela) + [""]
    r += ["Histórico da divisão:", ""]
    r += [f"- {data}: {texto}" for data, texto in bracol.HISTORICO_DIVISAO]
    return "\n".join(r) + "\n"


def _secao_duplicatas(linhas, ctx) -> list[str]:
    """Seção do relatório com os grupos de duplicatas e folhas repetidas e os pares a conferir."""
    por_id = {linha["id"]: linha for linha in linhas}
    presentes = [linha for linha in linhas if linha["presente"]]
    exatas = sum(1 for q in Counter(linha["sha256"] for linha in presentes).values() if q > 1)
    por_grupo = Counter(linha["grupo"] for linha in presentes)
    grupos = sorted((g for g, q in por_grupo.items() if q > 1), key=lambda g: int(g.split("-")[1]))
    r = ["## Duplicatas e folhas repetidas", ""]
    r += [f"Uma imagem entra no grupo de outra (de forma transitiva) se o SHA-256 for igual, se o "
          f"pHash do quadro ficar a até {bracol.LIMIAR_QUASE_DUPLICATA} bits, se o pHash da folha "
          f"ficar a até {bracol.LIMIAR_QUASE_DUPLICATA_FOLHA} bits, ou se o par estiver na "
          "lista de folhas repetidas conferidas visualmente em 07/10/2026 (`PARES_MESMA_FOLHA`). "
          "Um grupo "
          "nunca se divide entre splits.", ""]
    r += [f"- Duplicatas exatas (mesmo SHA-256): {fmt.n(exatas)} grupo(s).",
          f"- Grupos com mais de uma imagem: {fmt.n(len(grupos))}.", ""]
    if grupos:
        linhas_grupos = []
        for g in grupos:
            membros = sorted(linha["id"] for linha in presentes if linha["grupo"] == g)
            for k, a in enumerate(membros):
                for b in membros[k + 1:]:
                    linhas_grupos.append([f"`{g}`", *_descrever_par(por_id[a], por_id[b])])
        r += fmt.tabela(["grupo", "ids", "classe e severidade", "split", "quadro (bits)",
                         "folha (bits)", "critério"], linhas_grupos, alinhar="llllrrl") + [""]
    nome = {linha["id"]: linha["classe"] or "(classe 5)" for linha in linhas}
    for titulo, pares in (("pHash do quadro", ctx["pares_quadro"]),
                          ("pHash da folha", ctx["pares_folha"])):
        r += [f"Os 10 pares mais próximos NÃO agrupados, pelo {titulo}:", ""]
        r += fmt.tabela(["id A", "id B", "distância (bits)", "classe A", "classe B"],
                        [[a, b, d, nome[a], nome[b]] for d, a, b in pares], alinhar="rrrll") + [""]
    conferir = [(a, b) for a, b in bracol.PARES_PARA_CONFERIR
                if por_id.get(a, {}).get("presente") and por_id.get(b, {}).get("presente")]
    if conferir:
        r += ["Pares para conferir (folhas escuras parecidas), conferidos visualmente em "
              "07/10/2026, em cópias reduzidas: pareceram folhas diferentes e ficam fora dos "
              "grupos; continuam listados no relatório para nova conferência nos originais.", ""]
        r += fmt.tabela(["ids", "classe e severidade", "split", "quadro (bits)", "folha (bits)",
                         "critério"],
                        [_descrever_par(por_id[a], por_id[b]) for a, b in conferir],
                        alinhar="lllrrl") + [""]
    return r


def _descrever_par(a: dict, b: dict) -> list:
    """Ids, rótulos, splits, distâncias nos dois hashes e critério de agrupamento de um par."""
    def rotulo(linha):
        return f"{linha['classe'] or '(classe 5)'}, severidade {linha['severity']}"

    def split(linha):
        return linha["split"] or "(excluída)"

    quadro = bracol.distancia_hamming(a["phash"], b["phash"])
    folha = bracol.distancia_hamming(a["phash_folha"], b["phash_folha"])
    criterios = [nome for nome, vale in (
        ("SHA-256", a["sha256"] == b["sha256"]),
        ("quadro", quadro <= bracol.LIMIAR_QUASE_DUPLICATA),
        ("folha", folha <= bracol.LIMIAR_QUASE_DUPLICATA_FOLHA),
        ("lista", (a["id"], b["id"]) in bracol.PARES_MESMA_FOLHA),
    ) if vale]
    rotulos = rotulo(a) if rotulo(a) == rotulo(b) else f"{rotulo(a)} / {rotulo(b)}"
    splits = split(a) if split(a) == split(b) else f"{split(a)} / {split(b)}"
    agrupado = " + ".join(criterios) or ("transitivo" if a["grupo"] == b["grupo"] else "nenhum")
    return [f"{a['id']} / {b['id']}", rotulos, splits, quadro, folha, agrupado]


# ----------------------------------------------------------------------------- execução
def gerar(entrada: Path, manifest: Path, relatorio: Path, seed: int = bracol.SEED_PADRAO,
          refazer_divisao: bool = False, verificar: bool = False, raiz: Path = bracol.RAIZ_REPO,
          ausentes_esperados=bracol.IDS_AUSENTES_ESPERADOS) -> int:
    """Gera o manifest (ou, com verificar=True, só o confere) e grava o relatório.

    Devolve o código de saída: 0 sem erros, 1 com erros. Com erros, e sempre no modo
    verificar, o manifest não é alterado: só o relatório é gravado.
    """
    if verificar and refazer_divisao:
        raise ErroFatal("--verificar e --refazer-divisao não podem ser usados juntos")
    raiz, entrada = raiz.resolve(), entrada.resolve()
    manifest, relatorio = manifest.resolve(), relatorio.resolve()
    arquivo_csv, pasta_imagens = localizar_entrada(entrada)
    if not arquivo_csv.is_relative_to(raiz):
        raise ErroFatal(f"a entrada precisa ficar dentro do repositório ({raiz}): {entrada}")
    anterior = ler_manifest_anterior(manifest)
    if verificar and not anterior:
        raise ErroFatal(f"manifest não encontrado: {manifest}; rode sem --verificar para gerá-lo")
    fmt.dizer("Modo: " + ("verificação (o manifest não é alterado)" if verificar else "gerar"))
    fmt.dizer(f"Entrada: {_exibir(arquivo_csv.parent, raiz)}")

    oc = Ocorrencias()
    linhas, csv_com_erro = ler_csv(arquivo_csv, oc)
    imagens, ilegiveis = inspecionar_imagens(pasta_imagens, oc)
    fixos = {}
    if not refazer_divisao:
        fixos = {i: linha["split"] for i, linha in anterior.items() if linha.get("split")}
    linhas_manifest, divisao = montar_manifest(
        linhas, imagens, raiz, seed, fixos, ausentes_esperados, csv_com_erro | ilegiveis, oc
    )
    if verificar:
        conferir_com_versionado(linhas_manifest, anterior, oc)
    else:
        _comparar_com_anterior(linhas_manifest, anterior, fixos, oc)
    presentes = [linha for linha in linhas_manifest if linha["presente"]]
    grupo_de = {linha["id"]: linha["grupo"] for linha in presentes}
    ctx = {
        "entrada": _exibir(arquivo_csv.parent, raiz),
        "csv_sha256": hashlib.sha256(arquivo_csv.read_bytes()).hexdigest(),
        "manifest": _exibir(manifest, raiz),
        "seed": seed,
        "refazer_divisao": refazer_divisao,
        "imagens": imagens,
        "pares_quadro": bracol.pares_mais_proximos(presentes, chave="phash", grupos=grupo_de),
        "pares_folha": bracol.pares_mais_proximos(presentes, chave="phash_folha", grupos=grupo_de),
        "ausentes_esperados": set(ausentes_esperados),
    }
    relatorio.parent.mkdir(parents=True, exist_ok=True)
    relatorio.write_text(gerar_relatorio(linhas_manifest, ctx, oc), encoding="utf-8", newline="\n")
    if not oc.erros and not verificar:
        escrever_manifest(linhas_manifest, manifest)

    elegiveis = Counter(linha["split"] for linha in linhas_manifest if not linha["excluida"])
    grupos = Counter(grupo_de.values())
    mais_perto = {chave: ctx[chave][0][0] if ctx[chave] else "-"
                  for chave in ("pares_quadro", "pares_folha")}
    fmt.dizer("")
    fmt.dizer(f"Imagens: {len(presentes)} com imagem de {len(linhas_manifest)} linhas do csv "
              f"({len(linhas_manifest) - len(presentes)} sem imagem)")
    fmt.dizer(f"Grupos com mais de uma imagem: {sum(1 for q in grupos.values() if q > 1)} | "
              f"par não agrupado mais próximo: quadro {mais_perto['pares_quadro']} bits, "
              f"folha {mais_perto['pares_folha']} bits")
    fmt.dizer(f"Elegíveis: {sum(elegiveis.values())} | treino {elegiveis['treino']} | "
              f"val {elegiveis['val']} | teste {elegiveis['teste']} | "
              f"divisão: {divisao['mantidas']} mantidas, {divisao['novas']} novas")
    fmt.dizer(f"Erros: {len(oc.erros)} | avisos: {len(oc.avisos)}")
    for erro in oc.erros[:10]:
        fmt.dizer(f"  ERRO: {erro}")
    if len(oc.erros) > 10:
        fmt.dizer(f"  ... e mais {len(oc.erros) - 10} (ver o relatório)")
    fmt.dizer(f"Relatório: {_exibir(relatorio, raiz)}")
    if oc.erros and verificar:
        fmt.dizer("Manifest NÃO confere com os dados: veja os erros acima e no relatório.")
    elif oc.erros:
        fmt.dizer("Manifest NÃO gravado: corrija os erros e rode de novo.")
    elif verificar:
        fmt.dizer(f"Manifest confere com os dados: {_exibir(manifest, raiz)}")
    else:
        fmt.dizer(f"Manifest: {_exibir(manifest, raiz)} ({len(linhas_manifest)} linhas)")
    return 1 if oc.erros else 0


def _exibir(caminho: Path, raiz: Path) -> str:
    """Caminho relativo à raiz do repo em formato POSIX (absoluto, se estiver fora dela)."""
    return caminho.relative_to(raiz).as_posix() if caminho.is_relative_to(raiz) else str(caminho)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gera o manifest do BRACOL e o relatorio de integridade (ver data/README.md)."
    )
    ap.add_argument("--entrada", type=Path, default=bracol.PASTA_BRACOL,
                    help="pasta com a copia do BRACOL (padrao: data/raw/bracol/bracol_completo)")
    ap.add_argument("--manifest", type=Path, default=bracol.MANIFEST_BRACOL,
                    help="arquivo do manifest (padrao: data/manifests/bracol.csv)")
    ap.add_argument("--relatorio", type=Path, default=RELATORIO_PADRAO,
                    help="relatorio de integridade (padrao: data/reports/integridade_bracol.md)")
    ap.add_argument("--seed", type=int, default=bracol.SEED_PADRAO,
                    help=f"semente da divisao (padrao: {bracol.SEED_PADRAO})")
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--verificar", action="store_true",
                      help="so confere dados x manifest: grava o relatorio, nunca o manifest")
    modo.add_argument("--refazer-divisao", action="store_true",
                      help="ignora a divisao do manifest anterior e distribui tudo de novo")
    args = ap.parse_args(argv)
    try:
        return gerar(args.entrada, args.manifest, args.relatorio, args.seed,
                     refazer_divisao=args.refazer_divisao, verificar=args.verificar)
    except ErroFatal as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
