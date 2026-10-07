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
import unicodedata
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image

import bracol

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
    _dizer(f"Abrindo {len(arquivos)} imagens (decodificação completa, SHA-256 e pHash)...")
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
                    "formato": im.format,
                    "modo": im.mode,
                }
        except Exception as e:  # truncado, corrompido ou não é imagem: o Pillow usa vários tipos
            oc.erros.append(f"imagem ilegível: {p.name} ({type(e).__name__}: {e})")
            ilegiveis.add(i)
        if n % 250 == 0:
            _dizer(f"  ...{n}/{len(arquivos)}")
    return imagens, ilegiveis


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
        oc.erros.append(f"imagens ausentes fora da lista esperada: ids {_faixas(inesperados)}")

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
            **{c: imagem[c] if imagem else "" for c in ("largura", "altura", "sha256", "phash")},
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


def conferir_com_versionado(manifest, versionado, oc):
    """Modo --verificar: o manifest recalculado precisa bater com o versionado. O SHA-256 e as
    outras colunas são comparados exatamente; o pHash, com tolerância de TOLERANCIA_PHASH bits
    (outro decodificador JPEG pode mudar alguns bits)."""
    colunas = list(next(iter(versionado.values())))
    if colunas != bracol.COLUNAS_MANIFEST:
        oc.erros.append(
            f"colunas do manifest versionado: {colunas}; esperado: {bracol.COLUNAS_MANIFEST}"
        )
        return
    atuais = {linha["id"]: linha for linha in manifest}
    so_no_versionado = sorted(set(versionado) - set(atuais))
    so_nos_dados = sorted(set(atuais) - set(versionado))
    if so_no_versionado:
        oc.erros.append(
            f"ids no manifest versionado que não estão nos dados: {_faixas(so_no_versionado)}"
        )
    if so_nos_dados:
        oc.erros.append(f"ids nos dados que faltam no manifest versionado: {_faixas(so_nos_dados)}")
    divergentes = {}
    for i in sorted(set(atuais) & set(versionado)):
        for coluna in bracol.COLUNAS_MANIFEST:
            atual, gravado = str(atuais[i][coluna]), versionado[i][coluna]
            if coluna == "phash" and atual and gravado:
                confere = bracol.distancia_hamming(atual, gravado) <= bracol.TOLERANCIA_PHASH
            else:
                confere = atual == gravado
            if not confere:
                divergentes.setdefault(coluna, []).append(i)
    for coluna in bracol.COLUNAS_MANIFEST:
        if coluna in divergentes:
            ids = _faixas(divergentes[coluna])
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
        oc.avisos.append(f"rótulos diferentes do manifest anterior: ids {_faixas(rotulo_mudou)}")
    if imagem_mudou:
        oc.avisos.append(
            f"imagens diferentes do manifest anterior (SHA-256 mudou): ids {_faixas(imagem_mudou)}"
        )
    com_split = {linha["id"] for linha in manifest if linha["split"]}
    descartados = sorted(set(fixos) - com_split)
    if descartados:
        oc.avisos.append(
            "atribuições de split descartadas (deixaram de ser elegíveis): "
            f"ids {_faixas(descartados)}"
        )


def escrever_manifest(linhas: list[dict], caminho: Path) -> None:
    """Grava o manifest em UTF-8 sem BOM e com fim de linha LF (mesmos bytes em qualquer SO)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=bracol.COLUNAS_MANIFEST, lineterminator="\n")
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
            f"> **Cópia PARCIAL do BRACOL:** {_n(len(presentes))} de {_n(len(linhas))} imagens. "
            "Resultados com ela não são comparáveis com Esgario et al. (2020).",
            ">",
        ]
    else:
        r += [f"> **Cópia completa do BRACOL:** {_n(len(presentes))} imagens.", ">"]
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
    r += _tabela(["item", "valor"], [
        ["entrada", f"`{ctx['entrada']}`"],
        ["dataset.csv", f"{_n(len(linhas))} linhas, SHA-256 `{ctx['csv_sha256']}`"],
        ["manifest", f"`{ctx['manifest']}`"],
        ["proporções", ", ".join(f"{s} {p}%" for s, p in bracol.PROPORCOES.items())],
        ["seed", ctx["seed"]],
        ["divisão", divisao],
        ["pHash", f"{bracol.PHASH_TAMANHO ** 2} bits (`hash_size={bracol.PHASH_TAMANHO}`); "
                  f"quase-duplicata a até {bracol.LIMIAR_QUASE_DUPLICATA} bits"],
    ], alinhar="ll") + [""]

    r += ["## CSV x arquivos", ""]
    r += _tabela(["", "quantidade"], [
        ["linhas no dataset.csv", _n(len(linhas))],
        ["imagens `<id>.jpg` lidas", _n(len(ctx["imagens"]))],
        ["imagens sem linha no csv", _n(len(ctx["imagens"]) - len(presentes))],
        ["ids sem imagem", _n(len(ausentes))],
        ["ids sem imagem, esperados (truncamento do zip)", _n(len(ausentes) - len(inesperados))],
        ["ids sem imagem, inesperados", _n(len(inesperados))],
    ]) + [""]
    if ausentes:
        r += [f"Ids sem imagem: {_faixas(ausentes)}.", ""]
    tabela = []
    for classe in [*bracol.CLASSES, ""]:
        da_classe = [linha for linha in linhas if linha["classe"] == classe]
        sem = sum(1 for linha in da_classe if not linha["presente"])
        tabela.append([classe or "(classe 5)", _n(len(da_classe)), _n(len(da_classe) - sem),
                       _n(sem), _pct(sem, len(da_classe))])
    tabela.append(["**total**", _n(len(linhas)), _n(len(presentes)), _n(len(ausentes)),
                   _pct(len(ausentes), len(linhas))])
    r += _tabela(["classe", "no csv", "com imagem", "sem imagem", "perda"], tabela) + [""]

    r += ["## Imagens", ""]
    tipos = Counter(
        (d["formato"], d["modo"], f"{d['largura']}x{d['altura']}") for d in ctx["imagens"].values()
    )
    r += _tabela(["formato", "modo", "resolução", "imagens"],
                 [[*tipo, _n(q)] for tipo, q in sorted(tipos.items())], alinhar="lllr") + [""]
    r += ["As imagens desta tabela abriram e decodificaram por completo; as ilegíveis aparecem "
          "nos erros.", ""]

    r += ["## Duplicatas", ""]
    exatas = sum(1 for q in Counter(linha["sha256"] for linha in presentes).values() if q > 1)
    por_grupo = Counter(linha["grupo"] for linha in presentes)
    grupos = sorted((g for g, q in por_grupo.items() if q > 1), key=lambda g: int(g.split("-")[1]))
    r += [
        f"- Duplicatas exatas (mesmo SHA-256): {_n(exatas)} grupo(s).",
        f"- Grupos com mais de uma imagem (SHA-256 igual ou pHash a até "
        f"{bracol.LIMIAR_QUASE_DUPLICATA} bits): {_n(len(grupos))}.",
    ]
    for g in grupos:
        membros = ", ".join(str(linha["id"]) for linha in presentes if linha["grupo"] == g)
        r.append(f"  - `{g}`: ids {membros}")
    r += ["", "Pares mais próximos pelo pHash (para conferir o limiar):", ""]
    nome = {linha["id"]: linha["classe"] or "(classe 5)" for linha in linhas}
    r += _tabela(["id A", "id B", "distância (bits)", "classe A", "classe B"],
                 [[a, b, d, nome[a], nome[b]] for d, a, b in ctx["pares"]], alinhar="rrrll") + [""]

    r += ["## Exclusões", ""]
    motivos = Counter(linha["motivo_exclusao"] for linha in linhas if linha["excluida"])
    r += _tabela(["motivo", "linhas"], [
        *([f"`{m}`", _n(q)] for m, q in sorted(motivos.items())),
        ["**total excluídas**", _n(sum(motivos.values()))],
        ["**elegíveis**", _n(len(elegiveis))],
    ]) + [""]

    classe5 = [linha for linha in linhas if linha["predominant_stress"] == bracol.PS_DESCONHECIDO]
    com_imagem = [linha for linha in classe5 if linha["presente"]]
    sem_imagem = [str(linha["id"]) for linha in classe5 if not linha["presente"]]
    r += ["## Classe 5 (predominant_stress = 5)", ""]
    r += ["Significado desconhecido (os autores foram consultados): fica fora de classificação, "
          "severidade e multirrótulo até a resposta. Não atribuir classe por palpite.", ""]
    r += [f"Linhas: {_n(len(classe5))}, {_n(len(com_imagem))} com imagem "
          f"(sem imagem: {', '.join(sem_imagem) or 'nenhuma'}).", ""]
    combinacoes = Counter(
        "+".join(c for c in bracol.PS_PARA_COLUNA.values() if linha[c]) or "(nenhum)"
        for linha in com_imagem
    )
    r += _tabela(["estresses marcados (com imagem)", "imagens"],
                 [[c, _n(q)] for c, q in sorted(combinacoes.items(), key=lambda x: (-x[1], x[0]))])
    r += [""]

    r += ["## Divisão", ""]
    por_classe = Counter((linha["classe"], linha["split"]) for linha in elegiveis)
    tabela = []
    for classe in bracol.CLASSES:
        valores = [por_classe[classe, s] for s in bracol.SPLITS]
        tabela.append([classe, *map(_n, valores), _n(sum(valores))])
    totais = Counter(linha["split"] for linha in elegiveis)
    tabela.append(["**total**", *(_n(totais[s]) for s in bracol.SPLITS), _n(len(elegiveis))])
    r += _tabela(["classe", *bracol.SPLITS, "total"], tabela) + [""]
    por_severidade = Counter((linha["severity"], linha["split"]) for linha in elegiveis)
    tabela = []
    for nivel, descricao in bracol.SEVERIDADES.items():
        valores = [por_severidade[nivel, s] for s in bracol.SPLITS]
        tabela.append([f"{nivel}: {descricao}", *map(_n, valores),
                       _pct(por_severidade[nivel, "teste"], sum(valores))])
    r += _tabela(["severidade", *bracol.SPLITS, "% no teste"], tabela)
    return "\n".join(r) + "\n"


def _tabela(cabecalho, linhas, alinhar=None) -> list[str]:
    """Tabela markdown. `alinhar` tem uma letra por coluna (l ou r); o padrão é a primeira à
    esquerda e as outras à direita."""
    alinhar = alinhar or "l" + "r" * (len(cabecalho) - 1)
    saida = ["| " + " | ".join(cabecalho) + " |",
             "|" + "|".join("---" if a == "l" else "---:" for a in alinhar) + "|"]
    return saida + ["| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas]


def _n(valor: int) -> str:
    """Inteiro com ponto de milhar: 1747 -> 1.747."""
    return f"{valor:,}".replace(",", ".")


def _pct(parte: int, total: int) -> str:
    """Percentual com vírgula decimal: 14,8%."""
    return f"{100 * parte / total:.1f}%".replace(".", ",") if total else "-"


def _faixas(ids) -> str:
    """Ids em faixas: [7, 8, 9, 69, 70] -> '7-9, 69-70'."""
    faixas = []
    for i in sorted(ids):
        if faixas and i == faixas[-1][1] + 1:
            faixas[-1][1] = i
        else:
            faixas.append([i, i])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in faixas)


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
    _dizer("Modo: " + ("verificação (o manifest não é alterado)" if verificar else "gerar"))
    _dizer(f"Entrada: {_exibir(arquivo_csv.parent, raiz)}")

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
    ctx = {
        "entrada": _exibir(arquivo_csv.parent, raiz),
        "csv_sha256": hashlib.sha256(arquivo_csv.read_bytes()).hexdigest(),
        "manifest": _exibir(manifest, raiz),
        "seed": seed,
        "refazer_divisao": refazer_divisao,
        "imagens": imagens,
        "pares": bracol.pares_mais_proximos(presentes),
        "ausentes_esperados": set(ausentes_esperados),
    }
    relatorio.parent.mkdir(parents=True, exist_ok=True)
    relatorio.write_text(gerar_relatorio(linhas_manifest, ctx, oc), encoding="utf-8", newline="\n")
    if not oc.erros and not verificar:
        escrever_manifest(linhas_manifest, manifest)

    elegiveis = Counter(linha["split"] for linha in linhas_manifest if not linha["excluida"])
    grupos = Counter(linha["grupo"] for linha in presentes)
    _dizer("")
    _dizer(f"Imagens: {len(presentes)} com imagem de {len(linhas_manifest)} linhas do csv "
           f"({len(linhas_manifest) - len(presentes)} sem imagem)")
    _dizer(f"Grupos com mais de uma imagem: {sum(1 for q in grupos.values() if q > 1)} | "
           f"par mais próximo: {ctx['pares'][0][0] if ctx['pares'] else '-'} bits")
    _dizer(f"Elegíveis: {sum(elegiveis.values())} | treino {elegiveis['treino']} | "
           f"val {elegiveis['val']} | teste {elegiveis['teste']} | "
           f"divisão: {divisao['mantidas']} mantidas, {divisao['novas']} novas")
    _dizer(f"Erros: {len(oc.erros)} | avisos: {len(oc.avisos)}")
    for erro in oc.erros[:10]:
        _dizer(f"  ERRO: {erro}")
    if len(oc.erros) > 10:
        _dizer(f"  ... e mais {len(oc.erros) - 10} (ver o relatório)")
    _dizer(f"Relatório: {_exibir(relatorio, raiz)}")
    if oc.erros and verificar:
        _dizer("Manifest NÃO confere com os dados: veja os erros acima e no relatório.")
    elif oc.erros:
        _dizer("Manifest NÃO gravado: corrija os erros e rode de novo.")
    elif verificar:
        _dizer(f"Manifest confere com os dados: {_exibir(manifest, raiz)}")
    else:
        _dizer(f"Manifest: {_exibir(manifest, raiz)} ({len(linhas_manifest)} linhas)")
    return 1 if oc.erros else 0


def _exibir(caminho: Path, raiz: Path) -> str:
    """Caminho relativo à raiz do repo em formato POSIX (absoluto, se estiver fora dela)."""
    return caminho.relative_to(raiz).as_posix() if caminho.is_relative_to(raiz) else str(caminho)


def _dizer(texto: str = "") -> None:
    """Mensagem de terminal só em ASCII. Os acentos saem porque, no Windows, a saída
    redirecionada vai em cp1252 e os acentos viram lixo."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    print(sem_acento, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gera o manifest do BRACOL e o relatorio de integridade (ver data/README.md)."
    )
    ap.add_argument("--entrada", type=Path, default=bracol.PASTA_RAW_BRACOL,
                    help="pasta com a copia do BRACOL (padrao: data/raw/bracol)")
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
        _dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
