"""
Contrato dos dados do BRACOL neste projeto: classes, mapeamento dos rótulos originais,
regras de exclusão e caminhos padrão.

É o único lugar onde essas definições ficam. organize_dataset.py, eda_bracol.py, os testes
e a Frente 9 importam daqui (com a pasta data/ no sys.path: `import bracol`).

O BRACOL usado aqui é PARCIAL: 1.401 de 1.747 imagens (ver data/README.md).
"""
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------ caminhos
RAIZ_REPO = Path(__file__).resolve().parents[1]
PASTA_RAW_BRACOL = RAIZ_REPO / "data" / "raw" / "bracol"
MANIFEST_BRACOL = RAIZ_REPO / "data" / "manifests" / "bracol.csv"
PASTA_REPORTS = RAIZ_REPO / "data" / "reports"

# ------------------------------------------------------------------------- classes
# Ordem do RF01. O índice de uma classe para o modelo é CLASSES.index(nome), NUNCA o
# código predominant_stress do csv: com o código, bicho-mineiro (1) e ferrugem (2) se trocam.
CLASSES = ["saudavel", "ferrugem", "bicho_mineiro", "phoma", "cercosporiose"]

# Código predominant_stress do dataset.csv do BRACOL -> classe do projeto.
PS_PARA_CLASSE = {
    0: "saudavel",
    1: "bicho_mineiro",
    2: "ferrugem",
    3: "phoma",
    4: "cercosporiose",
}

# Código 5: significado desconhecido (62 linhas no csv completo). Os autores foram
# consultados; até a resposta, essas linhas ficam fora de todas as tarefas.
PS_DESCONHECIDO = 5

RESSALVA_PHOMA_CERCOSPORA = (
    "A correspondência das colunas 'phoma' e 'cercospora' do dataset.csv (códigos 3 e 4 de "
    "predominant_stress) com as classes 'brown leaf spot' e 'cercospora leaf spot' de "
    "Esgario et al. (2020) ainda não foi confirmada. Aqui elas viram as classes 'phoma' e "
    "'cercosporiose' do projeto."
)

# Coluna severity: fração da área da folha afetada.
SEVERIDADES = {
    0: "saudável (<0,1%)",
    1: "muito baixa (0,1-5%)",
    2: "baixa (5-10%)",
    3: "alta (10-15%)",
    4: "muito alta (>15%)",
}

# ----------------------------------------------------------------------- exclusões
MOTIVO_AUSENTE = "imagem_ausente"
MOTIVO_CLASSE_5 = "predominant_stress_5"

# Ids sem imagem na cópia recuperada. O zip publicado (DOI 10.17632/yy2k5y8mxg.1) não tem
# índice central, guarda as entradas em ordem alfabética do nome e termina no meio de
# 688.jpg: faltam exatamente os ids cujo nome vem depois de "688" nessa ordem.
IDS_AUSENTES_ESPERADOS = frozenset([*range(7, 10), *range(69, 100), *range(688, 1000)])

# ---------------------------------------------------------------- hashes e divisão
# pHash de 256 bits: imagehash.phash(img, hash_size=PHASH_TAMANHO). O padrão de 64 bits não
# serve aqui: folhas diferentes (137 e 1510) ficam a 2 bits. Medido no BRACOL com 256 bits:
# re-salvar, redimensionar ou mudar o brilho em 15% muda no máximo 18 bits; folhas distintas
# ficam a 48 ou mais. Recorte, rotação e nova foto da mesma folha não são detectados.
PHASH_TAMANHO = 16
LIMIAR_QUASE_DUPLICATA = 24

SPLITS = ("treino", "val", "teste")
PROPORCOES = {"treino": 70, "val": 15, "teste": 15}  # em %, somam 100
SEED_PADRAO = 42
_DESEMPATE = ("teste", "val", "treino")  # ordem fixa quando cotas ou déficits empatam


def classe_do_projeto(ps: int) -> str | None:
    """Classe do projeto para um código predominant_stress (inteiro de 0 a 5).

    Devolve None para o código 5 (significado desconhecido) e levanta ValueError para
    qualquer outro valor.
    """
    if ps == PS_DESCONHECIDO:
        return None
    if ps not in PS_PARA_CLASSE:
        raise ValueError(f"predominant_stress inválido: {ps!r} (esperado inteiro de 0 a 5)")
    return PS_PARA_CLASSE[ps]


def motivos_exclusao(ps: int, presente: bool) -> list[str]:
    """Motivos para a linha ficar fora de treino, val e teste, em ordem fixa.

    Lista vazia quer dizer linha elegível. No manifest, os motivos são unidos com ";".
    """
    classe = classe_do_projeto(ps)  # também valida o código
    motivos = []
    if not presente:
        motivos.append(MOTIVO_AUSENTE)
    if classe is None:
        motivos.append(MOTIVO_CLASSE_5)
    return motivos


# ----------------------------------------------------------------------- agrupamento
def agrupar_por_hash(
    itens, limiar: int = LIMIAR_QUASE_DUPLICATA, fonte: str = "bracol"
) -> dict[int, str]:
    """Junta no mesmo grupo as duplicatas exatas (sha256 igual) e as quase-duplicatas
    (pHash a no máximo `limiar` bits), de forma transitiva.

    itens: dicts com "id", "sha256" e "phash" (hex) das imagens presentes.
    Devolve {id: grupo}, com grupo = "<fonte>-<menor id do grupo>".
    """
    itens = sorted(itens, key=lambda item: item["id"])
    for item in itens:
        if not item["sha256"] or not item["phash"]:
            raise ValueError(f"id {item['id']} sem sha256 ou phash")
    if not itens:
        return {}

    pai = list(range(len(itens)))  # união por componentes conexos; a raiz é o menor índice

    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    def unir(i, j):
        ri, rj = raiz(i), raiz(j)
        if ri != rj:
            pai[max(ri, rj)] = min(ri, rj)

    primeiro_com_sha = {}
    for i, item in enumerate(itens):
        unir(i, primeiro_com_sha.setdefault(item["sha256"], i))
    for i, j in _pares_ate(_matriz_de_bits([item["phash"] for item in itens]), limiar):
        unir(i, j)
    return {item["id"]: f"{fonte}-{itens[raiz(i)]['id']}" for i, item in enumerate(itens)}


def _matriz_de_bits(hashes_hex):
    """Hashes hex de mesmo tamanho (múltiplo de 16) -> matriz uint64, uma linha por hash."""
    tamanhos = {len(h) for h in hashes_hex}
    if len(tamanhos) != 1 or next(iter(tamanhos)) % 16:
        raise ValueError(
            f"hashes precisam ter o mesmo tamanho, múltiplo de 16 hex: {sorted(tamanhos)}"
        )
    dados = b"".join(bytes.fromhex(h) for h in hashes_hex)
    return np.frombuffer(dados, dtype=">u8").astype(np.uint64).reshape(len(hashes_hex), -1)


def _pares_ate(matriz, limiar, bloco=None):
    """Pares (i, j), com i < j, a distância de Hamming <= limiar. Calcula em blocos de linhas
    para não montar a matriz N x N inteira de uma vez (escala para fontes grandes)."""
    n, palavras = matriz.shape
    bloco = bloco or max(1, (1 << 20) // (n * palavras))
    for inicio in range(0, n, bloco):
        xor = matriz[inicio:inicio + bloco, None, :] ^ matriz[None, :, :]
        dist = np.bitwise_count(xor).sum(axis=2)
        for i, j in zip(*np.nonzero(dist <= limiar)):
            i += inicio
            if i < j:
                yield int(i), int(j)


# --------------------------------------------------------------------------- divisão
def dividir(itens, proporcoes=PROPORCOES, seed: int = SEED_PADRAO, fixos=None) -> dict[int, str]:
    """Divide as linhas elegíveis em treino/val/teste, estratificando por classe.

    itens: dicts com "id", "grupo", "classe" e "severity" (só linhas elegíveis).
    fixos: {id: split} de uma divisão anterior. Grupos com imagens já atribuídas ficam onde
        estão e só os grupos novos são distribuídos (divisão estável).
    Devolve {id: split}.

    A unidade é o grupo: um grupo nunca se divide. Em cada classe, as cotas saem pelo método
    dos maiores restos; os grupos novos, ordenados por severidade e por um sorteio
    determinístico (sha256 de "seed:grupo"), vão um a um para o split com maior déficit em
    relação à cota, o que espalha val e teste por todos os níveis de severidade.
    """
    if sorted(proporcoes) != sorted(SPLITS) or sum(proporcoes.values()) != 100:
        raise ValueError(f"proporcoes precisam ter {SPLITS} e somar 100: {proporcoes}")
    fixos = fixos or {}
    invalidos = {s for s in fixos.values() if s not in SPLITS}
    if invalidos:
        raise ValueError(f"split desconhecido em fixos: {sorted(invalidos)}")

    grupos = {}
    vistos = set()
    for item in itens:
        if not item["classe"]:
            raise ValueError(f"id {item['id']} sem classe: linhas excluídas não entram na divisão")
        if item["id"] in vistos:
            raise ValueError(f"id repetido: {item['id']}")
        vistos.add(item["id"])
        grupos.setdefault(item["grupo"], []).append(item)

    por_classe = {}
    for nome, membros in grupos.items():
        membros.sort(key=lambda item: item["id"])
        anteriores = {fixos[item["id"]] for item in membros if item["id"] in fixos}
        if len(anteriores) > 1:
            raise ValueError(
                f"grupo {nome} tem imagens em splits diferentes {sorted(anteriores)}; "
                "use --refazer-divisao"
            )
        split_fixo = anteriores.pop() if anteriores else None
        # Classe e severidade do grupo: as do menor id (grupos mistos viram aviso no relatório).
        por_classe.setdefault(membros[0]["classe"], []).append((nome, membros, split_fixo))

    resultado = {}
    for grupos_da_classe in por_classe.values():
        cota = _cotas(sum(len(membros) for _, membros, _ in grupos_da_classe), proporcoes)
        ocupado = Counter()
        novos = []
        for nome, membros, split_fixo in grupos_da_classe:
            if split_fixo:
                ocupado[split_fixo] += len(membros)
                resultado.update((item["id"], split_fixo) for item in membros)
            else:
                novos.append((nome, membros))
        novos.sort(key=lambda g: (g[1][0]["severity"], _sorteio(seed, g[0])))

        vaga = {s: max(cota[s] - ocupado[s], 0) for s in SPLITS}
        total_vaga = sum(vaga.values())  # >= número de imagens novas
        atribuido = Counter()
        processadas = 0
        for _, membros in novos:
            processadas += len(membros)
            cabem = [s for s in SPLITS if atribuido[s] + len(membros) <= vaga[s]]
            candidatos = cabem or [s for s in SPLITS if atribuido[s] < vaga[s]] or list(SPLITS)
            # Déficit = quanto o split deveria ter até aqui menos o que já tem, multiplicado
            # por total_vaga para comparar só inteiros.
            split = max(
                candidatos,
                key=lambda s: (
                    vaga[s] * processadas - atribuido[s] * total_vaga,
                    -_DESEMPATE.index(s),
                ),
            )
            atribuido[split] += len(membros)
            resultado.update((item["id"], split) for item in membros)
    return resultado


def _sorteio(seed, grupo):
    """Chave de ordenação determinística: não depende da ordem de entrada nem do PYTHONHASHSEED."""
    return hashlib.sha256(f"{seed}:{grupo}".encode()).hexdigest()


def _cotas(n, proporcoes):
    """Imagens por split para n imagens, pelo método dos maiores restos (aritmética inteira)."""
    cota = {s: n * proporcoes[s] // 100 for s in SPLITS}
    resto = {s: n * proporcoes[s] % 100 for s in SPLITS}
    sobra = n - sum(cota.values())
    for s in sorted(SPLITS, key=lambda s: (-resto[s], _DESEMPATE.index(s)))[:sobra]:
        cota[s] += 1
    return cota
