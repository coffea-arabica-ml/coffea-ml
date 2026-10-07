"""
Contrato dos dados do BRACOL neste projeto: classes, mapeamento dos rótulos originais,
regras de exclusão e caminhos padrão.

É o único lugar onde essas definições ficam. organize_dataset.py, eda_bracol.py, os testes
e a Frente 9 importam daqui (com a pasta data/ no sys.path: `import bracol`).

O BRACOL usado aqui é PARCIAL: 1.401 de 1.747 imagens (ver data/README.md).
"""
from pathlib import Path

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
