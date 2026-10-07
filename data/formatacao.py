"""
Formatação compartilhada pelos relatórios (organize_dataset.py e eda_bracol.py): números em
português (1.747; 14,8%), faixas de ids, tabelas markdown e mensagens de terminal sem acento.
"""
import unicodedata


def n(valor: int) -> str:
    """Inteiro com ponto de milhar: 1747 -> 1.747."""
    return f"{valor:,}".replace(",", ".")


def decimal(valor: float, casas: int = 1) -> str:
    """Número com vírgula decimal: 165.25 -> 165,3."""
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(parte: float, total: float, casas: int = 1) -> str:
    """Percentual com vírgula decimal: 14,8%."""
    return f"{decimal(100 * parte / total, casas)}%" if total else "-"


def intervalos(ids) -> list[tuple[int, int]]:
    """Ids em intervalos contíguos: [7, 8, 9, 69, 70] -> [(7, 9), (69, 70)]."""
    saida = []
    for i in sorted(ids):
        if saida and i == saida[-1][1] + 1:
            saida[-1] = (saida[-1][0], i)
        else:
            saida.append((i, i))
    return saida


def faixas(ids) -> str:
    """Ids em faixas: [7, 8, 9, 69, 70] -> '7-9, 69-70'."""
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in intervalos(ids))


def tabela(cabecalho, linhas, alinhar=None) -> list[str]:
    """Tabela markdown. `alinhar` tem uma letra por coluna (l ou r); o padrão é a primeira à
    esquerda e as outras à direita."""
    alinhar = alinhar or "l" + "r" * (len(cabecalho) - 1)
    saida = ["| " + " | ".join(cabecalho) + " |",
             "|" + "|".join("---" if a == "l" else "---:" for a in alinhar) + "|"]
    return saida + ["| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas]


def dizer(texto: str = "") -> None:
    """Mensagem de terminal só em ASCII. Os acentos saem porque, no Windows, a saída
    redirecionada vai em cp1252 e os acentos viram lixo."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    print(sem_acento, flush=True)
