"""Invariantes do manifest versionado do JMuBEN (data/manifests/jmuben.csv).

Não precisam das imagens: rodam em qualquer máquina com o repositório. Os números fixos abaixo
são os da cópia local de 09/10/2026; mudar qualquer um deles exige uma decisão explícita do
gestor e um commit próprio.
"""
import csv
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import pytest

import bracol
import jmuben

pytestmark = pytest.mark.skipif(
    not jmuben.MANIFEST_JMUBEN.is_file(), reason="manifest ainda não gerado"
)

SELECIONADAS = {"saudavel": 12, "ferrugem": 186, "bicho_mineiro": 135, "phoma": 122,
                "cercosporiose": 51}


@pytest.fixture(scope="module")
def linhas():
    with open(jmuben.MANIFEST_JMUBEN, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == jmuben.COLUNAS_MANIFEST
        return list(leitor)


def pasta(linha):
    return PurePosixPath(linha["caminho"]).parts[3]


def test_uma_linha_por_conteudo_distinto_e_sem_split(linhas):
    assert len(linhas) == 3757
    assert [int(linha["id"]) for linha in linhas] == list(range(1, 3758))
    assert [linha["caminho"] for linha in linhas] == sorted(linha["caminho"] for linha in linhas)
    assert len({linha["sha256"] for linha in linhas}) == 3757
    assert "split" not in linhas[0]  # fonte auxiliar: nunca validação nem teste
    assert {linha["fonte"] for linha in linhas} == {"jmuben"}
    for linha in linhas:
        caminho = PurePosixPath(linha["caminho"])
        assert not caminho.is_absolute() and "\\" not in linha["caminho"]
        assert caminho.parts[:3] == ("data", "raw", "jmuben")
        assert (linha["classe"], linha["subconjunto"]) == jmuben.PASTAS[pasta(linha)]


def test_arquivos_por_pasta_e_subconjunto(linhas):
    copias = Counter()
    for linha in linhas:
        copias[pasta(linha)] += int(linha["copias"])
    assert copias == {"Cerscospora": 7681, "Healthy": 18984, "Leaf rust": 8336, "Miner": 16978,
                      "Phoma": 6571}
    por_subconjunto = Counter()
    for linha in linhas:
        por_subconjunto[linha["subconjunto"]] += int(linha["copias"])
    # O JMuBEN publicado tem 22.591 imagens; a cópia local tem 22.588 (registrado, sem investigar).
    assert por_subconjunto == {"jmuben": 22588, "jmuben2": 35962}


def test_so_o_ilegivel_esperado_fica_excluido(linhas):
    excluidas = [linha for linha in linhas if linha["uso"] != jmuben.USO_TREINO]
    assert [linha["caminho"] for linha in excluidas] == ["data/raw/jmuben/Healthy/2 (691).jpg"]
    linha = excluidas[0]
    assert (linha["uso"], linha["motivo"], linha["grupo"], linha["selecionada"]) == \
        (jmuben.USO_EXCLUIDA, jmuben.MOTIVO_ILEGIVEL, "", "0")


def test_grupos_por_classe_e_nenhum_misto(linhas):
    classes = defaultdict(set)
    for linha in linhas:
        if linha["grupo"]:
            classes[linha["grupo"]].add(linha["classe"])
    assert all(len(c) == 1 for c in classes.values())
    assert Counter(next(iter(c)) for c in classes.values()) == {
        "saudavel": 12, "ferrugem": 406, "bicho_mineiro": 309, "phoma": 177, "cercosporiose": 80,
    }


def test_selecao_um_por_grupo_e_motivos(linhas):
    selecionadas = [linha for linha in linhas if linha["selecionada"] == "1"]
    assert Counter(linha["classe"] for linha in selecionadas) == SELECIONADAS
    assert len({linha["grupo"] for linha in selecionadas}) == len(selecionadas) == 506
    assert all(linha["uso"] == jmuben.USO_TREINO and linha["motivo"] == ""
               for linha in selecionadas)
    representados = {linha["grupo"] for linha in selecionadas}
    for linha in linhas:
        if linha["uso"] == jmuben.USO_TREINO and linha["selecionada"] == "0":
            esperado = (jmuben.MOTIVO_GRUPO_REPRESENTADO if linha["grupo"] in representados
                        else jmuben.MOTIVO_ACIMA_DO_TETO)
            assert linha["motivo"] == esperado


def test_selecao_segue_a_regra(linhas):
    itens = [{**linha, "id": int(linha["id"])} for linha in linhas]
    gravadas = {item["id"] for item in itens if item["selecionada"] == "1"}
    assert jmuben.selecionar(itens) == gravadas


def test_teto_e_metade_do_treino_do_bracol():
    if not bracol.MANIFEST_BRACOL.is_file():
        pytest.skip("manifest do BRACOL ainda não gerado")
    treino = Counter(bracol.ler_manifest(split="treino")["classe"])
    assert jmuben.CAP_POR_CLASSE == {c: treino[c] // 2 for c in bracol.CLASSES}


def test_selecao_fixada_por_hash(linhas):
    # Ids selecionados em ordem crescente, unidos por ";", sem espaços. Valor de 09/10/2026,
    # conferido com um cálculo independente (grupos pelo scipy e o sorteio reescrito à parte).
    ids = [int(linha["id"]) for linha in linhas if linha["selecionada"] == "1"]
    esperado = "6b206f932ee7c561ac1ac82749e20e0d612c81bab882a948c82f001ed4ddb1a5"
    assert jmuben.hash_da_selecao(ids) == esperado


def test_ler_manifest_no_manifest_versionado():
    selecionadas = jmuben.ler_manifest()
    assert len(selecionadas) == 506
    assert Counter(selecionadas["classe"]) == SELECIONADAS
    assert len(jmuben.ler_manifest(teto=0)) == 0
    dez = jmuben.ler_manifest(teto=10)
    assert len(dez) == 50 and set(dez["id"]) <= set(selecionadas["id"])
