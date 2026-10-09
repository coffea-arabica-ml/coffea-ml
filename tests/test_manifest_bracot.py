"""Invariantes do manifest versionado do BRACOT (data/manifests/bracot.csv).

Não precisam das fotos: rodam em qualquer máquina com o repositório. Os números fixos abaixo são
os de 09/10/2026; mudar qualquer um deles exige uma decisão explícita do gestor e um commit
próprio.
"""
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import PurePosixPath

import pytest

import bracot

pytestmark = pytest.mark.skipif(
    not bracot.MANIFEST_BRACOT.is_file(), reason="manifest ainda não gerado"
)

PASTA = {"treino": "train", "teste": "test"}


@pytest.fixture(scope="module")
def linhas():
    with open(bracot.MANIFEST_BRACOT, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == bracot.COLUNAS_MANIFEST
        return list(leitor)


def test_uma_linha_por_foto(linhas):
    ids = [linha["id"] for linha in linhas]
    assert len(ids) == len(set(ids)) == 300 and ids == sorted(ids)
    assert {linha["fonte"] for linha in linhas} == {"bracot"}
    assert len({linha["sha256"] for linha in linhas}) == 300  # nenhuma duplicata exata
    for linha in linhas:
        pasta = PASTA[linha["split"]]
        caminho = PurePosixPath(linha["caminho"])
        assert caminho.parts[:5] == ("data", "raw", "bracot", "bracot-data", pasta)
        assert caminho.name == f"{linha['id']}.jpg"
        coco = bracot.SPLITS_DOS_AUTORES[pasta][1]
        assert linha["anotacao"] == f"data/raw/bracot/bracot-data/{pasta}/{coco}"
        assert (linha["largura"], linha["altura"]) == ("4032", "3024")


def test_divisao_dos_autores_e_folhas(linhas):
    assert Counter(linha["split"] for linha in linhas) == {"treino": 240, "teste": 60}
    folhas = Counter()
    for linha in linhas:
        folhas[linha["split"]] += int(linha["n_folhas"])
    assert folhas == {"treino": 1318, "teste": 344}
    assert {int(linha["n_folhas"]) for linha in linhas} <= set(range(2, 13))
    assert all(0 < float(linha["area_coberta"]) < 1 for linha in linhas)


def test_datas_e_dias(linhas):
    dias = Counter((linha["data_hora"][:10], linha["split"]) for linha in linhas)
    assert dias == {("2019-08-31", "treino"): 109, ("2019-08-31", "teste"): 15,
                    ("2019-12-08", "treino"): 131, ("2019-12-08", "teste"): 45}
    for linha in linhas:
        assert datetime.fromisoformat(linha["data_hora"]) == bracot.data_hora_do_nome(linha["id"])


def test_cenas(linhas):
    membros = defaultdict(list)
    for linha in linhas:
        membros[linha["cena"]].append(linha["id"])
    assert len(membros) == 90
    assert all(cena == min(ids) for cena, ids in membros.items())  # o id da primeira foto
    datas = {linha["id"]: datetime.fromisoformat(linha["data_hora"]) for linha in linhas}
    assert bracot.cenas(datas) == {linha["id"]: linha["cena"] for linha in linhas}


def test_teste_sobreposto_fica_na_cena_da_foto_de_treino(linhas):
    split = {linha["id"]: linha["split"] for linha in linhas}
    cena = {linha["id"]: linha["cena"] for linha in linhas}
    for teste, treino in bracot.TESTE_SOBREPOSTO:
        assert (split[teste], split[treino]) == ("teste", "treino")
        assert cena[teste] == cena[treino]  # o intervalo de 10 s junta cada par numa cena


def test_divisao_fixada_por_hash(linhas):
    # "id:split" em ordem de id, unidos por ";". Valor de 09/10/2026, conferido com um cálculo
    # independente a partir das pastas train/ e test/.
    esperado = "890c3fb2c5065fedd3a5b3807cbcd082070dcfd24d75b579b263a352853c0317"
    assert bracot.hash_da_divisao(linhas) == esperado


def test_ler_manifest_no_manifest_versionado():
    assert len(bracot.ler_manifest()) == 300
    teste = bracot.ler_manifest(split="teste")
    assert len(teste) == 60 and int(teste["n_folhas"].sum()) == 344
