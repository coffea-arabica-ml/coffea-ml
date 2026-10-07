"""Invariantes do manifest versionado (data/manifests/bracol.csv).

Não precisam das imagens: rodam em qualquer máquina com o repositório. Os números fixos abaixo
são os do subconjunto PARCIAL; se chegar a cópia completa do BRACOL, atualize-os de propósito.
"""
import csv
from collections import Counter
from pathlib import PurePosixPath

import pytest

import bracol

pytestmark = pytest.mark.skipif(
    not bracol.MANIFEST_BRACOL.is_file(), reason="manifest ainda não gerado"
)


@pytest.fixture(scope="module")
def linhas():
    with open(bracol.MANIFEST_BRACOL, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == bracol.COLUNAS_MANIFEST
        return list(leitor)


def test_uma_linha_por_id_do_csv_original(linhas):
    assert [int(linha["id"]) for linha in linhas] == list(range(1, 1748))
    assert {linha["fonte"] for linha in linhas} == {"bracol"}
    for linha in linhas:
        if linha["caminho"]:
            caminho = PurePosixPath(linha["caminho"])
            assert not caminho.is_absolute() and "\\" not in linha["caminho"]
            assert caminho.parts[:3] == ("data", "raw", "bracol")


def test_contagens_do_subconjunto_parcial(linhas):
    presentes = [linha for linha in linhas if linha["presente"] == "1"]
    assert len(presentes) == 1401
    assert Counter(linha["classe"] or "classe_5" for linha in presentes) == {
        "saudavel": 142,
        "bicho_mineiro": 253,
        "ferrugem": 465,
        "phoma": 346,
        "cercosporiose": 136,
        "classe_5": 59,
    }
    ausentes = {int(linha["id"]) for linha in linhas if linha["presente"] == "0"}
    assert ausentes == bracol.IDS_AUSENTES_ESPERADOS


def test_classe_e_motivos_seguem_as_regras(linhas):
    for linha in linhas:
        ps = int(linha["predominant_stress"])
        assert linha["classe"] == (bracol.classe_do_projeto(ps) or "")
        motivos = bracol.motivos_exclusao(ps, linha["presente"] == "1")
        assert linha["motivo_exclusao"] == ";".join(motivos)
        assert linha["excluida"] == ("1" if motivos else "0")


def test_excluidas_sem_split_e_elegiveis_com_split(linhas):
    for linha in linhas:
        if linha["excluida"] == "1":
            assert linha["split"] == ""
        else:
            assert linha["split"] in bracol.SPLITS


def test_nenhum_grupo_em_dois_splits(linhas):
    splits_do_grupo = {}
    for linha in linhas:
        if linha["split"]:
            splits_do_grupo.setdefault(linha["grupo"], set()).add(linha["split"])
    assert all(len(splits) == 1 for splits in splits_do_grupo.values())


def test_teste_so_tem_bracol(linhas):
    assert {linha["fonte"] for linha in linhas if linha["split"] == "teste"} == {"bracol"}


def test_proporcoes_por_classe(linhas):
    contagem = Counter((linha["classe"], linha["split"]) for linha in linhas if linha["split"])
    for classe in bracol.CLASSES:
        n = sum(contagem[classe, s] for s in bracol.SPLITS)
        for split, pct in bracol.PROPORCOES.items():
            assert abs(contagem[classe, split] - n * pct / 100) <= 1


def test_ler_manifest_no_manifest_versionado():
    assert len(bracol.ler_manifest()) == 1342
    assert len(bracol.ler_manifest(split="teste")) == 202
    assert len(bracol.ler_manifest(incluir_excluidas=True)) == 1747
