"""Invariantes do manifest versionado (data/manifests/bracol.csv).

Não precisam das imagens: rodam em qualquer máquina com o repositório. Os números fixos abaixo
são os da cópia completa do BRACOL, adotada em 07/10/2026; mudar qualquer um deles exige uma
decisão explícita e um commit próprio.
"""
import csv
import hashlib
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


def test_contagens_da_copia_completa(linhas):
    # Trocado de propósito em 07/10/2026; na cópia parcial eram 1.401 imagens
    # (142, 253, 465, 346, 136 e 59).
    presentes = [linha for linha in linhas if linha["presente"] == "1"]
    assert len(presentes) == 1747
    assert Counter(linha["classe"] or "classe_5" for linha in presentes) == {
        "saudavel": 272,
        "bicho_mineiro": 387,
        "ferrugem": 531,
        "phoma": 348,
        "cercosporiose": 147,
        "classe_5": 62,
    }
    ausentes = {int(linha["id"]) for linha in linhas if linha["presente"] == "0"}
    assert ausentes == bracol.IDS_AUSENTES_ESPERADOS == frozenset()


def test_grupos_sao_exatamente_as_folhas_repetidas_conferidas(linhas):
    # Grupos com mais de uma imagem = os 8 pares de PARES_MESMA_FOLHA (conferidos visualmente em
    # 07/10/2026). Um grupo novo, ou um par que deixe de se agrupar, tem de ser revisto à mão.
    membros = {}
    for linha in linhas:
        if linha["grupo"]:
            membros.setdefault(linha["grupo"], set()).add(int(linha["id"]))
    grupos = {frozenset(ids) for ids in membros.values() if len(ids) > 1}
    assert grupos == {frozenset(par) for par in bracol.PARES_MESMA_FOLHA}


def test_folhas_repetidas_ficam_no_mesmo_split(linhas):
    split = {int(linha["id"]): linha["split"] for linha in linhas}
    assert all(split[a] == split[b] for a, b in bracol.PARES_MESMA_FOLHA)


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


def test_divisao_fixada_por_hash(linhas):
    # Fixa a divisão vigente: linhas elegíveis em ordem de id, no formato "id:split" unidas por
    # ";", sem espaços e sem quebra de linha no fim.
    # Trocado de propósito em 07/10/2026: cópia completa, divisão refeita do zero (seed 42) com
    # os grupos das folhas repetidas (ver HISTORICO_DIVISAO). O valor foi calculado de forma
    # independente a partir do dataset.csv e de bracol.dividir com esses 7 grupos de elegíveis.
    # O valor anterior, da cópia parcial, era
    # c3703a39d085a803b6188a10a45d35d5c650b89fdeaf10f02906f8ff0cc1ae00.
    texto = ";".join(
        f"{linha['id']}:{linha['split']}"
        for linha in sorted(linhas, key=lambda linha: int(linha["id"]))
        if linha["motivo_exclusao"] == ""
    )
    esperado = "ecc98bfc9827448747f7451814904d5d9f26fdfc264b3f6ca69e6b37df989baa"
    assert hashlib.sha256(texto.encode("utf-8")).hexdigest() == esperado


def test_ler_manifest_no_manifest_versionado():
    # Trocado de propósito em 07/10/2026; na cópia parcial eram 1.342 elegíveis e 202 no teste.
    assert len(bracol.ler_manifest()) == 1685
    assert len(bracol.ler_manifest(split="teste")) == 253
    assert len(bracol.ler_manifest(incluir_excluidas=True)) == 1747
