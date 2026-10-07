"""Testes do mapeamento de classes e das regras de exclusão (data/bracol.py)."""
import pytest

import bracol


def test_classes_na_ordem_do_rf01():
    assert bracol.CLASSES == ["saudavel", "ferrugem", "bicho_mineiro", "phoma", "cercosporiose"]


def test_codigos_do_bracol_viram_as_classes_certas():
    assert bracol.PS_PARA_CLASSE == {
        0: "saudavel",
        1: "bicho_mineiro",
        2: "ferrugem",
        3: "phoma",
        4: "cercosporiose",
    }
    assert set(bracol.PS_PARA_CLASSE.values()) == set(bracol.CLASSES)
    for ps, nome in bracol.PS_PARA_CLASSE.items():
        assert bracol.classe_do_projeto(ps) == nome


def test_indice_da_classe_nao_e_o_codigo_original():
    # Armadilha: usar predominant_stress como índice troca bicho-mineiro e ferrugem.
    assert bracol.CLASSES.index(bracol.classe_do_projeto(1)) == 2
    assert bracol.CLASSES.index(bracol.classe_do_projeto(2)) == 1


def test_codigo_5_nao_tem_classe():
    assert bracol.classe_do_projeto(bracol.PS_INDETERMINADO) is None


@pytest.mark.parametrize("ps", [-1, 6, 7, "1", None])
def test_codigo_invalido_levanta_erro(ps):
    with pytest.raises(ValueError):
        bracol.classe_do_projeto(ps)


@pytest.mark.parametrize(
    ("ps", "presente", "esperado"),
    [
        (0, True, []),
        (2, True, []),
        (5, True, ["predominant_stress_5"]),
        (2, False, ["imagem_ausente"]),
        (5, False, ["imagem_ausente", "predominant_stress_5"]),
    ],
)
def test_motivos_de_exclusao(ps, presente, esperado):
    assert bracol.motivos_exclusao(ps, presente) == esperado


def test_motivos_de_exclusao_validam_o_codigo():
    with pytest.raises(ValueError):
        bracol.motivos_exclusao(9, True)


def test_copia_completa_nao_tem_ausentes_esperados():
    # Mudança intencional em 07/10/2026: a cópia completa tem todas as imagens. A cópia parcial
    # não tinha os 346 ids que vinham depois de "688" na ordem alfabética do zip truncado.
    assert bracol.IDS_AUSENTES_ESPERADOS == frozenset()


def test_caminhos_partem_da_raiz_do_repo():
    assert bracol.RAIZ_REPO.is_absolute()
    assert (bracol.RAIZ_REPO / "data" / "bracol.py").is_file()
    assert bracol.MANIFEST_BRACOL == bracol.RAIZ_REPO / "data" / "manifests" / "bracol.csv"
