"""Testes das métricas (evaluation/metrics.py), com casos pequenos feitos à mão."""
import pytest

import bracol
import metrics

# 10 imagens das 5 classes, com 2 erros: uma saudável vista como ferrugem e um bicho-mineiro
# visto como phoma.
Y_TRUE = [0, 0, 0, 0, 1, 1, 2, 2, 3, 4]
Y_PRED = [0, 0, 0, 1, 1, 1, 2, 3, 3, 4]


def test_acuracia_e_metricas_por_classe_na_ordem_de_classes():
    m = metrics.calcular_metricas(Y_TRUE, Y_PRED)
    assert m["classes"] == bracol.CLASSES
    assert (m["n"], m["acertos"]) == (10, 8)
    assert m["acuracia"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx([3 / 4, 1, 1 / 2, 1, 1])
    assert m["precisao"] == pytest.approx([1, 2 / 3, 1, 1 / 2, 1])
    assert m["f1"] == pytest.approx([6 / 7, 4 / 5, 2 / 3, 2 / 3, 1])
    assert m["suporte"] == [4, 2, 2, 1, 1]


def test_f1_macro_e_a_media_simples_dos_f1():
    # (6/7 + 4/5 + 2/3 + 2/3 + 1) / 5 = 419/525
    assert metrics.calcular_metricas(Y_TRUE, Y_PRED)["f1_macro"] == pytest.approx(419 / 525)


def test_matriz_de_confusao_em_inteiros_linhas_verdadeiras():
    assert metrics.calcular_metricas(Y_TRUE, Y_PRED)["matriz_confusao"] == [
        [3, 1, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]


def test_classe_nunca_prevista_tem_precisao_zero():
    m = metrics.calcular_metricas([0, 1, 2, 3, 4], [0, 1, 2, 3, 3])
    assert m["precisao"][4] == 0 and m["recall"][4] == 0 and m["f1"][4] == 0


def test_indice_de_classe_invalido_levanta_erro():
    with pytest.raises(ValueError):
        metrics.calcular_metricas([0, 5], [0, 1])
    with pytest.raises(ValueError):
        metrics.calcular_metricas([0, 1], [0])


def test_wilson_de_90_por_cento_com_253_imagens():
    baixo, alto = metrics.intervalo_wilson(0.9, 253)
    assert (round(100 * baixo, 1), round(100 * alto, 1)) == (85.7, 93.1)


def test_wilson_nos_extremos_fica_entre_0_e_1():
    baixo, alto = metrics.intervalo_wilson(1.0, 22)
    assert alto == pytest.approx(1.0) and 0.8 < baixo < 0.9
    baixo, alto = metrics.intervalo_wilson(0.0, 22)
    assert baixo == pytest.approx(0.0) and 0.1 < alto < 0.2


def test_acuracia_vem_com_o_intervalo_de_wilson():
    m = metrics.calcular_metricas(Y_TRUE, Y_PRED)
    assert m["acuracia_ic95"] == pytest.approx(list(metrics.intervalo_wilson(0.8, 10)))


def test_severidade_ignora_linhas_sem_alvo():
    # Erros nas 5 linhas com alvo: 0, 1, 0, 2, 0. A última linha (-1) fica de fora.
    ms = metrics.calcular_metricas_severidade([0, 1, 2, 3, 4, -1], [0, 2, 2, 1, 4, 3])
    assert ms["n"] == 5
    assert ms["mae"] == pytest.approx(3 / 5)
    assert ms["acuracia_exata"] == pytest.approx(3 / 5)
    assert ms["acuracia_tolerancia_1"] == pytest.approx(4 / 5)
    assert ms["suporte"] == [1, 1, 1, 1, 1]


def test_kappa_quadratico_feito_a_mao():
    # Pesos (i - j)^2: discordância observada = 1 + 4 = 5; esperada = (30 + 15 + 2*10 + 30)/5
    # = 19 (níveis previstos: 0, 1, 2, 2 e 4). Kappa = 1 - 5/19 = 14/19.
    ms = metrics.calcular_metricas_severidade([0, 1, 2, 3, 4], [0, 2, 2, 1, 4])
    assert ms["kappa_quadratico"] == pytest.approx(14 / 19)


def test_kappa_perfeito_e_indefinido():
    perfeito = metrics.calcular_metricas_severidade([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert perfeito["kappa_quadratico"] == pytest.approx(1.0) and perfeito["mae"] == 0
    # Um só nível nas duas listas: o kappa não é definido.
    assert metrics.calcular_metricas_severidade([1, 1, 1], [1, 1, 1])["kappa_quadratico"] is None


def test_severidade_sem_nenhuma_linha_com_alvo():
    ms = metrics.calcular_metricas_severidade([-1, -1], [0, 3])
    assert ms["n"] == 0 and ms["mae"] is None and ms["kappa_quadratico"] is None
    assert ms["suporte"] == [0, 0, 0, 0, 0]


def test_trecho_do_relatorio_traz_as_metricas_principais():
    texto = "\n".join(metrics.trecho_relatorio(
        metrics.calcular_metricas(Y_TRUE, Y_PRED),
        metrics.calcular_metricas_severidade([0, 1, 2, 3, 4], [0, 2, 2, 1, 4]),
    ))
    assert "80,0% (8 de 10 imagens" in texto
    assert "kappa quadrático ponderado: 0,737" in texto
    assert "| cercosporiose | 1 | 100,0% | 100,0% | 100,0% |" in texto
