"""Testes do plano de dobras do diagnóstico por blocos (evaluation/diagnostico_blocos.py).

Só funções puras: nenhum teste treina modelo, lê imagem ou abre o teste. O plano real usa só
os manifests versionados.
"""
import numpy as np
import pandas as pd
import pytest

import bracol
import diagnostico_blocos as diag


def _pool_sintetico() -> pd.DataFrame:
    """Ids 1 a 500 sem o 150 (fora das elegíveis), classes em faixas de 10 ids, e as imagens
    30 e 330 no mesmo grupo (a mesma folha em blocos diferentes: 0 e 3)."""
    ids = [i for i in range(1, 501) if i != 150]
    grupos = {i: f"g{i}" for i in ids}
    grupos[330] = "g30"
    return pd.DataFrame({
        "fonte": "bracol", "id": ids,
        "classe": [bracol.CLASSES[(i // 10) % 5] for i in ids],
        "grupo": [grupos[i] for i in ids],
    })


@pytest.fixture(scope="module")
def pool():
    return _pool_sintetico()


@pytest.fixture(scope="module")
def planos(pool):
    return diag.planejar(pool, k=5, bloco=100, purga=20)


def test_cada_imagem_fica_de_fora_uma_vez_por_protocolo(pool, planos):
    for dobras in planos.values():
        fora = np.sort(np.concatenate([d["fora"] for d in dobras]))
        assert fora.tolist() == sorted(pool["id"])
        assert 150 not in fora  # o id excluído não aparece
    # em blocos, cada dobra tem o bloco dela inteiro
    assert planos["blocos"][1]["fora"].tolist() == [i for i in range(101, 201) if i != 150]


def test_nenhum_id_do_treino_a_ate_20_ids_de_um_bloco_de_fora(planos):
    for dobra in planos["blocos"]:
        assert diag._distancia_aos_blocos(dobra["treino"], dobra["fora"]).min() > 20
    dobra_0 = planos["blocos"][0]  # bloco 1-100 de fora: 101 a 120 purgados, 121 no treino
    assert set(range(101, 121)) <= set(dobra_0["purga_distancia"])
    assert 121 in dobra_0["treino"]


def test_nenhum_grupo_nos_dois_lados(pool, planos):
    grupo = pool.set_index("id")["grupo"]
    for dobras in planos.values():
        for dobra in dobras:
            assert not set(grupo.loc[dobra["treino"]]) & set(grupo.loc[dobra["fora"]])
    # 30 (bloco 0) e 330 (bloco 3) são a mesma folha: quando um fica de fora, o outro sai do
    # treino pelo grupo (está longe demais para a purga por distância)
    assert 330 in planos["blocos"][0]["purga_grupo"]
    assert 30 in planos["blocos"][3]["purga_grupo"]


def test_treino_do_aleatorio_tem_o_mesmo_n_do_de_blocos(planos):
    for blocos, aleatorio in zip(planos["blocos"], planos["aleatorio"]):
        assert len(aleatorio["treino"]) == len(blocos["treino"])
        assert not set(aleatorio["treino"]) & set(aleatorio["fora"])
        assert len(aleatorio["descartada_sorteio"]) > 0


def test_conferir_plano_aceita_o_plano_e_recusa_uma_violacao(pool, planos):
    diag.conferir_plano(pool, planos, bloco=100, purga=20)
    violado = {p: [dict(d) for d in dobras] for p, dobras in planos.items()}
    dobra = violado["blocos"][0]
    dobra["treino"] = np.sort(np.append(dobra["treino"], 110))  # 110 está na zona de purga
    dobra["purga_distancia"] = dobra["purga_distancia"][dobra["purga_distancia"] != 110]
    with pytest.raises(ValueError):
        diag.conferir_plano(pool, violado, bloco=100, purga=20)


def test_plano_por_imagem_da_um_papel_por_dobra(pool, planos):
    plano = diag.plano_por_imagem(pool, planos, bloco=100)
    assert len(plano) == len(pool)
    for protocolo, dobras in planos.items():
        for d, dobra in enumerate(dobras):
            coluna = plano[f"{protocolo}_d{d}"]
            assert set(coluna) <= set(diag.PAPEIS)
            assert (coluna == "fora").sum() == len(dobra["fora"])
        assert plano[f"dobra_fora_{protocolo}"].between(0, 4).all()


def test_faixas_da_regra_de_leitura():
    assert diag.faixa_de_d(-2) == diag.faixa_de_d(3) == "pouco atalho de sessão detectado"
    assert diag.faixa_de_d(3.1) == diag.faixa_de_d(8) == "atenção"
    assert diag.faixa_de_d(8.1) == "atalho relevante"


def test_comparacao_pareada():
    base = {"id": [1, 2, 3, 4], "classe_verdadeira": ["phoma"] * 4}
    aleatorio = pd.DataFrame({**base,
                              "classe_prevista": ["phoma", "phoma", "ferrugem", "ferrugem"]})
    blocos = pd.DataFrame({**base, "classe_prevista": ["phoma", "ferrugem", "phoma", "ferrugem"]})
    assert diag.comparacao_pareada(aleatorio, blocos) == {
        "n": 4, "certas_nos_dois": 1, "so_no_aleatorio": 1, "so_em_blocos": 1,
        "erradas_nos_dois": 1}


def test_plano_real_passa_na_condicao_de_parada_so_com_k_5():
    # Decisão do gestor (09/10/2026): K=5, porque K=4 deixava saudavel com 47% no treino.
    pool = diag.montar_pool()  # treino + val, só dos manifests
    assert len(pool) == 1432
    planos = diag.planejar(pool)
    diag.conferir_plano(pool, planos)
    tabela = diag.tabela_das_dobras(pool, planos)
    menor = tabela.loc[tabela["menor_fracao_treino"].idxmin()]
    assert menor["menor_fracao_treino"] >= diag.LIMIAR_CLASSE
    assert menor["classe_menor_fracao"] == "cercosporiose"
    com_k_4 = diag.tabela_das_dobras(pool, diag.planejar(pool, k=4))
    assert com_k_4["menor_fracao_treino"].min() < diag.LIMIAR_CLASSE
