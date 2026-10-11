"""Testes do diagnóstico de campo do passo 5a (evaluation/campo_diagnostico.py): AUC com
bootstrap agrupado, ponto de operação com no máximo 5% das D marcadas saudáveis, sonda linear com
dobras agrupadas, controles e o efeito do limiar no BRACOL.

Sem pesos, sem GPU e sem data/raw: vetores e tabelas pequenos.
"""
import numpy as np
import pandas as pd

import campo_diagnostico as cd
import metrics


# --------------------------------------------------------------------------------- AUC
def test_auc_bootstrap_separacao_perfeita_e_deterministico():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    escore = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.95, 0.05])
    grupos = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    r = cd.auc_bootstrap(y, escore, grupos, n=200, seed=1)
    assert r["auc"] == 1.0 and r["ic95"] == [1.0, 1.0]
    assert r == cd.auc_bootstrap(y, escore, grupos, n=200, seed=1)


def test_auc_bootstrap_descarta_reamostras_com_uma_classe_so():
    # cada grupo tem uma classe só: sorteios só com "a" ou só com "b" não dão AUC
    y = np.array([1, 1, 0, 0])
    escore = np.array([0.9, 0.6, 0.4, 0.1])
    r = cd.auc_bootstrap(y, escore, np.array(["a", "a", "b", "b"]), n=400, seed=0)
    assert 0 < r["reamostras_validas"] < 400
    assert r["auc"] == 1.0


def test_auc_com_uma_classe_so_e_nan():
    assert np.isnan(cd.auc([1, 1], [0.2, 0.3]))


# ------------------------------------------------------------------ ponto de operação
def test_ponto_de_operacao_marca_no_maximo_5_por_cento_das_d():
    p_d = np.linspace(0.0, 0.5, 46)  # 46 D: floor(0,05 x 46) = 2 podem passar
    p_s = np.array([0.6, 0.49, 0.45, 0.1])
    y = np.r_[np.zeros(46, int), np.ones(4, int)]
    ponto = cd.ponto_de_operacao(y, np.r_[p_d, p_s])
    assert ponto["d_marcadas_saudaveis"] == 2 and ponto["n_d"] == 46
    assert ponto["limiar"] == np.sort(p_d)[-3]  # o terceiro maior P das D
    assert (ponto["acertos_s"], ponto["n_s"]) == (2, 4)  # 0,6 e 0,49 passam do limiar (~0,478)
    assert ponto["recall_s_ic95"] == list(metrics.intervalo_wilson(0.5, 4))


def test_ponto_de_operacao_com_poucas_d_nao_deixa_nenhuma_passar():
    y = np.array([0, 0, 0, 1, 1])  # 3 D: floor(0,15) = 0
    ponto = cd.ponto_de_operacao(y, np.array([0.2, 0.7, 0.4, 0.8, 0.5]))
    assert ponto["limiar"] == 0.7 and ponto["d_marcadas_saudaveis"] == 0
    assert ponto["acertos_s"] == 1


# --------------------------------------------------------------------------- sonda
def _por_grupo(n_grupos=10, por_grupo=6, dimensao=20, seed=0):
    """Rótulo definido pelo grupo e um vetor próprio de cada grupo nas features: só dá para
    acertar fora da dobra decorando o grupo."""
    gerador = np.random.default_rng(seed)
    grupos = np.repeat(np.arange(n_grupos), por_grupo)
    y = grupos % 2
    assinatura = gerador.normal(0, 3, (n_grupos, dimensao))
    x = assinatura[grupos] + gerador.normal(0, 0.1, (len(grupos), dimensao))
    return x, y, grupos


def test_sonda_separa_sinal_linear_verdadeiro():
    gerador = np.random.default_rng(0)
    y = np.r_[np.zeros(30, int), np.ones(30, int)]
    x = gerador.normal(0, 1, (60, 8))
    x[:, 0] += 4 * y
    escores = cd.sonda_linear(x, y, np.arange(60) % 10)
    assert not np.isnan(escores).any()
    assert cd.auc(y, escores) > 0.95


def test_sonda_agrupada_nao_aprende_a_identidade_do_grupo():
    # com um conjunto só, a AUC agrupada varia muito (é quase sorteio); a média de 5 não
    vazados, agrupados = [], []
    for seed in range(5):
        x, y, grupos = _por_grupo(seed=seed)
        vazados.append(cd.auc(y, cd.sonda_linear(x, y, np.arange(len(y)))))  # cada folha um grupo
        agrupados.append(cd.auc(y, cd.sonda_linear(x, y, grupos)))
    assert min(vazados) > 0.95
    assert np.mean(agrupados) < 0.7


# ------------------------------------------------------------------------ controles
def test_controles_por_sessao_e_pelo_filtro_de_fragmentos():
    tabela = pd.DataFrame({
        "foto": ["20190831_1", "20190831_2", "20191208_1", "20191208_2", "20191208_3",
                 "20191208_4"],
        "fracao_visivel": [1.0, 0.4, 0.9, 0.6, 0.3, 1.2],
        "p_saudavel_atual_B": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        "sonda_atual": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        "sonda_v0": [0.1, 0.9, 0.8, 0.2, 0.7, 0.3]})
    y = np.array([1, 0, 1, 0, 1, 0])
    r = cd.controles(tabela, y)
    assert set(r["subconjuntos"]) == {"sessão 20190831", "sessão 20191208",
                                      "ficam no filtro de fragmentos"}
    assert (r["subconjuntos"]["sessão 20190831"]["S"],
            r["subconjuntos"]["sessão 20190831"]["D"]) == (1, 1)
    filtro = r["subconjuntos"]["ficam no filtro de fragmentos"]
    assert (filtro["S"], filtro["D"]) == (2, 2)  # saem as de fração 0,4 e 0,3
    assert filtro["sonda do v0"] == 0.5


def test_efeito_no_bracol_conta_d_que_viram_saudaveis_pela_regra_do_limiar():
    predicoes = pd.DataFrame({
        "classe_verdadeira": ["saudavel", "saudavel", "ferrugem", "phoma", "phoma"],
        "classe_prevista": ["saudavel", "phoma", "ferrugem", "phoma", "phoma"],
        "prob_saudavel": [0.9, 0.2, 0.05, 0.12, 0.01]})
    r = cd.efeito_no_bracol(0.1, predicoes)
    assert (r["n_s"], r["n_d"]) == (2, 3)
    assert r["argmax"]["d_previstas_saudaveis"] == 0 and r["argmax"]["s_previstas_doentes"] == 1
    assert r["regra_limiar"]["d_previstas_saudaveis"] == 1  # a phoma com P(saudável) de 0,12
    assert r["regra_limiar"]["s_previstas_doentes"] == 0
