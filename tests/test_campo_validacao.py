"""Testes da validação de campo da regra em duas etapas (evaluation/campo_validacao.py): desfechos
das regras, proporções com Wilson, cenários dos "?", curva do limiar de saudável e tipos de
folha. Sem pesos, sem GPU e sem data/raw: tabelas pequenas.
"""
import json

import numpy as np
import pandas as pd
import pytest

import campo_validacao as cv
import inferencia as inf
import metrics


def tabela():
    """6 folhas: 2 S, 2 D e 2 "?", com P(saudável) e desfechos prontos."""
    return pd.DataFrame({
        "foto": [f"f{k}" for k in range(6)],
        "rotulo": ["S", "S", "D", "D", "?", "?"],
        "p_saudavel": [0.5, 0.05, 0.1, 0.01, 0.3, 0.02],
        "observacao": ["brilho", "", "fragmento", "", "fragmento; brilho", ""],
        "borda": [False, True, False, False, True, False],
        "fracao_visivel": [1.0, 0.4, 1.0, 1.0, 0.8, 1.0],
        "desfecho_duas_etapas": ["saudavel", "descartada", "saudavel", "doenca", "saudavel",
                                 "doenca"],
        "desfecho_argmax_061": ["descartada"] * 3 + ["doenca", "descartada", "doenca"]})


def test_desfecho_das_regras():
    duas = inf.Configuracao(decisao="duas_etapas", limiar_doenca=0.65)
    assert cv.desfecho([0.1, 0.6, 0.1, 0.1, 0.1], duas) == "saudavel"  # P(saudável) >= 0,083
    assert cv.desfecho([0.02, 0.4, 0.38, 0.1, 0.1], duas) == "descartada"  # 0,41 < 0,65
    assert cv.desfecho([0.02, 0.88, 0.04, 0.04, 0.02], duas) == "doenca"
    assert cv.desfecho([0.5, 0.2, 0.1, 0.1, 0.1], inf.Configuracao()) == "descartada"
    assert cv.desfecho([0.5, 0.2, 0.1, 0.1, 0.1],
                       inf.Configuracao(limiar_confianca=0.0)) == "saudavel"


def test_regras_comparadas_usam_os_limiares_fixados():
    r = cv.regras(0.65)
    assert r["duas_etapas"].limiar_saudavel == 0.083 == cv.LIMIAR_SAUDAVEL
    assert r["duas_etapas"].limiar_doenca == 0.65
    assert r["argmax_061"].limiar_confianca == 0.61 and r["argmax_061"].decisao == "argmax"


def test_medir_conta_saudaveis_e_descartadas_com_wilson():
    f = cv.com_rotulos(tabela(), "sem_interrogacao")
    r = cv.medir(f, ["duas_etapas", "argmax_061"])
    assert (r["S"], r["D"]) == (2, 2)
    duas = r["regras"]["duas_etapas"]
    assert (duas["saudavel_dado_s"]["sucessos"], duas["saudavel_dado_d"]["sucessos"]) == (1, 1)
    assert duas["descartada_dado_s"]["sucessos"] == 1
    assert duas["saudavel_dado_s"]["ic95"] == list(metrics.intervalo_wilson(0.5, 2))
    assert r["regras"]["argmax_061"]["saudavel_dado_s"]["sucessos"] == 0
    assert r["auc"]["auc"] == pytest.approx(0.75)


def test_cenarios_dos_interrogacoes():
    assert len(cv.com_rotulos(tabela(), "sem_interrogacao")) == 4
    como_s = cv.com_rotulos(tabela(), "interrogacao_como_S")
    como_d = cv.com_rotulos(tabela(), "interrogacao_como_D")
    assert (int(como_s["y"].sum()), int(como_d["y"].sum())) == (4, 2)
    assert len(como_s) == len(como_d) == 6


def test_sensibilidade_e_so_informacao_em_limiares_fixos():
    f = cv.com_rotulos(tabela(), "sem_interrogacao")
    pontos = cv.sensibilidade(f, (0.01, 0.083, 0.2))
    assert [p["limiar"] for p in pontos] == [0.01, 0.083, 0.2]
    assert [p["recall_s"]["sucessos"] for p in pontos] == [2, 1, 1]
    assert [p["falso_saudavel_d"]["sucessos"] for p in pontos] == [2, 1, 0]


def test_tipos_de_folha_e_fracao_de_interrogacoes():
    tipos = cv.tipos_de_folha(tabela())
    assert tipos["fragmento (observação)"].tolist() == [False, False, True, False, True, False]
    assert tipos["fração visível < 0,5 (cadeia)"].sum() == 1
    assert tipos["brilho (observação)"].sum() == 2
    assert tipos["nenhum desses"].tolist() == [False, False, False, True, False, True]
    r = cv.interrogacoes_por_tipo(tabela())
    assert (r["todas"]["sucessos"], r["todas"]["n"]) == (2, 6)
    assert (r["fragmento (observação)"]["sucessos"], r["fragmento (observação)"]["n"]) == (1, 2)
    assert set(r) == set(cv.TIPOS)


def test_limiar_de_doenca_vem_do_run_e_falha_sem_ele(tmp_path):
    with pytest.raises(cv.train.ErroFatal):
        cv.limiar_doenca_proposto(tmp_path)
    (tmp_path / "metricas.json").write_text(json.dumps({"duas_etapas": {"proposta": 0.65}}),
                                            encoding="utf-8")
    assert cv.limiar_doenca_proposto(tmp_path) == 0.65


def test_analisar_separa_os_conjuntos():
    f = pd.concat([tabela().assign(conjunto="antigas"),
                   tabela().assign(conjunto="novas", foto=lambda t: t["foto"] + "n")],
                  ignore_index=True)
    f["desfecho_argmax_sem_corte"] = "doenca"
    r = cv.analisar(f, 0.65)
    assert r["contagens"]["juntas"] == {"S": 4, "D": 4, "?": 4}
    assert r["por_conjunto"]["novas"]["S"] == 2 and r["por_conjunto"]["juntas"]["S"] == 4
    assert set(r["limites"]["novas"]) == set(cv.CENARIOS)
    assert np.isclose(r["limiar_saudavel"], 0.083)
