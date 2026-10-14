"""Testes do passo 6, a cadeia final (decisões do gestor, 11/10/2026): a regra de decisão por
caminho em model/inferencia.py (as duas etapas só nos recortes por detecção no modo B; o argmax com
corte 0,61 na foto inteira), as análises por caminho de model/avaliar_pipeline.py e o manifesto
dos pesos finais (model/pesos/MANIFESTO.md).

Sem treino e sem data/raw: o detector e o classificador falsos de tests/test_inferencia.py e
tabelas pequenas. A conferência do SHA-256 dos pesos pula se eles não estiverem na máquina.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

import avaliar_pipeline as ap
import detector
import inferencia as inf
from test_inferencia import deteccao, detectar_com, foto, modelos_falsos  # noqa: F401

MANIFESTO = inf.RAIZ_REPO / "model" / "pesos" / "MANIFESTO.md"
# P(saudável) = 0,30: as duas etapas chamariam saudável; o argmax fica com ferrugem (0,70)
FERRUGEM_FORTE = [0.30, 0.70, 0.0, 0.0, 0.0]
# P(saudável) = 0,10, ferrugem 0,60: saudável nas duas etapas; abaixo de 0,61 no argmax
SAUDAVEL_FRACA = [0.10, 0.60, 0.10, 0.10, 0.10]


# ------------------------------------------------------------------ regra do caminho
def test_decisao_do_caminho():
    final = inf.Configuracao()
    assert final.decisao == "por_caminho" and final.recorte == "B"
    assert inf.decisao_do_caminho(final, "deteccao") == "duas_etapas"
    assert inf.decisao_do_caminho(final, "foto_inteira") == "argmax"
    assert inf.decisao_do_caminho(final, "deteccao", "A") == "argmax"  # t_s só foi fixado no B
    assert inf.decisao_do_caminho(inf.Configuracao(recorte="A"), "deteccao") == "argmax"
    for decisao in ("argmax", "duas_etapas"):  # os padrões anteriores valem em todo caminho
        config = inf.Configuracao(decisao=decisao)
        assert {inf.decisao_do_caminho(config, p) for p in ("deteccao", "foto_inteira")} == {
            decisao}
    resolvida = inf.config_do_caminho(final, "deteccao")
    assert resolvida.decisao == "duas_etapas" and inf.limiar_de_corte(resolvida) == 0.65
    assert inf.limiar_de_corte(inf.config_do_caminho(final, "foto_inteira")) == 0.61


# ----------------------------------------------------------------- caminho 1: foto inteira
@pytest.mark.parametrize("score", [0.8, 0.1])  # regra 1 (folha grande) e regra 2 (fraca)
def test_foto_inteira_fica_no_argmax_com_corte_061(detectar_com, score):  # noqa: F811
    folha = (100, 50, 500, 250) if score == 0.8 else (250, 100, 350, 160)
    detectar_com([deteccao(*folha, score)])
    resposta = inf.diagnosticar(foto(folha), inf.Configuracao(),
                                modelos_falsos([FERRUGEM_FORTE], [2]))
    unica, = resposta["folhas"]
    assert (unica["categoria"], unica["confianca"]) == ("ferrugem", 0.7)  # não vira saudável
    detalhes = resposta["detalhes"]
    assert (detalhes["plano"], detalhes["regra"]) == ("foto_inteira", 1 if score == 0.8 else 2)
    assert (detalhes["decisao"], detalhes["decisao_configurada"],
            detalhes["limiar_confianca"]) == ("argmax", "por_caminho", 0.61)
    assert "limiar_saudavel" not in detalhes


def test_foto_inteira_abaixo_de_061_e_baixa_confianca(detectar_com):  # noqa: F811
    folha = (100, 50, 500, 250)
    detectar_com([deteccao(*folha, 0.8)])
    resposta = inf.diagnosticar(foto(folha), inf.Configuracao(),
                                modelos_falsos([SAUDAVEL_FRACA], [2]))
    assert (resposta["status"], resposta["tipo"]) == ("erro", "baixa_confianca")
    assert resposta["detalhes"]["plano"] == "foto_inteira"


# ------------------------------------------------------- caminho 2: recortes por detecção
def test_recortes_por_deteccao_usam_as_duas_etapas(detectar_com):  # noqa: F811
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100)]
    detectar_com([deteccao(*folhas[0], 0.9), deteccao(*folhas[1], 0.8)])
    modelos = modelos_falsos([[0.02, 0.88, 0.04, 0.04, 0.02],   # ferrugem, renormalizada 0,90
                              [0.02, 0.40, 0.38, 0.10, 0.10]],  # renormalizada 0,41: sai
                             [3, 2])
    resposta = inf.diagnosticar(foto(*folhas), inf.Configuracao(), modelos)
    unica, = resposta["folhas"]
    assert unica["categoria"] == "ferrugem"
    assert unica["confianca"] == pytest.approx(0.88 / 0.98, abs=1e-4)
    detalhes = resposta["detalhes"]
    assert (detalhes["plano"], detalhes["decisao"], detalhes["decisao_configurada"],
            detalhes["limiar_saudavel"], detalhes["limiar_doenca"],
            detalhes["abaixo_da_confianca"]) == ("deteccao", "duas_etapas", "por_caminho", 0.083,
                                                 0.65, 1)
    assert "limiar_confianca" not in detalhes


def test_recorte_por_deteccao_vira_saudavel_abaixo_de_061(detectar_com):  # noqa: F811
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100)]
    detectar_com([deteccao(*f, 0.9) for f in folhas])
    probabilidades = [SAUDAVEL_FRACA, [0.20, 0.50, 0.10, 0.10, 0.10]]
    resposta = inf.diagnosticar(foto(*folhas), inf.Configuracao(),
                                modelos_falsos(probabilidades, [3, 3]))
    assert resposta["status"] == "sucesso"
    primeira, segunda = resposta["folhas"]
    for folha, p_saudavel in ((primeira, 0.10), (segunda, 0.20)):
        assert (folha["categoria"], folha["severidade"]) == ("saudavel", "saudavel")
        assert folha["confianca"] == pytest.approx(p_saudavel)  # P(saudável), sem corte
    # no argmax antigo, as mesmas folhas ficariam abaixo de 0,61 e a foto, em baixa_confianca
    antiga = inf.diagnosticar(foto(*folhas), inf.Configuracao(decisao="argmax"),
                              modelos_falsos(probabilidades, [3, 3]))
    assert antiga["tipo"] == "baixa_confianca"


def test_modo_a_fica_no_argmax(detectar_com):  # noqa: F811
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100)]
    detectar_com([deteccao(*f, 0.9) for f in folhas])
    resposta = inf.diagnosticar(foto(*folhas), inf.Configuracao(recorte="A"),
                                modelos_falsos([SAUDAVEL_FRACA] * 2, [3, 3]))
    assert resposta["tipo"] == "baixa_confianca"
    assert resposta["detalhes"]["decisao"] == "argmax"


# ------------------------------------------------------------ análises do avaliar_pipeline
def tabela_t1():
    """4 imagens: 2 pela foto inteira, 1 pela detecção e 1 sem folha (regra 3)."""
    return pd.DataFrame({
        "plano": ["foto_inteira", "foto_inteira", "deteccao", "nenhuma"],
        "classe": [inf.SAUDAVEL, 1, inf.SAUDAVEL, 1],
        "cadeia_B_probs": [np.array(SAUDAVEL_FRACA), np.array(FERRUGEM_FORTE),
                           np.array(SAUDAVEL_FRACA), None]})


def test_t1_por_caminho_aplica_a_regra_de_cada_caminho():
    r = ap.t1_por_caminho(tabela_t1(), inf.Configuracao())
    assert set(r) == {"argmax", "duas_etapas", "por_caminho"} and set(r["argmax"]) == set(
        ap.CAMINHOS)
    final, argmax, duas = r["por_caminho"], r["argmax"], r["duas_etapas"]
    assert final["foto_inteira"] == argmax["foto_inteira"]
    assert final["deteccao"] == duas["deteccao"]
    assert final["deteccao"]["saudavel_dado_s"]["sucessos"] == 1  # a saudável fraca fica
    assert argmax["deteccao"]["cobertura"]["sucessos"] == 0  # 0,60 < 0,61
    assert duas["foto_inteira"]["saudavel_dado_d"]["sucessos"] == 1  # ferrugem com P(s) 0,30
    todas = final["todas"]
    assert (todas["imagens"], todas["com_previsao"], todas["cobertura"]["sucessos"]) == (4, 3, 2)
    assert todas["acuracia_das_mantidas"]["p"] == 1.0


def test_t2_por_regra_conta_folhas_e_fotos():
    folhas = pd.DataFrame({"foto": ["a", "a", "b"], "plano": ["deteccao"] * 3,
                           **{f"prob_{c}_B": v for c, v in zip(
                               inf.CATEGORIAS, np.array([SAUDAVEL_FRACA, FERRUGEM_FORTE,
                                                         [0.02, 0.40, 0.38, 0.1, 0.1]]).T)}})
    fotos = pd.DataFrame({"foto": ["a", "b", "c", "d"],
                          "plano": ["deteccao", "deteccao", "fragmentos", "nenhuma"]})
    r = ap.t2_por_regra(folhas, fotos, inf.Configuracao())
    assert r["por_caminho"] == r["duas_etapas"]  # no T2, todas as folhas vêm da detecção
    duas, argmax = r["duas_etapas"], r["argmax"]
    assert duas["desfechos"] == {"saudavel": 2, "doenca": 0, "descartada": 1}
    assert argmax["desfechos"] == {"saudavel": 0, "doenca": 1, "descartada": 2}
    assert argmax["classes_mantidas"]["ferrugem"] == 1
    assert duas["fotos"] == {"total": 4, "com_resposta": 1, "com_doenca": 0,
                             "baixa_confianca_corte": 1, "baixa_confianca_fragmentos": 1,
                             "planta_nao_identificada": 1}


def test_comparar_com_runs_confere_o_recalculado(tmp_path):
    def gravar(nome, decisao, limiar_chave, t2_baixa):
        pasta = tmp_path / nome
        pasta.mkdir()
        configuracao = {"limiar_confianca": 0.61, "decisao": decisao}
        if decisao == "duas_etapas":
            configuracao["limiar_doenca"] = 0.65
        (pasta / "config.json").write_text(json.dumps(
            {"criado_em": "2026-10-11T10:00:00", "configuracao": configuracao}), encoding="utf-8")
        (pasta / "metricas.json").write_text(json.dumps({
            "t1": {"variantes": {"cadeia_B": {"metricas": {"acuracia": 2 / 3}, "mantidas": 2,
                                              "acuracia_no_limiar": 1.0}}},
            "t2": {"folhas": 3, "fotos_baixa_confianca": {"B": {limiar_chave: t2_baixa}}},
            "t3": {"mediana": {"total_s": 0.9}, "maximo": {"total_s": 1.5}}}), encoding="utf-8")

    gravar("pipeline_padrao_b", "argmax", "0.61", 1)
    gravar("pipeline_duas_etapas", "duas_etapas", "0.65", 0)
    t1r = {regra: {"todas": {"acuracia_sem_corte": 2 / 3, "cobertura": {"sucessos": 2},
                             "acuracia_das_mantidas": {"p": 1.0}}}
           for regra in ("argmax", "duas_etapas")}
    t2r = {"argmax": {"folhas": 3, "fotos": {"baixa_confianca_corte": 1}},
           "duas_etapas": {"folhas": 3, "fotos": {"baixa_confianca_corte": 1}}}
    r = ap.comparar_com_runs(t1r, t2r, tmp_path)
    assert r["pipeline_padrao_b"]["confere"]
    assert not r["pipeline_duas_etapas"]["confere"]  # 1 foto recalculada x 0 gravada
    assert r["pipeline_padrao_b"]["gravado"]["t3_mediana_s"] == 0.9
    assert ap.comparar_com_runs(t1r, t2r, tmp_path / "vazia") == {
        "pipeline_padrao_b": None, "pipeline_duas_etapas": None}


def test_limitacoes_de_campo_contam_as_interrogacoes_chamadas_saudaveis(tmp_path):
    def regra(s, n_s, d, n_d):
        return {"duas_etapas": {"saudavel_dado_s": ap._proporcao(s, n_s),
                                "saudavel_dado_d": ap._proporcao(d, n_d)}}

    conjunto = {"contagens": {"S": 10, "D": 20, "?": 8},
                "por_conjunto": {"regras": regra(8, 10, 1, 20)},
                "limites": {"sem_interrogacao": {"regras": regra(8, 10, 1, 20)},
                            "interrogacao_como_D": {"regras": regra(8, 10, 6, 28)}}}
    v = {"limiar_saudavel": 0.083, "limiar_doenca_proposto": 0.65,
         **{chave: {c: conjunto[chave] for c in ("antigas", "novas", "juntas")}
            for chave in ("contagens", "por_conjunto", "limites")}}
    caminho = tmp_path / "validacao.json"
    caminho.write_text(json.dumps(v), encoding="utf-8")
    r = ap.limitacoes_de_campo(caminho)
    interrogacoes = r["novas"]["interrogacao_chamada_saudavel"]
    assert (interrogacoes["sucessos"], interrogacoes["n"]) == (5, 8)  # 6 - 1 de 28 - 20
    assert r["juntas"]["saudavel_dado_d"]["sucessos"] == 1 and r["novas"]["?"] == 8
    assert ap.limitacoes_de_campo(tmp_path / "nao_existe.json") is None


# ------------------------------------------------------------------- pesos finais
def pesos_do_manifesto() -> dict:
    """{caminho relativo: (bytes, sha256)} das linhas da tabela do MANIFESTO.md."""
    texto = MANIFESTO.read_text(encoding="utf-8")
    return {caminho: (int(tamanho.replace(".", "")), sha)
            for caminho, tamanho, sha in re.findall(
                r"\|\s*`([^`]+\.pt)`\s*\|\s*([\d.]+)\s*\|\s*`([0-9a-f]{64})`", texto)}


def test_manifesto_lista_os_pesos_que_a_cadeia_carrega():
    pesos = pesos_do_manifesto()
    esperados = {(inf.PASTA_CLASSIFICADOR / "melhor.pt").relative_to(inf.RAIZ_REPO).as_posix(),
                 detector.PESOS_PADRAO.relative_to(inf.RAIZ_REPO).as_posix()}
    assert esperados <= set(pesos)


@pytest.mark.parametrize("arquivo", [inf.PASTA_CLASSIFICADOR / "melhor.pt",
                                     detector.PESOS_PADRAO])
def test_pesos_locais_batem_com_o_manifesto(arquivo):
    if not arquivo.is_file():
        pytest.skip("pesos fora desta máquina")
    tamanho, sha = pesos_do_manifesto()[arquivo.relative_to(inf.RAIZ_REPO).as_posix()]
    assert arquivo.stat().st_size == tamanho
    assert hashlib.sha256(arquivo.read_bytes()).hexdigest() == sha
