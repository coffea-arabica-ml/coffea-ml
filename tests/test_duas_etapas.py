"""Testes do passo 5b: a regra em duas etapas (model/inferencia.py), a calibração do limiar de
doença e as análises dela (model/avaliar_pipeline.py) e a rotulagem de validação nas fotos de
treino do detector (evaluation/campo.py).

Sem treino, sem pesos e sem data/raw: probabilidades e tabelas pequenas, e o detector e o
classificador falsos de tests/test_inferencia.py. O teste que lê o manifest versionado do BRACOT
pula se ele faltar.
"""
import math

import numpy as np
import pandas as pd
import pytest

import avaliar_pipeline as ap
import campo
import inferencia as inf
from test_inferencia import deteccao, detectar_com, foto, modelos_falsos  # noqa: F401

DUAS = inf.Configuracao(decisao="duas_etapas", limiar_doenca=0.65)


def p(saudavel, ferrugem, bicho, phoma, cerco):
    return np.array([saudavel, ferrugem, bicho, phoma, cerco])


# ------------------------------------------------------------------ configuração
def test_padroes_da_regra_e_validacao():
    config = inf.Configuracao()
    assert (config.decisao, config.limiar_saudavel, config.limiar_doenca) == ("argmax", 0.083,
                                                                               None)
    assert inf.limiar_de_corte(config) == config.limiar_confianca
    assert inf.limiar_de_corte(DUAS) == 0.65
    with pytest.raises(ValueError):
        inf.Configuracao(decisao="duas_etapas")  # sem limiar de doença
    with pytest.raises(ValueError):
        inf.Configuracao(decisao="outra", limiar_doenca=0.5)


# ------------------------------------------------------------------- aplicar_decisao
def test_argmax_igual_ao_de_antes():
    probs = p(0.1, 0.6, 0.1, 0.1, 0.1)
    assert inf.aplicar_decisao(probs, inf.Configuracao()) == (1, 0.6, 0.6)


def test_primeira_etapa_saudavel_sem_corte_com_maior_ou_igual():
    classe, confianca, corte = inf.aplicar_decisao(p(0.083, 0.617, 0.1, 0.1, 0.1), DUAS)
    assert (classe, confianca, corte) == (inf.SAUDAVEL, 0.083, math.inf)
    assert inf.aplicar_decisao(p(0.0829, 0.6171, 0.1, 0.1, 0.1), DUAS)[0] != inf.SAUDAVEL


def test_segunda_etapa_renormaliza_entre_as_doencas():
    classe, confianca, corte = inf.aplicar_decisao(p(0.05, 0.5, 0.25, 0.15, 0.05), DUAS)
    assert classe == inf.CATEGORIAS.index("ferrugem")
    assert confianca == corte == pytest.approx(0.5 / 0.95)


def test_doenca_renormalizada_do_pipeline_e_a_mesma_da_inferencia():
    gerador = np.random.default_rng(0)
    probs = np.zeros((50, 5))
    probs[:, inf.SAUDAVEL] = gerador.uniform(0, 0.08, 50)  # todas vão para a 2ª etapa
    probs[:, list(inf.DOENCAS)] = (gerador.dirichlet(np.ones(4), size=50)
                                   * (1 - probs[:, [inf.SAUDAVEL]]))
    classes, renormalizadas = ap.doenca_renormalizada(probs)
    for linha, classe, r in zip(probs, classes, renormalizadas):
        assert inf.aplicar_decisao(linha, DUAS)[:2] == (classe, pytest.approx(r))


# ------------------------------------------------------------------------- contrato
def test_contrato_na_regra_em_duas_etapas(detectar_com):  # noqa: F811
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100), (400, 150, 560, 230)]
    detectar_com([deteccao(*folhas[0], 0.9), deteccao(*folhas[1], 0.8),
                  deteccao(*folhas[2], 0.7)])
    modelos = modelos_falsos([[0.10, 0.60, 0.10, 0.10, 0.10],   # saudável pela 1ª etapa
                              [0.02, 0.88, 0.04, 0.04, 0.02],   # ferrugem, renormalizada 0,90
                              [0.02, 0.40, 0.38, 0.10, 0.10]],  # renormalizada 0,41: sai
                             [3, 2, 2])
    resposta = inf.diagnosticar(foto(*folhas), DUAS, modelos)
    primeira, segunda = resposta["folhas"]
    assert (primeira["categoria"], primeira["severidade"]) == ("saudavel", "saudavel")
    assert primeira["confianca"] == pytest.approx(0.10)  # abaixo de 0,61 e fica
    assert segunda["categoria"] == "ferrugem"
    assert segunda["confianca"] == pytest.approx(0.88 / 0.98, abs=1e-4)
    detalhes = resposta["detalhes"]
    assert (detalhes["decisao"], detalhes["limiar_saudavel"], detalhes["limiar_doenca"],
            detalhes["abaixo_da_confianca"]) == ("duas_etapas", 0.083, 0.65, 1)
    assert "limiar_confianca" not in detalhes


def test_contrato_padrao_continua_no_argmax(detectar_com):  # noqa: F811
    folha = (200, 30, 330, 100)
    detectar_com([deteccao(*folha, 0.9), deteccao(20, 200, 140, 260, 0.8)])
    modelos = modelos_falsos([[0.10, 0.60, 0.10, 0.10, 0.10]] * 2, [3, 3])
    resposta = inf.diagnosticar(foto(folha, (20, 200, 140, 260)), inf.Configuracao(), modelos)
    assert resposta["status"] == "erro" and resposta["tipo"] == "baixa_confianca"
    assert resposta["detalhes"]["decisao"] == "argmax"


# --------------------------------------------------------------- calibração (BRACOL)
def test_curva_das_doencas_so_usa_as_doentes():
    probs = np.array([p(0.9, 0.04, 0.03, 0.02, 0.01), p(0.01, 0.9, 0.05, 0.02, 0.02),
                      p(0.01, 0.3, 0.5, 0.1, 0.09)])
    curva = ap.curva_das_doencas(probs, np.array([0, 1, 1]))
    assert curva[0]["mantidas"] == 2  # a saudável fica de fora
    assert curva[0]["acuracia"] == 0.5  # a segunda doente vai para bicho_mineiro


def test_efeito_do_limiar_saudavel_conta_doentes_e_saudaveis():
    probs = np.array([p(0.5, 0.5, 0, 0, 0), p(0.1, 0.9, 0, 0, 0), p(0.05, 0.95, 0, 0, 0),
                      p(0.9, 0.1, 0, 0, 0), p(0.3, 0.7, 0, 0, 0)])
    verdadeiras = np.array([1, 1, 1, 0, 0])
    r = ap.efeito_do_limiar_saudavel(probs, verdadeiras, 0.083)
    assert (r["argmax"]["saudavel_dado_d"]["sucessos"], r["argmax"]["recall_saudavel"]["sucessos"]
            ) == (1, 1)  # o empate 0,5 x 0,5 fica com saudável no argmax
    assert (r["limiar"]["saudavel_dado_d"]["sucessos"], r["limiar"]["recall_saudavel"]["sucessos"]
            ) == (2, 2)


def test_regra_completa_conta_a_saudavel_mandada_para_a_2a_etapa_como_erro():
    probs = np.array([p(0.05, 0.9, 0.03, 0.01, 0.01),   # saudável de verdade, vai para a doença
                      p(0.01, 0.95, 0.02, 0.01, 0.01),  # ferrugem certa
                      p(0.5, 0.5, 0, 0, 0)])            # ferrugem vista como saudável
    r = ap.regra_completa(probs, np.array([0, 1, 1]), DUAS)
    assert r["cobertura"] == 1.0 and r["acuracia_das_mantidas"]["sucessos"] == 1
    assert r["cobertura_saudaveis"] == 1.0


def test_sensibilidade_sem_as_3_s_nao_muda_o_limiar(tmp_path):
    linhas = [{"id": f"d{k}", "rotulo": "D", "p_saudavel_atual_B": v}
              for k, v in enumerate(np.linspace(0.0, 0.2, 40))]
    linhas += [{"id": i, "rotulo": "S", "p_saudavel_atual_B": 0.05} for i in ap.S_COM_MARCA]
    linhas += [{"id": f"s{k}", "rotulo": "S", "p_saudavel_atual_B": 0.5} for k in range(7)]
    caminho = tmp_path / "folhas.csv"
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    r = ap.sensibilidade_campo(0.083, caminho)
    todas, sem = r["82 folhas"], r["sem as 3 S com marca"]
    assert todas["limiar_recalculado"] == sem["limiar_recalculado"]
    assert (todas["S"], sem["S"], todas["D"], sem["D"]) == (10, 7, 40, 40)
    assert (todas["recall_s"]["sucessos"], sem["recall_s"]["sucessos"]) == (7, 7)
    assert ap.sensibilidade_campo(0.083, tmp_path / "nao_existe.csv") is None


def test_metricas_t1_aplicam_o_corte_pelo_valor_de_corte():
    tabela = pd.DataFrame({"classe": [0, 1, 1], "severidade": [0, 2, 2]})
    for v in ap.VARIANTES_T1:
        tabela[f"{v}_classe"] = [0, 1, 2]
        tabela[f"{v}_nivel"] = [0, 2, 2]
        tabela[f"{v}_confianca"] = [0.1, 0.9, 0.5]
        tabela[f"{v}_corte"] = [math.inf, 0.9, 0.5]
    r = ap.metricas_t1(tabela, 0.65)["cadeia_B"]
    assert (r["mantidas"], r["acuracia_no_limiar"]) == (2, 1.0)  # a saudável fica sem corte
    assert r["saudavel_dado_d"]["sucessos"] == 0 and r["recall_saudavel"]["sucessos"] == 1


def test_duas_etapas_tem_nome_proprio_e_limiar_de_doenca_so_com_ela():
    with pytest.raises(SystemExit):
        ap.main(["--limiar-doenca", "0.5", "--so-constantes"])
    assert ap.NOME_DUAS_ETAPAS != ap.NOME_PADRAO


# ------------------------------------------------------------- rotulagem de validação
def test_configuracao_da_rotulagem_de_validacao_fica_fixa():
    config = campo.CONFIG_ROTULAGEM_TREINO
    assert (config.recorte, config.fracao_visivel_minima) == ("B", 0.5)
    assert campo.N_ROTULAGEM_TREINO == 160
    assert campo.MANIFEST_CAMPO_TREINO != campo.MANIFEST_CAMPO


def test_rotulagem_de_validacao_nao_sobrescreve_csv_com_rotulos(tmp_path):
    caminho = tmp_path / "campo_rotulado_treino.csv"
    caminho.write_text("id,rotulo,classe\nx_01,S,\n", encoding="utf-8")
    with pytest.raises(campo.train.ErroFatal):
        campo.gerar_rotulagem(None, tmp_path / "pranchas", caminho,
                              campo.CONFIG_ROTULAGEM_TREINO, ([], pd.Series(dtype=str)),
                              campo.N_ROTULAGEM_TREINO)


@pytest.mark.skipif(not campo.bracot.MANIFEST_BRACOT.is_file(), reason="manifest ausente")
def test_fotos_de_treino_sao_as_194_fora_da_validacao_e_do_teste():
    treino, _ = campo.fotos_de_treino()
    validacao, _ = campo.fotos_de_validacao()
    manifest = campo.bracot.ler_manifest()
    teste = set(manifest.loc[manifest["split"] == "teste", "id"])
    assert len(treino) == 194 and not set(treino) & set(validacao) and not set(treino) & teste
    cenas = manifest.set_index("id")["cena"]
    assert not set(cenas[treino]) & set(cenas[validacao])
