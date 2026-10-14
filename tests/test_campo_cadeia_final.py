"""Testes do passo 6b (evaluation/campo_cadeia_final.py): o destino de uma folha rotulada na
cadeia final, a regra do caminho em cada folha, com e sem o corte t_d, e os resumos por rótulo.
Sem pesos, sem GPU e sem data/raw: detecções sintéticas e tabelas pequenas.
"""
import pandas as pd
import pytest

import campo_cadeia_final as ccf
import inferencia as inf


def folha(x0, y0, x1, y1, score=0.9, borda=False):
    """Detecção já medida (como a de inferencia.detectar), numa foto de 300 x 600."""
    return {"poligono": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "caixa": [x0, y0, x1, y1],
            "score": score, "area": (x1 - x0) * (y1 - y0) / (300 * 600), "borda": borda,
            "colorida": 1.0}


def test_configuracoes_nao_mudam_a_cadeia_final():
    configs = ccf.configuracoes()
    assert configs["final"] == inf.Configuracao()
    sem = configs["sem_td"]
    assert sem.limiar_doenca == 0.0 and sem == inf.Configuracao(limiar_doenca=0.0)


def test_destino_classificada_fragmento_e_limite():
    grandes = [folha(20 + 110 * k, 40, 120 + 110 * k, 140, score=0.9 - 0.01 * k)
               for k in range(4)]
    pedaco = folha(0, 220, 30, 280, score=0.95, borda=True)  # 0,18 da mediana: fragmento
    deteccoes = [*grandes, pedaco]
    config = inf.Configuracao(max_folhas=3)
    preparo = inf.decidir(deteccoes, 300, 600, config)
    assert ccf.destino(grandes[0], deteccoes, preparo, config) == ("classificada", None)
    assert ccf.destino(pedaco, deteccoes, preparo, config)[0] == "fragmentos"
    motivo, detalhe = ccf.destino(grandes[3], deteccoes, preparo, config)
    assert motivo == "outro" and "max_folhas" in detalhe  # a 4ª de maior score fica de fora
    # com o filtro desligado (a cadeia da rotulagem antiga), o pedaço entra
    rotulagem = inf.Configuracao(max_folhas=3, fracao_visivel_minima=0.0)
    assert any(ccf._mesma(pedaco, f) for f in inf.decidir(deteccoes, 300, 600,
                                                           rotulagem)["folhas"])


def test_destino_quando_o_filtro_tira_todas():
    so_pedaco = folha(0, 220, 30, 280, borda=True)
    preparo = {"plano": "fragmentos", "folhas": []}
    assert ccf.destino(so_pedaco, [so_pedaco], preparo, inf.Configuracao())[0] == "fragmentos"


@pytest.mark.parametrize("probs, final, sem_td", [
    ([0.10, 0.60, 0.10, 0.10, 0.10], ("saudavel", None), ("saudavel", None)),
    ([0.02, 0.88, 0.04, 0.04, 0.02], ("doenca", None), ("doenca", None)),
    ([0.02, 0.40, 0.38, 0.10, 0.10], ("descartada", "corte_td"), ("doenca", None)),
])
def test_decidir_folha_na_deteccao_com_e_sem_td(probs, final, sem_td):
    configs = ccf.configuracoes()
    for cenario, esperado in (("final", final), ("sem_td", sem_td)):
        r = ccf.decidir_folha(probs, "deteccao", configs[cenario])
        assert (r["desfecho"], r["motivo"]) == esperado and r["regra"] == "duas_etapas"
    r = ccf.decidir_folha([0.02, 0.40, 0.38, 0.10, 0.10], "deteccao", configs["final"])
    assert r["classe"] == "ferrugem" and r["confianca"] == pytest.approx(0.40 / 0.98)


def test_decidir_folha_na_foto_inteira_fica_no_argmax():
    configs = ccf.configuracoes()
    for cenario in ccf.CENARIOS:  # o t_d não vale na foto inteira
        r = ccf.decidir_folha([0.10, 0.60, 0.10, 0.10, 0.10], "foto_inteira", configs[cenario])
        assert (r["desfecho"], r["motivo"], r["regra"]) == ("descartada", "outro", "argmax")
        assert "0,61" in r["detalhe"]


def tabela():
    """6 folhas: 2 S, 3 D e 1 "?", com os desfechos dos dois cenários."""
    return pd.DataFrame({
        "conjunto": ["antigas"] * 3 + ["novas"] * 3,
        "rotulo": ["S", "S", "D", "D", "D", "?"],
        "desfecho_final": ["saudavel", "descartada", "doenca", "descartada", "saudavel",
                           "descartada"],
        "motivo_final": [None, "fragmentos", None, "corte_td", None, "outro"],
        "classe_final": ["saudavel", None, "ferrugem", "phoma", "saudavel", None],
        "confianca_final": [0.2, None, 0.9, 0.5, 0.1, None],
        "desfecho_sem_td": ["saudavel", "descartada", "doenca", "doenca", "saudavel",
                            "descartada"],
        "motivo_sem_td": [None, "fragmentos", None, None, None, "outro"],
        "classe_sem_td": ["saudavel", None, "ferrugem", "phoma", "saudavel", None],
        "confianca_sem_td": [0.2, None, 0.9, 0.5, 0.1, None]})


def test_resumo_por_rotulo_soma_os_desfechos_e_os_motivos():
    r = ccf.resumo_por_rotulo(tabela(), "final")
    d = r["D"]
    assert d["n"] == 3
    assert [d[x]["sucessos"] for x in ccf.DESFECHOS] == [1, 1, 1]
    assert sum(m["sucessos"] for m in d["motivos"].values()) == d["descartada"]["sucessos"]
    assert d["motivos"]["corte_td"]["sucessos"] == 1 and d["classes_doentes"]["ferrugem"] == 1
    assert r["S"]["motivos"]["fragmentos"]["sucessos"] == 1
    sem = ccf.resumo_por_rotulo(tabela(), "sem_td")["D"]
    assert (sem["doenca"]["sucessos"], sem["classes_doentes"]["phoma"]) == (2, 1)


def test_doentes_mantidas_classes_e_confianca():
    r = ccf.doentes_mantidas(tabela(), "sem_td")
    todas = r["todas"]
    assert todas["n"] == 2 and todas["classes"]["phoma"]["sucessos"] == 1
    assert todas["confianca"]["mediana"] == pytest.approx(0.7)
    assert todas["confianca_por_classe"]["bicho_mineiro"] is None
    assert r["S"]["n"] == 0 and r["S"]["confianca"] is None


def test_analisar_separa_e_junta_os_conjuntos():
    r = ccf.analisar(tabela().assign(destino="classificada"))
    assert r["juntas"]["contagens"] == {"S": 2, "D": 3, "?": 1}
    assert r["antigas"]["final"]["rotulos"]["D"]["n"] == 1
    assert r["novas"]["sem_td"]["rotulos"]["D"]["doenca"]["sucessos"] == 1


def test_conferir_com_a_validacao(tmp_path):
    folhas = pd.DataFrame({"conjunto": ["novas", "novas"], "id": ["a_01", "b_01"],
                           "destino": ["classificada", "fragmentos"],
                           "plano_final": ["deteccao", "deteccao"],
                           "desfecho_final": ["saudavel", "descartada"],
                           **{f"p_{c}": [0.1, None] for c in ccf.DOENCAS}})
    antes = pd.DataFrame({"conjunto": ["novas"], "id": ["a_01"],
                          "desfecho_duas_etapas": ["saudavel"],
                          **{f"p_{c}": [0.1001] for c in ccf.DOENCAS}})
    caminho = tmp_path / "folhas.csv"
    antes.to_csv(caminho, index=False)
    r = ccf.conferir(folhas, caminho)
    assert (r["folhas"], r["desfechos_diferentes"], r["sem_par"]) == (1, 0, 0)
    assert r["diferenca_maxima"] == pytest.approx(1e-4)
    assert ccf.conferir(folhas, tmp_path / "nao_existe.csv") is None


def test_nada_le_o_teste():
    fonte = (ccf.RAIZ_REPO / "evaluation" / "campo_cadeia_final.py").read_text(encoding="utf-8")
    assert "montar_teste" not in fonte and '"teste"' not in fonte
