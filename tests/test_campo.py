"""Testes do passo 4 (classificador robusto ao campo): rotulagem cega, rótulos e proxies de
campo (evaluation/campo.py), filtro de fragmentos e deslocamento do recorte (model/inferencia.py)
e as peças do treino de campo (model/treinar_campo.py): máscaras, fundos, corte, erasing e o
dataset.

Sem treino, sem pesos e sem data/raw: imagens sintéticas e tabelas pequenas. Os dois testes que
leem arquivos versionados (manifest do BRACOT, config do diagnóstico) pulam se faltarem.
"""
import math

import cv2
import numpy as np
import pandas as pd
import pytest

import campo
import dados
import inferencia as inf
import metrics
import train
import treinar_campo as tc

VERDE = (40, 150, 40)


# ------------------------------------------------------------------------ rotulagem
def test_sorteio_da_rotulagem_espalha_pelas_fotos_e_e_deterministico():
    folhas = {"a": 5, "b": 2, "c": 3}
    sorteio = campo.sortear_rotulagem(folhas, n=6, seed=42)
    assert sorteio == campo.sortear_rotulagem(folhas, n=6, seed=42)
    assert pd.Series([f for f, _ in sorteio]).value_counts().to_dict() == {"a": 2, "b": 2, "c": 2}
    assert len(set(sorteio)) == 6 and all(1 <= k <= folhas[f] for f, k in sorteio)
    assert len(campo.sortear_rotulagem(folhas, n=50)) == 10  # não passa do total


def csv_de_rotulos(caminho, linhas, separador=",", codificacao="utf-8"):
    tabela = pd.DataFrame(linhas, columns=campo.COLUNAS_CAMPO).fillna("")
    tabela.to_csv(caminho, index=False, sep=separador, encoding=codificacao)


def linha(ident, rotulo="", classe="", caixa=(0.1, 0.1, 0.3, 0.3)):
    foto, folha = ident.rsplit("_", 1)
    return {"id": ident, "foto": foto, "folha": int(folha), "prancha": 1, "numero": 1,
            "score": 0.9, "caixa_x0": caixa[0], "caixa_y0": caixa[1], "caixa_x1": caixa[2],
            "caixa_y1": caixa[3], "rotulo": rotulo, "classe": classe, "observacao": ""}


def test_ler_rotulos_sem_rotulo_e_none_e_aceita_csv_do_excel(tmp_path):
    caminho = tmp_path / "campo.csv"
    csv_de_rotulos(caminho, [linha("f1_01"), linha("f1_02")])
    assert campo.ler_rotulos(caminho) is None
    assert campo.ler_rotulos(tmp_path / "nao_existe.csv") is None
    csv_de_rotulos(caminho, [linha("f1_01", "s"), linha("f1_02", "D", "Ferrugem"),
                             linha("f1_03")], separador=";", codificacao="cp1252")
    rotulos = campo.ler_rotulos(caminho)
    assert rotulos["rotulo"].tolist() == ["S", "D"] and rotulos["classe"].tolist() == ["",
                                                                                       "ferrugem"]
    csv_de_rotulos(caminho, [linha("f1_01", "X")])
    with pytest.raises(ValueError):
        campo.ler_rotulos(caminho)


def test_gerar_rotulagem_nao_sobrescreve_csv_com_rotulos(tmp_path):
    caminho = tmp_path / "campo.csv"
    csv_de_rotulos(caminho, [linha("f1_01", "S")])
    with pytest.raises(train.ErroFatal):
        campo.gerar_rotulagem(None, tmp_path / "pranchas", caminho)
    assert campo.ler_rotulos(caminho)["rotulo"].tolist() == ["S"]


def test_metricas_rotuladas_com_wilson_e_sem_os_nao_sei():
    tabela = pd.DataFrame({
        "rotulo": ["S", "S", "S", "S", "D", "D", "D", "?"],
        "classe": ["", "", "", "", "ferrugem", "phoma", "", ""],
        "prevista": ["saudavel", "ferrugem", "saudavel", "phoma", "ferrugem", "saudavel",
                     "ferrugem", "phoma"]})
    m = campo.metricas_rotuladas(tabela)
    assert (m["S"], m["D"], m["nao_sei"]) == (4, 3, 1)
    assert m["doenca_dado_S"]["valor"] == 0.5 and m["doenca_dado_S"]["n"] == 4
    assert m["doenca_dado_S"]["ic95"] == pytest.approx(list(metrics.intervalo_wilson(0.5, 4)))
    assert m["saudavel_dado_D"]["valor"] == pytest.approx(1 / 3)
    assert (m["classe_certa_dado_D"]["n"], m["classe_certa_dado_D"]["valor"]) == (2, 0.5)
    vazio = campo.metricas_rotuladas(tabela[tabela["rotulo"] == "?"])
    assert vazio["doenca_dado_S"] == {"n": 0, "valor": None, "ic95": [None, None]}


def test_casar_rotulos_confere_a_caixa():
    rotulos = pd.DataFrame([linha("f1_01", "S"), linha("f1_02", "D", caixa=(0.5, 0.5, 0.6, 0.6))])
    folhas = pd.DataFrame([{"id": "f1_01", "caixa_x0": 0.11, "caixa_y0": 0.1, "caixa_x1": 0.3,
                            "caixa_y1": 0.31},
                           {"id": "f1_02", "caixa_x0": 0.1, "caixa_y0": 0.1, "caixa_x1": 0.2,
                            "caixa_y1": 0.2}])  # a numeração mudou: outra folha
    assert campo.casar_rotulos(rotulos, folhas)["id"].tolist() == ["f1_01"]


# --------------------------------------------------------------- proxies e fragmentos
def test_resumir_proxies():
    folhas = pd.DataFrame({
        "borda": [True, False, False, False],
        "classe_A": ["ferrugem", "ferrugem", "ferrugem", "saudavel"],
        "classe_B": ["ferrugem", "phoma", "ferrugem", "saudavel"],
        "confianca_A": [0.5, 0.6, 0.7, 0.8], "confianca_B": [0.4] * 4,
        "severidade_A": ["baixa", "baixa", "muito_baixa", "saudavel"],
        "severidade_B": ["baixa"] * 4, "estavel_A": [1.0, 0.5, 1.0, 1.0],
        "estavel_B": [1.0] * 4})
    px = campo.resumir_proxies(folhas)
    assert px["concordancia_A_B"] == 0.75
    assert px["A"]["classe_mais_comum"] == 0.75 and px["A"]["fracao_saudavel"] == 0.25
    assert px["A"]["estabilidade_media"] == pytest.approx(0.875)
    assert px["A"]["estaveis_em_todas"] == 0.75
    assert px["A"]["borda"]["folhas"] == 1 and px["A"]["severidades"]["baixa"] == 2


def test_fracao_visivel_preenchimento_e_filtro_de_fragmentos():
    inteira = {"area": 0.04, "borda": False, "poligono": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    pedaco = {"area": 0.01, "borda": True, "poligono": [[0, 0], [10, 0], [0, 10]]}
    folhas = [inteira, pedaco]
    assert inf.fracao_visivel(pedaco, folhas) == pytest.approx(0.25)
    assert inf.preenchimento(inteira["poligono"]) == pytest.approx(1.0)
    assert inf.preenchimento(pedaco["poligono"]) == pytest.approx(0.5)
    assert inf.fragmento(pedaco, folhas, inf.Configuracao())  # ligado em 0,5 (11/10/2026)
    assert not inf.fragmento(pedaco, folhas, inf.Configuracao(fracao_visivel_minima=0.0))
    assert inf.fragmento(pedaco, folhas, inf.Configuracao(fracao_visivel_minima=0.3))
    assert inf.fragmento(pedaco, folhas, inf.Configuracao(preenchimento_minimo=0.6))
    tabela = pd.DataFrame({"id": ["a", "b"], "borda": [False, True],
                           "fracao_visivel": [1.0, 0.25], "preenchimento": [1.0, 0.5]})
    resumo = campo.resumir_fragmentos(tabela)
    corte = next(c for c in resumo["cortes"] if c["filtro"] == "fracao_visivel"
                 and c["limiar"] == 0.3)
    assert (corte["cortadas"], corte["cortadas_na_borda"]) == (1, 1)


def test_deslocamento_move_a_folha_e_a_mascara_juntas():
    imagem = np.full((600, 800, 3), 50, dtype=np.uint8)
    pontos = cv2.boxPoints(((400, 300), (300, 60), 0))
    cv2.fillPoly(imagem, [np.round(pontos).astype(np.int32)], VERDE)
    for modo in ("A", "B"):
        parado = inf.recortar_folha(imagem, pontos, modo, 0.1, 200)
        movido = inf.recortar_folha(imagem, pontos, modo, 0.1, 200, deslocamento=(0.05, 0.0))
        xs_parado = np.nonzero(np.all(parado == VERDE, axis=2))[1]
        xs_movido = np.nonzero(np.all(movido == VERDE, axis=2))[1]
        assert xs_movido.mean() - xs_parado.mean() == pytest.approx(0.05 * 360, abs=1.5)
    assert np.all(movido[90, 5] == 200)  # modo B: o lado de onde a folha saiu continua cinza
    assert campo.perturbar(np.full((2, 2, 3), 100, np.uint8), "brilho", 0.1)[0, 0, 0] == 110


# --------------------------------------------------------------------- treino de campo
def folha_sintetica(altura=224, largura=448):
    imagem = np.full((altura, largura, 3), 215, dtype=np.uint8)
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.ellipse(mascara, (largura // 2, altura // 2), (170, 60), 0, 0, 360, 1, -1)
    imagem[mascara.astype(bool)] = VERDE
    return imagem, mascara.astype(bool)


def test_segmentacao_simples_acha_a_folha_no_fundo_liso():
    imagem, mascara = folha_sintetica()
    imagem[100:110, 200:215] = (200, 200, 200)  # lesão clara no meio: o furo é preenchido
    assert tc.iou(tc.segmentar_fundo_liso(imagem), mascara) > 0.95


def test_escolher_mascara():
    _, folha = folha_sintetica()
    metade = folha.copy()
    metade[:, :224] = False
    assert tc.escolher_mascara(None, folha)[1] == "simples"
    escolhida, regra, _ = tc.escolher_mascara(folha, folha)
    assert regra == "uniao" and escolhida.sum() == folha.sum()
    assert tc.escolher_mascara(metade, folha)[1] == "simples"  # não concordam
    assert tc.escolher_mascara(None, np.zeros_like(folha))[0] is None  # implausível


def test_trocar_fundo_so_mexe_fora_da_mascara():
    imagem, mascara = folha_sintetica()
    fundo = np.full_like(imagem, 7)
    trocada = tc.trocar_fundo(imagem, mascara, fundo)
    assert np.array_equal(trocada[mascara], imagem[mascara])
    assert np.all(trocada[0, 0] == 7) and np.all(trocada[112, 224] == VERDE)


def test_jmuben_completado_ate_2_por_1_sem_distorcer():
    quadrado = np.zeros((128, 128, 3), dtype=np.uint8)
    quadrado[:, :64] = 255
    saida = tc.completar_2_por_1(quadrado, np.full((tc.ALTURA, tc.LARGURA, 3), 9, np.uint8))
    assert saida.shape == (224, 448, 3)
    colunas = np.nonzero(saida[112, :, 0] != 9)[0]
    assert (colunas.min(), colunas.max()) == (112, 335)  # 224 px de largura, centrado
    assert np.all(saida[112, 112:220] == 255) and np.all(saida[112, 228:336] == 0)


def test_corte_pela_borda_tira_no_maximo_15_por_cento_da_folha():
    imagem, mascara = folha_sintetica()
    gerador = np.random.default_rng(0)
    for _ in range(30):
        cortada = tc.cortar_pela_borda(imagem, mascara, gerador)
        restante = np.all(cortada == VERDE, axis=2).sum() / mascara.sum()
        assert 1 - tc.CORTE_MAXIMO - 0.02 <= restante <= 0.97
        assert np.all(cortada[~np.all(cortada == imagem, axis=2)] == tc.CINZA)


def test_apagar_faz_buracos_pequenos():
    imagem = np.zeros((224, 448, 3), dtype=np.uint8)
    gerador = np.random.default_rng(1)
    for _ in range(30):
        apagada = tc.apagar(imagem, gerador)
        mudou = np.any(apagada != 0, axis=2).mean()
        assert mudou <= tc.APAGAR_BURACOS * tc.APAGAR_AREA_MAXIMA + 0.01


def test_fundo_aleatorio_tem_o_tamanho_da_entrada():
    gerador = np.random.default_rng(2)
    banco = [np.full((224, 448, 3), 50, np.uint8)]
    for _ in range(20):
        assert tc.fundo_aleatorio(gerador, banco).shape == (224, 448, 3)
    assert tc.fundo_aleatorio(gerador, []).dtype == np.uint8


def dataset_sintetico(variante, monkeypatch, **probabilidades):
    for nome, valor in probabilidades.items():
        monkeypatch.setattr(tc, nome, valor)
    imagem, mascara = folha_sintetica()
    cache = tc.Cache()
    cache.imagens[("bracol", 1)] = imagem
    df = pd.DataFrame([{"fonte": "bracol", "id": 1, "caminho": "x.jpg", "classe": "ferrugem",
                        "severity": 2}])
    return tc.DatasetCampo(df, variante, cache, mascaras={1: mascara},
                           banco=[np.full((224, 448, 3), 60, np.uint8)])


def test_dataset_de_campo_tira_a_severidade_das_imagens_cortadas(monkeypatch):
    imagem, classe, severidade = dataset_sintetico("v0", monkeypatch)[0]
    assert tuple(imagem.shape) == (3, 224, 448) and (classe, severidade) == (1, 2)
    _, _, severidade = dataset_sintetico("v1g", monkeypatch, P_CORTE=1.0)[0]
    assert severidade == dados.SEM_SEVERIDADE
    _, _, severidade = dataset_sintetico("v1g", monkeypatch, P_CORTE=0.0, P_APAGAR=1.0)[0]
    assert severidade == dados.SEM_SEVERIDADE
    _, _, severidade = dataset_sintetico("v2", monkeypatch, P_CORTE=0.0, P_APAGAR=0.0,
                                         P_FUNDO=1.0)[0]
    assert severidade == 2  # a troca de fundo mantém a severidade


@pytest.mark.skipif(not (tc.PASTA_DIAGNOSTICO / "config.json").is_file(),
                    reason="diagnóstico por blocos ausente")
def test_receita_do_v0_e_a_da_validacao_cruzada_antiga():
    receita = tc.conferir_receita_v0()
    assert receita["igual"], receita["diferentes"]


@pytest.mark.skipif(not campo.bracot.MANIFEST_BRACOT.is_file(), reason="manifest ausente")
def test_fundos_vem_de_cenas_fora_da_validacao():
    fundos = tc.fotos_de_fundo()
    manifest = campo.bracot.ler_manifest(split="treino")
    validacao = set(campo.td.dividir_validacao(manifest))
    assert len(fundos) == 194 and not set(fundos["id"]) & validacao
    assert not set(fundos["cena"]) & set(manifest.loc[manifest["id"].isin(validacao), "cena"])


def test_variantes_e_teto_do_jmuben():
    assert set(tc.VARIANTES) == {"v0", "v1", "v1f", "v1g", "v2", "v3", "v3s"}
    assert tc.TETO_SEM_PHOMA_CERCOSPORIOSE == {"phoma": 0, "cercosporiose": 0}
    assert not any(tc.VARIANTES["v0"][k] for k in ("fotometrica", "corte", "apagar", "fundo"))
    assert math.isclose(tc.CORTE_MAXIMO, 0.15)


def test_metricas_rotuladas_separam_as_folhas_s_com_brilho():
    tabela = pd.DataFrame({
        "rotulo": ["S", "S", "S", "D"], "classe": ["", "", "", ""],
        "observacao": ["brilho forte", "clara; brilho", "", "lesao"],
        "prevista": ["ferrugem", "saudavel", "phoma", "ferrugem"]})
    m = campo.metricas_rotuladas(tabela)
    assert (m["doenca_dado_S_com_brilho"]["n"], m["doenca_dado_S_com_brilho"]["valor"]) == (2, 0.5)
    assert (m["doenca_dado_S_sem_brilho"]["n"], m["doenca_dado_S_sem_brilho"]["valor"]) == (1, 1.0)


def test_fragmento_nos_rotulos():
    casados = pd.DataFrame({"id": ["a", "b", "c"], "rotulo": ["?", "S", "?"], "classe": [""] * 3,
                            "observacao": ["fragmento", "Fragmento limpo", ""]})
    folhas = pd.DataFrame({"id": ["a", "b", "c"], "fracao_visivel": [0.2, 0.4, 1.0],
                           "borda": [True, True, False]})
    r = campo._fragmento_nos_rotulos(casados, folhas)
    assert (r["marcadas"], r["na_borda_entre_marcadas"]) == (2, 2)
    assert r["nao_sei_entre_marcadas"]["valor"] == 0.5 and r["nao_sei_entre_demais"]["valor"] == 1
    assert r["fracao_visivel_mediana_marcadas"] == pytest.approx(0.3)
