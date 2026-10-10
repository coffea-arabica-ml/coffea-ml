"""Testes da checagem de atalho (evaluation/atalho.py e evaluation/gradcam.py).

Usam dados sintéticos e o modelo sem pré-treino: nenhum teste lê data/raw, abre o run nem
baixa pesos.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import atalho
import bracol
import gradcam
import train


# -------------------------------------------------------------------------- Grad-CAM
def test_gerar_gradcam_devolve_mapa_do_tamanho_da_entrada():
    torch.manual_seed(0)
    modelo = train.criar_modelo("resnet50", pretrained=False)
    imagem = torch.randn(3, 64, 128)
    for entrada, alvo in ((imagem, None), (imagem.unsqueeze(0), 2)):
        mapa = gradcam.gerar_gradcam(modelo, entrada, alvo=alvo)
        assert mapa.shape == (64, 128) and mapa.dtype == np.float32
        assert 0 <= mapa.min() and mapa.max() <= 1 + 1e-6
    assert not modelo.training  # fica em eval


def test_gerar_gradcam_recusa_lote_com_mais_de_uma_imagem():
    modelo = train.criar_modelo("resnet50", pretrained=False)
    with pytest.raises(ValueError):
        gradcam.gerar_gradcam(modelo, torch.randn(2, 3, 64, 128))


# ------------------------------------------------------------- A. posição dos ids
# Elegíveis sintéticas: ids 1 a 13 sem o 4 (fora das elegíveis); a de 1 a 7, b de 8 a 13.
ELEGIVEIS = pd.DataFrame({
    "id": [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "classe": ["a"] * 6 + ["b"] * 6,
})


def test_pureza_local_usa_os_5_vizinhos_de_cada_lado_entre_as_elegiveis():
    pureza = atalho.pureza_local(ELEGIVEIS, [1, 5, 7, 13])
    # id 1: só os 5 seguintes (2, 3, 5, 6, 7), todos a. id 5: antes 1, 2, 3 (a); depois
    # 6, 7 (a) e 8, 9, 10 (b): 5 de 8, pulando o id 4. id 7: 5 a antes e 5 b depois.
    assert pureza.to_dict() == {1: 1.0, 5: 0.625, 7: 0.5, 13: 1.0}
    with pytest.raises(ValueError):
        atalho.pureza_local(ELEGIVEIS, [4])


def test_grupos_de_pureza_com_wilson():
    grupos = atalho.grupos_de_pureza([1.0, 0.625, 0.5, 1.0, 0.6], [True, True, False, False, True])
    assert grupos["grupo"].tolist() == ["1.0", "0.6_a_1.0", "abaixo_de_0.6"]
    assert grupos["n"].tolist() == [2, 2, 1] and grupos["acertos"].tolist() == [1, 2, 0]
    baixo, alto = grupos.loc[0, ["ic95_baixo", "ic95_alto"]]
    assert (baixo, alto) == pytest.approx(atalho.metrics.intervalo_wilson(0.5, 2))
    # Com as classes, cada grupo traz a sua mistura de classes.
    classes = ["phoma", "ferrugem", "phoma", "phoma", "cercosporiose"]
    com_classes = atalho.grupos_de_pureza([1.0, 0.625, 0.5, 1.0, 0.6], [True] * 5, classes)
    assert com_classes["phoma"].tolist() == [2, 0, 1]
    assert com_classes["cercosporiose"].tolist() == [0, 1, 0]


def test_faixas_de_id_contam_acertos_e_classes():
    predicoes = pd.DataFrame({
        "id": [1, 2, 3, 5, 6, 7],
        "classe_verdadeira": ["saudavel", "ferrugem", "ferrugem", "phoma", "phoma", "phoma"],
        "classe_prevista": ["saudavel", "ferrugem", "phoma", "phoma", "phoma", "saudavel"],
    })
    faixas = atalho.faixas_de_id(predicoes, maior_id=7, tamanho=3)
    assert faixas["faixa"].tolist() == ["1-3", "4-6", "7-7"]
    assert faixas["n"].tolist() == [3, 2, 1] and faixas["acertos"].tolist() == [2, 2, 0]
    assert faixas["ferrugem"].tolist() == [2, 0, 0] and faixas["phoma"].tolist() == [0, 2, 1]
    assert atalho.faixa_do_id(7, maior_id=7, tamanho=3) == "7-7"
    assert atalho.faixa_do_id(1650, maior_id=1747) == "1601-1747"


# ------------------------------------------------------------- B. controle de fundo
def test_mascara_borda_tem_a_area_esperada_em_448x224():
    assert atalho.mascara_borda(224, 448, 0.15).mean() == pytest.approx(0.512, abs=0.001)
    assert atalho.mascara_borda(224, 448, 0.08).mean() == pytest.approx(0.296, abs=0.001)
    mapa = np.zeros((224, 448))
    mapa[100, 200] = 1  # só no interior
    assert atalho.fracao_na_borda(mapa, atalho.mascara_borda(224, 448)) == 0


def test_caracteristicas_de_cor_de_uma_imagem_constante():
    rgb = np.full((8, 16, 3), (200, 50, 50), dtype=np.uint8)
    caracteristicas = atalho.caracteristicas_de_cor(rgb, atalho.mascara_borda(8, 16, 0.25))
    valores = dict(zip(atalho.NOMES_CARACTERISTICAS, caracteristicas))
    assert len(caracteristicas) == 12
    assert valores["media_R"] == 200 and valores["media_G"] == 50 and valores["desvio_R"] == 0


def test_controle_de_fundo_roda_em_dados_sinteticos():
    gerador = np.random.default_rng(0)
    cores = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]  # uma cor de fundo por classe

    def conjunto(por_classe):
        imagens, rotulos = [], []
        for rotulo, cor in enumerate(cores):
            for _ in range(por_classe):
                ruido = gerador.integers(-20, 21, (32, 64, 3))
                imagens.append(np.clip(np.array(cor) + ruido, 0, 255).astype(np.uint8))
                rotulos.append(rotulo)
        return imagens, np.array(rotulos)

    (treino, y_treino), (val, y_val) = conjunto(10), conjunto(5)
    mascara = atalho.mascara_borda(32, 64, 0.15)
    x_treino = np.array([atalho.caracteristicas_de_cor(im, mascara) for im in treino])
    x_val = np.array([atalho.caracteristicas_de_cor(im, mascara) for im in val])
    resultado = atalho.controle_de_fundo(x_treino, y_treino, x_val, y_val)
    assert resultado["val"]["acuracia"] == 1.0 and resultado["acuracia_treino"] == 1.0
    assert resultado["val"]["classes"] == bracol.CLASSES
