"""Testes das partes puras de evaluation/cor_de_folha.py (passo 8): o histograma conjunto de
matiz e brilho numa máscara, o corte por brilho mínimo, a faixa mínima (inclusive a que passa do
179 ao 0) e a fração dentro da faixa. Sem pesos e sem data/raw."""
import cv2
import numpy as np

import cor_de_folha as cf


def _hist(pesos: dict[int, int]) -> np.ndarray:
    hist = np.zeros(cf.MATIZES, dtype=np.int64)
    for matiz, n in pesos.items():
        hist[matiz] = n
    return hist


def _quadrado(lado: int):
    return [[0, 0], [lado - 1, 0], [lado - 1, lado - 1], [0, lado - 1]]


def test_histograma_conta_so_os_coloridos_da_mascara():
    hsv = np.zeros((10, 20, 3), dtype=np.uint8)
    hsv[:, :10] = (60, 200, 200)  # metade esquerda: verde saturado
    hsv[:, 10:] = (60, 5, 200)  # metade direita: mesmo matiz, sem saturação
    hist, total = cf.histograma(hsv, [[0, 0], [19, 0], [19, 9], [0, 9]])
    assert hist.shape == (cf.MATIZES, cf.BRILHOS)
    assert total == 200
    assert hist[60, 200 // cf.BRILHO_PASSO] == 100 and hist.sum() == 100


def test_matiz_descarta_os_pixels_escuros():
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:5] = (105, 200, 40)  # azul escuro (como a sombra)
    hsv[5:] = (60, 200, 200)  # verde claro
    hist, _ = cf.histograma(hsv, _quadrado(10))
    assert cf.matiz(hist, 0)[105] == 50 and cf.matiz(hist, 0)[60] == 50
    assert cf.matiz(hist, 48)[105] == 0 and cf.matiz(hist, 48)[60] == 50


def test_faixa_minima_simples():
    hist = _hist({40: 50, 41: 30, 42: 19, 100: 1})
    assert cf.faixa_minima(hist[None], 0.99) == (40, 42)


def test_faixa_minima_passa_do_179_ao_0():
    hist = _hist({178: 40, 179: 30, 0: 20, 1: 10})
    assert cf.faixa_minima(hist[None], 1.0) == (178, 1)


def test_faixa_minima_pesa_cada_folha_igual():
    grande = _hist({50: 1_000_000})
    pequena = _hist({90: 10})
    assert cf.faixa_minima(np.stack([grande, pequena]), 1.0) == (50, 90)
    assert cf.faixa_minima(np.stack([grande, pequena]), 0.5) == (50, 50)


def test_na_faixa_com_e_sem_volta():
    hist = _hist({178: 4, 0: 3, 1: 2, 50: 1})
    assert cf.na_faixa(hist, (178, 1)) == 9
    assert cf.na_faixa(hist, (40, 60)) == 1


def test_fracoes_com_brilho_minimo_e_mascara_vazia():
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:5] = (60, 200, 40)  # verde escuro
    hsv[5:] = (60, 200, 200)  # verde claro
    hist, total = cf.histograma(hsv, _quadrado(10))
    colorida, folha = cf.fracoes(hist[None], np.array([total]), (24, 90), 48)
    assert colorida[0] == 1.0 and folha[0] == 0.5
    vazia = np.zeros((1, cf.MATIZES, cf.BRILHOS))
    colorida, folha = cf.fracoes(vazia, np.array([0]), (24, 90), 0)
    assert colorida[0] == 0 and folha[0] == 0


def test_cadeia_usa_a_faixa_e_o_corte_medidos():
    """Os padrões de inferencia.Configuracao são os do run cor_de_folha com brilho mínimo 0
    (decisão do gestor, 30/10/2026: o brilho mínimo não foi adotado)."""
    import json
    medicao = json.loads((cf.PASTA_SAIDA / "medicao.json").read_text(encoding="utf-8"))
    adotada = next(v for v in medicao["variantes"] if v["brilho_minimo"] == 0)
    config = cf.inf.Configuracao()
    assert list(config.faixa_matiz_folha) == adotada["faixa_matiz"]
    assert config.fracao_cor_de_folha_minima == adotada["corte"]


def test_cadeias_de_rotulagem_sem_o_filtro_de_matiz():
    """Os ids das folhas rotuladas não podem mudar com o filtro novo."""
    import campo
    import teste_final
    for config in (campo.CONFIG_ROTULAGEM, campo.CONFIG_ROTULAGEM_TREINO,
                   teste_final.CONFIG_ROTULAGEM_TESTE):
        assert config.fracao_cor_de_folha_minima == 0.0


def test_hsv_reduzido_igual_ao_da_cadeia():
    imagem = np.zeros((3000, 4000, 3), dtype=np.uint8)
    imagem[..., 1] = 180
    hsv, escala = cf.hsv_reduzido(imagem)
    assert max(hsv.shape[:2]) == cf.inf.LADO_COR
    assert escala == cf.inf.LADO_COR / 4000
    assert hsv[0, 0, 0] == cv2.cvtColor(np.uint8([[[0, 180, 0]]]), cv2.COLOR_RGB2HSV)[0, 0, 0]
