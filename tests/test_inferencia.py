"""Testes da inferência ponta a ponta (model/inferencia.py), da curva risco-cobertura
(evaluation/metrics.py) e das partes puras de model/avaliar_pipeline.py.

Sem treino, sem pesos e sem data/raw: o detector é trocado por uma função que devolve detecções
prontas, o classificador por um módulo que devolve probabilidades prontas, e as fotos são
sintéticas (folhas verdes sobre fundo cinza).
"""
import math
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import avaliar_pipeline as ap
import dados
import detector
import inferencia as inf
import metrics

CINZA_FOTO = 128
VERDE = (30, 150, 30)


class ClassificadorFalso(torch.nn.Module):
    """Devolve, em ordem, as probabilidades e os níveis de severidade pedidos e guarda o formato
    de cada lote."""

    def __init__(self, probabilidades, niveis):
        super().__init__()
        self.probabilidades = torch.tensor(probabilidades, dtype=torch.float32)
        self.niveis = torch.tensor(niveis)
        self.lotes = []

    def forward(self, x):
        n = x.shape[0]
        self.lotes.append(tuple(x.shape))
        severidade = torch.full((n, 5), -10.0)
        severidade[torch.arange(n), self.niveis[:n]] = 10.0
        return torch.log(self.probabilidades[:n]), severidade


def modelos_falsos(probabilidades, niveis):
    return inf.Modelos(classificador=ClassificadorFalso(probabilidades, niveis),
                       transformacao=dados.transformacao_avaliacao(64, 32), detector=None,
                       dispositivo=torch.device("cpu"), nome_classificador="falso",
                       nome_detector="falso")


def probs(classe: str, confianca: float):
    """Probabilidades com `confianca` na classe e o resto dividido entre as outras."""
    p = np.full(5, (1 - confianca) / 4)
    p[inf.CATEGORIAS.index(classe)] = confianca
    return p.tolist()


def retangulo(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def foto(*folhas, altura=300, largura=600):
    """Foto cinza com retângulos verdes nas posições das folhas."""
    imagem = np.full((altura, largura, 3), CINZA_FOTO, dtype=np.uint8)
    for x0, y0, x1, y1 in folhas:
        imagem[y0:y1, x0:x1] = VERDE
    return imagem


def deteccao(x0, y0, x1, y1, score):
    return {"poligono": retangulo(x0, y0, x1, y1), "caixa": [x0, y0, x1, y1], "score": score}


@pytest.fixture
def detectar_com(monkeypatch):
    """Troca o detector por uma lista de detecções prontas."""
    def trocar(deteccoes):
        monkeypatch.setattr(detector, "detectar_folhas",
                            lambda imagem, limiar, **_: [d for d in deteccoes
                                                         if d["score"] >= limiar])
    return trocar


# ------------------------------------------------------------------------- contrato
def test_contrato_com_varias_folhas_filtros_e_coerencia(detectar_com):
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100), (400, 150, 560, 230), (20, 200, 35, 212)]
    imagem = foto(*folhas)
    detectar_com([deteccao(*folhas[0], 0.9), deteccao(*folhas[1], 0.8),
                  deteccao(*folhas[2], 0.5),
                  deteccao(*folhas[3], 0.7),  # 15 x 12 px: abaixo da área mínima
                  deteccao(150, 200, 300, 280, 0.6),  # só fundo cinza: o filtro de cor tira
                  deteccao(450, 20, 550, 90, 0.2)])  # abaixo do limiar do detector
    modelos = modelos_falsos([probs("saudavel", 0.9), probs("ferrugem", 0.8),
                              probs("phoma", 0.4)], [3, 0, 2])
    resposta = inf.diagnosticar(imagem, inf.Configuracao(limiar_confianca=0.61), modelos)
    assert resposta["status"] == "sucesso"
    assert modelos.classificador.lotes == [(3, 3, 32, 64)]  # um lote com as 3 utilizáveis
    primeira, segunda = resposta["folhas"]  # a phoma (0,4) ficou abaixo do limiar
    assert [primeira["id"], segunda["id"]] == ["folha-1", "folha-2"]
    assert (primeira["categoria"], primeira["severidade"]) == ("saudavel", "saudavel")
    assert (segunda["categoria"], segunda["severidade"]) == ("ferrugem", "muito_baixa")
    for folha in resposta["folhas"]:
        assert {"id", "categoria", "severidade", "regiao", "confianca", "scoreDeteccao", "caixa",
                "cortadaNaBorda"} <= set(folha)
        assert 0 <= folha["regiao"]["x"] <= 1 and 0 <= folha["regiao"]["y"] <= 1
        assert inf.RAIO_MINIMO <= folha["regiao"]["raio"] <= inf.RAIO_MAXIMO
        assert all(0 <= v <= 1 for v in folha["caixa"])
    assert primeira["regiao"] == pytest.approx({"x": 80 / 600, "y": 50 / 300, "raio": round(
        math.sqrt(120 * 60 / math.pi) / 300, 4)}, abs=1e-4)
    assert (primeira["confianca"], primeira["scoreDeteccao"]) == (0.9, 0.9)
    detalhes = resposta["detalhes"]
    assert (detalhes["plano"], detalhes["deteccoes"], detalhes["cortadas_por_area"],
            detalhes["cortadas_por_cor"], detalhes["abaixo_da_confianca"]) == (
        "deteccao", 5, 1, 1, 1)


def test_max_folhas_fica_com_as_de_maior_score(detectar_com):
    folhas = [(20 + 110 * k, 40, 120 + 110 * k, 100) for k in range(5)]
    detectar_com([deteccao(*f, s) for f, s in zip(folhas, (0.5, 0.9, 0.6, 0.8, 0.7))])
    modelos = modelos_falsos([probs("ferrugem", 0.9)] * 2, [2, 2])
    resposta = inf.diagnosticar(foto(*folhas), inf.Configuracao(max_folhas=2), modelos)
    assert [f["scoreDeteccao"] for f in resposta["folhas"]] == [0.9, 0.8]
    assert resposta["detalhes"]["cortadas_por_limite"] == 3
    assert modelos.classificador.lotes == [(2, 3, 32, 64)]


@pytest.mark.parametrize("nivel, esperado", [(0, "muito_baixa"), (1, "muito_baixa"),
                                             (2, "baixa"), (3, "alta"), (4, "muito_alta")])
def test_mapa_de_niveis_com_a_coerencia_do_frontend(nivel, esperado):
    assert inf.severidade_do_contrato("ferrugem", nivel) == esperado
    assert inf.severidade_do_contrato("saudavel", nivel) == "saudavel"
    assert inf.NIVEIS_CONTRATO == ("saudavel", "muito_baixa", "baixa", "alta", "muito_alta")
    assert inf.CATEGORIAS == ("saudavel", "ferrugem", "bicho_mineiro", "phoma", "cercosporiose")


# --------------------------------------------------------------- folha única e erros
def test_regra_1_uma_folha_grande_usa_a_foto_inteira(detectar_com):
    folha = (100, 50, 500, 250)  # caixa com 44% da foto
    detectar_com([deteccao(*folha, 0.8)])
    modelos = modelos_falsos([probs("phoma", 0.9)], [1])
    resposta = inf.diagnosticar(foto(folha), inf.Configuracao(), modelos)
    assert resposta["detalhes"]["plano"] == "foto_inteira" and resposta["detalhes"]["regra"] == 1
    assert modelos.classificador.lotes == [(1, 3, 32, 64)]
    unica, = resposta["folhas"]
    assert unica["scoreDeteccao"] == 0.8 and unica["regiao"]["x"] == pytest.approx(0.5)
    assert unica["caixa"] == pytest.approx([100 / 600, 50 / 300, 500 / 600, 250 / 300], abs=1e-4)


def test_regra_2_deteccao_fraca_com_cor_de_folha_usa_a_foto_inteira(detectar_com):
    folha = (250, 100, 350, 160)
    detectar_com([deteccao(*folha, 0.1)])  # abaixo de 0,35, acima de 0,05
    resposta = inf.diagnosticar(foto(folha), inf.Configuracao(),
                                modelos_falsos([probs("saudavel", 0.95)], [0]))
    assert (resposta["detalhes"]["plano"], resposta["detalhes"]["regra"]) == ("foto_inteira", 2)
    unica, = resposta["folhas"]
    assert unica["regiao"] == {"x": 0.5, "y": 0.5, "raio": 0.5}
    assert unica["scoreDeteccao"] is None and unica["caixa"] == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("deteccoes", [
    [],  # nenhuma detecção
    [deteccao(250, 100, 350, 160, 0.1)],  # detecção fraca, mas só fundo cinza (sem cor)
    [deteccao(20, 20, 30, 30, 0.9)],  # no limiar, mas abaixo da área mínima e sem cor
])
def test_regra_3_sem_folha_utilizavel_e_planta_nao_identificada(detectar_com, deteccoes):
    detectar_com(deteccoes)
    modelos = modelos_falsos([probs("saudavel", 0.9)], [0])
    resposta = inf.diagnosticar(foto(), inf.Configuracao(), modelos)
    assert (resposta["status"], resposta["tipo"]) == ("erro", "planta_nao_identificada")
    assert resposta["mensagem"] == inf.MENSAGENS["planta_nao_identificada"]
    assert resposta["detalhes"]["regra"] == 3 and modelos.classificador.lotes == []


def test_todas_abaixo_do_limiar_de_confianca_e_baixa_confianca(detectar_com):
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100)]
    detectar_com([deteccao(*f, 0.9) for f in folhas])
    resposta = inf.diagnosticar(foto(*folhas), inf.Configuracao(limiar_confianca=0.61),
                                modelos_falsos([probs("phoma", 0.5), probs("ferrugem", 0.6)],
                                               [1, 1]))
    assert (resposta["status"], resposta["tipo"]) == ("erro", "baixa_confianca")
    assert resposta["detalhes"]["abaixo_da_confianca"] == 2


def test_imagem_que_nao_e_rgb_e_formato_invalido():
    resposta = inf.diagnosticar(np.zeros((10, 10), dtype=np.uint8), inf.Configuracao(),
                                modelos_falsos([probs("saudavel", 0.9)], [0]))
    assert resposta["tipo"] == "formato_invalido"


# ------------------------------------------------------------------- arquivos
def test_decodificar_imagem_e_a_mesma_leitura_de_ler_imagem(tmp_path):
    imagem = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    caminho = tmp_path / "fólha.png"  # com acento, como no Windows
    cv2.imencode(".png", imagem)[1].tofile(caminho)
    assert np.array_equal(dados.ler_imagem(caminho), dados.decodificar_imagem(caminho.read_bytes()))
    assert np.array_equal(dados.ler_imagem(caminho), imagem[..., ::-1])
    with pytest.raises(ValueError):
        dados.decodificar_imagem(b"isto nao e uma imagem")


def test_diagnosticar_arquivo_aceita_bytes_e_caminho(tmp_path, detectar_com):
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100)]
    caminho = tmp_path / "foto.png"
    cv2.imencode(".png", foto(*folhas)[..., ::-1])[1].tofile(caminho)
    detectar_com([deteccao(*f, 0.9) for f in folhas])
    respostas = [inf.diagnosticar_arquivo(entrada, inf.Configuracao(limiar_confianca=0.5),
                                          modelos_falsos([probs("ferrugem", 0.9)] * 2, [2, 2]))
                 for entrada in (caminho, caminho.read_bytes())]
    assert respostas[0] == respostas[1] and respostas[0]["status"] == "sucesso"
    assert inf.diagnosticar_arquivo(b"\x89PNG quebrado")["tipo"] == "formato_invalido"
    with pytest.raises(FileNotFoundError):
        inf.diagnosticar_arquivo(tmp_path / "nao_existe.jpg")


def png_so_cabecalho(largura: int, altura: int) -> bytes:
    """PNG com o cabeçalho de uma imagem largura x altura e dados que não decodificam: só a
    leitura do cabeçalho funciona."""
    def bloco(tipo, dados_):
        return (struct.pack(">I", len(dados_)) + tipo + dados_
                + struct.pack(">I", zlib.crc32(tipo + dados_) & 0xFFFFFFFF))
    cabecalho = struct.pack(">IIBBBBB", largura, altura, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + bloco(b"IHDR", cabecalho) + bloco(b"IDAT", b"\x00" * 16)
            + bloco(b"IEND", b""))


@pytest.mark.parametrize("largura, altura, trecho", [(9000, 8000, "9000 x 8000 pixels"),
                                                     (20000, 10000, "passa de")])
def test_arquivo_muito_grande_pelo_cabecalho(largura, altura, trecho):
    conteudo = png_so_cabecalho(largura, altura)
    resposta = inf.diagnosticar_arquivo(conteudo)  # sem modelos: nada é decodificado
    assert (resposta["status"], resposta["tipo"]) == ("erro", "arquivo_muito_grande")
    assert trecho in resposta["mensagem"]
    assert inf.dimensoes_pelo_cabecalho(png_so_cabecalho(400, 300)) == (400, 300)


# ------------------------------------------------------------------------ geometria
@pytest.mark.parametrize("angulo", [0, 30, 90, -45, 135])
@pytest.mark.parametrize("modo", ["A", "B"])
def test_recorte_deita_a_folha_com_margem_em_2_por_1(angulo, modo):
    imagem = np.full((600, 800, 3), 50, dtype=np.uint8)
    pontos = cv2.boxPoints(((400, 300), (300, 60), angulo))
    cv2.fillPoly(imagem, [np.round(pontos).astype(np.int32)], VERDE)
    recorte = inf.recortar_folha(imagem, pontos, modo, margem=0.1, cinza=200)
    assert recorte.shape == (180, 360, 3)  # 300 x (1 + 2 x 0,1) = 360, metade na altura
    ys, xs = np.nonzero(np.all(recorte == VERDE, axis=2))
    assert 295 <= xs.max() - xs.min() + 1 <= 302 and 55 <= ys.max() - ys.min() + 1 <= 62
    assert xs.mean() / 360 == pytest.approx(0.5, abs=0.02)
    assert recorte[2, 2].tolist() == ([200] * 3 if modo == "B" else [50] * 3)


def test_recorte_a_de_folha_na_borda_completa_com_cinza():
    imagem = np.full((200, 300, 3), 50, dtype=np.uint8)
    imagem[80:120, 0:150] = VERDE  # folha cortada pela borda esquerda
    recorte = inf.recortar_folha(imagem, retangulo(0, 80, 150, 120), "A", 0.17, 197)
    meio = recorte.shape[0] // 2  # o sentido do eixo é arbitrário: o lado de fora pode vir
    pontas = [recorte[meio, 0].tolist(), recorte[meio, -1].tolist()]  # em qualquer ponta
    assert [197] * 3 in pontas and [50] * 3 in pontas  # fora da foto e o fundo da foto
    assert inf.cortada_pela_borda(retangulo(0, 80, 150, 120), 200, 300)
    assert not inf.cortada_pela_borda(retangulo(10, 80, 150, 120), 200, 300)


def test_recorte_da_foto_inteira_deita_e_completa_ate_2_por_1():
    assert inf.recorte_da_foto_inteira(np.zeros((100, 200, 3), np.uint8), 197).shape == (
        100, 200, 3)
    retrato = inf.recorte_da_foto_inteira(np.zeros((400, 300, 3), np.uint8), 197)
    assert retrato.shape == (300, 600, 3) and retrato[0, 0].tolist() == [197] * 3
    assert retrato[150, 300].tolist() == [0, 0, 0]


def test_regiao_da_folha_e_filtro_de_cor():
    regiao = inf.regiao_da_folha(retangulo(100, 50, 300, 150), 300, 600)
    assert regiao == pytest.approx({"x": 200 / 600, "y": 100 / 300,
                                    "raio": math.sqrt(200 * 100 / math.pi) / 300}, abs=1e-4)
    assert inf.regiao_da_folha(retangulo(0, 0, 2, 2), 300, 600)["raio"] == inf.RAIO_MINIMO
    assert inf.regiao_da_folha(retangulo(0, 0, 600, 300), 300, 600)["raio"] == inf.RAIO_MAXIMO
    assert inf.regiao_da_folha(None, 300, 600) == {"x": 0.5, "y": 0.5, "raio": 0.5}
    saturacao = cv2.cvtColor(foto((0, 0, 300, 300)), cv2.COLOR_RGB2HSV)[..., 1]
    assert inf.fracao_colorida(saturacao, retangulo(0, 0, 299, 299), 30) == pytest.approx(1.0)
    assert inf.fracao_colorida(saturacao, retangulo(150, 0, 449, 299), 30) == pytest.approx(
        0.5, abs=0.01)


def test_configuracao_recusa_modo_de_recorte_desconhecido():
    with pytest.raises(ValueError):
        inf.Configuracao(recorte="C")


# ------------------------------------------------------------------ calibração (RF07)
def test_curva_risco_cobertura():
    curva = metrics.curva_risco_cobertura([0.9, 0.8, 0.6, 0.4], [True, True, False, True],
                                          [0.5, 0.7, 0.95], classes_verdadeiras=[0, 1, 0, 1],
                                          classes=["a", "b"])
    meio, alto, vazio = curva
    assert (meio["mantidas"], meio["cobertura"], meio["acuracia"]) == (3, 0.75,
                                                                        pytest.approx(2 / 3))
    assert meio["cobertura_por_classe"] == {"a": 1.0, "b": 0.5}
    assert (alto["mantidas"], alto["acuracia"], alto["risco"]) == (2, 1.0, 0.0)
    assert vazio["mantidas"] == 0 and math.isnan(vazio["acuracia"])
    with pytest.raises(ValueError):
        metrics.curva_risco_cobertura([0.9], [True, False], [0.5])


def test_propor_limiar_e_o_menor_que_chega_ao_alvo():
    curva = [{"limiar": 0.5, "mantidas": 10, "acuracia": 0.9},
             {"limiar": 0.6, "mantidas": 8, "acuracia": 0.95},
             {"limiar": 0.7, "mantidas": 6, "acuracia": 0.97}]
    assert ap.propor_limiar(curva)["limiar"] == 0.6
    assert ap.propor_limiar(curva, alvo=0.99) is None


def test_nativo_converte_os_tipos_do_numpy():
    assert ap.nativo({"a": np.int64(3), "b": [np.float32(0.5)], np.int64(1): np.array([1, 2])}) \
        == {"a": 3, "b": [0.5], "1": [1, 2]}


def test_recortes_dentro_do_repositorio_sao_recusados():
    with pytest.raises(SystemExit):
        ap.main(["--recortes", str(ap.RAIZ_REPO / "model" / "runs" / "x"), "--so-constantes"])


def test_inferencia_e_avaliacao_nao_leem_os_testes():
    for modulo in (inf, ap):
        fonte = Path(modulo.__file__).read_text(encoding="utf-8")
        assert "montar_teste" not in fonte
        assert 'caminho_coco("teste")' not in fonte and 'ler_poligonos("teste")' not in fonte
