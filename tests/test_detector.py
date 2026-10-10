"""Testes do detector de folhas (model/detector.py, model/treinar_detector.py e
evaluation/metricas_deteccao.py).

Nenhum teste treina, roda o YOLO ou lê data/raw: os modelos são falsos e as imagens, sintéticas.
O teste da divisão de validação lê só as linhas de treino do manifest versionado do BRACOT, e
nenhum lê as anotações de teste dos autores. O teste do modo offline importa o ultralytics num
processo separado (o import muda o estado global), com a configuração numa pasta temporária.
"""
import hashlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import bracot
import dados
import detector
import metricas_deteccao as md
import train
import treinar_detector as td

ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def quadrado(x0, y0, lado):
    return np.array([[x0, y0], [x0 + lado, y0], [x0 + lado, y0 + lado], [x0, y0 + lado]],
                    dtype=np.float64)


def deteccao(poligono, score):
    """Uma detecção no formato de detector.detectar_folhas."""
    poligono = np.asarray(poligono, dtype=np.float64)
    return {"poligono": poligono.tolist(),
            "caixa": [*poligono.min(axis=0), *poligono.max(axis=0)], "score": score}


# Foto sintética de 1000 x 500 (largura x altura), avaliada em escala 1 (lado 1000), com duas folhas
# anotadas de 200 x 200. A região anotada vai de x 80 a 620 e de y 80 a 320 (margem de 20 px).
ANOTADAS = [quadrado(100, 100, 200), quadrado(400, 100, 200)]


def resumo(deteccoes, anotadas=ANOTADAS):
    return md.resumir_foto(anotadas, deteccoes, altura=500, largura=1000, lado=1000)


# ------------------------------------------------------------------- preparo dos dados
def test_rotulo_yolo_normaliza_e_corta_na_borda():
    poligonos = [np.array([[-2.0, 10.0], [50.0, 20.0], [104.0, 60.0]]),
                 [10, 0, 20, 0, 20, 80]]  # também no formato plano do COCO
    assert td.rotulo_yolo(poligonos, largura=100, altura=80) == (
        "0 0.000000 0.125000 0.500000 0.250000 1.000000 0.750000\n"
        "0 0.100000 0.000000 0.200000 0.000000 0.200000 1.000000\n")


def test_cinza_fora_da_regiao_mantem_o_casco_e_a_margem():
    imagem = np.full((100, 200, 3), 7, dtype=np.uint8)
    # Em pixels da foto original (400 x 200; escala 0,5). Na imagem, em (x, y), os quadrados vão
    # de (10, 10) a (30, 30) e de (150, 60) a (170, 80). Margem: 2% de 200 = 4 px.
    saida = td.cinza_fora_da_regiao(imagem, [quadrado(20, 20, 40), quadrado(300, 120, 40)], 0.5)
    assert saida.dtype == np.uint8 and (imagem == 7).all()  # a original não muda
    # (y, x): nos quadrados, entre eles (dentro do casco) e a 3 px do casco (na margem)
    for y, x in ((20, 20), (70, 160), (45, 90), (20, 7), (83, 160)):
        assert (saida[y, x] == 7).all(), (y, x)
    # nos cantos e a 7 px ou mais do casco
    for y, x in ((0, 0), (99, 0), (0, 199), (99, 199), (20, 3), (2, 20), (89, 160)):
        assert (saida[y, x] == td.CINZA).all(), (y, x)


@pytest.fixture
def bracot_sintetico(tmp_path, monkeypatch):
    """Quatro fotos PNG sintéticas (240 x 160, fundo 200), cada uma com uma folha anotada no
    centro; a "b" vai para a validação. O preparo reduz o lado maior a 120 px."""
    linhas, poligonos = [], {}
    for i in ("a", "b", "c", "d"):
        caminho = tmp_path / "fotos" / f"{i}.png"
        caminho.parent.mkdir(exist_ok=True)
        cv2.imencode(".png", np.full((160, 240, 3), 200, dtype=np.uint8))[1].tofile(caminho)
        linhas.append({"id": i, "caminho": str(caminho), "cena": i})
        poligonos[i] = [quadrado(100, 60, 40)]
    coco = tmp_path / "coco.json"
    coco.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bracot, "ler_manifest", lambda split=None: pd.DataFrame(linhas))
    monkeypatch.setattr(bracot, "caminho_coco", lambda split: coco)
    monkeypatch.setattr(td, "ler_poligonos", lambda split: poligonos)
    monkeypatch.setattr(td, "dividir_validacao", lambda manifest: ["b"])
    monkeypatch.setattr(td, "LADO_PREPARADO", 120)
    return SimpleNamespace(pasta=tmp_path / "yolo", poligonos=poligonos)


def test_preparar_poe_cinza_so_no_treino_e_grava_rotulos_e_yaml(bracot_sintetico):
    pasta = bracot_sintetico.pasta
    preparo = td.preparar(pasta)
    assert preparo["resumo"] == {"treino": {"fotos": 3, "folhas": 3, "cenas": 3},
                                 "val": {"fotos": 1, "folhas": 1, "cenas": 1}}
    assert sorted(p.name for p in (pasta / "images" / "treino").iterdir()) == [
        "a.jpg", "c.jpg", "d.jpg"]
    treino = dados.ler_imagem(pasta / "images" / "treino" / "a.jpg")
    val = dados.ler_imagem(pasta / "images" / "val" / "b.jpg")
    assert treino.shape == val.shape == (80, 120, 3)
    # JPEG: tolerância de 3. Canto: cinza no treino, a foto na validação; a folha, igual nos dois.
    assert np.abs(treino[2, 2].astype(int) - td.CINZA).max() <= 3
    assert np.abs(val[2, 2].astype(int) - 200).max() <= 3
    assert np.abs(treino[40, 60].astype(int) - 200).max() <= 3
    assert (pasta / "labels" / "val" / "b.txt").read_text(encoding="utf-8") == td.rotulo_yolo(
        bracot_sintetico.poligonos["b"], 240, 160)
    yaml = (pasta / "dados.yaml").read_text(encoding="utf-8")
    assert "train: images/treino\nval: images/val\nnames:\n  0: folha\n" in yaml


def test_preparar_e_idempotente_e_refaz_se_a_divisao_mudar(bracot_sintetico, monkeypatch):
    pasta = bracot_sintetico.pasta
    primeiro = td.preparar(pasta)
    leituras = []
    ler = dados.ler_imagem
    monkeypatch.setattr(dados, "ler_imagem",
                        lambda caminho: leituras.append(caminho) or ler(caminho))
    assert td.preparar(pasta) == primeiro and leituras == []  # nada é relido
    monkeypatch.setattr(td, "dividir_validacao", lambda manifest: ["c"])
    assert td.preparar(pasta)["marca"]["validacao"] == ["c"] and len(leituras) == 4
    assert [p.name for p in (pasta / "images" / "val").iterdir()] == ["c.jpg"]


def test_preparar_recusa_pasta_com_arquivos_sem_preparo_json(bracot_sintetico):
    pasta = bracot_sintetico.pasta
    pasta.mkdir()
    (pasta / "outro.txt").write_text("x", encoding="utf-8")
    with pytest.raises(train.ErroFatal):
        td.preparar(pasta)
    assert (pasta / "outro.txt").is_file()


@pytest.mark.skipif(not bracot.MANIFEST_BRACOT.is_file(), reason="manifest ainda não gerado")
def test_validacao_do_detector_e_estavel_e_separa_cenas_inteiras():
    treino = bracot.ler_manifest(split="treino")
    ids = td.dividir_validacao(treino)
    assert ids == sorted(ids) == td.dividir_validacao(treino)
    validacao = treino["id"].isin(ids)
    assert (len(ids), treino.loc[validacao, "cena"].nunique(),
            int(treino.loc[validacao, "n_folhas"].sum())) == (46, 24, 248)
    assert not set(treino.loc[validacao, "cena"]) & set(treino.loc[~validacao, "cena"])
    assert treino.loc[validacao, "data_hora"].dt.date.nunique() == 2  # os dois dias
    # O hash fixa as 46 fotos de 10/10/2026: mudar a validação pede decisão do gestor.
    assert hashlib.sha256(",".join(ids).encode()).hexdigest() == (
        "888ee1fdad3936699482177d00630d729c90b3b1b4f8a4073bac12109a175685")


# ----------------------------------------------------------------------- métricas
def test_fracao_fora_mede_a_area_fora_da_regiao():
    regiao = np.zeros((10, 10), dtype=bool)
    regiao[:, :5] = True
    mascaras = np.zeros((3, 10, 10), dtype=bool)
    mascaras[0, 0:2, 0:4] = True  # toda dentro
    mascaras[1, 0:2, 3:7] = True  # metade fora: não conta como fora (precisa de mais da metade)
    np.testing.assert_allclose(md.fracao_fora(mascaras, regiao), [0.0, 0.5, 1.0])  # vazia: 1


def test_ap_e_1_com_as_previsoes_perfeitas():
    m = md.avaliar([resumo([deteccao(p, s) for p, s in zip(ANOTADAS, (0.9, 0.8))])])
    for modo in md.MODOS:
        for tipo in md.TIPOS:
            assert m[modo][tipo] == {"ap50": pytest.approx(1.0), "ap50_95": pytest.approx(1.0)}


def test_previsao_fora_da_regiao_so_e_ignorada_no_modo_regiao():
    fora = deteccao(quadrado(850, 350, 100), 0.95)  # no canto da foto, longe das folhas
    entre = deteccao(quadrado(320, 150, 60), 0.85)  # entre as duas folhas: erro nos dois modos
    r = resumo([fora, deteccao(ANOTADAS[0], 0.9), entre, deteccao(ANOTADAS[1], 0.8)])
    assert r["fora"].tolist() == [True, False, False, False]  # em ordem de score
    m = md.avaliar([r])
    # Região: acerto, erro, acerto (a de 0,95 fica de fora): precisão 1 até o recall 0,5 e 2/3
    # daí até 1, nos 101 pontos de recall.
    ate_metade = int(np.sum(md.PONTOS_RECALL <= 0.5))
    assert m["regiao"]["mascara"]["ap50"] == pytest.approx(
        (ate_metade + (101 - ate_metade) * 2 / 3) / 101)
    # Conservador: erro, acerto, erro, acerto: a precisão máxima à direita é 0,5 em todo o recall.
    assert m["conservador"]["mascara"]["ap50"] == pytest.approx(0.5)
    assert m["conservador"]["caixa"]["ap50"] == pytest.approx(0.5)
    regiao, conservador = (td._ponto(m[modo]["curva"], 0.05) for modo in md.MODOS)
    assert (regiao["acertos"], regiao["erros"], regiao["ignoradas"], regiao["fora_da_regiao"],
            regiao["previstas"]) == (2, 1, 1, 1, 4)
    assert regiao["f1"] == pytest.approx(0.8) and conservador["f1"] == pytest.approx(2 / 3)
    assert (conservador["erros"], conservador["ignoradas"]) == (2, 0)
    # F1 de 0,8 em todos os limiares até 0,80: no empate, fica o menor.
    assert md.melhor_limiar(m["regiao"]["curva"])["limiar"] == pytest.approx(0.05)
    assert md.contagens([r], 0.9) == [
        {"anotadas": 2, "previstas": 2, "acertos": 1, "fora_da_regiao": 1}]


def test_ap50_95_cai_com_a_previsao_deslocada():
    # Deslocada 40 px: IoU de 160/240 = 0,67 na caixa (e quase o mesmo na máscara): acerta nos
    # limiares de IoU de 0,50 a 0,65, 4 dos 10.
    anotada = [quadrado(100, 100, 200)]
    m = md.avaliar([resumo([deteccao(quadrado(140, 100, 200), 0.9)], anotadas=anotada)])
    for tipo in md.TIPOS:
        assert m["regiao"][tipo]["ap50"] == pytest.approx(1.0)
        assert m["regiao"][tipo]["ap50_95"] == pytest.approx(0.4)


def test_avaliar_junta_as_previsoes_de_todas_as_fotos():
    perfeita = resumo([deteccao(ANOTADAS[0], 0.9)], anotadas=ANOTADAS[:1])
    sem_previsao = resumo([], anotadas=ANOTADAS[1:])
    m = md.avaliar([perfeita, sem_previsao])
    assert (m["n_fotos"], m["n_anotadas"], m["n_previstas"]) == (2, 2, 1)
    # precisão 1 até o recall 0,5 e nada depois
    assert m["conservador"]["mascara"]["ap50"] == pytest.approx(
        np.sum(md.PONTOS_RECALL <= 0.5) / 101)


# ---------------------------------------------------------------------- detector
class DetectorFalso:
    """Imita o YOLO do ultralytics: guarda a entrada e devolve um resultado pronto."""

    def __init__(self, resultado):
        self.resultado = resultado
        self.chamadas = []

    def predict(self, imagem, **kwargs):
        self.chamadas.append((imagem, kwargs))
        return [self.resultado]


def test_detectar_folhas_reduz_poe_moldura_e_volta_para_a_foto():
    # Foto 300 x 400 (altura x largura), vermelha em RGB: ampliada para 480 x 640 (escala 1,6) e
    # com a moldura de 16 px, a entrada fica 512 x 672, em BGR.
    foto = np.zeros((300, 400, 3), dtype=np.uint8)
    foto[..., 0] = 200
    m = detector.MOLDURA
    mascaras = np.zeros((3, 512, 672), dtype=bool)
    mascaras[0, m + 160:m + 320, 8:m + 480] = True  # começa dentro da moldura
    mascaras[1, 50, 50] = True  # um pixel só: sem polígono, fica de fora
    mascaras[2, m + 20:m + 60, m + 500:] = True  # vai até a borda direita da entrada
    mascaras[2, m + 400, m + 20] = True  # migalha: o polígono é o do maior pedaço
    resultado = SimpleNamespace(
        masks=SimpleNamespace(data=torch.from_numpy(mascaras.astype(np.float32))),
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[8, m + 160, m + 480, m + 320], [50, 50, 51, 51],
                               [m + 500, m + 20, 672, m + 60]], dtype=torch.float32),
            conf=torch.tensor([0.5, 0.9, 0.8])),
        orig_shape=(512, 672))
    falso = DetectorFalso(resultado)
    folhas = detector.detectar_folhas(foto, 0.25, detector=falso, dispositivo="cpu")
    entrada, argumentos = falso.chamadas[0]
    assert entrada.shape == (512, 672, 3) and entrada.flags["C_CONTIGUOUS"]
    assert entrada[100, 100].tolist() == [0, 0, 200] and entrada[0, 0].tolist() == [114] * 3
    assert argumentos == {"imgsz": 672, "conf": 0.25, "max_det": 300, "device": "cpu",
                          "verbose": False}
    assert [f["score"] for f in folhas] == pytest.approx([0.8, 0.5])  # sem a de um pixel
    segunda, primeira = folhas
    # Em pixels da foto: (x - 16) / 1,6, cortado em 0 e na largura (400).
    assert primeira["caixa"] == pytest.approx([0, 100, 300, 200])
    np.testing.assert_allclose(np.min(primeira["poligono"], axis=0), [0, 100])
    np.testing.assert_allclose(np.max(primeira["poligono"], axis=0), [299.4, 199.4])
    assert segunda["caixa"] == pytest.approx([312.5, 12.5, 400, 37.5])
    assert np.min(segunda["poligono"], axis=0)[0] == pytest.approx(312.5)  # sem a migalha
    assert np.max(segunda["poligono"], axis=0)[0] == pytest.approx(400)


def test_detectar_folhas_recusa_imagem_sem_tres_canais():
    with pytest.raises(ValueError):
        detector.detectar_folhas(np.zeros((30, 40), dtype=np.uint8), 0.5,
                                 detector=DetectorFalso(None))


def test_converter_resultado_sem_mascaras_devolve_lista_vazia():
    assert detector.converter_resultado(SimpleNamespace(masks=None, boxes=None)) == []


def test_poligono_da_mascara_desfaz_o_letterbox():
    # Foto 392 x 672 em entrada 416 x 672: 12 px de borda em cima e embaixo.
    mascara = np.zeros((416, 672), dtype=bool)
    mascara[22:42, 50:100] = True
    poligono = detector.poligono_da_mascara(mascara, (392, 672))
    assert np.min(poligono, axis=0).tolist() == [50, 10]
    assert np.max(poligono, axis=0).tolist() == [99, 29]
    # Foto 1280 x 960 em entrada 640 x 640: ganho 0,5 e 80 px de borda dos lados.
    mascara = np.zeros((640, 640), dtype=bool)
    mascara[100:200, 180:280] = True
    poligono = detector.poligono_da_mascara(mascara, (1280, 960))
    assert np.min(poligono, axis=0).tolist() == [200, 200]
    assert np.max(poligono, axis=0).tolist() == [398, 398]
    assert detector.poligono_da_mascara(np.zeros((8, 8), dtype=bool), (8, 8)).shape == (0, 2)


SCRIPT_OFFLINE = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
os.environ["YOLO_CONFIG_DIR"] = sys.argv[2]
os.environ["YOLO_OFFLINE"] = "false"  # o detector impõe o modo offline mesmo assim
import detector
detector.importar_ultralytics()
import ultralytics.data.utils as utils_de_dados
from ultralytics.utils import ONLINE, SETTINGS
assert os.environ["YOLO_OFFLINE"] == "true" and os.environ["YOLO_AUTOINSTALL"] == "false"
assert not ONLINE and SETTINGS["sync"] is False
assert utils_de_dados.check_font("Arial.ttf") is None
assert not list(Path(sys.argv[2]).rglob("*.ttf"))
print("ok")
"""


@pytest.mark.skipif(not ULTRALYTICS, reason="ultralytics não instalado")
def test_importar_ultralytics_desliga_a_rede(tmp_path):
    """Modo offline, sync desligado e a pré-carga da fonte (que baixaria a Arial.ttf de
    ultralytics.com) trocada por uma que não baixa nada."""
    saida = subprocess.run(
        [sys.executable, "-c", SCRIPT_OFFLINE, str(td.RAIZ_REPO / "model"), str(tmp_path)],
        capture_output=True, text=True, timeout=300)
    assert saida.returncode == 0, saida.stderr[-2000:]
    assert saida.stdout.strip().endswith("ok")


# ---------------------------------------------------------------- relatório e teste
def test_argumentos_do_treino_ficam_com_caminhos_relativos():
    modelo = SimpleNamespace(ckpt={"train_args": {
        "model": str(td.RAIZ_REPO / "model" / "pesos" / "yolo11s-seg.pt"), "epochs": 60,
        "device": 0, "data": "relativo.yaml", "outro": Path("x"), "classes": None}})
    assert td.argumentos_do_treino(modelo) == {
        "model": "model/pesos/yolo11s-seg.pt", "epochs": 60, "device": 0,
        "data": "relativo.yaml", "outro": "x", "classes": None}
    assert td.argumentos_do_treino(SimpleNamespace()) == {}


@pytest.mark.skipif(not ULTRALYTICS, reason="ultralytics não instalado")
def test_relatorio_monta_com_dados_sinteticos(capsys):
    anotada = [quadrado(100, 100, 200)]
    resumos = [resumo([deteccao(anotada[0], 0.9), deteccao(quadrado(850, 350, 100), 0.7)],
                      anotadas=anotada)] * 2
    saida, metricas = td.metricas_da_validacao(resumos, limiar_alto=0.8)
    assert saida["limiar_escolhido"] == pytest.approx(0.05)  # F1 1 até 0,90: fica o menor
    assert set(saida["limiares"]) == {"0.05", "0.8"}
    assert saida["limiares"]["0.05"]["fora_da_regiao"] == 2
    saida["ultralytics_val"] = {t: {"ap50": 0.9, "ap50_95": 0.7} for t in ("caixa", "mascara")}
    etapas = ("leitura_s", "deteccao_s", "classificacao_s", "total_s")
    saida["rnf02"] = {"threads_torch": 8, "fotos": 2, "folhas_mediana": 2.0, "acima_do_limite": 0,
                      "mediana": dict.fromkeys(etapas, 0.5), "maximo": dict.fromkeys(etapas, 1.0)}
    manifest = pd.DataFrame({"id": ["a", "b", "c"], "cena": ["a", "b", "b"]})
    config = td.montar_config("detector_x", manifest, ["a"], dict.fromkeys("abc", anotada),
                              {"epochs": 2})
    assert config["dados"]["treino"] == {"fotos": 2, "cenas": 1, "folhas": 2}
    texto = td.relatorio(config, saida, metricas, pd.DataFrame({"time": [30.0, 61.0]}),
                         "Nota da periferia.")
    for trecho in ("Nota da periferia.", "| região (principal) |", "mais alto (para cortar",
                   "Arial.ttf", "O teste dos autores não foi lido nem avaliado",
                   "2 épocas em 1,0 min"):
        assert trecho in texto, trecho
    td._imprimir_resumo(saida)
    assert capsys.readouterr().out.isascii()


def test_resumo_do_teste_com_e_sem_as_fotos_sobrepostas():
    anotada = [quadrado(100, 100, 200)]
    perfeita = resumo([deteccao(anotada[0], 0.9)], anotadas=anotada)
    # IoU de 120/280 com a folha e quase toda na região: erro nos dois modos, e a folha fica sem
    # par.
    com_erro = resumo([deteccao(quadrado(180, 100, 200), 0.95)], anotadas=anotada)
    resultado = td.resumo_do_teste({"a": perfeita, "b": perfeita, "c": com_erro}, 0.5, ["c", "x"])
    assert {k: (v["fotos"], v["anotadas"]) for k, v in resultado.items()} == {
        "todas": (3, 3), "sem_sobrepostas": (2, 2)}
    sem = resultado["sem_sobrepostas"]
    for modo in md.MODOS:
        for tipo in md.TIPOS:
            assert sem["metricas"][modo][tipo]["ap50"] == pytest.approx(1.0)
            assert resultado["todas"]["metricas"][modo][tipo]["ap50"] < 1
    todas_no_limiar, sem_no_limiar = (r["no_limiar"]["regiao"] for r in resultado.values())
    assert todas_no_limiar["erros"] == 1
    assert (sem_no_limiar["erros"], sem_no_limiar["limiar"]) == (0, 0.5)


def test_registrar_uso_do_teste_cria_o_arquivo_e_acrescenta_linhas(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "PASTA_RUNS", tmp_path / "runs")
    td.registrar_uso_do_teste("detector_a")
    td.registrar_uso_do_teste("detector_b")
    texto = (tmp_path / "runs" / td.USO_DO_TESTE).read_text(encoding="utf-8")
    assert texto.startswith(td.CABECALHO_USO_DO_TESTE)
    linhas = texto[len(td.CABECALHO_USO_DO_TESTE):].splitlines()
    assert [linha.split("|")[2].strip() for linha in linhas] == ["detector_a", "detector_b"]


def test_avaliar_teste_exige_a_validacao_antes_e_nao_registra_uso(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "PASTA_RUNS", tmp_path)
    with pytest.raises(train.ErroFatal):
        td.avaliar_teste("detector_x")
    assert not (tmp_path / td.USO_DO_TESTE).exists()


def test_so_avaliar_teste_le_as_anotacoes_de_teste():
    fonte = Path(td.__file__).read_text(encoding="utf-8")
    assert fonte.count('ler_poligonos("teste")') == 1
    assert 'ler_poligonos("teste")' in inspect.getsource(td.avaliar_teste)
    for modulo in (td, detector, md):
        assert 'caminho_coco("teste")' not in Path(modulo.__file__).read_text(encoding="utf-8")
