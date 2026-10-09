"""Testes do bracot.py com uma cópia mínima e falsa do BRACOT (não precisam das fotos reais)."""
import csv
import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

import bracot

LARGURA, ALTURA = 80, 60
# Dois retângulos por foto: áreas 200 e 400 numa foto de 80x60 (4.800 px), 12,5% coberta.
RETANGULOS = [[10, 10, 30, 10, 30, 20, 10, 20], [40, 30, 60, 30, 60, 50, 40, 50]]
FOTOS = {
    "train": ["20190831_100000", "20190831_100005", "20190831_100100"],
    "test": ["20190831_100008", "20191208_090000"],
}


def pasta(repo, nome):
    return repo / "data" / "raw" / "bracot" / "bracot-data" / nome


def caminho_coco(repo, nome):
    return pasta(repo, nome) / bracot.SPLITS_DOS_AUTORES[nome][1]


def escrever_anotacoes(repo, nome, poligonos):
    """COCO e VIA de um split. poligonos: {id da foto: [polígonos]}. O campo area recebe a área
    do bbox, como no export do VIA dos autores."""
    imagens, anotacoes, via = [], [], {}
    for k, (foto, segs) in enumerate(sorted(poligonos.items())):
        arquivo = f"{foto}.jpg"
        imagens.append({"id": k, "width": LARGURA, "height": ALTURA, "file_name": arquivo})
        regioes = []
        for seg in segs:
            xs, ys = seg[0::2], seg[1::2]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            anotacoes.append({"id": len(anotacoes), "image_id": k, "segmentation": [seg],
                              "area": bbox[2] * bbox[3], "bbox": bbox, "iscrowd": 0,
                              "category_id": 0})
            regioes.append({"shape_attributes": {"name": "polygon", "all_points_x": xs,
                                                 "all_points_y": ys}, "region_attributes": {}})
        tamanho = (pasta(repo, nome) / arquivo).stat().st_size
        via[f"{arquivo}{tamanho}"] = {"filename": arquivo, "size": tamanho, "regions": regioes,
                                      "file_attributes": {}}
    coco = {"images": imagens, "annotations": anotacoes,
            "categories": [{"supercategory": "leaf", "id": 0, "name": "leaf"}]}
    caminho_coco(repo, nome).write_text(json.dumps(coco), encoding="utf-8")
    (pasta(repo, nome) / bracot.ARQUIVO_VIA).write_text(json.dumps(via), encoding="utf-8")


def mudar_coco(repo, nome, mudanca):
    """Aplica `mudanca(coco)` ao COCO gravado de um split."""
    coco = json.loads(caminho_coco(repo, nome).read_text(encoding="utf-8"))
    mudanca(coco)
    caminho_coco(repo, nome).write_text(json.dumps(coco), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """Repositório falso: 3 fotos em train/ e 2 em test/, com 2 folhas anotadas em cada."""
    semente = 0
    for nome, fotos in FOTOS.items():
        pasta(tmp_path, nome).mkdir(parents=True)
        for foto in fotos:
            semente += 1
            ruido = np.random.default_rng(semente).integers(0, 256, (ALTURA, LARGURA, 3),
                                                            dtype=np.uint8)
            Image.fromarray(ruido).save(pasta(tmp_path, nome) / f"{foto}.jpg", quality=90)
        escrever_anotacoes(tmp_path, nome, {foto: RETANGULOS for foto in fotos})
    return tmp_path


def gerar(repo, **opcoes):
    padrao = dict(
        entrada=repo / "data" / "raw" / "bracot" / "bracot-data",
        manifest=repo / "data" / "manifests" / "bracot.csv",
        relatorio=repo / "data" / "reports" / "integridade_bracot.md",
        raiz=repo,
        sobrepostos=(),
    )
    return bracot.gerar(**{**padrao, **opcoes})


def manifest(repo):
    with open(repo / "data" / "manifests" / "bracot.csv", newline="", encoding="utf-8") as f:
        return {linha["id"]: linha for linha in csv.DictReader(f)}


def relatorio(repo):
    return (repo / "data" / "reports" / "integridade_bracot.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------- funções puras
def test_cenas_cortam_so_acima_do_intervalo():
    base = datetime(2019, 8, 31, 10, 0, 0)
    segundos = {"a": 0, "b": 5, "c": 15, "d": 26}  # intervalos de 5, 10 e 11 s
    datas = {i: base + timedelta(seconds=s) for i, s in segundos.items()}
    assert bracot.cenas(datas, intervalo=10) == {"a": "a", "b": "a", "c": "a", "d": "d"}


def test_cenas_nao_dependem_da_ordem_da_entrada():
    datas = {"x": datetime(2019, 1, 1, 0, 0, 30), "y": datetime(2019, 1, 1, 0, 0, 0)}
    assert bracot.cenas(datas, intervalo=60) == {"x": "y", "y": "y"}


def test_area_do_poligono():
    quadrado = [0, 0, 10, 0, 10, 10, 0, 10]
    assert bracot.area_do_poligono(quadrado) == 100
    assert bracot.area_do_poligono(quadrado + [0, 0]) == 100  # primeiro ponto repetido no fim
    assert bracot.area_do_poligono([0, 0, 4, 0, 0, 3]) == 6


def test_data_hora_do_nome():
    assert bracot.data_hora_do_nome("20190831_163057") == datetime(2019, 8, 31, 16, 30, 57)
    assert bracot.data_hora_do_nome("20191340_250000") is None


def test_caminho_coco_por_split(tmp_path):
    assert bracot.caminho_coco("treino", tmp_path) == tmp_path / "train" / "train_annotation_coco.json"
    assert bracot.caminho_coco("teste", tmp_path) == tmp_path / "test" / "test_annotation_coco.json"
    with pytest.raises(ValueError, match="split desconhecido"):
        bracot.caminho_coco("val", tmp_path)


# --------------------------------------------------------------------------- gerar
def test_gera_uma_linha_por_foto(repo):
    assert gerar(repo) == 0
    m = manifest(repo)
    assert list(m) == sorted(FOTOS["train"] + FOTOS["test"])
    assert list(next(iter(m.values()))) == bracot.COLUNAS_MANIFEST
    linha = m["20190831_100008"]
    assert (linha["fonte"], linha["split"], linha["n_folhas"]) == ("bracot", "teste", "2")
    assert linha["caminho"] == "data/raw/bracot/bracot-data/test/20190831_100008.jpg"
    assert linha["anotacao"] == "data/raw/bracot/bracot-data/test/test_annotation_coco.json"
    assert (linha["largura"], linha["altura"], linha["area_coberta"]) == ("80", "60", "0.1250")
    assert linha["data_hora"] == "2019-08-31T10:00:08"
    assert len(linha["sha256"]) == 64 and len(linha["phash"]) == 64
    assert {m[i]["split"] for i in FOTOS["train"]} == {"treino"}
    assert "**OK:** nenhum erro" in relatorio(repo)


def test_cenas_no_manifest(repo):
    gerar(repo)
    cena = {i: linha["cena"] for i, linha in manifest(repo).items()}
    # 10:00:00, :05 e :08 (teste) ficam juntas; 10:01:00 e o outro dia ficam sozinhas.
    assert cena == {"20190831_100000": "20190831_100000", "20190831_100005": "20190831_100000",
                    "20190831_100008": "20190831_100000", "20190831_100100": "20190831_100100",
                    "20191208_090000": "20191208_090000"}
    assert "Cenas com fotos de treino e de teste: 1" in relatorio(repo)


def test_campo_area_igual_ao_bbox_vira_aviso(repo):
    assert gerar(repo) == 0
    assert "o campo area do COCO é a área do bbox em 10 de 10 anotações" in relatorio(repo)


@pytest.mark.parametrize(("mudanca", "mensagem"), [
    (lambda c: c["annotations"].append({**c["annotations"][0], "id": 99, "image_id": 77}),
     "anotações sem imagem: ids 99"),
    (lambda c: c.update(annotations=[a for a in c["annotations"] if a["image_id"] != 0]),
     "imagens sem anotação: 20190831_100000.jpg"),
    (lambda c: c["annotations"][1].update(id=0), "id de anotação repetido: 0"),
    (lambda c: c["categories"].append({"id": 1, "name": "symptom"}), "categorias inesperadas"),
    (lambda c: c["images"][0].update(width=81), "tem 81x60 no COCO e 80x60 no arquivo"),
    (lambda c: c["annotations"][0].update(segmentation=[[1, 1, 5, 5]]),
     "polígono com 4 coordenadas"),
])
def test_problemas_no_coco_sao_erro(repo, mudanca, mensagem):
    mudar_coco(repo, "train", mudanca)
    assert gerar(repo) == 1
    assert not (repo / "data" / "manifests" / "bracot.csv").exists()
    assert mensagem in relatorio(repo)


def test_poligono_na_borda_e_aviso_e_alem_da_tolerancia_e_erro(repo):
    def passar(excesso):
        def mudanca(coco):
            coco["annotations"][0]["segmentation"] = [[10, 10, LARGURA + excesso, 10,
                                                       LARGURA + excesso, 20, 10, 20]]
        return mudanca

    mudar_coco(repo, "test", passar(3))
    assert gerar(repo) == 0
    assert "1 polígonos passam até 3 px da borda" in relatorio(repo)
    mudar_coco(repo, "test", passar(bracot.TOLERANCIA_BORDA + 1))
    assert gerar(repo) == 1
    assert f"passa {bracot.TOLERANCIA_BORDA + 1} px da borda" in relatorio(repo)


def test_via_com_tamanho_diferente_e_erro(repo):
    via = json.loads((pasta(repo, "train") / bracot.ARQUIVO_VIA).read_text(encoding="utf-8"))
    next(iter(via.values()))["size"] = 1
    (pasta(repo, "train") / bracot.ARQUIVO_VIA).write_text(json.dumps(via), encoding="utf-8")
    assert gerar(repo) == 1
    assert "tem 1 bytes no VIA" in relatorio(repo)


def test_arquivo_fora_do_padrao_vira_aviso(repo):
    (pasta(repo, "train") / "Thumbs.db").write_bytes(b"cache do Windows")
    assert gerar(repo) == 0
    assert "ignorado em train/: Thumbs.db" in relatorio(repo)


def test_sobrepostos_conferidos_contra_a_divisao(repo):
    assert gerar(repo, sobrepostos=(("20190831_100008", "20190831_100005"),)) == 0
    assert "| 20190831_100008 | 20190831_100005 | 3 |" in relatorio(repo)
    assert gerar(repo, sobrepostos=(("20190831_100005", "20190831_100008"),)) == 1
    assert "TESTE_SOBREPOSTO desatualizado" in relatorio(repo)


def test_rodar_de_novo_gera_os_mesmos_bytes(repo):
    gerar(repo)
    arquivo_manifest = repo / "data" / "manifests" / "bracot.csv"
    arquivo_relatorio = repo / "data" / "reports" / "integridade_bracot.md"
    antes = arquivo_manifest.read_bytes(), arquivo_relatorio.read_bytes()
    gerar(repo)
    assert (arquivo_manifest.read_bytes(), arquivo_relatorio.read_bytes()) == antes
    assert b"\r\n" not in antes[0] and b"\r\n" not in antes[1]


def test_saida_de_terminal_sem_acento(repo, capsys):
    mudar_coco(repo, "train", lambda c: c["annotations"][1].update(id=0))  # força um erro
    gerar(repo)
    saida = capsys.readouterr().out
    assert saida.isascii()
    assert "Manifest NAO gravado" in saida


# ----------------------------------------------------------------------- --verificar
def test_verificar_confere_e_acusa_anotacao_mudada(repo):
    gerar(repo)
    arquivo = repo / "data" / "manifests" / "bracot.csv"
    antes = arquivo.read_bytes()
    assert gerar(repo, verificar=True) == 0
    mudar_coco(repo, "train", lambda c: c["annotations"][0].update(
        segmentation=[[10, 10, 50, 10, 50, 20, 10, 20]]))
    assert gerar(repo, verificar=True) == 1
    assert "coluna area_coberta diverge do manifest versionado: ids 20190831_100000" in \
        relatorio(repo)
    assert arquivo.read_bytes() == antes  # o --verificar nunca grava o manifest


def test_verificar_sem_manifest_e_erro_fatal(repo):
    with pytest.raises(bracot.od.ErroFatal, match="rode sem --verificar"):
        gerar(repo, verificar=True)


# ---------------------------------------------------------------------- ler_manifest
def test_ler_manifest_com_tipos_e_split(repo):
    gerar(repo)
    caminho = repo / "data" / "manifests" / "bracot.csv"
    df = bracot.ler_manifest(caminho)
    assert len(df) == 5 and set(df["split"]) == {"treino", "teste"}
    assert df["n_folhas"].dtype == "int64" and df["area_coberta"].dtype == "float64"
    assert df.loc[df["id"] == "20191208_090000", "data_hora"].item() == datetime(2019, 12, 8, 9)
    assert len(bracot.ler_manifest(caminho, split="teste")) == 2
    with pytest.raises(ValueError, match="split desconhecido"):
        bracot.ler_manifest(caminho, split="val")
