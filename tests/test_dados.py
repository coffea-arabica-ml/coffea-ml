"""Testes da montagem dos dados da Frente 9 (model/dados.py).

Só leem os manifests versionados: nenhum teste lê data/raw, abre imagem do dataset ou baixa
alguma coisa.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import bracol
import dados


def test_rotulo_e_o_indice_em_classes_nunca_o_codigo_original():
    assert [dados.rotulo(c) for c in bracol.CLASSES] == [0, 1, 2, 3, 4]
    # predominant_stress 1 (miner) é bicho_mineiro, índice 2; 2 (rust) é ferrugem, índice 1.
    assert dados.rotulo(bracol.classe_do_projeto(1)) == 2
    assert dados.rotulo(bracol.classe_do_projeto(2)) == 1


def test_dataset_usa_classes_index_e_mantem_a_severidade_sem_alvo():
    df = pd.DataFrame({
        "fonte": ["bracol", "jmuben", "bracol"], "id": [7, 7, 9],
        "caminho": ["a.jpg", "b.jpg", "c.jpg"],
        "classe": ["bicho_mineiro", "ferrugem", "saudavel"], "severity": [3, -1, 0],
    })
    conjunto = dados.FolhasDataset(df, transform=None)  # não lê nenhuma imagem
    assert len(conjunto) == 3
    assert conjunto.classes.tolist() == [2, 1, 0]
    assert conjunto.severidades.tolist() == [3, dados.SEM_SEVERIDADE, 0]


def test_treino_val_e_teste_vem_dos_manifests_versionados():
    treino, val, teste = dados.montar_treino(), dados.montar_val(), dados.montar_teste()
    assert (len(treino), len(val), len(teste)) == (1180, 252, 253)
    for df in (treino, val, teste):
        assert list(df.columns) == dados.COLUNAS
        assert set(df["fonte"]) == {"bracol"}
        assert df["severity"].between(0, 4).all()
    assert (val["classe"] == "cercosporiose").sum() == 22
    ids = [set(df["id"]) for df in (treino, val, teste)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_jmuben_so_entra_no_treino_e_sem_alvo_de_severidade():
    treino = dados.montar_treino(auxiliar="jmuben")
    auxiliar = treino[treino["fonte"] == "jmuben"]
    assert len(treino) == 1180 + 506 and len(auxiliar) == 506
    assert (auxiliar["severity"] == dados.SEM_SEVERIDADE).all()
    menor = dados.montar_treino(auxiliar="jmuben", teto=10)
    assert (menor["fonte"] == "jmuben").sum() == 50  # 10 por classe
    assert set(dados.montar_val()["fonte"]) == set(dados.montar_teste()["fonte"]) == {"bracol"}


def test_fonte_auxiliar_desconhecida_ou_teto_sem_auxiliar_levantam_erro():
    with pytest.raises(ValueError):
        dados.montar_treino(auxiliar="bracot")
    with pytest.raises(ValueError):
        dados.montar_treino(teto=10)


def test_sampler_da_o_mesmo_peso_total_a_cada_classe():
    classes = ["ferrugem"] * 90 + ["phoma"] * 9 + ["cercosporiose"]
    sampler = dados.criar_sampler(pd.DataFrame({"classe": classes}))
    rotulos = np.array(classes)
    total = {c: float(sampler.weights[torch.from_numpy(rotulos == c)].sum()) for c in set(classes)}
    assert total == pytest.approx({"ferrugem": 1.0, "phoma": 1.0, "cercosporiose": 1.0})
    assert sampler.num_samples == len(classes) and sampler.replacement


def test_sampler_sorteia_as_classes_em_proporcao_parecida():
    classes = ["ferrugem"] * 900 + ["phoma"] * 90 + ["cercosporiose"] * 10
    sorteados = [classes[i] for i in dados.criar_sampler(pd.DataFrame({"classe": classes}))]
    contagem = pd.Series(sorteados).value_counts(normalize=True)
    assert contagem.between(0.25, 0.42).all() and len(contagem) == 3


def test_transformacoes_entregam_tensor_canais_altura_largura():
    imagem = np.random.default_rng(0).integers(0, 256, (1024, 2048, 3), dtype=np.uint8)
    for transform in (dados.transformacao_treino(), dados.transformacao_avaliacao()):
        saida = transform(image=imagem)["image"]
        assert saida.shape == (3, dados.ALTURA_PADRAO, dados.LARGURA_PADRAO) == (3, 224, 448)
        assert saida.dtype == torch.float32
    avaliacao = dados.transformacao_avaliacao()
    assert torch.equal(avaliacao(image=imagem)["image"], avaliacao(image=imagem)["image"])


def test_leitura_converte_bgr_para_rgb(monkeypatch):
    bgr = np.zeros((2, 4, 3), np.uint8)
    bgr[..., 2] = 255  # vermelho, na ordem BGR do OpenCV
    monkeypatch.setattr(dados.cv2, "imread", lambda caminho, modo: bgr)
    rgb = dados.ler_imagem("qualquer.jpg")
    assert (rgb[..., 0] == 255).all() and (rgb[..., 2] == 0).all()


def test_imagem_ausente_da_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrada"):
        dados.ler_imagem(tmp_path / "nao_existe.jpg")
