"""Testes do treino (model/train.py): perda multitarefa com a severidade mascarada, formas das
saídas do modelo e o fluxo de ponta a ponta (treino e --avaliar-teste).

Nenhum teste lê data/raw nem baixa pesos (pretrained=False). O fluxo de ponta a ponta usa o
backbone minúsculo test_resnet do timm e imagens sintéticas geradas na memória.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import bracol
import dados
import train


# ------------------------------------------------------------------- modelo e perda
def test_saidas_do_modelo_tem_formas_batch_por_5():
    modelo = train.criar_modelo("resnet50", pretrained=False).eval()
    with torch.no_grad():
        classe, severidade = modelo(torch.randn(3, 3, 112, 224))
    assert classe.shape == (3, len(bracol.CLASSES)) == (3, 5)
    assert severidade.shape == (3, len(bracol.SEVERIDADES)) == (3, 5)


def test_severidade_menos_1_fica_fora_da_perda():
    torch.manual_seed(0)
    logits_classe, logits_sev = torch.randn(4, 5), torch.randn(4, 5)
    classes = torch.tensor([0, 1, 2, 3])
    severidades = torch.tensor([2, -1, 0, -1])
    total, perda_classe, perda_sev = train.perda_multitarefa(
        logits_classe, logits_sev, classes, severidades, peso_sev=0.5
    )
    so_com_alvo = F.cross_entropy(logits_sev[[0, 2]], severidades[[0, 2]])
    assert torch.isclose(perda_sev, so_com_alvo)
    assert torch.isclose(perda_classe, F.cross_entropy(logits_classe, classes))
    assert torch.isclose(total, perda_classe + 0.5 * so_com_alvo)


def test_lote_sem_nenhum_alvo_de_severidade_nao_da_nan():
    logits_classe = torch.randn(3, 5, requires_grad=True)
    logits_sev = torch.randn(3, 5, requires_grad=True)
    total, perda_classe, perda_sev = train.perda_multitarefa(
        logits_classe, logits_sev, torch.tensor([0, 1, 4]), torch.full((3,), -1)
    )
    assert perda_sev.item() == 0.0
    assert torch.isfinite(total) and torch.isclose(total, perda_classe)
    total.backward()
    assert torch.isfinite(logits_classe.grad).all()
    assert (logits_sev.grad == 0).all()


# --------------------------------------------------------- pesos pré-treinados (cache)
@pytest.fixture
def create_model_espiao(monkeypatch):
    """Troca timm.create_model por um espião: guarda os argumentos e devolve o backbone
    minúsculo test_resnet sem pesos, sem baixar nada."""
    original = train.timm.create_model
    chamadas = []

    def espiao(nome, **kwargs):
        chamadas.append({"nome": nome, **kwargs})
        return original("test_resnet", pretrained=False, num_classes=0)

    monkeypatch.setattr(train.timm, "create_model", espiao)
    return chamadas


def test_com_pesos_no_cache_o_timm_recebe_o_arquivo(monkeypatch, create_model_espiao):
    monkeypatch.setattr(train, "pesos_em_cache", lambda backbone: "cache/model.safetensors")
    train.criar_modelo("resnet50", pretrained=True)
    assert create_model_espiao == [{
        "nome": "resnet50", "pretrained": True, "num_classes": 0,
        "pretrained_cfg_overlay": {"file": "cache/model.safetensors"},
    }]


def test_sem_pesos_no_cache_o_timm_baixa_normalmente(monkeypatch, capsys, create_model_espiao):
    # Máquina nova ou Colab: sem o arquivo no cache, o timm segue o caminho normal (baixa do
    # huggingface). Nada força o modo offline.
    monkeypatch.setattr(train, "pesos_em_cache", lambda backbone: None)
    train.criar_modelo("resnet50", pretrained=True)
    assert create_model_espiao == [{"nome": "resnet50", "pretrained": True, "num_classes": 0}]
    assert "Baixando os pesos pre-treinados de resnet50" in capsys.readouterr().out


def test_sem_pretreino_nao_procura_pesos(monkeypatch, create_model_espiao):
    def nao_chamar(backbone):
        raise AssertionError("não devia procurar pesos")

    monkeypatch.setattr(train, "pesos_em_cache", nao_chamar)
    train.criar_modelo("resnet50", pretrained=False)
    assert create_model_espiao == [{"nome": "resnet50", "pretrained": False, "num_classes": 0}]


def test_pesos_em_cache_so_le_o_cache_local(monkeypatch, tmp_path):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    assert train.pesos_em_cache("resnet50") is None  # cache vazio: o timm vai baixar
    # Cache no formato do huggingface: refs/main aponta para o snapshot com o arquivo.
    repositorio = tmp_path / "models--timm--resnet50.a1_in1k"
    commit = "0123456789abcdef0123456789abcdef01234567"
    (repositorio / "refs").mkdir(parents=True)
    (repositorio / "refs" / "main").write_text(commit)
    (repositorio / "snapshots" / commit).mkdir(parents=True)
    (repositorio / "snapshots" / commit / "model.safetensors").write_bytes(b"pesos")
    assert Path(train.pesos_em_cache("resnet50")) == (
        repositorio / "snapshots" / commit / "model.safetensors"
    )


# ------------------------------------------------------------------- ponta a ponta
ARGS_PEQUENOS = ["--backbone", "test_resnet", "--sem-pretreino", "--largura", "128",
                 "--altura", "64", "--batch", "8", "--workers", "0", "--dispositivo", "cpu"]
ARQUIVOS_DO_RUN = ["config.json", "historico.csv", "matriz_confusao_val.png", "melhor.pt",
                   "metricas_val.json", "predicoes_val.csv", "relatorio.md"]


def _conjunto(n: int, primeiro_id: int) -> pd.DataFrame:
    """n linhas sintéticas no formato de dados.COLUNAS, com as 5 classes em rodízio."""
    classes = [bracol.CLASSES[i % 5] for i in range(n)]
    return pd.DataFrame({
        "fonte": "bracol",
        "id": range(primeiro_id, primeiro_id + n),
        "caminho": [f"sintetico/{primeiro_id + i}.jpg" for i in range(n)],
        "classe": classes,
        "severity": [0 if c == "saudavel" else 1 + i % 4 for i, c in enumerate(classes)],
    })


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """model/runs/ num diretório temporário; treino (20), validação (10) e teste (10)
    sintéticos, com imagens geradas na hora."""
    monkeypatch.setattr(train, "PASTA_RUNS", tmp_path)
    monkeypatch.setattr(train, "PROGRESSO_A_CADA", 1)  # passa pela mensagem de progresso
    monkeypatch.setattr(dados, "montar_treino", lambda: _conjunto(20, 1))
    monkeypatch.setattr(dados, "montar_val", lambda: _conjunto(10, 101))
    monkeypatch.setattr(dados, "montar_teste", lambda: _conjunto(10, 201))
    gerador = np.random.default_rng(0)
    monkeypatch.setattr(dados, "ler_imagem",
                        lambda caminho: gerador.integers(0, 256, (64, 128, 3), dtype=np.uint8))
    return tmp_path


def test_treino_grava_o_run_completo(runs, capsys):
    assert train.main(["--nome", "t", "--epocas", "2", *ARGS_PEQUENOS]) == 0
    saida = capsys.readouterr().out
    assert "lote 1/3 | perda " in saida and "Epoca 2/2 | perda treino" in saida
    assert saida.isascii()  # terminal sem acento nem símbolos
    pasta = runs / "t"
    assert sorted(p.name for p in pasta.iterdir()) == ARQUIVOS_DO_RUN

    historico = pd.read_csv(pasta / "historico.csv")
    assert historico["epoca"].tolist() == [1, 2]
    for coluna in ["perda_treino", "perda_val", "acuracia_val", "f1_macro_val", "mae_sev_val",
                   "kappa_sev_val", "tempo_s", "memoria_gpu_gb",
                   *[f"recall_val_{c}" for c in bracol.CLASSES]]:
        assert coluna in historico.columns
    assert historico["memoria_gpu_gb"].isna().all()  # em CPU, sem memória de GPU

    config = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    assert config["backbone"] == "test_resnet" and config["dados"]["val"]["n"] == 10
    assert config["commit"] is None or len(config["commit"]) == 40

    checkpoint = torch.load(pasta / "melhor.pt", map_location="cpu", weights_only=True)
    assert checkpoint["classes"] == bracol.CLASSES
    assert checkpoint["tamanho_entrada"] == {"largura": 128, "altura": 64}
    assert checkpoint["media"] == list(dados.MEDIA) and checkpoint["ordem_canais"] == "RGB"

    predicoes = pd.read_csv(pasta / "predicoes_val.csv")
    probabilidades = [f"prob_{c}" for c in bracol.CLASSES]
    assert list(predicoes.columns) == ["id", "classe_verdadeira", "classe_prevista",
                                       *probabilidades, "severidade_verdadeira",
                                       "severidade_prevista"]
    assert predicoes["id"].tolist() == list(range(101, 111))
    assert np.allclose(predicoes[probabilidades].sum(axis=1), 1, atol=1e-4)

    relatorio = (pasta / "relatorio.md").read_text(encoding="utf-8")
    assert bracol.RESSALVA_PHOMA_CERCOSPORA in relatorio
    assert "Esgario et al. (2020)" in relatorio
    assert "Cercosporiose: 2 imagens na validação" in relatorio


def test_treino_nao_sobrescreve_um_run_existente(runs):
    assert train.main(["--nome", "t", "--epocas", "1", *ARGS_PEQUENOS]) == 0
    assert train.main(["--nome", "t", "--epocas", "1", *ARGS_PEQUENOS]) == 1


def test_avaliar_teste_grava_resultados_e_registra_cada_uso(runs):
    assert train.main(["--nome", "t", "--epocas", "1", *ARGS_PEQUENOS]) == 0
    avaliar = ["--avaliar-teste", "--run", "t", "--workers", "0", "--dispositivo", "cpu"]
    assert train.main(avaliar) == 0
    assert train.main(avaliar) == 0
    pasta = runs / "t"
    for nome in ["predicoes_teste.csv", "metricas_teste.json", "matriz_confusao_teste.png",
                 "relatorio_teste.md"]:
        assert (pasta / nome).is_file()
    assert pd.read_csv(pasta / "predicoes_teste.csv")["id"].tolist() == list(range(201, 211))
    relatorio = (pasta / "relatorio_teste.md").read_text(encoding="utf-8")
    assert "Cercosporiose: 2 imagens no teste" in relatorio
    assert bracol.RESSALVA_PHOMA_CERCOSPORA in relatorio
    uso = (runs / train.USO_DO_TESTE).read_text(encoding="utf-8").splitlines()
    assert sum(linha.endswith("| t |") for linha in uso) == 2


def test_avaliar_teste_exige_run_existente(runs):
    assert train.main(["--avaliar-teste", "--run", "nao_existe"]) == 1
    assert not (runs / train.USO_DO_TESTE).exists()
    with pytest.raises(SystemExit):
        train.main(["--avaliar-teste"])
