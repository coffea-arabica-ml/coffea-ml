"""Testes do modo gerar do organize_dataset.py, com uma cópia mínima e falsa do BRACOL."""
import csv

import numpy as np
import pytest
from PIL import Image

import bracol
import organize_dataset as od

# id, predominant_stress, miner, rust, phoma, cercospora, severity
LINHAS_CSV = [
    (1, 0, 0, 0, 0, 0, 0),
    (2, 1, 1, 0, 0, 0, 1),
    (3, 2, 0, 1, 0, 0, 2),
    (4, 3, 0, 0, 1, 0, 1),
    (5, 4, 0, 0, 0, 1, 1),
    (6, 5, 0, 1, 0, 1, 2),  # classe 5
    (7, 2, 0, 1, 0, 0, 1),  # sem imagem: o único ausente esperado nestes testes
]


def pasta_imagens(repo):
    return repo / "data" / "raw" / "bracol" / "copia" / "leaf" / "images"


def escrever_csv(repo, linhas):
    pasta_imagens(repo).mkdir(parents=True, exist_ok=True)
    with open(pasta_imagens(repo).parent / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f, lineterminator="\n")
        escritor.writerow(bracol.COLUNAS_CSV)
        escritor.writerows(linhas)


def salvar_imagem(caminho, semente):
    """JPEG pequeno de ruído: cada semente dá um pHash bem diferente dos outros."""
    ruido = np.random.default_rng(semente).integers(0, 256, (32, 64, 3), dtype=np.uint8)
    Image.fromarray(ruido).save(caminho, quality=90)


@pytest.fixture
def repo(tmp_path):
    """Repositório falso com uma cópia mínima do BRACOL em data/raw/bracol/copia/leaf/."""
    escrever_csv(tmp_path, LINHAS_CSV)
    for i in range(1, 7):
        salvar_imagem(pasta_imagens(tmp_path) / f"{i}.jpg", semente=i)
    return tmp_path


def gerar(repo, **opcoes):
    """Roda o modo gerar no repositório falso."""
    padrao = dict(
        entrada=repo / "data" / "raw" / "bracol",
        manifest=repo / "data" / "manifests" / "bracol.csv",
        relatorio=repo / "data" / "reports" / "integridade_bracol.md",
        raiz=repo,
        ausentes_esperados={7},
    )
    return od.gerar(**{**padrao, **opcoes})


def manifest(repo):
    with open(repo / "data" / "manifests" / "bracol.csv", newline="", encoding="utf-8") as f:
        return {int(linha["id"]): linha for linha in csv.DictReader(f)}


def relatorio(repo):
    return (repo / "data" / "reports" / "integridade_bracol.md").read_text(encoding="utf-8")


def test_gera_uma_linha_por_id_do_csv(repo):
    assert gerar(repo) == 0
    with open(repo / "data" / "manifests" / "bracol.csv", newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == bracol.COLUNAS_MANIFEST
        assert [int(linha["id"]) for linha in leitor] == list(range(1, 8))
    assert "**OK:** nenhum erro" in relatorio(repo)
    assert "Cópia PARCIAL" in relatorio(repo)


def test_linhas_elegiveis_tem_classe_split_e_hashes(repo):
    gerar(repo)
    m = manifest(repo)
    esperado = {1: "saudavel", 2: "bicho_mineiro", 3: "ferrugem", 4: "phoma", 5: "cercosporiose"}
    for i, classe in esperado.items():
        assert m[i]["classe"] == classe
        assert m[i]["split"] in bracol.SPLITS
        assert (m[i]["excluida"], m[i]["motivo_exclusao"]) == ("0", "")
    assert m[1]["caminho"] == "data/raw/bracol/copia/leaf/images/1.jpg"
    assert (m[1]["largura"], m[1]["altura"]) == ("64", "32")
    assert len(m[1]["sha256"]) == 64 and len(m[1]["phash"]) == 64
    assert m[1]["grupo"] == "bracol-1"


def test_classe_5_e_ausente_ficam_excluidos_e_sem_split(repo):
    gerar(repo)
    m = manifest(repo)
    assert (m[6]["presente"], m[6]["classe"], m[6]["split"]) == ("1", "", "")
    assert (m[6]["excluida"], m[6]["motivo_exclusao"]) == ("1", "predominant_stress_5")
    assert (m[7]["presente"], m[7]["caminho"], m[7]["sha256"], m[7]["split"]) == ("0", "", "", "")
    assert (m[7]["excluida"], m[7]["motivo_exclusao"]) == ("1", "imagem_ausente")


def test_rodar_de_novo_gera_os_mesmos_bytes(repo):
    gerar(repo)
    arquivo_manifest = repo / "data" / "manifests" / "bracol.csv"
    arquivo_relatorio = repo / "data" / "reports" / "integridade_bracol.md"
    primeiro_manifest = arquivo_manifest.read_bytes()
    primeiro_relatorio = arquivo_relatorio.read_bytes()
    gerar(repo)  # agora com o manifest anterior: a divisão é mantida
    assert arquivo_manifest.read_bytes() == primeiro_manifest
    assert arquivo_relatorio.read_bytes() == primeiro_relatorio
    gerar(repo, refazer_divisao=True)  # do zero, com a mesma seed
    assert arquivo_manifest.read_bytes() == primeiro_manifest
    assert b"\r\n" not in primeiro_manifest and b"\r\n" not in primeiro_relatorio


def test_divisao_do_manifest_anterior_e_mantida(repo):
    gerar(repo)
    m = manifest(repo)
    outro = next(s for s in bracol.SPLITS if s != m[1]["split"])
    m[1]["split"] = outro  # simula uma divisão anterior diferente da que sairia agora
    od.escrever_manifest([m[i] for i in sorted(m)], repo / "data" / "manifests" / "bracol.csv")
    assert gerar(repo) == 0
    assert manifest(repo)[1]["split"] == outro


def test_imagem_sem_linha_no_csv_e_erro_e_nao_grava_o_manifest(repo):
    salvar_imagem(pasta_imagens(repo) / "8.jpg", semente=8)
    assert gerar(repo) == 1
    assert not (repo / "data" / "manifests" / "bracol.csv").exists()
    assert "imagem sem linha no dataset.csv: 8.jpg" in relatorio(repo)


def test_jpeg_truncado_e_erro(repo):
    imagem = pasta_imagens(repo) / "3.jpg"
    imagem.write_bytes(imagem.read_bytes()[:300])
    assert gerar(repo) == 1
    assert "imagem ilegível: 3.jpg" in relatorio(repo)


def test_imagem_ausente_fora_da_lista_esperada_e_erro(repo):
    (pasta_imagens(repo) / "2.jpg").unlink()
    assert gerar(repo) == 1
    assert "imagens ausentes fora da lista esperada: ids 2" in relatorio(repo)


@pytest.mark.parametrize(("valor", "mensagem"), [("7", "fora de 0 a 5"), ("x", "não é inteiro")])
def test_valor_invalido_no_csv_e_erro(repo, valor, mensagem):
    linhas = [list(linha) for linha in LINHAS_CSV]
    linhas[2][1] = valor  # predominant_stress do id 3
    escrever_csv(repo, linhas)
    assert gerar(repo) == 1
    assert mensagem in relatorio(repo)


def test_arquivo_fora_do_padrao_vira_aviso(repo):
    (pasta_imagens(repo) / "Thumbs.db").write_bytes(b"cache do Windows")
    assert gerar(repo) == 0
    assert "ignorado em images/: Thumbs.db" in relatorio(repo)


def test_entrada_ambigua_e_erro_fatal(repo):
    outra = repo / "data" / "raw" / "bracol" / "outra"
    (outra / "images").mkdir(parents=True)
    (outra / "dataset.csv").write_text("id\n", encoding="utf-8")
    with pytest.raises(od.ErroFatal, match="exatamente 1"):
        gerar(repo)


def test_entrada_fora_do_repositorio_e_erro_fatal(repo, tmp_path_factory):
    with pytest.raises(od.ErroFatal, match="dentro do repositório"):
        gerar(repo, raiz=tmp_path_factory.mktemp("outro_repo"))


def test_saida_de_terminal_sem_acento(repo, capsys):
    (pasta_imagens(repo) / "2.jpg").unlink()  # força mensagens de erro, escritas com acento
    gerar(repo)
    saida = capsys.readouterr().out
    assert saida.isascii()
    assert "Manifest NAO gravado" in saida


# ------------------------------------------------------------------------- --verificar
def reescrever_manifest(repo, mudancas):
    """Simula um manifest versionado editado. mudancas: {id: {coluna: valor}}."""
    m = manifest(repo)
    for i, colunas in mudancas.items():
        m[i].update(colunas)
    od.escrever_manifest([m[i] for i in sorted(m)], repo / "data" / "manifests" / "bracol.csv")


def test_verificar_confere_sem_alterar_nada(repo):
    gerar(repo)
    arquivo_manifest = repo / "data" / "manifests" / "bracol.csv"
    arquivo_relatorio = repo / "data" / "reports" / "integridade_bracol.md"
    antes = arquivo_manifest.read_bytes(), arquivo_relatorio.read_bytes()
    assert gerar(repo, verificar=True) == 0
    assert (arquivo_manifest.read_bytes(), arquivo_relatorio.read_bytes()) == antes


def test_verificar_acusa_manifest_editado_e_nao_o_corrige(repo):
    gerar(repo)
    reescrever_manifest(repo, {1: {"severity": "1"}})
    assert gerar(repo, verificar=True) == 1
    assert "coluna severity diverge do manifest versionado: ids 1" in relatorio(repo)
    assert manifest(repo)[1]["severity"] == "1"  # o --verificar nunca grava o manifest


def test_verificar_acusa_imagem_trocada(repo):
    gerar(repo)
    salvar_imagem(pasta_imagens(repo) / "2.jpg", semente=99)
    assert gerar(repo, verificar=True) == 1
    assert "coluna sha256 diverge do manifest versionado: ids 2" in relatorio(repo)
    assert "coluna phash diverge do manifest versionado: ids 2" in relatorio(repo)


def test_verificar_tolera_poucos_bits_de_diferenca_no_phash(repo):
    gerar(repo)
    original = manifest(repo)[1]["phash"]
    poucos = f"{int(original, 16) ^ 0b111:064x}"  # 3 bits: outro decodificador JPEG, por exemplo
    reescrever_manifest(repo, {1: {"phash": poucos}})
    assert gerar(repo, verificar=True) == 0
    muitos = f"{int(original, 16) ^ ((1 << 20) - 1):064x}"  # 20 bits: acima da tolerância
    reescrever_manifest(repo, {1: {"phash": muitos}})
    assert gerar(repo, verificar=True) == 1


def test_verificar_sem_manifest_e_erro_fatal(repo):
    with pytest.raises(od.ErroFatal, match="rode sem --verificar"):
        gerar(repo, verificar=True)


def test_verificar_e_refazer_divisao_nao_combinam():
    with pytest.raises(SystemExit):
        od.main(["--verificar", "--refazer-divisao"])


# ------------------------------------------------------------------------ ler_manifest
def test_ler_manifest_devolve_elegiveis_com_tipos_fixos(repo):
    gerar(repo)
    caminho = repo / "data" / "manifests" / "bracol.csv"
    elegiveis = bracol.ler_manifest(caminho)
    assert sorted(elegiveis["id"]) == [1, 2, 3, 4, 5]
    assert elegiveis["id"].dtype == "int64" and elegiveis["severity"].dtype == "int64"
    assert elegiveis["classe"].map(bracol.CLASSES.index).between(0, 4).all()
    todas = bracol.ler_manifest(caminho, incluir_excluidas=True)
    assert len(todas) == 7
    assert todas.loc[todas["id"] == 6, "classe"].item() == ""  # classe 5: texto vazio, não NaN
    assert todas.loc[todas["id"] == 7, "largura"].isna().all()  # sem imagem
    for split in bracol.SPLITS:
        assert set(bracol.ler_manifest(caminho, split=split)["split"]) <= {split}
    with pytest.raises(ValueError, match="split desconhecido"):
        bracol.ler_manifest(caminho, split="validacao")
