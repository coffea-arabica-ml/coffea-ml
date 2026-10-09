"""Testes do jmuben.py com uma cópia mínima e falsa do JMuBEN (não precisam das imagens reais)."""
import csv
import random

import numpy as np
import pytest
from PIL import Image

import bracol
import jmuben


def imagem_suave(semente, tamanho=(64, 64)):
    """Imagem RGB lisa (soma de manchas gaussianas). O pHash dela é estável, e sementes
    diferentes dão recortes bem diferentes (98 bits ou mais entre as sementes de 1 a 20)."""
    rng = np.random.default_rng(semente)
    largura, altura = tamanho
    y, x = np.mgrid[0:altura, 0:largura]
    canais = []
    for _ in range(3):
        canal = np.zeros((altura, largura))
        for _ in range(6):
            cx, cy, s = rng.uniform(0, largura), rng.uniform(0, altura), rng.uniform(4, 16)
            canal += rng.uniform(-1, 1) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        canais.append(canal)
    matriz = np.stack(canais, axis=-1)
    matriz = (matriz - matriz.min()) / (matriz.max() - matriz.min()) * 200 + 25
    return Image.fromarray(matriz.astype(np.uint8))


def pasta(repo, nome):
    return repo / "data" / "raw" / "jmuben" / nome


@pytest.fixture
def repo(tmp_path):
    """Repositório falso: 3 recortes por pasta, mais uma cópia exata, uma cópia girada, uma
    variante mais escura e um arquivo ilegível."""
    semente = 0
    for nome in jmuben.PASTAS:
        pasta(tmp_path, nome).mkdir(parents=True)
        for k in range(1, 4):
            semente += 1
            imagem_suave(semente).save(pasta(tmp_path, nome) / f"1 ({k}).png")
    miner = pasta(tmp_path, "Miner")
    (miner / "2 (1).png").write_bytes((miner / "1 (1).png").read_bytes())  # cópia exata
    with Image.open(miner / "1 (2).png") as im:
        im.transpose(Image.Transpose.ROTATE_90).save(miner / "3 (2).png")  # girada
    with Image.open(pasta(tmp_path, "Phoma") / "1 (1).png") as im:
        im.point(lambda v: int(v * 0.8)).save(pasta(tmp_path, "Phoma") / "4 (1).png")  # escura
    (pasta(tmp_path, "Healthy") / "9 (9).jpg").write_bytes(b"\xff\xd8 nao e um JPEG inteiro")
    return tmp_path


def gerar(repo, **opcoes):
    padrao = dict(
        entrada=repo / "data" / "raw" / "jmuben",
        manifest=repo / "data" / "manifests" / "jmuben.csv",
        relatorio=repo / "data" / "reports" / "integridade_jmuben.md",
        raiz=repo,
        ilegiveis_esperados=frozenset({"Healthy/9 (9).jpg"}),
    )
    return jmuben.gerar(**{**padrao, **opcoes})


def manifest(repo):
    with open(repo / "data" / "manifests" / "jmuben.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def por_arquivo(repo):
    """{"<Pasta>/<nome>": linha} pela primeira cópia de cada conteúdo."""
    return {linha["caminho"].removeprefix("data/raw/jmuben/"): linha for linha in manifest(repo)}


def relatorio(repo):
    return (repo / "data" / "reports" / "integridade_jmuben.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------- hash canônico
@pytest.mark.parametrize("tamanho", [(64, 64), (48, 64)])
def test_hash_canonico_igual_nas_8_transformacoes(tamanho):
    im = imagem_suave(7, tamanho)
    base = jmuben.hash_canonico(im)
    assert len(base) == 64  # 256 bits em hexadecimal
    for transformacao in jmuben._TRANSFORMACOES[1:]:
        assert jmuben.hash_canonico(im.transpose(transformacao)) == base


def test_hash_canonico_estavel_com_mudanca_de_brilho():
    for semente in (1, 2, 3):
        im = imagem_suave(semente)
        base = jmuben.hash_canonico(im)
        for fator in (0.8, 1.15):
            variante = im.point(lambda v: min(255, int(v * fator)))
            assert bracol.distancia_hamming(base, jmuben.hash_canonico(variante)) <= \
                jmuben.LIMIAR_HASH_CANONICO


def test_recortes_diferentes_ficam_longe():
    a, b = jmuben.hash_canonico(imagem_suave(1)), jmuben.hash_canonico(imagem_suave(2))
    assert bracol.distancia_hamming(a, b) > jmuben.LIMIAR_HASH_CANONICO


# --------------------------------------------------------------------------- seleção
def linhas_sinteticas():
    """Por classe, 6 grupos de 1 a 3 membros, e uma linha excluída que nunca entra."""
    linhas = []
    for classe in bracol.CLASSES:
        for g in range(6):
            grupo = f"jmuben-{len(linhas) + 1}"
            for _ in range(1 + g % 3):
                linhas.append({"id": len(linhas) + 1, "classe": classe, "grupo": grupo,
                               "uso": jmuben.USO_TREINO})
        linhas.append({"id": len(linhas) + 1, "classe": classe, "grupo": "",
                       "uso": jmuben.USO_EXCLUIDA})
    return linhas


def test_selecao_deterministica_um_por_grupo_e_no_teto():
    linhas = linhas_sinteticas()
    teto = dict.fromkeys(bracol.CLASSES, 4)
    escolhidos = jmuben.selecionar(linhas, teto)
    for _ in range(5):
        embaralhadas = linhas[:]
        random.Random(_).shuffle(embaralhadas)
        assert jmuben.selecionar(embaralhadas, teto) == escolhidos
    por_id = {linha["id"]: linha for linha in linhas}
    assert all(por_id[i]["uso"] == jmuben.USO_TREINO for i in escolhidos)
    for classe in bracol.CLASSES:
        da_classe = [por_id[i] for i in escolhidos if por_id[i]["classe"] == classe]
        assert len(da_classe) == 4
        assert len({linha["grupo"] for linha in da_classe}) == 4  # um por grupo


def test_teto_maior_que_os_grupos_nao_repete_grupo():
    linhas = linhas_sinteticas()
    escolhidos = jmuben.selecionar(linhas, dict.fromkeys(bracol.CLASSES, 100))
    assert len(escolhidos) == 6 * len(bracol.CLASSES)  # 6 grupos por classe, 1 de cada


def test_selecao_de_uma_classe_nao_depende_das_outras():
    linhas = linhas_sinteticas()
    teto = dict.fromkeys(bracol.CLASSES, 3)
    antes = jmuben.selecionar(linhas, teto)
    sem_ferrugem = [linha for linha in linhas if linha["classe"] != "ferrugem"]
    outro_teto = {**teto, "phoma": 1}
    depois = jmuben.selecionar(sem_ferrugem, outro_teto)
    classe = {linha["id"]: linha["classe"] for linha in linhas}
    for c in ("saudavel", "bicho_mineiro", "cercosporiose"):
        assert {i for i in antes if classe[i] == c} == {i for i in depois if classe[i] == c}


def test_classe_fora_do_teto_fica_sem_selecao():
    linhas = linhas_sinteticas()
    escolhidos = jmuben.selecionar(linhas, {"saudavel": 2})
    assert {linha["classe"] for linha in linhas if linha["id"] in escolhidos} == {"saudavel"}


def test_teto_padrao_e_metade_do_treino_do_bracol():
    # Treino do BRACOL por classe (data/README.md): 190, 372, 271, 244 e 103.
    treino = {"saudavel": 190, "ferrugem": 372, "bicho_mineiro": 271, "phoma": 244,
              "cercosporiose": 103}
    assert jmuben.CAP_POR_CLASSE == {c: n // 2 for c, n in treino.items()}


# --------------------------------------------------------------------------- gerar
def test_gera_uma_linha_por_conteudo_distinto_sem_split(repo):
    assert gerar(repo) == 0
    linhas = manifest(repo)
    assert list(linhas[0]) == jmuben.COLUNAS_MANIFEST and "split" not in linhas[0]
    # 15 recortes + girada + escura + ilegível; a cópia exata não vira linha
    assert len(linhas) == 18
    assert [int(linha["id"]) for linha in linhas] == list(range(1, 19))
    assert [linha["caminho"] for linha in linhas] == sorted(linha["caminho"] for linha in linhas)
    assert {linha["fonte"] for linha in linhas} == {"jmuben"}
    assert "**OK:** nenhum erro" in relatorio(repo)


def test_colunas_de_cada_linha(repo):
    gerar(repo)
    m = por_arquivo(repo)
    assert m["Miner/1 (1).png"]["copias"] == "2"  # a cópia exata "2 (1).png" conta aqui
    assert (m["Miner/1 (1).png"]["classe"], m["Miner/1 (1).png"]["subconjunto"]) == \
        ("bicho_mineiro", "jmuben2")
    assert (m["Cerscospora/1 (1).png"]["classe"], m["Cerscospora/1 (1).png"]["subconjunto"]) == \
        ("cercosporiose", "jmuben")
    assert (m["Leaf rust/1 (1).png"]["largura"], m["Leaf rust/1 (1).png"]["altura"]) == ("64", "64")
    assert len(m["Phoma/1 (2).png"]["sha256"]) == 64
    assert len(m["Phoma/1 (2).png"]["phash_canonico"]) == 64


def test_copia_girada_e_variante_escura_ficam_no_grupo_do_original(repo):
    gerar(repo)
    m = por_arquivo(repo)
    assert m["Miner/3 (2).png"]["grupo"] == m["Miner/1 (2).png"]["grupo"]
    assert m["Phoma/4 (1).png"]["grupo"] == m["Phoma/1 (1).png"]["grupo"]
    assert m["Phoma/1 (1).png"]["grupo"] == f"jmuben-{m['Phoma/1 (1).png']['id']}"
    grupos = {linha["grupo"] for linha in manifest(repo) if linha["grupo"]}
    assert len(grupos) == 15  # um por recorte distinto


def test_um_selecionado_por_grupo_e_motivo_nos_outros(repo):
    gerar(repo)
    linhas = [linha for linha in manifest(repo) if linha["uso"] == jmuben.USO_TREINO]
    selecionados = {}
    for linha in linhas:
        if linha["selecionada"] == "1":
            assert linha["grupo"] not in selecionados
            selecionados[linha["grupo"]] = linha["id"]
            assert linha["motivo"] == ""
        else:
            assert linha["motivo"] == jmuben.MOTIVO_GRUPO_REPRESENTADO
    assert len(selecionados) == 15  # o teto padrão é maior que os grupos


def test_ilegivel_esperado_fica_excluido_com_motivo(repo):
    gerar(repo)
    ilegivel = por_arquivo(repo)["Healthy/9 (9).jpg"]
    assert (ilegivel["uso"], ilegivel["motivo"], ilegivel["selecionada"]) == \
        (jmuben.USO_EXCLUIDA, jmuben.MOTIVO_ILEGIVEL, "0")
    assert (ilegivel["grupo"], ilegivel["phash_canonico"], ilegivel["largura"]) == ("", "", "")


def test_ilegivel_fora_da_lista_e_erro_e_nao_grava_o_manifest(repo):
    assert gerar(repo, ilegiveis_esperados=frozenset()) == 1
    assert not (repo / "data" / "manifests" / "jmuben.csv").exists()
    assert "arquivo ilegível: Healthy/9 (9).jpg" in relatorio(repo)


def test_teto_menor_deixa_grupos_de_fora(repo):
    assert gerar(repo, teto=dict.fromkeys(bracol.CLASSES, 1)) == 0
    linhas = manifest(repo)
    selecionadas = [linha for linha in linhas if linha["selecionada"] == "1"]
    assert sorted(linha["classe"] for linha in selecionadas) == sorted(bracol.CLASSES)
    assert jmuben.MOTIVO_ACIMA_DO_TETO in {linha["motivo"] for linha in linhas}


def test_pasta_de_classe_faltando_e_erro(repo):
    for arquivo in pasta(repo, "Phoma").iterdir():
        arquivo.unlink()
    pasta(repo, "Phoma").rmdir()
    assert gerar(repo) == 1
    assert "pasta de classe não encontrada: Phoma" in relatorio(repo)


def test_rodar_de_novo_gera_os_mesmos_bytes(repo):
    gerar(repo)
    arquivo_manifest = repo / "data" / "manifests" / "jmuben.csv"
    arquivo_relatorio = repo / "data" / "reports" / "integridade_jmuben.md"
    antes = arquivo_manifest.read_bytes(), arquivo_relatorio.read_bytes()
    gerar(repo)
    assert (arquivo_manifest.read_bytes(), arquivo_relatorio.read_bytes()) == antes
    assert b"\r\n" not in antes[0] and b"\r\n" not in antes[1]


def test_saida_de_terminal_sem_acento(repo, capsys):
    gerar(repo, ilegiveis_esperados=frozenset())  # força a mensagem de erro, escrita com acento
    saida = capsys.readouterr().out
    assert saida.isascii()
    assert "Manifest NAO gravado" in saida


# ----------------------------------------------------------------------- --verificar
def test_verificar_confere_sem_alterar_nada(repo):
    gerar(repo)
    arquivo_manifest = repo / "data" / "manifests" / "jmuben.csv"
    antes = arquivo_manifest.read_bytes()
    assert gerar(repo, verificar=True) == 0
    assert arquivo_manifest.read_bytes() == antes


def test_verificar_acusa_arquivo_novo_e_arquivo_trocado(repo):
    gerar(repo)
    miner = pasta(repo, "Miner")
    (miner / "5 (1).png").write_bytes((miner / "1 (1).png").read_bytes())  # mais uma cópia
    assert gerar(repo, verificar=True) == 1
    assert "coluna copias diverge do manifest versionado" in relatorio(repo)
    imagem_suave(99).save(pasta(repo, "Leaf rust") / "1 (2).png")  # conteúdo trocado
    assert gerar(repo, verificar=True) == 1
    assert "coluna sha256 diverge do manifest versionado" in relatorio(repo)


def test_verificar_tolera_poucos_bits_no_hash_canonico(repo):
    gerar(repo)
    arquivo = repo / "data" / "manifests" / "jmuben.csv"
    linhas = manifest(repo)
    original = linhas[0]["phash_canonico"]
    linhas[0]["phash_canonico"] = f"{int(original, 16) ^ 0b111:064x}"  # 3 bits
    jmuben.od.escrever_manifest(linhas, arquivo, colunas=jmuben.COLUNAS_MANIFEST)
    assert gerar(repo, verificar=True) == 0
    linhas[0]["phash_canonico"] = f"{int(original, 16) ^ ((1 << 20) - 1):064x}"  # 20 bits
    jmuben.od.escrever_manifest(linhas, arquivo, colunas=jmuben.COLUNAS_MANIFEST)
    assert gerar(repo, verificar=True) == 1
    assert "coluna phash_canonico diverge do manifest versionado: ids 1" in relatorio(repo)


def test_verificar_sem_manifest_e_erro_fatal(repo):
    with pytest.raises(jmuben.od.ErroFatal, match="rode sem --verificar"):
        gerar(repo, verificar=True)


# ---------------------------------------------------------------------- ler_manifest
@pytest.fixture
def caminho_manifest(repo):
    """Manifest falso com 3 grupos selecionados por classe."""
    for nome in jmuben.PASTAS:
        for k in range(4, 6):  # mais 2 recortes distintos por pasta: 5 grupos
            imagem_suave(100 + 10 * list(jmuben.PASTAS).index(nome) + k).save(
                pasta(repo, nome) / f"1 ({k}).png")
    assert gerar(repo, teto=dict.fromkeys(bracol.CLASSES, 3)) == 0
    return repo / "data" / "manifests" / "jmuben.csv"


def test_ler_manifest_devolve_so_as_selecionadas(caminho_manifest):
    df = jmuben.ler_manifest(caminho_manifest)
    assert len(df) == 3 * len(bracol.CLASSES)
    assert set(df["uso"]) == {jmuben.USO_TREINO} and set(df["selecionada"]) == {1}
    assert "split" not in df.columns
    assert df["id"].dtype == "int64" and df["copias"].dtype == "int64"
    assert df["classe"].map(bracol.CLASSES.index).between(0, 4).all()


def test_teto_do_leitor_corta_em_subconjuntos_encaixados(caminho_manifest):
    ids = {t: set(jmuben.ler_manifest(caminho_manifest, teto=t)["id"]) for t in (0, 1, 2, 3, 50)}
    assert ids[0] == set()
    assert ids[1] <= ids[2] <= ids[3] == ids[50]  # o teto só reduz
    assert len(ids[1]) == len(bracol.CLASSES) and len(ids[2]) == 2 * len(bracol.CLASSES)
    so_saudavel = jmuben.ler_manifest(caminho_manifest, teto={"saudavel": 0})
    assert "saudavel" not in set(so_saudavel["classe"])
    assert len(so_saudavel) == 3 * (len(bracol.CLASSES) - 1)  # as outras classes ficam inteiras


@pytest.mark.parametrize("teto", [-1, 1.5, "10", {"folha": 3}, {"saudavel": -2}, True])
def test_teto_invalido_e_erro(caminho_manifest, teto):
    with pytest.raises(ValueError):
        jmuben.ler_manifest(caminho_manifest, teto=teto)
