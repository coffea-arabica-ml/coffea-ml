"""Testes do passo 7 (model/teste_final.py e os parâmetros novos de avaliar_pipeline,
treinar_detector, campo e campo_cadeia_final): o log único, as conferências antes de abrir, a
rodada única com nova tentativa só depois de erro, o sorteio com no máximo 2 folhas por foto, o
T1 só no modo B, o avaliar_teste numa pasta nova e a análise da parte B.

Nada aqui lê os testes do BRACOL ou do BRACOT: fotos sintéticas, detector e classificador falsos
(de tests/test_inferencia.py), resumos de detecção sintéticos e tabelas pequenas.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import avaliar_pipeline as ap
import campo
import campo_cadeia_final as ccf
import campo_validacao as cv
import inferencia as inf
import teste_final as tf
import train
import treinar_detector as td
from test_detector import ANOTADAS, deteccao as deteccao_md, quadrado, resumo
from test_inferencia import deteccao, detectar_com, foto, modelos_falsos, probs  # noqa: F401


# ------------------------------------------------------------------------- conferências
def test_conferir_pesos_com_o_manifesto(tmp_path):
    peso = tmp_path / "a.pt"
    peso.write_bytes(b"pesos")
    sha = hashlib.sha256(b"pesos").hexdigest()
    manifesto = tmp_path / "MANIFESTO.md"
    manifesto.write_text(f"| `x/a.pt` | 5 | `{sha}` | teste |\n", encoding="utf-8")
    r = tf.conferir_pesos(manifesto, {"c": ("x/a.pt", peso)})
    assert r["c"] == {"arquivo": "x/a.pt", "bytes": 5, "sha256": sha}
    peso.write_bytes(b"outros")
    with pytest.raises(train.ErroFatal):
        tf.conferir_pesos(manifesto, {"c": ("x/a.pt", peso)})
    with pytest.raises(train.ErroFatal):
        tf.conferir_pesos(manifesto, {"c": ("x/b.pt", peso)})


def test_configuracao_do_codigo_e_a_declarada_pelo_gestor():
    efetiva = tf.configuracao_efetiva()
    tf.conferir_configuracao(efetiva)  # a cadeia congelada do passo 6
    assert efetiva["configuracao"]["decisao"] == "por_caminho"
    for mudanca in ({"limiar_doenca": 0.6}, {"recorte": "A"}, {"limiar_deteccao": 0.3}):
        errada = {**efetiva, "configuracao": {**efetiva["configuracao"], **mudanca}}
        with pytest.raises(train.ErroFatal):
            tf.conferir_configuracao(errada)
    with pytest.raises(train.ErroFatal):
        tf.conferir_configuracao({**efetiva, "pesos": {**efetiva["pesos"], "detector": "x.pt"}})


def test_alteracoes_de_codigo_ignoram_runs_e_dados():
    status = [" M model/inferencia.py", "?? model/runs/x/y.json", " M README.md",
              "?? tests/novo.py", "R  evaluation/a.py -> evaluation/b.py",
              "?? data/manifests/x.csv"]
    assert tf.alteracoes_de_codigo(status) == ["model/inferencia.py", "tests/novo.py",
                                               "evaluation/b.py"]


def test_abrir_cria_o_log_uma_vez_e_recusa_codigo_nao_commitado(tmp_path, monkeypatch):
    log = tmp_path / "uso.md"
    monkeypatch.setattr(tf, "conferir_pesos", lambda: {"classificador": {
        "arquivo": "m.pt", "bytes": 1, "sha256": "ab" * 32}})
    monkeypatch.setattr(tf, "estado_git", lambda: ("c0ffee" * 6, [" M tests/x.py"]))
    with pytest.raises(train.ErroFatal):
        tf.abrir(log)
    assert not log.exists()
    monkeypatch.setattr(tf, "estado_git", lambda: ("c0ffee" * 6, ["?? model/runs/a.json"]))
    assert tf.abrir(log) == 0
    texto = log.read_text(encoding="utf-8")
    for trecho in ("única abertura dos testes", "c0ffee", "`'por_caminho'`", "## O que será "
                   "medido", "Uso 3B", "semente 42", "| data e hora | uso | evento |"):
        assert trecho in texto, trecho
    with pytest.raises(train.ErroFatal):
        tf.abrir(log)


# ---------------------------------------------------------------- rodada única e log
@pytest.mark.parametrize("historico, nova, decisao, esperado", [
    ([], None, None, "teste_bracol"),
    ([], "x", None, "erro"),
    ([tf.INICIO, tf.ERRO], None, None, "erro"),
    ([tf.INICIO, tf.ERRO], "corrigido", None, "teste_bracol"),
    ([tf.INICIO], "morreu sem erro", None, "teste_bracol"),
    ([tf.INICIO, tf.FIM], None, None, "erro"),
    ([tf.INICIO, tf.FIM], None, "pode, 14/10/2026", "teste_bracol_rodada2"),
    ([tf.INICIO, tf.ERRO, tf.NOVA, tf.INICIO, tf.FIM], "x", None, "erro"),
])
def test_autorizar(historico, nova, decisao, esperado):
    if esperado == "erro":
        with pytest.raises(train.ErroFatal):
            tf.autorizar("1", historico, nova, decisao)
    else:
        assert tf.autorizar("1", historico, nova, decisao).name == esperado


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    """Log, pastas e conferências trocados por versões em tmp_path (nada do repositório)."""
    log = tmp_path / "uso.md"
    log.write_text("# log\n\n## Registro\n\n| data e hora | uso | evento | detalhes |\n"
                   "|---|---|---|---|\n", encoding="utf-8")
    monkeypatch.setattr(tf, "LOG", log)
    monkeypatch.setattr(tf, "PASTAS", {u: tmp_path / f"saida_{u}" for u in tf.USOS})
    for nome in ("preparar", "conferir_pesos", "conferir_configuracao"):
        monkeypatch.setattr(tf, nome, lambda *a, **k: None)
    monkeypatch.setattr(tf, "configuracao_efetiva", lambda: {})
    monkeypatch.setattr(tf, "estado_git", lambda: ("c0ffee" * 6, []))
    return tmp_path


def test_erro_apaga_a_temporaria_e_so_a_nova_tentativa_roda(ambiente):
    def falha(pasta):
        pasta.mkdir(parents=True)
        (pasta / "parcial.txt").write_text("x", encoding="utf-8")
        raise RuntimeError("bug")

    def certo(pasta):
        pasta.mkdir(parents=True)
        (pasta / "a.txt").write_text("ok", encoding="utf-8")
        return {"resumo": "pronto", "conferencias_falhas": ["T1 x contrato: 1 divergência"]}

    with pytest.raises(RuntimeError):
        tf.executar_uso("1", funcoes={"1": falha})
    assert not (ambiente / "saida_1").exists() and not (ambiente / "saida_1.tmp").exists()
    assert tf.eventos("1") == [tf.INICIO, tf.ERRO]
    with pytest.raises(train.ErroFatal):
        tf.executar_uso("1", funcoes={"1": certo})  # sem --nova-tentativa
    assert tf.executar_uso("1", nova_tentativa="corrigido o bug", funcoes={"1": certo}) == 0
    assert (ambiente / "saida_1" / "a.txt").read_text(encoding="utf-8") == "ok"
    assert tf.eventos("1") == [tf.INICIO, tf.ERRO, tf.NOVA, tf.INICIO, tf.CONFERENCIA, tf.FIM,
                               tf.ARQUIVO]
    assert hashlib.sha256(b"ok").hexdigest() in tf.LOG.read_text(encoding="utf-8")
    with pytest.raises(train.ErroFatal):  # já rodou: só com a decisão do gestor
        tf.executar_uso("1", funcoes={"1": certo})
    assert tf.eventos("2") == []


def test_uso_sem_log_nao_roda(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "LOG", tmp_path / "nao_existe.md")
    with pytest.raises(train.ErroFatal):
        tf.executar_uso("1", funcoes={"1": lambda p: {}})


# ---------------------------------------------------------- uso 3A: sorteio e pranchas
def test_sorteio_padrao_nao_muda_e_max_por_foto_limita():
    folhas = {"a": 3, "b": 1, "c": 4}
    assert campo.sortear_rotulagem(folhas, n=6, seed=42) == [
        ("c", 3), ("b", 1), ("a", 1), ("c", 1), ("a", 2), ("c", 4)]  # o de antes do passo 7
    limitado = campo.sortear_rotulagem(folhas, n=100, seed=42, max_por_foto=2)
    assert pd.Series([f for f, _ in limitado]).value_counts().to_dict() == {"a": 2, "c": 2,
                                                                            "b": 1}
    assert set(limitado) <= set(campo.sortear_rotulagem(folhas, n=100, seed=42))


def gravar_foto(caminho: Path, *folhas):
    cv2.imencode(".png", foto(*folhas)[..., ::-1])[1].tofile(caminho)
    return caminho


def test_rotulagem_de_teste_sem_classificador_e_com_contagens(tmp_path, detectar_com):  # noqa
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100), (400, 150, 520, 220)]
    detectar_com([deteccao(*f, 0.9) for f in folhas])
    caminhos = pd.Series({n: str(gravar_foto(tmp_path / f"{n}.png", *folhas))
                          for n in ("f1", "f2")})
    so_detector = inf.Modelos(classificador=None, transformacao=None, detector=None,
                              dispositivo=modelos_falsos([probs("saudavel", 0.9)], [0])
                              .dispositivo, nome_classificador=None, nome_detector="falso")
    contagens = {}
    csv = tmp_path / "campo_rotulado_teste.csv"
    r = campo.gerar_rotulagem(so_detector, tmp_path / "pranchas", csv, tf.CONFIG_ROTULAGEM_TESTE,
                              (["f1", "f2"], caminhos), n=4, max_por_foto=tf.MAX_POR_FOTO,
                              contagens=contagens)
    tabela = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert list(tabela.columns) == campo.COLUNAS_CAMPO and (tabela["rotulo"] == "").all()
    assert tabela["foto"].value_counts().to_dict() == {"f1": 2, "f2": 2}
    assert len(r["pranchas"]) == 1 and r["pranchas"][0].is_file()
    assert set(contagens) == {"f1", "f2"} and contagens["f1"]["plano"] == "deteccao"
    assert contagens["f1"]["folhas"] == 3 and "cortadas_por_fragmento" in contagens["f1"]
    resumo_a = tf.resumo_parte_a(pd.DataFrame([{"foto": f, **c, "sorteadas": 2}
                                               for f, c in contagens.items()]), 1)
    assert (resumo_a["folhas"], resumo_a["sorteadas"], resumo_a["fotos_por_sorteadas"]) == (
        6, 4, {2: 2})


# ----------------------------------------------------------------- uso 1: T1 no modo B
def test_t1_so_no_modo_b_com_outras_imagens(tmp_path, detectar_com):  # noqa: F811
    folhas = [(20, 20, 140, 80), (200, 30, 330, 100)]
    detectar_com([deteccao(*f, 0.9) for f in folhas])
    imagens = pd.DataFrame({"id": [7], "classe": ["ferrugem"], "severity": [2],
                            "caminho": [str(gravar_foto(tmp_path / "x.png", *folhas))]})
    modelos = modelos_falsos([probs("ferrugem", 0.9)] * 2, [2, 2])
    tabela = ap.t1(modelos, inf.Configuracao(), imagens=imagens, modos=("B",))
    assert "cadeia_A_classe" not in tabela and "recorte_A_classe" not in tabela
    assert tabela.loc[0, "plano"] == "deteccao" and tabela.loc[0, "cadeia_B_decisao"] == (
        "duas_etapas")
    assert set(ap.metricas_t1(tabela)) == {"inteira", "cadeia_B", "recorte_B"}
    r = ap.t1_por_caminho(tabela, inf.Configuracao(), decisoes=("por_caminho",))
    assert set(r) == {"por_caminho"} and r["por_caminho"]["deteccao"]["imagens"] == 1


def test_conferir_contrato_e_desfechos():
    tabela = pd.DataFrame({"id": [1, 2, 3, 4], "plano": ["foto_inteira", "foto_inteira",
                                                         "deteccao", "nenhuma"],
                           "cadeia_B_mantida": [True, False, True, None],
                           "cadeia_B_classe": [1, 2, 0, None]})
    contrato = pd.DataFrame({"id": [1, 2, 3, 4], "plano": ["foto_inteira", "foto_inteira",
                                                           "deteccao", "nenhuma"],
                             "status": ["sucesso", "sucesso", "sucesso", "erro"],
                             "categoria": ["ferrugem", "phoma", "saudavel", None],
                             "tipo": [None, None, None, "planta_nao_identificada"]})
    r = tf.conferir_contrato(tabela, contrato)
    assert (r["foto_inteira"], r["divergentes"], r["ids_divergentes"],
            r["planos_diferentes"]) == (2, 1, [2], 0)
    assert tf.desfechos_t1(tabela) == {"mantida": 2, "descartada_corte": 1, "fragmentos": 0,
                                       "sem_folha": 1, "total": 4}
    resumo_c = tf.resumo_contrato(contrato.assign(folhas=[1, 1, 2, 0]))
    assert resumo_c["desfechos"] == {"sucesso": 3, "erro: planta_nao_identificada": 1}


# ------------------------------------------------------------- uso 2: detecção em pasta nova
def test_avaliar_teste_numa_pasta_nova_com_contagens(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "PASTA_RUNS", tmp_path / "runs")
    (tmp_path / "runs" / "det").mkdir(parents=True)
    (tmp_path / "runs" / "det" / "metricas_val.json").write_text(
        json.dumps({"limiar_escolhido": 0.5}), encoding="utf-8")
    perfeita = resumo([deteccao_md(ANOTADAS[0], 0.9), deteccao_md(ANOTADAS[1], 0.8)])
    errada = resumo([deteccao_md(quadrado(600, 300, 50), 0.95)], anotadas=[ANOTADAS[0]])
    monkeypatch.setattr(td.detector, "carregar_detector", lambda pesos: None)
    monkeypatch.setattr(td.bracot, "ler_manifest", lambda split: pd.DataFrame(
        {"id": ["a", "b"], "caminho": ["a.jpg", "b.jpg"]}))
    monkeypatch.setattr(td.bracot, "TESTE_SOBREPOSTO", (("b", "x"),))
    monkeypatch.setattr(td, "ler_poligonos", lambda split: {})
    monkeypatch.setattr(td, "resumir_fotos", lambda *a, **k: ([perfeita, errada], None))
    chamadas = []
    saida = tmp_path / "saida"
    assert td.avaliar_teste("det", saida=saida, registrar=chamadas.append) == 0
    assert chamadas == ["det"] and not (tmp_path / "runs" / td.USO_DO_TESTE).exists()
    resultado = json.loads((saida / "metricas_teste.json").read_text(encoding="utf-8"))
    assert resultado["sem_sobrepostas"]["fotos"] == 1 and resultado["todas"]["fotos"] == 2
    contagens = pd.read_csv(saida / "contagens_teste.csv")
    assert contagens["sobreposta"].tolist() == [False, True]
    assert contagens["anotadas"].tolist() == [len(ANOTADAS), 1]
    assert not (tmp_path / "runs" / "det" / "metricas_teste.json").exists()
    with pytest.raises(train.ErroFatal):
        td.avaliar_teste("det", saida=saida, registrar=chamadas.append)
    r = tf.resumo_contagens(contagens)
    assert r["fotos"] == 2 and r["fotos_com_mais_previstas"] + r["fotos_com_menos_previstas"] \
        + r["fotos_iguais"] == 2


# --------------------------------------------------------------- uso 3B: análise pré-registrada
def folhas_rotuladas():
    """12 folhas de 6 fotos: 4 S, 4 D e 4 "?", com os desfechos dos dois cenários."""
    gerador = np.random.default_rng(0)
    linhas = []
    for k, rotulo in enumerate(["S", "D", "?"] * 4):
        p = {"S": 0.4, "D": 0.03, "?": 0.1}[rotulo] + gerador.uniform(0, 0.05)
        regra = ccf.decidir_folha([p, 1 - p - 0.03, 0.01, 0.01, 0.01], "deteccao",
                                  ccf.configuracoes()["final"])
        sem = ccf.decidir_folha([p, 1 - p - 0.03, 0.01, 0.01, 0.01], "deteccao",
                                ccf.configuracoes()["sem_td"])
        linhas.append({"conjunto": "teste", "id": f"f{k // 2}_{k % 2 + 1:02d}",
                       "foto": f"f{k // 2}", "rotulo": rotulo, "destino": "classificada",
                       "p_saudavel": p, **{f"{c}_final": v for c, v in regra.items()},
                       **{f"{c}_sem_td": v for c, v in sem.items()}})
    return pd.DataFrame(linhas)


def test_analise_da_parte_b():
    r = tf.analisar_parte_b(folhas_rotuladas())
    assert r["contagens"] == {"S": 4, "D": 4, "?": 4}
    assert set(r["separacao"]) == set(cv.CENARIOS)
    sem_interrogacao = r["separacao"]["sem_interrogacao"]
    assert (sem_interrogacao["S"], sem_interrogacao["D"]) == (4, 4)
    assert sem_interrogacao["auc"]["auc"] == pytest.approx(1.0)
    assert sem_interrogacao["regras"]["final"]["saudavel_dado_s"]["sucessos"] == 4
    assert r["final"]["rotulos"]["D"]["n"] == 4 and set(r["final"]["rotulos"]) == {"S", "D", "?"}
    assert r["separacao"]["interrogacao_como_D"]["D"] == 8
    texto = tf.relatorio_uso_3b({**ap.nativo(r), "uso": "3b", "criado_em": "2026-10-14T12:00:00",
                                 "nao_casados": 0, "planos": {"deteccao": 6},
                                 "referencias": tf.referencias_campo()}, None)
    assert "## D evidentes" in texto and "antigas" in texto


# ------------------------------------------------------------------- guardas de leitura
def test_so_os_modulos_de_teste_leem_os_testes():
    for modulo in (inf, ap, campo, ccf, cv):
        fonte = Path(modulo.__file__).read_text(encoding="utf-8")
        assert "montar_teste" not in fonte and 'split="teste"' not in fonte, modulo.__name__
    fonte = Path(tf.__file__).read_text(encoding="utf-8")
    assert "montar_teste" in fonte and 'ler_poligonos("teste")' not in fonte
    assert 'caminho_coco("teste")' not in fonte
