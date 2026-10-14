"""
Passo 7: a única abertura dos testes fechados, com a cadeia congelada (Frente 9).

Autorização: "pode" do gestor em 14/10/2026 (11:39). Política de descarte fechada: t_d = 0,65
mantido (decisão do gestor, 14/10/2026). Tudo fica num log só, model/runs/uso_do_teste_final.md,
que neste passo substitui os registros model/runs/uso_do_teste.md (classificador) e
model/runs/uso_do_teste_detector.md (detector): nenhum dos dois é criado.

- --abrir: cria o log antes de qualquer leitura do teste. Recusa se o log já existe ou se há
  alteração não commitada em model/ (fora de model/runs/), evaluation/ ou tests/. Grava a data e
  a hora, o SHA-256 dos pesos conferido com model/pesos/MANIFESTO.md, o commit, o git status, a
  configuração efetiva lida do código (conferida com a declarada pelo gestor) e a lista exata do
  que será medido.
- --uso 1: teste do BRACOL (253 imagens) pela cadeia final e, como informação, pelo
  classificador sozinho; model/runs/teste_bracol/.
- --uso 2: detecção no teste do BRACOT (60 fotos; com e sem TESTE_SOBREPOSTO);
  model/runs/teste_bracot_deteccao/.
- --uso 3a: folhas de campo das 58 fotos de teste do BRACOT (sem TESTE_SOBREPOSTO): pranchas
  cegas, CSV vazio e só contagens, com o classificador nem carregado;
  model/runs/teste_bracot_folhas/parte_a/.
- --uso 3b: só com o CSV rotulado e a ordem do gestor: a regra congelada nessas folhas;
  model/runs/teste_bracot_folhas/parte_b/.
- --relatorio --uso N: só refaz o relatorio.md do uso a partir do resultado.json, sem ler o teste.

Cada uso roda uma vez e grava numa pasta nova (recusa sobrescrever). As saídas vão para uma pasta
temporária, renomeada no fim; se der erro, ela é apagada e o log registra o erro. Nova tentativa
só com --nova-tentativa "o que foi corrigido", depois de um erro registrado. Uma conferência
interna que falha sem exceção vai para o log e para o relatório como saiu; outra rodada de um uso
terminado só com --decisao-do-gestor "texto, data".

O código de saída é 0 sem erros e 1 com erros. A saída de terminal fica sem acento (terminal do
Windows); os arquivos gravados são UTF-8.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import avaliar_pipeline as ap  # noqa: E402
import bracol  # noqa: E402
import bracot  # noqa: E402
import campo  # noqa: E402
import campo_cadeia_final as ccf  # noqa: E402
import campo_diagnostico as cd  # noqa: E402
import campo_validacao as cv  # noqa: E402
import dados  # noqa: E402
import detector  # noqa: E402
import formatacao as fmt  # noqa: E402
import inferencia as inf  # noqa: E402
import metrics  # noqa: E402
import train  # noqa: E402
import treinar_detector as td  # noqa: E402

LOG = train.PASTA_RUNS / "uso_do_teste_final.md"
MANIFESTO = RAIZ_REPO / "model" / "pesos" / "MANIFESTO.md"
USOS = ("1", "2", "3a", "3b")
PASTAS = {"1": train.PASTA_RUNS / "teste_bracol",
          "2": train.PASTA_RUNS / "teste_bracot_deteccao",
          "3a": train.PASTA_RUNS / "teste_bracot_folhas" / "parte_a",
          "3b": train.PASTA_RUNS / "teste_bracot_folhas" / "parte_b"}
NOME_USO = {"1": "teste do BRACOL", "2": "detecção no teste do BRACOT",
            "3a": "folhas de campo do teste do BRACOT, parte A",
            "3b": "folhas de campo do teste do BRACOT, parte B"}
# Uso 3: as pranchas ficam fora do git; o CSV, com as colunas dos outros dois, é versionado.
PASTA_ROTULAGEM_TESTE = RAIZ_REPO / "data" / "processed" / "campo_rotulagem_teste"
MANIFEST_CAMPO_TESTE = RAIZ_REPO / "data" / "manifests" / "campo_rotulado_teste.csv"
# A cadeia da rotulagem de teste: a final, fixada aqui para que os ids não mudem.
CONFIG_ROTULAGEM_TESTE = inf.Configuracao(recorte="B", fracao_visivel_minima=0.5)
MAX_POR_FOTO = 2
SEED = campo.SEED
# Quem rotulou as folhas de teste: o registro do commit que trouxe os rótulos.
ORIGEM_ROTULOS_TESTE = ("rotuladas pelo gestor a partir das pranchas cegas, antes de qualquer "
                        "predição (commit `1c59792`)")
# Os pesos finais (os caminhos são os da tabela do MANIFESTO.md) e a cadeia congelada como o
# gestor a declarou no passo 7: conferidos com o código antes de cada leitura do teste.
PESOS = {"classificador": ("model/runs/base_resnet50_bracol/melhor.pt",
                           inf.PASTA_CLASSIFICADOR / "melhor.pt"),
         "detector": ("model/runs/detector_base/weights/last.pt", detector.PESOS_PADRAO)}
DECLARADA = {"decisao": "por_caminho", "recorte": "B", "fracao_visivel_minima": 0.5,
             "limiar_saudavel": 0.083, "limiar_doenca": 0.65, "limiar_confianca": 0.61}
CAMINHOS_DE_CODIGO = ("model/", "evaluation/", "tests/")
# Os números de referência saem destes arquivos, nunca digitados.
REFERENCIAS = {
    "classificador_val": train.PASTA_RUNS / "base_resnet50_bracol" / "metricas_val.json",
    "cadeia_val": train.PASTA_RUNS / "pipeline_final" / "metricas.json",
    "cv": train.PASTA_RUNS / "diagnostico_blocos" / "metricas_blocos.json",
    "detector_val": train.PASTA_RUNS / "detector_base" / "metricas_val.json",
    "detector_val_contagens": train.PASTA_RUNS / "detector_base" / "contagens_val.csv",
    "campo_6b": train.PASTA_RUNS / "campo_cadeia_final" / "campo.json",
    "campo_validacao": train.PASTA_RUNS / "campo_validacao_duas_etapas" / "validacao.json",
}
INICIO, FIM, ERRO, NOVA = "início", "fim", "erro", "nova tentativa"
CONFERENCIA, DECISAO, ARQUIVO = "conferência falhou", "decisão do gestor", "arquivo"
LEITURA, RELATORIO = "teste lido", "relatório refeito"
CAMINHOS_T1 = {"foto_inteira": "foto inteira (regras 1 e 2)", "deteccao": "recortes por detecção",
               "todas": "todas"}


# ------------------------------------------------------------------------- conferências
def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def pesos_do_manifesto(texto: str) -> dict:
    """{caminho relativo: (bytes, sha256)} das linhas da tabela do MANIFESTO.md."""
    return {caminho: (int(tamanho.replace(".", "")), sha)
            for caminho, tamanho, sha in re.findall(
                r"\|\s*`([^`]+\.pt)`\s*\|\s*([\d.]+)\s*\|\s*`([0-9a-f]{64})`", texto)}


def conferir_pesos(manifesto: Path = MANIFESTO, pesos: dict | None = None) -> dict:
    """Tamanho e SHA-256 de cada peso final, conferidos com o MANIFESTO.md (ErroFatal se não
    baterem)."""
    tabela = pesos_do_manifesto(Path(manifesto).read_text(encoding="utf-8"))
    resultado = {}
    for nome, (relativo, arquivo) in (pesos or PESOS).items():
        if relativo not in tabela:
            raise train.ErroFatal(f"{relativo} não está no MANIFESTO.md")
        if not Path(arquivo).is_file():
            raise train.ErroFatal(f"peso ausente: {relativo}")
        tamanho, sha = Path(arquivo).stat().st_size, sha256(arquivo)
        if (tamanho, sha) != tabela[relativo]:
            raise train.ErroFatal(f"{relativo} não bate com o MANIFESTO.md (bytes {tamanho}, "
                                  f"SHA-256 {sha})")
        resultado[nome] = {"arquivo": relativo, "bytes": tamanho, "sha256": sha}
    return resultado


def _relativo(caminho: Path) -> str:
    return Path(caminho).resolve().relative_to(RAIZ_REPO).as_posix()


def configuracao_efetiva() -> dict:
    """A configuração da cadeia lida do código, os pesos que ela carrega e o limiar de detecção
    escolhido na validação do detector."""
    validacao = json.loads(REFERENCIAS["detector_val"].read_text(encoding="utf-8"))
    return {"configuracao": asdict(inf.Configuracao()),
            "pesos": {"classificador": _relativo(inf.PASTA_CLASSIFICADOR / "melhor.pt"),
                      "detector": _relativo(detector.PESOS_PADRAO)},
            "limiar_deteccao_da_validacao": validacao["limiar_escolhido"]}


def conferir_configuracao(efetiva: dict) -> None:
    """ErroFatal se a configuração do código não for a declarada pelo gestor no passo 7."""
    c = efetiva["configuracao"]
    divergencias = [f"{k}: código {c[k]!r}, declarado {v!r}" for k, v in DECLARADA.items()
                    if c[k] != v]
    divergencias += [f"pesos do {nome}: código {efetiva['pesos'][nome]}, declarado {relativo}"
                     for nome, (relativo, _) in PESOS.items()
                     if efetiva["pesos"][nome] != relativo]
    if c["limiar_deteccao"] != efetiva["limiar_deteccao_da_validacao"]:
        divergencias.append(f"limiar_deteccao {c['limiar_deteccao']} diferente do escolhido na "
                            f"validação do detector ({efetiva['limiar_deteccao_da_validacao']})")
    if divergencias:
        raise train.ErroFatal("configuração diferente da declarada: " + "; ".join(divergencias))


def estado_git() -> tuple[str, list[str]]:
    """O commit atual e as linhas do git status --porcelain (com os não rastreados)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ_REPO, capture_output=True, text=True,
                              check=True).stdout

    return git("rev-parse", "HEAD").strip(), git("status", "--porcelain",
                                                 "--untracked-files=all").splitlines()


def alteracoes_de_codigo(status: list[str]) -> list[str]:
    """Os caminhos do git status em model/ (fora de model/runs/), evaluation/ ou tests/."""
    caminhos = [linha[3:].split(" -> ")[-1].strip().strip('"') for linha in status]
    return [c for c in caminhos
            if c.startswith(CAMINHOS_DE_CODIGO) and not c.startswith("model/runs/")]


# --------------------------------------------------------------------------------- log
def o_que_sera_medido() -> list[str]:
    """A lista exata dos usos, gravada no log antes da abertura (pré-registro)."""
    sobrepostas = ", ".join(t for t, _ in bracot.TESTE_SOBREPOSTO)
    return [
        f"1. **Uso 1, {NOME_USO['1']}** (as 253 imagens de `dados.montar_teste()`), em "
        "`model/runs/teste_bracol/`:",
        "   - **Cadeia final inteira:** `avaliar_pipeline.t1` só no modo B, como no T1 do "
        "`pipeline_final`.",
        "     - Por caminho: a foto inteira (argmax com corte 0,61), a detecção (duas etapas) e "
        "a regra 3.",
        "     - Métricas: acurácia com Wilson, F1 macro, recall por classe, matriz de confusão, "
        "severidade (exata e ±1), cobertura e acurácia das mantidas e doentes chamadas "
        "saudáveis.",
        "   - **O que o usuário recebe:** `inferencia.diagnosticar` em cada imagem (sucesso, "
        "`baixa_confianca`, `planta_nao_identificada` e exceções), conferido com o T1 nas "
        "imagens da foto inteira.",
        "   - **Classificador sozinho, sem corte (só informação):** pelo caminho do "
        "`metricas_val.json` (`train.avaliar`, com AMP), com as mesmas métricas, MAE e kappa da "
        "severidade.",
        "   - **Comparação, sem ajuste:** com a validação e com os protocolos por blocos e "
        "aleatório do `diagnostico_blocos`, lidos dos arquivos.",
        f"2. **Uso 2, {NOME_USO['2']}** (as 60 fotos), em `model/runs/teste_bracot_deteccao/`:",
        "   - `treinar_detector.avaliar_teste(\"detector_base\")`, com o limiar da validação "
        "(0,35).",
        "   - Métricas: AP50 e AP50-95 de máscara e de caixa, nos modos região e conservador, e "
        "precisão, recall e F1 no limiar.",
        f"   - Com as 60 fotos e sem as 2 de `TESTE_SOBREPOSTO` ({sobrepostas}), mais a contagem "
        "de folhas por foto, anotadas contra previstas.",
        "   - Sem o validador do ultralytics, que exigiria copiar as fotos de teste.",
        f"3. **Uso 3A, {NOME_USO['3a']}** (as 58 fotos, sem as de `TESTE_SOBREPOSTO`), em "
        "`model/runs/teste_bracot_folhas/parte_a/`:",
        "   - Detector, área, cor, limite e filtro de fragmentos (0,5); as anotações de teste não "
        "são lidas.",
        f"   - Sorteio de até {MAX_POR_FOTO} folhas por foto, com a semente {SEED} "
        "(`campo.sortear_rotulagem`), sem o classificador carregado.",
        "   - Pranchas cegas em `data/processed/campo_rotulagem_teste/` (fora do git) e o CSV "
        "vazio `data/manifests/campo_rotulado_teste.csv`. Só contagens: nenhuma predição.",
        f"4. **Uso 3B, {NOME_USO['3b']}** (só com o CSV rotulado e o \"pode avaliar\" do "
        "gestor), em `model/runs/teste_bracot_folhas/parte_b/`. A estrutura é a do passo 6b, e a "
        "cadeia congelada é o resultado principal:",
        "   - destino de S, D e \"?\": saudável, doente mantida (com as classes), descartada por "
        "fragmento, por t_d ou por outro motivo, com Wilson;",
        "   - D evidentes que chegam ao usuário como doente, como saudável ou somem;",
        "   - classes e confiança das doentes mantidas;",
        "   - saudável dado S e dado D, com Wilson;",
        f"   - AUC de P(saudável) com bootstrap por foto ({fmt.n(cd.BOOTSTRAP)} reamostras, "
        f"semente {cd.SEED});",
        "   - os \"?\" como D e como S.",
        "   - Os números com o corte t_d desligado entram só como informação, declarada aqui.",
        "   - Comparação, sem ajuste, com as antigas e as novas do passo 6b.",
    ]


def montar_abertura(agora: datetime, commit: str, status: list[str], pesos: dict,
                    efetiva: dict) -> str:
    c = efetiva["configuracao"]
    linhas = [
        "# Uso único dos testes fechados (passo 7)", "",
        "**Autorização:** \"pode\" do gestor em 14/10/2026 (11:39). **Política de descarte "
        "fechada:** t_d = 0,65 mantido (decisão do gestor, 14/10/2026). **Esta é a única "
        "abertura dos testes.** Nada é retreinado, nenhum limiar ou padrão muda, e os testes não "
        "servem a nenhuma escolha. Este log substitui, neste passo, os registros "
        "`uso_do_teste.md` e `uso_do_teste_detector.md`, que não são criados.", "",
        "## Abertura, antes de ler qualquer teste", "",
        f"- **Data e hora:** {agora:%d/%m/%Y %H:%M:%S}.",
        f"- **Commit:** `{commit}`. Nenhuma alteração não commitada em `model/` (fora de "
        "`model/runs/`), `evaluation/` ou `tests/`.",
        "- **git status --porcelain:** " + ("vazio." if not status else ""), ""]
    if status:
        linhas += ["```", *status, "```", ""]
    linhas += ["**Pesos finais**, conferidos com `model/pesos/MANIFESTO.md`:", ""]
    linhas += fmt.tabela(["peso", "arquivo", "bytes", "SHA-256"],
                         [[nome, f"`{p['arquivo']}`", fmt.n(p["bytes"]), f"`{p['sha256']}`"]
                          for nome, p in pesos.items()], "llrl")
    linhas += ["", "**Configuração efetiva**, lida do código (`asdict(inferencia.Configuracao())`) "
               "e conferida com a declarada pelo gestor:", ""]
    linhas += fmt.tabela(["parâmetro", "valor"], [[k, f"`{v!r}`"] for k, v in c.items()], "ll")
    linhas += ["", f"- Pesos que a cadeia carrega: `{efetiva['pesos']['classificador']}` e "
               f"`{efetiva['pesos']['detector']}`.",
               "- Limiar de detecção escolhido na validação do detector: "
               f"{efetiva['limiar_deteccao_da_validacao']}.",
               f"- Semente do sorteio das folhas do uso 3: {SEED}.", "",
               "## O que será medido (só isto)", "", *o_que_sera_medido(), "",
               "## Regras de execução", "",
               "- **Saídas:** cada uso roda uma vez e grava numa pasta nova, sem sobrescrever.",
               "- **Erro de código:** a pasta temporária é apagada e o erro vai para este log. "
               "Uma nova tentativa só roda com o motivo registrado.",
               "- **Conferência interna que falha sem exceção:** fica registrada e relatada como "
               "saiu. Outra rodada só com decisão do gestor, registrada aqui.", "",
               "## Registro", "",
               "| data e hora | uso | evento | detalhes |", "|---|---|---|---|"]
    return "\n".join(linhas) + "\n"


def abrir(log: Path | None = None) -> int:
    log = Path(log or LOG)
    if log.exists():
        raise train.ErroFatal(f"{train._exibir(log)} já existe: os testes já foram abertos")
    commit, status = estado_git()
    codigo = alteracoes_de_codigo(status)
    if codigo:
        raise train.ErroFatal("há alteração não commitada no código: " + ", ".join(codigo)
                              + ". Commite antes de abrir os testes")
    pesos = conferir_pesos()
    efetiva = configuracao_efetiva()
    conferir_configuracao(efetiva)
    log.write_text(montar_abertura(datetime.now(), commit, status, pesos, efetiva),
                   encoding="utf-8", newline="\n")
    fmt.dizer(f"Log criado: {log} (commit {commit[:7]}); nenhum teste foi lido")
    return 0


def registrar(uso: str, evento: str, detalhes: str, log: Path | None = None) -> None:
    texto = " ".join(str(detalhes).replace("|", "/").split())
    with open(log or LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"| {datetime.now():%d/%m/%Y %H:%M:%S} | {uso} | {evento} | {texto} |\n")


def eventos(uso: str, log: Path | None = None) -> list[str]:
    """Os eventos de um uso no registro do log, em ordem."""
    texto = Path(log or LOG).read_text(encoding="utf-8")
    registro = texto.split("## Registro", 1)[1] if "## Registro" in texto else ""
    saida = []
    for linha in registro.splitlines():
        celulas = [c.strip() for c in linha.strip().strip("|").split("|")]
        if len(celulas) == 4 and celulas[1] == uso:
            saida.append(celulas[2])
    return saida


def autorizar(uso: str, eventos_do_uso: list[str], nova_tentativa: str | None,
              decisao: str | None) -> Path:
    """A pasta de saída desta rodada, ou ErroFatal se a rodada não é permitida: a primeira é
    livre; depois de um erro (ou de um início sem fim), só com nova_tentativa; depois de um
    fim, só com a decisão do gestor, numa pasta nova."""
    principais = [e for e in eventos_do_uso if e in (INICIO, FIM, ERRO)]
    ultimo = principais[-1] if principais else None
    if ultimo is None and (nova_tentativa or decisao):
        raise train.ErroFatal("é a primeira rodada deste uso: sem --nova-tentativa nem "
                              "--decisao-do-gestor")
    if ultimo in (INICIO, ERRO) and not nova_tentativa:
        raise train.ErroFatal("a última rodada deste uso não terminou: rode de novo só com "
                              "--nova-tentativa \"o que foi corrigido\"")
    if ultimo == FIM and not decisao:
        raise train.ErroFatal("este uso já rodou: outra rodada só com a autorização do gestor "
                              "(--decisao-do-gestor \"texto, data\")")
    rodadas = principais.count(FIM)
    pasta = PASTAS[uso]
    return pasta if not rodadas else pasta.with_name(f"{pasta.name}_rodada{rodadas + 1}")


def _temporaria(final: Path) -> Path:
    return final.with_name(final.name + ".tmp")


def destinos_extras(uso: str) -> list[Path]:
    """Saídas do uso fora da pasta dele (o uso 3A grava as pranchas e o CSV de rotulagem)."""
    return [PASTA_ROTULAGEM_TESTE, MANIFEST_CAMPO_TESTE] if uso == "3a" else []


def preparar(uso: str) -> None:
    """O que precisa existir (ou não) antes de abrir o teste do uso, sem ler o teste."""
    for nome, arquivo in REFERENCIAS.items():
        if not arquivo.is_file():
            raise train.ErroFatal(f"falta a referência {nome}: {train._exibir(arquivo)}")
    for final in destinos_extras(uso):
        if final.exists():
            raise train.ErroFatal(f"{train._exibir(final)} já existe: não sobrescrevo")
    if uso == "3b":
        if not PASTAS["3a"].is_dir():
            raise train.ErroFatal("a parte A do uso 3 não rodou")
        bruto = pd.read_csv(MANIFEST_CAMPO_TESTE, dtype=str, keep_default_na=False)
        rotulos = campo.ler_rotulos(MANIFEST_CAMPO_TESTE)
        if rotulos is None or len(rotulos) != len(bruto):
            raise train.ErroFatal("o CSV de rotulagem de teste não está todo rotulado")


def executar_uso(uso: str, nova_tentativa: str | None = None, decisao: str | None = None,
                 funcoes: dict | None = None) -> int:
    """Roda um uso: confere, registra o início, calcula numa pasta temporária, renomeia e
    registra o fim com o SHA-256 de cada arquivo (ou o erro, apagando a temporária)."""
    if not LOG.is_file():
        raise train.ErroFatal(f"{train._exibir(LOG)} não existe: rode --abrir antes")
    funcoes = funcoes or FUNCOES
    pasta = autorizar(uso, eventos(uso), nova_tentativa, decisao)
    if pasta.exists():
        raise train.ErroFatal(f"{train._exibir(pasta)} já existe: não sobrescrevo")
    preparar(uso)
    conferir_pesos()
    conferir_configuracao(configuracao_efetiva())
    commit, status = estado_git()
    codigo = alteracoes_de_codigo(status)
    if decisao:
        registrar(uso, DECISAO, decisao)
    if nova_tentativa:
        registrar(uso, NOVA, nova_tentativa)
    registrar(uso, INICIO, f"{NOME_USO[uso]}; commit {commit[:7]}; código não commitado: "
                           f"{', '.join(codigo) or 'nenhum'}; pasta {_relativo_ou_nome(pasta)}")
    temporarias = [_temporaria(pasta), *(_temporaria(f) for f in destinos_extras(uso))]
    for t in temporarias:
        _apagar(t)
    try:
        saida = funcoes[uso](_temporaria(pasta))
        _temporaria(pasta).rename(pasta)
        for final in destinos_extras(uso):
            final.parent.mkdir(parents=True, exist_ok=True)
            _temporaria(final).rename(final)
    except BaseException as e:
        for t in temporarias:
            _apagar(t)
        registrar(uso, ERRO, f"{type(e).__name__}: {e}")
        raise
    for falha in saida.get("conferencias_falhas", []):
        registrar(uso, CONFERENCIA, falha)
    registrar(uso, FIM, saida.get("resumo", ""))
    for arquivo in _arquivos(pasta, destinos_extras(uso)):
        registrar(uso, ARQUIVO, f"{_relativo_ou_nome(arquivo)}, SHA-256 {sha256(arquivo)}")
    fmt.dizer(f"Uso {uso} concluido: {saida.get('resumo', '')}")
    return 0


def _apagar(caminho: Path) -> None:
    if caminho.is_dir():
        shutil.rmtree(caminho)
    elif caminho.exists():
        caminho.unlink()


def _arquivos(pasta: Path, extras) -> list[Path]:
    todos = sorted(p for p in pasta.rglob("*") if p.is_file())
    for extra in extras:
        todos += sorted(p for p in extra.rglob("*") if p.is_file()) if extra.is_dir() else [extra]
    return todos


# --------------------------------------------------------------------- uso 1: BRACOL
def uso_1(pasta: Path) -> dict:
    """Teste do BRACOL: o classificador sozinho (o caminho do metricas_val.json, como
    informação), a cadeia final (T1 no modo B, por caminho) e o contrato de cada imagem."""
    pasta.mkdir(parents=True)
    teste = dados.montar_teste()
    registrar("1", LEITURA, f"{len(teste)} imagens do teste do BRACOL (dados.montar_teste)")
    dispositivo = train.escolher_dispositivo("auto")
    amp = dispositivo.type == "cuda"
    modelo, checkpoint = train.carregar_modelo(inf.PASTA_CLASSIFICADOR, dispositivo)
    tamanho = checkpoint["tamanho_entrada"]
    loader = train.criar_loader(teste, dados.transformacao_avaliacao(tamanho["largura"],
                                                                     tamanho["altura"]),
                                checkpoint["config"]["batch"], train.WORKERS_PADRAO, pin=amp)
    fmt.dizer(f"Classificador sozinho: {len(teste)} imagens ({dispositivo})...")
    aval = train.avaliar(modelo, loader, dispositivo, checkpoint["config"]["peso_sev"], amp)
    m = metrics.calcular_metricas(aval["classe"], aval["classe_prevista"])
    ms = metrics.calcular_metricas_severidade(aval["severidade"], aval["severidade_prevista"])
    train.gravar_resultados(pasta, "teste", teste, aval, m, ms, checkpoint["epoca"])
    del modelo
    config = inf.Configuracao()
    modelos = inf.carregar_modelos(dispositivo=str(dispositivo))
    fmt.dizer("Cadeia final (T1, modo B)...")
    tabela = ap.t1(modelos, config, imagens=teste, modos=("B",))
    t1 = ap.resumir_t1(tabela, pasta / "predicoes_teste.csv")
    por_caminho = ap.t1_por_caminho(tabela, config, decisoes=("por_caminho",))["por_caminho"]
    mantidas = tabela[tabela["cadeia_B_mantida"].fillna(False).astype(bool)]
    m_mantidas = metrics.calcular_metricas(mantidas["classe"].to_numpy(),
                                           mantidas["cadeia_B_classe"].astype(int).to_numpy())
    fmt.dizer("O que o usuario recebe (inferencia.diagnosticar)...")
    contrato = pd.DataFrame([contrato_da_imagem(linha, config, modelos)
                             for linha in teste.itertuples()])
    conferencia = conferir_contrato(tabela, contrato)
    falhas = ([f"contrato x T1: {conferencia['divergentes']} divergências nas "
               f"{conferencia['foto_inteira']} imagens da foto inteira; planos diferentes em "
               f"{conferencia['planos_diferentes']} imagens"]
              if conferencia["divergentes"] or conferencia["planos_diferentes"] else [])
    resultado = {
        "imagens": len(teste), "epoca": checkpoint["epoca"], "dispositivo": str(dispositivo),
        "classificador": {"classificacao": m, "severidade": ms},
        "cadeia": {"t1": t1, "por_caminho": por_caminho, "mantidas": m_mantidas,
                   "desfechos": desfechos_t1(tabela)},
        "contrato": resumo_contrato(contrato), "conferencia_contrato": conferencia,
        "conferencia_inteira": t1["conferencia"], "referencias": referencias_uso_1()}
    metrics.salvar_figura(metrics.desenhar_matriz_confusao(
        t1["variantes"]["cadeia_B"]["metricas"]["matriz_confusao"],
        f"Matriz de confusão: cadeia final no teste, sem corte "
        f"({fmt.n(t1['variantes']['cadeia_B']['metricas']['n'])} imagens)"),
        pasta / "matriz_confusao_cadeia_teste.png")
    ap.gravar_csv(tabela.drop(columns=[c for c in tabela.columns if c.endswith("_probs")])
                  .assign(classe=lambda t: t["classe"].map(lambda k: bracol.CLASSES[k])),
                  pasta / "t1_imagens.csv")
    ap.gravar_csv(contrato, pasta / "contrato.csv")
    _gravar_resultado(pasta, "1", resultado)
    cadeia = t1["variantes"]["cadeia_B"]
    return {"resumo": f"cadeia final: acurácia sem corte {_pct(cadeia['metricas']['acuracia'])}, "
                      f"mantidas {cadeia['mantidas']} de {len(teste)}, acurácia das mantidas "
                      f"{_pct(cadeia['acuracia_no_limiar'])}; classificador sozinho: acurácia "
                      f"{_pct(m['acuracia'])}, F1 macro {_pct(m['f1_macro'])}",
            "conferencias_falhas": falhas}


def contrato_da_imagem(linha, config, modelos) -> dict:
    """A resposta do contrato para uma imagem (inferencia.diagnosticar), resumida."""
    imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
    try:
        r = inf.diagnosticar(imagem, config, modelos)
    except Exception as e:  # uma exceção da cadeia é um resultado: conta e segue
        return {"id": linha.id, "status": "excecao", "tipo": f"{type(e).__name__}: {e}",
                "plano": None, "decisao": None, "folhas": 0, "categoria": None,
                "confianca": None}
    detalhes, folhas = r.get("detalhes") or {}, r.get("folhas") or []
    tipo = r.get("tipo")
    if tipo == "baixa_confianca" and r.get("mensagem") == inf.MENSAGENS["fragmentos"]:
        tipo = "baixa_confianca_fragmentos"
    return {"id": linha.id, "status": r["status"], "tipo": tipo,
            "plano": detalhes.get("plano"), "decisao": detalhes.get("decisao"),
            "folhas": len(folhas), "categoria": folhas[0]["categoria"] if folhas else None,
            "confianca": folhas[0]["confianca"] if folhas else None}


def conferir_contrato(tabela: pd.DataFrame, contrato: pd.DataFrame) -> dict:
    """O contrato contra o T1: o plano de cada imagem e, nas da foto inteira (uma folha só), se
    a folha chega ao usuário e com a mesma classe."""
    t = tabela.set_index("id")
    c = contrato.set_index("id").loc[t.index]
    planos_diferentes = int((t["plano"] != c["plano"]).sum())
    inteira = t["plano"] == "foto_inteira"
    divergentes = []
    for i in t.index[inteira]:
        mantida = bool(t.at[i, "cadeia_B_mantida"])
        chega = c.at[i, "status"] == "sucesso"
        if mantida != chega or (chega and c.at[i, "categoria"]
                                != inf.CATEGORIAS[int(t.at[i, "cadeia_B_classe"])]):
            divergentes.append(int(i))
    return {"foto_inteira": int(inteira.sum()), "divergentes": len(divergentes),
            "ids_divergentes": divergentes, "planos_diferentes": planos_diferentes}


def desfechos_t1(tabela: pd.DataFrame) -> dict:
    """O destino da folha principal de cada imagem na cadeia B (T1)."""
    plano = tabela["plano"]
    mantida = tabela["cadeia_B_mantida"].fillna(False).astype(bool)
    com = tabela["cadeia_B_classe"].notna()
    return {"mantida": int((com & mantida).sum()), "descartada_corte": int((com & ~mantida).sum()),
            "fragmentos": int((plano == "fragmentos").sum()),
            "sem_folha": int((plano == "nenhuma").sum()), "total": len(tabela)}


def resumo_contrato(contrato: pd.DataFrame) -> dict:
    status = contrato["status"].where(contrato["status"] != "erro",
                                      "erro: " + contrato["tipo"].fillna(""))
    return {"imagens": len(contrato), "desfechos": status.value_counts().to_dict(),
            "excecoes": contrato.loc[contrato["status"] == "excecao", "tipo"].tolist(),
            "folhas_por_resposta": contrato.loc[contrato["status"] == "sucesso", "folhas"]
            .value_counts().sort_index().to_dict()}


def referencias_uso_1() -> dict:
    """Os números de referência do uso 1, lidos dos arquivos (com o caminho de cada um)."""
    val = json.loads(REFERENCIAS["classificador_val"].read_text(encoding="utf-8"))
    cadeia = json.loads(REFERENCIAS["cadeia_val"].read_text(encoding="utf-8"))
    cv_ = json.loads(REFERENCIAS["cv"].read_text(encoding="utf-8"))
    return {
        "classificador_val": {"arquivo": _relativo(REFERENCIAS["classificador_val"]),
                              "classificacao": val["classificacao"],
                              "severidade": val.get("severidade")},
        "cadeia_val": {"arquivo": _relativo(REFERENCIAS["cadeia_val"]),
                       "cadeia_B": cadeia["t1"]["variantes"]["cadeia_B"],
                       "por_caminho": cadeia["por_caminho"]["t1"]["por_caminho"],
                       "regras": cadeia["t1"]["regras"]},
        "cv": {"arquivo": _relativo(REFERENCIAS["cv"]),
               **{p: cv_["protocolos"][p]["classificacao"] for p in ("blocos", "aleatorio")}}}


# ------------------------------------------------------------- uso 2: detecção BRACOT
def uso_2(pasta: Path) -> dict:
    """Detecção no teste do BRACOT: avaliar_teste do detector_base numa pasta nova, com as
    contagens por foto, e a comparação com a validação."""
    pasta.mkdir(parents=True)
    td.avaliar_teste(td.NOME_PADRAO, saida=pasta, registrar=lambda run: registrar(
        "2", LEITURA, f"fotos e anotações de teste do BRACOT lidas e avaliadas ({run})"))
    teste = json.loads((pasta / "metricas_teste.json").read_text(encoding="utf-8"))
    contagens = pd.read_csv(pasta / "contagens_teste.csv")
    val = json.loads(REFERENCIAS["detector_val"].read_text(encoding="utf-8"))
    limiar = f"{val['limiar_escolhido']:g}"
    resultado = {
        "teste": teste,
        "contagens": {"todas": resumo_contagens(contagens),
                      "sem_sobrepostas": resumo_contagens(contagens[~contagens["sobreposta"]])},
        "referencias": {
            "validacao": {"arquivo": _relativo(REFERENCIAS["detector_val"]),
                          "fotos": val["n_fotos"], "anotadas": val["n_anotadas"],
                          "metricas": val["metricas"],
                          "no_limiar": {m: val["limiares"][limiar][m] for m in ("regiao",
                                                                               "conservador")}},
            "validacao_contagens": {"arquivo": _relativo(REFERENCIAS["detector_val_contagens"]),
                                    **resumo_contagens(pd.read_csv(
                                        REFERENCIAS["detector_val_contagens"]))}}}
    _gravar_resultado(pasta, "2", resultado)
    todas = teste["todas"]["metricas"]["regiao"]["mascara"]
    return {"resumo": f"AP50 de máscara (região), 60 fotos: {_pct(todas['ap50'])}; sem as "
                      "sobrepostas: "
                      f"{_pct(teste['sem_sobrepostas']['metricas']['regiao']['mascara']['ap50'])}",
            "conferencias_falhas": []}


def resumo_contagens(c: pd.DataFrame) -> dict:
    """Folhas por foto, anotadas contra previstas (no limiar), de md.contagens."""
    diferenca = c["previstas"] - c["anotadas"]
    return {"fotos": len(c), "anotadas": int(c["anotadas"].sum()),
            "previstas": int(c["previstas"].sum()), "acertos": int(c["acertos"].sum()),
            "fora_da_regiao": int(c["fora_da_regiao"].sum()),
            "anotadas_por_foto": _faixa(c["anotadas"]),
            "previstas_por_foto": _faixa(c["previstas"]),
            "diferenca_por_foto": {**_faixa(diferenca), "media": float(diferenca.mean())},
            "fotos_com_mais_previstas": int((diferenca > 0).sum()),
            "fotos_com_menos_previstas": int((diferenca < 0).sum()),
            "fotos_iguais": int((diferenca == 0).sum())}


def _faixa(serie) -> dict:
    s = pd.Series(serie, dtype=float)
    return {"mediana": float(s.median()), "min": float(s.min()), "max": float(s.max())}


# ----------------------------------------------------- uso 3: folhas de campo, parte A
def fotos_de_teste_sem_sobrepostas() -> tuple[list[str], pd.Series]:
    """As fotos de teste do BRACOT sem as de TESTE_SOBREPOSTO (só ids e caminhos das imagens;
    as anotações de teste não são lidas aqui)."""
    manifest = bracot.ler_manifest(split="teste")
    sobrepostas = {t for t, _ in bracot.TESTE_SOBREPOSTO}
    return sorted(set(manifest["id"]) - sobrepostas), manifest.set_index("id")["caminho"]


def modelos_so_detector():
    """Só o detector: sem o classificador, nenhuma predição de classe é possível."""
    import torch

    return inf.Modelos(classificador=None, transformacao=None,
                       detector=detector.carregar_detector(detector.PESOS_PADRAO),
                       dispositivo=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       nome_classificador=None, nome_detector="detector_base")


def uso_3a(pasta: Path) -> dict:
    """Pranchas cegas e CSV vazio das folhas das fotos de teste, e só contagens."""
    pasta.mkdir(parents=True)
    ids, caminhos = fotos_de_teste_sem_sobrepostas()
    registrar("3a", LEITURA, f"{len(ids)} fotos de teste do BRACOT (só as imagens)")
    contagens = {}
    csv_temporario = _temporaria(MANIFEST_CAMPO_TESTE)
    r = campo.gerar_rotulagem(modelos_so_detector(), _temporaria(PASTA_ROTULAGEM_TESTE),
                              csv_temporario, CONFIG_ROTULAGEM_TESTE, (ids, caminhos),
                              n=MAX_POR_FOTO * len(ids), max_por_foto=MAX_POR_FOTO,
                              contagens=contagens)
    sorteadas = pd.read_csv(csv_temporario, dtype=str).groupby("foto").size()
    tabela = pd.DataFrame([{"foto": f, **c, "sorteadas": int(sorteadas.get(f, 0))}
                           for f, c in contagens.items()])
    ap.gravar_csv(tabela, pasta / "contagens.csv")
    resultado = resumo_parte_a(tabela, len(r["pranchas"]))
    _gravar_resultado(pasta, "3a", resultado)
    return {"resumo": f"{resultado['sorteadas']} folhas sorteadas de {resultado['fotos']} fotos "
                      f"em {resultado['pranchas']} pranchas; o filtro de fragmentos tirou "
                      f"{resultado['cortadas']['fragmento']} folhas",
            "conferencias_falhas": []}


def resumo_parte_a(tabela: pd.DataFrame, pranchas: int) -> dict:
    sorteadas = tabela["sorteadas"]
    return {"fotos": len(tabela), "sobrepostas_fora": [t for t, _ in bracot.TESTE_SOBREPOSTO],
            "planos": tabela["plano"].value_counts().to_dict(),
            "deteccoes_no_limiar": int(tabela["deteccoes"].sum()),
            "cortadas": {"area": int(tabela["cortadas_por_area"].sum()),
                         "cor": int(tabela["cortadas_por_cor"].sum()),
                         "fragmento": int(tabela["cortadas_por_fragmento"].sum()),
                         "limite": int(tabela["cortadas_por_limite"].sum())},
            "fotos_com_fragmento_cortado": int((tabela["cortadas_por_fragmento"] > 0).sum()),
            "folhas": int(tabela["folhas"].sum()), "folhas_por_foto": _faixa(tabela["folhas"]),
            "sorteadas": int(sorteadas.sum()),
            "fotos_por_sorteadas": {int(k): int(v) for k, v in
                                    sorteadas.value_counts().sort_index().items()},
            "pranchas": pranchas, "max_por_foto": MAX_POR_FOTO, "semente": SEED,
            "pasta_pranchas": _relativo_ou_nome(PASTA_ROTULAGEM_TESTE),
            "csv": _relativo_ou_nome(MANIFEST_CAMPO_TESTE)}


def _relativo_ou_nome(caminho: Path) -> str:
    try:
        return _relativo(caminho)
    except ValueError:
        return str(caminho)


# ----------------------------------------------------- uso 3: folhas de campo, parte B
def uso_3b(pasta: Path) -> dict:
    """A regra congelada nas folhas rotuladas das fotos de teste (estrutura do passo 6b)."""
    pasta.mkdir(parents=True)
    ids, caminhos = fotos_de_teste_sem_sobrepostas()
    registrar("3b", LEITURA, f"{len(ids)} fotos de teste do BRACOT e os rótulos de "
                             f"{_relativo_ou_nome(MANIFEST_CAMPO_TESTE)}")
    configs = ccf.configuracoes()
    modelos = inf.carregar_modelos(dispositivo=cd._dispositivo())
    folhas, nao_casados, planos = ccf.folhas_do_conjunto(
        "teste", modelos, configs, (MANIFEST_CAMPO_TESTE, CONFIG_ROTULAGEM_TESTE, (ids, caminhos)))
    resultado = {**analisar_parte_b(folhas), "nao_casados": nao_casados,
                 "planos": pd.Series(planos).value_counts().to_dict(),
                 "referencias": referencias_campo()}
    ap.gravar_csv(folhas, pasta / "folhas.csv")
    _gravar_resultado(pasta, "3b", resultado)
    d = resultado["final"]["rotulos"]["D"]
    return {"resumo": f"D evidentes: {d['n']}; chegam como doente {_pct(d['doenca']['p'])}, como "
                      f"saudável {_pct(d['saudavel']['p'])}, somem {_pct(d['descartada']['p'])}",
            "conferencias_falhas": [f"{nao_casados} rótulos sem par na cadeia"] if nao_casados
            else []}


def analisar_parte_b(folhas: pd.DataFrame) -> dict:
    """A estrutura do passo 6b nas folhas de teste (destinos e doentes mantidas, com e sem t_d) e
    a separação S x D nas classificadas: saudável dado S e D com Wilson e a AUC de P(saudável)
    com bootstrap por foto, sem os "?" e com eles como S ou como D."""
    resultado = {"contagens": {r: int((folhas["rotulo"] == r).sum()) for r in ccf.ROTULOS},
                 "destinos": folhas["destino"].value_counts().to_dict(),
                 **{cenario: {"rotulos": ccf.resumo_por_rotulo(folhas, cenario),
                              "doentes_mantidas": ccf.doentes_mantidas(folhas, cenario)}
                    for cenario in ccf.CENARIOS}}
    classificadas = folhas[folhas["destino"] == "classificada"]
    resultado["separacao"] = {
        cenario: cv.medir(cv.com_rotulos(classificadas, cenario), list(ccf.CENARIOS))
        for cenario in cv.CENARIOS}
    return resultado


def referencias_campo() -> dict:
    seis_b = json.loads(REFERENCIAS["campo_6b"].read_text(encoding="utf-8"))
    validacao = json.loads(REFERENCIAS["campo_validacao"].read_text(encoding="utf-8"))
    return {"campo_6b": {"arquivo": _relativo(REFERENCIAS["campo_6b"]),
                         **{c: seis_b["por_conjunto"][c]["final"]["rotulos"]
                            for c in ("antigas", "novas", "juntas")}},
            "auc_validacao": {"arquivo": _relativo(REFERENCIAS["campo_validacao"]),
                              **{c: validacao["por_conjunto"][c]["auc"]
                                 for c in ("antigas", "novas", "juntas")}}}


FUNCOES = {"1": uso_1, "2": uso_2, "3a": uso_3a, "3b": uso_3b}


# ------------------------------------------------------------------------- relatórios
def _pct(valor, casas: int = 1) -> str:
    return ap._pct(valor, casas)


def _prop(p) -> str:
    return ap._prop(p)


def _ic(m: dict) -> str:
    return (f"{_pct(m['acuracia'])} ({_pct(m['acuracia_ic95'][0])} a "
            f"{_pct(m['acuracia_ic95'][1])})")


def _gravar_resultado(pasta: Path, uso: str, resultado: dict) -> None:
    resultado = ap.nativo({"uso": uso, "criado_em": datetime.now().isoformat(timespec="seconds"),
                           **resultado})
    train.gravar_json(resultado, pasta / "resultado.json")
    (pasta / "relatorio.md").write_text(RELATORIOS[uso](resultado, _leitura(pasta)),
                                        encoding="utf-8", newline="\n")


def _leitura(pasta: Path) -> str | None:
    nota = pasta / "leitura.md"
    return nota.read_text(encoding="utf-8").strip() if nota.is_file() else None


def _cabecalho(uso: str, r: dict) -> list[str]:
    return [f"# Uso {uso}: {NOME_USO[uso]} (passo 7)", "",
            f"Gerado por `python model/teste_final.py --uso {uso}` em "
            f"{datetime.fromisoformat(r['criado_em']):%d/%m/%Y %H:%M}. **Uso único e declarado "
            "dos testes**, com a cadeia congelada; nada foi ajustado nem escolhido por estes "
            "números. Início, fim, arquivos e SHA-256 em `model/runs/uso_do_teste_final.md`.", ""]


def relatorio_uso_1(r: dict, leitura: str | None) -> str:
    t1, ref = r["cadeia"]["t1"], r["referencias"]
    cadeia, cadeia_val = t1["variantes"]["cadeia_B"], ref["cadeia_val"]["cadeia_B"]
    m, ms = r["classificador"]["classificacao"], r["classificador"]["severidade"]
    linhas = _cabecalho("1", r) + [
        "## Cadeia final (o resultado principal)", "",
        f"As {r['imagens']} imagens de teste do BRACOL pela cadeia final inteira, no modo B, "
        "com o mesmo código do T1 da validação (`avaliar_pipeline.t1`). A cadeia decide o "
        "caminho de cada imagem: foto inteira (argmax com corte 0,61) ou detecção (duas etapas). "
        "A folha avaliada é a de maior área.", "",
        "**Por caminho, teste contra validação** (validação: "
        f"`{ref['cadeia_val']['arquivo']}`):", ""]
    linhas += fmt.tabela(
        ["caminho", "conjunto", "imagens", "acurácia sem corte", "cobertura",
         "acurácia das mantidas (IC 95%)", "doentes chamadas saudáveis (IC 95%)"],
        [[CAMINHOS_T1[c], nome, str(x["imagens"]), _pct(x["acuracia_sem_corte"]),
          _pct(x["cobertura"]["p"]), _prop(x["acuracia_das_mantidas"]),
          _prop(x["saudavel_dado_d"])]
         for c in CAMINHOS_T1 for nome, x in (("teste", r["cadeia"]["por_caminho"][c]),
                                              ("validação", ref["cadeia_val"]["por_caminho"][c]))],
        "llrrrll")
    d = r["cadeia"]["desfechos"]
    regras = t1["regras"]
    linhas += ["", f"**Caminhos no teste:** regra 1 em {regras['regra_1']} imagens, regra 2 em "
               f"{regras['regra_2']}, detecção em {regras['deteccao']} e regra 3 (sem folha) em "
               f"{regras['regra_3']}. **A folha principal:** mantida em {d['mantida']}, "
               f"descartada pelo corte em {d['descartada_corte']}, tirada pelo filtro de "
               f"fragmentos em {d['fragmentos']} e sem folha em {d['sem_folha']}, de "
               f"{d['total']}.", "",
               "**Cadeia B, todas as imagens: teste contra validação.**", ""]
    linhas += fmt.tabela(
        ["conjunto", "com previsão", "acurácia sem corte (IC 95%)", "F1 macro",
         *(f"recall {c}" for c in bracol.CLASSES), "severidade: exata / ±1",
         "no corte: cobertura / acurácia"],
        [[nome, _pct(x["cobertura_sem_limiar"]), _ic(x["metricas"]),
          _pct(x["metricas"]["f1_macro"]), *(_pct(v) for v in x["metricas"]["recall"]),
          f"{_pct(x['severidade']['acuracia_exata'])} / "
          f"{_pct(x['severidade']['acuracia_tolerancia_1'])}",
          f"{_pct(x['cobertura_no_limiar'])} / {_pct(x['acuracia_no_limiar'])}"]
         for nome, x in (("teste", cadeia), ("validação", cadeia_val))])
    linhas += ["", "Matriz de confusão da cadeia no teste, sem corte (linhas: verdadeira; "
               "colunas: prevista):", ""]
    linhas += fmt.tabela(["classe verdadeira", *bracol.CLASSES],
                         [[c, *map(fmt.n, linha)] for c, linha in
                          zip(bracol.CLASSES, cadeia["metricas"]["matriz_confusao"])])
    mm = r["cadeia"]["mantidas"]
    linhas += ["", "![Matriz de confusão da cadeia no teste](matriz_confusao_cadeia_teste.png)",
               "", f"Só as mantidas no corte ({mm['n']}): acurácia {_ic(mm)}, F1 macro "
               f"{_pct(mm['f1_macro'])}.", ""]
    linhas += fmt.tabela(["classe verdadeira", *bracol.CLASSES],
                         [[c, *map(fmt.n, linha)] for c, linha in
                          zip(bracol.CLASSES, mm["matriz_confusao"])])
    con, conf = r["contrato"], r["conferencia_contrato"]
    linhas += ["", "## O que o usuário recebe", "",
               f"`inferencia.diagnosticar` em cada uma das {con['imagens']} imagens: "
               + ", ".join(f"{k}: {v}" for k, v in con["desfechos"].items())
               + f". Folhas por resposta de sucesso: {con['folhas_por_resposta']}. Exceções: "
               f"{len(con['excecoes'])}.", "",
               f"**Conferência com o T1:** planos diferentes em {conf['planos_diferentes']} "
               f"imagens; nas {conf['foto_inteira']} imagens da foto inteira, "
               f"{conf['divergentes']} divergências (folha que chega ou não, e a classe)"
               + (f": ids {conf['ids_divergentes']}" if conf["divergentes"] else "") + ". "
               f"Imagem inteira da cadeia contra o classificador sozinho: "
               f"{r['conferencia_inteira']}.", "",
               "## Classificador sozinho, sem corte (só informação)", "",
               f"O caminho do `metricas_val.json` (`train.avaliar`, com AMP, época {r['epoca']}), "
               "na imagem inteira, sem cadeia nem corte.", ""]
    linhas += metrics.trecho_relatorio(m, ms)
    linhas += ["", "![Matriz de confusão do classificador no teste](matriz_confusao_teste.png)",
               "", "**Teste contra validação e validação cruzada** (os mesmos números dos "
               "arquivos; a validação cruzada usa modelos de 20 épocas, mais fracos que o run "
               "base, nas 1.432 imagens de treino e validação):", ""]
    fontes = (("teste", "este run", m),
              ("validação", ref["classificador_val"]["arquivo"],
               ref["classificador_val"]["classificacao"]),
              ("validação cruzada, protocolo por blocos (o CV realista)", ref["cv"]["arquivo"],
               ref["cv"]["blocos"]),
              ("validação cruzada, protocolo aleatório", ref["cv"]["arquivo"],
               ref["cv"]["aleatorio"]))
    linhas += fmt.tabela(
        ["fonte", "arquivo", "imagens", "acurácia (IC 95%)", "F1 macro",
         *(f"recall {c}" for c in bracol.CLASSES)],
        [[nome, f"`{arquivo}`" if arquivo != "este run" else arquivo, fmt.n(x["n"]), _ic(x),
          _pct(x["f1_macro"]), *(_pct(v) for v in x["recall"])]
         for nome, arquivo, x in fontes], "llrlllllll")
    linhas += ["", "## Leitura", "", leitura or "Ainda não escrita (falta o leitura.md).", "",
               "## Ressalvas", "",
               f"- **Phoma e cercosporiose:** {bracol.RESSALVA_PHOMA_CERCOSPORA}",
               train._nota_cercosporiose(m["suporte"][bracol.CLASSES.index("cercosporiose")],
                                         "no teste"),
               "- **Comparação com Esgario et al. (2020):** a divisão é outra (ver "
               "`data/README.md`); os números não são diretamente comparáveis.", ""]
    return "\n".join(linhas) + "\n"


def relatorio_uso_2(r: dict, leitura: str | None) -> str:
    teste, ref = r["teste"], r["referencias"]
    val = ref["validacao"]
    linhas = _cabecalho("2", r) + [
        f"As fotos de teste do BRACOT pelo `detector_base` (`weights/last.pt`), com as métricas "
        f"da validação (`evaluation/metricas_deteccao.py`) e o limiar escolhido nela "
        f"({fmt.decimal(teste['limiar'], 2)}). Modo região: a previsão com mais da metade da "
        "máscara fora da região anotada é ignorada (a anotação é parcial); modo conservador: "
        "ela conta como erro. Sem o validador do ultralytics, que exigiria copiar as fotos de "
        f"teste. Validação: `{val['arquivo']}`.", ""]
    conjuntos = (("teste, 60 fotos", teste["todas"]["fotos"], teste["todas"]["anotadas"],
                  teste["todas"]["metricas"], teste["todas"]["no_limiar"]),
                 ("teste, sem as 2 sobrepostas", teste["sem_sobrepostas"]["fotos"],
                  teste["sem_sobrepostas"]["anotadas"], teste["sem_sobrepostas"]["metricas"],
                  teste["sem_sobrepostas"]["no_limiar"]),
                 ("validação", val["fotos"], val["anotadas"], val["metricas"], val["no_limiar"]))
    linhas += fmt.tabela(
        ["conjunto", "fotos", "folhas anotadas", "modo", "AP50 máscara", "AP50-95 máscara",
         "AP50 caixa", "AP50-95 caixa", "precisão no limiar", "recall no limiar",
         "F1 no limiar"],
        [[nome, str(fotos), str(anotadas), td.NOMES_DOS_MODOS[modo],
          _pct(met[modo]["mascara"]["ap50"]), _pct(met[modo]["mascara"]["ap50_95"]),
          _pct(met[modo]["caixa"]["ap50"]), _pct(met[modo]["caixa"]["ap50_95"]),
          _pct(lim[modo]["precisao"]), _pct(lim[modo]["recall"]), _pct(lim[modo]["f1"])]
         for nome, fotos, anotadas, met, lim in conjuntos for modo in ("regiao", "conservador")],
        "lrrlrrrrrrr")
    linhas += ["", "**Folhas por foto, anotadas contra previstas no limiar** "
               f"(`contagens_teste.csv`; validação: `{ref['validacao_contagens']['arquivo']}`):",
               ""]
    linhas += fmt.tabela(
        ["conjunto", "fotos", "anotadas", "previstas", "acertos", "fora da região",
         "anotadas por foto (mediana; mín. a máx.)", "previstas por foto (mediana; mín. a máx.)",
         "previstas - anotadas por foto (média; mediana; mín. a máx.)",
         "fotos com mais / menos / iguais"],
        [[nome, str(c["fotos"]), str(c["anotadas"]), str(c["previstas"]), str(c["acertos"]),
          str(c["fora_da_regiao"]), _faixa_texto(c["anotadas_por_foto"]),
          _faixa_texto(c["previstas_por_foto"]),
          f"{fmt.decimal(c['diferenca_por_foto']['media'], 1)}; "
          + _faixa_texto(c["diferenca_por_foto"]),
          f"{c['fotos_com_mais_previstas']} / {c['fotos_com_menos_previstas']} / "
          f"{c['fotos_iguais']}"]
         for nome, c in (("teste, 60 fotos", r["contagens"]["todas"]),
                         ("teste, sem as 2 sobrepostas", r["contagens"]["sem_sobrepostas"]),
                         ("validação", ref["validacao_contagens"]))], "lrrrrrllll")
    linhas += ["", "Como a anotação é parcial, previstas a mais não são necessariamente erros "
               "(o modo região as separa).", "", "## Leitura", "",
               leitura or "Ainda não escrita (falta o leitura.md).", ""]
    return "\n".join(linhas) + "\n"


def _faixa_texto(f: dict) -> str:
    return (f"{fmt.decimal(f['mediana'], 1)}; {fmt.decimal(f['min'], 0)} a "
            f"{fmt.decimal(f['max'], 0)}")


def relatorio_uso_3a(r: dict, leitura: str | None) -> str:
    c = r["cortadas"]
    linhas = _cabecalho("3a", r) + [
        f"As {r['fotos']} fotos de teste do BRACOT, sem as de `TESTE_SOBREPOSTO` "
        f"({', '.join(r['sobrepostas_fora'])}), pela cadeia final até as folhas: detector, área "
        "mínima, filtro de cor, limite de folhas e filtro de fragmentos (0,5). **O classificador "
        "não foi carregado:** nenhuma predição foi calculada. As anotações de teste não foram "
        "lidas.", "",
        f"- **Planos das fotos:** {r['planos']}.",
        f"- **Detecções no limiar:** {r['deteccoes_no_limiar']}; cortadas pela área "
        f"{c['area']}, pela cor {c['cor']}, pelo filtro de fragmentos {c['fragmento']} (em "
        f"{r['fotos_com_fragmento_cortado']} fotos) e pelo limite {c['limite']}.",
        f"- **Folhas que ficam:** {r['folhas']}; por foto, {_faixa_texto(r['folhas_por_foto'])}.",
        f"- **Sorteio:** até {r['max_por_foto']} folhas por foto, semente {r['semente']} "
        f"(`campo.sortear_rotulagem`): {r['sorteadas']} folhas; fotos por número de folhas "
        f"sorteadas: {r['fotos_por_sorteadas']}.",
        f"- **Pranchas cegas:** {r['pranchas']}, em `{r['pasta_pranchas']}` (fora do git), no "
        "formato das anteriores: id numerado, recorte B grande e A pequeno, sem predição nem "
        "score.",
        f"- **CSV vazio:** `{r['csv']}`, com as colunas dos outros dois.",
        "- Contagens por foto em `contagens.csv`.", "",
        "## Leitura", "", leitura or "Ainda não escrita (falta o leitura.md).", ""]
    return "\n".join(linhas) + "\n"


def relatorio_uso_3b(r: dict, leitura: str | None) -> str:
    linhas = _cabecalho("3b", r) + [
        "As folhas rotuladas das fotos de teste do BRACOT pela cadeia final congelada (estrutura "
        "do passo 6b, `evaluation/campo_cadeia_final.py`). Os números com o corte t_d "
        "desligado são só informação, declarada antes da abertura.", "",
        f"**Contagens:** {r['contagens']}; destinos na cadeia: {r['destinos']}; planos das "
        f"fotos: {r['planos']}; rótulos sem par: {r['nao_casados']}.", "",
        "## Destino de S, D e \"?\" (cadeia congelada)", ""]
    linhas += fmt.tabela(
        ["rótulo", "folhas", "chamada saudável", "chamada doente e mantida", "classes",
         "descartada", *(f"descartada: {ccf.NOME_MOTIVO[x]}" for x in ccf.MOTIVOS)],
        [[rot, str(x["n"]), _prop(x["saudavel"]), _prop(x["doenca"]),
          ccf._classes(x["classes_doentes"]), _prop(x["descartada"]),
          *(_prop(x["motivos"][mot]) for mot in ccf.MOTIVOS)]
         for rot, x in r["final"]["rotulos"].items()], "lrlllllll")
    linhas += ["", "## D evidentes: o que chega ao usuário", ""]
    linhas += fmt.tabela(
        ["cenário", "D", "como doente", "como saudável", "somem",
         *(f"somem: {ccf.NOME_MOTIVO[x]}" for x in ccf.MOTIVOS)],
        [[ccf.NOME_CENARIO[cen], str(x["n"]), _prop(x["doenca"]), _prop(x["saudavel"]),
          _prop(x["descartada"]), *(_prop(x["motivos"][mot]) for mot in ccf.MOTIVOS)]
         for cen in ccf.CENARIOS for x in (r[cen]["rotulos"]["D"],)], "lrllllll")
    linhas += ["", "## Doentes mantidas: classes e confiança", "",
               "As classes são as previstas; os rótulos de campo não têm classe.", ""]
    linhas += fmt.tabela(
        ["cenário", "rótulo", "mantidas", *ccf.DOENCAS, "confiança"],
        [[ccf.NOME_CENARIO[cen], grupo, str(x["n"]),
          *(_prop(x["classes"][cl]) if x["n"] else "-" for cl in ccf.DOENCAS),
          ccf._conf(x["confianca"])]
         for cen in ccf.CENARIOS for grupo, x in r[cen]["doentes_mantidas"].items()],
        "llrlllll")
    linhas += ["", "## Separação S x D nas folhas classificadas", ""]
    linhas += fmt.tabela(
        ["cenário dos \"?\"", "S / D", "AUC de P(saudável) (IC 95%, bootstrap por foto)",
         "saudável dado S (final)", "saudável dado D (final)", "saudável dado D (sem t_d)"],
        [[cv.NOME_CENARIO[cen], f"{x['S']} / {x['D']}", cv._auc(x["auc"]),
          _prop(x["regras"]["final"]["saudavel_dado_s"]),
          _prop(x["regras"]["final"]["saudavel_dado_d"]),
          _prop(x["regras"]["sem_td"]["saudavel_dado_d"])]
         for cen, x in r["separacao"].items()], "llllll")
    ref = r["referencias"]
    linhas += ["", "## Comparação com o passo 6b, sem ajuste", "",
               f"Fontes: `{ref['campo_6b']['arquivo']}` e `{ref['auc_validacao']['arquivo']}`.",
               ""]
    linhas += fmt.tabela(
        ["conjunto", "S chamada saudável", "D chamada saudável", "D chamada doente e mantida",
         "D descartada", "AUC de P(saudável)"],
        [["teste", _prop(r["final"]["rotulos"]["S"]["saudavel"]),
          _prop(r["final"]["rotulos"]["D"]["saudavel"]),
          _prop(r["final"]["rotulos"]["D"]["doenca"]),
          _prop(r["final"]["rotulos"]["D"]["descartada"]),
          cv._auc(r["separacao"]["sem_interrogacao"]["auc"])],
         *([c, _prop(ref["campo_6b"][c]["S"]["saudavel"]),
            _prop(ref["campo_6b"][c]["D"]["saudavel"]), _prop(ref["campo_6b"][c]["D"]["doenca"]),
            _prop(ref["campo_6b"][c]["D"]["descartada"]), cv._auc(ref["auc_validacao"][c])]
           for c in ("antigas", "novas", "juntas"))], "llllll")
    linhas += ["", "## Leitura", "", leitura or "Ainda não escrita (falta o leitura.md).", "",
               "## Limites", "",
               f"- Folhas (S, D ou \"?\") {ORIGEM_ROTULOS_TESTE}; nenhuma D tem classe.",
               "- Mesma lavoura das outras rotulagens, em fotos e cenas do teste dos autores.",
               "- As folhas de teste foram sorteadas já com o filtro de fragmentos ligado; as "
               "antigas do passo 6b, sem ele (por isso o filtro tira folhas só nas antigas).",
               ""]
    return "\n".join(linhas) + "\n"


RELATORIOS = {"1": relatorio_uso_1, "2": relatorio_uso_2, "3a": relatorio_uso_3a,
              "3b": relatorio_uso_3b}


def refazer_relatorio(uso: str) -> int:
    """Só refaz o relatorio.md da última rodada do uso a partir do resultado.json."""
    rodadas = sorted(p for p in PASTAS[uso].parent.glob(PASTAS[uso].name + "*")
                     if p.is_dir() and not p.name.endswith(".tmp"))
    if not rodadas:
        raise train.ErroFatal(f"o uso {uso} ainda não rodou")
    pasta = rodadas[-1]
    resultado = json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
    (pasta / "relatorio.md").write_text(RELATORIOS[uso](resultado, _leitura(pasta)),
                                        encoding="utf-8", newline="\n")
    registrar(uso, RELATORIO, f"{_relativo_ou_nome(pasta / 'relatorio.md')}, SHA-256 "
                              f"{sha256(pasta / 'relatorio.md')}")
    fmt.dizer(f"Relatorio refeito: {pasta / 'relatorio.md'}")
    return 0


# -------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Passo 7: uso unico dos testes fechados. Detalhes no docstring.")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--abrir", action="store_true",
                       help="cria o log antes de qualquer leitura do teste")
    grupo.add_argument("--uso", choices=USOS, help="roda um uso (uma vez)")
    parser.add_argument("--relatorio", action="store_true",
                        help="com --uso: so refaz o relatorio.md a partir do resultado.json")
    parser.add_argument("--nova-tentativa", help="o que foi corrigido (so depois de um erro)")
    parser.add_argument("--decisao-do-gestor",
                        help="autorizacao do gestor para outra rodada (texto, data)")
    args = parser.parse_args(argv)
    try:
        if args.abrir:
            return abrir()
        if args.relatorio:
            return refazer_relatorio(args.uso)
        return executar_uso(args.uso, args.nova_tentativa, args.decisao_do_gestor)
    except (train.ErroFatal, FileNotFoundError, ValueError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
