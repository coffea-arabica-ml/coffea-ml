"""
Treino e avaliação do detector de folhas (RF09, Frente 9, passo 2): YOLO11s-seg (ultralytics)
ajustado nas fotos de treino do BRACOT.

- Validação: cenas inteiras do treino dos autores, separadas com bracol.dividir (estratificado
  por dia, grupo = cena, semente 42). Nunca fotos soltas.
- Anotação parcial (decisão do gestor, 10/10/2026): no treino, a foto vira cinza (114, o cinza do
  letterbox do YOLO) fora da região anotada, que é o casco convexo das folhas anotadas mais 2% do
  lado maior. A validação e a inferência usam a foto inteira. As métricas saem nos modos
  "regiao" (principal) e "conservador" (evaluation/metricas_deteccao.py).
- O teste dos autores (60 fotos) fica fechado: só --avaliar-teste lê as anotações dele, no fim,
  e cada uso fica em model/runs/uso_do_teste_detector.md.

As fotos preparadas (reduzidas; as de treino, com o cinza) ficam em data/processed/bracot_yolo/,
fora do git: são derivadas do BRACOT, e nenhuma é versionada.

Uso (de qualquer pasta, com o venv ativo):
    python model/treinar_detector.py --preparar                       só prepara os dados YOLO
    python model/treinar_detector.py --nome detector_base --epocas 60 prepara (se faltar), treina
                                                                      e avalia na validação
    python model/treinar_detector.py --avaliar --run detector_base    refaz avaliação e relatório
    python model/treinar_detector.py --avaliar-teste --run NOME       só no fim (teste fechado)

O código de saída é 0 sem erros e 1 com erros.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "evaluation"):  # bracol, bracot e formatacao; metricas_deteccao
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import bracol
import bracot
import dados
import detector
import formatacao as fmt
import metricas_deteccao as md
import train

PASTA_YOLO = RAIZ_REPO / "data" / "processed" / "bracot_yolo"  # fora do git
NOME_PADRAO = "detector_base"
LADO_PREPARADO = 1280  # lado maior das fotos preparadas para o YOLO
LADO_AVALIACAO = 1024  # lado maior das máscaras na avaliação
CINZA = detector.CINZA  # o cinza do letterbox e do mosaic do YOLO
BATCH = 16
WORKERS = 0
SEED = bracol.SEED_PADRAO
PROPORCAO_VAL = 20  # % das fotos do treino dos autores, em cenas inteiras
CONF_AVALIACAO = 0.001  # score mínimo das previsões guardadas para as curvas
MAX_DET_AVALIACAO = 100
RUN_CLASSIFICADOR = "base_resnet50_bracol"  # para medir o RNF02 com o classificador junto
LIMITE_RNF02 = 5.0  # segundos por foto (diagnóstico inteiro)
NOMES_DOS_MODOS = {"regiao": "região (principal)", "conservador": "conservador"}  # no relatório
USO_DO_TESTE = "uso_do_teste_detector.md"  # em model/runs/
CABECALHO_USO_DO_TESTE = """# Uso do conjunto de teste do detector

Cada linha é uma chamada de `python model/treinar_detector.py --avaliar-teste --run <nome>`. O
teste (as 60 fotos de teste dos autores do BRACOT) mede o detector escolhido; escolher modelo,
limiar ou hiperparâmetros olhando para ele invalida a medida.

| data | run |
|---|---|
"""
DECISOES = (
    ("09/10/2026", "o BRACOT é a fonte do detector, com a divisão oficial dos autores (240 "
                   "treino / 60 teste); o teste fica fechado até o modelo final."),
    ("10/10/2026", "anotação parcial: no treino, cinza fora do casco das folhas anotadas (margem "
                   "de 2% do lado maior); validação e inferência na foto inteira; métrica "
                   "principal ignorando a previsão com mais da metade da máscara fora da região, "
                   "mais a conservadora; contar e olhar as detecções da periferia."),
    ("10/10/2026", "ferramenta: YOLO11s-seg (ultralytics), com a licença AGPL-3.0 aceita para "
                   "este projeto acadêmico de repositórios públicos."),
    ("10/10/2026", "instalação: OpenCV só headless 5.0.0.93; ultralytics 8.4.175 com --no-deps "
                   "e as 5 dependências que faltam, fixadas; o pip check acusa o opencv-python, o "
                   "que é esperado."),
)
# Acessos à rede do passo 2 (10/10/2026): metadados (só JSON e cabeçalho), pip e o peso.
ACESSOS_A_REDE = (
    ("metadados", "https://pypi.org/pypi/ultralytics/json"),
    ("metadados", "https://pypi.org/pypi/polars/json"),
    ("metadados", "https://pypi.org/pypi/polars-runtime-32/json"),
    ("metadados", "https://pypi.org/pypi/nvidia-ml-py/json"),
    ("metadados", "https://pypi.org/pypi/ultralytics-thop/json"),
    ("metadados", "https://pypi.org/pypi/ultralytics-platform/json"),
    ("metadados", "https://api.github.com/repos/ultralytics/assets/releases/tags/v8.3.0"),
    ("cabeçalho", "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"),
    ("cabeçalho", "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"),
    ("pip", "índice do PyPI para opencv-python-headless 5.0.0.93 (o wheel veio do cache local)"),
    ("pip", "wheels do PyPI: ultralytics 8.4.175, polars 2.0.0, polars-runtime-32 2.0.0, "
            "nvidia-ml-py 13.615.71, ultralytics-thop 2.2.2, ultralytics-platform 0.1.83"),
    ("peso", detector.URL_PESO_PRETREINADO),
    ("download não autorizado", "https://ultralytics.com/assets/Arial.ttf (fonte de 755 KB, "
                                "baixada pela checagem do dataset do ultralytics no treino de 3 "
                                "épocas que mediu o tempo; o arquivo foi apagado e a pré-carga, "
                                "desligada em detector.importar_ultralytics)"),
)


# --------------------------------------------------------------------------- dados
def ler_poligonos(split: str) -> dict[str, list[np.ndarray]]:
    """{id da foto: [polígonos (k, 2) em pixels]} do COCO dos autores. O de teste só é lido por
    avaliar_teste: o teste fica fechado até o modelo final."""
    coco = json.loads(bracot.caminho_coco(split).read_text(encoding="utf-8"))
    nome = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    poligonos = {n: [] for n in nome.values()}
    for anotacao in coco["annotations"]:
        for segmento in anotacao["segmentation"]:
            poligonos[nome[anotacao["image_id"]]].append(
                np.asarray(segmento, dtype=np.float64).reshape(-1, 2))
    return poligonos


def dividir_validacao(manifest_treino, seed: int = SEED, proporcao: int = PROPORCAO_VAL):
    """Ids das fotos de validação: cenas inteiras do treino dos autores, com bracol.dividir
    (estratificado pelo dia; grupo = cena), em ordem."""
    itens = [{"id": linha.id, "grupo": linha.cena,
              "classe": linha.data_hora.strftime("%Y-%m-%d"), "severity": 0}
             for linha in manifest_treino.itertuples()]
    divisao = bracol.dividir(itens, {"treino": 100 - proporcao, "val": proporcao, "teste": 0},
                             seed=seed)
    if "teste" in divisao.values():
        raise ValueError("a divisão de validação mandou uma cena para 'teste'")
    return sorted(i for i, split in divisao.items() if split == "val")


def rotulo_yolo(poligonos, largura: int, altura: int) -> str:
    """Rótulo YOLO-seg: uma linha "0 x1 y1 x2 y2 ..." por polígono, normalizada e cortada em
    [0, 1] (alguns polígonos dos autores passam até 4 px da borda)."""
    linhas = []
    for poligono in poligonos:
        xy = np.clip(np.asarray(poligono, dtype=np.float64).reshape(-1, 2) / [largura, altura],
                     0, 1)
        linhas.append("0 " + " ".join(f"{v:.6f}" for v in xy.ravel()))
    return "\n".join(linhas) + "\n"


def cinza_fora_da_regiao(imagem, poligonos, escala: float) -> np.ndarray:
    """Cópia da imagem com CINZA fora da região anotada (md.regiao_anotada, a mesma da métrica).
    poligonos em pixels da foto original; escala = imagem / foto original."""
    altura, largura = imagem.shape[:2]
    saida = imagem.copy()
    saida[~md.regiao_anotada(poligonos, altura, largura, escala)] = CINZA
    return saida


def preparar(pasta: Path = PASTA_YOLO) -> dict:
    """Fotos e rótulos no formato do YOLO em `pasta` (fora do git): treino com cinza fora da
    região anotada, validação com a foto inteira, todas reduzidas a LADO_PREPARADO. Refaz do
    zero se os parâmetros ou a divisão mudarem; senão, só devolve o resumo gravado."""
    manifest = bracot.ler_manifest(split="treino")
    poligonos = ler_poligonos("treino")
    validacao = set(dividir_validacao(manifest))
    marca = {"lado": LADO_PREPARADO, "cinza": CINZA, "margem": md.MARGEM_REGIAO,
             "validacao": sorted(validacao),
             "sha256_coco": hashlib.sha256(bracot.caminho_coco("treino").read_bytes()).hexdigest()}
    arquivo = pasta / "preparo.json"
    if arquivo.is_file():
        anterior = json.loads(arquivo.read_text(encoding="utf-8"))
        if anterior["marca"] == marca:
            return anterior
        shutil.rmtree(pasta)  # derivado nosso (tem o preparo.json): refeito do zero
    elif pasta.exists() and any(pasta.iterdir()):
        raise train.ErroFatal(f"{train._exibir(pasta)} existe sem preparo.json; confira e apague")
    fmt.dizer(f"Preparando {len(manifest)} fotos em {train._exibir(pasta)}...")
    resumo = {s: {"fotos": 0, "folhas": 0, "cenas": set()} for s in ("treino", "val")}
    for split in resumo:
        (pasta / "images" / split).mkdir(parents=True)
        (pasta / "labels" / split).mkdir(parents=True)
    for n, linha in enumerate(manifest.itertuples(), start=1):
        split = "val" if linha.id in validacao else "treino"
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        altura, largura = imagem.shape[:2]
        escala = LADO_PREPARADO / max(altura, largura)
        reduzida = cv2.resize(imagem, (round(largura * escala), round(altura * escala)),
                              interpolation=cv2.INTER_AREA)
        if split == "treino":
            reduzida = cinza_fora_da_regiao(reduzida, poligonos[linha.id], escala)
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(reduzida, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise train.ErroFatal(f"não foi possível gravar a foto preparada de {linha.id}")
        (pasta / "images" / split / f"{linha.id}.jpg").write_bytes(jpeg.tobytes())
        (pasta / "labels" / split / f"{linha.id}.txt").write_text(
            rotulo_yolo(poligonos[linha.id], largura, altura), encoding="utf-8", newline="\n")
        resumo[split]["fotos"] += 1
        resumo[split]["folhas"] += len(poligonos[linha.id])
        resumo[split]["cenas"].add(linha.cena)
        if n % 60 == 0:
            fmt.dizer(f"  ...{n}/{len(manifest)}")
    (pasta / "dados.yaml").write_text(
        f"path: {pasta.as_posix()}\ntrain: images/treino\nval: images/val\nnames:\n  0: folha\n",
        encoding="utf-8", newline="\n")
    preparo = {"marca": marca, "resumo": {s: {"fotos": r["fotos"], "folhas": r["folhas"],
                                              "cenas": len(r["cenas"])}
                                          for s, r in resumo.items()}}
    arquivo.write_text(json.dumps(preparo, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
    return preparo


# -------------------------------------------------------------------------- treino
def conferir_peso() -> None:
    """O peso pré-treinado tem de ser o baixado da release oficial (SHA-256 registrado)."""
    arquivo = detector.PESO_PRETREINADO
    if not arquivo.is_file():
        raise train.ErroFatal(f"peso pré-treinado não encontrado: {train._exibir(arquivo)} "
                              f"(origem: {detector.URL_PESO_PRETREINADO})")
    if hashlib.sha256(arquivo.read_bytes()).hexdigest() != detector.SHA256_PESO_PRETREINADO:
        raise train.ErroFatal(f"o SHA-256 de {train._exibir(arquivo)} não é o registrado")


def treinar(nome: str, epocas: int, workers: int = WORKERS, projeto: Path | None = None) -> Path:
    """Ajusta o YOLO11s-seg nas fotos preparadas: épocas fixas, sem parada antecipada, sem AMP
    (a checagem de AMP do ultralytics baixaria outro peso) e sem gráficos (não grava imagens).
    O modelo do run é o weights/last.pt."""
    import torch

    projeto = Path(projeto or train.PASTA_RUNS)
    if (projeto / nome).exists():
        raise train.ErroFatal(f"o run {train._exibir(projeto / nome)} já existe")
    conferir_peso()
    yolo = detector.importar_ultralytics()
    modelo = yolo(str(detector.PESO_PRETREINADO))
    modelo.train(data=str(PASTA_YOLO / "dados.yaml"), imgsz=detector.TAMANHO_ENTRADA,
                 epochs=epocas, patience=epocas, batch=BATCH, seed=SEED, deterministic=True,
                 workers=workers, amp=False, plots=False, cache="ram", project=str(projeto),
                 name=nome, exist_ok=False, device=0 if torch.cuda.is_available() else "cpu")
    return projeto / nome


# ----------------------------------------------------------------------- avaliação
def resumir_fotos(modelo, ids, caminhos, poligonos, dispositivo=None):
    """resumir_foto (metricas_deteccao) de cada foto, com as previsões do modelo na foto inteira
    (score >= CONF_AVALIACAO). Devolve os resumos e as detecções, na ordem de ids."""
    resumos, deteccoes = [], []
    for i in ids:
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[i])
        folhas = detector.detectar_folhas(imagem, CONF_AVALIACAO, detector=modelo,
                                          dispositivo=dispositivo, max_det=MAX_DET_AVALIACAO)
        altura, largura = imagem.shape[:2]
        resumos.append(md.resumir_foto(poligonos[i], folhas, altura, largura, LADO_AVALIACAO))
        deteccoes.append(folhas)
    return resumos, deteccoes


def validar_com_ultralytics(modelo) -> dict:
    """mAP do próprio ultralytics na validação (foto inteira: comparável ao modo conservador)."""
    import torch

    with tempfile.TemporaryDirectory() as temporaria:  # o validador cria uma pasta de saída
        m = modelo.val(data=str(PASTA_YOLO / "dados.yaml"), split="val",
                       imgsz=detector.TAMANHO_ENTRADA, batch=BATCH, workers=0, plots=False,
                       project=temporaria, name="val", verbose=False,
                       device=0 if torch.cuda.is_available() else "cpu")
    return {"caixa": {"ap50": float(m.box.map50), "ap50_95": float(m.box.map)},
            "mascara": {"ap50": float(m.seg.map50), "ap50_95": float(m.seg.map)}}


def medir_rnf02(pesos: Path, ids, caminhos, limiar: float) -> dict:
    """Tempo do diagnóstico inteiro em CPU, por foto: leitura, detecção (YOLO em PyTorch, 640
    px) e classificação das folhas detectadas (o classificador do run base, 448x224, recorte
    pela caixa). A primeira foto aquece e fica de fora."""
    import torch

    cpu = torch.device("cpu")
    yolo = detector.importar_ultralytics()
    modelo = yolo(str(pesos))
    classificador, _ = train.carregar_modelo(train.PASTA_RUNS / RUN_CLASSIFICADOR, cpu)
    transformacao = dados.transformacao_avaliacao()
    tempos = []
    for k, i in enumerate([ids[0], *ids]):
        inicio = time.perf_counter()
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[i])
        lida = time.perf_counter()
        folhas = detector.detectar_folhas(imagem, limiar, detector=modelo, dispositivo="cpu")
        detectada = time.perf_counter()
        if folhas:
            recortes = []
            for folha in folhas:
                x0, y0, x1, y1 = (int(round(v)) for v in folha["caixa"])
                recorte = imagem[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
                recortes.append(transformacao(image=recorte)["image"])
            with torch.inference_mode():
                classificador(torch.stack(recortes))
        fim = time.perf_counter()
        if k:
            tempos.append({"id": i, "folhas": len(folhas), "leitura_s": lida - inicio,
                           "deteccao_s": detectada - lida, "classificacao_s": fim - detectada,
                           "total_s": fim - inicio})
    tabela = pd.DataFrame(tempos)
    return {"threads_torch": torch.get_num_threads(), "fotos": len(tabela),
            "mediana": {c: float(tabela[c].median()) for c in tabela.columns if c.endswith("_s")},
            "maximo": {c: float(tabela[c].max()) for c in tabela.columns if c.endswith("_s")},
            "folhas_mediana": float(tabela["folhas"].median()),
            "acima_do_limite": int((tabela["total_s"] > LIMITE_RNF02).sum())}


def metricas_da_validacao(resumos, limiar_alto: float | None = None) -> tuple[dict, dict]:
    """As métricas da validação a partir do resumir_foto de cada foto: AP nos dois modos, a
    curva por limiar, o limiar escolhido (maior F1 da métrica principal, na grade padrão) e, em
    cada limiar relatado (o escolhido e o limiar_alto, se houver), os pontos das duas curvas e
    as contagens. Devolve (saida, metricas): saida é o metricas_val.json sem o ultralytics_val e
    o rnf02; metricas, a saída de md.avaliar."""
    grade = (md.LIMIARES_SCORE if limiar_alto is None
             else np.union1d(md.LIMIARES_SCORE, [limiar_alto]))
    metricas = md.avaliar(resumos, grade)
    escolhido = md.melhor_limiar([p for p in metricas["regiao"]["curva"]
                                  if p["limiar"] in md.LIMIARES_SCORE])
    limiares = [escolhido["limiar"]] + ([limiar_alto] if limiar_alto is not None else [])
    resumo_limiares = {}
    for limiar in limiares:
        por_foto = md.contagens(resumos, limiar)
        resumo_limiares[f"{limiar:g}"] = {
            "regiao": _ponto(metricas["regiao"]["curva"], limiar),
            "conservador": _ponto(metricas["conservador"]["curva"], limiar),
            "anotadas": sum(c["anotadas"] for c in por_foto),
            "previstas": sum(c["previstas"] for c in por_foto),
            "fora_da_regiao": sum(c["fora_da_regiao"] for c in por_foto),
            "fotos_com_previsao_fora": sum(c["fora_da_regiao"] > 0 for c in por_foto),
        }
    saida = {"limiar_escolhido": escolhido["limiar"], "limiares": resumo_limiares,
             "metricas": {modo: {t: metricas[modo][t] for t in md.TIPOS} for modo in md.MODOS},
             "n_fotos": metricas["n_fotos"], "n_anotadas": metricas["n_anotadas"]}
    return saida, metricas


def avaliar_validacao(run: str, limiar_alto: float | None = None) -> int:
    """Avalia o weights/last.pt do run na validação, mede o RNF02 e grava config, métricas,
    curva, contagens e relatório na pasta do run."""
    pasta = train.PASTA_RUNS / run
    pesos = pasta / "weights" / "last.pt"
    if not pesos.is_file():
        raise train.ErroFatal(f"pesos não encontrados: {train._exibir(pesos)}")
    manifest = bracot.ler_manifest(split="treino")
    caminhos = manifest.set_index("id")["caminho"]
    ids = dividir_validacao(manifest)
    poligonos = ler_poligonos("treino")
    modelo = detector.carregar_detector(pesos)
    fmt.dizer(f"Avaliando {train._exibir(pesos)} em {len(ids)} fotos de validação...")
    resumos, _ = resumir_fotos(modelo, ids, caminhos, poligonos)
    saida, metricas = metricas_da_validacao(resumos, limiar_alto)
    pd.DataFrame([{"id": i, **c}
                  for i, c in zip(ids, md.contagens(resumos, saida["limiar_escolhido"]))]
                 ).to_csv(pasta / "contagens_val.csv", index=False, lineterminator="\n")
    curva = pd.DataFrame([{"modo": modo, **ponto} for modo in md.MODOS
                          for ponto in metricas[modo]["curva"]])
    curva.to_csv(pasta / "curva_limiar_val.csv", index=False, float_format="%.6f",
                 lineterminator="\n")
    fmt.dizer("Conferindo com o validador do ultralytics...")
    saida["ultralytics_val"] = validar_com_ultralytics(modelo)
    fmt.dizer("Medindo o RNF02 em CPU (detector + classificador)...")
    saida["rnf02"] = medir_rnf02(pesos, ids, caminhos, saida["limiar_escolhido"])
    resultados = pasta / "results.csv"
    if resultados.is_file():
        historico = pd.read_csv(resultados)
        historico.columns = [c.strip() for c in historico.columns]
        historico.to_csv(pasta / "historico.csv", index=False, lineterminator="\n")
    pd.DataFrame([{"id": i, "dia": manifest.set_index("id").loc[i, "data_hora"].date(),
                   "cena": manifest.set_index("id").loc[i, "cena"],
                   "folhas": len(poligonos[i])} for i in ids]
                 ).to_csv(pasta / "fotos_validacao.csv", index=False, lineterminator="\n")
    train.gravar_json(saida, pasta / "metricas_val.json")
    config = montar_config(run, manifest, ids, poligonos, argumentos_do_treino(modelo))
    train.gravar_json(config, pasta / "config.json")
    nota = pasta / "periferia.md"
    (pasta / "relatorio.md").write_text(
        relatorio(config, saida, metricas, historico_de(pasta),
                  nota.read_text(encoding="utf-8").strip() if nota.is_file() else None),
        encoding="utf-8", newline="\n")
    _imprimir_resumo(saida)
    return 0


def _ponto(curva, limiar: float) -> dict:
    """O ponto da curva no limiar (o limiar tem de estar na grade da curva)."""
    return next(p for p in curva if abs(p["limiar"] - limiar) < 1e-9)


def historico_de(pasta: Path):
    arquivo = pasta / "historico.csv"
    return pd.read_csv(arquivo) if arquivo.is_file() else None


# ----------------------------------------------------------------------- teste (fechado)
def resumo_do_teste(resumos: dict, limiar: float, sobrepostas) -> dict:
    """Métricas do teste com todas as fotos e sem as de TESTE_SOBREPOSTO (que repetem folhas do
    treino). resumos: {id: resumir_foto(...)}."""
    resultado = {}
    for nome, ids in (("todas", sorted(resumos)),
                      ("sem_sobrepostas", sorted(set(resumos) - set(sobrepostas)))):
        m = md.avaliar([resumos[i] for i in ids])
        resultado[nome] = {
            "fotos": len(ids), "anotadas": m["n_anotadas"],
            "metricas": {modo: {t: m[modo][t] for t in md.TIPOS} for modo in md.MODOS},
            "no_limiar": {modo: _ponto(m[modo]["curva"], limiar) for modo in md.MODOS},
        }
    return resultado


def registrar_uso_do_teste(run: str) -> None:
    """Acrescenta uma linha (data, run) a PASTA_RUNS/uso_do_teste_detector.md."""
    arquivo = train.PASTA_RUNS / USO_DO_TESTE
    if not arquivo.is_file():
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        arquivo.write_text(CABECALHO_USO_DO_TESTE, encoding="utf-8", newline="\n")
    with open(arquivo, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"| {datetime.now():%d/%m/%Y %H:%M} | {run} |\n")


def avaliar_teste(run: str) -> int:
    """Só no fim, por ordem do gestor: avalia o last.pt do run no teste dos autores, com o limiar
    escolhido na validação, com e sem TESTE_SOBREPOSTO, e registra o uso."""
    pasta = train.PASTA_RUNS / run
    arquivo_val = pasta / "metricas_val.json"
    if not arquivo_val.is_file():
        raise train.ErroFatal("avalie na validação antes (o limiar sai dela)")
    limiar = json.loads(arquivo_val.read_text(encoding="utf-8"))["limiar_escolhido"]
    modelo = detector.carregar_detector(pasta / "weights" / "last.pt")
    manifest = bracot.ler_manifest(split="teste")
    poligonos = ler_poligonos("teste")  # o único lugar que lê as anotações de teste
    ids = list(manifest["id"])
    resumos, _ = resumir_fotos(modelo, ids, manifest.set_index("id")["caminho"], poligonos)
    resultado = resumo_do_teste(dict(zip(ids, resumos)), limiar,
                                [teste for teste, _ in bracot.TESTE_SOBREPOSTO])
    registrar_uso_do_teste(run)
    train.gravar_json({"limiar": limiar, **resultado}, pasta / "metricas_teste.json")
    fmt.dizer(f"Teste: {json.dumps(resultado['todas']['metricas'], ensure_ascii=True)}")
    return 0


# --------------------------------------------------------------------------- relatório
def argumentos_do_treino(modelo) -> dict:
    """Os argumentos do treino que o checkpoint do ultralytics guarda (train_args), com os
    caminhos relativos à raiz do repositório: o args.yaml do run, com os caminhos absolutos da
    máquina, fica fora do git."""
    argumentos = (getattr(modelo, "ckpt", None) or {}).get("train_args") or {}
    saida = {}
    for chave, valor in argumentos.items():
        if isinstance(valor, (str, Path)) and Path(valor).is_absolute():
            valor = train._exibir(Path(valor))
        elif not isinstance(valor, (str, int, float, bool, list, type(None))):
            valor = str(valor)
        saida[chave] = valor
    return saida


def montar_config(run: str, manifest, ids_val, poligonos, argumentos: dict | None = None) -> dict:
    """config.json do run: decisões, dados, versões, peso, argumentos do treino e cada acesso à
    rede."""
    commit, alterado = train.estado_git()
    validacao = manifest[manifest["id"].isin(ids_val)]
    treino = manifest[~manifest["id"].isin(ids_val)]
    return {
        "run": run, "criado_em": datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "commit_com_alteracoes": alterado,
        "decisoes_do_gestor": [{"data": d, "texto": t} for d, t in DECISOES],
        "modelo": {"arquitetura": "YOLO11s-seg",
                   "ultralytics": importlib.metadata.version("ultralytics"),
                   "peso_pretreinado": detector.URL_PESO_PRETREINADO,
                   "sha256_peso": detector.SHA256_PESO_PRETREINADO, "licenca": "AGPL-3.0"},
        "treino": {"imgsz": detector.TAMANHO_ENTRADA, "batch": BATCH, "seed": SEED,
                   "workers": WORKERS, "amp": False, "plots": False, "cache": "ram",
                   "lado_preparado": LADO_PREPARADO, "cinza": CINZA,
                   "margem_regiao": md.MARGEM_REGIAO, "modelo_oficial": "weights/last.pt",
                   "argumentos_ultralytics": argumentos or {}},
        "inferencia": {"lado": detector.TAMANHO_ENTRADA, "reducao": "INTER_AREA",
                       "moldura": detector.MOLDURA, "cinza": detector.CINZA},
        "dados": {
            "fonte": "BRACOT, treino dos autores (240 fotos); teste fechado",
            "divisao_validacao": f"bracol.dividir, estratificado por dia, grupo = cena, semente "
                                 f"{SEED}, {PROPORCAO_VAL}% das fotos",
            "treino": {"fotos": len(treino), "cenas": int(treino["cena"].nunique()),
                       "folhas": int(sum(len(poligonos[i]) for i in treino["id"]))},
            "validacao": {"fotos": len(validacao), "cenas": int(validacao["cena"].nunique()),
                          "folhas": int(sum(len(poligonos[i]) for i in validacao["id"]))},
        },
        "avaliacao": {"lado_mascaras": LADO_AVALIACAO, "conf_minima": CONF_AVALIACAO,
                      "fracao_fora_max": md.FRACAO_FORA_MAX},
        "acessos_a_rede": [{"tipo": t, "url": u} for t, u in ACESSOS_A_REDE],
        "versoes": train.versoes(),
    }


def relatorio(config: dict, saida: dict, metricas: dict, historico, periferia) -> str:
    """relatorio.md do detector (UTF-8). periferia: a descrição, escrita à mão em periferia.md
    depois de olhar as fotos (sem imagens no repositório), das detecções fora da região
    anotada."""
    d = config["dados"]
    rnf = saida["rnf02"]
    linhas = [
        f"# Detector de folhas: run `{config['run']}`", "",
        f"Gerado por `python model/treinar_detector.py` em {_data(config['criado_em'])}, "
        f"{_origem(config)}. YOLO11s-seg (ultralytics {config['modelo']['ultralytics']}), "
        "ajustado nas fotos de treino do BRACOT. **O teste dos autores não foi lido nem "
        "avaliado.**", "",
        "## Decisões do gestor", "",
        *(f"- **{x['data']}:** {x['texto']}" for x in config["decisoes_do_gestor"]), "",
        "## Dados", "",
        f"- Treino: {d['treino']['fotos']} fotos, {d['treino']['cenas']} cenas, "
        f"{fmt.n(d['treino']['folhas'])} folhas, com cinza ({CINZA}) fora da região anotada.",
        f"- Validação: {d['validacao']['fotos']} fotos, {d['validacao']['cenas']} cenas, "
        f"{fmt.n(d['validacao']['folhas'])} folhas, na foto inteira ({d['divisao_validacao']}; "
        "lista em `fotos_validacao.csv`).", "",
        "## Treino", "",
        f"- Peso inicial: `{config['modelo']['peso_pretreinado']}` (SHA-256 "
        f"`{config['modelo']['sha256_peso']}`), licença AGPL-3.0.",
        f"- {detector.TAMANHO_ENTRADA} px, batch {BATCH}, semente {SEED}, determinístico, sem "
        "AMP (a checagem de AMP do ultralytics baixaria outro peso), sem gráficos (não grava "
        f"imagens), workers {config['treino']['workers']}. O modelo do run é o "
        "`weights/last.pt`, depois das épocas fixas. Os demais argumentos do ultralytics estão "
        "no `config.json`.",
        f"- Inferência (`detector.detectar_folhas`, a mesma da avaliação e do RNF02): a foto "
        f"inteira reduzida a {detector.TAMANHO_ENTRADA} px no lado maior (INTER_AREA), com "
        f"moldura de {detector.MOLDURA} px de cinza {detector.CINZA}, como o validador do "
        "ultralytics monta a entrada. Sem a moldura, a borda da foto coincide com a borda da "
        "entrada, o que o treino quase não mostra; a comparação está no docstring de "
        "`model/detector.py`.",
    ]
    if historico is not None and len(historico):
        tempo = float(historico["time"].iloc[-1])
        linhas.append(f"- {len(historico)} épocas em {fmt.decimal(tempo / 60, 1)} min "
                      f"({fmt.decimal(tempo / len(historico), 1)} s por época, com a validação "
                      "do ultralytics); histórico em `historico.csv`.")
    linhas += ["", "## Validação", "",
               "**Como ler.** A métrica principal (região) ignora a previsão sem folha "
               "anotada correspondente que tem mais da metade da máscara fora da região "
               "anotada; a conservadora conta toda previsão sem par como erro, mesmo que seja "
               "uma folha real que os autores não contornaram.", ""]
    linhas += fmt.tabela(
        ["modo", "AP50 máscara", "AP50-95 máscara", "AP50 caixa", "AP50-95 caixa"],
        [[NOMES_DOS_MODOS[modo], *(_pct(saida["metricas"][modo][t][k])
                                   for t in ("mascara", "caixa") for k in ("ap50", "ap50_95"))]
         for modo in md.MODOS],
    )
    u = saida["ultralytics_val"]
    linhas += ["", f"Conferência com o validador do ultralytics (foto inteira, comparável ao "
                   f"conservador): máscara AP50 {_pct(u['mascara']['ap50'])} e AP50-95 "
                   f"{_pct(u['mascara']['ap50_95'])}; caixa AP50 {_pct(u['caixa']['ap50'])} e "
                   f"AP50-95 {_pct(u['caixa']['ap50_95'])}. Pequenas diferenças são esperadas: o "
                   "ultralytics interpola a curva de precisão até o recall 1, casa previsões e "
                   "folhas de outro jeito (cada previsão só concorre pela folha de maior IoU), "
                   "mede a máscara em 1/4 da entrada e reduz a foto preparada de 1.280 px com "
                   "INTER_LINEAR.", "",
               "### Curva por limiar de score (máscara, IoU 0,5)", ""]
    regiao = {p["limiar"]: p for p in metricas["regiao"]["curva"]}
    conservador = {p["limiar"]: p for p in metricas["conservador"]["curva"]}
    linhas += fmt.tabela(
        ["limiar", "previstas", "fora da região", "precisão (região)", "recall", "F1 (região)",
         "precisão (conservador)", "F1 (conservador)"],
        [[fmt.decimal(lim, 2), fmt.n(r["previstas"]), fmt.n(r["fora_da_regiao"]),
          _pct(r["precisao"]), _pct(r["recall"]), _pct(r["f1"]),
          _pct(conservador[lim]["precisao"]), _pct(conservador[lim]["f1"])]
         for lim, r in regiao.items()],
    )
    linhas += ["", f"Arquivos: `curva_limiar_val.csv` e `contagens_val.csv` (por foto, no "
                   f"limiar escolhido).", "", "### Limiar", ""]
    for chave, ponto in saida["limiares"].items():
        rotulo = ("escolhido (maior F1 da métrica principal)"
                  if float(chave) == saida["limiar_escolhido"]
                  else "mais alto (para cortar a periferia)")
        regiao_p, conservador_p = ponto["regiao"], ponto["conservador"]
        linhas.append(
            f"- **{fmt.decimal(float(chave), 2)}, {rotulo}:** {fmt.n(ponto['previstas'])} folhas "
            f"previstas para {fmt.n(ponto['anotadas'])} anotadas; "
            f"{fmt.n(ponto['fora_da_regiao'])} previsões fora da região (em "
            f"{ponto['fotos_com_previsao_fora']} fotos), que não contam na métrica principal. "
            f"Região: precisão {_pct(regiao_p['precisao'])}, recall {_pct(regiao_p['recall'])}, "
            f"F1 {_pct(regiao_p['f1'])}. Conservador: precisão "
            f"{_pct(conservador_p['precisao'])}, F1 {_pct(conservador_p['f1'])}.")
    linhas += ["", "### Detecções da periferia (olhadas nas fotos, sem imagens no repositório)", "",
               periferia or "Ainda não descritas (falta o periferia.md do run).", "",
               "## RNF02 (diagnóstico inteiro em até 5 s, em CPU)", "",
               f"Medido nesta máquina, em CPU ({rnf['threads_torch']} threads do torch), nas "
               f"{rnf['fotos']} fotos de validação, depois de uma passada de aquecimento: "
               f"leitura da foto, detecção (redução, moldura e YOLO em PyTorch, "
               f"{detector.TAMANHO_ENTRADA} px, no limiar escolhido) e classificação das folhas "
               f"detectadas (o classificador `{RUN_CLASSIFICADOR}`, recorte pela caixa).", ""]
    linhas += fmt.tabela(
        ["etapa", "mediana (s)", "máximo (s)"],
        [[nome, fmt.decimal(rnf["mediana"][chave], 2), fmt.decimal(rnf["maximo"][chave], 2)]
         for nome, chave in (("leitura", "leitura_s"), ("detecção", "deteccao_s"),
                             ("classificação", "classificacao_s"), ("**total**", "total_s"))],
    )
    linhas += ["", f"Mediana de {fmt.decimal(rnf['folhas_mediana'], 0)} folhas por foto; "
                   f"{rnf['acima_do_limite']} fotos passaram de {fmt.decimal(LIMITE_RNF02, 0)} s. "
                   "A CPU do backend pode ser mais lenta que esta.", "",
               "## Licença", "",
               "O ultralytics e os pesos YOLO são AGPL-3.0 (decisão do gestor, 10/10/2026): "
               "aceita para este projeto acadêmico de repositórios públicos. Se o backend servir "
               "o modelo pela rede, o código-fonte do serviço tem de ficar disponível a quem o "
               "usa.", "",
               "## Acessos à rede neste passo", "",
               *(f"- {x['tipo']}: {x['url']}" for x in config["acessos_a_rede"]), "",
               "## Limites", "",
               "- A anotação é parcial: mesmo na região, folhas não contornadas viram erro.",
               "- O casco convexo é uma aproximação da área que os autores olharam.",
               f"- A validação tem {d['validacao']['fotos']} fotos das duas sessões de fotos do "
               "BRACOT (um dia cada). O teste dos autores vem das mesmas sessões, e 27 cenas "
               "misturam treino e teste (55 das 60 fotos de teste; `data/README.md`): o teste "
               "deve sair otimista para fotos de outras lavouras.",
               "- O limiar e a moldura da entrada foram escolhidos na própria validação, o que a "
               "deixa um pouco otimista; o teste, fechado, mede sem essa escolha.",
               "- Uma semente só; a GPU não é determinística, e o ultralytics avisa que o cache "
               "em RAM também pode mudar o resultado entre execuções.", ""]
    return "\n".join(linhas) + "\n"


def _imprimir_resumo(saida: dict) -> None:
    for modo in md.MODOS:
        m = saida["metricas"][modo]
        fmt.dizer(f"{modo}: mascara AP50 {_pct(m['mascara']['ap50'])}, AP50-95 "
                  f"{_pct(m['mascara']['ap50_95'])} | caixa AP50 {_pct(m['caixa']['ap50'])}")
    ponto = saida["limiares"][f"{saida['limiar_escolhido']:g}"]
    fmt.dizer(f"limiar {saida['limiar_escolhido']}: {ponto['previstas']} previstas, "
              f"{ponto['anotadas']} anotadas, {ponto['fora_da_regiao']} fora da regiao")
    fmt.dizer(f"RNF02: total mediano {fmt.decimal(saida['rnf02']['mediana']['total_s'], 2)} s, "
              f"maximo {fmt.decimal(saida['rnf02']['maximo']['total_s'], 2)} s")


def _origem(config: dict) -> str:
    if not config["commit"]:
        return "fora de um repositório git"
    return (f"no commit `{config['commit'][:7]}`"
            + (" (com alterações não commitadas)" if config["commit_com_alteracoes"] else ""))


def _data(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")


def _pct(valor) -> str:
    return "-" if valor is None or pd.isna(valor) else fmt.pct(valor, 1)


# -------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Treino e avaliacao do detector de folhas (YOLO11s-seg no BRACOT). "
                    "Detalhes no docstring de model/treinar_detector.py.")
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--preparar", action="store_true", help="so prepara os dados YOLO")
    modo.add_argument("--avaliar", action="store_true",
                      help="avalia um run na validacao e refaz o relatorio")
    modo.add_argument("--avaliar-teste", action="store_true",
                      help="so no fim: avalia um run no teste dos autores e registra o uso")
    ap.add_argument("--nome", default=NOME_PADRAO, help=f"nome do run (padrao: {NOME_PADRAO})")
    ap.add_argument("--epocas", type=int, help="epocas fixas do treino")
    ap.add_argument("--workers", type=int, default=WORKERS, help="processos do DataLoader")
    ap.add_argument("--projeto", type=Path,
                    help="pasta dos runs (padrao: model/runs); fora dela, so treina")
    ap.add_argument("--run", help="com --avaliar ou --avaliar-teste: o nome do run")
    ap.add_argument("--limiar-alto", type=float,
                    help="com --avaliar: um segundo limiar para comparar (periferia)")
    args = ap.parse_args(argv)
    try:
        if args.preparar:
            fmt.dizer(json.dumps(preparar()["resumo"]))
            return 0
        if args.avaliar or args.avaliar_teste:
            if not args.run:
                ap.error("--avaliar e --avaliar-teste precisam de --run")
            if args.avaliar_teste:
                return avaliar_teste(args.run)
            return avaliar_validacao(args.run, args.limiar_alto)
        if not args.epocas:
            ap.error("o treino precisa de --epocas")
        fmt.dizer(json.dumps(preparar()["resumo"]))
        treinar(args.nome, args.epocas, args.workers, args.projeto)
        if args.projeto is None:
            return avaliar_validacao(args.nome)
        return 0
    except (train.ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
