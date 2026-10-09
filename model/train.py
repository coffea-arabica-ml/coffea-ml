"""
Treino-base do classificador (Frente 9, passo 1): backbone pré-treinado do timm com duas
cabeças lineares sobre as features, classe (5 saídas, na ordem de CLASSES) e severidade (níveis
0 a 4), e perda = CE(classe) + peso_sev * CE(severidade).

Neste passo o treino é só com o BRACOL. Validação e teste ficam intactos; o RD02 entra no
treino como amostragem balanceada por classe e augmentation moderada, em memória
(model/dados.py). O teste só é avaliado no comando separado --avaliar-teste, e cada chamada
fica registrada em model/runs/uso_do_teste.md.

Uso (de qualquer pasta, com o venv ativo):
    python model/train.py                              treina e grava em model/runs/<nome>/
    python model/train.py --nome NOME                  o mesmo, com o nome escolhido
    python model/train.py --rapido                     teste de fumaça: 48 imagens de treino e
                                                       24 de validação, 224x112, 2 épocas, em
                                                       model/runs/_rapido/
    python model/train.py --avaliar-teste --run NOME   avalia o melhor checkpoint no teste
    python model/train.py --help                       todos os parâmetros

Cada treino grava em model/runs/<nome>/: config.json, historico.csv, melhor.pt (fora do git),
predicoes_val.csv, metricas_val.json, matriz_confusao_val.png e relatorio.md.
O código de saída é 0 sem erros e 1 com erros.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# A barra de progresso do download dos pesos usa caracteres que o terminal do Windows não
# mostra. A variável precisa vir antes do import do timm.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "evaluation"):  # bracol e formatacao; metrics
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import bracol
import dados
import formatacao as fmt
import metrics

PASTA_RUNS = RAIZ_REPO / "model" / "runs"
NOME_RAPIDO = "_rapido"
USO_DO_TESTE = "uso_do_teste.md"  # em PASTA_RUNS
N_NIVEIS = len(bracol.SEVERIDADES)  # severidade: níveis 0 a 4
WORKERS_PADRAO = 0 if sys.platform == "win32" else 2
# Modo --rapido: teste de fumaça do fluxo inteiro; não mede o modelo.
RAPIDO = {"treino": 48, "val": 24, "largura": 224, "altura": 112, "epocas": 2}
PROGRESSO_A_CADA = 10  # lotes entre as mensagens de progresso dentro de uma época
CABECALHO_USO_DO_TESTE = """# Uso do conjunto de teste

Cada linha é uma chamada de `python model/train.py --avaliar-teste --run <nome>`. O teste (só
BRACOL) mede o modelo escolhido; escolher modelo ou hiperparâmetros olhando para ele invalida
a medida.

| data | run |
|---|---|
"""


class ErroFatal(Exception):
    """Problema que impede continuar: run já existente, checkpoint ausente ou incompatível."""


# ------------------------------------------------------------------------------ modelo
class ModeloMultitarefa(nn.Module):
    """Backbone do timm sem classificador e duas cabeças lineares sobre as mesmas features:
    classe (len(CLASSES) saídas) e severidade (N_NIVEIS saídas, níveis 0 a 4)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.cabeca_classe = nn.Linear(backbone.num_features, len(bracol.CLASSES))
        self.cabeca_severidade = nn.Linear(backbone.num_features, N_NIVEIS)

    def forward(self, x):
        features = self.backbone(x)
        return self.cabeca_classe(features), self.cabeca_severidade(features)


def criar_modelo(backbone: str = "resnet50", pretrained: bool = True) -> ModeloMultitarefa:
    """Modelo multitarefa sobre timm.create_model(backbone, num_classes=0).

    Com pretrained=True, os pesos do ImageNet vêm do cache do huggingface e só a primeira vez
    baixa: sem o arquivo do cache, o timm consultaria a internet a cada execução.
    """
    extra = {}
    if pretrained:
        arquivo = pesos_em_cache(backbone)
        if arquivo:
            extra["pretrained_cfg_overlay"] = {"file": arquivo}
        else:
            fmt.dizer(f"Baixando os pesos pré-treinados de {backbone} "
                      f"({origem_dos_pesos(backbone)}); só desta vez, eles ficam no cache do "
                      "huggingface...")
    return ModeloMultitarefa(
        timm.create_model(backbone, pretrained=pretrained, num_classes=0, **extra)
    )


def origem_dos_pesos(backbone: str) -> str | None:
    """Repositório do huggingface com os pesos pré-treinados (ex.: timm/resnet50.a1_in1k)."""
    try:
        cfg = timm.models.get_pretrained_cfg(backbone)
    except RuntimeError:  # arquitetura conhecida com etiqueta inválida
        return None
    return cfg.hf_hub_id if cfg else None


def pesos_em_cache(backbone: str) -> str | None:
    """Arquivo dos pesos pré-treinados no cache local do huggingface, ou None se ainda não foram
    baixados. Não acessa a internet."""
    from huggingface_hub import try_to_load_from_cache

    repositorio = origem_dos_pesos(backbone)
    if not repositorio:
        return None
    for nome in ("model.safetensors", "pytorch_model.bin"):
        arquivo = try_to_load_from_cache(repositorio, nome)
        if isinstance(arquivo, str):
            return arquivo
    return None


def perda_multitarefa(logits_classe, logits_sev, alvo_classe, alvo_sev, peso_sev: float = 1.0):
    """CE(classe) + peso_sev * CE(severidade).

    As linhas com severidade -1 (sem alvo) ficam fora da média da severidade. Num lote sem
    nenhum alvo, essa parcela é 0: a média do PyTorch daria 0/0 = NaN. Devolve
    (total, perda_classe, perda_severidade).
    """
    perda_classe = F.cross_entropy(logits_classe, alvo_classe)
    com_alvo = (alvo_sev != dados.SEM_SEVERIDADE).sum()
    perda_sev = F.cross_entropy(logits_sev, alvo_sev, ignore_index=dados.SEM_SEVERIDADE,
                                reduction="sum") / com_alvo.clamp(min=1)
    return perda_classe + peso_sev * perda_sev, perda_classe, perda_sev


# ------------------------------------------------------------------------- ambiente
def semear(seed: int) -> None:
    """Sementes do random, do numpy e do torch (CPU e CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def escolher_dispositivo(pedido: str = "auto") -> torch.device:
    """cuda, se houver; senão mps (Mac); senão cpu. Ou o dispositivo pedido."""
    if pedido != "auto":
        return torch.device(pedido)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def descrever_cuda() -> str:
    """Se há CUDA e, se não há, por quê."""
    if torch.cuda.is_available():
        return f"sim ({torch.cuda.get_device_name(0)})"
    if torch.version.cuda is None:
        return f"nao (torch {torch.__version__}, compilado sem CUDA)"
    return f"nao (torch com CUDA {torch.version.cuda}, mas nenhuma GPU visivel)"


def estado_git() -> tuple[str | None, bool | None]:
    """Hash do commit atual e se há alterações fora de model/runs/ (commitadas ou não, inclusive
    arquivos novos); (None, None) sem git."""
    try:
        def git(*argumentos):
            return subprocess.run(["git", *argumentos], cwd=RAIZ_REPO, capture_output=True,
                                  text=True, timeout=30, check=True).stdout

        commit = git("rev-parse", "HEAD").strip()
        alteracoes = git("status", "--porcelain", "--", ".", ":(exclude)model/runs")
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(alteracoes.strip())


def versoes() -> dict:
    """Versões do ambiente, para o config.json."""
    import albumentations
    import cv2
    import sklearn

    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "timm": timm.__version__, "albumentations": albumentations.__version__,
        "opencv": cv2.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "sistema": platform.platform(),
    }


# ----------------------------------------------------------------------- treino e aval
def criar_loader(df, transform, batch: int, workers: int, sampler=None, pin: bool = False):
    """DataLoader do FolhasDataset: com o sampler no treino; sem ele, na ordem do DataFrame."""
    return DataLoader(
        dados.FolhasDataset(df, transform), batch_size=batch, sampler=sampler, shuffle=False,
        num_workers=workers, worker_init_fn=dados.semear_worker if workers else None,
        pin_memory=pin,
    )


def treinar_epoca(modelo, loader, otimizador, escalador, dispositivo, peso_sev: float,
                  amp: bool) -> list[float]:
    """Uma época de treino. Devolve as médias por imagem de (total, classe, severidade)."""
    modelo.train()
    somas = torch.zeros(3, dtype=torch.float64)
    vistas = 0
    inicio = time.perf_counter()
    for lote, (imagens, classes, severidades) in enumerate(loader, start=1):
        imagens = imagens.to(dispositivo, non_blocking=True)
        classes = classes.to(dispositivo, non_blocking=True)
        severidades = severidades.to(dispositivo, non_blocking=True)
        with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=amp):
            logits_classe, logits_sev = modelo(imagens)
            perdas = perda_multitarefa(logits_classe, logits_sev, classes, severidades, peso_sev)
        otimizador.zero_grad(set_to_none=True)
        escalador.scale(perdas[0]).backward()
        escalador.step(otimizador)
        escalador.update()
        somas += torch.tensor([p.item() for p in perdas], dtype=torch.float64) * len(classes)
        vistas += len(classes)
        if lote % PROGRESSO_A_CADA == 0 and lote < len(loader):
            fmt.dizer(f"  lote {lote}/{len(loader)} | perda "
                      f"{fmt.decimal(float(somas[0]) / vistas, 3)} | "
                      f"{fmt.decimal(time.perf_counter() - inicio, 0)} s")
    return (somas / vistas).tolist()


@torch.no_grad()
def avaliar(modelo, loader, dispositivo, peso_sev: float, amp: bool) -> dict:
    """Predições e perdas num conjunto sem augmentation, na ordem do DataFrame."""
    modelo.eval()
    somas = torch.zeros(3, dtype=torch.float64)
    probabilidades, sev_previstas, classes, severidades = [], [], [], []
    for imagens, classe, severidade in loader:
        imagens = imagens.to(dispositivo, non_blocking=True)
        with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=amp):
            logits_classe, logits_sev = modelo(imagens)
            perdas = perda_multitarefa(logits_classe, logits_sev, classe.to(dispositivo),
                                       severidade.to(dispositivo), peso_sev)
        somas += torch.tensor([p.item() for p in perdas], dtype=torch.float64) * len(classe)
        probabilidades.append(torch.softmax(logits_classe.float(), dim=1).cpu())
        sev_previstas.append(logits_sev.argmax(dim=1).cpu())
        classes.append(classe)
        severidades.append(severidade)
    probabilidades = torch.cat(probabilidades).numpy()
    classes = torch.cat(classes).numpy()
    return {
        "probabilidades": probabilidades,
        "classe": classes,
        "classe_prevista": probabilidades.argmax(axis=1),
        "severidade": torch.cat(severidades).numpy(),
        "severidade_prevista": torch.cat(sev_previstas).numpy(),
        "perdas": (somas / len(classes)).tolist(),
    }


def treinar(args, comando: str) -> int:
    """Treino com validação a cada época; grava o run em PASTA_RUNS/<nome>/."""
    if args.rapido:
        args.largura, args.altura, args.epocas = (RAPIDO["largura"], RAPIDO["altura"],
                                                  RAPIDO["epocas"])
        nome = NOME_RAPIDO
    else:
        nome = args.nome or f"{args.backbone}_{datetime.now():%Y%m%d_%H%M%S}"
    pasta = PASTA_RUNS / nome
    if not args.rapido and pasta.exists() and any(pasta.iterdir()):
        raise ErroFatal(f"o run {_exibir(pasta)} já existe: escolha outro --nome")
    commit = estado_git()  # antes de criar a pasta do run

    semear(args.seed)
    dispositivo = escolher_dispositivo(args.dispositivo)
    amp = dispositivo.type == "cuda"
    treino, val = dados.montar_treino(), dados.montar_val()
    if args.rapido:
        treino = dados.amostra_estratificada(treino, RAPIDO["treino"], args.seed)
        val = dados.amostra_estratificada(val, RAPIDO["val"], args.seed)
    transform_treino = dados.transformacao_treino(args.largura, args.altura, args.seed)
    loader_treino = criar_loader(treino, transform_treino, args.batch, args.workers,
                                 sampler=dados.criar_sampler(treino, args.seed), pin=amp)
    loader_val = criar_loader(val, dados.transformacao_avaliacao(args.largura, args.altura),
                              args.batch, args.workers, pin=amp)
    config = montar_config(args, nome, comando, commit, dispositivo, amp, treino, val,
                           transform_treino)

    fmt.dizer(f"Run: {_exibir(pasta)}")
    fmt.dizer(f"Dispositivo: {dispositivo} | CUDA: {descrever_cuda()} | "
              f"AMP: {'sim' if amp else 'não'} | workers: {args.workers}")
    fmt.dizer(f"Treino: {fmt.n(len(treino))} imagens ({_fontes(config['dados']['treino'])}) | "
              f"validação: {fmt.n(len(val))} ({_fontes(config['dados']['val'])}) | entrada "
              f"{args.largura}x{args.altura} | batch {args.batch} | até {args.epocas} épocas | "
              f"{args.backbone}{'' if args.sem_pretreino else ' pré-treinado'}")
    modelo = criar_modelo(args.backbone, pretrained=not args.sem_pretreino).to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=args.epocas)
    escalador = torch.amp.GradScaler("cuda", enabled=amp)

    pasta.mkdir(parents=True, exist_ok=True)
    gravar_json(config, pasta / "config.json")

    historico, melhor, sem_melhora = [], None, 0
    situacao = f"limite de {args.epocas} épocas"
    for epoca in range(1, args.epocas + 1):
        inicio = time.perf_counter()
        lr = otimizador.param_groups[0]["lr"]
        perdas_treino = treinar_epoca(modelo, loader_treino, otimizador, escalador, dispositivo,
                                      args.peso_sev, amp)
        aval = avaliar(modelo, loader_val, dispositivo, args.peso_sev, amp)
        agendador.step()
        segundos = time.perf_counter() - inicio
        m = metrics.calcular_metricas(aval["classe"], aval["classe_prevista"])
        ms = metrics.calcular_metricas_severidade(aval["severidade"], aval["severidade_prevista"])
        historico.append(linha_historico(epoca, lr, perdas_treino, aval["perdas"], m, ms,
                                         segundos))
        gravar_historico(historico, pasta / "historico.csv")
        melhorou = melhor is None or m["f1_macro"] > melhor["m"]["f1_macro"]
        if melhorou:
            melhor, sem_melhora = {"epoca": epoca, "m": m, "ms": ms}, 0
            torch.save(montar_checkpoint(modelo, config, epoca, m["f1_macro"]),
                       pasta / "melhor.pt")
            gravar_resultados(pasta, "val", val, aval, m, ms, epoca)
        else:
            sem_melhora += 1
        fmt.dizer(_resumo_da_epoca(epoca, args.epocas, perdas_treino, aval["perdas"], m, ms,
                                   segundos, melhorou))
        parar = sem_melhora >= args.paciencia
        if parar:
            situacao = (f"parada antecipada na época {epoca}: {args.paciencia} épocas sem "
                        "melhorar o F1 macro da validação")
        andamento = situacao if parar or epoca == args.epocas else (
            f"em andamento: época {epoca} de até {args.epocas}")
        (pasta / "relatorio.md").write_text(
            relatorio_treino(config, historico, melhor, andamento), encoding="utf-8",
            newline="\n")
        if parar:
            break

    tempo_medio = sum(linha["tempo_s"] for linha in historico) / len(historico)
    fmt.dizer(f"Melhor época: {melhor['epoca']} de {len(historico)} | "
              f"F1 macro val {_pct(melhor['m']['f1_macro'])} | "
              f"acurácia val {_pct(melhor['m']['acuracia'])}")
    fmt.dizer(f"Fim: {situacao} | tempo médio por época: {fmt.decimal(tempo_medio, 1)} s")
    arquivos = ", ".join(sorted(p.name for p in pasta.iterdir()))
    fmt.dizer(f"Arquivos em {_exibir(pasta)}: {arquivos}")
    return 0


def avaliar_teste(args) -> int:
    """Avalia o melhor checkpoint de um run no teste (só BRACOL) e registra o uso."""
    pasta = PASTA_RUNS / args.run
    arquivo = pasta / "melhor.pt"
    if not arquivo.is_file():
        raise ErroFatal(f"checkpoint não encontrado: {_exibir(arquivo)}")
    checkpoint = torch.load(arquivo, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != bracol.CLASSES:
        raise ErroFatal(f"as classes do checkpoint ({checkpoint['classes']}) não são as de CLASSES")
    preprocessamento = (checkpoint["media"], checkpoint["desvio"], checkpoint["interpolacao"])
    if preprocessamento != (list(dados.MEDIA), list(dados.DESVIO), dados.NOME_INTERPOLACAO):
        raise ErroFatal("o pré-processamento do checkpoint não é o de model/dados.py")
    dispositivo = escolher_dispositivo(args.dispositivo)
    amp = dispositivo.type == "cuda"
    modelo = criar_modelo(checkpoint["backbone"], pretrained=False)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.to(dispositivo)

    teste = dados.montar_teste()
    tamanho = checkpoint["tamanho_entrada"]
    loader = criar_loader(teste, dados.transformacao_avaliacao(tamanho["largura"],
                                                               tamanho["altura"]),
                          args.batch, args.workers, pin=amp)
    fmt.dizer(f"Teste: {fmt.n(len(teste))} imagens do BRACOL | run {args.run} | "
              f"época {checkpoint['epoca']} | dispositivo {dispositivo}")
    aval = avaliar(modelo, loader, dispositivo, checkpoint["config"]["peso_sev"], amp)
    registrar_uso_do_teste(args.run)
    m = metrics.calcular_metricas(aval["classe"], aval["classe_prevista"])
    ms = metrics.calcular_metricas_severidade(aval["severidade"], aval["severidade_prevista"])
    gravar_resultados(pasta, "teste", teste, aval, m, ms, checkpoint["epoca"])
    (pasta / "relatorio_teste.md").write_text(
        relatorio_teste(args.run, checkpoint, m, ms), encoding="utf-8", newline="\n")
    fmt.dizer(f"Acurácia teste {_pct(m['acuracia'])} | F1 macro teste {_pct(m['f1_macro'])}")
    fmt.dizer(f"Arquivos em {_exibir(pasta)}: relatorio_teste.md, metricas_teste.json, "
              f"predicoes_teste.csv e matriz_confusao_teste.png; uso registrado em "
              f"{_exibir(PASTA_RUNS / USO_DO_TESTE)}")
    return 0


def registrar_uso_do_teste(run: str) -> None:
    """Acrescenta uma linha (data, run) a PASTA_RUNS/uso_do_teste.md."""
    arquivo = PASTA_RUNS / USO_DO_TESTE
    if not arquivo.is_file():
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        arquivo.write_text(CABECALHO_USO_DO_TESTE, encoding="utf-8", newline="\n")
    with open(arquivo, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"| {datetime.now():%d/%m/%Y %H:%M} | {run} |\n")


# ---------------------------------------------------------------------------- arquivos
def montar_config(args, nome, comando, commit, dispositivo, amp, treino, val,
                  transform_treino) -> dict:
    """Tudo o que define o run, para o config.json e o checkpoint."""
    def resumo(df):
        return {
            "n": len(df),
            "por_fonte": {f: int(q) for f, q in df["fonte"].value_counts().sort_index().items()},
            "por_classe": {c: int((df["classe"] == c).sum()) for c in bracol.CLASSES},
        }

    config = {
        "nome": nome,
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "comando": comando,
        "commit": commit[0],
        "commit_com_alteracoes": commit[1],
        "rapido": args.rapido,
        "backbone": args.backbone,
        "pretreino": not args.sem_pretreino,
        "pesos": None if args.sem_pretreino else origem_dos_pesos(args.backbone),
        "largura": args.largura,
        "altura": args.altura,
        "epocas": args.epocas,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "agendador": "CosineAnnealingLR, T_max = epocas, um passo por época",
        "paciencia": args.paciencia,
        "peso_sev": args.peso_sev,
        "seed": args.seed,
        "workers": args.workers,
        "dispositivo": str(dispositivo),
        "amp": amp,
        "classes": list(bracol.CLASSES),
        "niveis_severidade": list(bracol.SEVERIDADES),
        "media": list(dados.MEDIA),
        "desvio": list(dados.DESVIO),
        "interpolacao": dados.NOME_INTERPOLACAO,
        "amostragem": "WeightedRandomSampler: peso 1/frequência da classe no treino, "
                      "len(treino) sorteios por época, com reposição",
        "augmentation": dados.descrever_transformacao(transform_treino),
        "dados": {"treino": resumo(treino), "val": resumo(val)},
        "versoes": versoes(),
    }
    return json.loads(json.dumps(config))  # só tipos do JSON: o checkpoint abre com weights_only


def montar_checkpoint(modelo, config: dict, epoca: int, f1_macro: float) -> dict:
    """melhor.pt: os pesos e o que o backend precisa para usá-los (classes, tamanho de entrada,
    interpolação, ordem dos canais, média e desvio). Abre com torch.load(weights_only=True)."""
    return {
        "state_dict": modelo.state_dict(),
        "backbone": config["backbone"],
        "classes": list(bracol.CLASSES),
        "niveis_severidade": list(bracol.SEVERIDADES),
        "tamanho_entrada": {"largura": config["largura"], "altura": config["altura"]},
        "interpolacao": dados.NOME_INTERPOLACAO,
        "ordem_canais": "RGB",
        "media": list(dados.MEDIA),
        "desvio": list(dados.DESVIO),
        "epoca": epoca,
        "f1_macro_val": f1_macro,
        "config": config,
    }


def gravar_resultados(pasta: Path, conjunto: str, df, aval: dict, m: dict, ms: dict,
                      epoca: int) -> None:
    """predicoes_<conjunto>.csv, metricas_<conjunto>.json e matriz_confusao_<conjunto>.png."""
    predicoes = pd.DataFrame({
        "id": df["id"].to_numpy(),
        "classe_verdadeira": df["classe"].to_numpy(),
        "classe_prevista": [bracol.CLASSES[i] for i in aval["classe_prevista"]],
        **{f"prob_{c}": aval["probabilidades"][:, k] for k, c in enumerate(bracol.CLASSES)},
        "severidade_verdadeira": aval["severidade"],
        "severidade_prevista": aval["severidade_prevista"],
    })
    predicoes.to_csv(pasta / f"predicoes_{conjunto}.csv", index=False, float_format="%.6f",
                     lineterminator="\n")
    gravar_json({"conjunto": conjunto, "epoca": epoca, "classificacao": m, "severidade": ms},
                pasta / f"metricas_{conjunto}.json")
    nome = {"val": "validação", "teste": "teste"}[conjunto]
    figura = metrics.desenhar_matriz_confusao(
        m["matriz_confusao"], f"Matriz de confusão: {nome} ({fmt.n(m['n'])} imagens)"
    )
    metrics.salvar_figura(figura, pasta / f"matriz_confusao_{conjunto}.png")


def gravar_json(objeto, caminho: Path) -> None:
    caminho.write_text(json.dumps(_arredondar(objeto), ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8", newline="\n")


def linha_historico(epoca, lr, perdas_treino, perdas_val, m, ms, segundos) -> dict:
    """Uma linha do historico.csv: perdas, métricas da validação e tempo da época."""
    linha = {"epoca": epoca, "lr": lr}
    for conjunto, perdas in (("treino", perdas_treino), ("val", perdas_val)):
        linha.update(zip([f"perda_{conjunto}", f"perda_classe_{conjunto}",
                          f"perda_sev_{conjunto}"], perdas))
    linha.update({"acuracia_val": m["acuracia"], "f1_macro_val": m["f1_macro"]})
    linha.update({f"recall_val_{c}": r for c, r in zip(m["classes"], m["recall"])})
    linha.update({"mae_sev_val": ms["mae"], "kappa_sev_val": ms["kappa_quadratico"],
                  "tempo_s": round(segundos, 1)})
    return _arredondar(linha)


def gravar_historico(historico: list[dict], caminho: Path) -> None:
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(historico[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(historico)


def _arredondar(objeto, digitos: int = 6):
    """Floats com 6 algarismos significativos, em qualquer profundidade (json e csv legíveis)."""
    if isinstance(objeto, float):
        return float(f"{objeto:.{digitos}g}")
    if isinstance(objeto, dict):
        return {k: _arredondar(v, digitos) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return [_arredondar(v, digitos) for v in objeto]
    return objeto


# ------------------------------------------------------------------------- relatórios
def relatorio_treino(config: dict, historico: list[dict], melhor: dict, situacao: str) -> str:
    """relatorio.md do run: configuração, dados, métricas da melhor época na validação,
    histórico e ressalvas."""
    d = config["dados"]
    treino, val = d["treino"]["por_classe"], d["val"]["por_classe"]
    linhas = [f"# Treino-base do classificador: run `{config['nome']}`", ""]
    if config["rapido"]:
        linhas += [
            f"> **Execução rápida (`--rapido`):** teste de fumaça do fluxo, com "
            f"{fmt.n(d['treino']['n'])} imagens de treino e {fmt.n(d['val']['n'])} de "
            f"validação, entrada {config['largura']}x{config['altura']} e {config['epocas']} "
            "épocas. Os números abaixo não medem o modelo.", "",
        ]
    linhas += [f"Gerado por `{config['comando']}` em {_data(config['criado_em'])}, "
               f"{_origem(config)}.", ""]

    pesos = (f"pré-treinado no ImageNet (pesos `{config['pesos']}`)" if config["pretreino"]
             else "sem pré-treino (pesos aleatórios)")
    tempo_medio = sum(linha["tempo_s"] for linha in historico) / len(historico)
    linhas += [
        "## Configuração", "",
        f"- **Modelo:** `{config['backbone']}` do timm, {pesos}, com duas cabeças lineares sobre "
        f"as features: classe ({len(config['classes'])} saídas, na ordem de `CLASSES`) e "
        "severidade (níveis 0 a 4).",
        f"- **Perda:** CE(classe) + {_numero(config['peso_sev'])} × CE(severidade); as linhas sem "
        "alvo de severidade ficam fora da média.",
        f"- **Entrada:** {config['largura']}x{config['altura']} (largura x altura), "
        f"redimensionada com `{config['interpolacao']}` e normalizada com a média e o desvio do "
        "ImageNet.",
        f"- **Otimização:** AdamW (lr {_numero(config['lr'])}, weight decay "
        f"{_numero(config['weight_decay'])}), cosseno, até {config['epocas']} épocas, batch "
        f"{config['batch']}, parada antecipada pelo F1 macro da validação (paciência "
        f"{config['paciencia']}), semente {config['seed']}.",
        f"- **RD02, só no treino:** amostragem balanceada por classe (peso 1/frequência da "
        f"classe, {fmt.n(d['treino']['n'])} sorteios por época, com reposição) e augmentation "
        f"moderada, em memória: espelhamento horizontal e vertical, rotação de até "
        f"{dados.ROTACAO_MAX}° e brilho e contraste de até "
        f"±{fmt.decimal(100 * dados.BRILHO_CONTRASTE_MAX, 0)}%. A validação fica intacta.",
        f"- **Execução:** {config['dispositivo']}, {'com' if config['amp'] else 'sem'} AMP; "
        f"{len(historico)} épocas rodadas, {fmt.decimal(tempo_medio, 1)} s por época em média; "
        f"{situacao}.",
        "",
        "## Dados", "",
    ]
    linhas += fmt.tabela(
        ["classe", "treino", "validação"],
        [[c, fmt.n(treino[c]), fmt.n(val[c])] for c in bracol.CLASSES]
        + [["**total**", f"**{fmt.n(d['treino']['n'])}**", f"**{fmt.n(d['val']['n'])}**"]],
    )
    linhas += ["", f"- **Fontes:** treino com {_fontes(d['treino'])}; validação com "
                   f"{_fontes(d['val'])}."]
    if "jmuben" not in d["treino"]["por_fonte"]:
        linhas.append("- O JMuBEN não entra neste treino.")
    linhas += [_nota_cercosporiose(val["cercosporiose"], "na validação"), ""]

    linhas += [f"## Validação (melhor época: {melhor['epoca']})", ""]
    linhas += metrics.trecho_relatorio(melhor["m"], melhor["ms"])
    linhas += ["", "![Matriz de confusão da validação](matriz_confusao_val.png)", ""]

    linhas += ["## Histórico por época", ""]
    linhas += fmt.tabela(
        ["época", "perda treino", "perda val", "acurácia val", "F1 macro val",
         "recall cercosporiose", "MAE sev", "kappa sev", "tempo (s)"],
        [[str(h["epoca"]) + (" (melhor)" if h["epoca"] == melhor["epoca"] else ""),
          fmt.decimal(h["perda_treino"], 3), fmt.decimal(h["perda_val"], 3),
          _pct(h["acuracia_val"]), _pct(h["f1_macro_val"]),
          _pct(h["recall_val_cercosporiose"]), _talvez(h["mae_sev_val"]),
          _talvez(h["kappa_sev_val"]), fmt.decimal(h["tempo_s"], 1)] for h in historico],
    )
    linhas += ["", "O recall de cada classe, por época, está em `historico.csv`.", ""]
    return "\n".join(linhas + _ressalvas()) + "\n"


def relatorio_teste(run: str, checkpoint: dict, m: dict, ms: dict) -> str:
    """relatorio_teste.md: métricas do melhor checkpoint no teste."""
    cercosporiose = m["suporte"][bracol.CLASSES.index("cercosporiose")]
    linhas = [
        f"# Teste do run `{run}`", "",
        f"Gerado por `python model/train.py --avaliar-teste --run {run}` em "
        f"{datetime.now():%d/%m/%Y %H:%M}, com o melhor checkpoint do treino (época "
        f"{checkpoint['epoca']}; F1 macro da validação {_pct(checkpoint['f1_macro_val'])}).", "",
        f"- Teste só do BRACOL ({fmt.n(m['n'])} imagens), sem augmentation. Este uso está "
        f"registrado em `model/runs/{USO_DO_TESTE}`.",
        _nota_cercosporiose(cercosporiose, "no teste"), "",
        "## Métricas", "",
    ]
    linhas += metrics.trecho_relatorio(m, ms)
    linhas += ["", "![Matriz de confusão do teste](matriz_confusao_teste.png)", ""]
    return "\n".join(linhas + _ressalvas(teste=True)) + "\n"


def _ressalvas(teste: bool = False) -> list[str]:
    uso = ("- **Teste:** cada avaliação no teste fica registrada em "
           f"`model/runs/{USO_DO_TESTE}`; escolher modelo ou hiperparâmetros pelo teste invalida a "
           "medida." if teste else
           "- **Teste:** não usado aqui. Ele só é avaliado com `python model/train.py "
           f"--avaliar-teste --run <nome>`, e cada uso fica em `model/runs/{USO_DO_TESTE}`.")
    return [
        "## Ressalvas", "",
        f"- **Phoma e cercosporiose:** {bracol.RESSALVA_PHOMA_CERCOSPORA}",
        "- **Comparação com Esgario et al. (2020):** as 1.685 imagens elegíveis são as mesmas, "
        "mas a divisão é outra. Aqui ela é estratificada por classe (seed 42) e mantém as folhas "
        "repetidas no mesmo split; a deles, segundo o repositório dos autores, é aleatória "
        "(seed 150), 70/15/15 com rotação em 5 dobras, com entrada de 224x224. Os números não são "
        "diretamente comparáveis (ver `data/README.md`).",
        uso, "",
    ]


def _nota_cercosporiose(n: int, onde: str) -> str:
    """Quantas imagens de cercosporiose há no conjunto ("na validação", "no teste")."""
    if not n:
        return f"- **Cercosporiose: nenhuma imagem {onde}.**"
    return (f"- **Cercosporiose: {fmt.n(n)} imagens {onde}.** Cada uma vale "
            f"{fmt.decimal(100 / n, 1)} pontos de recall dessa classe, que por isso oscila muito.")


def _resumo_da_epoca(epoca, epocas, perdas_treino, perdas_val, m, ms, segundos,
                     melhorou) -> str:
    return (f"Epoca {epoca}/{epocas} | perda treino {fmt.decimal(perdas_treino[0], 3)} "
            f"(classe {fmt.decimal(perdas_treino[1], 3)}, sev {fmt.decimal(perdas_treino[2], 3)}) "
            f"| perda val {fmt.decimal(perdas_val[0], 3)} | acc {_pct(m['acuracia'])} | "
            f"F1 macro {_pct(m['f1_macro'])} | MAE sev {_talvez(ms['mae'])} | "
            f"kappa {_talvez(ms['kappa_quadratico'])} | {fmt.decimal(segundos, 1)} s"
            + (" | melhor" if melhorou else ""))


def _fontes(resumo: dict) -> str:
    """Imagens por fonte, do resumo dos dados no config: 'bracol 1.180'."""
    return ", ".join(f"{f} {fmt.n(q)}" for f, q in resumo["por_fonte"].items())


def _origem(config: dict) -> str:
    if not config["commit"]:
        return "fora de um repositório git"
    alteracoes = " (com alterações não commitadas)" if config["commit_com_alteracoes"] else ""
    return f"no commit `{config['commit'][:7]}`{alteracoes}"


def _data(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")


def _numero(valor: float) -> str:
    """Número curto com vírgula: 0.0001 -> 0,0001; 1.0 -> 1."""
    return f"{valor:g}".replace(".", ",")


def _pct(fracao: float) -> str:
    return fmt.pct(fracao, 1)


def _talvez(valor) -> str:
    """Decimal com 3 casas, ou "-" se indefinido."""
    return "-" if valor is None else fmt.decimal(valor, 3)


def _exibir(caminho: Path) -> str:
    """Caminho relativo à raiz do repositório em formato POSIX (absoluto, se estiver fora)."""
    caminho = Path(caminho)
    if caminho.is_relative_to(RAIZ_REPO):
        return caminho.relative_to(RAIZ_REPO).as_posix()
    return str(caminho)


# -------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(
        description="Treino-base do classificador de estresse e severidade (Frente 9). "
                    "Detalhes no docstring de model/train.py."
    )
    ap.add_argument("--nome", help="nome do run em model/runs/ (padrao: <backbone>_<data>_<hora>)")
    ap.add_argument("--backbone", default="resnet50", help="modelo do timm (padrao: resnet50)")
    ap.add_argument("--sem-pretreino", action="store_true",
                    help="comeca com pesos aleatorios, sem os do ImageNet")
    ap.add_argument("--largura", type=int, default=dados.LARGURA_PADRAO,
                    help=f"largura da entrada (padrao: {dados.LARGURA_PADRAO})")
    ap.add_argument("--altura", type=int, default=dados.ALTURA_PADRAO,
                    help=f"altura da entrada (padrao: {dados.ALTURA_PADRAO})")
    ap.add_argument("--epocas", type=int, default=30, help="maximo de epocas (padrao: 30)")
    ap.add_argument("--batch", type=int, default=32, help="imagens por lote (padrao: 32)")
    ap.add_argument("--lr", type=float, default=1e-4, help="taxa de aprendizado (padrao: 1e-4)")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="weight decay do AdamW (padrao: 1e-4)")
    ap.add_argument("--paciencia", type=int, default=8,
                    help="epocas sem melhorar o F1 macro da validacao ate parar (padrao: 8)")
    ap.add_argument("--peso-sev", type=float, default=1.0,
                    help="peso da perda de severidade (padrao: 1.0)")
    ap.add_argument("--seed", type=int, default=bracol.SEED_PADRAO,
                    help=f"semente (padrao: {bracol.SEED_PADRAO})")
    ap.add_argument("--workers", type=int, default=WORKERS_PADRAO,
                    help=f"processos do DataLoader (padrao: 0 no Windows, 2 nos demais; "
                         f"aqui {WORKERS_PADRAO})")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="padrao: auto (cuda, se houver; senao mps; senao cpu)")
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--rapido", action="store_true",
                      help=f"teste de fumaca: {RAPIDO['treino']} imagens de treino e "
                           f"{RAPIDO['val']} de val, {RAPIDO['largura']}x{RAPIDO['altura']}, "
                           f"{RAPIDO['epocas']} epocas, em model/runs/{NOME_RAPIDO}/ (ignora "
                           "--nome, --epocas, --largura e --altura)")
    modo.add_argument("--avaliar-teste", action="store_true",
                      help="avalia no teste o melhor checkpoint do run --run (nao treina)")
    ap.add_argument("--run", help="com --avaliar-teste: o nome do run em model/runs/")
    args = ap.parse_args(argv)
    if args.avaliar_teste != bool(args.run):
        ap.error("--avaliar-teste e --run vao juntos")
    if min(args.epocas, args.batch, args.paciencia, args.largura, args.altura) < 1:
        ap.error("--epocas, --batch, --paciencia, --largura e --altura precisam ser >= 1")
    try:
        if args.avaliar_teste:
            return avaliar_teste(args)
        return treinar(args, " ".join(["python model/train.py", *argv]))
    except (ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
