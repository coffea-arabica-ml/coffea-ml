"""
Diagnóstico por blocos de ids (Frente 9, passo 1d): mede o atalho de sessão de fotos por
validação cruzada, com dois protocolos sobre o mesmo pool (treino + validação do BRACOL, 1.432
imagens).

- Blocos: blocos de 100 ids consecutivos (1-100, 101-200, ...); o bloco i fica de fora na
  dobra i mod 5. O treino da dobra é o resto do pool menos as imagens a até 20 ids de um bloco
  de fora (purga) e as que têm o grupo (folha repetida) no conjunto de fora. Se os blocos de ids
  acompanham as sessões de fotos, as imagens de fora vêm de sessões que o modelo não viu.
- Aleatório: StratifiedGroupKFold (5 dobras, estratificado pela classe, grupos do manifest),
  com o treino de cada dobra sorteado até o mesmo número de imagens do treino de blocos da mesma
  dobra, para a comparação não misturar "menos dados" com "sessão".

A medida do atalho é D = F1 macro (aleatório) - F1 macro (blocos), sobre as predições de fora
da dobra juntas: as mesmas 1.432 imagens nos dois protocolos. Cada protocolo treina 5 modelos
com a configuração do run base, mas com 20 épocas fixas e as métricas da última época, sem
parada antecipada e sem checkpoint. O controle de cor do passo 1c roda nas mesmas dobras.

Decisão do gestor (09/10/2026): K=5 porque K=4 deixava a classe saudável com 47% das imagens
no treino da dobra 1.

Usa só treino e validação do BRACOL: o teste nunca é lido. Grava em
model/runs/diagnostico_blocos/ e é retomável: pula as dobras que já têm predições.

Uso (de qualquer pasta, com o venv ativo):
    python evaluation/diagnostico_blocos.py                    plano, treinos, controle e relatório
    python evaluation/diagnostico_blocos.py --so-plano         só o plano das dobras
    python evaluation/diagnostico_blocos.py --max-treinos 4    para depois de 4 treinos (por partes)
    python evaluation/diagnostico_blocos.py --dispositivo cpu  padrão: auto (cuda, se houver)

O código de saída é 0 sem erros e 1 com erros ou com a condição de parada.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model"):  # bracol e formatacao; dados e train
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import atalho
import bracol
import dados
import formatacao as fmt
import metrics
import train

NOME = "diagnostico_blocos"  # pasta em model/runs/
RUN_BASE = "base_resnet50_bracol"  # a configuração de treino vem do config.json dele
K = 5
BLOCO = 100
PURGA = 20
EPOCAS = 20
SEED = bracol.SEED_PADRAO
LIMIAR_CLASSE = 0.5  # fração mínima do pool de cada classe no treino de cada dobra
PROTOCOLOS = ("blocos", "aleatorio")
NOME_PROTOCOLO = {"blocos": "blocos", "aleatorio": "aleatório"}
VARIANTES_COR = ("imagem_inteira", "borda_8")
DECISAO_K = ("09/10/2026", "K=5 porque K=4 deixava a classe saudável com 47% das imagens no "
             "treino da dobra 1")
# Regra de leitura de D, em pontos de F1 macro, definida pelo gestor antes dos números.
FAIXAS_D = ((3, "pouco atalho de sessão detectado"), (8, "atenção"),
            (math.inf, "atalho relevante"))
PAPEIS = ("treino", "fora", "purga_distancia", "purga_grupo", "descartada_sorteio")
ABREVIATURA = {"saudavel": "sau", "ferrugem": "fer", "bicho_mineiro": "bic", "phoma": "pho",
               "cercosporiose": "cer"}


class CondicaoDeParada(Exception):
    """O plano deixa alguma classe com menos de LIMIAR_CLASSE do pool no treino de uma dobra."""


# ------------------------------------------------------------------------- dobras
def montar_pool() -> pd.DataFrame:
    """Treino + validação do BRACOL (dados.montar_treino e dados.montar_val), com o grupo do
    manifest, em ordem de id."""
    pool = pd.concat([dados.montar_treino(), dados.montar_val()], ignore_index=True)
    grupo_do_id = bracol.ler_manifest().set_index("id")["grupo"]
    pool = pool.assign(grupo=pool["id"].map(grupo_do_id))
    return pool.sort_values("id").reset_index(drop=True)


def dobras_por_blocos(pool, k: int = K, bloco: int = BLOCO, purga: int = PURGA) -> list[dict]:
    """Protocolo blocos: o bloco b (ids b*bloco+1 a (b+1)*bloco) fica de fora na dobra b mod k.

    O treino de cada dobra é o resto do pool menos (a) as imagens a até `purga` ids de um bloco
    de fora e (b) as imagens cujo grupo aparece no conjunto de fora. Devolve, por dobra, os ids
    de cada papel de PAPEIS.
    """
    ids = pool["id"].to_numpy()
    bloco_do_id = (ids - 1) // bloco
    n_blocos = int(bloco_do_id.max()) + 1
    dobras = []
    for d in range(k):
        blocos_fora = list(range(d, n_blocos, k))
        fora = np.isin(bloco_do_id, blocos_fora)
        perto = np.zeros(len(ids), dtype=bool)
        for b in blocos_fora:
            perto |= (ids >= b * bloco + 1 - purga) & (ids <= (b + 1) * bloco + purga)
        perto &= ~fora
        grupos_fora = set(pool.loc[fora, "grupo"])
        do_grupo = pool["grupo"].isin(grupos_fora).to_numpy() & ~fora & ~perto
        dobras.append(_dobra(fora=ids[fora], treino=ids[~(fora | perto | do_grupo)],
                             purga_distancia=ids[perto], purga_grupo=ids[do_grupo]))
    return dobras


def dobras_aleatorias(pool, n_treino: list[int], k: int = K, seed: int = SEED) -> list[dict]:
    """Protocolo aleatório: StratifiedGroupKFold (estratificado pela classe, grupos do
    manifest). O treino de cada dobra é sorteado (estratificado pela classe, mesma semente) até
    n_treino[d] imagens; as que sobram ficam como descartada_sorteio."""
    from sklearn.model_selection import StratifiedGroupKFold

    divisor = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    ids = pool["id"].to_numpy()
    dobras = []
    for d, (i_treino, i_fora) in enumerate(divisor.split(pool, pool["classe"],
                                                         groups=pool["grupo"])):
        candidatas = pool.iloc[i_treino]
        if n_treino[d] > len(candidatas):
            raise ValueError(f"dobra {d}: o aleatório tem só {len(candidatas)} imagens de treino,"
                             f" menos que as {n_treino[d]} pedidas")
        sorteio = (candidatas if n_treino[d] == len(candidatas)
                   else dados.amostra_estratificada(candidatas, n_treino[d], seed))
        treino = sorteio["id"].to_numpy()
        dobras.append(_dobra(fora=ids[i_fora], treino=treino,
                             descartada_sorteio=np.setdiff1d(candidatas["id"].to_numpy(), treino)))
    return dobras


def _dobra(**ids_por_papel) -> dict:
    """{papel: ids em ordem} com todos os PAPEIS; os que não vierem ficam vazios."""
    return {papel: np.sort(np.asarray(ids_por_papel.get(papel, []), dtype=np.int64))
            for papel in PAPEIS}


def planejar(pool, k: int = K, bloco: int = BLOCO, purga: int = PURGA,
             seed: int = SEED) -> dict[str, list[dict]]:
    """As dobras dos dois protocolos, com o treino do aleatório pareado ao de blocos."""
    blocos = dobras_por_blocos(pool, k, bloco, purga)
    return {"blocos": blocos,
            "aleatorio": dobras_aleatorias(pool, [len(d["treino"]) for d in blocos], k, seed)}


def conferir_plano(pool, planos, bloco: int = BLOCO, purga: int = PURGA) -> None:
    """Levanta ValueError se o plano quebrar uma regra: cada imagem fora uma vez por protocolo;
    um papel só por imagem em cada dobra; nenhum grupo no treino e fora ao mesmo tempo; nenhum
    id do treino de blocos a até `purga` ids de um bloco de fora; treino do aleatório com o
    mesmo n do de blocos em cada dobra."""
    ids = np.sort(pool["id"].to_numpy())
    grupo = pool.set_index("id")["grupo"]
    for protocolo, dobras in planos.items():
        if not np.array_equal(np.sort(np.concatenate([d["fora"] for d in dobras])), ids):
            raise ValueError(f"{protocolo}: cada imagem do pool tem de ficar de fora uma vez")
        for k, dobra in enumerate(dobras):
            if not np.array_equal(np.sort(np.concatenate([dobra[p] for p in PAPEIS])), ids):
                raise ValueError(f"{protocolo}, dobra {k}: cada imagem tem de ter um papel só")
            comuns = set(grupo.loc[dobra["treino"]]) & set(grupo.loc[dobra["fora"]])
            if comuns:
                raise ValueError(f"{protocolo}, dobra {k}: grupos no treino e fora: {comuns}")
            if protocolo == "blocos" and len(dobra["treino"]):
                distancia = _distancia_aos_blocos(dobra["treino"], dobra["fora"], bloco)
                if distancia.min() <= purga:
                    raise ValueError(f"blocos, dobra {k}: treino a {distancia.min()} ids de um "
                                     "bloco de fora")
    n_blocos = [len(d["treino"]) for d in planos["blocos"]]
    if [len(d["treino"]) for d in planos["aleatorio"]] != n_blocos:
        raise ValueError("o treino do aleatório não tem o mesmo n do de blocos em cada dobra")


def _distancia_aos_blocos(ids, ids_fora, bloco: int = BLOCO) -> np.ndarray:
    """Distância, em ids, de cada id ao bloco de fora mais próximo (0 dentro de um bloco)."""
    ids = np.asarray(ids)
    distancia = np.full(len(ids), np.iinfo(np.int64).max)
    for b in np.unique((np.asarray(ids_fora) - 1) // bloco):
        inicio, fim = b * bloco + 1, (b + 1) * bloco
        ate_o_bloco = np.where(ids < inicio, inicio - ids, np.where(ids > fim, ids - fim, 0))
        distancia = np.minimum(distancia, ate_o_bloco)
    return distancia


def tabela_das_dobras(pool, planos) -> pd.DataFrame:
    """Por protocolo e dobra: n de cada papel, imagens por classe no treino e fora, e a menor
    fração do pool de uma classe que fica no treino."""
    classe = pool.set_index("id")["classe"]
    total = classe.value_counts()
    linhas = []
    for protocolo, dobras in planos.items():
        for d, dobra in enumerate(dobras):
            no_treino = classe.loc[dobra["treino"]].value_counts()
            de_fora = classe.loc[dobra["fora"]].value_counts()
            fracoes = {c: no_treino.get(c, 0) / total[c] for c in bracol.CLASSES}
            pior = min(fracoes, key=fracoes.get)
            linhas.append({
                "protocolo": protocolo, "dobra": d,
                **{f"n_{papel}": len(dobra[papel]) for papel in PAPEIS},
                **{f"treino_{c}": int(no_treino.get(c, 0)) for c in bracol.CLASSES},
                **{f"fora_{c}": int(de_fora.get(c, 0)) for c in bracol.CLASSES},
                "classe_menor_fracao": pior, "menor_fracao_treino": fracoes[pior],
            })
    return pd.DataFrame(linhas)


def plano_por_imagem(pool, planos, bloco: int = BLOCO) -> pd.DataFrame:
    """plano_dobras.csv: uma linha por imagem do pool, com o bloco, a dobra em que ela fica de
    fora em cada protocolo e o papel dela em cada (protocolo, dobra)."""
    tabela = pool[["id", "classe", "grupo"]].copy()
    tabela["bloco"] = (tabela["id"] - 1) // bloco
    for protocolo, dobras in planos.items():
        dobra_fora = pd.Series(-1, index=tabela["id"])
        for d, dobra in enumerate(dobras):
            dobra_fora.loc[dobra["fora"]] = d
        tabela[f"dobra_fora_{protocolo}"] = dobra_fora.to_numpy()
    for protocolo, dobras in planos.items():
        for d, dobra in enumerate(dobras):
            papel = pd.Series("", index=tabela["id"], dtype=object)
            for nome in PAPEIS:
                papel.loc[dobra[nome]] = nome
            tabela[f"{protocolo}_d{d}"] = papel.to_numpy()
    return tabela


# ------------------------------------------------------------------------- treino
def treinar_dobra(protocolo: str, d: int, pool, dobra: dict, config_base: dict, dispositivo,
                  pasta: Path) -> float | None:
    """Treina o modelo de (protocolo, dobra) por EPOCAS épocas fixas, com a configuração do run
    base, e grava as predições da última época sobre o conjunto de fora. Pula se as predições
    já existem. Não salva o modelo. Devolve o tempo em segundos (None se pulou)."""
    arquivo = pasta / f"predicoes_{protocolo}_{d}.csv"
    if arquivo.is_file():
        return None
    inicio = time.perf_counter()
    train.semear(SEED)
    amp = dispositivo.type == "cuda"
    largura, altura, batch = config_base["largura"], config_base["altura"], config_base["batch"]
    treino = pool[pool["id"].isin(dobra["treino"])]
    fora = pool[pool["id"].isin(dobra["fora"])]
    loader_treino = train.criar_loader(
        treino, dados.transformacao_treino(largura, altura, SEED), batch, train.WORKERS_PADRAO,
        sampler=dados.criar_sampler(treino, SEED), pin=amp)
    loader_fora = train.criar_loader(fora, dados.transformacao_avaliacao(largura, altura), batch,
                                     train.WORKERS_PADRAO, pin=amp)
    modelo = train.criar_modelo(config_base["backbone"],
                                pretrained=config_base["pretreino"]).to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=config_base["lr"],
                                   weight_decay=config_base["weight_decay"])
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=EPOCAS)
    escalador = torch.amp.GradScaler("cuda", enabled=amp)
    historico = []
    for epoca in range(1, EPOCAS + 1):
        inicio_epoca = time.perf_counter()
        if amp:
            torch.cuda.reset_peak_memory_stats(dispositivo)
        lr = otimizador.param_groups[0]["lr"]
        perdas = train.treinar_epoca(modelo, loader_treino, otimizador, escalador, dispositivo,
                                     config_base["peso_sev"], amp)
        aval = train.avaliar(modelo, loader_fora, dispositivo, config_base["peso_sev"], amp)
        agendador.step()
        m = metrics.calcular_metricas(aval["classe"], aval["classe_prevista"])
        memoria = train.memoria_gpu_gb(dispositivo)
        historico.append(train._arredondar({
            "epoca": epoca, "lr": lr, "perda_treino": perdas[0],
            "perda_classe_treino": perdas[1], "perda_sev_treino": perdas[2],
            "perda_fora": aval["perdas"][0], "acuracia_fora": m["acuracia"],
            "f1_macro_fora": m["f1_macro"], "tempo_s": round(time.perf_counter() - inicio_epoca, 1),
            "memoria_gpu_gb": None if memoria is None else round(memoria, 2),
        }))
        train.gravar_historico(historico, pasta / f"historico_{protocolo}_{d}.csv")
        fmt.dizer(f"  {protocolo} dobra {d} | época {epoca}/{EPOCAS} | perda treino "
                  f"{fmt.decimal(perdas[0], 3)} | F1 macro fora {_pct(m['f1_macro'])} (só "
                  f"informativo) | {fmt.decimal(historico[-1]['tempo_s'], 1)} s")
    predicoes = train.tabela_de_predicoes(fora, aval).assign(protocolo=protocolo, dobra=d)
    temporario = arquivo.with_name(arquivo.name + ".tmp")  # só vira o arquivo final no fim
    predicoes.to_csv(temporario, index=False, float_format="%.6f", lineterminator="\n")
    temporario.replace(arquivo)
    return time.perf_counter() - inicio


# ------------------------------------------------------------------ controle de cor
def controle_de_cor(pool, planos, largura: int, altura: int) -> dict:
    """O controle de cor do passo 1c (atalho.prever_por_cor) nas mesmas dobras e nos mesmos
    treinos de cada protocolo, com as predições de fora da dobra juntas, para a imagem inteira e
    a borda de 8%."""
    caracteristicas = atalho.extrair_cor(pool, largura, altura)
    y = pool["classe"].map(dados.rotulo).to_numpy()
    posicao = pd.Series(np.arange(len(pool)), index=pool["id"])
    rotulos = {chave: rotulo for chave, rotulo, _ in atalho.VARIANTES_FUNDO}
    resultado = {}
    for protocolo, dobras in planos.items():
        resultado[protocolo] = {}
        for chave in VARIANTES_COR:
            x = caracteristicas[chave]
            previsto = np.full(len(pool), -1)
            por_dobra = []
            for d, dobra in enumerate(dobras):
                no_treino = posicao.loc[dobra["treino"]].to_numpy()
                de_fora = posicao.loc[dobra["fora"]].to_numpy()
                previsto[de_fora], _ = atalho.prever_por_cor(x[no_treino], y[no_treino],
                                                             x[de_fora])
                m = metrics.calcular_metricas(y[de_fora], previsto[de_fora])
                por_dobra.append({"dobra": d, "n": m["n"], "acuracia": m["acuracia"],
                                  "f1_macro": m["f1_macro"]})
            resultado[protocolo][chave] = {"rotulo": rotulos[chave],
                                           **metrics.calcular_metricas(y, previsto),
                                           "por_dobra": por_dobra}
    return resultado


# ---------------------------------------------------------------------- métricas
def juntar_predicoes(pasta: Path) -> dict[str, pd.DataFrame]:
    """As predições de fora da dobra de cada protocolo, juntas e em ordem de id."""
    return {
        protocolo: pd.concat([pd.read_csv(pasta / f"predicoes_{protocolo}_{d}.csv")
                              for d in range(K)]).sort_values("id").reset_index(drop=True)
        for protocolo in PROTOCOLOS
    }


def resumir(predicoes) -> dict:
    """Métricas de classificação e de severidade das predições juntas e por dobra."""
    def classificacao(tabela):
        return metrics.calcular_metricas(tabela["classe_verdadeira"].map(dados.rotulo),
                                         tabela["classe_prevista"].map(dados.rotulo))

    por_dobra = []
    for d, tabela in predicoes.groupby("dobra"):
        m = classificacao(tabela)
        por_dobra.append({"dobra": int(d), "n": m["n"], "acuracia": m["acuracia"],
                          "f1_macro": m["f1_macro"]})
    return {
        "classificacao": classificacao(predicoes),
        "severidade": metrics.calcular_metricas_severidade(predicoes["severidade_verdadeira"],
                                                           predicoes["severidade_prevista"]),
        "por_dobra": por_dobra,
    }


def comparacao_pareada(aleatorio, blocos) -> dict:
    """Quantas imagens cada protocolo acerta, imagem a imagem."""
    juntas = aleatorio[["id", "classe_verdadeira", "classe_prevista"]].merge(
        blocos[["id", "classe_prevista"]], on="id", suffixes=("_aleatorio", "_blocos"))
    certa_a = (juntas["classe_prevista_aleatorio"] == juntas["classe_verdadeira"]).to_numpy()
    certa_b = (juntas["classe_prevista_blocos"] == juntas["classe_verdadeira"]).to_numpy()
    return {"n": len(juntas), "certas_nos_dois": int((certa_a & certa_b).sum()),
            "so_no_aleatorio": int((certa_a & ~certa_b).sum()),
            "so_em_blocos": int((~certa_a & certa_b).sum()),
            "erradas_nos_dois": int((~certa_a & ~certa_b).sum())}


def faixa_de_d(d_pontos: float) -> str:
    """A faixa da regra de leitura para D (em pontos de F1 macro)."""
    return next(texto for limite, texto in FAIXAS_D if d_pontos <= limite)


def sha256_da_pasta(pasta: Path) -> dict[str, str]:
    """SHA-256 de cada arquivo da pasta, em qualquer profundidade: {caminho POSIX: hash}."""
    resultado = {}
    for arquivo in sorted(p for p in pasta.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with open(arquivo, "rb") as f:
            for pedaco in iter(lambda: f.read(1 << 20), b""):
                h.update(pedaco)
        resultado[arquivo.relative_to(pasta).as_posix()] = h.hexdigest()
    return resultado


# ---------------------------------------------------------------------- execução
def executar(dispositivo_pedido: str = "auto", so_plano: bool = False,
             max_treinos: int | None = None) -> int:
    """Plano, condição de parada, treinos que faltam, controle de cor, métricas e relatório."""
    inicio = time.perf_counter()
    pasta = train.PASTA_RUNS / NOME
    pasta_base = train.PASTA_RUNS / RUN_BASE
    arquivo_base = pasta_base / "config.json"
    if not arquivo_base.is_file():
        raise train.ErroFatal(f"config do run base não encontrado: {train._exibir(arquivo_base)}")
    config_base = json.loads(arquivo_base.read_text(encoding="utf-8"))

    pool = montar_pool()
    planos = planejar(pool)
    conferir_plano(pool, planos)
    tabela = tabela_das_dobras(pool, planos)
    _imprimir_dobras(tabela)
    abaixo = tabela[tabela["menor_fracao_treino"] < LIMIAR_CLASSE]
    if len(abaixo):
        raise CondicaoDeParada(
            "; ".join(f"{r.protocolo} dobra {r.dobra}: {r.classe_menor_fracao} com "
                      f"{_pct(r.menor_fracao_treino)} no treino" for r in abaixo.itertuples()))

    pasta.mkdir(parents=True, exist_ok=True)
    _gravar_plano(plano_por_imagem(pool, planos), pasta / "plano_dobras.csv")
    arquivo_config = pasta / "config.json"
    if not arquivo_config.is_file():  # na retomada, fica o da primeira execução
        train.gravar_json(montar_config(config_base, pool, tabela, sha256_da_pasta(pasta_base)),
                          arquivo_config)
    config = json.loads(arquivo_config.read_text(encoding="utf-8"))
    if so_plano:
        fmt.dizer(f"Plano gravado em {train._exibir(pasta)} (plano_dobras.csv e config.json)")
        return 0

    dispositivo = train.escolher_dispositivo(dispositivo_pedido)
    pendentes = [(p, d) for d in range(K) for p in PROTOCOLOS
                 if not (pasta / f"predicoes_{p}_{d}.csv").is_file()]
    fmt.dizer(f"Dispositivo: {dispositivo} | CUDA: {train.descrever_cuda()} | treinos que "
              f"faltam: {len(pendentes)} de {K * len(PROTOCOLOS)}")
    tempos = []
    for protocolo, d in pendentes[:max_treinos]:
        dobra = planos[protocolo][d]
        fmt.dizer(f"Treino {protocolo}, dobra {d}: {fmt.n(len(dobra['treino']))} imagens de "
                  f"treino, {fmt.n(len(dobra['fora']))} de fora, {EPOCAS} épocas")
        tempos.append(treinar_dobra(protocolo, d, pool, dobra, config_base, dispositivo, pasta))
        faltam = len(pendentes) - len(tempos)
        media = sum(tempos) / len(tempos)
        fmt.dizer(f"Fim de {protocolo}, dobra {d}: {fmt.decimal(tempos[-1] / 60, 1)} min | "
                  f"faltam {faltam} treinos, cerca de {fmt.decimal(faltam * media / 60, 0)} min")
    if len(pendentes) > len(tempos):
        fmt.dizer(f"Parado depois de {len(tempos)} treinos (--max-treinos). Rode de novo para "
                  "continuar.")
        return 0

    fmt.dizer("Controle de cor nas mesmas dobras (lendo as 1.432 imagens)...")
    controle = controle_de_cor(pool, planos, config_base["largura"], config_base["altura"])
    train.gravar_json(controle, pasta / "controle_cor_blocos.json")
    predicoes = juntar_predicoes(pasta)
    resumo = {p: resumir(predicoes[p]) for p in PROTOCOLOS}
    pareada = comparacao_pareada(predicoes["aleatorio"], predicoes["blocos"])
    pureza = atalho.pureza_local(bracol.ler_manifest(), pool["id"])
    grupos = {p: atalho.grupos_de_pureza(
        [pureza[int(i)] for i in predicoes[p]["id"]],
        predicoes[p]["classe_verdadeira"] == predicoes[p]["classe_prevista"],
        predicoes[p]["classe_verdadeira"]) for p in PROTOCOLOS}
    d_pontos = 100 * (resumo["aleatorio"]["classificacao"]["f1_macro"]
                      - resumo["blocos"]["classificacao"]["f1_macro"])
    historicos = {(p, d): pd.read_csv(pasta / f"historico_{p}_{d}.csv")
                  for p in PROTOCOLOS for d in range(K)}
    sha_depois = sha256_da_pasta(pasta_base)
    base_intacto = sha_depois == config["sha256_run_base_antes"]
    train.gravar_json({
        "d_pontos_f1_macro": d_pontos, "faixa": faixa_de_d(d_pontos),
        "protocolos": resumo, "pareada": pareada,
        "pureza_local": {p: g.to_dict(orient="records") for p, g in grupos.items()},
        "run_base_intacto": base_intacto,
    }, pasta / "metricas_blocos.json")
    texto = relatorio(config, tabela, resumo, d_pontos, pareada, grupos, controle, historicos,
                      base_intacto)
    (pasta / "relatorio_blocos.md").write_text(texto, encoding="utf-8", newline="\n")

    m_a, m_b = (resumo[p]["classificacao"] for p in PROTOCOLOS[::-1])
    fmt.dizer(f"Aleatório: acurácia {_pct(m_a['acuracia'])}, F1 macro {_pct(m_a['f1_macro'])} | "
              f"blocos: acurácia {_pct(m_b['acuracia'])}, F1 macro {_pct(m_b['f1_macro'])}")
    fmt.dizer(f"D = {_pontos(d_pontos)} de F1 macro: {faixa_de_d(d_pontos)}")
    fmt.dizer(f"Run base intacto (SHA-256 antes e depois): {'sim' if base_intacto else 'NAO'}")
    fmt.dizer(f"Arquivos em {train._exibir(pasta)} | tempo desta execução: "
              f"{fmt.decimal((time.perf_counter() - inicio) / 60, 1)} min")
    return 0


def montar_config(config_base: dict, pool, tabela, sha_base: dict) -> dict:
    """O config.json do diagnóstico: plano, configuração de treino, decisão e o SHA-256 dos
    arquivos do run base antes dos treinos."""
    commit, alterado = train.estado_git()
    largura, altura = config_base["largura"], config_base["altura"]
    return {
        "nome": NOME, "criado_em": datetime.now().isoformat(timespec="seconds"),
        "comando": "python evaluation/diagnostico_blocos.py",
        "commit": commit, "commit_com_alteracoes": alterado,
        "decisao_do_gestor": {"data": DECISAO_K[0], "texto": DECISAO_K[1]},
        "pool": {"origem": "dados.montar_treino() + dados.montar_val() (o teste não entra)",
                 "n": len(pool),
                 "por_classe": {c: int((pool["classe"] == c).sum()) for c in bracol.CLASSES}},
        "k": K, "bloco": BLOCO, "purga": PURGA, "semente": SEED, "limiar_classe": LIMIAR_CLASSE,
        "protocolos": {
            "blocos": f"bloco i (ids {BLOCO}i+1 a {BLOCO}(i+1)) fora na dobra i mod {K}; "
                      f"treino sem as imagens a até {PURGA} ids de um bloco de fora e sem os "
                      "grupos do conjunto de fora",
            "aleatorio": f"StratifiedGroupKFold(n_splits={K}, shuffle=True, random_state={SEED}) "
                         "pela classe, grupos do manifest; treino sorteado (estratificado, mesma "
                         "semente) até o n do treino de blocos da mesma dobra",
        },
        "treino": {
            "configuracao_do_run": RUN_BASE, "backbone": config_base["backbone"],
            "pretreino": config_base["pretreino"], "pesos": config_base["pesos"],
            "largura": largura, "altura": altura, "batch": config_base["batch"],
            "lr": config_base["lr"], "weight_decay": config_base["weight_decay"],
            "peso_sev": config_base["peso_sev"], "epocas": EPOCAS, "t_max": EPOCAS,
            "parada_antecipada": False, "metricas": "última época",
            "amp": "sim, na CUDA", "workers": train.WORKERS_PADRAO, "checkpoint": False,
            "amostragem": config_base["amostragem"],
            "augmentation": dados.descrever_transformacao(
                dados.transformacao_treino(largura, altura, SEED)),
        },
        "regra_de_leitura": {"d": "F1 macro (aleatório) - F1 macro (blocos), em pontos",
                             "faixas": [[None if math.isinf(limite) else limite, texto]
                                        for limite, texto in FAIXAS_D]},
        "dobras": tabela.to_dict(orient="records"),
        "run_base": RUN_BASE,
        "sha256_run_base_antes": sha_base,
        "versoes": train.versoes(),
    }


def _gravar_plano(plano: pd.DataFrame, caminho: Path) -> None:
    """Grava plano_dobras.csv; numa retomada, ele tem de ser idêntico ao da primeira execução."""
    texto = plano.to_csv(index=False, lineterminator="\n")
    if caminho.is_file() and caminho.read_text(encoding="utf-8") != texto:
        raise train.ErroFatal(f"o plano das dobras mudou desde a primeira execução "
                              f"({train._exibir(caminho)}); os treinos não podem ser misturados")
    caminho.write_text(texto, encoding="utf-8", newline="\n")


def _imprimir_dobras(tabela) -> None:
    for linha in tabela.itertuples():
        treino = " ".join(f"{ABREVIATURA[c]} {getattr(linha, f'treino_{c}')}"
                          for c in bracol.CLASSES)
        fora = " ".join(f"{ABREVIATURA[c]} {getattr(linha, f'fora_{c}')}" for c in bracol.CLASSES)
        fmt.dizer(f"{linha.protocolo:9s} dobra {linha.dobra} | treino {linha.n_treino:4d} | fora "
                  f"{linha.n_fora:3d} | purga {linha.n_purga_distancia + linha.n_purga_grupo:3d} | "
                  f"sorteio {linha.n_descartada_sorteio:3d} | treino: {treino} | fora: {fora} | "
                  f"menor: {linha.classe_menor_fracao} {_pct(linha.menor_fracao_treino)}")


# ----------------------------------------------------------------------- relatório
def relatorio(config, tabela, resumo, d_pontos, pareada, grupos, controle, historicos,
              base_intacto) -> str:
    """relatorio_blocos.md: a regra de leitura antes dos números, o plano, os resultados, o
    controle de cor e o que isto mostra e não mostra."""
    origem = (f"no commit `{config['commit'][:7]}`" + (" (com alterações não commitadas)"
                                                       if config["commit_com_alteracoes"] else "")
              if config["commit"] else "fora de um repositório git")
    t = config["treino"]
    tempos = [h["tempo_s"].sum() for h in historicos.values()]
    memorias = [h["memoria_gpu_gb"].max() for h in historicos.values()
                if h["memoria_gpu_gb"].notna().any()]
    linhas = [
        "# Diagnóstico por blocos de ids", "",
        f"Gerado por `{config['comando']}`. Plano criado em {_data(config['criado_em'])}, "
        f"{origem}. Pool: treino + validação do BRACOL ({fmt.n(config['pool']['n'])} imagens); "
        "o teste não foi lido. Os modelos deste diagnóstico não foram salvos.", "",
        f"**Decisão do gestor ({config['decisao_do_gestor']['data']}):** "
        f"{config['decisao_do_gestor']['texto']}.", "",
        "## Como ler (regra definida antes dos números)", "",
        "- **D = F1 macro (aleatório) - F1 macro (blocos)**, em pontos, sobre as predições de "
        "fora da dobra juntas: as mesmas imagens nos dois protocolos.",
        "- **D até 3 pontos:** pouco atalho de sessão detectado. **De 3 a 8:** atenção. **Acima "
        "de 8:** atalho relevante.",
        "- A queda só é chamada de **clara** se os intervalos de Wilson de 95% da acurácia dos "
        "dois protocolos não se tocarem.", "",
        "## Protocolos e dobras", "",
        f"- **Blocos:** blocos de {BLOCO} ids consecutivos (1-100, 101-200, ...); o bloco i fica "
        f"de fora na dobra i mod {K}. O treino da dobra tira as imagens a até {PURGA} ids de um "
        "bloco de fora (purga por distância) e as que têm o grupo (folha repetida) no conjunto de "
        "fora (purga por grupo).",
        f"- **Aleatório:** `StratifiedGroupKFold` em {K} dobras, estratificado pela classe, com "
        "os grupos do manifest. O treino de cada dobra é sorteado (estratificado, semente "
        f"{SEED}) até o mesmo número de imagens do treino de blocos da mesma dobra.",
        "- Cada imagem do pool fica de fora exatamente uma vez em cada protocolo.", "",
    ]
    linhas += fmt.tabela(
        ["protocolo", "dobra", "treino", "fora", "purga (distância)", "purga (grupo)",
         "descartadas no sorteio", "menor fração de uma classe no treino"],
        [[NOME_PROTOCOLO[r.protocolo], str(r.dobra), fmt.n(r.n_treino), fmt.n(r.n_fora),
          fmt.n(r.n_purga_distancia), fmt.n(r.n_purga_grupo), fmt.n(r.n_descartada_sorteio),
          f"{_pct(r.menor_fracao_treino)} ({r.classe_menor_fracao})"]
         for r in tabela.itertuples()],
    )
    linhas += ["", "Imagens por classe no treino e fora (saudavel / ferrugem / bicho_mineiro / "
               "phoma / cercosporiose):", ""]
    linhas += fmt.tabela(
        ["protocolo", "dobra", "treino", "fora"],
        [[NOME_PROTOCOLO[r.protocolo], str(r.dobra),
          " / ".join(fmt.n(getattr(r, f"treino_{c}")) for c in bracol.CLASSES),
          " / ".join(fmt.n(getattr(r, f"fora_{c}")) for c in bracol.CLASSES)]
         for r in tabela.itertuples()],
    )
    linhas += [
        "", f"Arquivos: `plano_dobras.csv` (o papel de cada imagem em cada dobra) e "
        "`config.json`.", "",
        f"**Treino:** a configuração do run `{t['configuracao_do_run']}` (`{t['backbone']}` "
        f"pré-treinado, {t['largura']}x{t['altura']}, AdamW com lr {_numero(t['lr'])} e weight "
        f"decay {_numero(t['weight_decay'])}, batch {t['batch']}, AMP, RD02, semente {SEED}), "
        f"mas com {t['epocas']} épocas fixas, cosseno de T_max {t['t_max']}, sem parada "
        "antecipada e com as métricas da última época. O F1 macro por época, em "
        "`historico_<protocolo>_<dobra>.csv`, é só informativo. Tempo: "
        f"{fmt.decimal(sum(tempos) / 60, 0)} min nos {len(tempos)} treinos"
        + (f"; pico de {fmt.decimal(max(memorias), 1)} GB de memória da GPU." if memorias
           else "."), "",
    ]
    linhas += _secao_resultados(resumo, d_pontos, pareada, grupos, historicos)
    linhas += _secao_controle(controle)
    linhas += _secao_leitura(resumo, d_pontos, pareada, grupos, controle, base_intacto)
    return "\n".join(linhas) + "\n"


def _secao_resultados(resumo, d_pontos, pareada, grupos, historicos) -> list[str]:
    m = {p: resumo[p]["classificacao"] for p in PROTOCOLOS}
    s = {p: resumo[p]["severidade"] for p in PROTOCOLOS}
    linhas = [f"## Resultado (fora da dobra, {fmt.n(m['blocos']['n'])} imagens por protocolo)",
              ""]
    linhas += fmt.tabela(
        ["protocolo", "acurácia", "IC 95% (Wilson)", "F1 macro", "MAE sev (informativo)",
         "kappa sev (informativo)"],
        [[NOME_PROTOCOLO[p], _pct(m[p]["acuracia"]), _intervalo(*m[p]["acuracia_ic95"]),
          _pct(m[p]["f1_macro"]), fmt.decimal(s[p]["mae"], 3), _decimal(s[p]["kappa_quadratico"])]
         for p in ("aleatorio", "blocos")],
    )
    linhas += ["", f"**D = {_pontos(d_pontos)} de F1 macro: {faixa_de_d(d_pontos)}.**", "",
               "Recall por classe:", ""]
    linhas += fmt.tabela(
        ["protocolo", *bracol.CLASSES],
        [[NOME_PROTOCOLO[p], *(_pct(r) for r in m[p]["recall"])] for p in ("aleatorio", "blocos")]
        + [["diferença (pontos)", *(fmt.decimal(100 * (a - b), 1) for a, b in
                                    zip(m["aleatorio"]["recall"], m["blocos"]["recall"]))]],
    )
    for p in ("aleatorio", "blocos"):
        linhas += ["", f"Matriz de confusão, {NOME_PROTOCOLO[p]} (linhas: verdadeira; colunas: "
                   "prevista):", ""]
        linhas += fmt.tabela(["classe verdadeira", *bracol.CLASSES],
                             [[c, *map(fmt.n, linha)] for c, linha in
                              zip(bracol.CLASSES, m[p]["matriz_confusao"])])
    linhas += ["", "### Por dobra", "",
               "A última época vale; o maior F1 macro ao longo das épocas é só informativo "
               "(escolher a época pelo conjunto de fora seria olhar a resposta).", ""]
    linhas += fmt.tabela(
        ["protocolo", "dobra", "fora", "acurácia", "F1 macro", "maior F1 macro nas épocas"],
        [[NOME_PROTOCOLO[p], str(f["dobra"]), fmt.n(f["n"]), _pct(f["acuracia"]),
          _pct(f["f1_macro"]), _maior_f1(historicos[(p, f["dobra"])])]
         for p in ("aleatorio", "blocos") for f in resumo[p]["por_dobra"]],
    )
    linhas += ["", "### Comparação pareada (imagem a imagem)", ""]
    linhas += fmt.tabela(
        ["certas nos dois", "certas só no aleatório", "certas só em blocos", "erradas nos dois"],
        [[fmt.n(pareada["certas_nos_dois"]), fmt.n(pareada["so_no_aleatorio"]),
          fmt.n(pareada["so_em_blocos"]), fmt.n(pareada["erradas_nos_dois"])]],
    )
    rotulos = dict(atalho.GRUPOS_PUREZA)
    linhas += ["", "### Por pureza local (a dos passos 1c, pela classe do manifest)", ""]
    linhas += fmt.tabela(
        ["pureza local", "imagens", "acurácia aleatório", "IC 95%", "acurácia blocos", "IC 95%"],
        [[rotulos[a["grupo"]], fmt.n(a["n"]), _pct(a["acuracia"]),
          _intervalo(a["ic95_baixo"], a["ic95_alto"]), _pct(b["acuracia"]),
          _intervalo(b["ic95_baixo"], b["ic95_alto"])]
         for (_, a), (_, b) in zip(grupos["aleatorio"].iterrows(), grupos["blocos"].iterrows())],
    )
    return [*linhas, "", "Arquivos: `predicoes_<protocolo>_<dobra>.csv` e `metricas_blocos.json`.",
            ""]


def _secao_controle(controle) -> list[str]:
    linhas = ["## Controle de cor nos dois protocolos", "",
              "A regressão logística só com a cor do passo 1c, ajustada no treino de cada dobra e "
              "medida no conjunto de fora, com as predições juntas.", ""]
    linhas += fmt.tabela(
        ["variante", "acurácia aleatório", "F1 macro aleatório", "acurácia blocos",
         "F1 macro blocos", "diferença de F1 (pontos)"],
        [[controle["aleatorio"][chave]["rotulo"],
          f"{_pct(controle['aleatorio'][chave]['acuracia'])} "
          f"({_intervalo(*controle['aleatorio'][chave]['acuracia_ic95'])})",
          _pct(controle["aleatorio"][chave]["f1_macro"]),
          f"{_pct(controle['blocos'][chave]['acuracia'])} "
          f"({_intervalo(*controle['blocos'][chave]['acuracia_ic95'])})",
          _pct(controle["blocos"][chave]["f1_macro"]),
          fmt.decimal(100 * (controle["aleatorio"][chave]["f1_macro"]
                             - controle["blocos"][chave]["f1_macro"]), 1)]
         for chave in VARIANTES_COR],
    )
    return [*linhas, "", "Arquivo: `controle_cor_blocos.json`.", ""]


def _secao_leitura(resumo, d_pontos, pareada, grupos, controle, base_intacto) -> list[str]:
    m_a, m_b = resumo["aleatorio"]["classificacao"], resumo["blocos"]["classificacao"]
    fatos = [f"D = {_pontos(d_pontos)} de F1 macro ({_pct(m_a['f1_macro'])} no aleatório e "
             f"{_pct(m_b['f1_macro'])} em blocos): pela regra, **{faixa_de_d(d_pontos)}**."]
    fatos.append(
        f"Acurácia: {_pct(m_a['acuracia'])} ({_intervalo(*m_a['acuracia_ic95'])}) no aleatório "
        f"e {_pct(m_b['acuracia'])} ({_intervalo(*m_b['acuracia_ic95'])}) em blocos. "
        + ("Os intervalos se tocam: a diferença não é clara com estes n."
           if _se_tocam(m_a["acuracia_ic95"], m_b["acuracia_ic95"]) else
           "**Os intervalos não se tocam: a queda é clara.**"))
    f1_por_dobra = {p: [f["f1_macro"] for f in resumo[p]["por_dobra"]] for p in PROTOCOLOS}
    fatos.append(f"Por dobra, o F1 macro vai de {_pct(min(f1_por_dobra['aleatorio']))} a "
                 f"{_pct(max(f1_por_dobra['aleatorio']))} no aleatório e de "
                 f"{_pct(min(f1_por_dobra['blocos']))} a {_pct(max(f1_por_dobra['blocos']))} "
                 "em blocos.")
    diferencas = {c: 100 * (a - b) for c, a, b in
                  zip(bracol.CLASSES, m_a["recall"], m_b["recall"])}
    maior = max(diferencas, key=diferencas.get)
    fatos.append(f"A maior queda de recall do aleatório para blocos é em {maior} "
                 f"({fmt.decimal(diferencas[maior], 1)} pontos).")
    fatos.append(f"Imagem a imagem: {fmt.n(pareada['so_no_aleatorio'])} certas só no aleatório "
                 f"e {fmt.n(pareada['so_em_blocos'])} certas só em blocos, de "
                 f"{fmt.n(pareada['n'])}.")
    rotulos = dict(atalho.GRUPOS_PUREZA)
    for (_, a), (_, b) in zip(grupos["aleatorio"].iterrows(), grupos["blocos"].iterrows()):
        separados = not _se_tocam((a["ic95_baixo"], a["ic95_alto"]),
                                  (b["ic95_baixo"], b["ic95_alto"]))
        texto = (f"Pureza local {rotulos[a['grupo']]} ({fmt.n(a['n'])} imagens): "
                 f"{_pct(a['acuracia'])} no aleatório e {_pct(b['acuracia'])} em blocos; "
                 + ("**os intervalos não se tocam.**" if separados else "os intervalos se tocam."))
        fatos.append(texto)
    for chave in VARIANTES_COR:
        a, b = controle["aleatorio"][chave], controle["blocos"][chave]
        separados = not _se_tocam(a["acuracia_ic95"], b["acuracia_ic95"])
        fatos.append(f"Controle de cor {a['rotulo']}: {_pct(a['acuracia'])} no aleatório e "
                     f"{_pct(b['acuracia'])} em blocos (F1 macro {_pct(a['f1_macro'])} e "
                     f"{_pct(b['f1_macro'])}, diferença de "
                     f"{_pontos(100 * (a['f1_macro'] - b['f1_macro']))}); "
                     + ("**os intervalos de acurácia não se tocam: a queda é clara.**"
                        if separados else "os intervalos de acurácia se tocam."))
    fatos.append(f"Run base `{RUN_BASE}` intacto (SHA-256 de todos os arquivos antes e depois): "
                 + ("sim." if base_intacto else "**NÃO**."))
    limites = [
        "Um D pequeno não prova que não há atalho: só mede o quanto o desempenho depende de ver "
        "imagens vizinhas por id no treino.",
        "As sessões de foto não são conhecidas: os blocos de ids são um proxy. Uma sessão pode "
        "atravessar blocos, e um bloco pode juntar sessões.",
        f"A purga de {PURGA} ids é arbitrária.",
        f"K={K}, com uma semente só.",
        "A GPU não é determinística: repetir os treinos muda os números em cerca de 1 ponto.",
        "Os protocolos também diferem na mistura de classes do treino (tabela das dobras): em "
        "blocos, uma classe concentrada num bloco de fora fica com menos imagens no treino "
        "daquela dobra.",
    ]
    return [
        "## O que isto mostra e o que NÃO mostra", "",
        "**Mostra (fatos medidos):**", "",
        *(f"- {f}" for f in fatos), "",
        "**Não mostra:**", "",
        *(f"- {item}" for item in limites), "",
    ]


def _se_tocam(a, b) -> bool:
    """Se dois intervalos (baixo, alto) se tocam ou se sobrepõem."""
    return not (a[1] < b[0] or b[1] < a[0])


def _pontos(valor: float) -> str:
    """Pontos percentuais com a concordância: '1,0 ponto', '-0,4 ponto', '7,6 pontos'."""
    return f"{fmt.decimal(valor, 1)} {'ponto' if abs(round(valor, 1)) < 2 else 'pontos'}"


def _maior_f1(historico) -> str:
    linha = historico.loc[historico["f1_macro_fora"].idxmax()]
    return f"{_pct(linha['f1_macro_fora'])} (época {int(linha['epoca'])})"


def _data(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")


def _numero(valor: float) -> str:
    return f"{valor:g}".replace(".", ",")


def _decimal(valor) -> str:
    return "-" if valor is None else fmt.decimal(valor, 3)


def _intervalo(baixo, alto) -> str:
    return f"{_pct(baixo)} a {_pct(alto)}"


def _pct(fracao) -> str:
    return "-" if fracao is None or pd.isna(fracao) else fmt.pct(fracao, 1)


# ---------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Diagnostico por blocos de ids (validacao cruzada so com treino e val do "
                    "BRACOL). Detalhes no docstring de evaluation/diagnostico_blocos.py."
    )
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"],
                    help="padrao: auto (cuda, se houver)")
    ap.add_argument("--so-plano", action="store_true",
                    help="so calcula, confere e grava o plano das dobras (nao treina)")
    ap.add_argument("--max-treinos", type=int, default=None,
                    help="para depois de N treinos (para rodar por partes; e retomavel)")
    args = ap.parse_args(argv)
    if args.max_treinos is not None and args.max_treinos < 1:
        ap.error("--max-treinos precisa ser >= 1")
    try:
        return executar(args.dispositivo, args.so_plano, args.max_treinos)
    except CondicaoDeParada as e:
        fmt.dizer(f"PARADA: classe com menos de {_pct(LIMIAR_CLASSE)} do pool no treino de uma "
                  f"dobra: {e}. Nada foi treinado.")
        return 1
    except (train.ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
