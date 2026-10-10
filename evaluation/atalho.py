"""
Checagem de atalho do classificador (Frente 9, passo 1c).

Mede se o modelo de um run pode ter aprendido a sessão de fotos (blocos de ids da mesma classe,
tom do fundo) em vez da lesão:
- A. Acurácia por posição na sequência de ids: faixas de 200 ids e pureza local (a classe dos
  vizinhos por id, entre as elegíveis de todos os splits, lida só do manifest).
- B. Controle de fundo: uma regressão logística só com estatísticas de cor (imagem inteira,
  borda de 15% e borda de 8%), ajustada no treino e medida na validação.
- C. Grad-CAM da cabeça de classe numa amostra da validação (os erros e 4 acertos por classe),
  com pranchas e a fração da energia do mapa na borda de 15%.

Usa só treino e validação do BRACOL: nenhuma imagem de teste é lida, e o teste não é avaliado.
Não retreina e não altera o run; grava tudo em model/runs/<run>/atalho/.

Uso (de qualquer pasta, com o venv ativo):
    python evaluation/atalho.py --run base_resnet50_bracol
    python evaluation/atalho.py --run NOME --dispositivo cuda    (padrão: cpu, reprodutível)

O código de saída é 0 sem erros e 1 com erros.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model"):  # bracol, eda_bracol e formatacao; dados e train
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import bracol
import dados
import formatacao as fmt
import gradcam
import metrics
import train

PASTA_SAIDA = "atalho"  # dentro da pasta do run
TAMANHO_FAIXA = 200
VIZINHOS = 5  # de cada lado
# Grupos de pureza local: (chave no csv, rótulo no relatório).
GRUPOS_PUREZA = (
    ("1.0", "1,0"),
    ("0.6_a_1.0", "de 0,6 a menos de 1,0"),
    ("abaixo_de_0.6", "abaixo de 0,6"),
)
BORDA_CAM = 0.15  # faixa de borda do proxy do Grad-CAM (fração de cada lado)
# Controle de fundo: (chave, rótulo, borda); borda None é a imagem inteira.
VARIANTES_FUNDO = (
    ("imagem_inteira", "(a) imagem inteira", None),
    ("borda_15", "(b) borda de 15%", 0.15),
    ("borda_8", "(c) borda de 8%", 0.08),
)
LIMIAR_DESTAQUE = 0.5  # acurácia do controle de fundo que ganha destaque no relatório
CANAIS = ("R", "G", "B", "H", "S", "V")
NOMES_CARACTERISTICAS = [f"{estatistica}_{canal}" for canal in CANAIS
                         for estatistica in ("media", "desvio")]
ACERTOS_POR_CLASSE = 4
LINHAS_POR_PRANCHA = 4
SEED = bracol.SEED_PADRAO


# --------------------------------------------------------------- A. posição dos ids
def faixas_de_id(predicoes, maior_id: int, tamanho: int = TAMANHO_FAIXA) -> pd.DataFrame:
    """Acurácia da validação por faixa de ids (1-200, 201-400, ...), com o intervalo de Wilson
    de 95% e as imagens de cada classe da faixa.

    predicoes: colunas id, classe_verdadeira e classe_prevista. maior_id fecha a última faixa.
    """
    acerto = predicoes["classe_verdadeira"] == predicoes["classe_prevista"]
    linhas = []
    for inicio in range(1, maior_id + 1, tamanho):
        fim = min(inicio + tamanho - 1, maior_id)
        na_faixa = predicoes["id"].between(inicio, fim)
        n, acertos = int(na_faixa.sum()), int((na_faixa & acerto).sum())
        baixo, alto = metrics.intervalo_wilson(acertos / n, n) if n else (np.nan, np.nan)
        linhas.append({
            "faixa": f"{inicio}-{fim}", "n": n, "acertos": acertos,
            "acuracia": acertos / n if n else np.nan, "ic95_baixo": baixo, "ic95_alto": alto,
            **{c: int((na_faixa & (predicoes["classe_verdadeira"] == c)).sum())
               for c in bracol.CLASSES},
        })
    return pd.DataFrame(linhas)


def faixa_do_id(i: int, maior_id: int, tamanho: int = TAMANHO_FAIXA) -> str:
    """A faixa de ids de um id, no formato de faixas_de_id ("1601-1747")."""
    inicio = (int(i) - 1) // tamanho * tamanho + 1
    return f"{inicio}-{min(inicio + tamanho - 1, maior_id)}"


def pureza_local(elegiveis, ids, vizinhos: int = VIZINHOS) -> pd.Series:
    """Fração dos vizinhos por id que têm a mesma classe: os `vizinhos` anteriores e os
    `vizinhos` seguintes na sequência das elegíveis (de qualquer split, sem a própria imagem).
    Nas pontas da sequência, conta só os vizinhos que existem.

    elegiveis: colunas id e classe de todas as linhas elegíveis do manifest.
    ids: as imagens medidas. Devolve uma Series indexada pelo id.
    """
    ordem = elegiveis.sort_values("id")
    posicao = {int(i): k for k, i in enumerate(ordem["id"])}
    classes = ordem["classe"].to_numpy()
    faltam = sorted({int(i) for i in ids} - set(posicao))
    if faltam:
        raise ValueError(f"ids fora das elegíveis: {faltam[:10]}")
    pureza = {}
    for i in ids:
        k = posicao[int(i)]
        antes, depois = classes[max(0, k - vizinhos):k], classes[k + 1:k + 1 + vizinhos]
        vizinhas = np.concatenate([antes, depois])
        pureza[int(i)] = float(np.mean(vizinhas == classes[k])) if len(vizinhas) else np.nan
    return pd.Series(pureza, name="pureza_local")


def grupo_de_pureza(pureza: float) -> str:
    """Chave do grupo de pureza: 1.0, 0.6_a_1.0 (de 0,6 a menos de 1,0) ou abaixo_de_0.6."""
    if pureza == 1:
        return GRUPOS_PUREZA[0][0]
    return GRUPOS_PUREZA[1][0] if pureza >= 0.6 else GRUPOS_PUREZA[2][0]


def grupos_de_pureza(purezas, acertos, classes=None) -> pd.DataFrame:
    """n, acertos, acurácia e intervalo de Wilson de 95% por grupo de pureza local e, se
    `classes` vier, as imagens de cada classe no grupo (a mistura de classes muda entre eles).

    purezas, acertos e classes: um valor por imagem, na mesma ordem.
    """
    grupos = np.array([grupo_de_pureza(p) for p in purezas])
    acertos = np.asarray(acertos, dtype=bool)
    linhas = []
    for chave, _ in GRUPOS_PUREZA:
        no_grupo = grupos == chave
        n, a = int(no_grupo.sum()), int(acertos[no_grupo].sum())
        baixo, alto = metrics.intervalo_wilson(a / n, n) if n else (np.nan, np.nan)
        linha = {"grupo": chave, "n": n, "acertos": a, "acuracia": a / n if n else np.nan,
                 "ic95_baixo": baixo, "ic95_alto": alto}
        if classes is not None:
            do_grupo = np.asarray(classes)[no_grupo]
            linha.update({c: int(np.sum(do_grupo == c)) for c in bracol.CLASSES})
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_de_erros(predicoes, pureza, maior_id: int) -> pd.DataFrame:
    """Os erros da validação, com as probabilidades da prevista e da verdadeira, a faixa de ids
    e a pureza local."""
    erros = predicoes[predicoes["classe_verdadeira"] != predicoes["classe_prevista"]]
    erros = erros.sort_values("id")
    return pd.DataFrame({
        "id": erros["id"].to_numpy(),
        "classe_verdadeira": erros["classe_verdadeira"].to_numpy(),
        "classe_prevista": erros["classe_prevista"].to_numpy(),
        "prob_prevista": [linha[f"prob_{linha['classe_prevista']}"]
                          for _, linha in erros.iterrows()],
        "prob_verdadeira": [linha[f"prob_{linha['classe_verdadeira']}"]
                            for _, linha in erros.iterrows()],
        "faixa_id": [faixa_do_id(i, maior_id) for i in erros["id"]],
        "pureza_local": [pureza[int(i)] for i in erros["id"]],
    })


# ------------------------------------------------------------- B. controle de fundo
def mascara_borda(altura: int, largura: int, borda: float = BORDA_CAM) -> np.ndarray:
    """Máscara (altura x largura) da faixa de borda: os pixels a até `borda` da largura das
    laterais ou a até `borda` da altura do topo e da base, arredondado para pixels inteiros."""
    bx, by = round(borda * largura), round(borda * altura)
    mascara = np.ones((altura, largura), dtype=bool)
    mascara[by:altura - by, bx:largura - bx] = False
    return mascara


def fracao_na_borda(mapa, mascara) -> float:
    """Fração da energia (a soma dos valores) do mapa que cai na máscara de borda."""
    total = float(np.sum(mapa))
    return float(np.sum(mapa[mascara])) / total if total > 0 else float("nan")


def caracteristicas_de_cor(rgb, mascara=None) -> np.ndarray:
    """Média e desvio de R, G, B e de H, S, V dos pixels de uma imagem RGB uint8, ou só dos da
    máscara. O HSV é o do OpenCV (H de 0 a 179; S e V de 0 a 255). Devolve 12 números, na ordem
    de NOMES_CARACTERISTICAS."""
    canais = np.concatenate([rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)], axis=2)
    pixels = canais.reshape(-1, 6) if mascara is None else canais[mascara]
    pixels = pixels.astype(np.float64)
    return np.column_stack([pixels.mean(axis=0), pixels.std(axis=0)]).ravel()


def extrair_cor(df, largura: int, altura: int) -> dict[str, np.ndarray]:
    """Características de cor de cada imagem de df, já redimensionada como o modelo a vê (antes
    da normalização), para cada variante de VARIANTES_FUNDO: {chave: matriz n x 12}. Lê com o
    FolhasDataset de model/dados.py."""
    conjunto = dados.FolhasDataset(df, dados.transformacao_sem_normalizar(largura, altura))
    mascaras = {chave: None if borda is None else mascara_borda(altura, largura, borda)
                for chave, _, borda in VARIANTES_FUNDO}
    linhas = {chave: [] for chave in mascaras}
    for i in range(len(conjunto)):
        rgb = conjunto[i][0]
        for chave, mascara in mascaras.items():
            linhas[chave].append(caracteristicas_de_cor(rgb, mascara))
    return {chave: np.array(valores) for chave, valores in linhas.items()}


def controle_de_fundo(x_treino, y_treino, x_val, y_val, seed: int = SEED) -> dict:
    """O modelo "burro": regressão logística só com as estatísticas de cor, padronizadas com a
    média e o desvio do treino. Ajusta no treino; mede na validação com
    metrics.calcular_metricas (y são índices em CLASSES)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    escala = StandardScaler().fit(x_treino)
    modelo = LogisticRegression(max_iter=5000, random_state=seed)
    modelo.fit(escala.transform(x_treino), y_treino)
    previsto_treino = modelo.predict(escala.transform(x_treino))
    return {
        "acuracia_treino": float(np.mean(previsto_treino == np.asarray(y_treino))),
        "val": metrics.calcular_metricas(y_val, modelo.predict(escala.transform(x_val))),
    }


# ----------------------------------------------------------------------- C. Grad-CAM
def amostra_gradcam(predicoes, por_classe: int = ACERTOS_POR_CLASSE,
                    seed: int = SEED) -> pd.DataFrame:
    """Os erros da validação e `por_classe` acertos de cada classe, sorteados com a semente
    (cada classe com o seu sorteio, na ordem de CLASSES). Acrescenta a coluna acerto."""
    acerto = predicoes["classe_verdadeira"] == predicoes["classe_prevista"]
    partes = [predicoes[~acerto].sort_values("id").assign(acerto=False)]
    for classe in bracol.CLASSES:
        da_classe = predicoes[acerto & (predicoes["classe_verdadeira"] == classe)]
        sorteio = da_classe.sample(n=min(por_classe, len(da_classe)), random_state=seed)
        partes.append(sorteio.sort_values("id").assign(acerto=True))
    return pd.concat(partes, ignore_index=True)


def mapas_da_amostra(modelo, amostra, val, largura: int, altura: int, dispositivo) -> list[dict]:
    """Para cada imagem da amostra: a imagem como o modelo a vê (RGB, sem normalizar), as
    probabilidades recalculadas em float32 e os mapas Grad-CAM da classe prevista pelo run e,
    nos erros, também da verdadeira, com a fração da energia na borda de 15%."""
    caminhos = val.set_index("id")["caminho"]
    para_o_modelo = dados.transformacao_avaliacao(largura, altura)
    para_ver = dados.transformacao_sem_normalizar(largura, altura)
    mascara = mascara_borda(altura, largura, BORDA_CAM)
    resultados = []
    for _, linha in amostra.iterrows():
        original = dados.ler_imagem(dados.RAIZ_REPO / caminhos[linha["id"]])
        tensor = para_o_modelo(image=original)["image"].unsqueeze(0).to(dispositivo)
        with torch.no_grad():
            logits = modelo(tensor)[0].float()
        probabilidades = torch.softmax(logits, dim=1)[0].cpu().numpy()
        papeis = [("prevista", linha["classe_prevista"])]
        if not linha["acerto"]:
            papeis.append(("verdadeira", linha["classe_verdadeira"]))
        mapas = []
        for papel, classe in papeis:
            mapa = gradcam.gerar_gradcam(modelo, tensor, alvo=bracol.CLASSES.index(classe))
            mapas.append({"papel": papel, "classe": classe, "mapa": mapa,
                          "fracao_borda": fracao_na_borda(mapa, mascara)})
        resultados.append({"linha": linha, "rgb": para_ver(image=original)["image"],
                           "probabilidades": probabilidades, "mapas": mapas})
    return resultados


def desenhar_prancha(itens: list[dict], titulo: str, colunas: int):
    """Prancha de Grad-CAM, no estilo das figuras da EDA, com o crédito do BRACOL.

    itens: dicts com rgb (a imagem), titulo (da linha) e paineis (dicts com mapa e titulo).
    Cada linha: a imagem e, ao lado, cada mapa sobreposto (rampa azul de um tom, mais opaca
    onde o mapa é maior), com o limite da borda de 15% tracejado.
    """
    import eda_bracol as eda  # estilo comum das figuras do projeto
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    altura_fig = 2.55 * len(itens) + 1.15
    with eda._estilo():
        fig = Figure(figsize=(4.3 * colunas + 0.2, altura_fig))
        fig.suptitle(titulo, x=0.01, y=1 - 0.1 / altura_fig, ha="left", va="top",
                     color=eda.TINTA)
        fig.text(0.01, 1 - 0.42 / altura_fig,
                 "Azul: onde o mapa Grad-CAM da cabeça de classe é maior (0 a 1, normalizado "
                 "por imagem).\nTracejado: limite da borda de 15%; \"borda\" é a fração da "
                 "energia do mapa fora dele.", fontsize=8, color=eda.TINTA_2, va="top",
                 linespacing=1.4)
        grade = fig.add_gridspec(len(itens), colunas, left=0.01, right=0.99,
                                 top=1 - 0.95 / altura_fig, bottom=0.3 / altura_fig,
                                 wspace=0.03, hspace=0.3)
        for r, item in enumerate(itens):
            ax = fig.add_subplot(grade[r, 0])
            ax.imshow(item["rgb"])
            ax.set_axis_off()
            ax.set_title(item["titulo"], loc="left", fontsize=9, color=eda.TINTA, pad=3)
            altura, largura = item["rgb"].shape[:2]
            bx, by = round(BORDA_CAM * largura), round(BORDA_CAM * altura)
            for c, painel in enumerate(item["paineis"], start=1):
                ax = fig.add_subplot(grade[r, c])
                ax.imshow(item["rgb"])
                ax.imshow(painel["mapa"], cmap=eda.RAMPA_AZUL, vmin=0, vmax=1,
                          alpha=0.75 * painel["mapa"])
                ax.add_patch(Rectangle((bx - 0.5, by - 0.5), largura - 2 * bx, altura - 2 * by,
                                       fill=False, linestyle="--", linewidth=1.0,
                                       edgecolor=eda.TINTA))
                ax.set_axis_off()
                ax.set_title(painel["titulo"], loc="left", fontsize=9, color=eda.TINTA_2, pad=3)
        fig.text(0.01, 0.08 / altura_fig, eda.CREDITO, fontsize=7.5, color=eda.TINTA_2)
    return fig


def gravar_pranchas(resultados, pasta: Path) -> dict[int, str]:
    """Pranchas JPEG dos erros e dos acertos (LINHAS_POR_PRANCHA por prancha). Devolve
    {id: nome da prancha}."""
    import eda_bracol as eda

    prancha_do_id = {}
    for erro in (True, False):
        grupo = [r for r in resultados if bool(r["linha"]["acerto"]) != erro]
        partes = [grupo[k:k + LINHAS_POR_PRANCHA] for k in range(0, len(grupo), LINHAS_POR_PRANCHA)]
        for numero, parte in enumerate(partes, start=1):
            nome = f"gradcam_{'erros' if erro else 'acertos'}_{numero}.jpg"
            itens = [_item_da_prancha(r) for r in parte]
            titulo = (f"Grad-CAM, {'erros' if erro else 'acertos'} da validação "
                      f"({numero} de {len(partes)})")
            eda._salvar(desenhar_prancha(itens, titulo, colunas=3 if erro else 2), pasta / nome)
            prancha_do_id.update({int(r["linha"]["id"]): nome for r in parte})
    return prancha_do_id


def _item_da_prancha(resultado: dict) -> dict:
    linha = resultado["linha"]
    prob = linha[f"prob_{linha['classe_prevista']}"]
    paineis = []
    for mapa in resultado["mapas"]:
        if linha["acerto"]:
            nome = f"CAM ({mapa['classe']})"
        else:
            nome = f"CAM da {mapa['papel']} ({mapa['classe']})"
        paineis.append({"mapa": mapa["mapa"],
                        "titulo": f"{nome} | borda: {_pct(mapa['fracao_borda'])}"})
    return {"rgb": resultado["rgb"], "paineis": paineis,
            "titulo": f"id {linha['id']}: {linha['classe_verdadeira']} -> "
                      f"{linha['classe_prevista']} ({fmt.decimal(prob, 2)})"}


# ---------------------------------------------------------------------- execução
def checar(run: str, dispositivo_pedido: str = "cpu") -> int:
    """A checagem inteira de um run; grava em PASTA_RUNS/<run>/atalho/."""
    inicio = time.perf_counter()
    torch.manual_seed(SEED)
    pasta_run = train.PASTA_RUNS / run
    arquivo_predicoes = pasta_run / "predicoes_val.csv"
    if not arquivo_predicoes.is_file():
        raise train.ErroFatal(f"predições da validação não encontradas: "
                              f"{train._exibir(arquivo_predicoes)}")
    predicoes = pd.read_csv(arquivo_predicoes)
    metricas_run = json.loads((pasta_run / "metricas_val.json").read_text(encoding="utf-8"))
    elegiveis = bracol.ler_manifest()  # id e classe das 1.685 elegíveis; nenhuma imagem
    treino, val = dados.montar_treino(), dados.montar_val()
    verdadeiras = val.set_index("id")["classe"].sort_index()
    if sorted(predicoes["id"]) != list(verdadeiras.index) or not (
            predicoes.set_index("id")["classe_verdadeira"].sort_index() == verdadeiras).all():
        raise train.ErroFatal("as predições do run não são da validação atual do manifest")
    dispositivo = train.escolher_dispositivo(dispositivo_pedido)
    modelo, checkpoint = train.carregar_modelo(pasta_run, dispositivo)
    largura = checkpoint["tamanho_entrada"]["largura"]
    altura = checkpoint["tamanho_entrada"]["altura"]
    pasta = pasta_run / PASTA_SAIDA
    pasta.mkdir(exist_ok=True)
    for antiga in pasta.glob("gradcam_*.jpg"):  # pranchas de uma execução anterior
        antiga.unlink()
    fmt.dizer(f"Run: {train._exibir(pasta_run)} (época {checkpoint['epoca']}) | saída: "
              f"{train._exibir(pasta)} | dispositivo: {dispositivo}")

    # A. posição na sequência de ids
    maior_id = int(elegiveis["id"].max())
    acerto = (predicoes["classe_verdadeira"] == predicoes["classe_prevista"]).to_numpy()
    faixas = faixas_de_id(predicoes, maior_id)
    pureza = pureza_local(elegiveis, predicoes["id"])
    grupos = grupos_de_pureza([pureza[int(i)] for i in predicoes["id"]], acerto,
                              predicoes["classe_verdadeira"])
    erros = tabela_de_erros(predicoes, pureza, maior_id)
    _gravar_csv(faixas, pasta / "faixas_id.csv")
    _gravar_csv(grupos, pasta / "pureza_local.csv")
    _gravar_csv(erros, pasta / "erros_val.csv")
    fmt.dizer(f"A. Faixas: acurácia de {_pct(faixas['acuracia'].min())} a "
              f"{_pct(faixas['acuracia'].max())} | pureza 1,0: {_resumo_grupo(grupos, 0)} | "
              f"abaixo de 0,6: {_resumo_grupo(grupos, 2)}")

    # B. controle de fundo
    fmt.dizer(f"B. Lendo {fmt.n(len(treino))} imagens de treino e {fmt.n(len(val))} de "
              "validação para o controle de fundo...")
    x_treino, x_val = extrair_cor(treino, largura, altura), extrair_cor(val, largura, altura)
    y_treino = treino["classe"].map(dados.rotulo).to_numpy()
    y_val = val["classe"].map(dados.rotulo).to_numpy()
    controle = montar_controle(x_treino, y_treino, x_val, y_val, val, metricas_run, largura,
                               altura)
    train.gravar_json(controle, pasta / "controle_fundo.json")
    majoritaria = controle["referencias"]["majoritaria"]["acuracia"]
    fmt.dizer("B. Controle de fundo (val): " + " | ".join(
        f"{v['rotulo']} {_pct(v['val']['acuracia'])}" for v in controle["variantes"].values())
        + f" | acaso {_pct(controle['referencias']['acaso'])}, majoritária {_pct(majoritaria)}")

    # C. Grad-CAM
    amostra = amostra_gradcam(predicoes)
    fmt.dizer(f"C. Grad-CAM de {len(amostra)} imagens ({int((~amostra['acerto']).sum())} erros)...")
    resultados = mapas_da_amostra(modelo, amostra, val, largura, altura, dispositivo)
    prancha_do_id = gravar_pranchas(resultados, pasta)
    tabela_amostra, tabela_borda = tabelas_gradcam(resultados, prancha_do_id)
    _gravar_csv(tabela_amostra, pasta / "gradcam_amostra.csv")
    _gravar_csv(tabela_borda, pasta / "gradcam_borda.csv")
    da_prevista = tabela_borda[tabela_borda["mapa"] == "prevista"]
    area_borda = float(mascara_borda(altura, largura, BORDA_CAM).mean())
    fmt.dizer(f"C. Energia do CAM na borda de 15%: média {_pct(da_prevista['fracao_borda'].mean())}"
              f" (uma CAM uniforme daria {_pct(area_borda)})")

    commit = train.estado_git()
    texto = relatorio(run, checkpoint, commit, dispositivo_pedido, faixas, grupos, erros,
                      controle, tabela_amostra, tabela_borda, area_borda, prancha_do_id)
    (pasta / "relatorio_atalho.md").write_text(texto, encoding="utf-8", newline="\n")
    arquivos = ", ".join(sorted(p.name for p in pasta.iterdir()))
    fmt.dizer(f"Arquivos em {train._exibir(pasta)}: {arquivos}")
    fmt.dizer(f"Tempo: {fmt.decimal(time.perf_counter() - inicio, 0)} s")
    return 0


def montar_controle(x_treino, y_treino, x_val, y_val, val, metricas_run, largura: int,
                    altura: int) -> dict:
    """O conteúdo de controle_fundo.json: as três variantes e as referências."""
    contagem = val["classe"].value_counts()
    majoritaria = contagem.index[0]
    variantes = {}
    for chave, rotulo, borda in VARIANTES_FUNDO:
        resultado = controle_de_fundo(x_treino[chave], y_treino, x_val[chave], y_val)
        area = 1.0 if borda is None else float(mascara_borda(altura, largura, borda).mean())
        variantes[chave] = {"rotulo": rotulo, "borda": borda, "fracao_da_area": area,
                            **resultado,
                            "acima_do_limiar": resultado["val"]["acuracia"] > LIMIAR_DESTAQUE}
    classificacao = metricas_run["classificacao"]
    return {
        "descricao": "Regressão logística (scikit-learn, padronização com média e desvio do "
                     "treino) só com média e desvio de R, G, B, H, S e V, na imagem já em "
                     f"{largura}x{altura}, antes da normalização. Ajuste no treino; medida na "
                     "validação.",
        "caracteristicas": NOMES_CARACTERISTICAS,
        "entrada": {"largura": largura, "altura": altura},
        "n_treino": len(y_treino),
        "n_val": len(y_val),
        "semente": SEED,
        "limiar_destaque": LIMIAR_DESTAQUE,
        "referencias": {
            "acaso": 1 / len(bracol.CLASSES),
            "majoritaria": {"classe": majoritaria,
                            "acuracia": int(contagem.iloc[0]) / len(val)},
            "modelo": {"epoca": metricas_run["epoca"], "acuracia": classificacao["acuracia"],
                       "f1_macro": classificacao["f1_macro"],
                       "recall": dict(zip(classificacao["classes"], classificacao["recall"]))},
        },
        "variantes": variantes,
    }


def tabelas_gradcam(resultados, prancha_do_id) -> tuple[pd.DataFrame, pd.DataFrame]:
    """gradcam_amostra.csv (uma linha por imagem) e gradcam_borda.csv (uma linha por mapa)."""
    amostra, borda = [], []
    for r in resultados:
        linha = r["linha"]
        k = bracol.CLASSES.index(linha["classe_prevista"])
        amostra.append({
            "id": int(linha["id"]), "classe_verdadeira": linha["classe_verdadeira"],
            "classe_prevista": linha["classe_prevista"],
            "acerto": "sim" if linha["acerto"] else "não",
            "prancha": prancha_do_id[int(linha["id"])],
            "prob_prevista_run": linha[f"prob_{linha['classe_prevista']}"],
            "prob_prevista_fp32": float(r["probabilidades"][k]),
            "prevista_fp32": bracol.CLASSES[int(np.argmax(r["probabilidades"]))],
        })
        for mapa in r["mapas"]:
            borda.append({
                "id": int(linha["id"]), "classe_verdadeira": linha["classe_verdadeira"],
                "classe_prevista": linha["classe_prevista"],
                "acerto": "sim" if linha["acerto"] else "não", "mapa": mapa["papel"],
                "classe_do_mapa": mapa["classe"], "fracao_borda": mapa["fracao_borda"],
            })
    return pd.DataFrame(amostra), pd.DataFrame(borda)


def _gravar_csv(tabela: pd.DataFrame, caminho: Path) -> None:
    tabela.to_csv(caminho, index=False, float_format="%.6f", lineterminator="\n")


# ------------------------------------------------------------------------- relatório
def relatorio(run, checkpoint, commit, dispositivo, faixas, grupos, erros, controle,
              tabela_amostra, tabela_borda, area_borda, prancha_do_id) -> str:
    """relatorio_atalho.md: as tabelas de A e B, o resumo de C e o que isto mostra e não
    mostra. Descreve fatos; não conclui que não há atalho."""
    comando = f"python evaluation/atalho.py --run {run}" + (
        "" if dispositivo == "cpu" else f" --dispositivo {dispositivo}")
    if commit[0]:
        origem = f"no commit `{commit[0][:7]}`" + (
            " (com alterações não commitadas)" if commit[1] else "")
    else:
        origem = "fora de um repositório git"
    linhas = [
        f"# Checagem de atalho: run `{run}`", "",
        f"Gerado por `{comando}` em {datetime.now():%d/%m/%Y %H:%M}, {origem}. O modelo é o "
        f"`melhor.pt` do run (época {checkpoint['epoca']}), sem retreino. Usa só treino e "
        "validação do BRACOL: nenhuma imagem de teste foi lida, e o teste não foi avaliado.", "",
        "**Por que checar.** As classes do BRACOL vêm em blocos de ids seguidos, e o brilho e o "
        "tom do fundo mudam por bloco (`data/reports/eda_bracol.md`, seções 2 e 8). Se cada "
        "bloco for uma sessão de fotos, o modelo pode ter aprendido a sessão (luz, fundo, "
        "câmera) em vez da lesão. A divisão treino/val/teste é por imagem, então as mesmas "
        "sessões aparecem nos três splits.", "",
    ]
    linhas += _secao_a(faixas, grupos, erros)
    linhas += _secao_b(controle)
    linhas += _secao_c(tabela_amostra, tabela_borda, area_borda, prancha_do_id)
    linhas += _secao_conclusao(faixas, grupos, controle, tabela_borda, area_borda)
    return "\n".join(linhas) + "\n"


def _secao_a(faixas, grupos, erros) -> list[str]:
    linhas = [
        "## A. Acurácia por posição na sequência de ids", "",
        "### Faixas de 200 ids (validação)", "",
    ]
    linhas += fmt.tabela(
        ["faixa de ids", "imagens", "acertos", "acurácia", "IC 95% (Wilson)", *bracol.CLASSES],
        [[f["faixa"], fmt.n(f["n"]), fmt.n(f["acertos"]), _pct(f["acuracia"]),
          _intervalo(f["ic95_baixo"], f["ic95_alto"]), *(fmt.n(f[c]) for c in bracol.CLASSES)]
         for _, f in faixas.iterrows()],
    )
    linhas += [
        "", "Arquivo: `faixas_id.csv`. As colunas das classes contam as imagens da validação "
        "na faixa.", "",
        "### Pureza local", "",
        f"Para cada imagem da validação, a fração dos {VIZINHOS * 2} vizinhos por id "
        f"({VIZINHOS} antes e {VIZINHOS} depois, entre as 1.685 elegíveis de qualquer split, "
        "pela classe do manifest) que têm a mesma classe dela. Nas pontas da sequência, contam "
        "só os vizinhos que existem. Pureza 1,0: a imagem está no meio de um bloco da própria "
        "classe.", "",
    ]
    rotulos = dict(GRUPOS_PUREZA)
    linhas += fmt.tabela(
        ["pureza local", "imagens", "acertos", "acurácia", "IC 95% (Wilson)", *bracol.CLASSES],
        [[rotulos[g["grupo"]], fmt.n(g["n"]), fmt.n(g["acertos"]), _pct(g["acuracia"]),
          _intervalo(g["ic95_baixo"], g["ic95_alto"]), *(fmt.n(g[c]) for c in bracol.CLASSES)]
         for _, g in grupos.iterrows()],
    )
    linhas += ["", "Arquivo: `pureza_local.csv`. As colunas das classes contam as imagens da "
               "validação no grupo.", "",
               f"### Os {len(erros)} erros da validação", ""]
    linhas += fmt.tabela(
        ["id", "verdadeira", "prevista", "prob. da prevista", "prob. da verdadeira",
         "faixa de ids", "pureza local"],
        [[str(e["id"]), e["classe_verdadeira"], e["classe_prevista"],
          fmt.decimal(e["prob_prevista"], 2), fmt.decimal(e["prob_verdadeira"], 2),
          e["faixa_id"], fmt.decimal(e["pureza_local"], 2)] for _, e in erros.iterrows()],
    )
    return [*linhas, "", "Arquivo: `erros_val.csv`.", ""]


def _secao_b(controle) -> list[str]:
    ref = controle["referencias"]
    linhas = [
        "## B. Controle de fundo (o modelo \"burro\")", "",
        f"Uma regressão logística só com a média e o desvio de R, G, B e de H, S, V (12 números "
        f"por imagem), medidos na imagem já em {controle['entrada']['largura']}x"
        f"{controle['entrada']['altura']}, antes da normalização. Padronização com a média e o "
        f"desvio do treino; ajuste nas {fmt.n(controle['n_treino'])} imagens de treino; medida "
        f"nas {fmt.n(controle['n_val'])} da validação. A borda é a faixa a até 15% (ou 8%) da "
        "largura das laterais e da altura do topo e da base.", "",
    ]
    linhas += fmt.tabela(
        ["variante", "área usada", "acurácia val", "IC 95% (Wilson)", "F1 macro val",
         "acurácia treino"],
        [[v["rotulo"], _pct(v["fracao_da_area"]), _pct(v["val"]["acuracia"]),
          _intervalo(*v["val"]["acuracia_ic95"]), _pct(v["val"]["f1_macro"]),
          _pct(v["acuracia_treino"])] for v in controle["variantes"].values()],
    )
    linhas += [
        "",
        f"Referências na validação: acaso {_pct(ref['acaso'])}; sempre a classe majoritária "
        f"({ref['majoritaria']['classe']}) {_pct(ref['majoritaria']['acuracia'])}; o modelo do "
        f"run (época {ref['modelo']['epoca']}) {_pct(ref['modelo']['acuracia'])}, com F1 macro "
        f"{_pct(ref['modelo']['f1_macro'])}.", "",
    ]
    acima = [v for v in controle["variantes"].values() if v["acima_do_limiar"]]
    if acima:
        lista = _lista([f"{v['rotulo']} {_pct(v['val']['acuracia'])}" for v in acima])
        linhas += [f"> **Destaque: só a cor já acerta mais de {_pct(LIMIAR_DESTAQUE)} da "
                   f"validação:** {lista}.", ""]
    recalls = [[v["rotulo"], *(_pct(r) for r in v["val"]["recall"])]
               for v in controle["variantes"].values()]
    linhas += ["Recall do controle por classe (validação):", ""]
    linhas += fmt.tabela(["variante", *bracol.CLASSES], recalls)
    return [*linhas, "", "Arquivo: `controle_fundo.json`, com as matrizes de confusão.", ""]


def _secao_c(tabela_amostra, tabela_borda, area_borda, prancha_do_id) -> list[str]:
    erros = tabela_amostra[tabela_amostra["acerto"] == "não"]
    da_prevista = tabela_borda[tabela_borda["mapa"] == "prevista"]
    da_verdadeira = tabela_borda[tabela_borda["mapa"] == "verdadeira"]
    linhas = [
        "## C. Grad-CAM", "",
        f"Amostra da validação: os {len(erros)} erros e {ACERTOS_POR_CLASSE} acertos por classe "
        f"sorteados com a semente {SEED} ({len(tabela_amostra) - len(erros)} acertos). O mapa "
        "é o Grad-CAM da cabeça de classe, no último bloco do layer4 da ResNet50, em float32 "
        "e sem AMP. Nos erros há dois mapas: o da classe prevista e o da verdadeira. Arquivos: "
        "`gradcam_amostra.csv` e `gradcam_borda.csv`.", "",
        "**Energia do mapa na borda de 15%.** É a fração da soma do mapa que cai na faixa a "
        f"até 15% de cada lado, que ocupa {_pct(area_borda)} da área: um mapa uniforme daria "
        f"{_pct(area_borda)}.", "",
    ]
    por_classe = []
    for classe in bracol.CLASSES:
        da_classe = da_prevista[da_prevista["classe_verdadeira"] == classe]["fracao_borda"]
        if len(da_classe):
            por_classe.append([classe, fmt.n(len(da_classe)), _pct(da_classe.mean()),
                               f"{_pct(da_classe.min())} a {_pct(da_classe.max())}"])
    linhas += ["Mapa da classe prevista, por classe verdadeira:", ""]
    linhas += fmt.tabela(["classe verdadeira", "imagens", "média na borda", "mín. a máx."],
                         por_classe)
    grupos = [
        ["acertos (mapa da prevista)", da_prevista[da_prevista["acerto"] == "sim"]],
        ["erros (mapa da prevista)", da_prevista[da_prevista["acerto"] == "não"]],
        ["erros (mapa da verdadeira)", da_verdadeira],
        ["todos (mapa da prevista)", da_prevista],
    ]
    linhas += ["", "Erros x acertos:", ""]
    linhas += fmt.tabela(["grupo", "mapas", "média na borda", "mín. a máx."],
                         [[nome, fmt.n(len(g)), _pct(g["fracao_borda"].mean()),
                           f"{_pct(g['fracao_borda'].min())} a {_pct(g['fracao_borda'].max())}"]
                          for nome, g in grupos if len(g)])
    mudou = tabela_amostra[tabela_amostra["prevista_fp32"] != tabela_amostra["classe_prevista"]]
    diferenca = (tabela_amostra["prob_prevista_fp32"] - tabela_amostra["prob_prevista_run"]).abs()
    linhas += [
        "",
        f"**Conferência.** As probabilidades da amostra, recalculadas em float32 (o run avaliou "
        f"com AMP), diferem das do `predicoes_val.csv` em no máximo "
        f"{fmt.decimal(diferenca.max(), 4)}"
        + (f"; a classe prevista mudou em {len(mudou)} imagem(ns) (ids "
           f"{', '.join(map(str, mudou['id']))}). Os mapas usam sempre a classe prevista pelo "
           "run." if len(mudou) else "; a classe prevista é a mesma nas "
           f"{len(tabela_amostra)} imagens."), "",
        "**Pranchas.** Cada linha mostra a imagem e o mapa sobreposto; o título traz o id, a "
        "classe verdadeira, a prevista e a probabilidade da prevista.", "",
    ]
    for nome in sorted(set(prancha_do_id.values()), key=_ordem_prancha):
        linhas.append(f"- [{nome}]({nome})")
    linhas += [
        "",
        "**É só um proxy.** A folha pode encostar na borda, então energia na borda não é "
        "sinônimo de fundo. O mapa sai de uma grade de 7x14 células (o layer4 em 448x224), "
        "reescalada para a imagem, e é normalizado por imagem: mostra onde o modelo olhou, de "
        "forma grosseira, e não quanto cada região pesou na decisão.", "",
    ]
    return linhas


def _secao_conclusao(faixas, grupos, controle, tabela_borda, area_borda) -> list[str]:
    fatos = []
    com_imagens = faixas[faixas["n"] > 0]
    pior, melhor = (com_imagens.loc[com_imagens["acuracia"].idxmin()],
                    com_imagens.loc[com_imagens["acuracia"].idxmax()])
    fatos.append(
        f"Por faixa de 200 ids, a acurácia vai de {_pct(pior['acuracia'])} (ids {pior['faixa']}, "
        f"{fmt.n(pior['n'])} imagens) a {_pct(melhor['acuracia'])} (ids {melhor['faixa']}, "
        f"{fmt.n(melhor['n'])} imagens). Com cerca de "
        f"{fmt.n(int(round(float(com_imagens['n'].median()))))} imagens por faixa, cada erro "
        f"vale uns {fmt.decimal(100 / float(com_imagens['n'].median()), 1)} pontos.")
    puro, misturado = grupos.iloc[0], grupos.iloc[2]
    if puro["n"] and misturado["n"]:
        ic_misturado = _intervalo(misturado["ic95_baixo"], misturado["ic95_alto"])
        ic_puro = _intervalo(puro["ic95_baixo"], puro["ic95_alto"])
        comparacao = (f"{_pct(misturado['acuracia'])} ({fmt.n(misturado['acertos'])} de "
                      f"{fmt.n(misturado['n'])}; IC {ic_misturado}) com pureza local abaixo de "
                      f"0,6, contra {_pct(puro['acuracia'])} ({fmt.n(puro['acertos'])} de "
                      f"{fmt.n(puro['n'])}; IC {ic_puro}) com pureza 1,0")
        if misturado["ic95_alto"] < puro["ic95_baixo"]:
            fatos.append(f"**A acurácia cai de forma clara nas imagens de vizinhança "
                         f"misturada:** {comparacao}. Os intervalos de Wilson não se tocam.")
        elif misturado["acuracia"] < puro["acuracia"]:
            fatos.append(f"A acurácia é menor nas imagens de vizinhança misturada: "
                         f"{comparacao}. Os intervalos se sobrepõem: com estes n, a diferença "
                         "fica dentro da incerteza.")
        else:
            fatos.append(f"A acurácia nas imagens de vizinhança misturada não é menor: "
                         f"{comparacao}.")
        recall = controle["referencias"]["modelo"]["recall"]
        dificil = min(recall, key=recall.get)
        fatos.append(f"Os grupos de pureza não têm a mesma mistura de classes: {dificil}, a "
                     f"classe de menor recall do modelo ({_pct(recall[dificil])}), tem "
                     f"{fmt.n(misturado[dificil])} das {fmt.n(misturado['n'])} imagens do grupo "
                     f"abaixo de 0,6 e {fmt.n(puro[dificil])} das {fmt.n(puro['n'])} do grupo "
                     "1,0.")
    fatos += _fatos_do_controle(controle)
    da_prevista = tabela_borda[tabela_borda["mapa"] == "prevista"]
    media = da_prevista["fracao_borda"].mean()
    if media > area_borda:
        fatos.append(f"**Em média, {_pct(media)} da energia do Grad-CAM cai na borda de 15%, "
                     f"mais que os {_pct(area_borda)} de um mapa uniforme.**")
    else:
        fatos.append(f"Em média, {_pct(media)} da energia do Grad-CAM (mapa da classe "
                     f"prevista) cai na borda de 15%, menos que os {_pct(area_borda)} de um "
                     "mapa uniforme: os mapas se concentram no interior da foto.")
    acima = da_prevista[da_prevista["fracao_borda"] > area_borda]
    fatos.append(f"{fmt.n(len(acima))} dos {fmt.n(len(da_prevista))} mapas da classe prevista "
                 f"põem na borda mais que {_pct(area_borda)} da energia"
                 + (f" (ids {', '.join(map(str, acima['id']))})." if len(acima) else "."))
    da_verdadeira = tabela_borda[tabela_borda["mapa"] == "verdadeira"]
    if len(da_verdadeira):
        erros_prevista = da_prevista[da_prevista["acerto"] == "não"]["fracao_borda"]
        maior = da_verdadeira.loc[da_verdadeira["fracao_borda"].idxmax()]
        fatos.append(f"Nos erros, o mapa da classe verdadeira põe em média "
                     f"{_pct(da_verdadeira['fracao_borda'].mean())} da energia na borda, "
                     f"contra {_pct(erros_prevista.mean())} do mapa da prevista; o maior valor "
                     f"é {_pct(maior['fracao_borda'])} (id {maior['id']}, mapa de "
                     f"{maior['classe_do_mapa']}).")
    limites = [
        "Um resultado bom aqui não prova que não há atalho. A validação tem as mesmas sessões "
        "de foto do treino (a divisão é por imagem, não por sessão), e um modelo que usa a "
        "sessão também acerta nela.",
        "A pureza local olha só a posição na sequência de ids. Ids vizinhos da mesma classe não "
        "são, com certeza, da mesma sessão: o BRACOL não traz data, câmera nem planta.",
        "A queda com a pureza baixa pode ter outra causa além da sessão: os grupos não têm a "
        "mesma mistura de classes, e uma classe difícil concentrada nas regiões misturadas "
        "também baixaria a acurácia delas.",
        "O controle de fundo mostra que a cor da borda carrega informação da classe, não que o "
        "modelo a use: o Grad-CAM, ao mesmo tempo, aponta para o interior da foto.",
        "O controle de fundo usa 12 estatísticas de cor por região. Uma rede pode usar pistas "
        "mais finas (textura do fundo, sombras, luz), então um controle baixo não exclui atalho "
        "de fundo. E a imagem inteira inclui a cor da folha e da lesão: o controle (a) alto, "
        "sozinho, não indica uso do fundo.",
        "A borda de 15% ocupa 51,2% da área, e a de 8%, 29,6%. Nas fotos do BRACOL, a folha pode "
        "entrar nessas faixas, então a borda não é só fundo.",
        "O Grad-CAM é grosso (grade de 7x14) e normalizado por imagem. A amostra tem só "
        f"{fmt.n(len(da_prevista))} imagens, e a validação, 252: os intervalos são largos.",
        "Uma evidência mais forte exigiria validar com blocos de ids inteiros fora do treino, "
        "ou com fotos de outra sessão ou do campo. É uma decisão do gestor.",
    ]
    return [
        "## O que isto mostra e o que NÃO mostra", "",
        "**Mostra (fatos medidos):**", "",
        *(f"- {f}" for f in fatos), "",
        "**Não mostra:**", "",
        *(f"- {item}" for item in limites), "",
    ]


def _fatos_do_controle(controle) -> list[str]:
    """As frases do controle de fundo para a seção de fatos do relatório."""
    ref = controle["referencias"]
    variantes = list(controle["variantes"].values())
    maj = ref["majoritaria"]
    valores = _lista([f"{v['rotulo']} {_pct(v['val']['acuracia'])}" for v in variantes])
    texto = (f"o controle de fundo acerta {valores} da validação, contra "
             f"{_pct(maj['acuracia'])} de prever sempre a classe majoritária ({maj['classe']}) "
             f"e {_pct(ref['acaso'])} do acaso")
    if all(v["val"]["acuracia_ic95"][0] > maj["acuracia"] for v in variantes):
        texto += "; o intervalo de Wilson de cada variante fica inteiro acima da majoritária"
    if any(v["acima_do_limiar"] for v in variantes):
        fatos = [f"**Só a cor já passa de {_pct(LIMIAR_DESTAQUE)} de acurácia:** {texto}."]
    else:
        fatos = [f"{texto[0].upper()}{texto[1:]}."]
    inteira, borda = controle["variantes"]["imagem_inteira"], controle["variantes"]["borda_8"]
    fatos.append(f"A borda de 8%, que ocupa {_pct(borda['fracao_da_area'])} da área, acerta "
                 f"{_pct(borda['val']['acuracia'])}; a imagem inteira, "
                 f"{_pct(inteira['val']['acuracia'])}.")
    recall = dict(zip(borda["val"]["classes"], borda["val"]["recall"]))
    alta, baixa = max(recall, key=recall.get), min(recall, key=recall.get)
    fatos.append(f"Só com a cor da borda de 8%, o recall vai de {_pct(recall[baixa])} ({baixa}) "
                 f"a {_pct(recall[alta])} ({alta})."
                 + (" Na EDA (seção 8), phoma é a classe de fundo mais amarelado."
                    if alta == "phoma" else ""))
    return fatos


def _lista(itens: list[str]) -> str:
    """Itens unidos em português: 'a, b e c'."""
    return itens[0] if len(itens) == 1 else f"{', '.join(itens[:-1])} e {itens[-1]}"


def _ordem_prancha(nome: str) -> tuple[int, int]:
    return (0 if "erros" in nome else 1, int(nome.rsplit("_", 1)[1].split(".")[0]))


def _resumo_grupo(grupos, k: int) -> str:
    g = grupos.iloc[k]
    return f"{_pct(g['acuracia'])} (n {g['n']})" if g["n"] else "sem imagens"


def _intervalo(baixo, alto) -> str:
    if pd.isna(baixo):
        return "-"
    return f"{_pct(baixo)} a {_pct(alto)}"


def _pct(fracao) -> str:
    return "-" if pd.isna(fracao) else fmt.pct(fracao, 1)


# -------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Checagem de atalho de um run, so com treino e validacao do BRACOL. "
                    "Detalhes no docstring de evaluation/atalho.py."
    )
    ap.add_argument("--run", required=True, help="nome do run em model/runs/")
    ap.add_argument("--dispositivo", default="cpu", choices=["cpu", "cuda", "auto"],
                    help="padrao: cpu (os mapas se repetem iguais entre execucoes)")
    args = ap.parse_args(argv)
    try:
        return checar(args.run, args.dispositivo)
    except (train.ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
