"""
Diagnóstico de campo sem treinar rede (Frente 9, passo 5a), nas folhas rotuladas S ou D do campo
rotulado (sem os "?"), com os checkpoints existentes:

a) AUC de P(saudável) separando S de D, nos modos A e B, para o modelo atual
   (base_resnet50_bracol) e as variantes v0, v1, v3 e v3s do passo 4, com intervalo por bootstrap
   agrupado por foto. Ponto de operação: o limiar em P(saudável) que marca como saudáveis no
   máximo 5% das D, e o recall de S nele, com Wilson. O limiar é escolhido nas mesmas folhas em
   que é medido: o número sai otimista.
b) Sonda linear: regressão logística regularizada sobre os embeddings congelados (a saída do
   backbone) do modelo atual e do v0, no modo B, S x D, com validação cruzada de 5 dobras
   agrupada por cena. AUC com intervalo. Só diagnóstico de sinal: não gera modelo.
c) Auditoria das S: prancha com as folhas S que o modelo atual (modo B) previu como doença com
   maior confiança (até 12), em resolução nativa da foto, e Grad-CAM de 8 delas, em
   data/processed/campo_diagnostico/ (fora do git).

Os rótulos de campo foram feitos por Claude por leitura visual das pranchas e aceitos pelo
gestor em 11/10/2026 (não são de especialista). Os testes do BRACOL e do BRACOT ficam fechados.
Saída: model/runs/campo_diagnostico/ (diagnostico.json, folhas.csv, sonda.csv e relatorio.md,
com a leitura escrita à mão em leitura.md).

Uso (de qualquer pasta, com o venv ativo):
    python evaluation/campo_diagnostico.py              # roda tudo (alguns minutos na GPU)
    python evaluation/campo_diagnostico.py --relatorio  # só refaz o relatorio.md do JSON
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import bracot  # noqa: E402
import campo  # noqa: E402
import dados  # noqa: E402
import formatacao as fmt  # noqa: E402
import inferencia as inf  # noqa: E402
import metrics  # noqa: E402
import train  # noqa: E402

PASTA_SAIDA = train.PASTA_RUNS / "campo_diagnostico"
PASTA_IMAGENS = RAIZ_REPO / "data" / "processed" / "campo_diagnostico"  # fora do git
MODELOS = {"atual": train.PASTA_RUNS / "base_resnet50_bracol",
           "v0": train.PASTA_RUNS / "campo_v0", "v1": train.PASTA_RUNS / "campo_v1",
           "v3": train.PASTA_RUNS / "campo_v3", "v3s": train.PASTA_RUNS / "campo_v3s"}
SONDA_MODELOS = ("atual", "v0")
MODOS = ("A", "B")
SAUDAVEL = inf.CATEGORIAS.index("saudavel")
BOOTSTRAP = 2000
SEED = 42
FRACAO_D_SAUDAVEL_MAX = 0.05
C_SONDA = 0.1  # regularização L2 da sonda (fixada antes de olhar o resultado)
C_SENSIBILIDADE = (0.01, 0.1, 1.0)
DOBRAS = 5
AUDITORIA = 12
GRADCAM = 8


# ------------------------------------------------------------------------- estatística
def auc(y, escore) -> float:
    """AUC (Mann-Whitney) de `escore` separando y=1 de y=0; NaN se faltar uma das classes."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, escore))


def auc_bootstrap(y, escore, grupos, n: int = BOOTSTRAP, seed: int = SEED) -> dict:
    """AUC e intervalo de 95% por bootstrap agrupado (sorteia grupos com reposição)."""
    y, escore, grupos = np.asarray(y), np.asarray(escore, dtype=float), np.asarray(grupos)
    unicos = np.unique(grupos)
    indices = {g: np.nonzero(grupos == g)[0] for g in unicos}
    gerador = np.random.default_rng(seed)
    valores = []
    for _ in range(n):
        sorteio = np.concatenate([indices[g] for g in gerador.choice(unicos, len(unicos))])
        valor = auc(y[sorteio], escore[sorteio])
        if not math.isnan(valor):
            valores.append(valor)
    return {"auc": auc(y, escore), "ic95": [float(np.percentile(valores, 2.5)),
                                            float(np.percentile(valores, 97.5))],
            "reamostras_validas": len(valores)}


def ponto_de_operacao(y, p_saudavel, fracao_d_max: float = FRACAO_D_SAUDAVEL_MAX) -> dict:
    """O limiar em P(saudável) que marca como saudáveis no máximo `fracao_d_max` das D (y=0):
    saudável se P > limiar, com o limiar no (k+1)-ésimo maior P das D, k = floor(fração x n_D).
    Devolve o limiar, quantas D passam e o recall de S (y=1), com Wilson."""
    y, p = np.asarray(y), np.asarray(p_saudavel, dtype=float)
    doentes = np.sort(p[y == 0])[::-1]
    k = int(math.floor(fracao_d_max * len(doentes)))
    limiar = float(doentes[k]) if k < len(doentes) else float("-inf")
    saudaveis = p[y == 1]
    acertos = int((saudaveis > limiar).sum())
    recall = acertos / len(saudaveis) if len(saudaveis) else float("nan")
    return {"limiar": limiar, "d_marcadas_saudaveis": int((doentes > limiar).sum()),
            "n_d": len(doentes), "recall_s": recall, "acertos_s": acertos,
            "n_s": len(saudaveis),
            "recall_s_ic95": list(metrics.intervalo_wilson(recall, len(saudaveis)))
            if len(saudaveis) else [float("nan")] * 2}


def sonda_linear(embeddings, y, grupos, c: float = C_SONDA, dobras: int = DOBRAS):
    """Escores fora da dobra de uma regressão logística L2 (dados padronizados no treino de cada
    dobra), com GroupKFold pelos grupos."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    escores = np.full(len(y), np.nan)
    for treino, fora in GroupKFold(n_splits=dobras).split(embeddings, y, grupos):
        modelo = make_pipeline(StandardScaler(),
                               LogisticRegression(C=c, max_iter=5000, random_state=SEED))
        modelo.fit(embeddings[treino], y[treino])
        escores[fora] = modelo.predict_proba(embeddings[fora])[:, 1]
    return escores


# ------------------------------------------------------------------------------ dados
def folhas_rotuladas():
    """As folhas S e D do campo rotulado, casadas com as folhas da cadeia de rotulagem, com os
    recortes A e B (como o classificador vê) e o necessário para a auditoria."""
    rotulos = campo.ler_rotulos()
    if rotulos is None:
        raise train.ErroFatal("data/manifests/campo_rotulado.csv ainda não tem rótulos")
    modelos = inf.carregar_modelos(MODELOS["atual"], dispositivo=_dispositivo())
    config = campo.CONFIG_ROTULAGEM
    manifest = bracot.ler_manifest(split="treino").set_index("id")
    linhas, recortes = [], {m: [] for m in MODOS}
    for foto, reduzida, escala, folhas in campo.percorrer_folhas(modelos, config):
        tabela = pd.DataFrame([{**{k: f[k] for k in ("id", "borda")},
                                **dict(zip(("caixa_x0", "caixa_y0", "caixa_x1", "caixa_y1"),
                                           f["caixa"]))} for f in folhas])
        casados = campo.casar_rotulos(rotulos[rotulos["foto"] == foto], tabela)
        casados = casados[casados["rotulo"].isin(["S", "D"])]
        por_id = {f["id"]: f for f in folhas}
        for r in casados.itertuples():
            folha = por_id[r.id]
            linhas.append({"id": r.id, "foto": foto, "cena": manifest.loc[foto, "cena"],
                           "rotulo": r.rotulo, "observacao": r.observacao,
                           "borda": folha["borda"], "fracao_visivel": folha["fracao_visivel"],
                           "poligono": folha["poligono"],
                           "caixa": folha["caixa"]})
            for modo in MODOS:
                recortes[modo].append(campo.recorte(reduzida, escala, folha, modo, config))
    return pd.DataFrame(linhas), recortes


def prever(pasta, recortes):
    """Probabilidades (n x 5) de um classificador nos recortes, em lotes."""
    modelos = inf.carregar_modelos(pasta, dispositivo=_dispositivo())
    probabilidades, _ = campo._classificar_em_lotes(recortes, modelos)
    return probabilidades


def embeddings(pasta, recortes) -> np.ndarray:
    """A saída do backbone (as features antes das duas cabeças), congelada, em float32."""
    import torch

    modelos = inf.carregar_modelos(pasta, dispositivo=_dispositivo())
    saidas = []
    with torch.inference_mode():
        for i in range(0, len(recortes), campo.LOTE):
            lote = torch.stack([modelos.transformacao(image=r)["image"]
                                for r in recortes[i:i + campo.LOTE]])
            saidas.append(modelos.classificador.backbone(lote.to(modelos.dispositivo))
                          .float().cpu().numpy())
    return np.concatenate(saidas)


def _dispositivo() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- controles
CONTROLE_COLUNAS = {"P(saudável) do atual, B": "p_saudavel_atual_B",
                    "sonda do atual": "sonda_atual", "sonda do v0": "sonda_v0"}


def controles(tabela: pd.DataFrame, y) -> dict:
    """Dois controles: a sessão de fotos (S e D poderiam vir de sessões diferentes, e a AUC
    mediria a sessão) e as folhas que ficam no filtro de fragmentos ligado em 11/10/2026 (a
    população que a cadeia classifica agora)."""
    y = np.asarray(y)
    sessao = tabela["foto"].str[:8].to_numpy()
    fica = (tabela["fracao_visivel"] >= inf.Configuracao().fracao_visivel_minima).to_numpy()
    subconjuntos = {f"sessão {s}": sessao == s for s in sorted(set(sessao))}
    subconjuntos["ficam no filtro de fragmentos"] = fica
    return {"auc_so_da_sessao": auc(y, (sessao == max(sessao)).astype(int)),
            "subconjuntos": {nome: {"S": int(y[m].sum()), "D": int((1 - y[m]).sum()),
                                    **{rotulo: auc(y[m], tabela.loc[m, coluna])
                                       for rotulo, coluna in CONTROLE_COLUNAS.items()}}
                             for nome, m in subconjuntos.items()}}


def efeito_no_bracol(limiar: float, predicoes: pd.DataFrame | None = None) -> dict:
    """O que a regra "saudável se P(saudável) > limiar" faria na validação do BRACOL, contra o
    argmax, nas predições da imagem inteira do modelo atual (predicoes_val.csv; nunca o teste)."""
    if predicoes is None:
        predicoes = pd.read_csv(MODELOS["atual"] / "predicoes_val.csv")
    doente = (predicoes["classe_verdadeira"] != "saudavel").to_numpy()
    regras = {"argmax": (predicoes["classe_prevista"] == "saudavel").to_numpy(),
              "regra_limiar": predicoes["prob_saudavel"].to_numpy() > limiar}
    resultado = {"limiar": limiar, "n_s": int((~doente).sum()), "n_d": int(doente.sum())}
    for nome, saudavel in regras.items():
        d_saudaveis, s_doentes = int((saudavel & doente).sum()), int((~saudavel & ~doente).sum())
        resultado[nome] = {
            "d_previstas_saudaveis": d_saudaveis, "s_previstas_doentes": s_doentes,
            "p_saudavel_dado_d": d_saudaveis / resultado["n_d"],
            "p_saudavel_dado_d_ic95": list(metrics.intervalo_wilson(
                d_saudaveis / resultado["n_d"], resultado["n_d"])),
            "p_doenca_dado_s": s_doentes / resultado["n_s"],
            "p_doenca_dado_s_ic95": list(metrics.intervalo_wilson(
                s_doentes / resultado["n_s"], resultado["n_s"]))}
    return resultado


# --------------------------------------------------------------------------- auditoria
def auditar(folhas: pd.DataFrame, probabilidades_b, recortes_b) -> list[dict]:
    """As S previstas como doença com maior confiança pelo modelo atual (modo B): prancha em
    resolução nativa e Grad-CAM de GRADCAM delas, em PASTA_IMAGENS."""
    import gradcam
    import torch

    prevista = probabilidades_b.argmax(axis=1)
    confianca = probabilidades_b.max(axis=1)
    candidatas = np.nonzero((folhas["rotulo"].to_numpy() == "S") & (prevista != SAUDAVEL))[0]
    escolhidas = candidatas[np.argsort(-confianca[candidatas])][:AUDITORIA]
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    _, caminhos = campo.fotos_de_validacao()
    itens = []
    for i in escolhidas:
        linha = folhas.iloc[i]
        foto = dados.ler_imagem(RAIZ_REPO / caminhos[linha["foto"]])
        altura, largura = foto.shape[:2]
        x0, y0, x1, y1 = (np.asarray(linha["caixa"]) * [largura, altura, largura, altura])
        margem = 0.05 * max(x1 - x0, y1 - y0)
        x0, y0 = int(max(0, x0 - margem)), int(max(0, y0 - margem))
        x1, y1 = int(min(largura, x1 + margem)), int(min(altura, y1 + margem))
        itens.append({"id": linha["id"], "prevista": inf.CATEGORIAS[prevista[i]],
                      "confianca": float(confianca[i]), "p_saudavel":
                      float(probabilidades_b[i, SAUDAVEL]), "observacao": linha["observacao"],
                      "nativo": foto[y0:y1, x0:x1].copy(), "indice": int(i)})
    _prancha_nativa(itens, PASTA_IMAGENS / "auditoria_s_nativa.jpg")
    modelos = inf.carregar_modelos(MODELOS["atual"], dispositivo=_dispositivo())
    ver = dados.transformacao_sem_normalizar()
    cartoes = []
    for item in itens[:GRADCAM]:
        recorte = recortes_b[item["indice"]]
        tensor = modelos.transformacao(image=recorte)["image"]
        with torch.enable_grad():
            mapa = gradcam.gerar_gradcam(modelos.classificador, tensor)
        base = ver(image=recorte)["image"]
        calor = cv2.applyColorMap((mapa * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        sobreposta = (0.55 * base + 0.45 * calor).astype(np.uint8)
        cartao = np.full((254, 904, 3), 255, np.uint8)
        cartao[30:, :448], cartao[30:, 456:] = base, sobreposta
        cv2.putText(cartao, f"{item['id']}: {item['prevista']} {item['confianca']:.2f}",
                    (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cartoes.append(cartao)
    _gravar_grade(cartoes, 2, "Grad-CAM (classe prevista) das folhas S previstas como doenca, "
                              "modelo atual, modo B", PASTA_IMAGENS / "auditoria_s_gradcam.jpg")
    for item in itens:
        del item["nativo"]
    return itens


def _prancha_nativa(itens, caminho: Path) -> None:
    """Os recortes em resolução nativa (sem reduzir), cada um numa célula do tamanho do maior."""
    if not itens:
        return
    altura = max(i["nativo"].shape[0] for i in itens) + 60
    largura = max(i["nativo"].shape[1] for i in itens)
    cartoes = []
    for item in itens:
        cartao = np.full((altura, largura, 3), 255, np.uint8)
        h, w = item["nativo"].shape[:2]
        cartao[60:60 + h, :w] = item["nativo"]
        texto = (f"{item['id']}: previsto {item['prevista']} {item['confianca']:.2f} "
                 f"(P saudavel {item['p_saudavel']:.2f}) {item['observacao']}")
        cv2.putText(cartao, texto[:110], (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cartoes.append(cartao)
    _gravar_grade(cartoes, 3, "Folhas S previstas como doenca (modelo atual, modo B), em "
                              "resolucao nativa", caminho)


def _gravar_grade(cartoes, colunas: int, titulo: str, caminho: Path) -> None:
    altura, largura = cartoes[0].shape[:2]
    linhas = math.ceil(len(cartoes) / colunas)
    escala_texto = max(0.8, largura / 900)
    grade = np.full((90 + linhas * (altura + 10) + 50, colunas * (largura + 10) + 10, 3), 255,
                    np.uint8)
    for k, cartao in enumerate(cartoes):
        y, x = 90 + (k // colunas) * (altura + 10), 10 + (k % colunas) * (largura + 10)
        grade[y:y + altura, x:x + largura] = cartao
    cv2.putText(grade, titulo, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, escala_texto, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(grade, campo.CREDITO_BRACOT, (10, grade.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                escala_texto * 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(grade, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        caminho.write_bytes(jpeg.tobytes())


# ----------------------------------------------------------------------------- relatório
def _pct(valor, casas: int = 1) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "-"
    return fmt.pct(valor, 1, casas)


def _dec(valor, casas: int = 2) -> str:
    return "-" if valor is None or math.isnan(valor) else fmt.decimal(valor, casas)


def _auc(r) -> str:
    return f"{_dec(r['auc'])} ({_dec(r['ic95'][0])} a {_dec(r['ic95'][1])})"


def relatorio(resultado: dict, leitura: str | None) -> str:
    n = resultado["folhas"]
    linhas = [
        "# Diagnóstico de campo sem treinar rede (passo 5a)", "",
        f"Gerado por `python evaluation/campo_diagnostico.py`. {n['S'] + n['D']} folhas rotuladas "
        f"do campo ({n['S']} S e {n['D']} D, sem os \"?\"), de {n['fotos']} fotos de validação do "
        f"BRACOT em {n['cenas']} cenas. Rótulos {campo.ORIGEM_ROTULOS}. Testes do BRACOL e do "
        "BRACOT fechados; checkpoints existentes, nenhum treino.", "",
        "## a) P(saudável) separa S de D?", "",
        f"AUC de P(saudável), com S positiva; intervalo de 95% por bootstrap agrupado por foto "
        f"({fmt.n(BOOTSTRAP)} reamostras). Ponto de operação: o limiar em P(saudável) com no "
        "máximo "
        f"{_pct(FRACAO_D_SAUDAVEL_MAX, 0)} das D marcadas como saudáveis, e o recall de S nele "
        "(Wilson). **O limiar é escolhido nas mesmas 82 folhas em que é medido: o recall sai "
        "otimista.**", ""]
    linhas += fmt.tabela(
        ["modelo", "modo", "AUC (IC 95%)", "limiar em P(saudável)", "D marcadas saudáveis",
         "recall de S (IC 95%)"],
        [[m, modo, _auc(r["auc"]), _dec(r["ponto"]["limiar"], 3),
          f"{r['ponto']['d_marcadas_saudaveis']} de {r['ponto']['n_d']}",
          f"{_pct(r['ponto']['recall_s'])} ({_pct(r['ponto']['recall_s_ic95'][0])} a "
          f"{_pct(r['ponto']['recall_s_ic95'][1])}; {r['ponto']['acertos_s']} de "
          f"{r['ponto']['n_s']})"]
         for m, por_modo in resultado["auc"].items() for modo, r in por_modo.items()], "llllll")
    bv = resultado["bracol_val"]
    linhas += ["", f"Efeito colateral no BRACOL: a regra \"saudável se P(saudável) > "
               f"{_dec(bv['limiar'], 3)}\" (o limiar do modelo atual no modo B) na validação do "
               f"BRACOL ({bv['n_s']} saudáveis e {bv['n_d']} doentes; predições da imagem inteira "
               "em `base_resnet50_bracol/predicoes_val.csv`, não a cadeia; nunca o teste), "
               "contra o argmax de hoje:", ""]
    linhas += fmt.tabela(
        ["regra", "P(prever saudável dado D)", "P(prever doença dado S)"],
        [[nome, f"{_pct(r['p_saudavel_dado_d'])} ({_pct(r['p_saudavel_dado_d_ic95'][0])} a "
                f"{_pct(r['p_saudavel_dado_d_ic95'][1])}; {r['d_previstas_saudaveis']} de "
                f"{bv['n_d']})",
          f"{_pct(r['p_doenca_dado_s'])} ({_pct(r['p_doenca_dado_s_ic95'][0])} a "
          f"{_pct(r['p_doenca_dado_s_ic95'][1])}; {r['s_previstas_doentes']} de {bv['n_s']})"]
         for nome, r in (("argmax (hoje)", bv["argmax"]),
                         (f"P(saudável) > {_dec(bv['limiar'], 3)}", bv["regra_limiar"]))],
        "lll")
    linhas += ["", "## b) Sonda linear nos embeddings congelados (modo B)", "",
               f"Regressão logística L2 (C = {fmt.decimal(C_SONDA, 1)}, fixado antes; dados "
               "padronizados no treino "
               f"de cada dobra) sobre a saída do backbone (2.048 números por folha), S x D, "
               f"validação cruzada de {DOBRAS} dobras agrupada por cena. AUC dos escores fora da "
               "dobra, com intervalo por bootstrap agrupado por foto. Só mede se há sinal linear "
               "nos embeddings; não gera modelo.", ""]
    linhas += fmt.tabela(
        ["embeddings", "AUC da sonda (IC 95%)",
         *(f"AUC com C = {fmt.decimal(c, 2)}" for c in C_SENSIBILIDADE),
         "AUC de P(saudável), modo B (mesmo modelo)"],
        [[m, _auc(r["auc"]), *(_dec(r["sensibilidade"][f"{c:g}"]) for c in C_SENSIBILIDADE),
          _auc(resultado["auc"][m]["B"]["auc"])]
         for m, r in resultado["sonda"].items()], "llllll")
    ctl = resultado["controles"]
    linhas += ["", "### Controles", "",
               "Se S e D viessem de sessões de fotos diferentes, a AUC poderia medir a sessão e "
               "não a folha. A sessão sozinha, como escore, dá AUC de "
               f"{_dec(ctl['auc_so_da_sessao'])}. A última linha refaz a AUC só nas folhas que "
               "ficam no filtro de fragmentos (fração visível de pelo menos "
               f"{fmt.decimal(inf.Configuracao().fracao_visivel_minima, 1)}), que é o que a "
               "cadeia classifica com os padrões de 11/10/2026 (sem intervalo: subconjuntos "
               "pequenos).", ""]
    linhas += fmt.tabela(["folhas", "S", "D", *(f"AUC: {r}" for r in CONTROLE_COLUNAS)],
                         [[nome, str(r["S"]), str(r["D"]),
                           *(_dec(r[rotulo]) for rotulo in CONTROLE_COLUNAS)]
                          for nome, r in ctl["subconjuntos"].items()], "lrrlll")
    aud = resultado["auditoria"]
    linhas += ["", "## c) Auditoria das S previstas como doença (modelo atual, modo B)", "",
               f"{len(aud)} folhas S que o modelo atual previu como doença com maior confiança. "
               "As pranchas ficam fora do git: `data/processed/campo_diagnostico/"
               "auditoria_s_nativa.jpg` (resolução nativa) e `auditoria_s_gradcam.jpg` (Grad-CAM "
               f"de {min(GRADCAM, len(aud))}).", ""]
    linhas += fmt.tabela(["folha", "prevista", "confiança", "P(saudável)", "observação"],
                         [[a["id"], a["prevista"], _pct(a["confianca"]), _pct(a["p_saudavel"]),
                           a["observacao"] or "-"] for a in aud], "lllll")
    linhas += ["", "## d) O que isso implica", "",
               leitura or "Ainda não escrito (falta o leitura.md).", ""]
    return "\n".join(linhas) + "\n"


# -------------------------------------------------------------------------------- main
def gravar_relatorio(resultado: dict) -> None:
    """relatorio.md a partir do resultado, com a leitura escrita à mão em leitura.md."""
    leitura = PASTA_SAIDA / "leitura.md"
    (PASTA_SAIDA / "relatorio.md").write_text(
        relatorio(resultado, leitura.read_text(encoding="utf-8").strip()
                  if leitura.is_file() else None), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico de campo sem treinar rede.")
    parser.add_argument("--relatorio", action="store_true",
                        help="só refaz o relatorio.md a partir do diagnostico.json")
    args = parser.parse_args()
    try:
        if args.relatorio:
            caminho = PASTA_SAIDA / "diagnostico.json"
            if not caminho.is_file():
                raise train.ErroFatal(f"{caminho} não existe: rode sem --relatorio antes")
            gravar_relatorio(json.loads(caminho.read_text(encoding="utf-8")))
            fmt.dizer(f"Relatorio refeito: {PASTA_SAIDA / 'relatorio.md'}")
            return 0
        fmt.dizer("Lendo as folhas rotuladas e os recortes...")
        folhas, recortes = folhas_rotuladas()
        y = (folhas["rotulo"] == "S").astype(int).to_numpy()
        fotos = folhas["foto"].to_numpy()
        resultado = {"folhas": {"S": int(y.sum()), "D": int((1 - y).sum()),
                                "fotos": int(folhas["foto"].nunique()),
                                "cenas": int(folhas["cena"].nunique())},
                     "auc": {}, "sonda": {}}
        tabela = folhas[["id", "foto", "cena", "rotulo", "observacao", "borda",
                         "fracao_visivel"]].copy()
        probabilidades_atual_b = None
        for nome, pasta in MODELOS.items():
            fmt.dizer(f"Modelo {nome}...")
            resultado["auc"][nome] = {}
            for modo in MODOS:
                p = prever(pasta, recortes[modo])
                tabela[f"p_saudavel_{nome}_{modo}"] = p[:, SAUDAVEL]
                resultado["auc"][nome][modo] = {
                    "auc": auc_bootstrap(y, p[:, SAUDAVEL], fotos),
                    "ponto": ponto_de_operacao(y, p[:, SAUDAVEL])}
                if nome == "atual" and modo == "B":
                    probabilidades_atual_b = p
        for nome in SONDA_MODELOS:
            fmt.dizer(f"Sonda linear: embeddings de {nome}...")
            emb = embeddings(MODELOS[nome], recortes["B"])
            escores = sonda_linear(emb, y, folhas["cena"].to_numpy())
            tabela[f"sonda_{nome}"] = escores
            resultado["sonda"][nome] = {
                "auc": auc_bootstrap(y, escores, fotos),
                "sensibilidade": {f"{c:g}": auc(y, sonda_linear(emb, y, folhas["cena"].to_numpy(),
                                                                 c=c))
                                  for c in C_SENSIBILIDADE}}
        resultado["controles"] = controles(tabela, y)
        resultado["bracol_val"] = efeito_no_bracol(
            resultado["auc"]["atual"]["B"]["ponto"]["limiar"])
        fmt.dizer("Auditoria das S...")
        resultado["auditoria"] = auditar(folhas, probabilidades_atual_b, recortes["B"])
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(PASTA_SAIDA / "folhas.csv", index=False, float_format="%.6f",
                      lineterminator="\n")
        train.gravar_json(campo._nativo(resultado), PASTA_SAIDA / "diagnostico.json")
        gravar_relatorio(resultado)
        for nome, por_modo in resultado["auc"].items():
            fmt.dizer(f"{nome}: AUC A {por_modo['A']['auc']['auc']:.2f} B "
                      f"{por_modo['B']['auc']['auc']:.2f}")
        for nome, r in resultado["sonda"].items():
            fmt.dizer(f"sonda {nome}: AUC {r['auc']['auc']:.2f}")
        return 0
    except (train.ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
