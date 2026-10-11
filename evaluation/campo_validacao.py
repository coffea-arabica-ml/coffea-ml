"""
Validação de campo da regra em duas etapas (Frente 9, passo 5b), sem reajustar nada com os
dados novos.

A regra já está fixada: limiar de saudável t_s = 0,083 (decisão do gestor de 11/10/2026, tirado
das 82 folhas antigas) e limiar de doença só como proposta (o do run pipeline_duas_etapas). O
modelo é o congelado (base_resnet50_bracol), no recorte B. Dois conjuntos rotulados:
- antigas: as 120 folhas do campo rotulado (fotos de validação do detector; 82 S ou D), na cadeia
  da rotulagem do passo 4 (campo.CONFIG_ROTULAGEM);
- novas: as 160 folhas da rotulagem de validação (fotos de treino do detector; uma por foto), na
  cadeia de 11/10/2026 (campo.CONFIG_ROTULAGEM_TREINO). O detector treinou nessas fotos: as
  máscaras saem mais limpas que as de fotos novas.

a) P(chamar saudável | S) e P(chamar saudável | D) com Wilson, e a AUC de P(saudável) com
   bootstrap agrupado por foto, na regra em duas etapas e no argmax (com 0,61 e sem corte).
b) O mesmo nas antigas, nas novas e juntas, e a curva de t_s (recall de S x falso saudável em D)
   só como informação: nenhum limiar novo é escolhido.
c) Qualidade da rotulagem: fração de "?" por tipo de folha (fragmento, borda, brilho) e os
   limites com os "?" tratados como S ou como D.
d) Relatório em model/runs/campo_validacao_duas_etapas/relatorio.md, com a leitura escrita à mão
   em leitura.md.

Os rótulos foram feitos por Claude por leitura visual das pranchas e aceitos pelo gestor em
11/10/2026 (não são de especialista). Nada de treino; os testes do BRACOL e do BRACOT ficam
fechados.

Uso (de qualquer pasta, com o venv ativo):
    python evaluation/campo_validacao.py              # roda tudo (alguns minutos na GPU)
    python evaluation/campo_validacao.py --relatorio  # só refaz o relatorio.md do JSON
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import campo  # noqa: E402
import campo_diagnostico as cd  # noqa: E402
import formatacao as fmt  # noqa: E402
import inferencia as inf  # noqa: E402
import metrics  # noqa: E402
import train  # noqa: E402

PASTA_SAIDA = train.PASTA_RUNS / "campo_validacao_duas_etapas"
RUN_DUAS_ETAPAS = train.PASTA_RUNS / "pipeline_duas_etapas"
FOLHAS_DIAGNOSTICO = train.PASTA_RUNS / "campo_diagnostico" / "folhas.csv"
LIMIAR_SAUDAVEL = inf.Configuracao().limiar_saudavel  # 0,083, fixado em 11/10/2026
CONJUNTOS = ("antigas", "novas", "juntas")
NOME_CONJUNTO = {"antigas": "antigas (82; t_s saiu delas)", "novas": "novas (fotos de treino)",
                 "juntas": "antigas e novas juntas"}
NOME_REGRA = {"duas_etapas": "duas etapas", "argmax_061": "argmax, corte 0,61",
              "argmax_sem_corte": "argmax, sem corte"}
LIMIARES_SENSIBILIDADE = (0.01, 0.02, 0.03, 0.05, 0.083, 0.1, 0.15, 0.2, 0.3, 0.5)
CENARIOS = ("sem_interrogacao", "interrogacao_como_S", "interrogacao_como_D")
NOME_CENARIO = {"sem_interrogacao": "sem os \"?\"", "interrogacao_como_S": "\"?\" como S",
                "interrogacao_como_D": "\"?\" como D"}
TIPOS = ("fragmento (observação)", "fração visível < 0,5 (cadeia)", "borda (cadeia)",
         "brilho (observação)", "nenhum desses", "todas")


# ---------------------------------------------------------------------------- regras
def limiar_doenca_proposto(run: Path = RUN_DUAS_ETAPAS) -> float:
    """O limiar de doença proposto no run pipeline_duas_etapas (só como proposta)."""
    caminho = run / "metricas.json"
    if not caminho.is_file():
        raise train.ErroFatal(f"{train._exibir(caminho)} não existe: rode o "
                              "avaliar_pipeline --decisao duas_etapas antes")
    proposta = json.loads(caminho.read_text(encoding="utf-8"))["duas_etapas"]["proposta"]
    if proposta is None:
        raise train.ErroFatal("o run pipeline_duas_etapas não tem proposta de limiar de doença")
    return float(proposta)


def regras(limiar_doenca: float) -> dict:
    """As regras comparadas, com os limiares fixados (nada se ajusta aqui)."""
    return {"duas_etapas": inf.Configuracao(decisao="duas_etapas", limiar_doenca=limiar_doenca),
            "argmax_061": inf.Configuracao(),
            "argmax_sem_corte": inf.Configuracao(limiar_confianca=0.0)}


def desfecho(probs, config) -> str:
    """O que a cadeia diz da folha: "saudavel", "doenca" ou "descartada" (abaixo do corte)."""
    classe, _, corte = inf.aplicar_decisao(probs, config)
    if corte < inf.limiar_de_corte(config):
        return "descartada"
    return "saudavel" if classe == inf.SAUDAVEL else "doenca"


# ------------------------------------------------------------------------ estatística
def proporcao(sucessos: int, n: int) -> dict:
    """Proporção com o intervalo de Wilson de 95% (n = 0: sem valor)."""
    if not n:
        return {"p": None, "ic95": [None, None], "sucessos": 0, "n": 0}
    return {"p": sucessos / n, "ic95": list(metrics.intervalo_wilson(sucessos / n, n)),
            "sucessos": int(sucessos), "n": int(n)}


def medir(folhas: pd.DataFrame, nomes_regras) -> dict:
    """Nas folhas com y (1 = S, 0 = D): por regra, P(chamar saudável | S e D) e P(descartada |
    S e D); e a AUC de P(saudável) com bootstrap agrupado por foto."""
    s = folhas["y"].to_numpy() == 1
    resultado = {"S": int(s.sum()), "D": int((~s).sum()), "regras": {}}
    resultado["auc"] = (cd.auc_bootstrap(folhas["y"], folhas["p_saudavel"], folhas["foto"])
                        if s.any() and (~s).any() else None)
    for regra in nomes_regras:
        d = folhas[f"desfecho_{regra}"].to_numpy()
        resultado["regras"][regra] = {
            f"{alvo}_dado_{rotulo}": proporcao(int((d[grupo] == alvo).sum()), int(grupo.sum()))
            for alvo in ("saudavel", "descartada", "doenca")
            for rotulo, grupo in (("s", s), ("d", ~s))}
    return resultado


def com_rotulos(folhas: pd.DataFrame, cenario: str) -> pd.DataFrame:
    """As folhas com y pelo cenário: sem os "?", ou com os "?" como S ou como D."""
    if cenario == "sem_interrogacao":
        f = folhas[folhas["rotulo"].isin(["S", "D"])].copy()
        f["y"] = (f["rotulo"] == "S").astype(int)
        return f
    como = "S" if cenario == "interrogacao_como_S" else "D"
    f = folhas.copy()
    f["y"] = (f["rotulo"].replace("?", como) == "S").astype(int)
    return f


def sensibilidade(folhas: pd.DataFrame, limiares=LIMIARES_SENSIBILIDADE) -> list[dict]:
    """Recall de S e falso saudável em D com P(saudável) >= t, para cada t (só informação)."""
    p, s = folhas["p_saudavel"].to_numpy(), folhas["y"].to_numpy() == 1
    return [{"limiar": float(t), "recall_s": proporcao(int((p[s] >= t).sum()), int(s.sum())),
             "falso_saudavel_d": proporcao(int((p[~s] >= t).sum()), int((~s).sum()))}
            for t in limiares]


def tipos_de_folha(folhas: pd.DataFrame) -> dict[str, np.ndarray]:
    """Máscaras dos tipos de folha: os da observação do rotulador e os da cadeia."""
    obs = folhas["observacao"].fillna("").str.lower()
    tipos = {"fragmento (observação)": obs.str.contains("fragmento").to_numpy(),
             "fração visível < 0,5 (cadeia)": (folhas["fracao_visivel"] < 0.5).to_numpy(),
             "borda (cadeia)": folhas["borda"].astype(bool).to_numpy(),
             "brilho (observação)": obs.str.contains("brilho").to_numpy()}
    algum = np.any(list(tipos.values()), axis=0)
    return {**tipos, "nenhum desses": ~algum, "todas": np.ones(len(folhas), bool)}


def interrogacoes_por_tipo(folhas: pd.DataFrame) -> dict:
    """Fração de "?" em cada tipo de folha, com Wilson."""
    duvida = (folhas["rotulo"] == "?").to_numpy()
    return {tipo: proporcao(int((duvida & m).sum()), int(m.sum()))
            for tipo, m in tipos_de_folha(folhas).items()}


# ------------------------------------------------------------------------------- dados
def folhas_do_conjunto(nome: str, modelos) -> tuple[pd.DataFrame, list, int]:
    """As folhas rotuladas de um conjunto (com os "?"), casadas pela caixa com a cadeia da
    rotulagem dele, e os recortes B. Devolve também quantos rótulos não casaram."""
    if nome == "antigas":
        manifest, config = campo.MANIFEST_CAMPO, campo.CONFIG_ROTULAGEM
        _, caminhos = campo.fotos_de_validacao()
    else:
        manifest, config = campo.MANIFEST_CAMPO_TREINO, campo.CONFIG_ROTULAGEM_TREINO
        treino, caminhos = campo.fotos_de_treino()
    rotulos = campo.ler_rotulos(manifest)
    if rotulos is None:
        raise train.ErroFatal(f"{train._exibir(manifest)} não tem rótulos")
    fotos = sorted(set(rotulos["foto"]))
    if nome == "novas" and not set(fotos) <= set(treino):
        raise train.ErroFatal("há fotos da rotulagem de validação fora do treino do detector")
    linhas, recortes = [], []
    for foto, reduzida, escala, folhas in campo.percorrer_folhas(modelos, config,
                                                                 (fotos, caminhos)):
        tabela = pd.DataFrame([{"id": f["id"], **dict(zip(("caixa_x0", "caixa_y0", "caixa_x1",
                                                              "caixa_y1"), f["caixa"]))}
                               for f in folhas])
        por_id = {f["id"]: f for f in folhas}
        for r in campo.casar_rotulos(rotulos[rotulos["foto"] == foto], tabela).itertuples():
            folha = por_id[r.id]
            linhas.append({"conjunto": nome, "id": r.id, "foto": foto, "rotulo": r.rotulo,
                           "observacao": r.observacao, "borda": bool(folha["borda"]),
                           "fracao_visivel": folha["fracao_visivel"]})
            recortes.append(campo.recorte(reduzida, escala, folha, "B", config))
    return pd.DataFrame(linhas), recortes, len(rotulos) - len(linhas)


def conferir_antigas(folhas: pd.DataFrame, caminho: Path = FOLHAS_DIAGNOSTICO) -> float | None:
    """A maior diferença de P(saudável) nas 82 antigas contra o campo_diagnostico (passo 5a)."""
    if not caminho.is_file():
        return None
    antes = pd.read_csv(caminho).set_index("id")["p_saudavel_atual_B"]
    agora = folhas.set_index("id")["p_saudavel"].reindex(antes.index)
    return float((agora - antes).abs().max())


# --------------------------------------------------------------------------- análise
def analisar(folhas: pd.DataFrame, limiar_doenca: float) -> dict:
    """a), b) e c) a partir da tabela de folhas com as probabilidades e os desfechos."""
    nomes_regras = list(regras(limiar_doenca))
    grupos = {"antigas": folhas[folhas["conjunto"] == "antigas"],
              "novas": folhas[folhas["conjunto"] == "novas"], "juntas": folhas}
    resultado = {"limiar_saudavel": LIMIAR_SAUDAVEL, "limiar_doenca_proposto": limiar_doenca,
                 "contagens": {c: g["rotulo"].value_counts().reindex(["S", "D", "?"],
                                                                     fill_value=0).to_dict()
                               for c, g in grupos.items()},
                 "por_conjunto": {}, "sensibilidade": {}, "interrogacoes": {}, "limites": {}}
    for nome, grupo in grupos.items():
        rotuladas = com_rotulos(grupo, "sem_interrogacao")
        resultado["por_conjunto"][nome] = medir(rotuladas, nomes_regras)
        resultado["sensibilidade"][nome] = sensibilidade(rotuladas)
        resultado["interrogacoes"][nome] = interrogacoes_por_tipo(grupo)
        resultado["limites"][nome] = {c: medir(com_rotulos(grupo, c), nomes_regras)
                                      for c in CENARIOS}
    return resultado


def figura_sensibilidade(resultado: dict):
    """Recall de S x falso saudável em D, por t_s, nos três conjuntos (só informação)."""
    import eda_bracol as eda

    cores = dict(zip(CONJUNTOS, ("#2a78d6", "#eb6834", "#1baf7a")))
    with eda._estilo():
        fig = eda._figura(7.6, 4.6, "Limiar de saudável: recall de S x falso saudável em D "
                                    "(só informação)")
        ax = fig.add_subplot()
        for nome in CONJUNTOS:
            pontos = resultado["sensibilidade"][nome]
            x = [p["falso_saudavel_d"]["p"] for p in pontos]
            y = [p["recall_s"]["p"] for p in pontos]
            ax.plot(x, y, color=cores[nome], linewidth=2, marker="o", markersize=4,
                    markeredgecolor=eda.SUPERFICIE, label=NOME_CONJUNTO[nome])
            fixo = next(p for p in pontos if abs(p["limiar"] - LIMIAR_SAUDAVEL) < 1e-9)
            ax.plot(fixo["falso_saudavel_d"]["p"], fixo["recall_s"]["p"], "o", markersize=10,
                    markerfacecolor="none", markeredgecolor=cores[nome], markeredgewidth=1.5)
        ax.set_xlabel("falso saudável em D (fração das D com P(saudável) >= t)")
        ax.set_ylabel("recall de S")
        ax.xaxis.set_major_formatter(lambda v, _: fmt.pct(v, 1, 0))
        ax.yaxis.set_major_formatter(lambda v, _: fmt.pct(v, 1, 0))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
        ax.legend(loc="lower right")
        ax.set_title("pontos: t = " + ", ".join(fmt.decimal(t, 3).rstrip("0").rstrip(",")
                                                for t in LIMIARES_SENSIBILIDADE)
                     + f" (da direita para a esquerda); círculo: t_s = "
                       f"{fmt.decimal(LIMIAR_SAUDAVEL, 3)}", loc="left", fontsize=8)
    return fig


# ------------------------------------------------------------------------- relatório
def _pct(valor, casas: int = 1) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "-"
    return fmt.pct(valor, 1, casas)


def _prop(p: dict | None) -> str:
    if not p or p.get("p") is None:
        return "-"
    return (f"{_pct(p['p'])} ({_pct(p['ic95'][0])} a {_pct(p['ic95'][1])}; {p['sucessos']} de "
            f"{p['n']})")


def _auc(r: dict | None) -> str:
    if not r:
        return "-"
    return (f"{fmt.decimal(r['auc'], 2)} ({fmt.decimal(r['ic95'][0], 2)} a "
            f"{fmt.decimal(r['ic95'][1], 2)})")


def relatorio(resultado: dict, leitura: str | None) -> str:
    t_s, t_d = resultado["limiar_saudavel"], resultado["limiar_doenca_proposto"]
    cont = resultado["contagens"]
    linhas = [
        "# Validação de campo da regra em duas etapas (passo 5b)", "",
        f"Gerado por `python evaluation/campo_validacao.py`. Modelo congelado "
        "(`base_resnet50_bracol`), recorte B, regra já fixada: **limiar de saudável "
        f"{fmt.decimal(t_s, 3)}** (decisão do gestor de 11/10/2026, tirado das 82 folhas antigas) "
        f"e limiar de doença {fmt.decimal(t_d, 2)} **só como proposta** (run "
        "`pipeline_duas_etapas`). Nada foi reajustado com as folhas novas. Rótulos "
        f"{campo.ORIGEM_ROTULOS}. Testes do BRACOL e do BRACOT fechados; nenhum treino.", "",
        f"- **Antigas:** {sum(cont['antigas'].values())} folhas das fotos de validação do "
        f"detector ({cont['antigas']['S']} S, {cont['antigas']['D']} D, "
        f"{cont['antigas']['?']} \"?\"). O t_s saiu destas folhas: nelas a medida é otimista.",
        f"- **Novas:** {sum(cont['novas'].values())} folhas, uma por foto, das fotos de treino do "
        f"detector ({cont['novas']['S']} S, {cont['novas']['D']} D, {cont['novas']['?']} "
        "\"?\"). É a validação de verdade da regra. O detector treinou nessas fotos, então as "
        "máscaras saem mais limpas que as de fotos novas; mesma lavoura e mesmas 2 sessões das "
        "antigas, em cenas diferentes.",
        f"- **Desfecho de cada folha:** \"saudável\", \"doença\" ou \"descartada\" (abaixo do "
        "corte). Na regra em duas etapas, chamar saudável é P(saudável) >= "
        f"{fmt.decimal(t_s, 3)} e não depende do limiar de doença.", ""]
    if resultado.get("conferencia_antigas") is not None:
        linhas += [f"Conferência: nas 82 antigas, P(saudável) difere no máximo "
                   f"{resultado['conferencia_antigas']:.1e} do `campo_diagnostico` (passo 5a). "
                   f"Rótulos não casados pela caixa: {resultado['nao_casados']}.", ""]
    # a)
    novas = resultado["por_conjunto"]["novas"]
    linhas += ["## a) Folhas novas: a regra fixada contra o argmax", "",
               f"AUC de P(saudável), S x D: {_auc(novas['auc'])} (bootstrap agrupado por foto; "
               "aqui uma folha por foto).", ""]
    linhas += fmt.tabela(
        ["regra", "chamar saudável dado S", "chamar saudável dado D", "descartada dado S",
         "descartada dado D"],
        [[NOME_REGRA[r], _prop(m["saudavel_dado_s"]), _prop(m["saudavel_dado_d"]),
          _prop(m["descartada_dado_s"]), _prop(m["descartada_dado_d"])]
         for r, m in novas["regras"].items()], "lllll")
    # b)
    linhas += ["", "## b) Antigas, novas e juntas", ""]
    linhas += fmt.tabela(
        ["conjunto", "S / D", "AUC de P(saudável) (IC 95%)", "regra",
         "chamar saudável dado S", "chamar saudável dado D"],
        [[NOME_CONJUNTO[c], f"{r['S']} / {r['D']}", _auc(r["auc"]), NOME_REGRA[regra],
          _prop(m["saudavel_dado_s"]), _prop(m["saudavel_dado_d"])]
         for c, r in resultado["por_conjunto"].items() for regra, m in r["regras"].items()],
        "lllll" + "l")
    linhas += ["", "### Sensibilidade do limiar de saudável (só informação, nenhum limiar novo)",
               "", "![Recall de S x falso saudável em D](sensibilidade.png)", ""]
    linhas += fmt.tabela(
        ["t", *(f"{NOME_CONJUNTO[c]}: recall de S / falso saudável em D" for c in CONJUNTOS)],
        [[fmt.decimal(p["limiar"], 3),
          *(f"{_pct(resultado['sensibilidade'][c][k]['recall_s']['p'])} / "
            f"{_pct(resultado['sensibilidade'][c][k]['falso_saudavel_d']['p'])}"
            for c in CONJUNTOS)]
         for k, p in enumerate(resultado["sensibilidade"]["novas"])], "llll")
    # c)
    linhas += ["", "## c) Qualidade da rotulagem", "",
               "Fração de \"?\" por tipo de folha. \"Fragmento\" e \"brilho\" vêm da observação "
               "escrita ao rotular (em parte, ela explica o \"?\"); \"borda\" e a fração visível "
               "vêm da cadeia. Nas novas, o filtro de fragmentos já estava ligado.", ""]
    linhas += fmt.tabela(
        ["tipo de folha", *(NOME_CONJUNTO[c] for c in CONJUNTOS)],
        [[tipo, *(_prop(resultado["interrogacoes"][c][tipo]) for c in CONJUNTOS)]
         for tipo in TIPOS], "llll")
    linhas += ["", "Limites com os \"?\" tratados como S ou como D (chamar saudável e AUC):", ""]
    linhas += fmt.tabela(
        ["conjunto", "cenário", "S / D", "AUC", "duas etapas: saudável dado S",
         "duas etapas: saudável dado D", "argmax 0,61: saudável dado S",
         "argmax 0,61: saudável dado D"],
        [[NOME_CONJUNTO[c], NOME_CENARIO[cen], f"{r['S']} / {r['D']}", _auc(r["auc"]),
          _prop(r["regras"]["duas_etapas"]["saudavel_dado_s"]),
          _prop(r["regras"]["duas_etapas"]["saudavel_dado_d"]),
          _prop(r["regras"]["argmax_061"]["saudavel_dado_s"]),
          _prop(r["regras"]["argmax_061"]["saudavel_dado_d"])]
         for c in CONJUNTOS for cen, r in resultado["limites"][c].items()], "llllllll")
    linhas += ["", "## d) Leitura e recomendação", "",
               leitura or "Ainda não escrito (falta o leitura.md).", ""]
    return "\n".join(linhas) + "\n"


def gravar_relatorio(resultado: dict) -> None:
    leitura = PASTA_SAIDA / "leitura.md"
    (PASTA_SAIDA / "relatorio.md").write_text(
        relatorio(resultado, leitura.read_text(encoding="utf-8").strip()
                  if leitura.is_file() else None), encoding="utf-8", newline="\n")


# -------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Validacao de campo da regra em duas etapas.")
    parser.add_argument("--relatorio", action="store_true",
                        help="so refaz o relatorio.md a partir do validacao.json")
    args = parser.parse_args()
    try:
        if args.relatorio:
            caminho = PASTA_SAIDA / "validacao.json"
            if not caminho.is_file():
                raise train.ErroFatal(f"{caminho} não existe: rode sem --relatorio antes")
            gravar_relatorio(json.loads(caminho.read_text(encoding="utf-8")))
            fmt.dizer(f"Relatorio refeito: {PASTA_SAIDA / 'relatorio.md'}")
            return 0
        limiar_doenca = limiar_doenca_proposto()
        modelos = inf.carregar_modelos(dispositivo=cd._dispositivo())
        tabelas, nao_casados = [], 0
        for nome in ("antigas", "novas"):
            fmt.dizer(f"Folhas {nome}: cadeia, recortes B e classificador...")
            folhas, recortes, faltam = folhas_do_conjunto(nome, modelos)
            nao_casados += faltam
            probabilidades, _ = campo._classificar_em_lotes(recortes, modelos)
            for k, classe in enumerate(inf.CATEGORIAS):
                folhas[f"p_{classe}"] = probabilidades[:, k]
            folhas["p_saudavel"] = probabilidades[:, inf.SAUDAVEL]
            for regra, config in regras(limiar_doenca).items():
                folhas[f"desfecho_{regra}"] = [desfecho(p, config) for p in probabilidades]
            tabelas.append(folhas)
        folhas = pd.concat(tabelas, ignore_index=True)
        resultado = analisar(folhas, limiar_doenca)
        resultado["conferencia_antigas"] = conferir_antigas(
            folhas[(folhas["conjunto"] == "antigas") & folhas["rotulo"].isin(["S", "D"])])
        resultado["nao_casados"] = nao_casados
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        folhas.drop(columns=["p_saudavel"]).to_csv(PASTA_SAIDA / "folhas.csv", index=False,
                                                   float_format="%.6f", lineterminator="\n")
        train.gravar_json(campo._nativo(resultado), PASTA_SAIDA / "validacao.json")
        metrics.salvar_figura(figura_sensibilidade(resultado), PASTA_SAIDA / "sensibilidade.png")
        gravar_relatorio(resultado)
        for nome in CONJUNTOS:
            r = resultado["por_conjunto"][nome]
            m = r["regras"]["duas_etapas"]
            fmt.dizer(f"{nome}: S {r['S']} D {r['D']} | duas etapas: saudavel|S "
                      f"{_pct(m['saudavel_dado_s']['p'])} saudavel|D "
                      f"{_pct(m['saudavel_dado_d']['p'])} | AUC {r['auc']['auc']:.2f}")
        return 0
    except (train.ErroFatal, FileNotFoundError, ValueError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
