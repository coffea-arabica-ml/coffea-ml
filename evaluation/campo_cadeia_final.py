"""
Números de campo da cadeia final (Frente 9, passo 6b), antes de abrir os testes, sem treino e sem
mudar nenhum limiar nem padrão.

A cadeia é a final (inferencia.Configuracao(): regra por caminho, recorte B, filtro de fragmentos
em 0,5), aplicada inteira a cada foto rotulada: detector, filtros, regras de folha única e regra
de decisão do caminho. As folhas são só as rotuladas que já existem (nada do teste):
- antigas: as 120 do campo rotulado (fotos de validação do detector; 36 S, 46 D e 38 "?"),
  rotuladas com o filtro de fragmentos desligado (campo.CONFIG_ROTULAGEM), então a cadeia final
  pode tirar algumas;
- novas: as 160 da rotulagem de validação (fotos de treino do detector; 41 S, 60 D e 59 "?"),
  rotuladas já com o filtro ligado (campo.CONFIG_ROTULAGEM_TREINO).
Cada rótulo casa com a folha da cadeia da rotulagem pelo id, conferido pela caixa
(campo.casar_rotulos); depois, a folha é procurada entre as da cadeia final, na mesma detecção.

a) Para S, D e "?": chamada saudável, chamada doente e mantida (com as classes) e descartada,
   pelo motivo (filtro de fragmentos, corte t_d ou outro), com Wilson; antigas, novas e juntas.
b) Nas D evidentes: quantas chegam ao usuário como doente, como saudável ou somem; e o mesmo com
   o corte t_d desligado no caminho de detecção (só informação: nenhum padrão muda).
c) Classes e confiança das folhas doentes mantidas.

Saída em model/runs/campo_cadeia_final/: relatorio.md, campo.json e folhas.csv, com a leitura
escrita à mão em leitura.md. Os rótulos foram feitos por Claude por leitura visual e aceitos pelo
gestor em 11/10/2026 (não são de especialista); nenhuma D tem classe. Os testes do BRACOL e do
BRACOT ficam fechados.

Uso (de qualquer pasta, com o venv ativo):
    python evaluation/campo_cadeia_final.py              # roda tudo (alguns minutos na GPU)
    python evaluation/campo_cadeia_final.py --relatorio  # só refaz o relatorio.md do JSON
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "model", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import campo  # noqa: E402
import campo_diagnostico as cd  # noqa: E402
import campo_validacao as cv  # noqa: E402
import dados  # noqa: E402
import formatacao as fmt  # noqa: E402
import inferencia as inf  # noqa: E402
import train  # noqa: E402

PASTA_SAIDA = train.PASTA_RUNS / "campo_cadeia_final"
FOLHAS_VALIDACAO = train.PASTA_RUNS / "campo_validacao_duas_etapas" / "folhas.csv"
CONJUNTOS = ("antigas", "novas", "juntas")
NOME_CONJUNTO = {"antigas": "antigas (fotos de validação do detector)",
                 "novas": "novas (fotos de treino do detector)", "juntas": "juntas"}
ROTULOS = ("S", "D", "?")
DESFECHOS = ("saudavel", "doenca", "descartada")
MOTIVOS = ("fragmentos", "corte_td", "outro")
NOME_MOTIVO = {"fragmentos": "filtro de fragmentos", "corte_td": "corte t_d",
               "outro": "outro motivo"}
CENARIOS = ("final", "sem_td")
NOME_CENARIO = {"final": "cadeia final (t_d = 0,65)",
                "sem_td": "t_d desligado na detecção (só informação)"}
TITULO_CENARIO = {"final": "Cadeia final (t_d = 0,65)",
                  "sem_td": "Com o corte t_d desligado na detecção (só informação)"}
DOENCAS = tuple(c for c in inf.CATEGORIAS if c != "saudavel")


# --------------------------------------------------------------------------- regras
def configuracoes() -> dict:
    """A cadeia final (os padrões, sem mudar nada) e, só para informação, a mesma com o corte
    t_d desligado: o limiar de doença em 0 deixa passar toda folha da 2ª etapa."""
    final = inf.Configuracao()
    return {"final": final, "sem_td": replace(final, limiar_doenca=0.0)}


def _mesma(folha: dict, outra: dict) -> bool:
    """A mesma detecção (a caixa em pixels, que vem do mesmo detector)."""
    return (folha.get("caixa") is not None and outra.get("caixa") is not None
            and [float(v) for v in folha["caixa"]] == [float(v) for v in outra["caixa"]])


def destino(folha: dict, deteccoes, preparo: dict, config) -> tuple[str, str | None]:
    """O que a cadeia final faz com uma folha que a cadeia da rotulagem classificava:
    ("classificada", None), se ela entra na regra de decisão; ("fragmentos", ...), se o filtro de
    fragmentos a tirou; ou ("outro", detalhe), por exemplo o limite de folhas por foto."""
    if any(_mesma(folha, f) for f in preparo["folhas"]):
        return "classificada", None
    coloridas = [d for d in deteccoes if inf.utilizavel(d, config)]
    if preparo["plano"] == "fragmentos":
        return "fragmentos", "o filtro tirou todas as folhas da foto"
    if inf.fragmento(folha, coloridas, config):
        return "fragmentos", "fração visível abaixo do mínimo"
    return "outro", "fora das max_folhas de maior score" if preparo["plano"] == "deteccao" else (
        f"plano {preparo['plano']}")


def decidir_folha(probs, plano: str, config) -> dict:
    """A regra do caminho (inferencia.config_do_caminho) numa folha classificada: desfecho
    ("saudavel", "doenca" ou "descartada"), motivo do descarte ("corte_td" nas duas etapas;
    "outro" no corte de 0,61 da foto inteira), classe prevista e confiança do contrato."""
    regra = inf.config_do_caminho(config, plano)
    classe, confianca, corte = inf.aplicar_decisao(probs, regra)
    saida = {"classe": inf.CATEGORIAS[classe], "confianca": confianca, "regra": regra.decisao,
             "motivo": None, "detalhe": None}
    if corte < inf.limiar_de_corte(regra):
        duas = regra.decisao == "duas_etapas"
        return {**saida, "desfecho": "descartada", "motivo": "corte_td" if duas else "outro",
                "detalhe": None if duas else "corte de 0,61 (argmax)"}
    return {**saida, "desfecho": "saudavel" if classe == inf.SAUDAVEL else "doenca"}


# --------------------------------------------------------------------------- cadeia
def fontes(nome: str):
    """O CSV dos rótulos, a configuração da rotulagem e as fotos (ids e caminhos) do conjunto."""
    if nome == "antigas":
        return campo.MANIFEST_CAMPO, campo.CONFIG_ROTULAGEM, campo.fotos_de_validacao()
    return campo.MANIFEST_CAMPO_TREINO, campo.CONFIG_ROTULAGEM_TREINO, campo.fotos_de_treino()


def folhas_do_conjunto(nome: str, modelos, configs: dict,
                       fontes_do_conjunto=None) -> tuple[pd.DataFrame, int, dict]:
    """As folhas rotuladas de um conjunto pela cadeia final: uma linha por folha casada, com o
    plano da foto, as probabilidades (se classificada) e o desfecho em cada cenário. Devolve
    também quantos rótulos não casaram e o plano final de cada foto. fontes_do_conjunto: (CSV
    dos rótulos, configuração da rotulagem, (ids, caminhos) das fotos) de outro conjunto; sem
    ela, as do conjunto pelo nome (fontes)."""
    manifest, config_rotulagem, (ids, caminhos) = fontes_do_conjunto or fontes(nome)
    rotulos = campo.ler_rotulos(manifest)
    if rotulos is None:
        raise train.ErroFatal(f"{train._exibir(manifest)} não tem rótulos")
    fotos = sorted(set(rotulos["foto"]))
    if not set(fotos) <= set(ids):
        raise train.ErroFatal(f"há fotos de {nome} fora das fotos esperadas do BRACOT")
    final = configs["final"]
    linhas, planos = [], {}
    for foto in fotos:
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        altura, largura = imagem.shape[:2]
        deteccoes = inf.detectar(imagem, final, modelos)  # a detecção não depende do filtro
        rotulagem = inf.decidir(deteccoes, altura, largura, config_rotulagem)
        preparo = inf.decidir(deteccoes, altura, largura, final)
        planos[foto] = preparo["plano"]
        if rotulagem["plano"] != "deteccao":
            continue
        por_id = {f"{foto}_{k:02d}": folha for k, folha in enumerate(rotulagem["folhas"], 1)}
        tabela = pd.DataFrame([{"id": i, **dict(zip(
            ("caixa_x0", "caixa_y0", "caixa_x1", "caixa_y1"),
            inf.caixa_relativa(f["caixa"], altura, largura)))} for i, f in por_id.items()])
        casados = campo.casar_rotulos(rotulos[rotulos["foto"] == foto], tabela)
        rotuladas = [(r, por_id[r.id]) for r in casados.itertuples()]
        destinos = [destino(folha, deteccoes, preparo, final) for _, folha in rotuladas]
        probs = _classificar(imagem, preparo, [f for (_, f), (d, _) in zip(rotuladas, destinos)
                                                if d == "classificada"], final, modelos)
        for (r, folha), (dest, detalhe) in zip(rotuladas, destinos):
            linha = {"conjunto": nome, "id": r.id, "foto": foto, "rotulo": r.rotulo,
                     "observacao": r.observacao, "borda": bool(folha["borda"]),
                     "plano_final": preparo["plano"], "destino": dest}
            p = probs.get(_chave(folha))
            for k, classe in enumerate(inf.CATEGORIAS):
                linha[f"p_{classe}"] = None if p is None else float(p[k])
            for cenario, config in configs.items():
                saida = (decidir_folha(p, preparo["plano"], config) if dest == "classificada"
                         else {"desfecho": "descartada", "motivo": dest, "detalhe": detalhe,
                               "classe": None, "confianca": None, "regra": None})
                linha.update({f"{chave}_{cenario}": valor for chave, valor in saida.items()})
            linhas.append(linha)
    return pd.DataFrame(linhas), len(rotulos) - len(linhas), planos


def _chave(folha: dict) -> tuple:
    return tuple(float(v) for v in folha["caixa"])


def _classificar(imagem, preparo: dict, folhas, config, modelos) -> dict:
    """As probabilidades de cada folha classificada, pela chave da caixa: os recortes da cadeia
    (inferencia.recortar; na foto inteira, a foto toda) num lote só."""
    if not folhas:
        return {}
    if preparo["plano"] == "foto_inteira":
        probabilidades, _ = inf.classificar(inf.recortar(imagem, preparo, config), modelos)
        return {_chave(folhas[0]): probabilidades[0]}
    probabilidades, _ = inf.classificar(inf.recortar(imagem, {**preparo, "folhas": folhas},
                                                     config), modelos)
    return {_chave(f): p for f, p in zip(folhas, probabilidades)}


# ----------------------------------------------------------------------- estatística
def estatisticas(valores) -> dict | None:
    v = np.asarray([x for x in valores if x is not None and not pd.isna(x)], dtype=np.float64)
    if not v.size:
        return None
    return {"n": int(v.size), "media": float(v.mean()), "mediana": float(np.median(v)),
            "p25": float(np.quantile(v, 0.25)), "p75": float(np.quantile(v, 0.75)),
            "min": float(v.min()), "max": float(v.max())}


def resumo_por_rotulo(folhas: pd.DataFrame, cenario: str) -> dict:
    """a) Por rótulo: chamada saudável, doente e mantida e descartada (com o motivo), com Wilson
    sobre todas as folhas do rótulo, e as classes das doentes mantidas."""
    resultado = {}
    for rotulo in ROTULOS:
        g = folhas[folhas["rotulo"] == rotulo]
        n = len(g)
        d, m = g[f"desfecho_{cenario}"], g[f"motivo_{cenario}"]
        doentes = g[d == "doenca"]
        resultado[rotulo] = {
            "n": n, **{x: cv.proporcao(int((d == x).sum()), n) for x in DESFECHOS},
            "motivos": {mot: cv.proporcao(int(((d == "descartada") & (m == mot)).sum()), n)
                        for mot in MOTIVOS},
            "classes_doentes": {c: int((doentes[f"classe_{cenario}"] == c).sum())
                                for c in DOENCAS}}
    return resultado


def doentes_mantidas(folhas: pd.DataFrame, cenario: str) -> dict:
    """c) Por grupo (S, D, "?" e todas): as folhas mantidas como doentes, com as classes (Wilson
    sobre as mantidas) e a confiança do contrato, no todo e por classe."""
    resultado = {}
    for grupo in (*ROTULOS, "todas"):
        g = folhas if grupo == "todas" else folhas[folhas["rotulo"] == grupo]
        m = g[g[f"desfecho_{cenario}"] == "doenca"]
        classes = m[f"classe_{cenario}"]
        resultado[grupo] = {
            "n": len(m),
            "classes": {c: cv.proporcao(int((classes == c).sum()), len(m)) for c in DOENCAS},
            "confianca": estatisticas(m[f"confianca_{cenario}"]),
            "confianca_por_classe": {c: estatisticas(m.loc[classes == c, f"confianca_{cenario}"])
                                     for c in DOENCAS}}
    return resultado


def conferir(folhas: pd.DataFrame, caminho: Path = FOLHAS_VALIDACAO) -> dict | None:
    """As folhas classificadas pela detecção contra a validação da regra em duas etapas
    (campo_validacao_duas_etapas, mesmos recortes B e mesmos t_s e t_d): a maior diferença das
    probabilidades e as folhas com desfecho diferente."""
    if not Path(caminho).is_file():
        return None
    antes = pd.read_csv(caminho, dtype={"id": str}).set_index(["conjunto", "id"])
    f = folhas[(folhas["destino"] == "classificada") & (folhas["plano_final"] == "deteccao")]
    f = f.set_index(["conjunto", "id"])
    comuns = f.index.intersection(antes.index)
    colunas = [f"p_{c}" for c in DOENCAS]
    diferenca = np.abs(f.loc[comuns, colunas].to_numpy(dtype=np.float64)
                       - antes.loc[comuns, colunas].to_numpy(dtype=np.float64))
    diferentes = (f.loc[comuns, "desfecho_final"] != antes.loc[comuns, "desfecho_duas_etapas"])
    return {"folhas": len(comuns), "sem_par": int(len(f) - len(comuns)),
            "diferenca_maxima": float(diferenca.max()) if diferenca.size else None,
            "desfechos_diferentes": int(diferentes.sum())}


def analisar(folhas: pd.DataFrame) -> dict:
    resultado = {}
    for conjunto in CONJUNTOS:
        f = folhas if conjunto == "juntas" else folhas[folhas["conjunto"] == conjunto]
        resultado[conjunto] = {
            "contagens": {r: int((f["rotulo"] == r).sum()) for r in ROTULOS},
            "destinos": f["destino"].value_counts().to_dict(),
            **{cenario: {"rotulos": resumo_por_rotulo(f, cenario),
                         "doentes_mantidas": doentes_mantidas(f, cenario)}
               for cenario in CENARIOS}}
    return resultado


# ------------------------------------------------------------------------- relatório
def _pct(valor) -> str:
    return cv._pct(valor)


def _prop(p: dict | None) -> str:
    return cv._prop(p)


def _por_conjunto(valores: dict) -> str:
    return ", ".join(f"{conj} {v}" for conj, v in valores.items())


def _classes(contagem: dict) -> str:
    return ", ".join(f"{c} {k}" for c, k in contagem.items() if k) or "-"


def _conf(e: dict | None) -> str:
    if not e:
        return "-"
    return (f"{fmt.decimal(e['mediana'], 2)} ({fmt.decimal(e['p25'], 2)} a "
            f"{fmt.decimal(e['p75'], 2)}; mín. {fmt.decimal(e['min'], 2)}, máx. "
            f"{fmt.decimal(e['max'], 2)})")


def relatorio(resultado: dict, leitura: str | None) -> str:
    c = resultado["configuracao"]
    por = resultado["por_conjunto"]
    confere = resultado["conferencia"]
    linhas = [
        "# Campo pela cadeia final (passo 6b)", "",
        f"Gerado por `python evaluation/campo_cadeia_final.py` em {resultado['criado_em']}, "
        f"{resultado['origem']}. Classificador `{resultado['modelos']['classificador']}`, "
        f"detector `{resultado['modelos']['detector']}`, dispositivo "
        f"{resultado['modelos']['dispositivo']}. **Nenhum limiar nem padrão mudou; nada do teste "
        "do BRACOL nem do BRACOT foi lido.**", "",
        "A cadeia final inteira (`inferencia.Configuracao()`) roda em cada foto rotulada: o "
        "detector, os filtros (fragmentos em "
        f"{fmt.decimal(c['fracao_visivel_minima'], 1)}), as regras de folha única e a regra do "
        f"caminho. Nos recortes por detecção (modo {c['recorte']}), as duas etapas: saudável com "
        f"P(saudável) >= {fmt.decimal(c['limiar_saudavel'], 3)}; senão, a doença com P "
        f"renormalizada, cortada em t_d = {fmt.decimal(c['limiar_doenca'], 2)}. Na foto inteira, "
        f"o argmax com corte {fmt.decimal(c['limiar_confianca'], 2)}. A `confianca` das doentes "
        "é a P renormalizada entre as 4 doenças (a do contrato).", "",
        "**As folhas** são só as rotuladas que já existem: as 120 antigas (rotuladas com o filtro "
        "de fragmentos desligado, então a cadeia final pode tirar algumas) e as 160 novas "
        "(rotuladas já com o filtro ligado). Cada rótulo casa com a sua folha pela caixa "
        "(IoU >= 0,5), e a folha é procurada na cadeia final, na mesma detecção. Rótulos sem "
        f"par: {_por_conjunto(resultado['nao_casados'])}.", "",
        "**Planos das fotos na cadeia final:** " + "; ".join(
            f"{conj}: " + ", ".join(f"{plano} {k}" for plano, k in planos.items())
            for conj, planos in resultado["planos"].items()) + ".", ""]
    if confere:
        linhas += [
            f"**Conferência.** Nas {confere['folhas']} folhas classificadas pela detecção, contra "
            "`campo_validacao_duas_etapas` (mesmos recortes B, t_s e t_d): diferença máxima das "
            f"probabilidades {confere['diferenca_maxima']:.1e}; desfechos diferentes: "
            f"{confere['desfechos_diferentes']}; folhas sem par: {confere['sem_par']}.", ""]
    # a)
    linhas += ["## a) S, D e \"?\": o que a cadeia final faz com cada folha", "",
               "Proporções sobre todas as folhas do rótulo, com Wilson de 95%. Descartada: a "
               "folha não chega ao usuário (se nenhuma folha da foto chega, a foto vira "
               "`baixa_confianca`).", ""]
    linhas += fmt.tabela(
        ["conjunto", "rótulo", "folhas", "chamada saudável", "chamada doente e mantida",
         "classes das doentes mantidas", "descartada", *(f"descartada: {NOME_MOTIVO[m]}"
                                                         for m in MOTIVOS)],
        [[NOME_CONJUNTO[conj], rot, str(r["n"]), _prop(r["saudavel"]), _prop(r["doenca"]),
          _classes(r["classes_doentes"]), _prop(r["descartada"]),
          *(_prop(r["motivos"][m]) for m in MOTIVOS)]
         for conj in CONJUNTOS for rot, r in por[conj]["final"]["rotulos"].items()],
        "llrlllllll")
    outros = resultado.get("detalhes_outro") or {}
    linhas += ["", "\"Outro motivo\": " + (", ".join(f"{k} ({v})" for k, v in outros.items())
                                           if outros else "nenhuma folha") + ".", ""]
    # b)
    linhas += ["## b) As D evidentes: o que chega ao usuário", "",
               "Com o corte t_d desligado no caminho de detecção (limiar de doença em 0) só para "
               "informação: nenhum padrão mudou. O filtro de fragmentos e a 1ª etapa não mudam.",
               ""]
    linhas += fmt.tabela(
        ["conjunto", "cenário", "D", "chegam como doente", "chegam como saudável", "somem",
         "somem: filtro de fragmentos", "somem: corte t_d", "somem: outro motivo"],
        [[NOME_CONJUNTO[conj], NOME_CENARIO[cen], str(r["n"]), _prop(r["doenca"]),
          _prop(r["saudavel"]), _prop(r["descartada"]),
          *(_prop(r["motivos"][m]) for m in MOTIVOS)]
         for conj in CONJUNTOS for cen in CENARIOS for r in (por[conj][cen]["rotulos"]["D"],)],
        "llrllllll")
    # c)
    linhas += ["", "## c) Folhas doentes mantidas: classes e confiança", "",
               "As classes são as previstas: **nenhuma D tem classe nos rótulos, então a classe "
               "não é verificada.** Confiança: mediana (p25 a p75; mínimo e máximo).", ""]
    for cen in CENARIOS:
        linhas += [f"**{TITULO_CENARIO[cen]}.**", ""]
        linhas += fmt.tabela(
            ["conjunto", "rótulo", "doentes mantidas", *DOENCAS, "confiança"],
            [[NOME_CONJUNTO[conj], grupo, str(r["n"]),
              *(_prop(r["classes"][cl]) if r["n"] else "-" for cl in DOENCAS),
              _conf(r["confianca"])]
             for conj in CONJUNTOS for grupo, r in por[conj][cen]["doentes_mantidas"].items()],
            "llrlllll")
        linhas += [""]
    juntas = por["juntas"]["final"]["doentes_mantidas"]["todas"]
    linhas += ["Confiança por classe prevista (juntas, cadeia final, todas as folhas mantidas "
               "como doentes):", ""]
    linhas += fmt.tabela(["classe", "folhas", "confiança"],
                         [[cl, str((juntas["confianca_por_classe"][cl] or {"n": 0})["n"]),
                           _conf(juntas["confianca_por_classe"][cl])] for cl in DOENCAS], "lrl")
    linhas += ["", "## Leitura", "", leitura or "Ainda não escrito (falta o leitura.md).", "",
               "## Limites", "",
               "- Rótulos feitos por Claude por leitura visual, aceitos pelo gestor em "
               "11/10/2026, sem especialista; 1 lavoura e 2 sessões.",
               "- Nenhuma D tem classe: a distribuição das classes descreve o que a cadeia "
               "responde, não se ela acerta. A severidade não é informativa no campo.",
               "- As novas vêm de fotos em que o detector treinou: as máscaras saem mais limpas "
               "que as de fotos novas.",
               "- As folhas foram sorteadas da cadeia da rotulagem de cada conjunto, não da "
               "cadeia final: nas antigas, entram fragmentos que a cadeia final não "
               "classificaria (o filtro estava desligado na rotulagem).",
               "- O cenário sem t_d é só informação: nenhum limiar foi ajustado com estes "
               "números.", ""]
    return "\n".join(linhas) + "\n"


def gravar_relatorio(resultado: dict) -> None:
    leitura = PASTA_SAIDA / "leitura.md"
    (PASTA_SAIDA / "relatorio.md").write_text(
        relatorio(resultado, leitura.read_text(encoding="utf-8").strip()
                  if leitura.is_file() else None), encoding="utf-8", newline="\n")


def _origem() -> str:
    commit, alterado = train.estado_git()
    if not commit:
        return "fora de um repositório git"
    return f"no commit `{commit[:7]}`" + (" (com alterações não commitadas)" if alterado else "")


# -------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Numeros de campo da cadeia final (passo 6b).")
    parser.add_argument("--relatorio", action="store_true",
                        help="so refaz o relatorio.md a partir do campo.json")
    args = parser.parse_args(argv)
    try:
        if args.relatorio:
            caminho = PASTA_SAIDA / "campo.json"
            if not caminho.is_file():
                raise train.ErroFatal(f"{caminho} não existe: rode sem --relatorio antes")
            gravar_relatorio(json.loads(caminho.read_text(encoding="utf-8")))
            fmt.dizer(f"Relatorio refeito: {PASTA_SAIDA / 'relatorio.md'}")
            return 0
        if PASTA_SAIDA.exists() and any(PASTA_SAIDA.iterdir()):
            raise train.ErroFatal(f"{train._exibir(PASTA_SAIDA)} já existe: não sobrescrevo")
        configs = configuracoes()
        dispositivo = cd._dispositivo()
        modelos = inf.carregar_modelos(dispositivo=dispositivo)
        tabelas, nao_casados, planos = [], {}, {}
        for nome in ("antigas", "novas"):
            fmt.dizer(f"Folhas {nome}: cadeia final...")
            folhas, nao_casados[nome], planos_fotos = folhas_do_conjunto(nome, modelos, configs)
            planos[nome] = pd.Series(planos_fotos).value_counts().to_dict()
            tabelas.append(folhas)
        folhas = pd.concat(tabelas, ignore_index=True)
        outro = folhas.loc[folhas["motivo_final"] == "outro", "detalhe_final"]
        resultado = {
            "criado_em": datetime.now().strftime("%d/%m/%Y %H:%M"), "origem": _origem(),
            "configuracao": {k: getattr(configs["final"], k) for k in (
                "decisao", "recorte", "fracao_visivel_minima", "limiar_saudavel",
                "limiar_doenca", "limiar_confianca", "max_folhas")},
            "modelos": {"classificador": modelos.nome_classificador,
                        "detector": modelos.nome_detector, "dispositivo": dispositivo},
            "nao_casados": nao_casados, "planos": planos,
            "detalhes_outro": outro.value_counts().to_dict(),
            "conferencia": conferir(folhas), "por_conjunto": analisar(folhas)}
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        folhas.to_csv(PASTA_SAIDA / "folhas.csv", index=False, float_format="%.6f",
                      lineterminator="\n")
        train.gravar_json(campo._nativo(resultado), PASTA_SAIDA / "campo.json")
        gravar_relatorio(resultado)
        for nome in CONJUNTOS:
            for cenario in CENARIOS:
                d = resultado["por_conjunto"][nome][cenario]["rotulos"]["D"]
                fmt.dizer(f"{nome} {cenario}: D {d['n']} | doente {_pct(d['doenca']['p'])} "
                          f"saudavel {_pct(d['saudavel']['p'])} somem "
                          f"{_pct(d['descartada']['p'])}")
        fmt.dizer(f"Conferencia: {resultado['conferencia']}; nao casados: {nao_casados}")
        return 0
    except (train.ErroFatal, FileNotFoundError, ValueError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
