"""
Avaliação ponta a ponta da inferência (Frente 9, passo 3), gravada em model/runs/<nome>/:

- Constantes medidas no TREINO do BRACOL: o cinza do fundo (borda das fotos), a margem do
  recorte (comprimento da folha sobre a largura da foto) e a cobertura da caixa nas fotos de
  folha única; e, no treino do BRACOT (sem a validação), a cobertura da maior caixa.
- Calibração do limiar de confiança (RF07): curva risco-cobertura com as predições fora da dobra
  do diagnóstico por blocos (protocolo aleatório) e com a validação do classificador avaliado.
- T1: as imagens de validação do BRACOL pela cadeia (recortes A e B) x a imagem inteira.
- T2: as fotos de validação do BRACOT (cenas inteiras do treino dos autores): classes,
  severidades e confiança por folha, nos recortes A e B; folhas cortadas pela borda x inteiras.
- T3: tempo da cadeia em CPU (RNF02).
- Com --decisao duas_etapas (passo 5b): a curva risco-cobertura do limiar de doença nas doentes
  fora da dobra, com a proposta; o efeito do limiar de saudável no BRACOL; a regra completa
  contra o argmax; e a sensibilidade do limiar de saudável no campo rotulado.
- Com a regra por caminho (o padrão da cadeia final, passo 6): o T1 e o T2 por caminho (foto
  inteira x recortes por detecção), nas três regras (argmax, duas etapas e por caminho), com as
  mesmas probabilidades; a comparação com os runs pipeline_padrao_b e pipeline_duas_etapas; e as
  limitações de campo, lidas de model/runs/campo_validacao_duas_etapas/validacao.json.

Os testes do BRACOL e do BRACOT ficam fechados: nada aqui lê o split de teste do BRACOL nem
as anotações de teste do BRACOT.

Uso (de qualquer pasta, com o venv ativo):
    python model/avaliar_pipeline.py                    tudo, com a cadeia final (regra por
                                                        caminho), em model/runs/pipeline_final/
    python model/avaliar_pipeline.py --decisao argmax   os padrões até o passo 5b, em
                                                        model/runs/pipeline_padrao_b/
    python model/avaliar_pipeline.py --so-constantes    só mede e mostra as constantes
    python model/avaliar_pipeline.py --relatorio --nome NOME
                                                        só refaz o relatorio.md do run
    python model/avaliar_pipeline.py --classificador PASTA --detector PESOS --nome NOME
                                                        outros modelos (recalibrar o limiar)
    python model/avaliar_pipeline.py --recortes PASTA   grava também os recortes do T2 e duas
                                                        pranchas em PASTA, fora do repositório
    python model/avaliar_pipeline.py --decisao duas_etapas
                                                        a regra em duas etapas (passo 5b), em
                                                        model/runs/pipeline_duas_etapas/: propõe
                                                        o limiar de doença no BRACOL e roda T1 a
                                                        T3 com ela, sem mudar o padrão

O código de saída é 0 sem erros e 1 com erros.
A saída de terminal fica sem acento (terminal do Windows); os arquivos gravados são UTF-8.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[1]
for _pasta in ("data", "evaluation"):
    if str(RAIZ_REPO / _pasta) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO / _pasta))

import atalho
import bracol
import bracot
import dados
import detector
import formatacao as fmt
import inferencia as inf
import metrics
import train
import treinar_detector as td

# O run de cada regra de decisão (o padrão de --nome): a cadeia final (passo 6), os padrões de
# 11/10/2026 com o argmax (recorte B e filtro de fragmentos) e a regra em duas etapas em todo
# caminho (passo 5b). O pipeline_base, do passo 3 (recorte A, sem filtro), fica como registro e não
# é sobrescrito por uma rodada sem --nome.
NOMES = {"por_caminho": "pipeline_final", "argmax": "pipeline_padrao_b",
         "duas_etapas": "pipeline_duas_etapas"}
NOME_REGRA = {"argmax": "argmax em todo caminho", "duas_etapas": "duas etapas em todo caminho",
              "por_caminho": "por caminho (cadeia final)"}
CAMINHOS = {"foto_inteira": "foto inteira (regras 1 e 2)", "deteccao": "recortes por detecção",
            "todas": "todas"}
VALIDACAO_CAMPO = train.PASTA_RUNS / "campo_validacao_duas_etapas" / "validacao.json"
PASTA_FORA_DA_DOBRA = train.PASTA_RUNS / "diagnostico_blocos"
ALVO_ACURACIA = 0.95  # acurácia das predições mantidas que a proposta de limiar busca
LIMIARES_CURVA = np.round(np.arange(0.30, 1.0, 0.01), 2)
# limiar de doença: a probabilidade renormalizada entre as 4 doenças nunca fica abaixo de 0,25
LIMIARES_DOENCA = np.round(np.arange(0.25, 1.0, 0.01), 2)
LIMIARES_DOENCA_DESTAQUE = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Sensibilidade do limiar de saudável (decisão do gestor, 11/10/2026): as 82 folhas S ou D do
# campo rotulado, com o P(saudável) do modelo atual no modo B, e as 3 S com marca na ponta.
FOLHAS_CAMPO = train.PASTA_RUNS / "campo_diagnostico" / "folhas.csv"
S_COM_MARCA = ("20191208_142146_02", "20191208_142156_03", "20191208_142829_05")
LIMIARES_T2 = (0.5, 0.6, 0.7, 0.8, 0.9)
LIMIARES_DESTAQUE = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
AMOSTRA_FOLHA_UNICA = 200  # imagens de treino do BRACOL para a margem e a cobertura
BORDA_FUNDO = 0.08  # faixa de borda do tom do fundo (a da checagem de atalho)
RECORTES_POR_PRANCHA = 12
CREDITO_BRACOT = ("Fotos: BRACOT (Krohling, Tozzi de Souza e Tassis, 2021), CC BY 4.0, "
                  "DOI 10.17632/pmkbyjpf6k.1")
SEED = bracol.SEED_PADRAO
MODOS = ("A", "B")
VARIANTES_T1 = ("inteira", "cadeia_A", "cadeia_B", "recorte_A", "recorte_B")
DESCRICAO_T1 = {
    "inteira": "imagem inteira (a referência)",
    "cadeia_A": "cadeia, recorte A",
    "cadeia_B": "cadeia, recorte B",
    "recorte_A": "só o recorte A (sem a regra 1)",
    "recorte_B": "só o recorte B (sem a regra 1)",
}
DECISOES = (
    ("10/10/2026", "o detector está bom o bastante: AP50 de máscara de 94,7% (região) e 88,1% "
                   "(conservador), limiar 0,35."),
    ("10/10/2026", "classificar também as folhas cortadas pela borda, com área mínima e corte "
                   "por confiança do classificador; reavaliar o filtro de borda olhando os "
                   "recortes."),
    ("10/10/2026", "folha única: a foto inteira só se houver detecção com score >= 0,05; sem "
                   "nenhuma, planta_nao_identificada."),
    ("10/10/2026", "arquivo_muito_grande acima de ~64 megapixels, com as dimensões lidas do "
                   "cabeçalho."),
    ("11/10/2026", "recorte B por padrão (o A continua como opção), pelos rótulos de campo do "
                   "passo 4."),
    ("11/10/2026", "filtro de fragmentos ligado com fracao_visivel_minima = 0,5; folha sozinha "
                   "inteira não é filtrada; se o filtro tirar todas as folhas, baixa_confianca "
                   "pedindo para enquadrar a folha inteira."),
    ("11/10/2026", "sem validação cruzada das variantes do passo 4: o modelo oficial segue o "
                   "base_resnet50_bracol. O limiar de confiança segue pendente."),
    ("11/10/2026", "base_resnet50_bracol e detector_base congelados como modelos finais, sem "
                   "retreino final; os testes fechados medem estes modelos uma única vez, no "
                   "fim."),
    ("11/10/2026", "regra em duas etapas: o limiar de saudável fica fixado antes de dados novos, "
                   "com no máximo 5% das D das 82 folhas rotuladas marcadas saudáveis (0,083 no "
                   "modelo atual, modo B), sem reajuste com dados novos; a regra fica desligada "
                   "por padrão até a validação."),
    ("11/10/2026", "os 82 rótulos de campo ficam congelados, inclusive as 3 S com marca na "
                   "ponta; o ponto de operação sem elas sai só como sensibilidade."),
    ("11/10/2026", "passo 6, cadeia final: a regra em duas etapas fica ligada por padrão só nos "
                   "recortes por detecção (modo B), com limiar de saudável 0,083 e limiar de "
                   "doença 0,65 (provisório); as regras de foto inteira (folha única) continuam "
                   "no argmax com corte 0,61."),
    ("11/10/2026", "passo 6: nenhum limiar é reajustado com dados de campo."),
    ("11/10/2026", "passo 6: as limitações de campo ficam registradas no relatório e nos READMEs "
                   "(rótulos de Claude sem especialista, uma lavoura e duas sessões, falso "
                   "saudável nas D evidentes, folhas ambíguas, classe da doença e severidade não "
                   "verificadas no campo)."),
)


# ------------------------------------------------------------------------ constantes
def medir_constantes(modelos, config) -> dict:
    """As constantes da cadeia medidas em dados de treino (nada é escolhido na validação)."""
    treino = dados.montar_treino()
    fmt.dizer(f"Constantes: tom do fundo em {len(treino)} imagens de treino do BRACOL...")
    fundo = medir_fundo(treino)
    amostra = dados.amostra_estratificada(treino, AMOSTRA_FOLHA_UNICA, SEED)
    fmt.dizer(f"Constantes: folhas de {len(amostra)} imagens de treino do BRACOL...")
    folha = medir_folha_unica(amostra, modelos, config)
    fmt.dizer("Constantes: fotos de treino do BRACOT (sem a validacao)...")
    plantas = medir_plantas(modelos, config)
    fmt.dizer("Constantes: cor das folhas anotadas no treino do BRACOT...")
    cor = medir_cor_das_folhas(config)
    return {"fundo": fundo, "folha_unica_bracol": folha, "plantas_bracot": plantas,
            "cor_folhas_bracot": cor}


def medir_fundo(treino) -> dict:
    """Mediana do tom da borda (BORDA_FUNDO) de cada imagem, já em 448x224, por classe; o cinza
    é a média das medianas de luminância das classes (cada classe com o mesmo peso)."""
    ver = dados.transformacao_sem_normalizar()
    linhas, borda = [], None
    for linha in treino.itertuples():
        rgb = ver(image=dados.ler_imagem(RAIZ_REPO / linha.caminho))["image"]
        if borda is None:
            borda = atalho.mascara_borda(rgb.shape[0], rgb.shape[1], BORDA_FUNDO)
        pixels = rgb[borda].astype(np.float64)
        r, g, b = np.median(pixels, axis=0)
        linhas.append({"classe": linha.classe, "r": r, "g": g, "b": b,
                       "luminancia": float(np.median(pixels.mean(axis=1)))})
    tabela = pd.DataFrame(linhas).groupby("classe")[["r", "g", "b", "luminancia"]].median()
    tabela = tabela.reindex(bracol.CLASSES)
    return {"imagens": len(linhas), "borda": BORDA_FUNDO,
            "por_classe": tabela.round(1).to_dict("index"),
            "cinza": int(round(tabela["luminancia"].mean()))}


def medir_folha_unica(amostra, modelos, config) -> dict:
    """Nas fotos de folha única do BRACOL: detecções utilizáveis, comprimento e espessura da
    maior folha (relativos à largura e à altura da foto) e cobertura da caixa dela."""
    linhas, cores = [], []
    for linha in amostra.itertuples():
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        altura, largura = imagem.shape[:2]
        deteccoes = inf.detectar(imagem, config, modelos)
        cores += [d["colorida"] for d in deteccoes
                  if d["score"] >= config.limiar_deteccao and d["area"] >= config.area_minima]
        uteis = _utilizaveis(deteccoes, config)
        if not uteis:
            linhas.append({"deteccoes": 0})
            continue
        maior = max(uteis, key=lambda d: d["area"])
        _, _, comprimento, espessura = inf.eixo_da_folha(maior["poligono"])
        linhas.append({"deteccoes": len(uteis), "comprimento": comprimento / largura,
                       "espessura": espessura / altura,
                       "cobertura": inf.cobertura_da_caixa(maior["caixa"], altura, largura)})
    t = pd.DataFrame(linhas)
    com = t[t["deteccoes"] > 0]
    comprimento = float(com["comprimento"].median())
    return {"imagens": len(t), "sem_deteccao": int((t["deteccoes"] == 0).sum()),
            "uma_deteccao": int((t["deteccoes"] == 1).sum()),
            "mais_de_uma": int((t["deteccoes"] > 1).sum()),
            "comprimento_mediano": comprimento,
            "espessura_mediana": float(com["espessura"].median()),
            "margem": (1 / comprimento - 1) / 2,
            "cobertura": _percentis(com["cobertura"]),
            "deteccoes_antes_da_cor": len(cores),
            "cortadas_pela_cor": int(sum(c < config.fracao_colorida_minima for c in cores)),
            "colorida_histograma": np.histogram(cores, bins=np.linspace(0, 1, 11))[0].tolist()}


def medir_plantas(modelos, config) -> dict:
    """Nas fotos de treino do BRACOT (sem as de validação): cobertura da maior caixa e quantas
    fotos têm uma só folha utilizável."""
    manifest = bracot.ler_manifest(split="treino")
    validacao = set(td.dividir_validacao(manifest))
    linhas = []
    for linha in manifest[~manifest["id"].isin(validacao)].itertuples():
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        altura, largura = imagem.shape[:2]
        uteis = _utilizaveis(inf.detectar(imagem, config, modelos), config)
        linhas.append({"deteccoes": len(uteis), "cobertura": max(
            (inf.cobertura_da_caixa(d["caixa"], altura, largura) for d in uteis), default=0.0)})
    t = pd.DataFrame(linhas)
    return {"fotos": len(t), "uma_deteccao": int((t["deteccoes"] == 1).sum()),
            "sem_deteccao": int((t["deteccoes"] == 0).sum()),
            "cobertura_maior": _percentis(t["cobertura"])}


def medir_cor_das_folhas(config, fotos: int = 80) -> dict:
    """Nas folhas anotadas de fotos de treino do BRACOT (sem a validação, sorteadas com a
    semente): a fração da máscara com saturação >= saturacao_minima, que o filtro de cor usa."""
    manifest = bracot.ler_manifest(split="treino")
    validacao = set(td.dividir_validacao(manifest))
    poligonos = td.ler_poligonos("treino")
    amostra = manifest[~manifest["id"].isin(validacao)].sample(fotos, random_state=SEED)
    fracoes = []
    for linha in amostra.itertuples():
        reduzida, escala = inf.reduzir(dados.ler_imagem(RAIZ_REPO / linha.caminho), inf.LADO_COR)
        saturacao = cv2.cvtColor(reduzida, cv2.COLOR_RGB2HSV)[..., 1]
        fracoes += [inf.fracao_colorida(saturacao, p * escala, config.saturacao_minima)
                    for p in poligonos[linha.id]]
    return {"fotos": fotos, "folhas": len(fracoes), **_percentis(fracoes),
            "p1": float(np.quantile(fracoes, 0.01)),
            "abaixo_do_minimo": int(sum(f < config.fracao_colorida_minima for f in fracoes))}


def _utilizaveis(deteccoes, config) -> list[dict]:
    return [d for d in deteccoes if inf.utilizavel(d, config)]


def _percentis(serie) -> dict:
    s = pd.Series(serie, dtype=float)
    return {"min": float(s.min()), "p5": float(s.quantile(0.05)), "p25": float(s.quantile(0.25)),
            "mediana": float(s.median()), "p95": float(s.quantile(0.95)), "max": float(s.max())}


# ----------------------------------------------------------------------- calibração
def ler_fora_da_dobra(pasta: Path) -> pd.DataFrame:
    """As predições fora da dobra do protocolo aleatório do diagnóstico por blocos."""
    arquivos = sorted(Path(pasta).glob("predicoes_aleatorio_*.csv"))
    if not arquivos:
        raise train.ErroFatal(f"sem predicoes_aleatorio_*.csv em {train._exibir(Path(pasta))}")
    return pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)


def curva_das_probabilidades(probabilidades, verdadeiras) -> list[dict]:
    probabilidades = np.asarray(probabilidades, dtype=np.float64)
    verdadeiras = np.asarray(verdadeiras)
    return metrics.curva_risco_cobertura(
        probabilidades.max(axis=1), probabilidades.argmax(axis=1) == verdadeiras,
        LIMIARES_CURVA, verdadeiras)


def propor_limiar(curva, alvo: float = ALVO_ACURACIA) -> dict | None:
    """O menor limiar em que a acurácia das mantidas chega ao alvo."""
    return next((p for p in curva if p["mantidas"] and p["acuracia"] >= alvo), None)


def calibrar(pasta_fora_da_dobra: Path, inteira_val: pd.DataFrame) -> dict:
    fora = ler_fora_da_dobra(pasta_fora_da_dobra)
    colunas = [f"prob_{c}" for c in bracol.CLASSES]
    verdadeiras = fora["classe_verdadeira"].map(bracol.CLASSES.index)
    curva_fora = curva_das_probabilidades(fora[colunas].to_numpy(), verdadeiras)
    curva_val = curva_das_probabilidades(np.stack(inteira_val["probs"]),
                                         inteira_val["classe"].to_numpy())
    proposta = propor_limiar(curva_fora)
    return {"fora_da_dobra": {"pasta": train._exibir(Path(pasta_fora_da_dobra)),
                              "n": len(fora),
                              "acuracia": float(np.mean(
                                  fora[colunas].to_numpy().argmax(axis=1) == verdadeiras)),
                              "curva": curva_fora},
            "validacao": {"n": len(inteira_val), "curva": curva_val},
            "alvo": ALVO_ACURACIA,
            "proposta": None if proposta is None else proposta["limiar"]}


# ------------------------------------------------------- regra em duas etapas (passo 5b)
def _proporcao(sucessos: int, n: int) -> dict:
    """Proporção com o intervalo de Wilson de 95% (n = 0: sem valor)."""
    if not n:
        return {"p": None, "ic95": [None, None], "sucessos": 0, "n": 0}
    return {"p": sucessos / n, "ic95": list(metrics.intervalo_wilson(sucessos / n, n)),
            "sucessos": int(sucessos), "n": int(n)}


def doenca_renormalizada(probabilidades) -> tuple[np.ndarray, np.ndarray]:
    """Por linha: a doença de maior probabilidade (índice em CLASSES) e a probabilidade dela
    renormalizada entre as 4 doenças (a 2ª etapa de inferencia.aplicar_decisao)."""
    p = np.asarray(probabilidades, dtype=np.float64)[:, list(inf.DOENCAS)]
    soma = p.sum(axis=1, keepdims=True)
    renormalizadas = np.divide(p, soma, out=np.full_like(p, 1 / len(inf.DOENCAS)),
                               where=soma > 0)
    return (np.asarray(inf.DOENCAS)[renormalizadas.argmax(axis=1)],
            renormalizadas.max(axis=1))


def curva_das_doencas(probabilidades, verdadeiras) -> list[dict]:
    """Curva risco-cobertura do limiar de doença, só nas folhas doentes: a doença prevista com o
    softmax renormalizado contra a verdadeira, cortada pela probabilidade renormalizada."""
    probabilidades = np.asarray(probabilidades, dtype=np.float64)
    verdadeiras = np.asarray(verdadeiras)
    doentes = verdadeiras != inf.SAUDAVEL
    prevista, renormalizada = doenca_renormalizada(probabilidades[doentes])
    return metrics.curva_risco_cobertura(renormalizada, prevista == verdadeiras[doentes],
                                         LIMIARES_DOENCA, verdadeiras[doentes])


def calibrar_doencas(pasta_fora_da_dobra: Path) -> dict:
    """A proposta do limiar de doença (2ª etapa): o menor em que a acurácia das doentes mantidas
    fora da dobra chega ao alvo, como na proposta do limiar de confiança. Só o BRACOL."""
    fora = ler_fora_da_dobra(pasta_fora_da_dobra)
    probabilidades = fora[[f"prob_{c}" for c in bracol.CLASSES]].to_numpy()
    verdadeiras = fora["classe_verdadeira"].map(bracol.CLASSES.index).to_numpy()
    doentes = verdadeiras != inf.SAUDAVEL
    prevista, _ = doenca_renormalizada(probabilidades[doentes])
    curva = curva_das_doencas(probabilidades, verdadeiras)
    proposta = propor_limiar(curva)
    return {"fora_da_dobra": {"pasta": train._exibir(Path(pasta_fora_da_dobra)),
                              "n": len(fora), "n_doentes": int(doentes.sum()),
                              "acuracia_sem_corte": float(np.mean(prevista
                                                                  == verdadeiras[doentes])),
                              "curva": curva},
            "alvo": ALVO_ACURACIA,
            "proposta": None if proposta is None else proposta["limiar"]}


def efeito_do_limiar_saudavel(probabilidades, verdadeiras, limiar_saudavel: float) -> dict:
    """Doentes marcadas saudáveis e recall das saudáveis: argmax contra P(saudável) >= limiar."""
    probabilidades = np.asarray(probabilidades, dtype=np.float64)
    doente = np.asarray(verdadeiras) != inf.SAUDAVEL
    regras = {"argmax": probabilidades.argmax(axis=1) == inf.SAUDAVEL,
              "limiar": probabilidades[:, inf.SAUDAVEL] >= limiar_saudavel}
    return {nome: {"saudavel_dado_d": _proporcao(int((s & doente).sum()), int(doente.sum())),
                   "recall_saudavel": _proporcao(int((s & ~doente).sum()), int((~doente).sum()))}
            for nome, s in regras.items()}


def regra_completa(probabilidades, verdadeiras, config) -> dict:
    """A regra da configuração inteira (as duas etapas e o corte; ou argmax e o corte) em
    predições com rótulo: cobertura, acurácia das mantidas e cobertura de saudáveis e doentes."""
    verdadeiras = np.asarray(verdadeiras)
    decisoes = [inf.aplicar_decisao(p, config) for p in np.asarray(probabilidades)]
    classes = np.array([d[0] for d in decisoes])
    mantidas = np.array([d[2] for d in decisoes]) >= inf.limiar_de_corte(config)
    acertos = classes == verdadeiras
    doente = verdadeiras != inf.SAUDAVEL
    return {"acuracia_sem_corte": float(acertos.mean()),
            "cobertura": float(mantidas.mean()),
            "acuracia_das_mantidas": _proporcao(int(acertos[mantidas].sum()),
                                                int(mantidas.sum())),
            "cobertura_saudaveis": float(mantidas[~doente].mean()) if (~doente).any() else None,
            "cobertura_doentes": float(mantidas[doente].mean()) if doente.any() else None}


def sensibilidade_campo(limiar_saudavel: float, caminho: Path = FOLHAS_CAMPO) -> dict | None:
    """O limiar de saudável nas 82 folhas do campo rotulado (P(saudável) do modelo atual no
    modo B, de model/runs/campo_diagnostico), com e sem as 3 S com marca na ponta: o limiar do
    ponto de operação recalculado (no máximo 5% das D) e, no limiar fixado, as D marcadas
    saudáveis e o recall das S. Os rótulos não mudam (decisão do gestor, 11/10/2026)."""
    import campo_diagnostico as cd

    if not Path(caminho).is_file():
        return None
    folhas = pd.read_csv(caminho)
    resultado = {}
    for nome, manter in (("82 folhas", np.ones(len(folhas), bool)),
                         ("sem as 3 S com marca", ~folhas["id"].isin(S_COM_MARCA).to_numpy())):
        f = folhas[manter]
        y = (f["rotulo"] == "S").astype(int).to_numpy()
        p = f["p_saudavel_atual_B"].to_numpy()
        resultado[nome] = {
            "S": int(y.sum()), "D": int((1 - y).sum()),
            "limiar_recalculado": cd.ponto_de_operacao(y, p)["limiar"],
            "d_marcadas_saudaveis": _proporcao(int((p[y == 0] >= limiar_saudavel).sum()),
                                               int((y == 0).sum())),
            "recall_s": _proporcao(int((p[y == 1] >= limiar_saudavel).sum()), int(y.sum()))}
    return resultado


def completar_duas_etapas(calibracao_doencas: dict, pasta_fora_da_dobra: Path,
                          inteira_val: pd.DataFrame, config) -> dict:
    """O resto da análise da regra em duas etapas, com o limiar de doença em uso: a curva na
    validação (imagem inteira, T1), o efeito do limiar de saudável no BRACOL, a regra completa
    contra o argmax com o limiar de confiança e a sensibilidade no campo rotulado."""
    fora = ler_fora_da_dobra(pasta_fora_da_dobra)
    probs_fora = fora[[f"prob_{c}" for c in bracol.CLASSES]].to_numpy()
    verdadeiras_fora = fora["classe_verdadeira"].map(bracol.CLASSES.index).to_numpy()
    probs_val = np.stack(inteira_val["probs"])
    verdadeiras_val = inteira_val["classe"].to_numpy()
    argmax = replace(config, decisao="argmax")
    fontes = {"fora_da_dobra": (probs_fora, verdadeiras_fora),
              "validacao": (probs_val, verdadeiras_val)}
    return {**calibracao_doencas,
            "validacao": {"n": len(inteira_val),
                          "n_doentes": int((verdadeiras_val != inf.SAUDAVEL).sum()),
                          "curva": curva_das_doencas(probs_val, verdadeiras_val)},
            "limiar_saudavel": config.limiar_saudavel, "limiar_doenca": config.limiar_doenca,
            "efeito_saudavel": {fonte: efeito_do_limiar_saudavel(p, v, config.limiar_saudavel)
                                for fonte, (p, v) in fontes.items()},
            "regra_completa": {fonte: {"duas_etapas": regra_completa(p, v, config),
                                       "argmax": regra_completa(p, v, argmax)}
                               for fonte, (p, v) in fontes.items()},
            # a regra completa em outros limiares de doença, com o limiar de saudável fixo
            "regra_completa_por_limiar": {
                fonte: {f"{t:g}": regra_completa(p, v, replace(config, limiar_doenca=t))
                        for t in sorted({*LIMIARES_DOENCA_DESTAQUE, config.limiar_doenca})}
                for fonte, (p, v) in fontes.items()},
            "campo": sensibilidade_campo(config.limiar_saudavel)}


def figura_risco_cobertura(calibracao: dict, limiar_em_uso: float,
                           titulo: str = "Curva risco-cobertura do limiar de confiança (RF07)",
                           destaques=LIMIARES_DESTAQUE, eixo_x: str = "imagens"):
    """Acurácia das mantidas x cobertura nas duas fontes, com os limiares em destaque."""
    import eda_bracol as eda

    n_fora, n_val = (calibracao[f].get("n_doentes", calibracao[f]["n"])
                     for f in ("fora_da_dobra", "validacao"))
    series = (("fora_da_dobra", f"fora da dobra ({fmt.n(n_fora)}, protocolo aleatório)",
               eda.AZUL),
              ("validacao", f"validação do classificador ({n_val})",
               eda.COR_CLASSE["ferrugem"]))
    with eda._estilo():
        fig = eda._figura(7.6, 4.4, titulo)
        ax = fig.add_subplot()
        menor = min(p["cobertura"] for chave, _, _ in series
                    for p in calibracao[chave]["curva"])
        inicio = max(0.0, math.floor((menor - 0.02) * 20) / 20)
        ax.axhline(ALVO_ACURACIA, color=eda.EIXO, linewidth=0.8, linestyle="--")
        ax.text(inicio + 0.005, ALVO_ACURACIA + 0.002, f"alvo {fmt.pct(ALVO_ACURACIA, 1, 0)}",
                color=eda.TINTA_2, va="bottom")
        for chave, rotulo, cor in series:
            curva = calibracao[chave]["curva"]
            x = [p["cobertura"] for p in curva]
            y = [p["acuracia"] for p in curva]
            ax.plot(x, y, color=cor, linewidth=2, label=rotulo)
            for p in curva:
                if any(abs(p["limiar"] - d) < 1e-9 for d in destaques):
                    ax.plot(p["cobertura"], p["acuracia"], "o", color=cor, markersize=5,
                            markeredgecolor=eda.SUPERFICIE, markeredgewidth=1)
                    if chave == "fora_da_dobra":
                        ax.annotate(fmt.decimal(p["limiar"], 2), (p["cobertura"], p["acuracia"]),
                                    textcoords="offset points", xytext=(4, -11),
                                    color=eda.TINTA_2, fontsize=8)
        proposta = calibracao["proposta"]
        if proposta is not None:
            p = next(q for q in calibracao["fora_da_dobra"]["curva"]
                     if abs(q["limiar"] - proposta) < 1e-9)
            ax.plot(p["cobertura"], p["acuracia"], "o", color=eda.TINTA, markersize=9,
                    markerfacecolor="none", markeredgewidth=1.5)
            ax.annotate(f"proposta {fmt.decimal(proposta, 2)}", (p["cobertura"], p["acuracia"]),
                        textcoords="offset points", xytext=(8, 10), ha="left", color=eda.TINTA)
        ax.set_xlim(inicio, 1.005)
        ax.set_xlabel(f"cobertura (fração das {eixo_x} mantidas)")
        ax.set_ylabel("acurácia das mantidas")
        ax.xaxis.set_major_formatter(lambda v, _: fmt.pct(v, 1, 0))
        ax.yaxis.set_major_formatter(lambda v, _: fmt.pct(v, 1, 0))
        ax.grid(axis="y")
        ax.legend(loc="lower left")
        ax.set_title(f"pontos: limiares {', '.join(fmt.decimal(d, 2) for d in destaques)};"
                     f" em uso na cadeia: {fmt.decimal(limiar_em_uso, 2)}", loc="left")
    return fig


# --------------------------------------------------------------------------- T1
def t1(modelos, config, imagens: pd.DataFrame | None = None, modos=MODOS) -> pd.DataFrame:
    """Cada imagem de validação do BRACOL (ou de `imagens`, com as colunas de dados.montar_val)
    pela imagem inteira, pela cadeia (regras completas) e só pelo recorte da maior folha (sem a
    regra 1), nos `modos` de recorte, pela regra de decisão do caminho de cada previsão (a imagem
    inteira é o caminho da foto inteira; o recorte, o da detecção). A classe é a da regra sem o
    corte; o valor de corte e se a folha fica no corte ficam registrados, e o corte entra nas
    métricas."""
    val = dados.montar_val() if imagens is None else imagens
    cadeia = replace(config, limiar_confianca=0.0)
    so_recorte = replace(cadeia, cobertura_folha_unica=math.inf)
    linhas = []
    for n, linha in enumerate(val.itertuples(), start=1):
        imagem = dados.ler_imagem(RAIZ_REPO / linha.caminho)
        altura, largura = imagem.shape[:2]
        deteccoes = inf.detectar(imagem, cadeia, modelos)
        preparo = inf.decidir(deteccoes, altura, largura, cadeia)
        preparo_recorte = inf.decidir(deteccoes, altura, largura, so_recorte)
        uteis = _utilizaveis(deteccoes, cadeia)
        registro = {"id": linha.id, "classe": bracol.CLASSES.index(linha.classe),
                    "severidade": int(linha.severity), "deteccoes": len(uteis),
                    "deteccoes_baixas": sum(d["score"] < cadeia.limiar_deteccao
                                            for d in deteccoes),
                    "cobertura_maior": max((inf.cobertura_da_caixa(d["caixa"], altura, largura)
                                            for d in uteis), default=0.0),
                    "cortadas_por_cor": preparo["contagens"]["cortadas_por_cor"],
                    "plano": preparo["plano"], "regra": preparo["regra"],
                    "folhas": len(preparo["folhas"])}
        inteira = _prever([inf.recorte_da_foto_inteira(imagem, config.cinza_fundo)], modelos,
                          config, "foto_inteira")[0]
        _registrar(registro, "inteira", inteira)
        for modo in modos:
            recorte = _folha_principal(imagem, preparo_recorte, config, modo, modelos, inteira)
            _registrar(registro, f"recorte_{modo}", recorte)
            principal = {"foto_inteira": inteira, "deteccao": recorte}.get(preparo["plano"])
            _registrar(registro, f"cadeia_{modo}", principal)
        linhas.append(registro)
        if n % 50 == 0:
            fmt.dizer(f"  T1: {n}/{len(val)}")
    return pd.DataFrame(linhas)


def _folha_principal(imagem, preparo, config, modo, modelos, inteira):
    """A previsão da maior folha do preparo (a foto inteira, se for esse o plano)."""
    if preparo["plano"] == "foto_inteira":
        return inteira
    if not preparo["folhas"]:
        return None
    previsoes = _prever(inf.recortar(imagem, preparo, config, modo), modelos, config,
                        preparo["plano"], modo)
    return previsoes[int(np.argmax([f["area"] for f in preparo["folhas"]]))]


def _prever(recortes, modelos, config, plano: str, modo: str | None = None) -> list[dict]:
    """As probabilidades e, pela regra de decisão do caminho (plano e modo de recorte;
    inferencia.config_do_caminho e aplicar_decisao), a classe, a confiança, o valor de corte e
    se a folha fica no corte, para cada recorte."""
    probabilidades, niveis = inf.classificar(recortes, modelos)
    regra = inf.config_do_caminho(config, plano, modo)
    previsoes = []
    for p, nivel in zip(probabilidades, niveis):
        classe, confianca, corte = inf.aplicar_decisao(p, regra)
        previsoes.append({"probs": p, "classe": classe, "confianca": confianca, "corte": corte,
                          "mantida": corte >= inf.limiar_de_corte(regra),
                          "decisao": regra.decisao, "nivel": int(nivel)})
    return previsoes


def _registrar(registro: dict, prefixo: str, previsao) -> None:
    for campo in ("classe", "confianca", "corte", "mantida", "decisao", "nivel", "probs"):
        registro[f"{prefixo}_{campo}"] = None if previsao is None else previsao[campo]


def metricas_t1(tabela: pd.DataFrame) -> dict:
    """Por variante presente na tabela: as métricas sem corte, as doentes marcadas saudáveis e o
    recall das saudáveis, e a cobertura e a acurácia com o corte da regra de cada previsão
    (coluna <variante>_mantida)."""
    resultado = {}
    for variante in (v for v in VARIANTES_T1 if f"{v}_classe" in tabela):
        com = tabela[tabela[f"{variante}_classe"].notna()]
        previstas = com[f"{variante}_classe"].astype(int).to_numpy()
        m = metrics.calcular_metricas(com["classe"].to_numpy(), previstas)
        ms = metrics.calcular_metricas_severidade(
            com["severidade"].to_numpy(),
            np.where(previstas == 0, 0, np.maximum(com[f"{variante}_nivel"].astype(int), 1)))
        mantidas = com[f"{variante}_mantida"].astype(bool)
        doente = com["classe"].to_numpy() != inf.SAUDAVEL
        saudavel = previstas == inf.SAUDAVEL
        resultado[variante] = {
            "cobertura_sem_limiar": len(com) / len(tabela), "metricas": m, "severidade": ms,
            "saudavel_dado_d": _proporcao(int((saudavel & doente).sum()), int(doente.sum())),
            "recall_saudavel": _proporcao(int((saudavel & ~doente).sum()),
                                          int((~doente).sum())),
            "decisoes": sorted(com[f"{variante}_decisao"].dropna().unique().tolist()),
            "mantidas": int(mantidas.sum()),
            "cobertura_no_limiar": float(mantidas.sum() / len(tabela)),
            "acuracia_no_limiar": float(np.mean(previstas[mantidas.to_numpy()]
                                                == com["classe"].to_numpy()[mantidas.to_numpy()]))
            if mantidas.any() else None}
    return resultado


# --------------------------------------------------------------------------- T2
def t2(modelos, config, pasta_recortes: Path | None = None):
    """Cada folha das fotos de validação do BRACOT, recortada nos modos A e B, pela regra de
    decisão do caminho (plano da foto e modo), com as probabilidades das classes (para comparar
    as regras com as mesmas previsões)."""
    manifest = bracot.ler_manifest(split="treino")
    caminhos = manifest.set_index("id")["caminho"]
    cadeia = replace(config, limiar_confianca=0.0)
    folhas, fotos = [], []
    if pasta_recortes:
        (pasta_recortes / "recortes").mkdir(parents=True, exist_ok=True)
    for foto in td.dividir_validacao(manifest):
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        preparo = inf.decidir(inf.detectar(imagem, cadeia, modelos), imagem.shape[0],
                              imagem.shape[1], cadeia)
        fotos.append({"foto": foto, "plano": preparo["plano"], "regra": preparo["regra"],
                      "folhas": len(preparo["folhas"]), **preparo["contagens"]})
        if not preparo["folhas"]:
            continue
        recortes = {modo: inf.recortar(imagem, preparo, cadeia, modo) for modo in MODOS}
        previsoes = {modo: _prever(recortes[modo], modelos, config, preparo["plano"], modo)
                     for modo in MODOS}
        for k, folha in enumerate(preparo["folhas"]):
            linha = {"foto": foto, "folha": k + 1, "plano": preparo["plano"],
                     "score": folha["score"], "area": folha["area"], "borda": folha["borda"]}
            for modo in MODOS:
                p = previsoes[modo][k]
                categoria = inf.CATEGORIAS[p["classe"]]
                linha.update({f"classe_{modo}": categoria, f"confianca_{modo}": p["confianca"],
                              f"corte_{modo}": p["corte"], f"mantida_{modo}": p["mantida"],
                              f"decisao_{modo}": p["decisao"],
                              f"severidade_{modo}": inf.severidade_do_contrato(categoria,
                                                                              p["nivel"]),
                              **{f"prob_{c}_{modo}": float(v)
                                 for c, v in zip(inf.CATEGORIAS, p["probs"])}})
            folhas.append(linha)
            if pasta_recortes:
                _gravar_recorte(pasta_recortes / "recortes" / f"{foto}_{k + 1:02d}.jpg",
                                [recortes[m][k] for m in MODOS], linha, modelos)
    return pd.DataFrame(folhas), pd.DataFrame(fotos)


def _entrada_do_modelo(recorte, modelos) -> np.ndarray:
    """O recorte como o classificador o vê, antes da normalização (448x224, INTER_AREA)."""
    redimensionar = modelos.transformacao.transforms[0]
    altura, largura = redimensionar.height, redimensionar.width
    return cv2.resize(recorte, (largura, altura), interpolation=dados.INTERPOLACAO)


def _gravar_recorte(caminho: Path, recortes, linha: dict, modelos) -> None:
    """A | B lado a lado, como o classificador vê, com a previsão (texto ASCII)."""
    partes = []
    for modo, recorte in zip(MODOS, recortes):
        img = cv2.copyMakeBorder(_entrada_do_modelo(recorte, modelos), 22, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        texto = (f"{modo}: {linha[f'classe_{modo}']} / {linha[f'severidade_{modo}']} "
                 f"{linha[f'confianca_{modo}']:.2f}")
        cv2.putText(img, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        partes.append(img)
    junto = cv2.hconcat([partes[0], np.full((partes[0].shape[0], 8, 3), 255, np.uint8),
                         partes[1]])
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(junto, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        caminho.write_bytes(jpeg.tobytes())


def escolher_para_pranchas(folhas: pd.DataFrame, n: int) -> pd.DataFrame:
    """n folhas variadas: um sorteio (semente) por classe prevista no modo A, em rodízio, com
    metade das vagas, quando houver, para folhas cortadas pela borda."""
    gerador = np.random.default_rng(SEED)
    embaralhadas = folhas.iloc[gerador.permutation(len(folhas))]
    filas = {(c, b): list(g.index) for (c, b), g in embaralhadas.groupby(["classe_A", "borda"])}
    escolhidas = []
    while len(escolhidas) < min(n, len(folhas)):
        for chave in sorted(filas, key=lambda k: (k[1], k[0])):
            if filas[chave] and len(escolhidas) < n:
                escolhidas.append(filas[chave].pop(0))
    return folhas.loc[escolhidas]


def gravar_pranchas(folhas_escolhidas: pd.DataFrame, pasta: Path) -> list[Path]:
    """Pranchas JPEG de RECORTES_POR_PRANCHA folhas cada (A | B), com a previsão."""
    import eda_bracol as eda

    caminhos = []
    blocos = [folhas_escolhidas.iloc[i:i + RECORTES_POR_PRANCHA]
              for i in range(0, len(folhas_escolhidas), RECORTES_POR_PRANCHA)]
    for numero, bloco in enumerate(blocos, start=1):
        with eda._estilo():
            fig = eda._figura(16, 2.05 * math.ceil(len(bloco) / 2) + 0.6,
                              f"Recortes do T2 (fotos de validação do BRACOT), prancha {numero}: "
                              "modo A (foto em volta) | modo B (fundo cinza)")
            eixos = fig.subplots(math.ceil(len(bloco) / 2), 2, squeeze=False)
            for ax in eixos.ravel():
                ax.axis("off")
            for ax, linha in zip(eixos.ravel(), bloco.itertuples()):
                imagem = cv2.imdecode(np.fromfile(
                    pasta / "recortes" / f"{linha.foto}_{linha.folha:02d}.jpg", np.uint8),
                    cv2.IMREAD_COLOR)[22:, :, ::-1]
                ax.imshow(imagem)
                ax.set_title(
                    f"{linha.foto}, folha {linha.folha}"
                    f"{' (cortada na borda)' if linha.borda else ''}"
                    f"\nA: {linha.classe_A} / {linha.severidade_A}, "
                    f"{fmt.pct(linha.confianca_A, 1, 0)}    B: {linha.classe_B} / "
                    f"{linha.severidade_B}, {fmt.pct(linha.confianca_B, 1, 0)}",
                    loc="left", fontsize=8)
            fig.text(0.01, 0.002, CREDITO_BRACOT, fontsize=7, color=eda.TINTA_2)
        caminho = pasta / f"prancha_{numero}.jpg"
        eda._salvar(fig, caminho)
        caminhos.append(caminho)
    return caminhos


def resumo_t2(folhas: pd.DataFrame, fotos: pd.DataFrame, config) -> dict:
    """Classes, severidades e confiança por modo; o corte da regra de cada folha (coluna
    mantida_<modo>) por folha e por foto (foto sem nenhuma folha mantida: baixa_confianca); e a
    distribuição do valor de corte em limiares fixos e nos limiares de corte da regra de cada
    modo."""
    limiares = sorted({*LIMIARES_T2, *(inf.limiar_de_corte(inf.config_do_caminho(
        config, plano, modo)) for plano in folhas["plano"].unique() for modo in MODOS)})
    resultado = {"fotos": len(fotos), "folhas": len(folhas),
                 "planos": fotos["plano"].value_counts().to_dict(),
                 "cortadas_por_area": int(fotos["cortadas_por_area"].sum()),
                 "cortadas_por_limite": int(fotos["cortadas_por_limite"].sum()),
                 "deteccoes": int(fotos["deteccoes"].sum()),
                 "na_borda": int(folhas["borda"].sum())}
    for modo in MODOS:
        conf = folhas[f"confianca_{modo}"]
        corte = folhas[f"corte_{modo}"].astype(float)
        mantida = folhas[f"mantida_{modo}"].astype(bool)
        borda = folhas["borda"].astype(bool)
        resultado[modo] = {
            "decisoes": folhas[f"decisao_{modo}"].value_counts().to_dict(),
            "classes": folhas[f"classe_{modo}"].value_counts().reindex(
                bracol.CLASSES, fill_value=0).to_dict(),
            "classes_mantidas": folhas.loc[mantida, f"classe_{modo}"].value_counts()
            .reindex(bracol.CLASSES, fill_value=0).to_dict(),
            "severidades": folhas[f"severidade_{modo}"].value_counts().reindex(
                inf.NIVEIS_CONTRATO, fill_value=0).to_dict(),
            "confianca_media": float(conf.mean()), "confianca_mediana": float(conf.median()),
            "abaixo": {f"{t:g}": float((corte < t).mean()) for t in limiares},
            "borda": {"folhas": int(borda.sum()), "confianca_media": float(conf[borda].mean()),
                      "abaixo_do_limiar": float((~mantida[borda]).mean())},
            "inteiras": {"folhas": int((~borda).sum()),
                         "confianca_media": float(conf[~borda].mean()),
                         "abaixo_do_limiar": float((~mantida[~borda]).mean())},
        }
    resultado["concordancia_A_B"] = float((folhas["classe_A"] == folhas["classe_B"]).mean())
    resultado["cortadas_por_cor"] = int(fotos["cortadas_por_cor"].sum())
    resultado["cortadas_por_fragmento"] = int(fotos.get("cortadas_por_fragmento",
                                                        pd.Series(dtype=int)).sum())
    maxima = folhas.groupby("foto")[[f"corte_{m}" for m in MODOS]].max()
    resultado["fotos_baixa_confianca"] = {
        modo: {f"{t:g}": int((maxima[f"corte_{modo}"] < t).sum()) for t in limiares}
        for modo in MODOS}
    resultado["fotos_baixa_confianca_na_regra"] = {
        modo: int((~folhas[f"mantida_{modo}"].astype(bool).groupby(folhas["foto"]).any()).sum())
        for modo in MODOS}
    return resultado


# --------------------------------------------------- regra por caminho (passo 6, cadeia final)
REGRAS_DE_DECISAO = ("argmax", "duas_etapas", "por_caminho")


def regras_comparadas(config, decisoes=REGRAS_DE_DECISAO) -> dict:
    """As regras de decisão pedidas (por padrão, as três), com os limiares da configuração."""
    return {d: replace(config, decisao=d) for d in decisoes}


def desfecho(probs, regra) -> tuple[int, str]:
    """A classe pela regra já resolvida no caminho e o que a cadeia faz com a folha:
    "saudavel", "doenca" ou "descartada" (abaixo do corte)."""
    classe, _, corte = inf.aplicar_decisao(probs, regra)
    if corte < inf.limiar_de_corte(regra):
        return classe, "descartada"
    return classe, "saudavel" if classe == inf.SAUDAVEL else "doenca"


def _sem_previsao(probs) -> bool:
    return probs is None or (isinstance(probs, float) and math.isnan(probs))


def _resumo_desfechos(t: pd.DataFrame) -> dict:
    """Imagens com rótulo: acurácia sem corte (nas que têm previsão), cobertura, acurácia das
    mantidas e as chamadas saudáveis entre as doentes e entre as saudáveis, com Wilson."""
    doente = (t["classe"] != inf.SAUDAVEL).to_numpy()
    com = t["prevista"].notna().to_numpy()
    acerto = com & (t["prevista"].to_numpy() == t["classe"].to_numpy())
    mantida = t["desfecho"].isin(["saudavel", "doenca"]).to_numpy()
    saudavel = (t["desfecho"] == "saudavel").to_numpy()
    return {"imagens": len(t), "com_previsao": int(com.sum()),
            "acuracia_sem_corte": float(acerto[com].mean()) if com.any() else None,
            "cobertura": _proporcao(int(mantida.sum()), len(t)),
            "acuracia_das_mantidas": _proporcao(int((acerto & mantida).sum()), int(mantida.sum())),
            "saudavel_dado_d": _proporcao(int((saudavel & doente).sum()), int(doente.sum())),
            "saudavel_dado_s": _proporcao(int((saudavel & ~doente).sum()), int((~doente).sum()))}


def t1_por_caminho(tabela: pd.DataFrame, config, decisoes=REGRAS_DE_DECISAO) -> dict:
    """O T1 da cadeia no modo B (a folha principal: a imagem inteira na foto inteira, o
    recorte B na detecção) separado pelo caminho da imagem, nas regras pedidas (por padrão, as
    três), com as mesmas probabilidades (coluna cadeia_B_probs). A imagem sem folha (regra 3)
    conta como não mantida."""
    resultado = {}
    for nome, cfg in regras_comparadas(config, decisoes).items():
        linhas = []
        for plano, classe, probs in zip(tabela["plano"], tabela["classe"],
                                        tabela["cadeia_B_probs"]):
            if _sem_previsao(probs):
                linhas.append({"plano": plano, "classe": classe, "prevista": None,
                               "desfecho": "sem_folha"})
                continue
            prevista, d = desfecho(probs, inf.config_do_caminho(cfg, plano, "B"))
            linhas.append({"plano": plano, "classe": classe, "prevista": prevista,
                           "desfecho": d})
        t = pd.DataFrame(linhas)
        resultado[nome] = {caminho: _resumo_desfechos(t if caminho == "todas"
                                                      else t[t["plano"] == caminho])
                           for caminho in CAMINHOS}
    return resultado


def t2_por_regra(folhas: pd.DataFrame, fotos: pd.DataFrame, config) -> dict:
    """As folhas do T2 no modo B (o da cadeia), nas três regras, com as mesmas probabilidades
    (colunas prob_<classe>_B): folhas chamadas saudáveis, doentes (por classe) e descartadas; e
    as fotos com resposta, com alguma doença na resposta, em baixa_confianca pelo corte, pelo
    filtro de fragmentos e sem folha (planta_nao_identificada)."""
    probs = folhas[[f"prob_{c}_B" for c in inf.CATEGORIAS]].to_numpy()
    sem_folhas = {"baixa_confianca_fragmentos": int((fotos["plano"] == "fragmentos").sum()),
                  "planta_nao_identificada": int((fotos["plano"] == "nenhuma").sum())}
    resultado = {}
    for nome, cfg in regras_comparadas(config).items():
        saidas = [desfecho(p, inf.config_do_caminho(cfg, plano, "B"))
                  for p, plano in zip(probs, folhas["plano"])]
        f = folhas.assign(prevista=[inf.CATEGORIAS[c] for c, _ in saidas],
                          desfecho=[d for _, d in saidas])
        mantida = f["desfecho"] != "descartada"
        por_foto = pd.DataFrame({"resposta": mantida, "doenca": f["desfecho"] == "doenca",
                                 "foto": f["foto"]}).groupby("foto").any()
        resultado[nome] = {
            "folhas": len(f),
            "desfechos": {d: int((f["desfecho"] == d).sum())
                          for d in ("saudavel", "doenca", "descartada")},
            "classes_mantidas": f.loc[mantida, "prevista"].value_counts()
            .reindex(inf.CATEGORIAS, fill_value=0).to_dict(),
            "fotos": {"total": len(fotos), "com_resposta": int(por_foto["resposta"].sum()),
                      "com_doenca": int(por_foto["doenca"].sum()),
                      "baixa_confianca_corte": int((~por_foto["resposta"]).sum()),
                      **sem_folhas}}
    return resultado


def ler_run(nome: str, pasta_runs: Path = train.PASTA_RUNS) -> tuple[dict, dict] | None:
    """O config.json e o metricas.json gravados num run (None se faltar algum)."""
    pasta = Path(pasta_runs) / nome
    if not all((pasta / a).is_file() for a in ("config.json", "metricas.json")):
        return None
    return tuple(json.loads((pasta / a).read_text(encoding="utf-8"))
                 for a in ("config.json", "metricas.json"))


def comparar_com_runs(por_caminho_t1: dict, por_regra_t2: dict,
                      pasta_runs: Path = train.PASTA_RUNS) -> dict:
    """Os números gravados nos runs pipeline_padrao_b (argmax) e pipeline_duas_etapas (T1 da
    cadeia B, T2 no modo B e T3) e os recalculados aqui com a mesma regra e as probabilidades
    deste run (conferência: as mesmas previsões dão os mesmos números)."""
    resultado = {}
    for regra in ("argmax", "duas_etapas"):
        lido = ler_run(NOMES[regra], pasta_runs)
        if lido is None:
            resultado[NOMES[regra]] = None
            continue
        config_run, m = lido
        c = config_run["configuracao"]
        limiar = c["limiar_doenca"] if regra == "duas_etapas" else c["limiar_confianca"]
        cadeia = m["t1"]["variantes"]["cadeia_B"]
        gravado = {"t1_acuracia_sem_corte": cadeia["metricas"]["acuracia"],
                   "t1_mantidas": cadeia["mantidas"],
                   "t1_acuracia_das_mantidas": cadeia["acuracia_no_limiar"],
                   "t2_folhas": m["t2"]["folhas"],
                   "t2_fotos_baixa_confianca": m["t2"]["fotos_baixa_confianca"]["B"][f"{limiar:g}"],
                   "t3_mediana_s": m["t3"]["mediana"]["total_s"],
                   "t3_maximo_s": m["t3"]["maximo"]["total_s"]}
        todas, t2r = por_caminho_t1[regra]["todas"], por_regra_t2[regra]
        recalculado = {"t1_acuracia_sem_corte": todas["acuracia_sem_corte"],
                       "t1_mantidas": todas["cobertura"]["sucessos"],
                       "t1_acuracia_das_mantidas": todas["acuracia_das_mantidas"]["p"],
                       "t2_folhas": t2r["folhas"],
                       "t2_fotos_baixa_confianca": t2r["fotos"]["baixa_confianca_corte"]}
        resultado[NOMES[regra]] = {
            "regra": regra, "limiar": limiar, "criado_em": config_run["criado_em"],
            "gravado": gravado, "recalculado": recalculado,
            "confere": all(math.isclose(recalculado[k], gravado[k], abs_tol=1e-6)
                           for k in recalculado)}
    return resultado


def limitacoes_de_campo(caminho: Path = VALIDACAO_CAMPO) -> dict | None:
    """Os números de campo das limitações da cadeia final (decisão do gestor, 11/10/2026), lidos
    da validação da regra em duas etapas: contagens de S, D e "?", saudável dado S e dado D e a
    fração de "?" que a regra chama saudável (com "?" como D, menos as D), por conjunto."""
    if not Path(caminho).is_file():
        return None
    v = json.loads(Path(caminho).read_text(encoding="utf-8"))
    resultado = {"arquivo": train._exibir(Path(caminho)), "limiar_saudavel": v["limiar_saudavel"],
                 "limiar_doenca": v["limiar_doenca_proposto"]}
    for conjunto in ("antigas", "novas", "juntas"):
        regra = v["por_conjunto"][conjunto]["regras"]["duas_etapas"]
        sem = v["limites"][conjunto]["sem_interrogacao"]["regras"]["duas_etapas"]
        como_d = v["limites"][conjunto]["interrogacao_como_D"]["regras"]["duas_etapas"]
        sucessos = como_d["saudavel_dado_d"]["sucessos"] - sem["saudavel_dado_d"]["sucessos"]
        n = como_d["saudavel_dado_d"]["n"] - sem["saudavel_dado_d"]["n"]
        resultado[conjunto] = {**v["contagens"][conjunto],
                               "saudavel_dado_s": regra["saudavel_dado_s"],
                               "saudavel_dado_d": regra["saudavel_dado_d"],
                               "interrogacao_chamada_saudavel": _proporcao(sucessos, n),
                               "saudavel_dado_d_interrogacao_como_d": como_d["saudavel_dado_d"]}
    return resultado


def analisar_por_caminho(tabela_t1: pd.DataFrame, folhas_t2: pd.DataFrame,
                         fotos_t2: pd.DataFrame, config) -> dict:
    t1r = t1_por_caminho(tabela_t1, config)
    t2r = t2_por_regra(folhas_t2, fotos_t2, config)
    return {"t1": t1r, "t2": t2r, "runs": comparar_com_runs(t1r, t2r),
            "campo": limitacoes_de_campo()}


# --------------------------------------------------------------------------- T3
def t3(pasta_classificador: Path, pesos_detector: Path, config) -> tuple[pd.DataFrame, dict]:
    """Tempo da cadeia em CPU, por foto de validação do BRACOT: leitura, detecção, recortes,
    classificação (um lote) e contrato. Os modelos de CPU são carregados à parte (o detector numa
    instância nova, para não herdar a da GPU). A primeira foto aquece e fica de fora. Pior caso:
    a classificação de max_folhas recortes."""
    import torch

    inicio = time.perf_counter()
    modelos = inf.carregar_modelos(pasta_classificador, pesos_detector, "cpu")
    modelos = replace(modelos, detector=detector.importar_ultralytics()(str(pesos_detector)))
    carga = time.perf_counter() - inicio
    manifest = bracot.ler_manifest(split="treino")
    caminhos = manifest.set_index("id")["caminho"]
    ids = td.dividir_validacao(manifest)
    linhas = []
    for k, foto in enumerate([ids[0], *ids]):
        t0 = time.perf_counter()
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        t1_ = time.perf_counter()
        deteccoes = inf.detectar(imagem, config, modelos)
        t2_ = time.perf_counter()
        preparo = inf.decidir(deteccoes, imagem.shape[0], imagem.shape[1], config)
        recortes = inf.recortar(imagem, preparo, config) if preparo["folhas"] else []
        t3_ = time.perf_counter()
        if recortes:
            probabilidades, niveis = inf.classificar(recortes, modelos)
            t4_ = time.perf_counter()
            resposta = inf.montar_resposta(preparo, probabilidades, niveis, config, modelos)
        else:
            t4_ = time.perf_counter()
            resposta = inf.resposta_de_erro("planta_nao_identificada")
        t5_ = time.perf_counter()
        if k:
            linhas.append({"foto": foto, "folhas": len(recortes), "leitura_s": t1_ - t0,
                           "deteccao_s": t2_ - t1_, "recortes_s": t3_ - t2_,
                           "classificacao_s": t4_ - t3_, "contrato_s": t5_ - t4_,
                           "total_s": t5_ - t0, "status": resposta["status"]})
    tabela = pd.DataFrame(linhas)
    # Pior caso: max_folhas recortes (os da foto repetidos) num lote só.
    pior = []
    for foto in ids[:10]:
        imagem = dados.ler_imagem(RAIZ_REPO / caminhos[foto])
        preparo = inf.decidir(inf.detectar(imagem, config, modelos), imagem.shape[0],
                              imagem.shape[1], config)
        recortes = inf.recortar(imagem, preparo, config) if preparo["folhas"] else []
        if not recortes:
            continue
        recortes = (recortes * config.max_folhas)[:config.max_folhas]
        t0 = time.perf_counter()
        inf.classificar(recortes, modelos)
        pior.append(time.perf_counter() - t0)
    etapas = [c for c in tabela.columns if c.endswith("_s")]
    por_folha = (tabela["recortes_s"] / tabela["folhas"].clip(lower=1)).median()
    resumo = {
        "threads_torch": torch.get_num_threads(), "fotos": len(tabela),
        "carga_dos_modelos_s": carga,
        "mediana": {c: float(tabela[c].median()) for c in etapas},
        "maximo": {c: float(tabela[c].max()) for c in etapas},
        "folhas_mediana": float(tabela["folhas"].median()),
        "folhas_maximo": int(tabela["folhas"].max()),
        "pior_caso": {
            "folhas": config.max_folhas,
            "classificacao_mediana_s": float(np.median(pior)),
            "classificacao_maximo_s": float(np.max(pior)),
            "total_estimado_s": float(tabela["leitura_s"].max() + tabela["deteccao_s"].max()
                                      + por_folha * config.max_folhas + np.max(pior)),
        },
        "acima_de_5s": int((tabela["total_s"] > 5).sum()),
    }
    return tabela, resumo


# ------------------------------------------------------------------------- relatório
def montar_config(nome, config, pasta_classificador, pesos_detector, pasta_fora, dispositivo,
                  constantes) -> dict:
    commit, alterado = train.estado_git()
    return {
        "run": nome, "criado_em": datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "commit_com_alteracoes": alterado,
        "decisoes_do_gestor": [{"data": d, "texto": t} for d, t in DECISOES],
        "configuracao": asdict(config),
        "modelos": {"classificador": train._exibir(Path(pasta_classificador).resolve()),
                    "detector": train._exibir(Path(pesos_detector).resolve()),
                    "fora_da_dobra": train._exibir(Path(pasta_fora).resolve())},
        "dispositivo_t1_t2": str(dispositivo), "dispositivo_t3": "cpu",
        "constantes_medidas": constantes,
        "acessos_a_rede": [],
        "versoes": train.versoes(),
    }


def _pct(valor, casas: int = 1) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "-"
    return fmt.pct(valor, 1, casas)


def _ic(m: dict) -> str:
    baixo, alto = m["acuracia_ic95"]
    return f"{_pct(baixo)} a {_pct(alto)}"


def relatorio(config_run: dict, metricas: dict, observacoes: str | None) -> str:
    c = config_run["configuracao"]
    cal = metricas["calibracao"]
    t1m, t2m, t3m = metricas["t1"], metricas["t2"], metricas["t3"]
    const = config_run["constantes_medidas"]
    final = c.get("decisao") == "por_caminho"
    linhas = [
        f"# Inferência ponta a ponta: run `{config_run['run']}`", "",
        f"Gerado por `python model/avaliar_pipeline.py` em {_data(config_run['criado_em'])}, "
        f"{_origem(config_run)}. Classificador `{config_run['modelos']['classificador']}`, "
        f"detector `{config_run['modelos']['detector']}`. **Os testes do BRACOL e do BRACOT não "
        "foram lidos nem avaliados.** Módulo: `model/inferencia.py`; interface no README "
        "(seção \"Interface para o backend\").", "",
        "## Decisões do gestor", "",
        *(f"- **{x['data']}:** {x['texto']}" for x in config_run["decisoes_do_gestor"]), "",
        "## Configuração da cadeia", "",
    ]
    linhas += fmt.tabela(["parâmetro", "valor", "origem"], [
        ["limiar_deteccao", fmt.decimal(c["limiar_deteccao"], 2), "decisão do gestor"],
        ["max_folhas", c["max_folhas"], "RNF02 (a validação do BRACOT tem até 13 detecções)"],
        ["area_minima", _pct(c["area_minima"]), "pedido do gestor (cortadas: T2)"],
        ["saturacao_minima / fracao_colorida_minima",
         f"{c['saturacao_minima']} / {_pct(c['fracao_colorida_minima'], 0)}",
         "filtro de cor, medido no treino (abaixo)"],
        ["limiar_confianca", fmt.decimal(c["limiar_confianca"], 2),
         "curva risco-cobertura (abaixo); vale na foto inteira e no modo A: decisão do gestor "
         "(11/10/2026, passo 6)" if final else
         "proposta da curva risco-cobertura; **aguarda a decisão do gestor**"],
        ["recorte", c["recorte"],
         "decisão do gestor (11/10/2026), pelos rótulos de campo do passo 4; no T1 só o recorte "
         f"acerta {_pct(t1m['variantes']['recorte_A']['metricas']['acuracia'])} no A e "
         f"{_pct(t1m['variantes']['recorte_B']['metricas']['acuracia'])} no B (abaixo)"],
        ["margem", _pct(c["margem"]), "medida no treino do BRACOL (abaixo)"],
        ["cinza_fundo", c["cinza_fundo"], "medido no treino do BRACOL (abaixo)"],
        ["cobertura_folha_unica", _pct(c["cobertura_folha_unica"], 0),
         "entre o BRACOL e o BRACOT de treino (abaixo)"],
        ["score_minimo_folha_unica", fmt.decimal(c["score_minimo_folha_unica"], 2),
         "decisão do gestor"],
        ["max_megapixels", fmt.decimal(c["max_megapixels"], 0), "decisão do gestor"],
        ["fracao_visivel_minima", fmt.decimal(c["fracao_visivel_minima"], 2),
         "filtro de fragmentos; decisão do gestor (11/10/2026)"],
        ["decisao", c.get("decisao", "argmax"),
         "decisão do gestor (11/10/2026, passo 6): duas etapas nos recortes por detecção "
         "(modo B); argmax com limiar_confianca na foto inteira e no modo A" if final else
         "argmax: o padrão; duas_etapas: proposta do passo 5b, desligada por padrão"],
        *([["limiar_saudavel", fmt.decimal(c["limiar_saudavel"], 3),
            "decisão do gestor (11/10/2026): fixado antes de dados novos"
            + (" e validado no campo (`campo_validacao_duas_etapas`), sem reajuste" if final
               else " (abaixo)")],
           ["limiar_doenca", fmt.decimal(c["limiar_doenca"], 2),
            "decisão do gestor (11/10/2026, passo 6): **provisório**; a proposta da curva das "
            "doenças no BRACOL (`pipeline_duas_etapas`)" if final else
            "proposta da curva das doenças (abaixo); **aguarda a decisão do gestor**"]]
          if c.get("decisao") in ("duas_etapas", "por_caminho") else []),
    ], "lll")
    linhas += _secao_constantes(const, c)
    linhas += _secao_calibracao(cal, c)
    if "duas_etapas" in metricas:
        linhas += _secao_duas_etapas(metricas["duas_etapas"], c)
    if "por_caminho" in metricas:
        linhas += _secao_por_caminho(metricas["por_caminho"], metricas, c)
    linhas += _secao_t1(t1m, c)
    linhas += _secao_t2(t2m, c, observacoes)
    linhas += _secao_t3(t3m, c)
    linhas += [
        "## Acessos à rede", "",
        "Nenhum: o classificador é montado sem pré-treino e lê só o `melhor.pt`; o detector roda "
        "com o modo offline do ultralytics imposto (`detector.importar_ultralytics`).", "",
        "## Limites", "",
        "- O limiar de confiança é calibrado em folhas do BRACOL (fundo claro, uma folha por "
        "foto); no campo (T2) a confiança é outra, e não há rótulo para medir a acurácia ali.",
        "- As predições fora da dobra vêm de modelos de 20 épocas, mais fracos que o run base: "
        "o limiar proposto tende a ser conservador para ele.",
        "- `especie_incorreta` não é produzido: não há dados de outras plantas. Uma foto sem "
        "café pode sair como folha de café com confiança alta.",
        "- O T2 não tem rótulo de classe: as observações dos recortes são uma inspeção visual, "
        "não uma medida.", ""]
    if final:
        linhas += _limitacoes_de_campo(metricas["por_caminho"].get("campo"), c)
    return "\n".join(linhas) + "\n"


def _limitacoes_de_campo(campo: dict | None, c: dict) -> list[str]:
    """As limitações de campo da cadeia final (decisão do gestor, 11/10/2026), com os números
    da validação da regra em duas etapas."""
    linhas = ["### Limitações de campo da cadeia final (decisão do gestor, 11/10/2026)", ""]
    if campo is None:
        return linhas + ["Sem a validação de campo (`campo_validacao_duas_etapas/validacao.json`)."
                         " Os números não puderam ser lidos.", ""]
    novas, juntas, antigas = campo["novas"], campo["juntas"], campo["antigas"]

    def folhas(conjunto):
        return conjunto["S"] + conjunto["D"]

    return linhas + [
        "- **Rótulos:** feitos por Claude por leitura visual das pranchas e aceitos pelo gestor "
        "em 11/10/2026, sem especialista.",
        f"- **Amostra:** {folhas(juntas)} folhas evidentes ({juntas['S']} S e {juntas['D']} D): "
        f"{folhas(antigas)} das fotos de validação do detector, de onde saiu o limiar de "
        f"saudável, e {folhas(novas)} das fotos de treino, a validação. São 1 lavoura e 2 "
        "sessões de fotos.",
        f"- **Falso saudável nas D evidentes:** {_prop(juntas['saudavel_dado_d'])} nas D das "
        f"duas rotulagens; nas novas, {_prop(novas['saudavel_dado_d'])}. Os dados não mostram "
        "que a regra respeita o teto de 5% usado para fixar o limiar de saudável.",
        "- **Folhas ambíguas (\"?\"):** a regra chama saudável "
        f"{_prop(novas['interrogacao_chamada_saudavel'])} das \"?\" novas e "
        f"{_prop(juntas['interrogacao_chamada_saudavel'])} das duas rotulagens juntas. Se elas "
        "forem doentes, o falso saudável sobe: nas novas, com os \"?\" como D, vai a "
        f"{_prop(novas['saudavel_dado_d_interrogacao_como_d'])}.",
        f"- **Recall das saudáveis:** {_prop(juntas['saudavel_dado_s'])} nas S das duas "
        f"rotulagens; nas novas, {_prop(novas['saudavel_dado_s'])}.",
        "- **Classe da doença no campo:** não foi verificada. Os rótulos de campo só dizem S ou "
        f"D, e o limiar de doença ({fmt.decimal(c['limiar_doenca'], 2)}, provisório) só foi "
        "calibrado no BRACOL.",
        "- **Severidade:** não é informativa no campo (passo 4). O nível sai do classificador "
        "treinado no BRACOL, e não há rótulo de severidade no campo.",
        f"- Fonte dos números: `{campo['arquivo']}`.", ""]


def _secao_constantes(const: dict, c: dict) -> list[str]:
    fundo, folha, plantas = const["fundo"], const["folha_unica_bracol"], const["plantas_bracot"]
    linhas = ["", "## Constantes medidas no treino", "",
              f"**Cinza do fundo.** Mediana do tom da borda de {_pct(fundo['borda'], 0)} de cada "
              f"uma das {fmt.n(fundo['imagens'])} imagens de treino do BRACOL, já em 448x224. "
              "O tom do fundo identifica a sessão de fotos (checagem de atalho), por isso cada "
              f"classe pesa igual: cinza neutro {fundo['cinza']} (em uso: {c['cinza_fundo']}).",
              ""]
    linhas += fmt.tabela(["classe", "R", "G", "B", "luminância"], [
        [classe, *(fmt.decimal(v[canal], 0) for canal in ("r", "g", "b")),
         fmt.decimal(v["luminancia"], 0)]
        for classe, v in fundo["por_classe"].items()])
    cor = const["cor_folhas_bracot"]
    linhas += ["", f"**Filtro de cor.** Nas fotos de folha sobre fundo liso do BRACOL, o detector "
                   f"também acha pedaços do fundo. Das {folha['deteccoes_antes_da_cor']} detecções "
                   f"no limiar nas {folha['imagens']} imagens de treino, "
                   f"{folha['cortadas_pela_cor']} têm menos de "
                   f"{_pct(c['fracao_colorida_minima'], 0)} da máscara com saturação >= "
                   f"{c['saturacao_minima']} e saem (histograma da fração colorida, de 0 a 1 em "
                   f"passos de 0,1: {folha['colorida_histograma']}). Nas {cor['folhas']} folhas "
                   f"anotadas de {cor['fotos']} fotos de treino do BRACOT, a fração colorida tem "
                   f"mediana {_pct(cor['mediana'])}, p1 {_pct(cor['p1'])} e mínimo "
                   f"{_pct(cor['min'])}; {cor['abaixo_do_minimo']} ficariam abaixo do mínimo."]
    cob = folha["cobertura"]
    linhas += ["", f"**Margem.** Em {fmt.n(folha['imagens'])} imagens de treino do BRACOL "
                   f"(amostra estratificada, semente {SEED}), a maior folha detectada tem "
                   f"comprimento mediano de {_pct(folha['comprimento_mediano'])} da largura da "
                   f"foto: margem de {_pct(folha['margem'])} de cada lado (em uso: "
                   f"{_pct(c['margem'])}). Detecções por imagem: nenhuma em "
                   f"{folha['sem_deteccao']}, uma em {folha['uma_deteccao']}, mais de uma em "
                   f"{folha['mais_de_uma']}.", "",
               f"**Cobertura da folha única.** A caixa da maior folha cobre de "
               f"{_pct(cob['min'])} a {_pct(cob['max'])} da foto no BRACOL (p5 {_pct(cob['p5'])},"
               f" mediana {_pct(cob['mediana'])}). Nas {plantas['fotos']} fotos de treino do "
               f"BRACOT (sem a validação), a maior caixa cobre no máximo "
               f"{_pct(plantas['cobertura_maior']['max'])} (p95 "
               f"{_pct(plantas['cobertura_maior']['p95'])}), e {plantas['uma_deteccao']} foto(s) "
               f"têm uma só folha utilizável. Regra 1 a partir de "
               f"{_pct(c['cobertura_folha_unica'], 0)}.", ""]
    return linhas


def _secao_calibracao(cal: dict, c: dict) -> list[str]:
    fora, val = cal["fora_da_dobra"], cal["validacao"]
    linhas = ["## Limiar de confiança (RF07)", "",
              f"Curva risco-cobertura: com o limiar t, a folha com probabilidade da classe "
              f"prevista abaixo de t sai da resposta (todas abaixo: `baixa_confianca`). Fonte "
              f"principal: as {fmt.n(fora['n'])} predições fora da dobra do diagnóstico por "
              f"blocos (`{fora['pasta']}`, protocolo aleatório; acurácia "
              f"{_pct(fora['acuracia'])} sem limiar). Conferência: as {val['n']} imagens de "
              "validação no classificador avaliado (imagem inteira, T1).", "",
              "![Curva risco-cobertura](risco_cobertura.png)", ""]
    pontos_fora = {round(p["limiar"], 2): p for p in fora["curva"]}
    pontos_val = {round(p["limiar"], 2): p for p in val["curva"]}
    destaque = sorted({*LIMIARES_DESTAQUE, *(x for x in (cal["proposta"],) if x is not None)})
    linhas += fmt.tabela(
        ["limiar", "cobertura (fora da dobra)", "acurácia das mantidas (IC 95%)",
         *(f"cobertura {cl}" for cl in bracol.CLASSES), "cobertura (val)", "acurácia (val)"],
        [[fmt.decimal(t, 2), _pct(pontos_fora[round(t, 2)]["cobertura"]),
          f"{_pct(pontos_fora[round(t, 2)]['acuracia'])} "
          f"({_pct(pontos_fora[round(t, 2)]['acuracia_ic95'][0])} a "
          f"{_pct(pontos_fora[round(t, 2)]['acuracia_ic95'][1])})",
          *(_pct(pontos_fora[round(t, 2)]["cobertura_por_classe"][cl]) for cl in bracol.CLASSES),
          _pct(pontos_val[round(t, 2)]["cobertura"]), _pct(pontos_val[round(t, 2)]["acuracia"])]
         for t in destaque])
    proposta = cal["proposta"]
    linhas += ["", (f"**Proposta: {fmt.decimal(proposta, 2)}**, o menor limiar em que a "
                    f"acurácia das mantidas fora da dobra chega a {_pct(cal['alvo'], 0)}. "
                    "**A decisão é do gestor.**" if proposta is not None else
                    f"Nenhum limiar chega a {_pct(cal['alvo'], 0)} de acurácia."),
               f"Em uso nesta avaliação: {fmt.decimal(c['limiar_confianca'], 2)}. Arquivo: "
               "`risco_cobertura.csv`.", ""]
    return linhas


def _corte(c: dict) -> str:
    """O corte da regra de decisão do run, para os títulos das tabelas."""
    duas = (f"saudável com P >= {fmt.decimal(c.get('limiar_saudavel', 0), 3)}; doença com P "
            f"renormalizada >= {fmt.decimal(c.get('limiar_doenca') or 0, 2)}")
    if c.get("decisao", "argmax") == "duas_etapas":
        return f"na regra em duas etapas ({duas})"
    if c.get("decisao") == "por_caminho":
        return (f"por caminho (foto inteira e modo A: confiança >= "
                f"{fmt.decimal(c['limiar_confianca'], 2)}; recortes por detecção no modo B: "
                f"{duas})")
    return f"no limiar {fmt.decimal(c['limiar_confianca'], 2)}"


def _valor_de_corte(c: dict) -> str:
    if c.get("decisao", "argmax") == "duas_etapas":
        return "P renormalizada da doença (as saudáveis da 1ª etapa não têm corte)"
    if c.get("decisao") == "por_caminho":
        return ("confiança na foto inteira e no modo A; P renormalizada da doença nos recortes do "
                "modo B (as saudáveis da 1ª etapa não têm corte)")
    return "confiança"


def _secao_por_caminho(pc: dict, metricas: dict, c: dict) -> list[str]:
    t_c, t_s, t_d = (fmt.decimal(c["limiar_confianca"], 2), fmt.decimal(c["limiar_saudavel"], 3),
                     fmt.decimal(c["limiar_doenca"], 2))
    linhas = [
        "## Regra de decisão por caminho (passo 6): a cadeia final", "",
        "Decisão do gestor (11/10/2026): as duas etapas (saudável com P(saudável) >= "
        f"{t_s}; senão, a doença com P renormalizada >= {t_d}, **provisório**) só nos recortes "
        f"por detecção, no modo B; na foto inteira (regras de folha única 1 e 2) e no modo A, o "
        f"argmax com corte {t_c}. Nenhum limiar foi reajustado com dados de campo. As três "
        "regras das tabelas abaixo usam as mesmas probabilidades deste run: só a regra muda. "
        "Por construção, a regra por caminho é igual ao argmax na foto inteira e às duas etapas "
        "nos recortes por detecção.", "",
        "### T1 (validação do BRACOL) por caminho", "",
        f"As {metricas['t1']['imagens']} imagens de validação do BRACOL, pela cadeia no modo B: "
        "a imagem inteira, se a regra 1 ou a 2 valeu, ou o recorte da maior folha (detecção). "
        "Cobertura: imagens com a folha principal mantida no corte; a imagem sem folha (regra 3) "
        "conta como não mantida. \"Chamadas saudáveis\": folha mantida como saudável.", ""]
    linhas += fmt.tabela(
        ["caminho", "regra", "imagens", "acurácia sem corte", "cobertura",
         "acurácia das mantidas (IC 95%)", "doentes chamadas saudáveis (IC 95%)",
         "saudáveis chamadas saudáveis (IC 95%)"],
        [[CAMINHOS[caminho], NOME_REGRA[regra], str(r["imagens"]), _pct(r["acuracia_sem_corte"]),
          _pct(r["cobertura"]["p"]), _prop(r["acuracia_das_mantidas"]),
          _prop(r["saudavel_dado_d"]), _prop(r["saudavel_dado_s"])]
         for caminho in CAMINHOS for regra, por_caminho in pc["t1"].items()
         for r in (por_caminho[caminho],)], "llrrrlll")
    t2r = pc["t2"]
    exemplo = next(iter(t2r.values()))
    linhas += ["", "### T2 (validação do BRACOT, sem rótulo) por regra, no modo B", "",
               f"As {exemplo['folhas']} folhas classificadas das {exemplo['fotos']['total']} "
               "fotos de validação do detector. Sem rótulo de classe: as contagens descrevem o "
               "que a cadeia responde, não se ela acerta.", ""]
    linhas += fmt.tabela(
        ["regra", "folhas saudáveis", "folhas doentes", "folhas descartadas",
         "doenças na resposta", "fotos com resposta", "fotos com alguma doença",
         "fotos em baixa_confianca (corte)", "fotos em baixa_confianca (fragmentos)",
         "planta_nao_identificada"],
        [[NOME_REGRA[regra], str(r["desfechos"]["saudavel"]), str(r["desfechos"]["doenca"]),
          str(r["desfechos"]["descartada"]),
          ", ".join(f"{cl} {r['classes_mantidas'][cl]}" for cl in bracol.CLASSES
                    if cl != "saudavel" and r["classes_mantidas"][cl]) or "-",
          str(r["fotos"]["com_resposta"]), str(r["fotos"]["com_doenca"]),
          str(r["fotos"]["baixa_confianca_corte"]), str(r["fotos"]["baixa_confianca_fragmentos"]),
          str(r["fotos"]["planta_nao_identificada"])]
         for regra, r in t2r.items()], "lrrrlrrrrr")
    linhas += _comparacao_com_runs(pc["runs"], metricas)
    return linhas + [""]


def _comparacao_com_runs(runs: dict, metricas: dict) -> list[str]:
    cadeia = metricas["t1"]["variantes"]["cadeia_B"]
    t2m, t3m = metricas["t2"], metricas["t3"]
    este = {"t1_acuracia_sem_corte": cadeia["metricas"]["acuracia"],
            "t1_mantidas": cadeia["mantidas"],
            "t1_acuracia_das_mantidas": cadeia["acuracia_no_limiar"],
            "t2_folhas": t2m["folhas"],
            "t2_fotos_baixa_confianca": t2m["fotos_baixa_confianca_na_regra"]["B"],
            "t3_mediana_s": t3m["mediana"]["total_s"], "t3_maximo_s": t3m["maximo"]["total_s"]}
    colunas = {**{nome: (r["gravado"] if r else None) for nome, r in runs.items()},
               NOMES["por_caminho"]: este}
    nomes = (("t1_acuracia_sem_corte", "T1, cadeia B: acurácia sem corte", _pct),
             ("t1_mantidas", f"T1, cadeia B: imagens mantidas no corte (de "
                             f"{metricas['t1']['imagens']})", str),
             ("t1_acuracia_das_mantidas", "T1, cadeia B: acurácia das mantidas", _pct),
             ("t2_folhas", "T2: folhas classificadas", str),
             ("t2_fotos_baixa_confianca", f"T2, modo B: fotos em baixa_confianca pelo corte (de "
                                          f"{t2m['fotos']})", str),
             ("t3_mediana_s", "T3: total mediano (s)", lambda v: fmt.decimal(v, 2)),
             ("t3_maximo_s", "T3: total máximo (s)", lambda v: fmt.decimal(v, 2)))
    linhas = ["", "### Comparação com os runs gravados", "",
              "Os números gravados nos runs anteriores, com os mesmos modelos e a mesma cadeia "
              "(recorte B e filtro de fragmentos); só a regra de decisão muda. O T3 é tempo de "
              "máquina e varia de uma rodada para outra.", ""]
    linhas += fmt.tabela(
        ["", *(f"`{n}` ({'gravado' if n != NOMES['por_caminho'] else 'este run'})"
               for n in colunas)],
        [[rotulo, *("-" if col is None or col.get(chave) is None else formato(col[chave])
                    for col in colunas.values())]
         for chave, rotulo, formato in nomes], "l" + "r" * len(colunas))
    conferencias = [f"`{nome}` ({NOME_REGRA[r['regra']]}): "
                    + ("os mesmos números" if r["confere"] else "**números diferentes**: "
                       + ", ".join(f"{k} {r['recalculado'][k]} x {r['gravado'][k]}"
                                   for k in r["recalculado"]
                                   if not math.isclose(r["recalculado"][k], r["gravado"][k],
                                                       abs_tol=1e-6)))
                    for nome, r in runs.items() if r]
    linhas += ["", "**Conferência.** A mesma regra aplicada às probabilidades deste run (T1 e T2) "
               "contra o gravado: " + "; ".join(conferencias) + "."]
    return linhas


def _prop(p: dict | None) -> str:
    """Proporção com o IC de Wilson e a contagem, ou "-"."""
    if not p or p.get("p") is None:
        return "-"
    return (f"{_pct(p['p'])} ({_pct(p['ic95'][0])} a {_pct(p['ic95'][1])}; {p['sucessos']} de "
            f"{p['n']})")


def _secao_duas_etapas(de: dict, c: dict) -> list[str]:
    fora = de["fora_da_dobra"]
    t_s, t_d = de["limiar_saudavel"], de["limiar_doenca"]
    linhas = [
        "## Regra em duas etapas (passo 5b): proposta, desligada por padrão", "",
        "**1ª etapa:** P(saudável) >= limiar_saudavel faz a folha saudável, sem o corte de "
        "confiança. **2ª etapa:** senão, a doença de maior probabilidade com o softmax "
        "renormalizado entre as 4 doenças, cortada por limiar_doenca (abaixo dele a folha sai, "
        "como hoje). O padrão da cadeia continua o argmax com o limiar de confiança "
        "(`inferencia.Configuracao.decisao`).", "",
        f"**Limiar de saudável: {fmt.decimal(t_s, 3)}**, fixado pelo gestor em 11/10/2026 antes "
        "de dados novos e sem reajuste: no máximo 5% das 46 D do campo rotulado marcadas "
        "saudáveis (modelo atual, modo B; `model/runs/campo_diagnostico`). Com >=, 0,083 deixa "
        "passar 2 das 46 D; o valor exato da 3ª D (0,08277) deixaria passar 3 (6,5%). O limiar "
        "saiu das mesmas folhas em que foi medido.", ""]
    if de.get("campo"):
        linhas += ["Sensibilidade (decisão do gestor, 11/10/2026: rótulos congelados, nenhum CSV "
                   "alterado). Sem as 3 S com marca na ponta (" + ", ".join(S_COM_MARCA) + "):",
                   ""]
        linhas += fmt.tabela(
            ["folhas", "S", "D", "limiar recalculado (no máximo 5% das D)",
             f"D com P(saudável) >= {fmt.decimal(t_s, 3)}",
             f"recall de S com P(saudável) >= {fmt.decimal(t_s, 3)}"],
            [[nome, str(r["S"]), str(r["D"]), fmt.decimal(r["limiar_recalculado"], 4),
              _prop(r["d_marcadas_saudaveis"]), _prop(r["recall_s"])]
             for nome, r in de["campo"].items()], "lrrlll")
        linhas += ["", "O limiar recalculado só depende das D, então tirar S não o muda; só o "
                   "recall de S muda.", ""]
    linhas += ["### Efeito do limiar de saudável no BRACOL", "",
               f"As {fmt.n(fora['n'])} predições fora da dobra (protocolo aleatório, modelos de "
               "20 épocas, imagem inteira) e as "
               f"{de['validacao']['n']} imagens de validação no classificador avaliado (imagem "
               "inteira, T1): o argmax contra P(saudável) >= "
               f"{fmt.decimal(t_s, 3)}.", ""]
    nomes = {"fora_da_dobra": "fora da dobra", "validacao": "validação (imagem inteira)"}
    linhas += fmt.tabela(
        ["fonte", "regra", "doentes marcadas saudáveis (IC 95%)", "recall das saudáveis (IC 95%)"],
        [[nomes[fonte], "argmax" if regra == "argmax" else f"P(saudável) >= {fmt.decimal(t_s, 3)}",
          _prop(r["saudavel_dado_d"]), _prop(r["recall_saudavel"])]
         for fonte, por_regra in de["efeito_saudavel"].items()
         for regra, r in por_regra.items()], "llll")
    linhas += ["", "### Limiar de doença (2ª etapa): curva risco-cobertura nas doentes", "",
               f"Só no BRACOL: as {fmt.n(fora['n_doentes'])} folhas doentes das predições fora "
               "da dobra (acurácia da doença renormalizada sem corte: "
               f"{_pct(fora['acuracia_sem_corte'])}). Com o limiar t, a doente com probabilidade "
               "renormalizada abaixo de t sai. Conferência: as "
               f"{de['validacao']['n_doentes']} doentes da validação (imagem inteira, T1).", "",
               "![Curva risco-cobertura do limiar de doença](risco_cobertura_doencas.png)", ""]
    pontos_fora = {round(p["limiar"], 2): p for p in fora["curva"]}
    pontos_val = {round(p["limiar"], 2): p for p in de["validacao"]["curva"]}
    destaque = sorted({*LIMIARES_DOENCA_DESTAQUE, *(x for x in (de["proposta"], t_d)
                                                    if x is not None)})
    doencas = [cl for cl in bracol.CLASSES if cl != "saudavel"]
    linhas += fmt.tabela(
        ["limiar", "cobertura (fora da dobra)", "acurácia das mantidas (IC 95%)",
         *(f"cobertura {cl}" for cl in doencas), "cobertura (val)", "acurácia (val)"],
        [[fmt.decimal(t, 2), _pct(pontos_fora[round(t, 2)]["cobertura"]),
          f"{_pct(pontos_fora[round(t, 2)]['acuracia'])} "
          f"({_pct(pontos_fora[round(t, 2)]['acuracia_ic95'][0])} a "
          f"{_pct(pontos_fora[round(t, 2)]['acuracia_ic95'][1])})",
          *(_pct(pontos_fora[round(t, 2)]["cobertura_por_classe"][cl]) for cl in doencas),
          _pct(pontos_val[round(t, 2)]["cobertura"]), _pct(pontos_val[round(t, 2)]["acuracia"])]
         for t in destaque])
    proposta = de["proposta"]
    linhas += ["", (f"**Proposta: {fmt.decimal(proposta, 2)}**, o menor limiar em que a "
                    f"acurácia das doentes mantidas fora da dobra chega a {_pct(de['alvo'], 0)} "
                    "(o mesmo critério da proposta do limiar de confiança). **Não adotada: a "
                    "decisão é do gestor.**" if proposta is not None else
                    f"Nenhum limiar chega a {_pct(de['alvo'], 0)} de acurácia."),
               f"Em uso nesta avaliação: {fmt.decimal(t_d, 2)}. Arquivo: "
               "`risco_cobertura_doencas.csv`.", "",
               "### A regra completa contra o argmax com o limiar de confiança", "",
               "Todas as predições, saudáveis e doentes: uma saudável com P(saudável) abaixo do "
               "limiar vai para a 2ª etapa e conta como erro.", ""]
    linhas += fmt.tabela(
        ["fonte", "regra", "acurácia sem corte", "cobertura", "acurácia das mantidas (IC 95%)",
         "cobertura das saudáveis", "cobertura das doentes"],
        [[nomes[fonte], (f"duas etapas ({fmt.decimal(t_s, 3)}; {fmt.decimal(t_d, 2)})"
                         if regra == "duas_etapas" else
                         f"argmax ({fmt.decimal(c['limiar_confianca'], 2)})"),
          _pct(r["acuracia_sem_corte"]), _pct(r["cobertura"]), _prop(r["acuracia_das_mantidas"]),
          _pct(r["cobertura_saudaveis"]), _pct(r["cobertura_doentes"])]
         for fonte, por_regra in de["regra_completa"].items()
         for regra, r in por_regra.items()], "lllllll")
    por_limiar = de["regra_completa_por_limiar"]
    linhas += ["", f"A regra completa em outros limiares de doença, com o de saudável fixo em "
               f"{fmt.decimal(t_s, 3)} (cobertura / acurácia das mantidas):", ""]
    linhas += fmt.tabela(
        ["limiar de doença", *(f"{nomes[f]}" for f in por_limiar)],
        [[fmt.decimal(float(t), 2), *(f"{_pct(por_limiar[f][t]['cobertura'])} / "
                                      f"{_pct(por_limiar[f][t]['acuracia_das_mantidas']['p'])}"
                                      for f in por_limiar)]
         for t in por_limiar["fora_da_dobra"]], "lll")
    return linhas + [""]


def _secao_t1(t1m: dict, c: dict) -> list[str]:
    tabela = t1m["variantes"]
    linhas = ["## T1: validação do BRACOL pela cadeia", "",
              f"As {t1m['imagens']} imagens de validação do BRACOL (uma folha por foto, rótulo "
              "conhecido) pela cadeia inteira e pela imagem inteira, com o mesmo código "
              "(`inferencia.classificar`). A folha avaliada é a de maior área. \"Só o recorte\" "
              "desliga a regra 1, para medir o recorte numa folha com rótulo. Regra de decisão: "
              f"{c.get('decisao', 'argmax')}, sem corte; a última coluna aplica o corte "
              f"({_corte(c)}).", ""]
    linhas += fmt.tabela(
        ["variante", "com previsão", "acurácia (IC 95%)", "F1 macro",
         *(f"recall {cl}" for cl in bracol.CLASSES), "doentes previstas saudáveis",
         "severidade: exata / ±1", "no corte: cobertura / acurácia"],
        [[DESCRICAO_T1[v], _pct(r["cobertura_sem_limiar"]),
          f"{_pct(r['metricas']['acuracia'])} ({_ic(r['metricas'])})",
          _pct(r["metricas"]["f1_macro"]), *(_pct(x) for x in r["metricas"]["recall"]),
          _prop(r.get("saudavel_dado_d")),
          f"{_pct(r['severidade']['acuracia_exata'])} / "
          f"{_pct(r['severidade']['acuracia_tolerancia_1'])}",
          f"{_pct(r['cobertura_no_limiar'])} / {_pct(r['acuracia_no_limiar'])}"]
         for v, r in tabela.items()])
    regras = t1m["regras"]
    linhas += ["", "**Regras de folha única nas imagens do BRACOL** (cadeia):", "",
               f"- regra 1 (uma folha, caixa com {_pct(c['cobertura_folha_unica'], 0)} da foto "
               f"ou mais): {regras['regra_1']};",
               f"- regra 2 (nenhuma folha utilizável, alguma detecção com cor de folha e score "
               f">= {fmt.decimal(c['score_minimo_folha_unica'], 2)}): {regras['regra_2']};",
               f"- regra 3 (`planta_nao_identificada`): {regras['regra_3']};",
               f"- pelas detecções: {regras['deteccao']}, das quais {regras['mais_de_uma']} com "
               "mais de uma folha.", "",
               "**Correção das regras depois do T1.** Na primeira rodada, sem o filtro de cor, o "
               "detector achava pedaços do fundo liso do BRACOL como folhas: 135 imagens saíam "
               "com mais de uma folha, e a regra 1 só pegava 105. O filtro de cor (constantes, "
               "acima) tira esses pedaços, e a regra 2 passou a exigir uma detecção com cor de "
               "folha, porque um pedaço de fundo não é sinal de folha.", "",
               f"Detecções utilizáveis por imagem: {t1m['deteccoes_por_imagem']}; o filtro de cor "
               f"tirou {t1m['cortadas_por_cor']} detecções de fundo. Conferência da "
               f"imagem inteira com o `predicoes_val.csv` do run: {t1m['conferencia']}. "
               "Arquivo: `t1_imagens.csv`.", ""]
    return linhas


def _secao_t2(t2m: dict, c: dict, observacoes: str | None) -> list[str]:
    linhas = ["## T2: validação do BRACOT (sem rótulo de classe)", "",
              f"As {t2m['fotos']} fotos de validação do detector (cenas inteiras do treino dos "
              f"autores): {t2m['deteccoes']} detecções no limiar, {t2m['cortadas_por_area']} "
              f"cortadas pela área mínima, {t2m['cortadas_por_cor']} pelo filtro de cor, "
              f"{t2m.get('cortadas_por_fragmento', 0)} pelo filtro de fragmentos, "
              f"{t2m['cortadas_por_limite']} pelo número máximo; "
              f"{t2m['folhas']} folhas classificadas, {t2m['na_borda']} cortadas pela borda. "
              f"Planos: {t2m['planos']}. Classe igual nos dois modos em "
              f"{_pct(t2m['concordancia_A_B'])} das folhas. Regra de decisão: "
              f"{c.get('decisao', 'argmax')}; corte {_corte(c)}; valor de corte: "
              f"{_valor_de_corte(c)}.", ""]
    linhas += fmt.tabela(["", *(f"modo {m}" for m in MODOS)], [
        *([f"folhas {cl}", *(str(t2m[m]["classes"][cl]) for m in MODOS)]
          for cl in bracol.CLASSES),
        *([f"severidade {s}", *(str(t2m[m]["severidades"][s]) for m in MODOS)]
          for s in inf.NIVEIS_CONTRATO),
        ["confiança média / mediana", *(f"{_pct(t2m[m]['confianca_media'])} / "
                                        f"{_pct(t2m[m]['confianca_mediana'])}" for m in MODOS)],
        *([f"mantidas no corte: {cl}", *(str(t2m[m]["classes_mantidas"][cl]) for m in MODOS)]
          for cl in bracol.CLASSES if "classes_mantidas" in t2m["A"]),
        *([f"valor de corte abaixo de {fmt.decimal(float(t), 2)}",
           *(_pct(t2m[m]["abaixo"][t]) for m in MODOS)]
          for t in t2m["A"]["abaixo"]),
        ["cortadas pela borda: confiança média", *(_pct(t2m[m]["borda"]["confianca_media"])
                                                   for m in MODOS)],
        ["inteiras: confiança média", *(_pct(t2m[m]["inteiras"]["confianca_media"])
                                        for m in MODOS)],
        ["cortadas pela borda: saem no corte",
         *(_pct(t2m[m]["borda"]["abaixo_do_limiar"]) for m in MODOS)],
        ["inteiras: saem no corte",
         *(_pct(t2m[m]["inteiras"]["abaixo_do_limiar"]) for m in MODOS)],
        *([f"fotos sem nenhuma folha com valor de corte >= {fmt.decimal(float(t), 2)} "
           "(baixa_confianca)",
           *(f"{t2m['fotos_baixa_confianca'][m][t]} de {t2m['fotos']}" for m in MODOS)]
          for t in t2m["fotos_baixa_confianca"]["A"]),
        *([["**fotos em baixa_confianca na regra do run**",
            *(f"{t2m['fotos_baixa_confianca_na_regra'][m]} de {t2m['fotos']}" for m in MODOS)]]
          if "fotos_baixa_confianca_na_regra" in t2m else []),
    ], "lrr")
    linhas += ["", "Arquivo: `t2_folhas.csv` (uma linha por folha) e `t2_fotos.csv`.", "",
               "### Observações escritas à mão (sem imagens no repositório)", "",
               observacoes or "Ainda não descrito (falta o observacoes.md do run).", ""]
    return linhas


def _secao_t3(t3m: dict, c: dict) -> list[str]:
    pior = t3m["pior_caso"]
    linhas = ["## T3: tempo em CPU (RNF02: diagnóstico em até 5 s)", "",
              f"Nesta máquina, em CPU ({t3m['threads_torch']} threads do torch), nas "
              f"{t3m['fotos']} fotos de validação do BRACOT (4032x3024), depois de uma passada de "
              f"aquecimento. Leitura dos pesos dos dois modelos, uma vez ao subir o backend: "
              f"{fmt.decimal(t3m['carga_dos_modelos_s'], 1)} s, com o torch e o ultralytics já "
              "importados (o import leva alguns segundos a mais na primeira vez).", ""]
    linhas += fmt.tabela(["etapa", "mediana (s)", "máximo (s)"], [
        [nome, fmt.decimal(t3m["mediana"][chave], 2), fmt.decimal(t3m["maximo"][chave], 2)]
        for nome, chave in (("leitura", "leitura_s"), ("detecção", "deteccao_s"),
                            ("recortes", "recortes_s"), ("classificação (lote)", "classificacao_s"),
                            ("contrato", "contrato_s"), ("**total**", "total_s"))])
    linhas += ["", f"Folhas por foto: mediana {fmt.decimal(t3m['folhas_mediana'], 0)}, máximo "
                   f"{t3m['folhas_maximo']}; {t3m['acima_de_5s']} fotos passaram de 5 s. Pior "
                   f"caso, com {pior['folhas']} folhas: a classificação leva "
                   f"{fmt.decimal(pior['classificacao_mediana_s'], 2)} s (máximo "
                   f"{fmt.decimal(pior['classificacao_maximo_s'], 2)} s), e o total estimado "
                   f"(maior leitura, maior detecção, recortes e classificação de {pior['folhas']}"
                   f" folhas) é {fmt.decimal(pior['total_estimado_s'], 2)} s. A CPU do backend "
                   "pode ser mais lenta que esta. Arquivo: `t3_tempos.csv`.", ""]
    return linhas


def _data(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")


def _origem(config: dict) -> str:
    if not config["commit"]:
        return "fora de um repositório git"
    return (f"no commit `{config['commit'][:7]}`"
            + (" (com alterações não commitadas)" if config["commit_com_alteracoes"] else ""))


# ------------------------------------------------------------------------- execução
def conferir_inteira(tabela: pd.DataFrame, arquivo: Path) -> str:
    """A imagem inteira da cadeia contra as predições gravadas pelo caminho do treino
    (train.avaliar, com AMP): o predicoes_val.csv do run ou outro arquivo no mesmo formato."""
    arquivo = Path(arquivo)
    if not arquivo.is_file():
        return f"sem {arquivo.name} no run"
    gravado = pd.read_csv(arquivo).set_index("id")
    colunas = [f"prob_{c}" for c in bracol.CLASSES]
    probs = np.stack(tabela["inteira_probs"])
    referencia = gravado.loc[tabela["id"], colunas].to_numpy()
    iguais = int((probs.argmax(axis=1) == referencia.argmax(axis=1)).sum())
    return (f"diferença máxima das probabilidades {np.abs(probs - referencia).max():.1e} "
            f"(o treino usou AMP), classe prevista igual em {iguais} de {len(tabela)}")


def resumir_t1(tabela: pd.DataFrame, referencia: Path) -> dict:
    """O resumo do T1; referencia: as predições da imagem inteira pelo caminho do treino
    (conferir_inteira)."""
    regras = tabela["regra"].fillna(0).astype(int)
    contagem = tabela["deteccoes"].value_counts().sort_index()
    return {"imagens": len(tabela),
            "variantes": metricas_t1(tabela),
            "regras": {"regra_1": int((regras == 1).sum()), "regra_2": int((regras == 2).sum()),
                       "regra_3": int((regras == 3).sum()),
                       "deteccao": int((tabela["plano"] == "deteccao").sum()),
                       "mais_de_uma": int(((tabela["plano"] == "deteccao")
                                           & (tabela["folhas"] > 1)).sum())},
            "deteccoes_por_imagem": ", ".join(f"{k}: {v}" for k, v in contagem.items()),
            "cortadas_por_cor": int(tabela["cortadas_por_cor"].sum()),
            "conferencia": conferir_inteira(tabela, referencia)}


def nativo(objeto):
    """Tipos do numpy como tipos do Python, em qualquer profundidade (para o json)."""
    if isinstance(objeto, dict):
        return {str(k): nativo(v) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple, np.ndarray)):
        return [nativo(v) for v in objeto]
    if isinstance(objeto, np.generic):
        return objeto.item()
    return objeto


def gravar_csv(tabela: pd.DataFrame, caminho: Path) -> None:
    tabela.to_csv(caminho, index=False, float_format="%.6f", lineterminator="\n")


def gravar_relatorio(pasta: Path, config_run: dict, metricas: dict) -> None:
    """relatorio.md do run, com as observações escritas à mão em observacoes.md."""
    nota = pasta / "observacoes.md"
    (pasta / "relatorio.md").write_text(
        relatorio(config_run, metricas,
                  nota.read_text(encoding="utf-8").strip() if nota.is_file() else None),
        encoding="utf-8", newline="\n")


def refazer_relatorio(nome: str) -> int:
    """Só refaz o relatorio.md a partir do config.json e do metricas.json gravados no run."""
    pasta = train.PASTA_RUNS / nome
    for arquivo in ("config.json", "metricas.json"):
        if not (pasta / arquivo).is_file():
            raise train.ErroFatal(f"{pasta / arquivo} não existe: rode a avaliação antes")
    gravar_relatorio(pasta, *(json.loads((pasta / a).read_text(encoding="utf-8"))
                              for a in ("config.json", "metricas.json")))
    fmt.dizer(f"Relatorio refeito: {pasta / 'relatorio.md'}")
    return 0


def curva_em_tabela(calibracao: dict) -> pd.DataFrame:
    """Os pontos das curvas risco-cobertura (fora da dobra e validação) numa tabela."""
    return pd.DataFrame([{"fonte": fonte, **{k: v for k, v in p.items()
                                             if k not in ("acuracia_ic95", "cobertura_por_classe")},
                          "ic95_baixo": p["acuracia_ic95"][0], "ic95_alto": p["acuracia_ic95"][1],
                          **{f"cobertura_{c}": v for c, v in p["cobertura_por_classe"].items()}}
                         for fonte in ("fora_da_dobra", "validacao")
                         for p in calibracao[fonte]["curva"]])


def configurar(args) -> tuple[inf.Configuracao, dict | None]:
    """A configuração do run: a padrão com a regra de --decisao (por caminho, a cadeia final; ou
    argmax) e o limiar de doença de --limiar-doenca, se vier; ou, com --decisao duas_etapas, a
    regra em duas etapas com o limiar de doença de --limiar-doenca ou, sem ele, o proposto pela
    curva das doenças fora da dobra (a proposta não muda o padrão da cadeia)."""
    if args.decisao != "duas_etapas":
        config = inf.Configuracao(decisao=args.decisao)
        if args.limiar_doenca is not None:
            config = replace(config, limiar_doenca=args.limiar_doenca)
        return config, None
    calibracao = calibrar_doencas(args.fora_da_dobra)
    limiar = args.limiar_doenca if args.limiar_doenca is not None else calibracao["proposta"]
    if limiar is None:
        raise train.ErroFatal("nenhum limiar de doença chega ao alvo: passe --limiar-doenca")
    fmt.dizer(f"Limiar de doenca proposto: {calibracao['proposta']}; em uso: {limiar}")
    return inf.Configuracao(decisao="duas_etapas", limiar_doenca=limiar), calibracao


def executar(args) -> int:
    import torch

    config, calibracao_doencas = configurar(args)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelos = inf.carregar_modelos(args.classificador, args.detector, dispositivo)
    constantes = medir_constantes(modelos, config)
    fmt.dizer(json.dumps({"cinza": constantes["fundo"]["cinza"],
                          "margem": round(constantes["folha_unica_bracol"]["margem"], 4),
                          "cobertura_bracol": constantes["folha_unica_bracol"]["cobertura"],
                          "cobertura_bracot": constantes["plantas_bracot"]["cobertura_maior"],
                          "bracot_uma_folha": constantes["plantas_bracot"]["uma_deteccao"]}))
    if args.so_constantes:
        return 0
    pasta = train.PASTA_RUNS / args.nome
    pasta.mkdir(parents=True, exist_ok=True)
    fmt.dizer("T1: validacao do BRACOL...")
    tabela_t1 = t1(modelos, config)
    inteira = tabela_t1[["classe", "inteira_probs"]].rename(columns={"inteira_probs": "probs"})
    calibracao = calibrar(args.fora_da_dobra, inteira)
    fmt.dizer(f"Limiar proposto: {calibracao['proposta']}")
    duas_etapas = (None if calibracao_doencas is None else
                   completar_duas_etapas(calibracao_doencas, args.fora_da_dobra, inteira, config))
    fmt.dizer("T2: validacao do BRACOT...")
    folhas_t2, fotos_t2 = t2(modelos, config, args.recortes)
    pranchas = []
    if args.recortes:
        pranchas = gravar_pranchas(escolher_para_pranchas(folhas_t2, 2 * RECORTES_POR_PRANCHA),
                                   args.recortes)
    fmt.dizer("T3: tempo em CPU...")
    tabela_t3, resumo_t3 = t3(args.classificador, args.detector, config)
    metricas = {"calibracao": calibracao,
                "t1": resumir_t1(tabela_t1, Path(args.classificador) / "predicoes_val.csv"),
                "t2": resumo_t2(folhas_t2, fotos_t2, config), "t3": resumo_t3}
    if config.decisao == "por_caminho":
        metricas["por_caminho"] = analisar_por_caminho(tabela_t1, folhas_t2, fotos_t2, config)
    gravar_csv(curva_em_tabela(calibracao), pasta / "risco_cobertura.csv")
    metrics.salvar_figura(figura_risco_cobertura(calibracao, config.limiar_confianca),
                          pasta / "risco_cobertura.png")
    if duas_etapas is not None:
        metricas["duas_etapas"] = duas_etapas
        gravar_csv(curva_em_tabela(duas_etapas), pasta / "risco_cobertura_doencas.csv")
        metrics.salvar_figura(
            figura_risco_cobertura(duas_etapas, config.limiar_doenca,
                                   "Curva risco-cobertura do limiar de doença (2ª etapa)",
                                   LIMIARES_DOENCA_DESTAQUE, "folhas doentes"),
            pasta / "risco_cobertura_doencas.png")
    gravar_csv(tabela_t1.drop(columns=[c for c in tabela_t1.columns if c.endswith("_probs")])
               .assign(classe=lambda t: t["classe"].map(lambda k: bracol.CLASSES[k])),
               pasta / "t1_imagens.csv")
    gravar_csv(folhas_t2, pasta / "t2_folhas.csv")
    gravar_csv(fotos_t2, pasta / "t2_fotos.csv")
    gravar_csv(tabela_t3, pasta / "t3_tempos.csv")
    config_run = montar_config(args.nome, config, args.classificador, args.detector,
                               args.fora_da_dobra, dispositivo, constantes)
    config_run, metricas = nativo(config_run), nativo(metricas)
    train.gravar_json(config_run, pasta / "config.json")
    train.gravar_json(metricas, pasta / "metricas.json")
    gravar_relatorio(pasta, config_run, metricas)
    for v in VARIANTES_T1:
        m = metricas["t1"]["variantes"][v]["metricas"]
        fmt.dizer(f"T1 {v}: acuracia {_pct(m['acuracia'])}, F1 macro {_pct(m['f1_macro'])}")
    fmt.dizer(f"T3: total mediano {fmt.decimal(resumo_t3['mediana']['total_s'], 2)} s, maximo "
              f"{fmt.decimal(resumo_t3['maximo']['total_s'], 2)} s")
    for caminho in pranchas:
        fmt.dizer(f"Prancha: {caminho}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Avaliacao ponta a ponta da inferencia (detector + recorte + classificador). "
                    "Detalhes no docstring de model/avaliar_pipeline.py.")
    padrao = inf.Configuracao().decisao
    ap.add_argument("--nome", help="pasta em model/runs (padrao: o run da regra de decisao: "
                                   + ", ".join(f"{d}: {n}" for d, n in NOMES.items()) + ")")
    ap.add_argument("--decisao", choices=inf.DECISOES, default=padrao,
                    help=f"regra de decisao do run (padrao: {padrao}, a da cadeia final; argmax "
                         "e duas_etapas valem em todo caminho)")
    ap.add_argument("--limiar-doenca", type=float,
                    help="com --decisao duas_etapas ou por_caminho: o limiar de doenca (sem ele, "
                         "o da Configuracao em por_caminho e o proposto pela curva das doencas "
                         "fora da dobra em duas_etapas)")
    ap.add_argument("--classificador", type=Path, default=inf.PASTA_CLASSIFICADOR,
                    help="pasta do run do classificador (com o melhor.pt)")
    ap.add_argument("--detector", type=Path, default=detector.PESOS_PADRAO,
                    help="pesos do detector (last.pt)")
    ap.add_argument("--fora-da-dobra", type=Path, default=PASTA_FORA_DA_DOBRA,
                    help="pasta com as predicoes_aleatorio_*.csv do diagnostico por blocos")
    ap.add_argument("--recortes", type=Path,
                    help="pasta FORA do repositorio para os recortes do T2 e as pranchas")
    ap.add_argument("--so-constantes", action="store_true",
                    help="so mede e mostra as constantes (nao grava nada)")
    ap.add_argument("--relatorio", action="store_true",
                    help="so refaz o relatorio.md do run (--nome) a partir do config.json e do "
                         "metricas.json, com o observacoes.md")
    args = ap.parse_args(argv)
    if args.limiar_doenca is not None and args.decisao == "argmax":
        ap.error("--limiar-doenca nao vale com --decisao argmax")
    if args.nome is None:
        args.nome = NOMES[args.decisao]
    if args.recortes and args.recortes.resolve().is_relative_to(RAIZ_REPO):
        ap.error("--recortes precisa ficar fora do repositorio (imagens do dataset)")
    try:
        if args.relatorio:
            return refazer_relatorio(args.nome)
        return executar(args)
    except (train.ErroFatal, FileNotFoundError) as e:
        fmt.dizer(f"ERRO: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
