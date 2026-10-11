"""
Inferência ponta a ponta (Frente 9, passo 3): a função que o backend chama. Recebe a foto de uma
planta, de um galho ou de uma folha e devolve o diagnóstico no contrato dos frontends
(coffea-web: docs/frontend-reference/03-contrato-api-mock.md e src/api/types.ts).

    sucesso: {"status": "sucesso", "folhas": [{"id", "categoria", "severidade",
              "regiao": {"x", "y", "raio"}, "confianca", "scoreDeteccao", "caixa",
              "cortadaNaBorda"}], "detalhes": {...}}
    erro:    {"status": "erro", "tipo", "mensagem", "detalhes"}

- O imagemUrl do contrato é do backend, que guarda a foto; sem ele, o frontend usa a foto local.
- Campos além dos do contrato (confianca, scoreDeteccao, caixa, cortadaNaBorda, detalhes) são
  permitidos: o normalizador do frontend os ignora.
- Erros produzidos: planta_nao_identificada, baixa_confianca (RF07), formato_invalido e
  arquivo_muito_grande (mais de max_megapixels). especie_incorreta não é produzido: não há dados
  de outras plantas para treinar essa recusa. O limite de 10 MB do arquivo fica no backend.

Cadeia: detector (model/detector.py) -> área mínima e número máximo de folhas -> regras de folha
única -> recorte de cada folha, deitada e com margem como no BRACOL -> classificador, num lote só
-> regra de decisão e corte (argmax com limiar de confiança; ou, desligada por padrão, as duas
etapas: saudável por P(saudável), senão a doença com o softmax renormalizado) -> contrato.

Uso (com model/ no sys.path):
    import inferencia
    inferencia.carregar_modelos()                         # uma vez, ao subir o backend
    resposta = inferencia.diagnosticar_arquivo(conteudo)  # bytes do upload ou caminho

Os pesos ficam fora do git: o melhor.pt do classificador (PASTA_CLASSIFICADOR) e o last.pt do
detector (detector.PESOS_PADRAO). Outros caminhos vão em carregar_modelos.
"""
import io
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RAIZ_REPO = Path(__file__).resolve().parents[1]
if str(RAIZ_REPO / "data") not in sys.path:
    sys.path.insert(0, str(RAIZ_REPO / "data"))

import bracol
import dados
import detector

PASTA_CLASSIFICADOR = RAIZ_REPO / "model" / "runs" / "base_resnet50_bracol"
CATEGORIAS = tuple(bracol.CLASSES)  # os nomes de CLASSES já são os do contrato
# Níveis 0 a 4 do modelo (bracol.SEVERIDADES) no contrato. O BRACOL só tem nível 0 nas folhas
# saudáveis e só 1 a 4 nas doentes (data/reports/eda_bracol.md, seção 3).
NIVEIS_CONTRATO = ("saudavel", "muito_baixa", "baixa", "alta", "muito_alta")
SAUDAVEL = CATEGORIAS.index("saudavel")
DOENCAS = tuple(i for i in range(len(CATEGORIAS)) if i != SAUDAVEL)
DECISOES = ("argmax", "duas_etapas")  # regras de decisão (Configuracao.decisao)
RAIO_MINIMO, RAIO_MAXIMO = 0.01, 0.5  # o mesmo limite do normalizador do frontend
LADO_TRABALHO = 2048  # a foto é reduzida a este lado maior antes dos recortes (o do BRACOL)
LADO_COR = 1024  # lado maior da cópia reduzida em que se mede a cor das máscaras
BORDA_DA_FOTO = 0.005  # fração do lado maior: polígono a até isto da borda = folha cortada
DILATACAO_MASCARA = 0.01  # fração do comprimento do recorte (modo B)
MENSAGENS = {
    "planta_nao_identificada": "Não encontramos folhas de café na foto. Fotografe a planta ou a "
                               "folha mais de perto, com boa luz.",
    "baixa_confianca": "Encontramos folhas, mas o diagnóstico não ficou confiável. Tente outra "
                       "foto, mais nítida e com boa luz.",
    "formato_invalido": "Não foi possível ler a imagem. Envie uma foto em JPG ou PNG.",
    "arquivo_muito_grande": "A imagem tem {largura} x {altura} pixels ({megapixels} "
                            "megapixels); o limite é {limite} megapixels.",
    # baixa_confianca quando o filtro de fragmentos tiraria todas as folhas
    "fragmentos": "Só aparecem pedaços de folha na foto. Enquadre a folha inteira e tente de "
                  "novo.",
}


@dataclass(frozen=True)
class Configuracao:
    """Parâmetros da cadeia. Os padrões e a origem de cada um estão no relatório de
    model/runs/pipeline_padrao_b/ (o de model/runs/pipeline_base/ é o do passo 3)."""
    limiar_deteccao: float = 0.35  # decisão do gestor, 10/10/2026 (run detector_base)
    max_folhas: int = 15  # as de maior score do detector (RNF02)
    area_minima: float = 0.005  # fração da foto ocupada pela máscara
    # Filtro de cor: nas fotos de folha sobre fundo liso (as do BRACOL), o detector também acha
    # pedaços do fundo. Folha utilizável tem ao menos fracao_colorida_minima da máscara com
    # saturação (HSV do OpenCV, 0 a 255) >= saturacao_minima. Medido no treino em 10/10/2026:
    # nas folhas anotadas do BRACOT, a mediana é 99% e o p1, 35%; no fundo do BRACOL, < 10%.
    saturacao_minima: int = 30
    fracao_colorida_minima: float = 0.25
    # RF07: probabilidade mínima da classe prevista. 0,61 é a PROPOSTA da curva risco-cobertura
    # (model/runs/pipeline_base/relatorio.md) e aguarda a decisão do gestor.
    limiar_confianca: float = 0.61
    # "B": fundo fora da máscara em cinza; "A": a foto em volta. B por padrão (decisão do gestor,
    # 11/10/2026): nos rótulos de campo do passo 4, P(prever doença | S) cai de 100% (A) para 69%
    # (B) no modelo oficial, com intervalos que não se tocam. No BRACOL (T1) o A rende mais,
    # porque o fundo das fotos carrega a sessão.
    recorte: str = "B"
    # Medidos no treino do BRACOL (model/avaliar_pipeline.py, 10/10/2026): a folha ocupa 74% da
    # largura da foto (margem de 17% de cada lado) e o fundo tem luminância mediana 197.
    margem: float = 0.17  # de cada lado, em fração do comprimento da folha
    cinza_fundo: int = 197  # cinza neutro do modo B e do que fica fora da foto
    cobertura_folha_unica: float = 0.30  # regra 1: fração da foto coberta pela caixa
    score_minimo_folha_unica: float = 0.05  # regra 2 (decisão do gestor, 10/10/2026)
    # Filtro de fragmentos (folhas cortadas pela borda): fração visível = área da folha / mediana
    # da área das folhas inteiras da mesma foto. Ligado em 0,5 (decisão do gestor, 11/10/2026):
    # nos rótulos de campo, 51% de "?" entre as que ele corta, contra 24% entre as que ficam. Uma
    # folha sozinha tem fração visível 1 e nunca sai. O preenchimento (área / retângulo mínimo)
    # não discriminou e fica desligado.
    fracao_visivel_minima: float = 0.5
    preenchimento_minimo: float = 0.0
    max_megapixels: float = 64.0
    # Regra de decisão (passo 5b; decisões do gestor de 11/10/2026). "argmax" (padrão): a classe
    # de maior probabilidade, cortada por limiar_confianca. "duas_etapas", desligada até a
    # validação no campo: P(saudável) >= limiar_saudavel faz a folha "saudavel", sem corte;
    # senão, a doença de maior probabilidade com o softmax renormalizado entre as 4 doenças,
    # cortada por limiar_doenca (abaixo dele a folha sai, como hoje). Ver aplicar_decisao.
    decisao: str = "argmax"
    # Fixado em 11/10/2026, antes de dados novos, e não se reajusta: no máximo 5% das 46 D do
    # campo rotulado marcadas saudáveis (modelo atual, modo B; model/runs/campo_diagnostico).
    # Com >=, 0,083 deixa passar 2 das 46 D e 30 das 36 S; o valor exato da 3ª D (0,08277)
    # deixaria passar 3 D (6,5%).
    limiar_saudavel: float = 0.083
    # Sem padrão: a proposta sai da curva risco-cobertura das doenças no BRACOL
    # (model/avaliar_pipeline.py --decisao duas_etapas) e espera a decisão do gestor.
    limiar_doenca: float | None = None

    def __post_init__(self):
        if self.recorte not in ("A", "B"):
            raise ValueError(f"recorte desconhecido: {self.recorte!r} (use 'A' ou 'B')")
        if self.decisao not in DECISOES:
            raise ValueError(f"decisão desconhecida: {self.decisao!r} (use {DECISOES})")
        if self.decisao == "duas_etapas" and self.limiar_doenca is None:
            raise ValueError("a decisão em duas etapas precisa de limiar_doenca")


@dataclass(frozen=True)
class Modelos:
    classificador: object  # train.ModeloMultitarefa, em eval
    transformacao: object  # dados.transformacao_avaliacao no tamanho do checkpoint
    detector: object  # o YOLO do ultralytics
    dispositivo: object  # torch.device
    nome_classificador: str
    nome_detector: str


_modelos = {}


# --------------------------------------------------------------------------- modelos
def carregar_modelos(pasta_do_classificador=PASTA_CLASSIFICADOR,
                     pesos_do_detector=detector.PESOS_PADRAO, dispositivo="cpu") -> Modelos:
    """Classificador (melhor.pt da pasta do run) e detector, uma vez por combinação de caminhos e
    dispositivo. Só lê arquivos locais: o classificador é montado sem pré-treino e o ultralytics
    roda com a rede desligada (detector.importar_ultralytics)."""
    import torch

    import train

    pasta = Path(pasta_do_classificador).resolve()
    pesos = Path(pesos_do_detector).resolve()
    dispositivo = torch.device(dispositivo)
    chave = (pasta, pesos, str(dispositivo))
    if chave not in _modelos:
        classificador, checkpoint = train.carregar_modelo(pasta, dispositivo)
        tamanho = checkpoint["tamanho_entrada"]
        _modelos[chave] = Modelos(
            classificador=classificador,
            transformacao=dados.transformacao_avaliacao(tamanho["largura"], tamanho["altura"]),
            detector=detector.carregar_detector(pesos), dispositivo=dispositivo,
            nome_classificador=pasta.name, nome_detector=_nome_do_detector(pesos))
    return _modelos[chave]


def _nome_do_detector(pesos: Path) -> str:
    """O nome do run (model/runs/<run>/weights/last.pt) ou o do arquivo."""
    return pesos.parents[1].name if pesos.parent.name == "weights" else pesos.name


# ------------------------------------------------------------------------- entradas
def diagnosticar_arquivo(entrada, config: Configuracao | None = None,
                         modelos: Modelos | None = None) -> dict:
    """diagnosticar a partir dos bytes do arquivo ou de um caminho. As dimensões vêm do
    cabeçalho, sem decodificar a imagem: acima de config.max_megapixels, o erro é
    arquivo_muito_grande. A decodificação é a do projeto (dados.decodificar_imagem: imdecode
    com a orientação EXIF)."""
    config = config or Configuracao()
    em_memoria = isinstance(entrada, (bytes, bytearray, memoryview))
    if not em_memoria and not Path(entrada).is_file():
        raise FileNotFoundError(f"imagem não encontrada: {entrada}")
    try:
        largura, altura = dimensoes_pelo_cabecalho(entrada)
    except ImagemGrandeDemais as e:
        return resposta_de_erro("arquivo_muito_grande", str(e))
    except (OSError, ValueError, SyntaxError):
        return resposta_de_erro("formato_invalido")
    if largura * altura > config.max_megapixels * 1e6:
        return resposta_de_erro("arquivo_muito_grande", MENSAGENS["arquivo_muito_grande"].format(
            largura=largura, altura=altura, megapixels=_decimal(largura * altura / 1e6),
            limite=_decimal(config.max_megapixels)))
    try:
        if em_memoria:
            imagem = dados.decodificar_imagem(entrada)
        else:
            imagem = dados.ler_imagem(entrada)
    except ValueError:
        return resposta_de_erro("formato_invalido")
    return diagnosticar(imagem, config, modelos)


class ImagemGrandeDemais(Exception):
    """A imagem passa do limite do próprio Pillow, que nem lê as dimensões."""


def dimensoes_pelo_cabecalho(entrada) -> tuple[int, int]:
    """(largura, altura) lidas do cabeçalho pelo Pillow, sem decodificar os pixels."""
    from PIL import Image

    fonte = io.BytesIO(entrada) if isinstance(entrada, (bytes, bytearray, memoryview)) else entrada
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        try:
            with Image.open(fonte) as imagem:
                return imagem.size
        except Image.DecompressionBombError as e:
            raise ImagemGrandeDemais(
                f"A imagem passa de {_decimal(2 * Image.MAX_IMAGE_PIXELS / 1e6)} megapixels, "
                "mais do que qualquer foto de celular.") from e


# --------------------------------------------------------------------------- cadeia
def diagnosticar(imagem_rgb, config: Configuracao | None = None,
                 modelos: Modelos | None = None) -> dict:
    """O diagnóstico de uma foto RGB (altura x largura x 3, uint8) no contrato dos frontends."""
    config = config or Configuracao()
    modelos = modelos or carregar_modelos()
    imagem = np.asarray(imagem_rgb)
    if imagem.ndim != 3 or imagem.shape[2] != 3 or imagem.dtype != np.uint8:
        return resposta_de_erro("formato_invalido")
    deteccoes = detectar(imagem, config, modelos)
    preparo = decidir(deteccoes, imagem.shape[0], imagem.shape[1], config)
    if preparo["plano"] == "fragmentos":
        return resposta_de_erro("baixa_confianca", MENSAGENS["fragmentos"],
                                _detalhes(preparo, config, modelos))
    if not preparo["folhas"]:
        return resposta_de_erro("planta_nao_identificada", detalhes=_detalhes(preparo, config,
                                                                                modelos))
    probabilidades, niveis = classificar(recortar(imagem, preparo, config), modelos)
    return montar_resposta(preparo, probabilidades, niveis, config, modelos)


def detectar(imagem, config: Configuracao, modelos: Modelos) -> list[dict]:
    """As detecções do detector até o menor dos dois limiares (o da cadeia e o da regra 2), com
    a área (fração da foto) e se a folha está cortada pela borda."""
    limiar = min(config.limiar_deteccao, config.score_minimo_folha_unica)
    dispositivo = 0 if modelos.dispositivo.type == "cuda" else "cpu"
    folhas = detector.detectar_folhas(imagem, limiar, detector=modelos.detector,
                                      dispositivo=dispositivo)
    altura, largura = imagem.shape[:2]
    reduzida, escala = reduzir(imagem, LADO_COR)
    saturacao = cv2.cvtColor(reduzida, cv2.COLOR_RGB2HSV)[..., 1]
    return [{**folha, "area": area_do_poligono(folha["poligono"]) / (altura * largura),
             "borda": cortada_pela_borda(folha["poligono"], altura, largura),
             "colorida": fracao_colorida(saturacao, np.asarray(folha["poligono"]) * escala,
                                         config.saturacao_minima)}
            for folha in folhas]


def utilizavel(deteccao: dict, config: Configuracao) -> bool:
    """Folha que entra na cadeia: no limiar do detector, com a área mínima e com cor de folha
    (o filtro de cor tira os pedaços de fundo liso)."""
    return (deteccao["score"] >= config.limiar_deteccao and deteccao["area"] >= config.area_minima
            and deteccao["colorida"] >= config.fracao_colorida_minima)


def decidir(deteccoes, altura: int, largura: int, config: Configuracao) -> dict:
    """Que folhas classificar, pelas regras da cadeia (puro: testável sem modelos).

    - Folhas utilizáveis (utilizavel: limiar, área mínima e cor), as max_folhas de maior score.
    - Regra 1: uma só folha utilizável, com a caixa cobrindo pelo menos cobertura_folha_unica
      da foto: classifica a foto inteira (foto de uma folha só, como as do BRACOL).
    - Regra 2: nenhuma folha utilizável, mas alguma detecção com cor de folha e score >=
      score_minimo_folha_unica: a foto inteira.
    - Regra 3: nenhuma das duas: sem folhas (planta_nao_identificada).
    - Se o filtro de fragmentos tirou todas as folhas utilizáveis: plano "fragmentos" (o
      diagnóstico vira baixa_confianca, pedindo para enquadrar a folha inteira).
    Devolve {"plano", "regra", "folhas", "contagens", "altura", "largura"}; cada folha da foto
    inteira tem "foto_inteira": True.
    """
    ordenadas = sorted(deteccoes, key=lambda d: -d["score"])
    no_limiar = [d for d in ordenadas if d["score"] >= config.limiar_deteccao]
    com_area = [d for d in no_limiar if d["area"] >= config.area_minima]
    coloridas = [d for d in com_area if utilizavel(d, config)]
    utilizaveis = [d for d in coloridas if not fragmento(d, coloridas, config)]
    contagens = {"deteccoes": len(no_limiar), "abaixo_do_limiar": len(ordenadas) - len(no_limiar),
                 "cortadas_por_area": len(no_limiar) - len(com_area),
                 "cortadas_por_cor": len(com_area) - len(coloridas),
                 "cortadas_por_fragmento": len(coloridas) - len(utilizaveis),
                 "cortadas_por_limite": max(0, len(utilizaveis) - config.max_folhas)}
    base = {"altura": altura, "largura": largura, "contagens": contagens}
    if len(utilizaveis) == 1 and cobertura_da_caixa(utilizaveis[0]["caixa"], altura,
                                                    largura) >= config.cobertura_folha_unica:
        return {**base, "plano": "foto_inteira", "regra": 1,
                "folhas": [{**utilizaveis[0], "foto_inteira": True}]}
    if coloridas and not utilizaveis:  # o filtro de fragmentos tirou todas
        return {**base, "plano": "fragmentos", "regra": None, "folhas": []}
    if not utilizaveis and any(d["score"] >= config.score_minimo_folha_unica
                               and d["colorida"] >= config.fracao_colorida_minima
                               for d in ordenadas):
        return {**base, "plano": "foto_inteira", "regra": 2,
                "folhas": [{"foto_inteira": True, "poligono": None, "caixa": None,
                            "score": None, "area": 1.0, "borda": False, "colorida": None}]}
    return {**base, "plano": "deteccao" if utilizaveis else "nenhuma",
            "regra": None if utilizaveis else 3, "folhas": utilizaveis[:config.max_folhas]}


def recortar(imagem, preparo: dict, config: Configuracao, modo: str | None = None) -> list:
    """Os recortes RGB (2:1) das folhas do preparo, no modo da configuração ou no pedido."""
    modo = modo or config.recorte
    if preparo["plano"] == "foto_inteira":
        return [recorte_da_foto_inteira(imagem, config.cinza_fundo)]
    reduzida, escala = reduzir(imagem, LADO_TRABALHO)
    return [recortar_folha(reduzida, np.asarray(folha["poligono"]) * escala, modo,
                           config.margem, config.cinza_fundo)
            for folha in preparo["folhas"]]


def classificar(recortes, modelos: Modelos) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades das classes (n x 5, na ordem de CATEGORIAS) e nível de severidade previsto
    (0 a 4) de cada recorte, num lote só, em float32."""
    import torch

    lote = torch.stack([modelos.transformacao(image=r)["image"] for r in recortes])
    with torch.inference_mode():
        logits_classe, logits_sev = modelos.classificador(lote.to(modelos.dispositivo))
    return (torch.softmax(logits_classe.float(), dim=1).cpu().numpy(),
            logits_sev.argmax(dim=1).cpu().numpy())


def aplicar_decisao(probs, config: Configuracao) -> tuple[int, float, float]:
    """A regra de decisão de uma folha (puro): (classe em CATEGORIAS, confiança informada no
    contrato, valor comparado com limiar_de_corte(config); a folha sai se ele ficar abaixo).

    - argmax: a classe de maior probabilidade; a confiança e o valor de corte são essa
      probabilidade.
    - duas_etapas: com P(saudável) >= limiar_saudavel, "saudavel", com confiança P(saudável) e
      sem corte (valor de corte infinito); senão, a doença de maior probabilidade entre as 4,
      com as probabilidades delas renormalizadas para somar 1; a confiança e o valor de corte
      são essa probabilidade renormalizada.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if config.decisao == "argmax":
        classe = int(np.argmax(probs))
        return classe, float(probs[classe]), float(probs[classe])
    if probs[SAUDAVEL] >= config.limiar_saudavel:
        return SAUDAVEL, float(probs[SAUDAVEL]), math.inf
    doencas = probs[list(DOENCAS)]
    renormalizadas = doencas / doencas.sum() if doencas.sum() > 0 else np.full(len(DOENCAS),
                                                                             1 / len(DOENCAS))
    k = int(np.argmax(renormalizadas))
    return DOENCAS[k], float(renormalizadas[k]), float(renormalizadas[k])


def limiar_de_corte(config: Configuracao) -> float:
    """O limiar que a regra de decisão aplica ao valor de corte de aplicar_decisao."""
    return config.limiar_doenca if config.decisao == "duas_etapas" else config.limiar_confianca


def montar_resposta(preparo: dict, probabilidades, niveis, config: Configuracao,
                    modelos: Modelos | None = None) -> dict:
    """O contrato a partir das folhas do preparo e das saídas do classificador: aplica a regra
    de decisão com o corte dela (aplicar_decisao) e numera as folhas que ficam, na ordem de
    score."""
    altura, largura = preparo["altura"], preparo["largura"]
    folhas = []
    for folha, probs, nivel in zip(preparo["folhas"], probabilidades, niveis):
        classe, confianca, corte = aplicar_decisao(probs, config)
        if corte < limiar_de_corte(config):
            continue
        categoria = CATEGORIAS[classe]
        folhas.append({
            "id": f"folha-{len(folhas) + 1}",
            "categoria": categoria,
            "severidade": severidade_do_contrato(categoria, int(nivel)),
            "regiao": regiao_da_folha(folha["poligono"], altura, largura),
            "confianca": round(confianca, 4),
            "scoreDeteccao": None if folha["score"] is None else round(float(folha["score"]), 4),
            "caixa": caixa_relativa(folha["caixa"], altura, largura),
            "cortadaNaBorda": bool(folha["borda"]),
        })
    detalhes = _detalhes(preparo, config, modelos,
                         abaixo_da_confianca=len(preparo["folhas"]) - len(folhas))
    if not folhas:
        return resposta_de_erro("baixa_confianca", detalhes=detalhes)
    return {"status": "sucesso", "folhas": folhas, "detalhes": detalhes}


def resposta_de_erro(tipo: str, mensagem: str | None = None, detalhes: dict | None = None) -> dict:
    resposta = {"status": "erro", "tipo": tipo, "mensagem": mensagem or MENSAGENS[tipo]}
    if detalhes is not None:
        resposta["detalhes"] = detalhes
    return resposta


def _detalhes(preparo: dict, config: Configuracao, modelos, abaixo_da_confianca: int = 0):
    limiares = ({"limiar_saudavel": config.limiar_saudavel,
                 "limiar_doenca": config.limiar_doenca} if config.decisao == "duas_etapas"
                else {"limiar_confianca": config.limiar_confianca})
    return {"plano": preparo["plano"], "regra": preparo["regra"], **preparo["contagens"],
            "abaixo_da_confianca": abaixo_da_confianca, "recorte": config.recorte,
            "decisao": config.decisao, **limiares,
            "classificador": modelos.nome_classificador if modelos else None,
            "detector": modelos.nome_detector if modelos else None}


# ------------------------------------------------------------------------- contrato
def severidade_do_contrato(categoria: str, nivel: int) -> str:
    """O nível do modelo no contrato, coerente com a categoria, como o normalizador do
    frontend: folha saudável tem severidade "saudavel"; folha doente com nível 0 vira
    "muito_baixa"."""
    if categoria == "saudavel":
        return "saudavel"
    return NIVEIS_CONTRATO[max(1, min(int(nivel), len(NIVEIS_CONTRATO) - 1))]


def regiao_da_folha(poligono, altura: int, largura: int) -> dict:
    """O círculo da folha no contrato: centro no centroide da máscara (relativo à largura e à
    altura) e raio do círculo de mesma área, relativo à menor dimensão, em [0,01; 0,5]. Sem
    polígono (a foto inteira), o círculo do meio da foto."""
    if poligono is None:
        return {"x": 0.5, "y": 0.5, "raio": RAIO_MAXIMO}
    pontos = np.asarray(poligono, dtype=np.float32).reshape(-1, 2)
    momentos = cv2.moments(pontos)
    if momentos["m00"] > 0:
        cx, cy = momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"]
    else:
        cx, cy = pontos.mean(axis=0)
    raio = math.sqrt(area_do_poligono(pontos) / math.pi) / min(altura, largura)
    return {"x": round(float(np.clip(cx / largura, 0, 1)), 4),
            "y": round(float(np.clip(cy / altura, 0, 1)), 4),
            "raio": round(float(np.clip(raio, RAIO_MINIMO, RAIO_MAXIMO)), 4)}


def caixa_relativa(caixa, altura: int, largura: int) -> list[float]:
    """[x0, y0, x1, y1] relativos (0 a 1) à largura e à altura; a foto inteira sem caixa."""
    if caixa is None:
        return [0.0, 0.0, 1.0, 1.0]
    x0, y0, x1, y1 = caixa
    return [round(float(np.clip(v, 0, 1)), 4)
            for v in (x0 / largura, y0 / altura, x1 / largura, y1 / altura)]


# ------------------------------------------------------------------------ geometria
def area_do_poligono(poligono) -> float:
    return float(abs(cv2.contourArea(np.asarray(poligono, dtype=np.float32).reshape(-1, 2))))


def fracao_colorida(saturacao, poligono, saturacao_minima: int) -> float:
    """Fração dos pixels da máscara do polígono com saturação >= saturacao_minima (saturacao: o
    canal S do HSV do OpenCV; poligono na escala dele). Máscara vazia: 0."""
    mascara = np.zeros(saturacao.shape, dtype=np.uint8)
    cv2.fillPoly(mascara, [np.round(np.asarray(poligono, dtype=np.float64).reshape(-1, 2))
                           .astype(np.int32)], 1)
    valores = saturacao[mascara.astype(bool)]
    return float((valores >= saturacao_minima).mean()) if valores.size else 0.0


def fracao_visivel(folha: dict, folhas) -> float:
    """Área da folha sobre a mediana da área das folhas inteiras (não cortadas pela borda) da
    mesma foto; sem nenhuma inteira, sobre a mediana de todas."""
    inteiras = [f["area"] for f in folhas if not f["borda"]]
    referencia = float(np.median(inteiras or [f["area"] for f in folhas]))
    return folha["area"] / referencia if referencia > 0 else 0.0


def preenchimento(poligono) -> float:
    """Área do polígono sobre a área do seu retângulo mínimo (cv2.minAreaRect)."""
    pontos = np.asarray(poligono, dtype=np.float32).reshape(-1, 2)
    _, (largura, altura), _ = cv2.minAreaRect(pontos)
    return area_do_poligono(pontos) / (largura * altura) if largura * altura > 0 else 0.0


def fragmento(folha: dict, folhas, config: Configuracao) -> bool:
    """Se a folha é um fragmento pelo filtro de fragmentos (desligado com os limiares em 0)."""
    if config.fracao_visivel_minima <= 0 and config.preenchimento_minimo <= 0:
        return False
    return (fracao_visivel(folha, folhas) < config.fracao_visivel_minima
            or preenchimento(folha["poligono"]) < config.preenchimento_minimo)


def cobertura_da_caixa(caixa, altura: int, largura: int) -> float:
    x0, y0, x1, y1 = caixa
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / (altura * largura)


def cortada_pela_borda(poligono, altura: int, largura: int, borda: float = BORDA_DA_FOTO) -> bool:
    """Se algum ponto do polígono está a até `borda` do lado maior da borda da foto."""
    pontos = np.asarray(poligono, dtype=np.float64).reshape(-1, 2)
    folga = borda * max(altura, largura)
    return bool((pontos[:, 0] <= folga).any() or (pontos[:, 1] <= folga).any()
                or (pontos[:, 0] >= largura - folga).any()
                or (pontos[:, 1] >= altura - folga).any())


def reduzir(imagem, lado: int) -> tuple[np.ndarray, float]:
    """A imagem com o lado maior reduzido a `lado` (INTER_AREA), se for maior, e a escala."""
    altura, largura = imagem.shape[:2]
    escala = min(1.0, lado / max(altura, largura))
    if escala == 1.0:
        return imagem, 1.0
    tamanho = (round(largura * escala), round(altura * escala))
    return cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA), escala


def eixo_da_folha(poligono) -> tuple[tuple[float, float], float, float, float]:
    """Centro, ângulo (graus) do eixo maior em relação à horizontal, comprimento e largura do
    retângulo mínimo (cv2.minAreaRect) do polígono."""
    retangulo = cv2.minAreaRect(np.asarray(poligono, dtype=np.float32).reshape(-1, 2))
    p0, p1, p2 = cv2.boxPoints(retangulo)[:3]
    lado_a, lado_b = p1 - p0, p2 - p1
    maior, menor = ((lado_a, lado_b) if np.hypot(*lado_a) >= np.hypot(*lado_b)
                    else (lado_b, lado_a))
    angulo = math.degrees(math.atan2(maior[1], maior[0]))
    return retangulo[0], angulo, float(np.hypot(*maior)), float(np.hypot(*menor))


def recortar_folha(imagem, poligono, modo: str, margem: float, cinza: int,
                   deslocamento: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """O recorte RGB de uma folha como as fotos do BRACOL: a folha deitada (o eixo maior na
    horizontal; o treino só girou até 15 graus, e os espelhamentos cobrem o sentido), centrada,
    com `margem` de cada lado e completada até a proporção 2:1, sem distorcer.

    Modo "A": a foto em volta da folha (o que cai fora da foto fica `cinza`). Modo "B": tudo
    fora da máscara (dilatada em DILATACAO_MASCARA do comprimento) fica `cinza`.
    poligono em pixels da imagem recebida. deslocamento: (dx, dy), em fração da largura e da
    altura do recorte, move a folha (e a máscara) dentro dele; serve às perturbações da
    avaliação de campo (evaluation/campo.py)."""
    centro, angulo, comprimento, espessura = eixo_da_folha(poligono)
    largura_saida = max(comprimento * (1 + 2 * margem), 2 * espessura * (1 + 2 * margem), 2.0)
    largura_saida, altura_saida = int(round(largura_saida)), max(1, int(round(largura_saida / 2)))
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    matriz[0, 2] += largura_saida / 2 - centro[0] + deslocamento[0] * largura_saida
    matriz[1, 2] += altura_saida / 2 - centro[1] + deslocamento[1] * altura_saida
    recorte = cv2.warpAffine(imagem, matriz, (largura_saida, altura_saida),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(cinza, cinza, cinza))
    if modo == "B":
        pontos = cv2.transform(np.asarray(poligono, dtype=np.float64).reshape(-1, 1, 2), matriz)
        mascara = np.zeros((altura_saida, largura_saida), dtype=np.uint8)
        cv2.fillPoly(mascara, [np.round(pontos).astype(np.int32)], 1)
        raio = int(round(DILATACAO_MASCARA * largura_saida))
        if raio > 0:
            mascara = cv2.dilate(mascara, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1)))
        recorte[mascara == 0] = cinza
    return recorte


def recorte_da_foto_inteira(imagem, cinza: int) -> np.ndarray:
    """A foto inteira como uma folha: deitada (retrato gira 90 graus) e completada até 2:1 com
    `cinza`, centrada, sem distorcer. Uma foto 2:1 (as do BRACOL) não muda."""
    if imagem.shape[0] > imagem.shape[1]:
        imagem = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    altura, largura = imagem.shape[:2]
    alvo_largura, alvo_altura = max(largura, 2 * altura), max(altura, math.ceil(largura / 2))
    if (alvo_largura, alvo_altura) == (largura, altura):
        return imagem
    dx, dy = alvo_largura - largura, alvo_altura - altura
    return cv2.copyMakeBorder(imagem, dy // 2, dy - dy // 2, dx // 2, dx - dx // 2,
                              cv2.BORDER_CONSTANT, value=(cinza, cinza, cinza))


def _decimal(valor: float) -> str:
    return f"{valor:.1f}".replace(".", ",")

