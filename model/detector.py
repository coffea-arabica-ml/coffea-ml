"""
Detector de folhas (RF09), interface do backend (Frente 6): acha cada folha numa foto de planta,
de galho ou de uma folha só e devolve o polígono, a caixa e o score de cada uma. O modelo é um
YOLO11s-seg (ultralytics) ajustado no BRACOT por model/treinar_detector.py.

Licença (decisão do gestor, 10/10/2026): o ultralytics e os pesos YOLO são AGPL-3.0, aceita para
este projeto acadêmico de repositórios públicos. Se o backend servir o modelo pela rede, o
código-fonte do serviço tem de ficar disponível a quem o usa.

Rede: importar_ultralytics() liga o modo offline do ultralytics (sem checagem de conexão, de
atualização nem telemetria), desliga a instalação automática de pacotes, troca a pré-carga da
fonte dos gráficos (que baixaria a Arial.ttf mesmo no modo offline) por uma função que não faz
nada e guarda a configuração dele em model/pesos/ (fora do git), com sync=False. Os pesos vêm de
arquivo local.

Entrada do YOLO (conferida na validação do run detector_base, 10/10/2026): a foto reduzida a 640
px no lado maior (INTER_AREA) e cercada por uma moldura de 16 px de cinza 114, como o validador do
ultralytics monta os lotes (672 x 512 numa foto 4:3). No treino, a foto sempre tem cinza em volta
(mosaico e letterbox); sem a moldura, a borda da foto coincide com a borda da entrada e as
previsões da periferia mudam. AP50 de máscara na validação, região / conservador: 94,6% / 88,1%
com a moldura, 92,3% / 84,4% sem ela e 93,2% / 85,2% com a foto quadrada (faixas de 80 px).

Uso:
    import detector
    folhas = detector.detectar_folhas(imagem_rgb, limiar=0.5, pesos="model/runs/.../last.pt")
"""
import os
from pathlib import Path

import cv2
import numpy as np

RAIZ_REPO = Path(__file__).resolve().parents[1]
PASTA_PESOS = RAIZ_REPO / "model" / "pesos"  # fora do git
PESO_PRETREINADO = PASTA_PESOS / "yolo11s-seg.pt"
# Origem do peso pré-treinado (COCO-seg), baixado uma vez em 10/10/2026 da release oficial.
URL_PESO_PRETREINADO = ("https://github.com/ultralytics/assets/releases/download/v8.3.0/"
                        "yolo11s-seg.pt")
SHA256_PESO_PRETREINADO = "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3"
VERSAO_ULTRALYTICS = "8.4.175"
TAMANHO_ENTRADA = 640  # lado maior da foto na entrada do YOLO (o do treino)
MOLDURA = 16  # px de cinza em volta da foto reduzida (ver o docstring do módulo)
CINZA = 114  # o cinza do letterbox e do mosaic do YOLO
PESOS_PADRAO = RAIZ_REPO / "model" / "runs" / "detector_base" / "weights" / "last.pt"
# Variáveis lidas pelo ultralytics no import. As de rede são impostas; as outras valem só se o
# ambiente não as definir. OMP_NUM_THREADS: sem ela, o ultralytics a fixa em 1, e a inferência em
# CPU ficaria com uma thread só.
REDE_DESLIGADA = {"YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false"}
AMBIENTE_ULTRALYTICS = {
    "YOLO_CONFIG_DIR": str(PASTA_PESOS),
    "OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 2) // 2)),
}

_detectores = {}


def importar_ultralytics():
    """Importa o ultralytics com a rede desligada (ver o docstring do módulo) e devolve a classe
    YOLO. Precisa vir antes de qualquer outro import do ultralytics no processo."""
    PASTA_PESOS.mkdir(parents=True, exist_ok=True)  # o ultralytics só aceita uma pasta existente
    os.environ.update(REDE_DESLIGADA)
    for nome, valor in AMBIENTE_ULTRALYTICS.items():
        os.environ.setdefault(nome, valor)
    import ultralytics.data.utils as utils_de_dados
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    if SETTINGS["sync"]:
        SETTINGS.update({"sync": False})
    # A checagem do dataset (check_det_dataset) baixa a fonte Arial.ttf de ultralytics.com mesmo
    # com YOLO_OFFLINE; a fonte só serve para gráficos, que ficam desligados (plots=False).
    utils_de_dados.check_font = _sem_fonte
    return YOLO


def _sem_fonte(*_args, **_kwargs):
    """Substitui a pré-carga da fonte do ultralytics: não baixa nada."""
    return None


def carregar_detector(pesos=PESOS_PADRAO):
    """O modelo YOLO de um arquivo de pesos local, carregado uma vez por caminho."""
    caminho = Path(pesos).resolve()
    if not caminho.is_file():
        raise FileNotFoundError(f"pesos do detector não encontrados: {caminho}")
    if caminho not in _detectores:
        _detectores[caminho] = importar_ultralytics()(str(caminho))
    return _detectores[caminho]


def detectar_folhas(imagem_rgb, limiar: float, detector=None, pesos=PESOS_PADRAO,
                    dispositivo=None, max_det: int = 300) -> list[dict]:
    """As folhas de uma foto, do maior score para o menor.

    imagem_rgb: array RGB uint8 (altura x largura x 3), como dados.ler_imagem devolve.
    limiar: score mínimo (o escolhido na validação está no relatório do run).
    detector: um modelo de carregar_detector; None carrega `pesos`.
    dispositivo: "cpu", "cuda" ou None (o ultralytics escolhe).
    Devolve dicts com "poligono" ([[x, y], ...]), "caixa" ([x0, y0, x1, y1]) e "score", em
    pixels da foto recebida.
    """
    if detector is None:
        detector = carregar_detector(pesos)
    entrada, escala = preparar_entrada(imagem_rgb)
    resultado = detector.predict(entrada, imgsz=TAMANHO_ENTRADA + 2 * MOLDURA, conf=limiar,
                                 max_det=max_det, device=dispositivo, verbose=False)[0]
    return converter_resultado(resultado, escala, MOLDURA, np.asarray(imagem_rgb).shape[:2])


def preparar_entrada(imagem_rgb) -> tuple[np.ndarray, tuple[float, float]]:
    """A entrada do YOLO: a foto reduzida a TAMANHO_ENTRADA no lado maior, com a moldura de
    MOLDURA px de CINZA, em BGR (um array numpy, para o ultralytics, é BGR). Devolve a entrada e
    a escala (x, y) da redução."""
    imagem = np.asarray(imagem_rgb)
    if imagem.ndim != 3 or imagem.shape[2] != 3:
        raise ValueError(f"a foto tem de ser RGB (altura x largura x 3), veio {imagem.shape}")
    altura, largura = imagem.shape[:2]
    fator = TAMANHO_ENTRADA / max(altura, largura)
    nova_largura, nova_altura = round(largura * fator), round(altura * fator)
    if (nova_largura, nova_altura) != (largura, altura):
        imagem = cv2.resize(imagem, (nova_largura, nova_altura),
                            interpolation=cv2.INTER_AREA if fator < 1 else cv2.INTER_LINEAR)
    imagem = cv2.copyMakeBorder(imagem, MOLDURA, MOLDURA, MOLDURA, MOLDURA,
                                cv2.BORDER_CONSTANT, value=(CINZA, CINZA, CINZA))
    return np.ascontiguousarray(imagem[..., ::-1]), (nova_largura / largura, nova_altura / altura)


def converter_resultado(resultado, escala=(1.0, 1.0), moldura: int = 0,
                        tamanho: tuple[int, int] | None = None) -> list[dict]:
    """Um resultado do ultralytics (com máscaras) como lista de dicts, do maior score para o
    menor, em pixels da foto: tira a moldura, desfaz a escala (x, y) e, com o tamanho (altura,
    largura) da foto, corta polígono e caixa nela. Detecções sem polígono (menos de 3 pontos)
    ficam de fora."""
    if resultado.masks is None or resultado.boxes is None:
        return []
    fator = np.array(escala, dtype=np.float64)
    caixas = (np.asarray(_numpy(resultado.boxes.xyxy), dtype=np.float64).reshape(-1, 4)
              - moldura) / np.tile(fator, 2)
    scores = np.asarray(_numpy(resultado.boxes.conf), dtype=np.float64).reshape(-1)
    mascaras = np.asarray(_numpy(resultado.masks.data)) > 0.5
    limite = None if tamanho is None else np.array(tamanho[::-1], dtype=np.float64)  # (x, y)
    folhas = []
    for mascara, caixa, score in zip(mascaras, caixas, scores):
        poligono = (poligono_da_mascara(mascara, resultado.orig_shape) - moldura) / fator
        if limite is not None:
            poligono = np.clip(poligono, 0, limite)
            caixa = np.clip(caixa, 0, np.tile(limite, 2))
        if len(poligono) >= 3:
            folhas.append({"poligono": poligono.round(1).tolist(),
                           "caixa": caixa.round(1).tolist(), "score": float(score)})
    return sorted(folhas, key=lambda folha: -folha["score"])


def poligono_da_mascara(mascara: np.ndarray, forma_imagem) -> np.ndarray:
    """O contorno externo de maior área de uma máscara do YOLO (na resolução da entrada, com o
    letterbox), em pixels da imagem que o YOLO recebeu (forma_imagem = (altura, largura)).

    O masks.xy do ultralytics emenda todos os pedaços da máscara num polígono só, com traços
    retos entre eles. Na validação do detector_base (10/10/2026), 28% das máscaras no limiar têm
    mais de um pedaço, quase sempre migalhas (mediana de 0,08% da área): fica o maior.
    """
    contornos = cv2.findContours(np.asarray(mascara, dtype=np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contornos:
        return np.zeros((0, 2))
    maior = max(contornos, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    # Desfaz o letterbox como o ultralytics (ops.scale_coords): ganho e borda centralizada.
    altura_m, largura_m = np.shape(mascara)[:2]
    altura, largura = forma_imagem[:2]
    ganho = min(altura_m / altura, largura_m / largura)
    borda = (round((largura_m - largura * ganho) / 2 - 0.1),
             round((altura_m - altura * ganho) / 2 - 0.1))
    return (maior - borda) / ganho


def _numpy(valor):
    """Tensor do torch (em qualquer dispositivo) ou array como numpy."""
    return valor.cpu().numpy() if hasattr(valor, "cpu") else np.asarray(valor)
