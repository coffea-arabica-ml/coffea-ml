"""Testes do agrupamento por hash e da divisão estratificada (data/bracol.py)."""
import random
from collections import Counter

import pytest

import bracol

# Imagens elegíveis por classe no subconjunto atual do BRACOL.
CONTAGENS_BRACOL = {
    "saudavel": 142,
    "bicho_mineiro": 253,
    "ferrugem": 465,
    "phoma": 346,
    "cercosporiose": 136,
}


def itens_sinteticos(contagens):
    """Linhas elegíveis falsas: ids sequenciais, um grupo por imagem, severidade de 0 a 4."""
    itens = []
    for classe, n in contagens.items():
        for k in range(n):
            i = len(itens) + 1
            itens.append({"id": i, "grupo": f"t-{i}", "classe": classe, "severity": k % 5})
    return itens


def contagem(divisao, itens):
    """{(classe, split): número de imagens}."""
    return Counter((item["classe"], divisao[item["id"]]) for item in itens)


# --------------------------------------------------------------------------- divisão
def test_cotas_no_tamanho_do_bracol():
    itens = itens_sinteticos(CONTAGENS_BRACOL)
    c = contagem(bracol.dividir(itens), itens)
    esperado = {  # treino, val, teste (tabela da decisão D2 do plano)
        "saudavel": (100, 21, 21),
        "bicho_mineiro": (177, 38, 38),
        "ferrugem": (325, 70, 70),
        "phoma": (242, 52, 52),
        "cercosporiose": (95, 20, 21),
    }
    for classe, cotas in esperado.items():
        assert (c[classe, "treino"], c[classe, "val"], c[classe, "teste"]) == cotas


def test_mesma_seed_da_o_mesmo_resultado_em_qualquer_ordem():
    itens = itens_sinteticos(CONTAGENS_BRACOL)
    embaralhados = itens[:]
    random.Random(0).shuffle(embaralhados)
    assert bracol.dividir(itens) == bracol.dividir(embaralhados)


def test_seed_diferente_muda_a_divisao():
    itens = itens_sinteticos(CONTAGENS_BRACOL)
    assert bracol.dividir(itens, seed=1) != bracol.dividir(itens, seed=2)


def test_severidade_fica_balanceada_dentro_da_classe():
    itens = itens_sinteticos({"ferrugem": 500})  # 100 imagens de cada severidade
    divisao = bracol.dividir(itens)
    for severidade in range(5):
        teste = sum(
            1 for item in itens if item["severity"] == severidade and divisao[item["id"]] == "teste"
        )
        assert 13 <= teste <= 17  # 15% de 100


def test_grupo_inteiro_vai_para_o_mesmo_split():
    itens = itens_sinteticos({"ferrugem": 60, "phoma": 40})
    for item in itens:  # grupos de até 3 imagens com ids consecutivos
        item["grupo"] = f"t-{(item['id'] - 1) // 3}"
    divisao = bracol.dividir(itens)
    splits_do_grupo = {}
    for item in itens:
        splits_do_grupo.setdefault(item["grupo"], set()).add(divisao[item["id"]])
    assert all(len(splits) == 1 for splits in splits_do_grupo.values())
    assert abs(contagem(divisao, itens)["ferrugem", "teste"] - 9) <= 3  # 15% de 60


def test_divisao_estavel_quando_chegam_imagens_novas():
    completo = itens_sinteticos(CONTAGENS_BRACOL)
    parcial = [item for item in completo if item["id"] % 5]  # tira 1 de cada 5
    antes = bracol.dividir(parcial)
    depois = bracol.dividir(completo, fixos=antes)
    assert all(depois[i] == split for i, split in antes.items())
    c = contagem(depois, completo)
    for classe, n in CONTAGENS_BRACOL.items():
        for split, pct in bracol.PROPORCOES.items():
            assert abs(c[classe, split] - n * pct / 100) <= 1


def test_fixos_de_imagens_que_sairam_sao_ignorados():
    itens = itens_sinteticos({"phoma": 20})
    antes = bracol.dividir(itens)
    depois = bracol.dividir(itens[:10], fixos=antes)
    assert depois == {i: antes[i] for i in depois}


def test_grupo_com_imagens_em_splits_diferentes_levanta_erro():
    itens = itens_sinteticos({"phoma": 2})
    for item in itens:
        item["grupo"] = "t-unico"
    with pytest.raises(ValueError, match="splits diferentes"):
        bracol.dividir(itens, fixos={1: "treino", 2: "teste"})


@pytest.mark.parametrize("classe", ["", None])
def test_linha_sem_classe_nao_entra_na_divisao(classe):
    itens = itens_sinteticos({"phoma": 3})
    itens[0]["classe"] = classe
    with pytest.raises(ValueError, match="sem classe"):
        bracol.dividir(itens)


@pytest.mark.parametrize(
    ("proporcoes", "fixos"),
    [
        ({"treino": 70, "val": 15, "teste": 14}, None),  # não soma 100
        ({"treino": 70, "validacao": 15, "teste": 15}, None),  # split desconhecido
        (bracol.PROPORCOES, {1: "validacao"}),  # split desconhecido em fixos
    ],
)
def test_parametros_invalidos_levantam_erro(proporcoes, fixos):
    with pytest.raises(ValueError):
        bracol.dividir(itens_sinteticos({"phoma": 10}), proporcoes=proporcoes, fixos=fixos)


def test_id_repetido_levanta_erro():
    itens = itens_sinteticos({"phoma": 3})
    itens.append(dict(itens[0], grupo="t-outro"))
    with pytest.raises(ValueError, match="repetido"):
        bracol.dividir(itens)


# ----------------------------------------------------------------------- agrupamento
BASE = random.Random(1).getrandbits(256)


def inverte(valor, n, a_partir_do_bit=0):
    """Inverte n bits seguidos: o resultado fica a distância de Hamming n do original."""
    return valor ^ (((1 << n) - 1) << a_partir_do_bit)


def item(i, phash, sha256=None):
    """Imagem falsa com pHash de 256 bits (64 hex, formato do imagehash com hash_size=16)."""
    return {"id": i, "sha256": sha256 or f"sha-{i}", "phash": f"{phash:064x}"}


def test_hashes_distantes_ficam_em_grupos_unitarios():
    rng = random.Random(7)
    itens = [item(i, rng.getrandbits(256)) for i in range(1, 51)]
    assert bracol.agrupar_por_hash(itens) == {i: f"bracol-{i}" for i in range(1, 51)}


def test_quase_duplicata_junta_ate_o_limiar():
    limiar = bracol.LIMIAR_QUASE_DUPLICATA
    itens = [
        item(10, BASE),
        item(11, inverte(BASE, limiar)),  # a 24 bits do 10: mesmo grupo
        item(12, inverte(BASE, limiar + 1, a_partir_do_bit=128)),  # a 25 do 10 e a 49 do 11
    ]
    assert bracol.agrupar_por_hash(itens) == {10: "bracol-10", 11: "bracol-10", 12: "bracol-12"}


def test_agrupamento_e_transitivo_e_usa_o_menor_id():
    meio = inverte(BASE, 20)
    ponta = inverte(meio, 20, a_partir_do_bit=100)  # a 40 bits de BASE, mas a 20 de meio
    itens = [item(22, ponta), item(21, meio), item(20, BASE)]  # fora de ordem de propósito
    assert bracol.agrupar_por_hash(itens) == {20: "bracol-20", 21: "bracol-20", 22: "bracol-20"}


def test_sha256_igual_junta_mesmo_com_phash_distante():
    oposto = BASE ^ ((1 << 256) - 1)  # todos os 256 bits diferentes
    itens = [item(1, BASE, sha256="igual"), item(2, oposto, sha256="igual")]
    assert bracol.agrupar_por_hash(itens) == {1: "bracol-1", 2: "bracol-1"}


def test_hashes_de_tamanhos_diferentes_levantam_erro():
    itens = [
        {"id": 1, "sha256": "a", "phash": "ff" * 32},
        {"id": 2, "sha256": "b", "phash": "ff" * 8},
    ]
    with pytest.raises(ValueError, match="mesmo tamanho"):
        bracol.agrupar_por_hash(itens)


def test_calculo_em_blocos_da_o_mesmo_resultado():
    rng = random.Random(3)
    hashes = [f"{rng.getrandbits(64):016x}" for _ in range(40)]  # 64 bits: muitos pares próximos
    matriz = bracol._matriz_de_bits(hashes)
    pares = set(bracol._pares_ate(matriz, 28, bloco=1000))
    assert pares  # garante que o teste compara alguma coisa
    assert set(bracol._pares_ate(matriz, 28, bloco=3)) == pares
