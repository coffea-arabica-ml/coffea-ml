"""
Recuperação de um zip cujo índice (a "lista de conteúdo" no final do arquivo) está danificado.

Em vez de confiar no índice, este script lê o zip do começo ao fim, entrada por entrada
(cada arquivo dentro do zip tem um cabeçalho próprio), confere a integridade (CRC-32) de
cada uma e, se você pedir, grava as íntegras numa pasta NOVA. Entradas com erro nunca
são gravadas.

Uso:
    python recuperar_zip.py                           -> só VERIFICA (não grava imagens)
    python recuperar_zip.py --extrair PASTA           -> verifica e grava as íntegras em PASTA
    python recuperar_zip.py "C:\\caminho\\arq.zip" --extrair PASTA

PASTA deve não existir ou estar vazia. Só usa a biblioteca padrão do Python.
O relatório é salvo em relatorio_recuperacao.txt, ao lado deste script.
"""
import argparse
import mmap
import os
import struct
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

CAMINHO_PADRAO = r"C:\Users\PICHAU\Downloads\BRACOL_coffee_leaf_ images_datasets.zip"
EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BLOCO = 1024 * 1024

SIG_LOCAL = b"PK\x03\x04"
SIG_CENTRAL = b"PK\x01\x02"
SIG_FIM = b"PK\x05\x06"
SIG_FIM64 = b"PK\x06\x06"
SIG_DESC = b"PK\x07\x08"
# assinatura, versão, flags, método, hora, data, crc, tam. compactado, tam. original, tam. nome, tam. extra
CAB_LOCAL = struct.Struct("<4s5H3L2H")

DESCRICAO = {
    "ok": "íntegro",
    "ok_sem_crc": "íntegro (sem CRC para conferir)",
    "crc": "CRC não confere (dados alterados)",
    "descompressao": "erro ao descompactar",
    "tamanho": "tamanho diferente do declarado",
    "incompleto": "incompleto (o arquivo termina antes do fim da entrada)",
    "sem_tamanho": "sem tamanho conhecido (não recuperável)",
    "metodo": "método de compressão não suportado",
    "nome_invalido": "nome de arquivo inseguro (não gravado)",
    "erro_gravacao": "erro ao gravar no disco",
}
BONS = ("ok", "ok_sem_crc")


# ---------------------------------------------------------------- utilitários
def buscar_proxima(f, inicio, tam):
    """Posição e tipo da próxima assinatura PK (local, central ou fim), a partir de `inicio`."""
    sigs = {SIG_LOCAL: "local", SIG_CENTRAL: "central", SIG_FIM: "fim", SIG_FIM64: "fim"}
    pos = inicio
    while pos < tam:
        f.seek(pos)
        bloco = f.read(BLOCO + 3)
        if len(bloco) < 4:
            return None
        melhor = None
        for sig, tipo in sigs.items():
            i = bloco.find(sig)
            if i != -1 and (melhor is None or i < melhor[0]):
                melhor = (i, tipo)
        if melhor is not None:
            return pos + melhor[0], melhor[1]
        if len(bloco) < BLOCO + 3:
            return None
        pos += BLOCO
    return None


def tamanhos_zip64(extra, csize, usize):
    i = 0
    while i + 4 <= len(extra):
        tag, tam = struct.unpack_from("<HH", extra, i)
        corpo = extra[i + 4:i + 4 + tam]
        if tag == 0x0001:
            j = 0
            if usize == 0xFFFFFFFF and len(corpo) >= j + 8:
                usize = struct.unpack_from("<Q", corpo, j)[0]
                j += 8
            if csize == 0xFFFFFFFF and len(corpo) >= j + 8:
                csize = struct.unpack_from("<Q", corpo, j)[0]
                j += 8
            break
        i += 4 + tam
    return csize, usize


def ler_cabecalho(f, pos):
    """Lê e valida um cabeçalho local. Devolve um dicionário, ou None se não parecer um cabeçalho real."""
    f.seek(pos)
    bruto = f.read(CAB_LOCAL.size)
    if len(bruto) < CAB_LOCAL.size:
        return None
    sig, _ver, flags, metodo, _h, _d, crc, csize, usize, nlen, elen = CAB_LOCAL.unpack(bruto)
    if sig != SIG_LOCAL or not (1 <= nlen <= 1024) or metodo > 99:
        return None
    nome_b = f.read(nlen)
    extra = f.read(elen)
    if len(nome_b) < nlen or len(extra) < elen:
        return None
    if any(b < 32 for b in nome_b):
        return None
    try:
        nome = nome_b.decode("utf-8")
    except UnicodeDecodeError:
        nome = nome_b.decode("cp437", "replace")
    csize, usize = tamanhos_zip64(extra, csize, usize)
    return {
        "nome": nome, "flags": flags, "metodo": metodo, "crc": crc,
        "csize": csize, "usize": usize,
        "inicio": pos + CAB_LOCAL.size + nlen + elen,
        "eh_pasta": nome.endswith("/"),
    }


def caminho_seguro(base, nome):
    """Caminho dentro de `base`, ou None se o nome tentar escapar da pasta (.., unidade, etc.)."""
    partes = [p for p in nome.replace("\\", "/").split("/") if p not in ("", ".")]
    if not partes or any(p == ".." for p in partes) or ":" in partes[0]:
        return None
    alvo = base.joinpath(*partes)
    try:
        alvo.resolve().relative_to(base.resolve())
    except ValueError:
        return None
    return alvo


def achar_fim_armazenado(f, inicio):
    """Entrada SEM compressão e SEM tamanho no cabeçalho: procura o descritor de dados
    (PK\\x07\\x08 + CRC + tamanho) cujo tamanho bata com a distância percorrida e cujo CRC bata
    com os dados. Devolve (tamanho, crc) ou None."""
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except (ValueError, OSError):
        return None
    try:
        q = mm.find(SIG_DESC, inicio)
        while q != -1:
            dist = q - inicio
            if q + 16 <= len(mm):
                crc = struct.unpack_from("<L", mm, q + 4)[0]
                tam32 = struct.unpack_from("<L", mm, q + 8)[0]
                tam64 = struct.unpack_from("<Q", mm, q + 8)[0] if q + 24 <= len(mm) else None
                if (tam32 == dist or tam64 == dist) and zlib.crc32(mm[inicio:q]) == crc:
                    return dist, crc
            q = mm.find(SIG_DESC, q + 1)
        return None
    finally:
        mm.close()


def crc_do_descritor(f, fim):
    """CRC gravado no descritor de dados que vem logo depois do fluxo, ou None se não houver."""
    f.seek(fim)
    b = f.read(8)
    if len(b) < 4 or b[:4] in (SIG_LOCAL, SIG_CENTRAL, SIG_FIM, SIG_FIM64):
        return None
    if b[:4] == SIG_DESC:
        return struct.unpack_from("<L", b, 4)[0] if len(b) >= 8 else None
    return struct.unpack_from("<L", b, 0)[0]


# ------------------------------------------------------------ leitura de uma entrada
def processar(f, cab, tam, destino):
    """Lê os dados de uma entrada, confere o CRC e, se `destino` foi dado, grava se íntegra.
    Devolve (status, mensagem, posição_onde_continuar)."""
    if cab["eh_pasta"]:
        return "ok", "", min(cab["inicio"] + cab["csize"], tam)

    status = None
    msg = ""
    proximo = cab["inicio"]
    estado = {"crc": 0, "total": 0}
    sink = None
    tmp = None
    caminho = None
    erro_gravacao = None

    if destino is not None:
        caminho = caminho_seguro(destino, cab["nome"])
        if caminho is not None:
            try:
                caminho.parent.mkdir(parents=True, exist_ok=True)
                tmp = caminho.with_name(caminho.name + ".parte")
                sink = open(tmp, "wb")
            except OSError as e:
                erro_gravacao = f"{type(e).__name__}: {e}"
                sink = None

    def emitir(dados):
        if dados:
            estado["crc"] = zlib.crc32(dados, estado["crc"])
            estado["total"] += len(dados)
            if sink is not None:
                sink.write(dados)

    tamanho_desconhecido = (
        (cab["flags"] & 0x08 and cab["csize"] == 0) or (cab["csize"] == 0 and cab["usize"] != 0)
    )
    if tamanho_desconhecido and cab["metodo"] == 0:
        achado = achar_fim_armazenado(f, cab["inicio"])
        if achado is not None:
            cab["csize"] = cab["usize"] = achado[0]
            cab["crc"] = achado[1]
            cab["flags"] &= ~0x08
            tamanho_desconhecido = False
    try:
        if cab["metodo"] == 8 and not tamanho_desconhecido:
            d = zlib.decompressobj(-15)
            restante = cab["csize"]
            f.seek(cab["inicio"])
            try:
                truncado = False
                while restante > 0:
                    bloco = f.read(min(BLOCO, restante))
                    if not bloco:
                        truncado = True
                        break
                    restante -= len(bloco)
                    emitir(d.decompress(bloco))
                if truncado:
                    status, proximo = "incompleto", tam
                else:
                    emitir(d.flush())
                    proximo = cab["inicio"] + cab["csize"]
                    if not d.eof:
                        status, msg = "descompressao", "fluxo deflate sem final"
            except zlib.error as e:
                status, msg = "descompressao", str(e)
                proximo = min(cab["inicio"] + cab["csize"], tam)

        elif cab["metodo"] == 0 and not tamanho_desconhecido:
            restante = cab["csize"]
            f.seek(cab["inicio"])
            truncado = False
            while restante > 0:
                bloco = f.read(min(BLOCO, restante))
                if not bloco:
                    truncado = True
                    break
                restante -= len(bloco)
                emitir(bloco)
            if truncado:
                status, proximo = "incompleto", tam
            else:
                proximo = cab["inicio"] + cab["csize"]

        elif cab["metodo"] == 8:  # deflate sem tamanho no cabeçalho: o próprio fluxo marca o fim
            d = zlib.decompressobj(-15)
            f.seek(cab["inicio"])
            consumido = 0
            fim_fluxo = None
            try:
                while True:
                    bloco = f.read(BLOCO)
                    if not bloco:
                        status, proximo = "incompleto", tam
                        break
                    emitir(d.decompress(bloco))
                    if d.eof:
                        consumido += len(bloco) - len(d.unused_data)
                        fim_fluxo = cab["inicio"] + consumido
                        break
                    consumido += len(bloco)
            except zlib.error as e:
                status, msg = "descompressao", str(e)
            if fim_fluxo is not None:
                proximo = fim_fluxo
                crc_desc = crc_do_descritor(f, fim_fluxo)
                if crc_desc is None:
                    status = "ok_sem_crc"
                elif crc_desc != estado["crc"]:
                    status, msg = "crc", f"esperado {crc_desc:08x}, obtido {estado['crc']:08x}"
                else:
                    status = "ok"
            elif status is None:
                status = "incompleto"

        elif cab["metodo"] == 0:
            status, proximo = "sem_tamanho", cab["inicio"]

        else:
            status = "metodo"
            msg = f"método {cab['metodo']}"
            proximo = min(cab["inicio"] + cab["csize"], tam) if cab["csize"] else cab["inicio"]

        # conferência final para as entradas com tamanho conhecido
        if status is None:
            if cab["flags"] & 0x08 and cab["crc"] == 0:
                status = "ok_sem_crc"
            elif estado["crc"] != cab["crc"]:
                status = "crc"
                msg = f"esperado {cab['crc']:08x}, obtido {estado['crc']:08x}"
            elif estado["total"] != cab["usize"] and cab["usize"] != 0xFFFFFFFF:
                status = "tamanho"
                msg = f"declarado {cab['usize']}, lido {estado['total']}"
            else:
                status = "ok"
    finally:
        if sink is not None:
            sink.close()

    if status in BONS and destino is not None:
        if caminho is None:
            status = "nome_invalido"
        elif erro_gravacao:
            status, msg = "erro_gravacao", erro_gravacao
        else:
            try:
                os.replace(tmp, caminho)
            except OSError as e:
                status, msg = "erro_gravacao", f"{type(e).__name__}: {e}"
    if tmp is not None and os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return status, msg, proximo


# ----------------------------------------------------------------------- principal
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("zip", nargs="?", default=None)
    ap.add_argument("--extrair", metavar="PASTA", default=None)
    args = ap.parse_args()

    linhas = []

    def p(texto=""):
        print(texto, flush=True)
        linhas.append(texto)

    caminho_zip = Path(args.zip) if args.zip else Path(CAMINHO_PADRAO)
    if not caminho_zip.exists() and not args.zip:
        achados = list(caminho_zip.parent.glob("BRACOL*.zip"))
        if len(achados) == 1:
            caminho_zip = achados[0]
    if not caminho_zip.exists():
        p(f"Arquivo não encontrado: {caminho_zip}")
        p('Passe o caminho completo: python recuperar_zip.py "C:\\pasta\\arquivo.zip"')
        return

    destino = None
    if args.extrair:
        destino = Path(args.extrair)
        if destino.exists() and any(destino.iterdir()):
            p(f"A pasta de destino já tem conteúdo: {destino}")
            p("Use uma pasta nova (ou vazia) para não misturar com outros arquivos.")
            return
        destino.mkdir(parents=True, exist_ok=True)

    tam = caminho_zip.stat().st_size
    p(f"Arquivo: {caminho_zip}")
    p(f"Tamanho: {tam:,} bytes ({tam / 1048576:.1f} MB)")
    p("Modo: " + (f"verificar e gravar as íntegras em {destino}" if destino else "somente verificar (nada é gravado)"))
    p()

    entradas = []  # (nome, status, msg, é_pasta)
    parou_em = None
    ultimo_fim = 0
    pos = 0
    with open(caminho_zip, "rb") as f:
        while True:
            achado = buscar_proxima(f, pos, tam)
            if achado is None:
                break
            pos_sig, tipo = achado
            if tipo != "local":
                parou_em = (pos_sig, tipo)
                break
            cab = ler_cabecalho(f, pos_sig)
            if cab is None:  # assinatura "por acaso" no meio de dados: ignora
                pos = pos_sig + 4
                continue
            status, msg, proximo = processar(f, cab, tam, destino)
            entradas.append((cab["nome"], status, msg, cab["eh_pasta"]))
            ultimo_fim = max(ultimo_fim, proximo)
            pos = max(proximo, pos_sig + 4)
            if len(entradas) % 250 == 0:
                print(f"  ...{len(entradas)} entradas lidas ({pos / 1048576:.0f} de {tam / 1048576:.0f} MB)", flush=True)

        f.seek(max(0, tam - 70000))
        cauda = f.read()
        n_indice = None
        if parou_em and parou_em[1] == "central":
            f.seek(parou_em[0])
            n_indice = f.read(64 * 1024 * 1024).count(SIG_CENTRAL)
    tem_fim = cauda.rfind(SIG_FIM) != -1 or cauda.rfind(SIG_FIM64) != -1

    arquivos = [e for e in entradas if not e[3]]
    pastas = [e for e in entradas if e[3]]
    por_status = Counter(e[1] for e in arquivos)
    bons = sum(por_status[s] for s in BONS)
    ruins = [e for e in arquivos if e[1] not in BONS]

    p(f"Índice do zip (registro de fim) no final do arquivo: {'presente' if tem_fim else 'AUSENTE'}")
    if parou_em:
        p(f"Início do índice central encontrado na posição {parou_em[0]:,} ({parou_em[1]})")
    if n_indice is not None:
        p(f"Entradas listadas no índice central: {n_indice} (lidas na varredura: {len(entradas)})")
    p(f"Leitura sequencial chegou até o byte {ultimo_fim:,} de {tam:,} (sobram {max(0, tam - ultimo_fim):,} bytes)")
    p()
    p(f"Entradas lidas: {len(entradas)} ({len(arquivos)} arquivos, {len(pastas)} pastas)")
    p(f"RESULTADO: {bons} arquivos íntegros, {len(ruins)} com problema")
    for s, q in por_status.most_common():
        p(f"  {DESCRICAO.get(s, s)}: {q}")
    p()

    ext = Counter(Path(e[0]).suffix.lower() or "(sem extensão)" for e in arquivos)
    p("Extensões: " + ", ".join(f"{k} ({v})" for k, v in ext.most_common()))
    ruins_nomes = {e[0] for e in ruins}
    por_pasta = defaultdict(lambda: [0, 0])
    for nome, status, _m, _d in arquivos:
        pasta = "/".join(nome.split("/")[:-1][:3]) or "(raiz)"
        por_pasta[pasta][0] += 1
        if nome in ruins_nomes:
            por_pasta[pasta][1] += 1
    p()
    p("Estrutura (pasta: arquivos / com problema):")
    for pasta in sorted(por_pasta):
        total, com_erro = por_pasta[pasta]
        p(f"  {pasta}: {total} / {com_erro}")

    outros = [e[0] for e in arquivos if Path(e[0]).suffix.lower() not in EXT_IMAGEM]
    p()
    p(f"Arquivos que não são imagem ({len(outros)}):")
    for nome in outros[:40]:
        p(f"  {nome}")
    if len(outros) > 40:
        p(f"  ... e mais {len(outros) - 40}")

    repetidos = [n for n, q in Counter(e[0] for e in arquivos).items() if q > 1]
    if repetidos:
        p()
        p(f"Nomes que aparecem mais de uma vez no zip: {len(repetidos)} (a última cópia vale)")

    p()
    if ruins:
        p("Arquivos com problema (até 25):")
        for nome, status, msg, _d in ruins[:25]:
            p(f"  {nome} -> {DESCRICAO.get(status, status)}" + (f" [{msg}]" if msg else ""))
        if len(ruins) > 25:
            p(f"  ... e mais {len(ruins) - 25}")
    else:
        p("Nenhum arquivo com problema.")

    if destino is not None:
        gravados = sum(1 for _r, _d, files in os.walk(destino) for _ in files)
        p()
        p(f"Gravados em {destino}: {gravados} arquivos")

    salvar(linhas)


def salvar(linhas):
    destino = Path(__file__).with_name("relatorio_recuperacao.txt")
    try:
        destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        print(f"\nRelatório salvo em: {destino}")
    except Exception as e:
        print(f"\n(Não consegui salvar o relatório: {e})")


if __name__ == "__main__":
    main()
    try:
        input("\nPressione Enter para fechar...")
    except EOFError:
        pass
