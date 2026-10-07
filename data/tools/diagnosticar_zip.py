"""
Diagnóstico do zip do BRACOL (SOMENTE LEITURA: não extrai nem altera nada).

Mostra:
  1. tamanho e SHA-256 do arquivo (serve de registro e para comparar downloads);
  2. a estrutura de pastas do conteúdo e os arquivos que não são imagem
     (anotações), que definem como o organize_dataset.py deve ler o dataset;
  3. quantos arquivos internos estão íntegros e quais estão com erro.

Uso:
    python diagnosticar_zip.py                    -> usa o caminho padrão abaixo
    python diagnosticar_zip.py "C:\\pasta\\arquivo.zip"

Só usa a biblioteca padrão do Python (nada para instalar).
O relatório também é salvo em relatorio_zip.txt, ao lado deste script.
"""
import hashlib
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CAMINHO_PADRAO = r"C:\Users\PICHAU\Downloads\BRACOL_coffee_leaf_ images_datasets.zip"
EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BLOCO = 1024 * 1024
NOMES_METODO = {0: "sem compressão", 8: "deflate", 9: "deflate64", 12: "bzip2", 14: "lzma"}


def sha256_do_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            pedaco = f.read(BLOCO)
            if not pedaco:
                break
            h.update(pedaco)
    return h.hexdigest()


def testar_entrada(zf, info):
    """Lê a entrada inteira. Devolve None se estiver íntegra, ou a mensagem de erro."""
    try:
        with zf.open(info) as f:
            while f.read(BLOCO):
                pass
        return None
    except Exception as e:  # CRC errado, dado comprimido ruim, método não suportado...
        return f"{type(e).__name__}: {e}"


def localizar_zip():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    caminho = Path(CAMINHO_PADRAO)
    if caminho.exists():
        return caminho
    candidatos = list(caminho.parent.glob("BRACOL*.zip"))
    if len(candidatos) == 1:
        return candidatos[0]
    return caminho


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    linhas = []

    def p(texto=""):
        print(texto, flush=True)
        linhas.append(texto)

    caminho = localizar_zip()
    if not caminho.exists():
        p(f"Arquivo não encontrado: {caminho}")
        p('Passe o caminho completo: python diagnosticar_zip.py "C:\\pasta\\arquivo.zip"')
        return

    tamanho = caminho.stat().st_size
    p(f"Arquivo: {caminho}")
    p(f"Tamanho: {tamanho:,} bytes ({tamanho / 1048576:.1f} MB)")
    p("Calculando SHA-256 (alguns segundos)...")
    p(f"SHA-256: {sha256_do_arquivo(caminho)}")
    p()

    try:
        zf = zipfile.ZipFile(caminho)
    except zipfile.BadZipFile as e:
        p(f"O Python NÃO consegue abrir este zip: {e}")
        p("Isso indica dano na estrutura do arquivo (a lista de conteúdo), e não só em um item interno.")
        salvar(linhas)
        return

    infos = zf.infolist()
    arquivos = [i for i in infos if not i.is_dir()]
    p(f"Entradas no zip: {len(infos)} ({len(arquivos)} arquivos, {len(infos) - len(arquivos)} pastas)")
    p(f"Tamanho descompactado declarado: {sum(i.file_size for i in arquivos):,} bytes")
    metodos = Counter(NOMES_METODO.get(i.compress_type, f"método {i.compress_type}") for i in arquivos)
    p("Métodos de compressão: " + ", ".join(f"{n} ({q})" for n, q in metodos.items()))
    extensoes = Counter(Path(i.filename).suffix.lower() or "(sem extensão)" for i in arquivos)
    p("Extensões: " + ", ".join(f"{e} ({q})" for e, q in extensoes.most_common()))
    p()

    p("Verificando cada arquivo interno (pode levar cerca de um minuto)...")
    ruins = []
    for n, info in enumerate(arquivos, 1):
        erro = testar_entrada(zf, info)
        if erro:
            ruins.append((info.filename, erro))
        if n % 250 == 0:
            print(f"  ...{n}/{len(arquivos)} verificados", flush=True)
    p()

    p(f"RESULTADO: {len(arquivos) - len(ruins)} arquivos íntegros, {len(ruins)} com erro.")
    p()

    ruins_set = {nome for nome, _ in ruins}
    por_pasta = defaultdict(lambda: [0, 0])  # [total, com erro]
    for info in arquivos:
        partes = info.filename.split("/")[:-1]
        pasta = "/".join(partes[:3]) or "(raiz)"
        por_pasta[pasta][0] += 1
        if info.filename in ruins_set:
            por_pasta[pasta][1] += 1
    p("Estrutura (pasta: arquivos / com erro):")
    for pasta in sorted(por_pasta):
        total, com_erro = por_pasta[pasta]
        p(f"  {pasta}: {total} / {com_erro}")
    p()

    outros = [i.filename for i in arquivos if Path(i.filename).suffix.lower() not in EXT_IMAGEM]
    p(f"Arquivos que não são imagem ({len(outros)}):")
    for nome in outros[:40]:
        p(f"  {nome}")
    if len(outros) > 40:
        p(f"  ... e mais {len(outros) - 40}")
    p()

    if ruins:
        p("Arquivos com erro (até 25):")
        for nome, erro in ruins[:25]:
            p(f"  {nome} -> {erro}")
        if len(ruins) > 25:
            p(f"  ... e mais {len(ruins) - 25}")
    else:
        p("Nenhum arquivo interno com erro. O conteúdo está íntegro.")

    salvar(linhas)


def salvar(linhas):
    destino = Path(__file__).with_name("relatorio_zip.txt")
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
