"""Teste de fumaça da EDA (data/eda_bracol.py) na cópia mínima e falsa do BRACOL dos testes do
organize_dataset.py: as 8 figuras e o resumo saem, e rodar de novo dá os mesmos bytes."""
import eda_bracol as eda
from test_organize import gerar as gerar_manifest
from test_organize import repo  # noqa: F401 (fixture usada abaixo)


def test_eda_gera_figuras_e_resumo_iguais_a_cada_execucao(repo):
    assert gerar_manifest(repo) == 0
    manifest = repo / "data" / "manifests" / "bracol.csv"
    saida = repo / "data" / "reports"
    arquivos = eda.gerar(manifest, saida, raiz=repo)
    figuras = [p for p in arquivos if p.parent.name == "figures"]
    assert len(figuras) == 8 and all(p.stat().st_size > 0 for p in figuras)
    primeira = {p: p.read_bytes() for p in arquivos}
    eda.gerar(manifest, saida, raiz=repo)
    assert {p: p.read_bytes() for p in arquivos} == primeira
    texto = (saida / "eda_bracol.md").read_text(encoding="utf-8")
    assert "Cópia PARCIAL" in texto
    assert "figures/08_brilho_e_cor_por_id.png" in texto
