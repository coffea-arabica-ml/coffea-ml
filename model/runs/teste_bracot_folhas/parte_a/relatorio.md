# Uso 3a: folhas de campo do teste do BRACOT, parte A (passo 7)

Gerado por `python model/teste_final.py --uso 3a` em 14/10/2026 12:13. **Uso único e declarado dos testes**, com a cadeia congelada; nada foi ajustado nem escolhido por estes números. Início, fim, arquivos e SHA-256 em `model/runs/uso_do_teste_final.md`.

As 58 fotos de teste do BRACOT, sem as de `TESTE_SOBREPOSTO` (20190831_164920, 20190831_165013), pela cadeia final até as folhas: detector, área mínima, filtro de cor, limite de folhas e filtro de fragmentos (0,5). **O classificador não foi carregado:** nenhuma predição foi calculada. As anotações de teste não foram lidas.

- **Planos das fotos:** {'deteccao': 58}.
- **Detecções no limiar:** 410; cortadas pela área 1, pela cor 1, pelo filtro de fragmentos 76 (em 41 fotos) e pelo limite 0.
- **Folhas que ficam:** 332; por foto, 5,0; 1 a 13.
- **Sorteio:** até 2 folhas por foto, semente 42 (`campo.sortear_rotulagem`): 115 folhas; fotos por número de folhas sorteadas: {'1': 1, '2': 57}.
- **Pranchas cegas:** 6, em `data/processed/campo_rotulagem_teste` (fora do git), no formato das anteriores: id numerado, recorte B grande e A pequeno, sem predição nem score.
- **CSV vazio:** `data/manifests/campo_rotulado_teste.csv`, com as colunas dos outros dois.
- Contagens por foto em `contagens.csv`.

## Leitura

Ainda não escrita (falta o leitura.md).

