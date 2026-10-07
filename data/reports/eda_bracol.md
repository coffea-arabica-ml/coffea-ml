# EDA do BRACOL (cópia parcial)

Gerado por `python data/eda_bracol.py` a partir de `data/manifests/bracol.csv`. Não editar à mão: rode o script de novo. O notebook `notebooks/01_eda.ipynb` mostra as mesmas tabelas e figuras; as figuras ficam em `data/reports/figures/` para a documentação (Frente 12).

> **Cópia PARCIAL do BRACOL:** 1.401 de 1.747 imagens. Resultados com ela não são comparáveis com Esgario et al. (2020).
>
> **Ressalva:** A correspondência das colunas 'phoma' e 'cercospora' do dataset.csv (códigos 3 e 4 de predominant_stress) com as classes 'brown leaf spot' e 'cercospora leaf spot' de Esgario et al. (2020) ainda não foi confirmada. Aqui elas viram as classes 'phoma' e 'cercosporiose' do projeto.
>
> **Classe 5** (`predominant_stress = 5`, significado desconhecido): 62 linhas, 59 com imagem. Fica fora de classificação, severidade e multirrótulo até a resposta dos autores.

Todas as 1.401 imagens têm 2048x1024 pixels, por isso não há figura de resolução.

## 1. Classes e o efeito dos ausentes

![Classes no dataset.csv completo e na cópia recuperada](figures/01_classes.png)

| classe | no csv | com imagem | sem imagem | perda |
|---|---:|---:|---:|---:|
| saudavel | 272 | 142 | 130 | 47,8% |
| ferrugem | 531 | 465 | 66 | 12,4% |
| bicho_mineiro | 387 | 253 | 134 | 34,6% |
| phoma | 348 | 346 | 2 | 0,6% |
| cercosporiose | 147 | 136 | 11 | 7,5% |
| classe 5 | 62 | 59 | 3 | 4,8% |

A perda não é uniforme: saudavel perdeu 47,8% e bicho_mineiro, 34,6%; phoma perdeu só 0,6%. Com isso, saudavel passa de 15,6% das linhas do csv para 10,1% das imagens presentes.

## 2. Onde estão os ausentes

![Ids do dataset.csv por classe, com e sem imagem](figures/02_ausentes_por_id.png)

Os ids sem imagem formam as faixas 7-9, 69-99, 688-999. A causa é o truncamento do zip publicado, que guarda as imagens em ordem alfabética do nome (ver `data/README.md`). Como as classes aparecem em blocos de ids no csv, perder faixas de ids vira perder classes de forma desigual.

## 3. Severidade por classe

![Severidade por classe](figures/03_severidade.png)

| classe | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| saudavel | 142 | 0 | 0 | 0 | 0 |
| ferrugem | 0 | 278 | 117 | 38 | 32 |
| bicho_mineiro | 0 | 195 | 44 | 14 | 0 |
| phoma | 0 | 202 | 99 | 29 | 16 |
| cercosporiose | 0 | 118 | 15 | 2 | 1 |

Níveis: 0 = saudável (<0,1%); 1 = muito baixa (0,1-5%); 2 = baixa (5-10%); 3 = alta (10-15%); 4 = muito alta (>15%) da área da folha.

O nível 1, muito baixa (0,1-5%), é o mais comum: 793 de 1.342 imagens elegíveis (59,1%). Combinações de classe e nível com menos de 5 imagens: bicho_mineiro/4: 0, cercosporiose/3: 2, cercosporiose/4: 1. Métricas de severidade nessas combinações ficam muito incertas.

## 4. Mais de um estresse na mesma folha

![Estresses marcados por folha](figures/04_estresses_por_folha.png)

| estresses marcados | imagens |
|---|---:|
| phoma | 339 |
| rust | 303 |
| miner | 204 |
| rust + cercospora | 158 |
| nenhum (saudável) | 142 |
| miner + rust | 89 |
| cercospora | 76 |
| miner + cercospora | 18 |
| phoma + cercospora | 7 |
| miner + phoma | 2 |
| miner + rust + cercospora | 2 |
| rust + phoma | 2 |

278 das 1.342 imagens elegíveis (20,7%) têm mais de um estresse marcado; a combinação mais comum é rust + cercospora (158). A classe do projeto usa só o estresse predominante; as colunas `miner`, `rust`, `phoma` e `cercospora` guardam o multirrótulo.

## 5. Exemplos por classe

![Exemplos por classe](figures/05_exemplos_por_classe.jpg)

Quatro imagens de treino por classe, sorteadas de forma determinística (seed 42); a última linha mostra a classe 5, que está excluída. Imagens: BRACOL (Krohling, Esgario e Ventura, 2019), CC BY 4.0, DOI 10.17632/yy2k5y8mxg.1.

## 6. Duplicatas e quase-duplicatas

![Distância de pHash até a imagem mais parecida](figures/06_vizinho_mais_proximo.png)

Para cada imagem, a distância de pHash (256 bits) até a imagem mais parecida. Nenhuma imagem fica a 24 bits ou menos de outra (o limiar de quase-duplicata); a menor distância é 48 bits e a mediana, 74 bits. Imagens com SHA-256 repetido: 0. O pHash pega a mesma foto re-salva, redimensionada ou com outro brilho; não pega a mesma folha fotografada de novo.

## 7. Divisão treino/val/teste

![Divisão treino/val/teste](figures/07_divisao.png)

| classe | treino | val | teste | total |
|---|---:|---:|---:|---:|
| saudavel | 100 | 21 | 21 | 142 |
| ferrugem | 325 | 70 | 70 | 465 |
| bicho_mineiro | 177 | 38 | 38 | 253 |
| phoma | 242 | 52 | 52 | 346 |
| cercosporiose | 95 | 20 | 21 | 136 |
| **total** | 939 | 201 | 202 | 1.342 |

Estratificada por classe (70/15/15, seed 42), com os níveis de severidade espalhados: a fração de teste fica entre 14,3% e 15,7% em todos os níveis. Com 202 imagens de teste, uma acurácia observada de 90% tem intervalo de confiança de 95% (Wilson) de 85,1% a 93,4%.

## 8. Brilho e cor média por id (indícios de sessões de foto)

![Brilho e cor média por id](figures/08_brilho_e_cor_por_id.png)

Medidas numa versão reduzida de cada imagem (256 px de largura):

- **luminância**: média da imagem em tons de cinza, de 0 (preto) a 255 (branco);
- **cor média**: média de R, G e B da imagem inteira (a faixa colorida da figura);
- **fundo**: faixas de cima e de baixo, 1/8 da altura cada, que nessas fotos em geral só têm fundo. O **tom** é azul menos vermelho: quanto mais negativo, mais amarelado.

Medianas por faixa de 100 ids (faixas sem imagem não aparecem); a dispersão do tom é a distância interquartil dentro da faixa:

| ids | imagens | luminância | luminância do fundo | tom do fundo | dispersão do tom | classes mais comuns |
|---|---:|---:|---:|---:|---:|---|
| 1-100 | 66 | 148,5 | 190,6 | -10,9 | 1,5 | bicho_mineiro 46, saudavel 16 |
| 101-200 | 100 | 142,8 | 179,2 | -9,6 | 1,5 | ferrugem 71, saudavel 28 |
| 201-300 | 100 | 169,4 | 202,7 | -16,2 | 5,7 | phoma 69, bicho_mineiro 19 |
| 301-400 | 100 | 179,5 | 211,1 | -16,8 | 0,9 | phoma 100 |
| 401-500 | 100 | 177,9 | 212,0 | -16,7 | 0,9 | phoma 95, cercosporiose 3 |
| 501-600 | 100 | 169,5 | 206,9 | -16,7 | 1,1 | phoma 56, ferrugem 15 |
| 601-700 | 87 | 168,4 | 211,1 | -16,7 | 1,1 | saudavel 26, phoma 19 |
| 901-1000 | 1 | 142,1 | 195,3 | -12,0 | 0,0 | bicho_mineiro 1 |
| 1001-1100 | 100 | 144,4 | 186,6 | -11,4 | 0,9 | bicho_mineiro 58, saudavel 16 |
| 1101-1200 | 100 | 142,9 | 182,0 | -11,2 | 1,1 | bicho_mineiro 56, saudavel 22 |
| 1201-1300 | 100 | 161,6 | 209,7 | -2,2 | 9,5 | ferrugem 61, bicho_mineiro 19 |
| 1301-1400 | 100 | 165,0 | 212,6 | -3,0 | 4,2 | ferrugem 73, bicho_mineiro 13 |
| 1401-1500 | 100 | 168,7 | 207,7 | -8,8 | 5,4 | ferrugem 66, cercosporiose 16 |
| 1501-1600 | 100 | 178,2 | 218,9 | -10,5 | 3,6 | ferrugem 64, cercosporiose 16 |
| 1601-1700 | 100 | 154,8 | 199,5 | -10,5 | 6,4 | ferrugem 56, cercosporiose 30 |
| 1701-1747 | 47 | 152,7 | 193,6 | -10,3 | 6,1 | ferrugem 20, cercosporiose 17 |

Medianas por classe:

| classe | imagens | luminância | luminância do fundo | tom do fundo |
|---|---:|---:|---:|---:|
| saudavel | 142 | 145,6 | 187,9 | -11,2 |
| ferrugem | 465 | 160,7 | 204,2 | -9,0 |
| bicho_mineiro | 253 | 148,2 | 188,9 | -11,2 |
| phoma | 346 | 176,3 | 209,7 | -16,7 |
| cercosporiose | 136 | 167,1 | 206,7 | -11,1 |
| classe 5 | 59 | 161,8 | 204,8 | -11,4 |

**O que a figura mostra.** Nesta cópia, brilho e cor mudam por blocos de ids: faixas vizinhas têm valores parecidos e há saltos entre blocos. Entre as faixas de 100 ids com pelo menos 20 imagens, a mediana do tom do fundo vai de -16,8 (ids 301-400: phoma 100) a -2,2 (ids 1201-1300: ferrugem 61, bicho_mineiro 19), e a da luminância do fundo, de 179,2 a 218,9. Os blocos coincidem em parte com os blocos de classe do csv: phoma, com 90% das imagens entre os ids 248 e 650, tem o fundo mais amarelado (mediana -16,7) do que as outras classes (-11,2 a -9,0). Dentro de uma mesma classe o tom também varia entre faixas: em ferrugem, a mediana por faixa com pelo menos 20 imagens dessa classe vai de -13,4 a -1,5. Em 8 das 15 faixas o tom do fundo é homogêneo (dispersão de no máximo 2); nas outras (201-300, 1201-1300, 1301-1400, 1401-1500, 1501-1600, 1601-1700, 1701-1747), a figura mostra um bloco que termina no meio da faixa ou dois níveis que se alternam.

**O que a figura não mostra.** Não prova que houve sessões de foto diferentes: o BRACOL não traz data, câmera nem planta, e a média da imagem inteira também depende do tamanho da folha e da cor das lesões. O padrão é compatível com sessões diferentes (luz, câmera ou balanço de branco), algumas concentradas em uma classe. Se for isso, um modelo pode aprender a cor do fundo junto com a lesão, e a divisão aleatória por imagem põe fotos da mesma sessão em treino e teste. Vale checar na Frente 9: Grad-CAM nas imagens de teste e acurácia por faixa de ids.
