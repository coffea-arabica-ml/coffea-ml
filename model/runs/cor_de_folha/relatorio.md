# Filtro de cor de folha (passo 8)

Decisão do gestor, 30/10/2026. Gerado por `evaluation/cor_de_folha.py`; números em `medicao.json`. Só o treino do BRACOT e as fotos sem planta; os testes fechados não são lidos.

## Método

- Pixel colorido: saturação >= 30 (HSV do OpenCV, na cópia reduzida a 1024 px, como a cadeia). Pixel com cor de folha: colorido, com brilho (V) >= o brilho mínimo e matiz dentro da faixa.
- Para cada brilho mínimo, a faixa é o menor arco de matiz que cobre 99,0% dos pixels coloridos e claros das 1318 folhas anotadas do treino (cada folha pesa igual), e o corte é o p1 da fração da máscara com cor de folha, arredondado para baixo.
- O brilho mínimo existe porque o matiz de pixel escuro é ruído: com brilho mínimo 0, as sombras das folhas levam a faixa até o azul, e um notebook azul-marinho passou.

Conferência do método: o p1 da fração colorida deu 34,0%; em 10/10/2026, 35%.

## Sensibilidade ao brilho mínimo

| brilho mínimo | faixa de matiz (H) | graus | corte | folhas anotadas abaixo | detecções cortadas | S cortadas | D cortadas | fotos sem planta com folha |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 24 a 110 | 48° a 220° | 33,0% | 12 | 11 de 1727 | 0 de 42 | 0 de 61 | 13 de 38 |
| 32 | 24 a 110 | 48° a 220° | 33,0% | 12 | 11 de 1727 | 0 de 42 | 0 de 61 | 13 de 38 |
| 48 | 24 a 110 | 48° a 220° | 33,0% | 12 | 11 de 1727 | 0 de 42 | 0 de 61 | 13 de 38 |
| 64 | 25 a 111 | 50° a 222° | 30,0% | 13 | 14 de 1727 | 0 de 42 | 0 de 61 | 13 de 38 |
| 80 | 24 a 111 | 48° a 222° | 25,0% | 14 | 15 de 1727 | 0 de 42 | 0 de 61 | 13 de 38 |
| 96 | 24 a 112 | 48° a 224° | 17,0% | 13 | 35 de 1727 | 2 de 42 | 0 de 61 | 15 de 38 |

Sem o filtro de matiz (só o de saturação, hoje): 32 de 38 fotos sem planta com folha.

## Fotos sem planta, por foto (folhas utilizáveis que sobram)

Conjunto de desenvolvimento desde 30/10/2026 (motivou o brilho mínimo): o número aqui sai otimista.

| foto | hoje | V >= 0 | V >= 32 | V >= 48 | V >= 64 | V >= 80 | V >= 96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| commons_backpacks_2.jpg | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_blue-objects_1.jpg | 4 | 3 | 3 | 2 | 2 | 3 | 3 |
| commons_blue-objects_2.jpg | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| commons_cardboard-boxes_1.jpg | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_cardboard-boxes_2.jpg | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_cats_1.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_coffee-mugs_1.jpg | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_coffee-mugs_2.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_coffee-mugs_3.jpg | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| commons_dogs_1.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_green-bags_1.jpg | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| commons_green-bags_2.jpg | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_green-bags_3.jpg | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| commons_green-balloons_1.jpg | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_green-balloons_2.jpg | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_green-bottles_1.jpg | 8 | 2 | 2 | 2 | 2 | 2 | 2 |
| commons_green-bottles_2.jpg | 3 | 2 | 2 | 2 | 2 | 2 | 2 |
| commons_green-bottles_3.jpg | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_green-cups_1.jpg | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| commons_green-cups_2.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_green-cups_3.jpg | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_green-mugs_1.jpg | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| commons_green-mugs_2.jpg | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| commons_green-mugs_3.jpg | 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| commons_green-t-shirts_1.jpg | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| commons_green-t-shirts_3.jpg | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_human-hands_3.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_kitchen-counters_1.jpg | 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| commons_laptops_1.jpg | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_laptops_2.jpg | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_red-objects_2.jpg | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_shoes_1.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_shoes_2.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_tennis-balls_2.jpg | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_wooden-tables_1.jpg | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_yellow-objects_1.jpg | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| commons_yellow-objects_3.jpg | 2 | 0 | 0 | 0 | 0 | 0 | 1 |
| teste-sem-planta.jpg | 4 | 3 | 3 | 3 | 2 | 2 | 2 |

## A cadeia inteira, sem e com o filtro adotado

Adotado (decisão do gestor, 30/10/2026): brilho mínimo 0 (não adotado), faixa 24 a 110 e corte 33,0%, em `inferencia.Configuracao`. Resposta de `inferencia.diagnosticar_arquivo` (detector, classificador e regras), que é a que chega ao usuário.

| conjunto | fotos | sucesso sem o filtro | sucesso com o filtro |
|---|---:|---:|---:|
| sem planta (desenvolvimento) | 38 | 23 | 9 |
| treino do BRACOT (1 a cada 8) | 30 | 28 | 28 |

Mudaram (sem planta (desenvolvimento)):

| foto | sem o filtro | com o filtro |
|---|---|---|
| commons_backpacks_2.jpg | baixa_confianca | planta_nao_identificada |
| commons_cardboard-boxes_2.jpg | baixa_confianca | planta_nao_identificada |
| commons_cats_1.jpg | baixa_confianca | planta_nao_identificada |
| commons_coffee-mugs_1.jpg | baixa_confianca | planta_nao_identificada |
| commons_coffee-mugs_3.jpg | sucesso, 2 folha(s) | planta_nao_identificada |
| commons_dogs_1.jpg | baixa_confianca | planta_nao_identificada |
| commons_green-bags_2.jpg | sucesso, 4 folha(s) | planta_nao_identificada |
| commons_green-balloons_1.jpg | sucesso, 1 folha(s) | planta_nao_identificada |
| commons_green-balloons_2.jpg | sucesso, 2 folha(s) | planta_nao_identificada |
| commons_green-bottles_1.jpg | sucesso, 7 folha(s) | sucesso, 2 folha(s) |
| commons_green-bottles_2.jpg | sucesso, 1 folha(s) | baixa_confianca |
| commons_green-cups_2.jpg | sucesso, 1 folha(s) | planta_nao_identificada |
| commons_green-cups_3.jpg | baixa_confianca | planta_nao_identificada |
| commons_green-mugs_3.jpg | sucesso, 2 folha(s) | sucesso, 1 folha(s) |
| commons_green-t-shirts_1.jpg | sucesso, 1 folha(s) | baixa_confianca |
| commons_human-hands_3.jpg | sucesso, 1 folha(s) | planta_nao_identificada |
| commons_kitchen-counters_1.jpg | sucesso, 2 folha(s) | sucesso, 1 folha(s) |
| commons_laptops_1.jpg | sucesso, 2 folha(s) | baixa_confianca |
| commons_laptops_2.jpg | sucesso, 3 folha(s) | planta_nao_identificada |
| commons_red-objects_2.jpg | baixa_confianca | planta_nao_identificada |
| commons_shoes_1.jpg | sucesso, 1 folha(s) | planta_nao_identificada |
| commons_shoes_2.jpg | sucesso, 1 folha(s) | planta_nao_identificada |
| commons_yellow-objects_1.jpg | sucesso, 2 folha(s) | planta_nao_identificada |
| commons_yellow-objects_3.jpg | sucesso, 1 folha(s) | planta_nao_identificada |

Mudaram (treino do BRACOT (1 a cada 8)):

| foto | sem o filtro | com o filtro |
|---|---|---|
| 20190831_163057.jpg | sucesso, 9 folha(s) | sucesso, 8 folha(s) |
| 20190831_164946.jpg | sucesso, 2 folha(s) | sucesso, 3 folha(s) |

## Limitações

- O corte sai das folhas anotadas do treino, nas fotos em que o detector treinou.
- Só folhas de campo do BRACOT: as folhas do BRACOL (foto inteira, fundo claro) não estão nesta máquina e não foram medidas.
- Objeto com cor de folha (verde, amarelo) passa pelo filtro.
- As fotos sem planta do Commons motivaram o brilho mínimo: o número honesto sai de um conjunto novo, que não guiou nenhuma escolha.
