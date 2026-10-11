**Efeito dos padrões de 11/10/2026 (recorte B e filtro de fragmentos em 0,5), contra o
`pipeline_base` (recorte A, sem filtro).** Mesmo classificador, mesmo detector e mesmas fotos;
nenhum recorte foi olhado neste run.

- **T1 (BRACOL):**
  - **Cadeia:** a que está em uso, a B, acerta 94,0%, contra 94,4% da A: queda de 0,4 ponto.
    Os intervalos se sobrepõem.
  - **Só o recorte (sem a regra 1):** cai de 92,4% (A) para 88,4% (B), uma queda de 4,0
    pontos. A regra 1 (foto inteira) cobre quase toda a diferença no BRACOL. O A rende mais lá
    porque o fundo das fotos carrega a sessão: o T1 não serve de evidência para o modo de
    recorte.
  - **Filtro de fragmentos:** tirou 1 folha de 1 imagem (id 1427, de 3 para 2 folhas), sem
    mudar a classe. Nas outras 251 imagens, as folhas, os planos, as regras e as classes são
    os mesmos do `pipeline_base`.
  - **Limiar de 0,61:** cobertura e acurácia iguais às do `pipeline_base`. Na cadeia B,
    cobertura de 91,7% e acurácia de 95,2%; na A, 92,9% e 96,2%.
- **T2 (campo, sem rótulo de classe):**
  - **Folhas classificadas:** caem de 318 para 239. O filtro de fragmentos tira 79, e as
    cortadas pela borda caem de 192 para 117.
  - **Os "?" dos rótulos de campo:** 51% entre as folhas que o filtro tira, contra 24% entre as
    que ficam (passo 4).
  - **Classes no modo B:**
    - saudável cai de 42 para 24 (de 13% para 10% das folhas);
    - ferrugem, de 126 para 96;
    - bicho-mineiro, de 70 para 56;
    - phoma, de 75 para 59;
    - cercosporiose, de 5 para 4.
  - **Fotos em `baixa_confianca` no limiar de 0,61:** no modo B, sobem de 20 para 25 de 46; no
    A, de 8 para 12. Nas 5 fotos a mais (4 no A), as únicas folhas acima do limiar eram as que
    o filtro tira.
  - **O filtro nunca tirou todas as folhas de uma foto.** A fração visível é relativa à
    mediana da foto, então pelo menos metade das folhas fica, e uma folha sozinha tem fração 1.
    Só com `preenchimento_minimo` ligado (hoje em 0) aparece o erro de fragmentos.
- **T3 (CPU):** continua abaixo dos 5 s do RNF02 (tabela acima).
  - **Contra o `pipeline_base`:** a mediana vai de 0,82 para 0,90 s, e o pior caso estimado
    (15 folhas), de 1,75 para 2,44 s.
  - **Parte disso é variação de medida nesta máquina.** A classificação de 15 folhas usa a
    mesma rede, com a mesma entrada nos dois modos, e mesmo assim foi de 1,25 para 1,40 s.
  - **Uma rodada anterior com os mesmos padrões** deu 0,88 e 2,38 s.
  - O modo B só acrescenta o preenchimento do fundo em cinza no recorte.
