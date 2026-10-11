**Folhas novas (a validação: o t_s não saiu delas).**

| | regra em duas etapas | argmax, corte 0,61 |
|---|---|---|
| chamar saudável dado S | 80,5% (66,0% a 89,8%) | 0,0% (0,0% a 8,6%) |
| chamar saudável dado D | 6,7% (2,6% a 15,9%) | 0,0% (0,0% a 6,0%) |

- **Em S, a diferença é sustentada.** O argmax com 0,61 descarta 40 das 41 S, porque nenhuma
  passa de 0,61. Mesmo sem corte, o argmax só chama saudável 19,5% (10,2% a 34,0%).
- **Em D, os intervalos se tocam**, então a diferença não é sustentada. Mas a estimativa da
  regra (6,7%) passa do teto de 5% usado para fixar o t_s, e o limite superior é 15,9%.
- **A AUC de P(saudável)** é 0,95 (0,89 a 0,99).

**As novas são piores que as 82 antigas nos três números, sem diferença sustentada.**
- O recall de S cai de 83,3% para 80,5%.
- O falso saudável em D sobe de 4,3% para 6,7%.
- A AUC vai de 0,96 para 0,95.
- Os intervalos se sobrepõem bastante. É o esperado de um limiar escolhido nas antigas e medido
  em folhas independentes.
- As máscaras das novas são mais limpas, porque o detector treinou nessas fotos. Isso, se algo,
  favorece as novas, e mesmo assim elas saíram um pouco piores.
- A curva de t_s fica só como informação, e o t_s continua 0,083. Nas novas, 0,10 daria 73,2% /
  1,7%, e 0,05 daria 95,1% / 20,0%.

**Os "?" são a maior incerteza.**
- **Quantos são:** 37% das folhas novas e 32% das antigas.
- **O que a regra faz com eles:** chama saudável 32 dos 59 "?" das novas e 28 dos 38 das
  antigas.
- **Limites, nas novas:**
  - com todos os "?" como D, o falso saudável em D vai a 30,3% (22,7% a 39,0%);
  - com todos como S, o recall de S cai para 65,0%.
- O risco real de chamar saudável uma folha doente depende justamente das folhas ambíguas, que
  estes rótulos (sem especialista) não resolvem.
- **Onde se concentram:** em fragmentos (50% nas novas) e na borda (43%).
- **Mesmo nas folhas sem nenhum desses tipos,** as novas têm 34% de "?", contra 15% nas antigas
  (os intervalos se tocam).

**A 2ª etapa (t_d = 0,65, proposta) não resolve as doenças no campo.**
- Das 60 D novas, 22 saem como doença e 34 são descartadas; o argmax com 0,61 dá 23 e 37.
- A cobertura das doentes fica baixa nas duas regras.

**Recomendação (a decisão é do gestor): ligar a regra por padrão, junto com a decisão do limiar
de doença.**
- **No campo, o padrão atual não chama saudável nenhuma folha.** São 0 das 77 S rotuladas, nos
  dois conjuntos, e toda folha saudável vira `baixa_confianca` ou, raramente, doença. A regra
  fixada chama saudável cerca de 80% das S em folhas independentes.
- **Custo no campo:** cerca de 7% das D evidentes viram saudáveis (até 16% pelo intervalo), e
  possivelmente cerca de 30% se as folhas ambíguas forem doentes.
- **Custo no BRACOL:**
  - fora da dobra, a regra completa fica abaixo do argmax (91,7% contra 95,1%, sustentado);
  - na validação do modelo congelado, a diferença não se sustenta (95,3% contra 96,1%).

Para ligar, faltam três decisões:
1. **O limiar de doença:** 0,65 é a proposta, e sem ele a regra não liga.
2. **A confiança informada das folhas saudáveis:** é P(saudável), por volta de 0,1. Vale
   decidir com o frontend como mostrar esse número, para não aparecer como "10% de confiança"
   num diagnóstico de folha saudável.
3. **Os limites, registrados no README:**
   - o falso saudável foi medido só em D evidentes;
   - os rótulos não são de especialista;
   - é uma lavoura só.

Se um falso saudável acima de 5% for inaceitável, não ligue. Os dados novos não mostram que a
regra respeita os 5% (6,7%, de 2,6% a 15,9%); só não mostram que ela falha.
