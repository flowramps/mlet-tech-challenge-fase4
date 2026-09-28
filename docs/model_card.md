# Model Card — Baseline de Risco de Crédito

| | |
|---|---|
| **Identificador** | `credito` — baseline de risco de inadimplência |
| **Versão** | 0.1.0 |
| **Tipo** | Classificador binário de probabilidade |
| **Arquitetura publicada** | `XGBClassifier` (300 árvores, profundidade 5, `scale_pos_weight` pela razão de classes), encapsulado em `sklearn.pipeline.Pipeline` |
| **Artefato** | `models/model.joblib` — modelo e metadados no mesmo pacote |
| **Semente** | 42 |
| **Reprodução** | `make data && make train` (monitoramento: `make monitor`) |

---

## Uso pretendido

### Para que serve

Estimar a probabilidade de um cliente entrar em dificuldade financeira grave (atraso de 90
dias ou mais) nos **próximos dois anos**, a partir de dez atributos de crédito e renda.

O modelo foi construído como **baseline de sustentação**: ele existe para dar um ponto de
referência estável contra o qual medir degradação, drift e o efeito de mudanças no dado. É
um instrumento de engenharia de confiabilidade antes de ser um instrumento de decisão.

### Para quem

Equipes de risco e de engenharia de ML que operam o modelo — não o cliente final, que
nunca interage com ele diretamente.

### Usos explicitamente fora de escopo

- **Decisão de crédito totalmente automatizada, sem revisão humana.** Ver a seção de
  conformidade.
- **Precificação** (definição de taxa ou limite): o modelo foi calibrado para ordenar
  risco, não para produzir uma probabilidade fidedigna em valor absoluto.
- **Populações fora da Referência**, em especial clientes sem renda declarada — o modelo
  nunca viu esse perfil. Ver limitações.
- **Qualquer uso em que a saída seja interpretada como característica da pessoa**, e não
  como estimativa estatística condicionada a dez variáveis.

---

## Dados de treino

### Origem

> Credit Fusion and Will Cukierski. *Give Me Some Credit.* Kaggle, 2011.

Obtido pela redistribuição do OpenML (id 46929, licença Public), que serve o mesmo arquivo
por URL pública com checksum verificado a cada execução. O acesso direto ao Kaggle exige
credencial e quebraria a reprodutibilidade em clone limpo e em CI.

Dados **históricos e anônimos**, sem identificador pessoal. Não há dado sensível nos termos
do art. 5º, II da LGPD entre as dez variáveis. `age` é o único atributo protegido presente.

### Dataset de Referência

| | Linhas | Positivos |
|---|---:|---:|
| Arquivo bruto | 150.000 | 10.026 (6,68%) |
| **Referência (após limpeza)** | **117.917** | **8.186 (6,94%)** |

32.083 linhas descartadas (retenção de 78,61%), contabilizadas por motivo: `renda_nula`
29.186 · `razao_divida_implausivel` 2.106 · `duplicata` 646 · `atraso_sentinela` 144 ·
`idade_invalida` 1 · `dependentes_nulo` 0 · `campo_invalido` 0. Os dois zeros são
medições: as linhas que eles filtrariam já foram removidas antes, ou não existem no
arquivo. Os filtros existem porque o contrato de ingestão exige o que eles exigem.

Partições estratificadas pelo alvo, semente 42: **treino 70.749 · validação 23.584 · teste
23.584**, com 6,94% de positivos em cada.

### Variáveis de entrada

`RevolvingUtilizationOfUnsecuredLines`, `age`, `NumberOfTime30-59DaysPastDueNotWorse`,
`DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`,
`NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`,
`NumberOfDependents`.

Todo lote — de treino ou de inferência — passa por um contrato de seis regras antes de
qualquer uso. As regras e os defeitos medidos que as justificam estão no README.

### Alvo

`inadimplente` — 1 se houve dificuldade financeira grave nos dois anos seguintes, 0 caso
contrário. **Desbalanceado: 6,94% de positivos.**

---

## Métricas

Todas medidas no **conjunto de teste** (23.584 linhas, nunca usado para escolher o
candidato), limiar de decisão 0,5.

### Seleção do candidato

O campeão foi escolhido pelo AUC-PR na **validação**, não no teste — escolher no teste
tornaria o teste parte do treino e a métrica publicada seria otimista por construção.

| Candidato (validação) | AUC-PR | AUC-ROC | Recall + | Precisão + | Acurácia |
|---|---:|---:|---:|---:|---:|
| `regressao_logistica` | 0,3469 | 0,8123 | 0,6237 | 0,2506 | 0,8444 |
| **`xgboost`** (campeão) | **0,3612** | 0,8437 | 0,6970 | 0,2325 | 0,8192 |

### Desempenho do modelo publicado

| Métrica | Valor | Leitura |
|---|---:|---|
| **AUC-PR** | **0,3716** | Métrica primária. Linha de base aleatória = 0,0694 (taxa de positivos): **5,4× melhor que o acaso** |
| **Recall da classe positiva** | **0,6915** | Métrica de negócio. 69,15% dos inadimplentes são identificados |
| Precisão da classe positiva | 0,2350 | Cerca de 3 em cada 4 recusas seriam bons pagadores |
| AUC-ROC | 0,8421 | Figura secundária: infla com desbalanceamento, não decide nada |
| Acurácia | 0,8223 | **Não é critério** — ver abaixo |
| Taxa de positivos | 0,0694 | |
| n | 23.584 | |

**Por que acurácia não é critério.** Um modelo constante que responde "não vai inadimplir"
para todo mundo atinge **0,9306 de acurácia** e **0,0000 de recall positivo** sobre o mesmo
conjunto de teste. Acurácia, neste regime, premia o modelo inútil.

O efeito do tratamento de desbalanceamento, medido com a mesma arquitetura e mesmos
hiperparâmetros, mudando apenas o peso da classe:

| Regressão logística (teste) | Recall + | Acurácia | AUC-PR |
|---|---:|---:|---:|
| sem `class_weight` | 0,1436 | **0,9346** | 0,3590 |
| com `class_weight="balanced"` | **0,6213** | 0,8474 | 0,3648 |

O modelo sem tratamento tem acurácia mais alta e encontra um quarto dos inadimplentes.

### Desempenho por faixa etária

`age` é atributo protegido. Afirmar que o modelo não discrimina sem medir por grupo é
afirmação sem evidência.

| Faixa | n | Positivos reais | Taxa de recusa | Taxa de aprovação | Recall + | Prob. média |
|---|---:|---:|---:|---:|---:|---:|
| 18-25 | 486 | 8,64% | 34,98% | 65,02% | 0,7857 | 0,381 |
| 26-40 | 5.399 | 10,58% | 31,91% | 68,09% | 0,7671 | 0,390 |
| 41-60 | 11.302 | 7,24% | 21,62% | 78,38% | 0,6822 | 0,304 |
| 61+ | 6.397 | 3,22% | 7,50% | 92,50% | 0,5000 | 0,166 |

**Como ler.** A taxa de recusa cai monotonicamente com a idade e a taxa de inadimplência
real também: a diferença de tratamento acompanha uma diferença de risco observada, não é
viés puro. Mas as proporções não batem — entre os extremos, a recusa varia **4,7×** e o
risco real apenas **2,7×**. O modelo é mais severo com clientes de 18 a 25 anos do que o
risco medido justifica.

Dois pontos de atenção adicionais:

- O recall na faixa **61+ é 0,5000**, o mais baixo da tabela: o modelo enxerga pior o
  inadimplente idoso. Metade deles passa.
- A faixa **18-25 tem apenas 486 linhas** no teste (2,1% do conjunto). As métricas dessa
  faixa têm intervalo de confiança largo e devem ser lidas com essa ressalva.

Estes números estão **medidos, não tratados**. A análise formal de *disparate impact* e
qualquer mitigação são trabalho da etapa de governança.

### Degradação sob drift

O modelo publicado foi aplicado, **sem retreino**, a seis lotes mensais de Produção
simulada construídos sobre a mesma partição de teste (23.584 linhas). O mês 0 é idêntico à
partição, byte a byte — é a linha de base, não uma aproximação dela. Os seis lotes passam
no contrato de dados com zero linhas reprovadas: **dado deslocado continua sendo dado
válido**, e nenhuma degradação abaixo vem de dado inválido.

| Lote | Rótulo = real | AUC-PR | Recall + | Piso (= taxa +) | Lift acima do piso | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|
| `mes_00` | 100,00% | 0,3716 | 0,6915 | 0,0694 | 0,3022 | 0,8421 |
| `mes_01` | 99,03% | 0,3478 | 0,6381 | 0,0791 | 0,2687 | 0,8109 |
| `mes_02` | 97,77% | 0,3233 | 0,5661 | 0,0918 | 0,2315 | 0,7686 |
| `mes_03` | 96,26% | 0,3093 | 0,5048 | 0,1068 | 0,2025 | 0,7279 |
| `mes_04` | 94,61% | 0,2944 | 0,4531 | 0,1233 | 0,1710 | 0,6992 |
| `mes_05` | 92,86% | 0,2882 | 0,4158 | 0,1408 | 0,1474 | 0,6645 |
| `mes_06` | 91,12% | 0,2814 | 0,3798 | 0,1582 | 0,1232 | 0,6342 |

**Como ler, e a armadilha.** O recall da classe positiva cai 45,1% — o modelo que
identificava 69,15% dos inadimplentes identifica 37,98%. A AUC-PR cai 24,3%, o que parece
moderado até se lembrar de que a linha de base aleatória da AUC-PR **é a taxa de positivos
do lote**, e essa taxa dobra (6,94% → 15,82%, 2,28×). O piso sobe enquanto a métrica desce.
O que resta acima do piso — poder discriminativo de verdade — cai de 0,3022 para 0,1232,
**59,2%**, e a AUC-ROC confirma por um caminho insensível à prevalência (−24,7%).
**Reportar AUC-PR sozinha sobre uma população em movimento subestima a degradação em mais
da metade.**

**De onde a queda vem.** Decompondo a perda de AUC-ROC do mês 6 (0,2079) por intervenção —
deslocando uma variável por vez e mantendo as demais fixas:

| Canal isolado | Queda | Fração |
|---|---:|---:|
| `MonthlyIncome` | 0,0017 | 0,8% |
| `DebtRatio` | 0,0047 | 2,3% |
| `NumberOfTime30-59DaysPastDueNotWorse` | 0,0142 | 6,8% |
| **concept drift** (`P(y\|X)`) | **0,0998** | **48,0%** |
| interação entre os quatro | 0,0876 | 42,1% |

As duas variáveis com PSI mais alto (`DebtRatio` 0,5046 e `MonthlyIncome` 0,2634) explicam
**3,1% da degradação entre as duas**. O concept drift, que nenhum PSI por feature enxerga
porque não move `P(X)`, explica quase metade. **Drift de feature é diagnóstico; a
degradação medida é o alarme** — e um alerta de drift não indica, por si, qual variável
investigar.

Esta decomposição só é possível porque a Produção é simulada e o processo gerador é
controlável. Ver as limitações abaixo.

---

## Limitações

- **A Referência não representa a população inteira.** 19,82% do arquivo bruto vem sem
  `MonthlyIncome` e é descartado. O modelo nunca viu esse perfil, e esse perfil existe em
  produção. Hoje o contrato o bloquearia em vez de pontuá-lo — o que é seguro, mas não é
  atendimento. Tratar essa fatia exige política de imputação ou modelo separado.
- **Limiar fixo em 0,5**, não otimizado para a assimetria de custo entre aprovar um
  inadimplente e recusar um bom pagador. O ponto de operação atual foi herdado do default,
  não escolhido por curva de custo.
- **Probabilidades não calibradas.** O modelo ordena risco bem (AUC-ROC 0,8421); não há
  garantia de que uma saída de 0,30 corresponda a 30% de inadimplência observada. Não usar
  a saída como probabilidade absoluta sem calibração.
- **Hiperparâmetros são padrões conservadores de baseline**, sem busca. Deliberado: só
  assim a comparação entre os dois candidatos é evidência, e não um vencedor ajustado a
  dedo.
- **Dado histórico de contexto diferente.** O dataset é de 2011, de crédito ao consumidor
  nos Estados Unidos. Aplicá-lo a outra jurisdição, outro produto ou outro período exige
  revalidação completa, não só retreino.
- **O gate compara contra os metadados gravados no artefato**, não contra uma reavaliação
  do incumbente sobre a partição corrente. Se as partições mudarem, a comparação passa a
  ser entre números medidos sobre conjuntos diferentes.
- **A degradação acima foi medida contra Produção simulada, não observada.** O processo
  gerador foi escrito por este repositório e calibrado contra o dado real, mas os números
  são evidência sobre o instrumento de monitoramento, não sobre o mundo. Não existe lote de
  produção real medido aqui.
- **A degradação é medida com rótulo instantâneo.** A simulação entrega o rótulo junto com
  as features. Em crédito o rótulo demora — "dificuldade grave nos próximos dois anos" só é
  observável dois anos depois —, e é esse atraso que torna a degradação silenciosa um
  problema operacional. A defasagem não está modelada.
- **A atribuição causal por intervenção não é reproduzível em produção.** Ela exige
  controlar o processo gerador para mover uma variável de cada vez. O substituto disponível
  em produção — importância por permutação sobre o lote degradado — mede sensibilidade do
  modelo, não causa do deslocamento, e não enxerga nem o concept drift nem o termo de
  interação, que juntos respondem por 90,1% da queda medida.
- **Sem rótulo, metade da degradação é invisível.** Os 48,0% atribuídos ao concept drift só
  aparecem porque a métrica foi calculada contra o rótulo verdadeiro. Nenhum detector de
  drift de feature acusaria essa parte.

---

## Riscos

| Risco | Descrição | Mitigação atual | Pendente |
|---|---|---|---|
| **Falso positivo em escala** | Precisão de 0,2350: cerca de 3 em cada 4 recusas atingem bons pagadores. Perda de receita e de relacionamento | Recall privilegiado por decisão explícita e documentada de custo | Curva de custo para escolher o limiar |
| **Viés etário** | Recusa varia 4,7× entre faixas contra 2,7× de variação no risco real. Severidade desproporcional com jovens | Medido e publicado a cada execução em `metrics.json` | Análise formal de *disparate impact* e mitigação |
| **Exclusão de quem não declara renda** | 19,82% do bruto. Hoje bloqueado no contrato, não pontuado | Bloqueio explícito e auditável, em vez de pontuação sobre dado inventado | Política de imputação ou modelo dedicado |
| **Degradação silenciosa** | O modelo continua respondendo com dado deslocado. Medido sob drift simulado: recall + cai 45,1% e o lift acima do piso 59,2%, sem uma única linha inválida | Histórico de execuções *append-only*; `make monitor` mede PSI/KS por feature, a degradação do campeão lote a lote e a decomposição causal, com veredito consolidado | Medição sobre lote de produção real, com a defasagem de rótulo modelada |
| **Alarmar na variável errada** | PSI aponta onde a distribuição se moveu, não quanto custou. As duas variáveis de PSI mais alto explicam 3,1% da degradação medida | O relatório consolidado publica a decomposição causal **antes** da tabela de PSI, e o resumo do gate carrega a ressalva embutida | Detecção de concept drift sem rótulo |
| **Dado quebrado na ingestão** | Nulo, sentinela, duplicata, valor implausível | Contrato de 6 regras bloqueando antes do treino e da inferência | Aplicação do mesmo contrato no endpoint de serviço |
| **Variável correlacionada com atributo protegido** | Renda e número de dependentes podem carregar sinal de gênero, raça ou região não observados | Nenhuma | Auditoria de *proxy* para atributos protegidos não presentes no dado |
| **Uso fora do escopo** | Interpretar a saída como precificação ou como característica da pessoa | Este documento | Contrato de uso junto à equipe consumidora |

---

## Conformidade — decisão automatizada e revisão humana

A **Lei nº 13.709/2018 (LGPD), art. 20**, garante ao titular o direito de solicitar revisão
de decisões tomadas unicamente com base em tratamento automatizado de dados pessoais que
afetem seus interesses — concessão de crédito é o exemplo nomeado no próprio caput. O mesmo
artigo assegura o direito a informação sobre os critérios e procedimentos utilizados.

Consequências diretas para este modelo:

1. **Nenhuma recusa pode ser final sem caminho de revisão humana.** O modelo produz uma
   recomendação, não um veredito.
2. **Toda decisão precisa ser explicável ao titular** em termos dos critérios utilizados.
3. **A auditabilidade é requisito, não conveniência.** É por isso que a política de limpeza
   contabiliza cada descarte por motivo, o histórico de treinos é *append-only* e o recorte
   etário é publicado a cada execução.

Esta etapa estabelece a **base técnica** que torna esses direitos exequíveis: dado
rastreável, decisão registrada, métrica por grupo medida. O processo de revisão em si —
fluxo de contestação, prazos, explicabilidade individual da predição e documentação de
conformidade — é desenvolvido na etapa de governança.

---

*Métricas do modelo publicado medidas na execução registrada em `metrics/metrics.json`,
reproduzível por `make data && make train`. Degradação sob drift e decomposição causal
medidas na execução registrada em `reports/monitoramento.json`, reproduzível por
`make monitor`. Os dois arquivos são gerados, não versionados.*
