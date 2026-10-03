# Governança — proteção de dados, equidade e causalidade

| | |
|---|---|
| **Escopo** | O modelo de risco publicado (`models/model.joblib`), a Referência sobre a qual ele treina e a API de scoring (`credito.api`) |
| **Normas** | Lei nº 13.709/2018 (LGPD) · Lei nº 12.414/2011 (Cadastro Positivo) · Lei nº 8.078/1990 (CDC), art. 43 |
| **Leitura complementar** | [`model_card.md`](model_card.md) (uso pretendido, métricas, riscos) · [`monitoring_plan.md`](monitoring_plan.md) (alarme de produção) |

Toda citação de lei abaixo foi conferida contra o texto oficial publicado no portal da
Presidência da República, na redação vigente — não citada de memória. Isso importou: o
art. 20 da LGPD já teve três redações, e a que está em vigor não diz o que muita gente
supõe (ver a seção de revisão de decisão automatizada).

Todo número abaixo vem de uma execução reproduzível: `make train` (equidade, gravada em
`metrics/metrics.json`) e `make auditar-privacidade` (risco de reidentificação).

---

## 1. Base legal para a decisão de crédito

**A hipótese que autoriza o tratamento é a proteção do crédito — LGPD, art. 7º, X**:
*"para a proteção do crédito, inclusive quanto ao disposto na legislação pertinente"*. A
legislação pertinente é a Lei nº 12.414/2011, que disciplina o histórico de crédito, e o
art. 43 do CDC, que disciplina cadastros de consumidores.

Quando a pontuação acontece porque o próprio cliente pediu crédito, soma-se o **art. 7º, V**:
*"quando necessário para a execução de contrato ou de procedimentos preliminares
relacionados a contrato do qual seja parte o titular, a pedido do titular dos dados"*. A
análise de risco é o procedimento preliminar do contrato de crédito.

**Por que não o consentimento (art. 7º, I).** Consentimento é a primeira hipótese da lista
e a escolha mais comum por reflexo — e é a errada aqui, por dois motivos que estão no
texto da lei:

- **Ele é revogável a qualquer momento** (art. 8º, §5º), e a revogação encerra o tratamento
  (art. 15, III). Uma análise de risco que o avaliado pode interromper no meio deixa de ser
  análise de risco.
- **Ele precisa ser livre** (art. 5º, XII: *"manifestação livre, informada e inequívoca"*).
  Condicionar a concessão do crédito ao "aceite" do tratamento não é manifestação livre —
  é a contrapartida de um contrato, que é exatamente o que o art. 7º, V descreve.

**Os princípios que a base legal não dispensa** (art. 6º): finalidade (I), adequação (II),
**necessidade** — *"limitação do tratamento ao mínimo necessário"* (III) —, transparência
(VI), **não discriminação** — *"impossibilidade de realização do tratamento para fins
discriminatórios ilícitos ou abusivos"* (IX) — e **responsabilização e prestação de
contas** (X). As seções seguintes são, cada uma, a demonstração de um deles.

---

## 2. Que dado é tratado, e como

### 2.1 A Referência: sem identificador, mas não anônima

O dataset de treino (*Give Me Some Credit*, redistribuído pelo OpenML) não tem nome, CPF,
endereço ou qualquer identificador direto. Nenhuma das dez variáveis é dado sensível nos
termos do art. 5º, II — a lista legal é origem racial ou étnica, convicção religiosa,
opinião política, filiação sindical, saúde, vida sexual, dado genético ou biométrico, e
**idade não está nela**. Idade também não está entre as informações sensíveis que a Lei
do Cadastro Positivo proíbe anotar (art. 3º, §3º, II).

**Isso não basta para chamar o dado de anonimizado.** A LGPD só tira um dado do regime de
dado pessoal quando a anonimização não pode ser revertida *"com esforços razoáveis"*
(art. 12). A pergunta certa não é "tem CPF?", é "uma linha pode ser apontada?". Medido com
k-anonimato (k = 5) sobre as 117.917 linhas da Referência:

| Quase-identificadores | Linhas únicas | Em grupo com menos de 5 |
|---|---:|---:|
| idade | 1 (0,00%) | 11 (0,01%) |
| idade + dependentes | 62 (0,05%) | 235 (0,20%) |
| idade + renda | 48.677 (**41,28%**) | 69.671 (59,08%) |
| idade + renda + dependentes | 62.852 (**53,30%**) | 89.165 (**75,62%**) |

**Mais da metade das pessoas da Referência é a única com aquela idade, aquela renda mensal
e aquele número de dependentes.** Quem conhece esses três fatos sobre alguém — e um
empregador, um parente ou um vizinho conhece — encontra a linha, e com ela o histórico de
atrasos e o rótulo de inadimplência. **A Referência é dado pessoal pseudonimizado, não
anonimizado**, e é tratada aqui sob o regime completo da LGPD. O model card das etapas
anteriores a descrevia como "anônima"; a descrição foi corrigida.

**A mitigação, medida.** Generalizar valor exato em faixa (`credito.governanca.privacidade.
generalizar`: faixa etária, quintil de renda, dependentes até "3 ou mais"):

| Quase-identificadores (generalizados) | Linhas únicas | Em grupo com menos de 5 | k mínimo |
|---|---:|---:|---:|
| faixa etária | 0 | 0 | 2.219 |
| faixa etária + dependentes | 0 | 0 | 18 |
| faixa etária + quintil de renda | 0 | 0 | 13 |
| faixa etária + quintil de renda + dependentes | 2 | **12 (0,01%)** | 1 |

De 75,62% para 0,01%. As 12 linhas restantes estão em combinações raras e precisam de
**supressão**, não de mais generalização. Com decil de renda em vez de quintil sobram 25 —
por isso o quintil.

A generalização **não entra no treino**: o modelo continua aprendendo sobre o valor exato,
porque é ele que tem poder preditivo. Ela é a transformação obrigatória antes de qualquer
uso da Referência que **não seja** treinar o modelo — conservação após o fim do
tratamento, estudo, compartilhamento (ver a seção 3).

### 2.2 Em produção: o que a API recebe, guarda e expõe

Num sistema real, o cliente que pede crédito chega com nome, CPF, endereço e conta. **Nada
disso entra no serviço de scoring.** `ScoreRequest` aceita exatamente as dez variáveis de
`credito.schema.FEATURES`; a ligação entre a pessoa e o vetor de features fica no sistema
de origem, que já precisa dela para o contrato.

E isso é propriedade do código, não promessa sobre quem chama: um CPF e um nome enviados
por engano junto com as features **são descartados na validação** (`extra="ignore"`,
declarado no `model_config` e travado por
`test_identificador_enviado_a_mais_e_descartado_antes_de_chegar_ao_modelo`) — não chegam
ao modelo, ao log nem à métrica. É o princípio da necessidade (art. 6º, III) aplicado na
fronteira do serviço.

**Mas o vetor de features ainda é dado pessoal.** Idade e renda exatas apontam sozinhas
41,28% das pessoas da Referência (seção 2.1). Tirar o CPF reduz o risco; não tira o dado do
regime da lei. O que o serviço faz com ele:

| O que | Onde fica | Contém dado pessoal? |
|---|---|---|
| O vetor de features da requisição | memória do processo, até a resposta | **sim** — fora o registro de decisão, não é gravado em lugar nenhum |
| **Registro de decisão** | `decisoes/decisoes.jsonl` (no container, o volume `decisoes`) | **sim** — o vetor de features de cada decisão, **sem identificador nenhum**; retenção de 5 anos (seção 3) |
| Log da aplicação | stdout do container | não — a aplicação só loga o carregamento do modelo; o log de acesso do uvicorn grava IP e porta de quem chamou, método, rota e status, nunca o corpo. Quem chama é o sistema de origem, não o titular — verificado enviando um CPF: zero ocorrências no log |
| Métricas Prometheus | `/metrics` | não — rótulos vêm de conjunto fechado (rota, método, status), nunca da entrada |
| Janela da taxa de aprovação | memória, últimas 2.000 decisões | não — só probabilidades, sem identidade; perdida ao reiniciar |

### 2.3 Os artefatos gerados

| Artefato | Contém linha individual? |
|---|---|
| `models/model.joblib` | não — parâmetros do pipeline e agregados das árvores |
| `metrics/metrics.json`, `metrics/training_history.jsonl` | não — métricas agregadas, inclusive por faixa etária |
| `reports/drift_mes_*.html` (Evidently) | **não — medido**, ver abaixo |
| `mlruns/` (MLflow) | não — parâmetros, métricas agregadas e os artefatos acima |

Os relatórios do Evidently pesam 4,72 MB cada — tamanho suficiente para embutirem linhas
cruas, não só histogramas. Medido em vez de suposto: os vetores dos gráficos guardam
números em precisão completa, então um valor individual apareceria inteiro; comparando os
3.832 pontos distintos dos 20 traços não constantes do relatório do mês 6 contra os
valores reais do lote, há 52 coincidências — contra 49 de um controle com os mesmos pontos
deslocados no sexto decimal. Mesma taxa, cerca de 1,3%: coincidência, não dado embutido. O maior bloco binário do arquivo,
126 KB em base64, é uma fonte tipográfica. **Os relatórios guardam estatística agregada,
não pessoas.**

---

## 3. Plano de retenção

O fim do tratamento acontece quando a finalidade é alcançada, quando o período de
tratamento acaba, a pedido do titular ou por determinação da autoridade (art. 15); depois
dele, o dado é eliminado, salvo conservação para obrigação legal, estudo, transferência
lícita ou uso exclusivo do controlador **desde que anonimizado** (art. 16). Os tetos que a
legislação de crédito já fixa: informação negativa por até **5 anos** (CDC, art. 43, §1º);
informação de adimplemento por até **15 anos** (Lei nº 12.414/2011, art. 14).

| Dado | Retenção | Depois | Fundamento |
|---|---|---|---|
| Vetor de features da requisição | **nenhuma** — descartado ao responder | — | necessidade (art. 6º, III) |
| Registro de decisão¹ (`id_decisao`, data, vetor de features, probabilidade, decisão, limiar, candidato, sha256 do modelo) | **5 anos** a partir da decisão | eliminação | revisão da decisão (art. 20; Lei nº 12.414, art. 5º, VI) e prestação de contas (art. 6º, X); prazo alinhado ao teto do CDC para informação negativa |
| Referência (treino) | enquanto o modelo treinado nela estiver em produção, mais um ciclo de retreino | generalização + supressão (seção 2.1), ou eliminação | finalidade (art. 6º, I); conservação só anonimizada (art. 16, IV) |
| `models/`, `metrics/`, `reports/`, `mlruns/` | enquanto o modelo correspondente estiver em produção, mais um ciclo | eliminação | auditoria do campeão contra o incumbente; agregados, sem linha individual |

O prazo de 5 anos do registro de decisão e o "mais um ciclo de retreino" são **decisões
deste projeto**, não números que a lei fixa: a lei dá os tetos e o critério (finalidade
alcançada); o prazo concreto é do controlador, e está declarado aqui para poder ser
contestado. A razão do "mais um ciclo": o gate de promoção compara o candidato novo contra
o incumbente, e auditar essa comparação depois exige o incumbente e os dados dele.

¹ **Como o registro funciona** (`credito.governanca.decisoes`). Cada `/score` gera um
`id_decisao` e o devolve na resposta; **a API não recebe identificador nenhum, nem
pseudônimo** — é o sistema de origem que guarda o vínculo entre o id e a pessoa. Isso é
mais forte que gravar um pseudônimo do cliente: o registro sozinho não liga decisão a
ninguém. A decisão é gravada em JSONL append-only, com `fsync`, **antes** de ser devolvida:
se a gravação falha, a API responde `503` e nenhuma decisão é emitida — uma decisão sem
registro não poderia ser revista. O sha256 do arquivo do modelo liga cada decisão ao modelo
exato que a tomou. A retenção é executável: `make expurgar-decisoes` remove o que passou de
5 anos (uma decisão exatamente no limite ainda fica), e é feito para rodar agendado — um
prazo que nada executa não elimina nada (art. 16). No container, o diretório de decisões é o
único caminho gravável; código e modelo são só-leitura para o processo.

Verificado contra mutação, inclusive a corrida que mais importa: uma decisão gravada
enquanto o expurgo reescreve o arquivo iria para o arquivo antigo e sumiria na troca. O
teste força essa corrida de forma determinística; ele falha 5 vezes em 5 sem a trava e
passa 5 em 5 com ela.

---

## 4. Equidade e mitigação de vieses

Idade não é dado sensível na LGPD, mas o princípio da não discriminação (art. 6º, IX) vale
para todo tratamento, e a autoridade nacional pode auditar *"aspectos discriminatórios em
tratamento automatizado"* (art. 20, §2º). Afirmar que o modelo não discrimina sem medir
por grupo seria afirmação sem evidência. Medido no conjunto de teste (23.584 linhas), com o
campeão publicado e o limiar de 0,5:

| Faixa | n | Aprovação | Razão 4/5 | Impacto adverso | Recall + | Distância ao maior recall |
|---|---:|---:|---:|:---:|---:|---:|
| 18-25 | 486 | 65,02% | **0,7030** | **sim** | 0,7857 | — (referência) |
| 26-40 | 5.399 | 68,09% | **0,7361** | **sim** | 0,7671 | 0,0186 |
| 41-60 | 11.302 | 78,38% | 0,8473 | não | 0,6822 | 0,1036 |
| 61+ | 6.397 | 92,50% | — (referência) | não | 0,5000 | **0,2857** |

As duas colunas vêm de duas métricas formais diferentes, e **elas apontam em direções
opostas**:

- **Razão de impacto adverso (regra dos 4/5)** — a taxa de aprovação de cada faixa sobre a
  da faixa mais aprovada; abaixo de 0,80 é tratado como evidência de impacto adverso. É o
  critério que reguladores aplicam a decisões de seleção. Por ele, **os jovens sofrem
  impacto adverso**: 18-25 e 26-40 ficam abaixo do corte.
- **Igualdade de oportunidade** — entre os que de fato inadimpliram, quantos o modelo
  identifica em cada faixa. Por ela, **quem sai pior são os idosos**: o modelo deixa passar
  metade dos inadimplentes 61+, contra pouco mais de um quinto dos 18-25.

A regra dos 4/5 não desconta a diferença de risco real — e ela existe: a inadimplência
observada cai de 8,64% (18-25) e 10,58% (26-40) para 3,22% (61+). Mas a desigualdade de
desfecho é **maior** que a desigualdade de risco: entre os extremos, a recusa varia 4,7×
e o risco real 2,7× (ver o model card). Parte da severidade com os jovens não é explicada
pelo risco que eles de fato carregam.

**Uma única métrica de equidade daria o veredito de uma direção e esconderia a outra.** É
por isso que as duas são publicadas em `metrics.json` a cada treino (`equidade.
quatro_quintos` e `equidade.oportunidade`), calculadas sobre o mesmo recorte por faixa que
o arquivo já publica.

### Mitigação: recomendada, medida antes de aplicada

O modelo publicado **não foi alterado** nesta etapa — a decisão foi medir formalmente e
recomendar, não retreinar. As opções, com o que cada uma custa:

1. **Limiar de decisão por faixa etária.** Fecha a razão 4/5 diretamente, aprovando mais
   jovens. Custo: usar idade explicitamente na regra de decisão é tratamento diferenciado
   declarado, e precisa de justificativa própria diante do art. 6º, IX — corrigir uma
   discriminação estatística com uma regra que distingue por idade pode ser lido como
   outra.
2. **Reponderação no treino.** Pesos por faixa e classe durante o treino, sem idade na regra
   de decisão. Custo: retreino, novo ciclo do gate de promoção e nova medição de AUC-PR e
   recall global — a mitigação pode derrubar o modelo abaixo dos pisos.
3. **Investigar o recall da faixa 61+ antes de qualquer ajuste de aprovação.** É o achado
   que a regra dos 4/5 sozinha não mostraria: o modelo aprova muito os idosos **e** enxerga
   mal o inadimplente idoso. Aprovar ainda mais jovens sem tratar isso desloca o erro, não o
   reduz.

**A regra para qualquer uma delas:** medir as **duas** métricas antes e depois. Uma
mitigação que fecha a razão 4/5 e abre a distância de recall não é mitigação, é troca de
grupo prejudicado. As duas funções existem (`credito.model.evaluate`) exatamente para que
essa medição seja um comando, não um projeto.

O que não está medido: **viés por atributo que o dado não tem.** Renda e número de
dependentes podem carregar sinal de gênero ou região, que não estão na Referência. Não há
como medir a equidade sobre um atributo ausente — está declarado como risco no model card.

---

## 5. Causalidade: por que o modelo decide o que decide, e o que muda isso

O art. 20, §1º obriga o controlador a fornecer *"informações claras e adequadas a respeito
dos critérios e dos procedimentos utilizados para a decisão automatizada"*, e a Lei do
Cadastro Positivo dá ao cadastrado o direito de *"conhecer os principais elementos e
critérios considerados para a análise de risco"* (art. 5º, IV). Esta seção consolida o que
o projeto mediu sobre esses critérios — e, principalmente, sobre **o que os faz deixar de
valer**.

**Os critérios.** O modelo usa as dez variáveis de `FEATURES` — utilização de crédito
rotativo, idade, três contagens de atraso, razão de endividamento, renda mensal, número de
linhas de crédito, de financiamentos imobiliários e de dependentes —, um classificador
XGBoost e um limiar de 0,5 sobre a probabilidade estimada de inadimplência nos dois anos
seguintes.

**O que os invalida, medido por intervenção.** Sob uma mudança econômica simulada de seis
meses, o modelo perdeu 0,2079 de AUC-ROC. Deslocando uma causa de cada vez e mantendo as
outras fixas (`credito.drift.causal`):

| Causa isolada | Fração da queda |
|---|---:|
| inflação da renda | 0,8% |
| aumento do endividamento | 2,3% |
| novo perfil de atraso | 6,8% |
| **mudança na relação entre perfil e inadimplência (concept drift)** | **48,0%** |
| interação entre as causas | 42,1% |

Três consequências para governança:

1. **Explicar uma decisão pela variável que "mais mudou" é explicar errado.** As duas
   variáveis com maior deslocamento de distribuição (endividamento e renda) respondem
   juntas por 3,1% da degradação. Uma explicação ao titular — ou a um revisor — que
   apontasse para elas descreveria o sintoma visível, não a causa.
2. **Os critérios podem deixar de valer sem que nada na entrada mude.** O concept drift não
   desloca nenhuma das dez variáveis: o mesmo perfil passa a ter outro risco. Por isso a
   informação sobre os critérios do art. 20, §1º precisa ser **versionada junto com o
   modelo e com o monitoramento** — os mesmos critérios, aplicados depois de uma mudança na
   relação entre perfil e risco, já não descrevem uma decisão bem fundamentada.
3. **A atribuição causal não existe em produção.** Ela exige controlar o processo gerador,
   o que só a simulação permite. Em produção, o que sobra é o alarme sem rótulo
   (`docs/monitoring_plan.md`) para saber **se** o modelo degradou, e a degradação medida
   com rótulo atrasado para saber **quanto** — não **por quê**.

O que o projeto **não** entrega: explicação de uma decisão individual (por que *este*
pedido foi recusado). Os critérios acima são globais. O registro de decisão (seção 3) já
guarda tudo o que essa explicação exigiria — as entradas e o modelo exato —; a explicação
em si é a continuidade natural dele.

---

## 6. Direitos do titular, e como esta arquitetura os atende

| Direito | Fundamento | Como é atendido | O que falta |
|---|---|---|---|
| Confirmação e acesso | LGPD, art. 18, I e II; Lei nº 12.414, art. 5º, II | No sistema de origem, que detém a identidade e o `id_decisao` de cada pedido; com ele, `make consultar-decisao` devolve o que o serviço de scoring gravou | — |
| Correção | LGPD, art. 18, III; Lei nº 12.414, art. 5º, III | Correção na origem e nova pontuação; o contrato de dados (`credito.contracts`) já bloqueia o dado inválido antes do modelo | — |
| Informação sobre critérios | LGPD, art. 20, §1º; Lei nº 12.414, art. 5º, IV | Este documento, a seção 5 e o model card | Explicação por decisão individual |
| **Revisão de decisão automatizada** | LGPD, art. 20; Lei nº 12.414, art. 5º, VI | Registro de decisão (seção 3) e reconstrução por `make consultar-decisao` — ver abaixo | Fluxo de revisão humana do controlador |
| Oposição | LGPD, art. 18, §2º | Pelo controlador, no canal do encarregado | — |
| Petição à autoridade | LGPD, art. 18, §1º | Direito do titular, independente da arquitetura | — |

**A revisão de decisão automatizada, e o que a lei de fato exige.** O texto original da
LGPD garantia revisão *"por pessoa natural"*. A Medida Provisória nº 869/2018 retirou essa
expressão, a Lei nº 13.853/2019 manteve a retirada, e o parágrafo que a reintroduziria foi
vetado. **A redação vigente garante o direito à revisão, mas não exige que ela seja feita
por um humano.** A Lei do Cadastro Positivo também fala em revisão sem exigir pessoa
natural.

**Este projeto adota revisão humana mesmo assim** — é uma decisão de projeto, mais rigorosa
que a lei, e está declarada como tal. A razão é a seção 4: um modelo cuja equidade aponta
para direções opostas conforme a métrica, e cujo recall cai à metade numa faixa etária, não
deveria ter suas recusas revistas por outro procedimento automatizado com os mesmos
critérios. A revisão precisa de alguém que possa decidir **contra** o modelo.

O que torna essa revisão possível hoje e o que ainda falta:

- **Hoje:** cada decisão é registrada antes de ser emitida, e
  `make consultar-decisao ID=<id_decisao>` a reconstrói: mostra as entradas, a probabilidade,
  o limiar e o modelo, e — se o modelo publicado ainda é o mesmo, pelo sha256 — **repontua e
  confere** que o resultado é idêntico. Verificado com o campeão real: uma recusa
  reconstruída idêntica até o último dígito. Não há rota HTTP para isso de propósito: o
  registro tem dado pessoal, e a API não tem autenticação; a consulta roda no servidor.
- **Falta, e é do controlador, não do código:** o fluxo humano de revisão (quem revê, em
  que prazo, como a decisão revista volta ao titular) e a indicação do encarregado pelo
  tratamento, cuja identidade e contato a lei manda divulgar publicamente (art. 41, §1º).

---

## Continuidade

O que esta etapa deixa especificado para a operação em produção, na ordem em que um
depende do outro:

1. **Explicação individual** sobre o registro de decisão, que já guarda as entradas e o
   modelo exato de cada decisão.
2. **Mitigação de viés** escolhida entre as opções da seção 4, aplicada com as duas
   métricas medidas antes e depois, e passando pelo mesmo gate de promoção de qualquer
   candidato.
3. **Relatório de impacto à proteção de dados** (art. 5º, XVII; art. 38), que este
   documento já estrutura: tipos de dados, metodologia, riscos medidos e mitigação.
