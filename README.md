# Risco de Crédito — Sustentação e Confiabilidade

> Camada de sustentação para um modelo de risco de crédito em produção: bloqueia dado
> inválido na ingestão, treina um baseline auditável sobre um Dataset de Referência limpo e
> só publica um modelo novo quando ele passa por um gate de qualidade com pisos medidos.

---

## O problema

Uma fintech tem um modelo de *credit scoring* em produção e suspeita que ele vem
degradando em silêncio. Silêncio é a palavra: o modelo continua respondendo, continua
rápido, continua com boa acurácia — e ainda assim pode estar identificando cada vez menos
inadimplentes.

Duas causas se confundem quando não há instrumentação:

- **Dado quebrado chegando na entrada.** Um campo que passa a vir nulo, uma coluna com
  código de ausência disfarçado de número. O modelo não reclama: ele pontua.
- **A população mudou.** O dado continua válido, mas não é mais o dado sobre o qual o
  modelo aprendeu.

São problemas diferentes e exigem respostas diferentes. Esta etapa resolve a primeira:
estabelece o Dataset de Referência, o contrato que barra dado inválido antes de qualquer
coisa acontecer, o baseline e o gate de promoção. A segunda — detecção de *drift* — é a
etapa seguinte, e o projeto foi desenhado para que ela entre sem reescrever nada daqui.

A distinção que o código sustenta: **contrato responde "este dado é válido?"** e é falha
dura quando a resposta é não. **Drift responde "este dado é o mesmo de antes?"** — e um
lote com drift passa no contrato, porque dado deslocado continua sendo dado válido.

---

## Dados

### Fonte

> Credit Fusion and Will Cukierski. *Give Me Some Credit.* Kaggle, 2011.

O arquivo é obtido pela **redistribuição do OpenML (id 46929, licença Public)**, que serve
o mesmo conteúdo por URL pública com checksum. O acesso direto ao Kaggle exige credencial
de conta; um `git clone` limpo e um job de CI não têm essa credencial, e depender dela
tornaria o projeto irreproduzível para qualquer pessoa que não fosse o autor. A escolha é
de engenharia, não de conveniência: reprodutibilidade em clone limpo vale mais do que
usar o endereço canônico.

A integridade é verificada a cada execução contra o md5 fixado em `config.py`
(`6d013a631b97b13d5372f097697e25a1`, 6,9 MB, formato ARFF). Cache íntegro não é rebaixado.

### Do arquivo bruto ao Dataset de Referência

| | Linhas | Positivos |
|---|---:|---:|
| Arquivo bruto | 150.000 | 10.026 (6,68%) |
| **Dataset de Referência** | **117.917** | **8.186 (6,94%)** |

Retenção de 78,61%. Cada linha descartada é contabilizada por motivo — descarte silencioso
é perda de auditoria:

| Motivo | Linhas |
|---|---:|
| `duplicata` | 646 |
| `renda_nula` | 29.186 |
| `dependentes_nulo` | 0 |
| `idade_invalida` | 1 |
| `atraso_sentinela` | 144 |
| `razao_divida_implausivel` | 2.106 |
| **Total** | **32.083** |

As contagens são **cumulativas, não independentes**: cada filtro opera sobre o que sobrou
do anterior, então a soma é o total real de linhas perdidas, sem dupla contagem.

### Três coisas que o dado revelou

Estes três achados são a razão de o contrato ter a forma que tem. Sem eles, as regras
seriam palpite.

**1. Renda nula e `DebtRatio` absurdo são um defeito só, não dois.**
29.731 linhas do bruto (19,82%) vêm sem `MonthlyIncome`. Nelas, `DebtRatio` guarda a
**dívida bruta** em vez da razão: 90,04% ficam acima de 10, contra 1,75% entre as linhas
que têm renda. O p95 de `DebtRatio` no bruto é 2.449 e o máximo é 329.664; restrito às
linhas com renda declarada, o p95 cai para 1,13. Barrar a renda nula corrige os dois
campos de uma vez.

**2. 100% das 3.924 linhas sem `NumberOfDependents` também estão sem `MonthlyIncome`.**
A ausência é inteiramente co-localizada — não são dois defeitos independentes, é um único
evento de ingestão quebrado com vários sintomas. É por isso que o descarte
`dependentes_nulo` marca **0**: o filtro de renda já removeu essas linhas antes. Esse zero
é evidência do achado, não sinal de regra morta a remover.

**3. As três colunas de atraso carregam os sentinelas 96 e 98 em 269 linhas.**
São marcas de "não disponível" herdadas da coleta original, não contagens. Um cliente que
declara 98 inadimplências de 90 dias é um registro **do tipo certo e semanticamente
impossível** — exatamente a classe de defeito que um contrato de dados intercepta e um
schema de tipos deixa passar.

*(Todos os números acima medidos sobre `data/raw/GiveMeSomeCredit.arff` nesta versão do
repositório.)*

### Duas propriedades que custaram a ser obtidas

**A Referência passa no próprio contrato.** A limpeza e o contrato são uma política só,
expressa duas vezes — e as constantes (`DEBT_RATIO_MAXIMO`, `IDADE_MINIMA`,
`ATRASO_MAXIMO_PLAUSIVEL`) são **importadas**, não repetidas. Um teste
(`test_referencia_limpa_passa_no_proprio_contrato`) garante isso. Num rascunho anterior as
duas divergiam, o que teria matado o pipeline na própria etapa de validação.

**O tratamento de desbalanceamento é verificado, não suposto.** Ver a seção de modelo.

---

## Contrato de dados

O contrato é a fronteira de entrada. Roda **antes** do treino e antes de qualquer
inferência — um modelo treinado sobre dado reprovado já nasce comprometido.

A implementação fica atrás de um `Protocol` (`credito.contracts.base.Validator`): as regras
são declaradas uma única vez em `rules.py`, independentes de biblioteca, e o executor
(Pandera) fica trocável sem reescrever regra nem teste de comportamento.

### As seis regras e o defeito medido atrás de cada uma

| Regra | Coluna | Exigência | Defeito medido que a justifica |
|---|---|---|---|
| `renda_nao_nula` | `MonthlyIncome` | não nula, `>= 0` | 29.731 linhas (19,82%) sem renda; 90,04% delas com `DebtRatio > 10` |
| `idade_plausivel` | `age` | entre 18 e 110 | `age` no bruto vai de **0 a 109**; 1 linha abaixo de 18 |
| `sem_duplicatas` | lote inteiro | sem linhas idênticas | 609 duplicatas exatas de linha inteira; 646 considerando só as *features* |
| `atraso_plausivel` | 3 colunas de atraso | `<= 20` | 269 linhas com os sentinelas 96 e 98 |
| `dependentes_nao_nulo` | `NumberOfDependents` | não nula, `>= 0` | 3.924 linhas (2,62%), 100% delas também sem renda |
| `razao_divida_plausivel` | `DebtRatio` | entre 0 e 10 | rede de segurança: mesmo com renda válida, 1,75% ainda passam de 10 |

O teto de 10 em `DebtRatio` é uma ordem de grandeza acima do p95 medido entre as linhas com
renda (1,13): generoso o bastante para não reprovar caso atípico legítimo, apertado o
bastante para barrar o defeito conhecido. O teto de 110 em `age` barra erro de digitação
sem reprovar o cliente de 109 anos que existe no arquivo.

### O bloqueio, demonstrado

`make demo-contrato` injeta 5 ocorrências de cada defeito em uma amostra limpa de 500
linhas e mostra os dois desfechos lado a lado. Saída real:

```
=== Lote da Referência (limpo) ===
linhas: 500 | válido: True
ingestão liberada

=== Lote adulterado ===
linhas: 505 | válido: False
  - razao_divida_plausivel (DebtRatio): 5 linha(s)
  - renda_nao_nula (MonthlyIncome): 5 linha(s)
  - dependentes_nao_nulo (NumberOfDependents): 5 linha(s)
  - atraso_plausivel (NumberOfTimes90DaysLate): 5 linha(s)
  - idade_plausivel (age): 5 linha(s)
  - sem_duplicatas (*): 5 linha(s)

INGESTÃO BLOQUEADA: 30 de 505 linha(s) reprovadas — ...
```

As seis regras disparam, cada uma com a contagem exata, e a ingestão é bloqueada. O
relatório é agregado (`lazy=True`): quem está corrigindo um lote precisa ver **todos** os
problemas de uma vez, não o primeiro.

---

## Modelo baseline

### Por que AUC-PR e recall da classe positiva, e não acurácia

A Referência tem **6,94% de positivos**. Sobre o conjunto de teste, um modelo constante que
responde "não vai inadimplir" para todo mundo obtém:

| Métrica | Modelo constante |
|---|---:|
| Acurácia | **0,9306** |
| Recall da classe positiva | **0,0000** |

93% de acurácia e zero inadimplentes identificados — que é a única coisa que se quer do
modelo. Acurácia, nesse regime, é uma métrica que **premia o modelo inútil**.

As duas métricas que o projeto usa para decidir:

- **AUC-PR** mede a qualidade do ordenamento onde a classe rara importa. A linha de base de
  um ordenador aleatório é a própria taxa de positivos, **0,0694** — é contra isso que o
  número do campeão deve ser lido.
- **Recall da classe positiva** é a métrica de negócio. Aprovar um inadimplente custa o
  valor emprestado; recusar um bom pagador custa a margem. A assimetria é enorme, e AUC-PR
  sozinho não a enxerga.

`auc_roc` é reportado como figura secundária: ele infla com o desbalanceamento e por isso
não decide nada aqui.

### O tratamento de desbalanceamento, medido

Não é refinamento: é o que separa um modelo de uma constante. Comparação real, mesma
arquitetura e mesmos hiperparâmetros, mudando **apenas** o peso da classe, treinada na
partição de treino e avaliada na partição de teste da Referência:

| Regressão logística | Recall positivo | Acurácia | AUC-PR |
|---|---:|---:|---:|
| sem `class_weight` | **0,1436** | 0,9346 | 0,3590 |
| com `class_weight="balanced"` | **0,6213** | 0,8474 | 0,3648 |

O modelo sem tratamento tem **acurácia mais alta** (0,9346 contra 0,8474) e identifica
**um quarto** dos inadimplentes. Quem olhasse só para acurácia escolheria o pior dos dois.
Um teste de controle (`test_nao_colapsa_na_classe_majoritaria`) mantém essa propriedade
verificada na suíte, para os dois candidatos.

### Candidatos e campeão

Dois candidatos, não um: a comparação transforma a escolha do modelo em evidência em vez de
preferência, ao custo de uma chamada a mais no pipeline. O campeão é escolhido pelo **AUC-PR
na validação** — escolher no teste transformaria o teste em parte do treino e a métrica
publicada nasceria otimista.

Partições estratificadas pelo alvo, semente 42: treino 70.749 · validação 23.584 · teste
23.584 linhas (6,94% de positivos em cada).

**Validação** (onde o campeão é escolhido):

| Candidato | AUC-PR | AUC-ROC | Recall + | Precisão + | Acurácia |
|---|---:|---:|---:|---:|---:|
| `regressao_logistica` | 0,3469 | 0,8123 | 0,6237 | 0,2506 | 0,8444 |
| **`xgboost`** | **0,3612** | 0,8437 | 0,6970 | 0,2325 | 0,8192 |

**Campeão `xgboost`, no conjunto de teste** (métricas publicadas):

| Métrica | Valor |
|---|---:|
| AUC-PR | **0,3716** |
| AUC-ROC | 0,8421 |
| Recall da classe positiva | **0,6915** |
| Precisão da classe positiva | 0,2350 |
| Acurácia | 0,8223 |
| Taxa de positivos | 0,0694 |
| Limiar | 0,5 |
| n | 23.584 |

AUC-PR de 0,3716 contra uma linha de base aleatória de 0,0694: **5,4 vezes** melhor que o
acaso. O modelo recupera 69,15% dos inadimplentes ao custo de uma precisão de 23,50% — um
ponto de operação deliberadamente deslocado para o recall, pela assimetria de custo.

### Recorte por faixa etária

Idade é atributo protegido. Afirmar que o modelo não discrimina sem medir por grupo é
afirmação sem evidência. Medido no conjunto de teste:

| Faixa | n | Positivos reais | Taxa de recusa | Recall + | Prob. média |
|---|---:|---:|---:|---:|---:|
| 18-25 | 486 | 8,64% | 34,98% | 0,7857 | 0,381 |
| 26-40 | 5.399 | 10,58% | 31,91% | 0,7671 | 0,390 |
| 41-60 | 11.302 | 7,24% | 21,62% | 0,6822 | 0,304 |
| 61+ | 6.397 | 3,22% | 7,50% | 0,5000 | 0,166 |

A taxa de recusa cai monotonicamente com a idade (34,98% → 7,50%), e a taxa de
inadimplência real também (8,64% → 3,22%). Ou seja: a diferença de tratamento acompanha uma
diferença de risco observada, não é um viés puro. Mas a razão entre os extremos é de **4,7×**
na recusa contra **2,7×** no risco real — o modelo é mais severo com os jovens do que o
risco medido justifica. O recall de 0,5000 na faixa 61+ também é o mais baixo da tabela: o
modelo enxerga pior o inadimplente idoso. São números para discutir viés com dado em vez de
adjetivo; a análise formal de *disparate impact* é trabalho da etapa de governança.

---

## Gate de qualidade

O gate decide se um candidato recém-treinado substitui o modelo em produção. São quatro
critérios: dois pisos absolutos e, quando já existe incumbente, duas não regressões.

| # | Critério | Valor |
|---|---|---|
| 1 | AUC-PR do candidato `>=` piso absoluto | **0,3216** |
| 2 | Recall positivo do candidato `>=` piso absoluto | **0,6415** |
| 3 | AUC-PR `>` o do modelo em produção | — |
| 4 | Recall positivo `>=` o do modelo em produção | — |

### Como os pisos foram calibrados

Os dois pisos ficaram em `0.0` até existir modelo treinado, **de propósito**: um piso
inventado antes da medição é enfeite que sempre passa. Foram fixados a partir das métricas
medidas do campeão no conjunto de teste, menos uma margem de 0,05:

| Piso | Medição de origem | Margem | Valor |
|---|---:|---:|---:|
| `min_auc_pr` | AUC-PR do `xgboost` no teste: 0,3716 | −0,05 | **0,3216** |
| `min_recall_positivo` | Recall + do `xgboost` no teste: 0,6915 | −0,05 | **0,6415** |

A margem absorve a variação natural entre retreinos (reordenação de partição, versão de
biblioteca) sem abrir espaço para uma queda real de qualidade passar batido. Um teste
(`test_pisos_do_gate_estao_calibrados`) trava os valores, para que zerá-los seja uma decisão
visível e não efeito colateral de alguém "destravando o pipeline".

### Os dois desfechos

A distinção mais importante desta etapa:

- **`QualityGateError` — piso absoluto violado.** O modelo não é utilizável: é defeito, o
  run **falha** (saída 1) e alguém precisa investigar.
- **`ModelNotPromoted` — candidato válido que não supera o incumbente.** Desfecho
  **normal**, não defeito. Sobre um dataset estático o retreino reproduz o incumbente, então
  esta é a saída esperada de toda execução periódica depois da primeira. O run **conclui**
  (saída 0) e a orquestração converte isso em *skip*.

Se fossem a mesma exceção, o pipeline periódico ficaria vermelho da segunda execução em
diante e o alarme perderia o sentido. As duas classes são deliberadamente **independentes**
— nenhuma herda da outra, e um teste garante isso.

### A prova dos dois desfechos, rodando

Sequência real a partir de `models/` e `metrics/` vazios:

**Execução 1 — primeira promoção**

```
$ make train
INFO referência com 117917 linhas; descartes: {'duplicata': 646, 'renda_nula': 29186, ...}
INFO candidato regressao_logistica treinado sobre 70749 linhas
INFO auc_pr=0.3469 auc_roc=0.8123 recall+=0.6237
INFO candidato xgboost treinado sobre 70749 linhas
INFO auc_pr=0.3612 auc_roc=0.8437 recall+=0.6970
INFO campeão na validação: xgboost (auc_pr 0.3612)
INFO auc_pr=0.3716 auc_roc=0.8421 recall+=0.6915
INFO promovido: xgboost
>>> exit=0

$ md5sum models/model.joblib metrics/metrics.json
567d4533bf2f46c425d22e59adcd8aac  models/model.joblib
f591612310ed38d3f9d47c0ef9828bd9  metrics/metrics.json
```

**Execução 2 — piso absoluto violado, o run falha**

```
$ CREDITO_MIN_AUC_PR=0.99 poetry run python -m credito.pipeline.training
...
INFO auc_pr=0.3716 auc_roc=0.8421 recall+=0.6915
ERROR gate de qualidade reprovou o candidato: auc_pr 0.3716 abaixo do piso 0.9900
>>> exit=1

$ md5sum models/model.joblib metrics/metrics.json
567d4533bf2f46c425d22e59adcd8aac  models/model.joblib     <- idêntico
f591612310ed38d3f9d47c0ef9828bd9  metrics/metrics.json    <- idêntico
```

**Execução 3 — retreino idempotente, o run conclui sem promover**

```
$ poetry run python -m credito.pipeline.training
...
INFO auc_pr=0.3716 auc_roc=0.8421 recall+=0.6915
INFO nada a promover: auc_pr 0.3716 não supera o modelo em produção (0.3716)
>>> exit=0

$ md5sum models/model.joblib metrics/metrics.json
567d4533bf2f46c425d22e59adcd8aac  models/model.joblib     <- idêntico
f591612310ed38d3f9d47c0ef9828bd9  metrics/metrics.json    <- idêntico
```

O checksum do modelo publicado é **byte a byte idêntico** nas três verificações: nenhum
candidato recusado encostou no que está servindo. `metrics.json` também não é reescrito por
candidato recusado — ele descreve **quem está em produção**, e um arquivo que descrevesse um
candidato nunca publicado seria enganoso sem que ninguém percebesse lendo o JSON.

### O histórico

`metrics/training_history.jsonl` é *append-only*, uma linha por execução, nunca reescrita —
uma escrita interrompida derruba a última linha, não o arquivo inteiro. As três execuções
acima ficaram registradas com seus motivos:

```
{"carimbo": "...", "candidato": "xgboost", "promovido": true,  "motivos": []}
{"carimbo": "...", "candidato": "xgboost", "promovido": false, "motivos": ["auc_pr 0.3716 abaixo do piso 0.9900", "auc_pr 0.3716 não supera o modelo em produção (0.3716)"]}
{"carimbo": "...", "candidato": "xgboost", "promovido": false, "motivos": ["auc_pr 0.3716 não supera o modelo em produção (0.3716)"]}
```

A execução reprovada é exatamente a que alguém vai querer auditar depois — por isso o
registro acontece **antes** de qualquer exceção ser levantada, e não some por o run ter
terminado em erro. É o que permite responder "por que o gate vem reprovando?" lendo, em vez
de adivinhando.

---

## Como executar

Pré-requisitos: **Python 3.12** e **Poetry 2.x** (validado com Python 3.12.13 e Poetry
2.3.2). Nenhuma credencial é necessária.

```bash
git clone <url-do-repositorio>
cd <diretorio-do-repositorio>

make install        # instala dependências e os hooks de pre-commit
make data           # baixa o dataset público (~6,9 MB) e verifica o md5
make train          # treina os candidatos, avalia e promove o campeão
```

Ao final, `make train` deixa em disco:

- `models/model.joblib` — o pipeline completo (pré-processamento embutido) junto dos
  metadados de como ele foi obtido;
- `metrics/metrics.json` — as métricas do modelo publicado, incluindo o recorte etário e a
  contabilidade de descartes da Referência;
- `metrics/training_history.jsonl` — uma linha por execução, com o veredito e os motivos.

Demais alvos:

```bash
make help           # lista tudo
make lint           # ruff check + ruff format --check
make test           # suíte com relatório de cobertura
make demo-contrato  # mostra o contrato bloqueando um lote adulterado
```

Rodar `make train` uma segunda vez **não falha**: o retreino reproduz o incumbente, o gate
recusa a promoção e o processo termina com saída 0. Esse é o comportamento correto.

Toda configuração é resolvida por variável de ambiente com o prefixo `CREDITO_`
(ver `.env.example` e `src/credito/config.py`); nenhuma é obrigatória — os defaults são os
valores embutidos em `Settings`.

---

## Estrutura do projeto

```
src/credito/
├── config.py               Fonte única de caminhos, semente e pisos do gate
├── data/
│   ├── download.py         Download com cache e verificação de md5
│   ├── arff.py             Leitor do formato em que o OpenML publica
│   ├── prepare.py          Política de limpeza auditável + partições estratificadas
│   └── adulterate.py       Injeção controlada dos defeitos medidos, para exercitar o contrato
├── contracts/
│   ├── base.py             Protocol Validator, ValidationResult, ContratoViolado
│   ├── rules.py            As 6 regras, declaradas uma vez, com o motivo medido de cada
│   └── pandera_backend.py  Execução das regras com Pandera, atrás do Protocol
├── model/
│   ├── train.py            Os dois candidatos, com tratamento de desbalanceamento
│   └── evaluate.py         AUC-PR, recall positivo e o recorte por faixa etária
└── pipeline/
    ├── steps.py            Gate: os 4 critérios e as 2 exceções distintas
    └── training.py         Encadeamento ponta a ponta

scripts/demo_contrato.py    Demonstração do bloqueio de ingestão
docs/model_card.md          Uso pretendido, métricas, limitações e riscos
tests/                      Suíte espelhando a estrutura de src/
```

`data/`, `models/`, `metrics/` e `reports/` são ignorados pelo git: são artefatos
reproduzíveis por `make data && make train`, e versioná-los inflaria o repositório sem
acrescentar informação.

---

## Roadmap

### Limitações conhecidas desta etapa

Dívida declarada é maturidade; dívida escondida que o leitor descobre é defeito.

- **Os 19,82% de linhas sem renda são descartados, não recuperados.** É a decisão certa
  para construir uma Referência confiável, mas significa que a Referência **não representa**
  a população inteira: ela exclui sistematicamente um perfil (quem não declara renda), que
  provavelmente tem risco diferente. Em produção, esse perfil vai chegar — e hoje ele seria
  bloqueado pelo contrato em vez de pontuado. Tratar essa fatia exige uma política de
  imputação ou um modelo separado, e nenhum dos dois cabia nesta etapa.
- **O limiar de decisão é 0,5 fixo**, não otimizado para a assimetria de custo que o próprio
  README argumenta existir. Escolher o limiar por curva de custo é trabalho pendente.
- **Hiperparâmetros são padrões conservadores de baseline**, sem busca. Deliberado: a
  comparação entre os dois candidatos só é evidência se o *ensemble* entrar como candidato
  honesto, não como vencedor ajustado a dedo.
- **A precisão da classe positiva é 0,2350** — cerca de três em cada quatro recusas do
  modelo seriam bons pagadores. Aceitável dado o custo assimétrico, mas é um número que
  precisa entrar em qualquer discussão de adoção, não ser escondido atrás do recall.
- **O recorte etário mostra assimetria não explicada inteiramente pelo risco** (4,7× de
  variação na recusa contra 2,7× no risco real). Está medido, não está tratado.
- **Não há separação Referência × Produção ainda**: existe só a Referência. O lote de
  produção e sua simulação entram na etapa seguinte.
- **O gate compara contra o incumbente pelos metadados gravados no `model.joblib`**, não
  por uma reavaliação do incumbente sobre o teste corrente. Se a partição mudar, a
  comparação passa a ser entre números medidos sobre conjuntos diferentes.

### Próximas etapas

- **Etapa 2 — Drift.** Lote de Produção simulado, PSI e KS sobre as *features*, alarme
  separado do contrato (dado deslocado é válido, e precisa de resposta diferente). CI/CD
  completo entra aqui.
- **Etapa 3 — Observabilidade.** Rastreamento de experimentos, métricas de serviço em
  Prometheus e painéis, ligando a métrica de modelo à métrica de infraestrutura.
- **Etapa 4 — Governança.** Análise formal de viés a partir do recorte já medido,
  documentação de conformidade e o direito de revisão humana sobre decisão automatizada.
