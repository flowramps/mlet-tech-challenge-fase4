# Changelog

Uma versão por etapa entregue. Cada etapa entrou por um pull request com CI verde, e cada
versão tem uma tag no commit de merge correspondente (`git checkout v0.2.0` devolve o
repositório exatamente como a Etapa 2 o deixou). Os números citados abaixo são os medidos
na própria etapa; os atuais estão no README.

## Não lançado

### Corrigido
- `/score` agora rejeita com HTTP 422 idade, renda, dependentes, razão de dívida e
  contadores de atraso fora do contrato antes de chamar o modelo. O schema Pydantic e o
  contrato Pandera consomem os mesmos limites de `credito.schema`.

## [1.1.1] — o timeout que a validação final achou

A validação final, num clone feito pela URL pública, reproduziu uma falha que já tinha
aparecido duas vezes e passado sem causa: o gerador de tráfego morria por timeout no meio
da rodada.

### Corrigido
- O gerador de tráfego conta um timeout como resultado, em vez de parar a rodada com
  traceback — era o que escondia a causa.

### Documentado
- A causa: a gravação do registro de decisão com `fsync` acopla a latência ao disco do
  host. Sob 3 GB de escrita concorrente, até 16,1 s (7,7 s sem o `fsync` — a escrita simples
  também trava). O `fsync` fica, pela durabilidade que o direito de revisão exige; o custo e
  o que fazer em produção estão no plano de monitoramento, na governança e no README.

## [1.1.0] — os logs de execução do pipeline, centralizados

A auditoria final contra o enunciado achou o critério de observabilidade parcial: ele pede
os logs de execução do pipeline — sucesso e falha na ingestão e na transformação, status
das tarefas, volume processado — e a estabilidade da infraestrutura. O MLflow só recebia o
resultado de um monitoramento bem-sucedido; um lote reprovado abortava sem rastro, e o
treino não ia para o MLflow.

### Adicionado
- `credito.tracking.execucao`: o run é aberto no início do pipeline. Status de cada tarefa
  (1/0, no mês do lote), volume processado e recusado por motivo, desfecho declarado; uma
  falha fecha o run `FAILED` com tipo e mensagem. "Nada a promover" termina `FINISHED`.
- O treino passa a ser registrado no MLflow (`credito-treino`), com as métricas de cada
  candidato e do campeão. Visto com o dado real: promovido, não promovido e um `FAILED` por
  piso violado.
- Métricas do processo da API (CPU, memória, reinício) e os painéis de estabilidade da
  infraestrutura no Grafana.
- O plano de monitoramento documenta cada métrica de saúde do pipeline e ganha o cenário
  de um run `FAILED` no playbook.
- Os relatórios HTML do Evidently anexados à release, com link no README.

### Corrigido
- Os prints do MLflow mostravam o painel lateral de assistente da ferramenta, com o nome de
  um produto de IA no seletor de modelo; recapturados sem ele.

## [1.0.0] — as dívidas declaradas, fechadas

As duas lacunas que a documentação das etapas declarava — e que impediam afirmar que o
alarme funciona e que o direito de revisão tem objeto.

### Adicionado
- **Registro de decisão de crédito** (`credito.governanca.decisoes`): cada `/score` devolve
  um `id_decisao` e grava a decisão antes de emiti-la — entradas, probabilidade, limiar e
  sha256 do modelo, sem identificador nenhum. Sem registro, sem decisão: falha de gravação
  vira 503. `make consultar-decisao` reconstrói e repontua (recusa real reconstruída
  idêntica); `make expurgar-decisoes` executa a retenção de 5 anos. No container, o único
  caminho gravável.
- **Calibração do alarme ao vivo** (`make calibrar-alarme`): nula da taxa de aprovação por
  tamanho de janela, limiar com 1% de falso alarme e poder por mês de drift. A nula do PSI
  do score, antes de um script descartado, sai do mesmo comando.
- **Regras de alerta no Prometheus**, provisionadas por arquivo e testadas com `promtool
  test rules` no CI: taxa de aprovação abaixo do limiar calibrado com a janela cheia, e 5xx
  sustentado (que inclui o 503 de "sem registro, sem decisão").
- `make traffic LOTE=mes_06`: tráfego com linhas reais de um mês com drift. Visto ao vivo:
  taxa de 79,9% com o mês 0, 74,8% com o mês 6, alerta disparado.

### Alterado
- Janela da taxa de aprovação de 500 para 2.000 decisões. Medido: com 500, o ruído (1,8
  p.p.) era maior que a queda de um mês inteiro (0,79 p.p.) e o alarme pegava só 68,1% das
  janelas no mês 6; com 2.000, pega 79,5% no mês 3 e 94,1% no mês 4.
- O painel de taxa de aprovação desenha o limiar calibrado.

### Corrigido
- O plano de monitoramento numerava os painéis numa ordem diferente da do dashboard, e o
  playbook mandava olhar o painel errado.
- Documentos afirmavam que o vetor de features nunca era gravado e que o serviço não
  guardava dado do titular — falso desde o registro de decisão; reescritos.

## [0.5.0] — maturidade e reprodutibilidade

Revisão do projeto inteiro depois das quatro etapas, para quem chega ao repositório sem
contexto conseguir entendê-lo e reproduzi-lo.

### Adicionado
- `make reproduzir`: do dataset a todo número publicado no README, num comando.
- Workflow de segurança (`security.yml`): Trivy na imagem em push, PR e semanalmente.
- Piso de cobertura de 95% no relatório (a suíte cobre 98%).
- Teste que segue estaticamente o fecho de imports da API e falha se ela alcançar uma
  dependência que a imagem não instala.
- Este changelog e as tags por etapa.

### Alterado
- Imagem de inferência de 3,4 GB para 1,25 GB: o `chown -R` duplicava o virtualenv numa
  camada nova, e MLflow, Evidently e Pandera iam para a imagem sem a API usar nenhum —
  agora num grupo de dependências próprio. Código e modelo ficam só-leitura para o processo.
- Imagem sem pip e com as atualizações de segurança do sistema: de 5 vulnerabilidades
  HIGH corrigíveis para zero.
- A versão é uma só: o OpenAPI anunciava 1.0.0 enquanto o pacote era 0.1.0; a API passa a
  ler do metadado do pacote.

### Corrigido
- Referências a artefatos internos de planejamento que tinham vazado para código, testes,
  dashboard e documentação — inclusive três citações a um arquivo inexistente.
- Números divergentes entre documentos (razões de latência, linhas citadas de código, "60"
  onde o medido é 62) e o tempo do `make monitor` remedido depois da integração com o MLflow.

### Removido
- Grupo de dependências `notebooks` (jupyter, matplotlib), sem notebook nem uso no código.

## [0.4.0] — 2026-10-03 — governança e proteção de dados

### Adicionado
- `docs/governanca.md`: base legal (LGPD, art. 7º, X e V — proteção do crédito, não
  consentimento), dado tratado, retenção com os tetos do CDC e do Cadastro Positivo,
  equidade, causalidade e direitos do titular. Toda citação conferida no texto oficial.
- Duas métricas formais de equidade, publicadas em `metrics.json`: a regra dos 4/5 aponta
  impacto adverso nos jovens (18-25: 0,7030); a igualdade de oportunidade aponta os idosos
  (61+: recall 0,5000).
- `make auditar-privacidade`: 53,30% da Referência é única em idade + renda + dependentes;
  a generalização reduz as linhas em grupo menor que 5 de 75,62% para 0,01%.
- A API descarta identificador enviado a mais, por decisão declarada e travada por teste.

### Corrigido
- A Referência era descrita como "anônima"; é pseudonimizada.
- A revisão humana era apresentada como exigência do art. 20; a redação vigente não a
  exige — é decisão do projeto, mais rigorosa que a lei.

## [0.3.0] — 2026-10-02 — observabilidade

### Adicionado
- Sinais sem rótulo (`credito.monitoring.proxies`) e a validação de qual antecipa a
  degradação real: os três empatam em ordem (Spearman ±1, p exato 0,002778) e
  `taxa_de_aprovacao` vence por magnitude.
- API de scoring instrumentada (`/health`, `/score`, `/metrics`), com cardinalidade de
  rótulo controlada pela rota declarada.
- Pilha Prometheus + Grafana com datasource e dashboard provisionados por arquivo.
- Cada `make monitor` registrado como um run do MLflow.
- `docs/monitoring_plan.md`: limiares medidos e playbook por cenário.

### Corrigido
- Buckets de latência de inferência calibrados com um modelo-dublê saturavam em `+Inf`
  contra o campeão real.
- O MLflow abria uma conexão de telemetria a cada execução; desligada.

## [0.2.0] — 2026-09-27 — simulação e detecção de drift

### Adicionado
- Seis lotes mensais simulados: três variáveis de *data drift* e um *concept drift*.
- PSI e KS próprios, com binning roteado por colapso medido.
- Nula empírica do PSI: a convenção 0,10 é 62x mais folgada que o p95 do acaso neste dado.
- Correção de Benjamini-Hochberg por lote e gate de severidade.
- Atribuição causal por intervenção: o concept drift responde por 48,0% da degradação; as
  duas variáveis de maior PSI, por 3,1%.
- Relatório HTML do Evidently — que declara "sem drift" no mês 6, pelo mecanismo de binning
  documentado no README.

## [0.1.0] — 2026-09-26 — contratos de dados, baseline e gate

### Adicionado
- Dataset de Referência com política de limpeza auditável (117.917 linhas, cada descarte
  contado por motivo).
- Contrato de dados com seis regras, bloqueando lote inválido antes do treino.
- Baseline XGBoost com tratamento de desbalanceamento: AUC-PR 0,3716, recall 0,6915.
- Gate de promoção com pisos medidos e duas exceções distintas — falha contra "nada a fazer".
- Recorte de métricas por faixa etária.

[1.1.1]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/releases/tag/v0.1.0
