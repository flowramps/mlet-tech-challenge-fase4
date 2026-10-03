# Changelog

Uma versão por etapa entregue. Cada etapa entrou por um pull request com CI verde, e cada
versão tem uma tag no commit de merge correspondente (`git checkout v0.2.0` devolve o
repositório exatamente como a Etapa 2 o deixou). Os números citados abaixo são os medidos
na própria etapa; os atuais estão no README.

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

[0.5.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/flowramps/mlet-tech-challenge-fase4/releases/tag/v0.1.0
