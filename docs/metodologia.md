# Metodologia e limites de inferência

## 1. Questão de pesquisa

Este estudo investiga se uma camada de sustentação de *machine learning* consegue:

1. impedir que dados inválidos alcancem o treinamento ou a inferência;
2. detectar deslocamentos graduais em dados válidos;
3. medir a degradação de um modelo mantido fixo;
4. produzir sinais operacionais antes da disponibilidade do rótulo; e
5. conservar evidências suficientes para auditoria e governança.

O objeto empírico é um **experimento reprodutível de engenharia**, construído sobre o
dataset histórico *Give Me Some Credit*. Não é uma validação clínica, regulatória ou de
negócio de um sistema de crédito brasileiro em produção.

## 2. Hipóteses operacionais

- **H1 — qualidade:** um contrato executável bloqueia violações estruturais e semânticas
  antes do consumo pelo modelo.
- **H2 — deslocamento:** PSI e KS identificam as distribuições de entrada alteradas pelo
  processo gerador sintético.
- **H3 — degradação:** mantendo o campeão fixo, o aumento programado da intensidade do
  cenário reduz seu poder discriminativo.
- **H4 — observabilidade:** ao menos um sinal disponível sem rótulo acompanha a ordem da
  degradação nos lotes simulados.

H1 e H2 são verificadas por testes de software e pelos lotes conhecidos do simulador. H3 e
H4 são avaliadas apenas dentro da trajetória sintética especificada neste repositório; não
constituem estimativas populacionais para uma carteira externa.

## 3. Desenho do estudo

A Referência é dividida aleatória e estratificadamente em treino, validação e teste. A
validação escolhe o candidato; o teste mede o campeão. A mesma partição de teste é então
usada como unidade de partida dos lotes mensais, sobre os quais o processo gerador aplica,
com intensidade crescente:

- deslocamento de renda;
- deslocamento de endividamento;
- mudança no perfil de atrasos; e
- mudança sintética em `P(y|X)` (*concept drift*).

O mês zero é o contrafactual interno sem transformação. Nos cenários isolados, uma
transformação é ligada por vez e as demais permanecem em intensidade zero. A diferença de
AUC-ROC contra o mês zero é o **efeito do fator dentro do simulador**. A diferença entre o
efeito conjunto e a soma dos efeitos isolados é registrada como interação.

Esse desenho suporta uma análise fatorial de sensibilidade do processo gerador conhecido.
Ele não identifica efeitos causais sobre pessoas, decisões econômicas ou uma população
real. Por compatibilidade, o módulo `credito.drift.causal` e a chave histórica
`atribuicao_causal_por_lote` conservam os nomes existentes; neste texto, sua interpretação
é sempre restrita ao simulador.

## 4. Estimandos e critérios de decisão

| Pergunta | Medida | Interpretação permitida |
|---|---|---|
| A distribuição mudou? | PSI e KS por feature | Evidência de deslocamento em `P(X)` |
| O campeão piorou? | queda de AUC-ROC, AUC-PR, recall e *lift* acima da prevalência | Desempenho na trajetória sintética |
| Qual fator mais moveu a métrica? | efeito isolado e termo de interação | Sensibilidade ao mecanismo do simulador |
| Há sinal antes do rótulo? | Spearman e poder do alarme | Associação preditiva no cenário simulado |
| O modelo é justo? | razão 4/5 e diferença de oportunidade por idade | Triagem exploratória de disparidade, não prova jurídica de discriminação |

O PSI é classificado pelos cortes convencionais de 0,10 e 0,25. A distribuição nula
empírica é publicada como diagnóstico, mas ainda não governa o gate. O KS recebe correção
de Benjamini–Hochberg dentro de cada lote. Essa correção não controla, por si, a taxa de
falso alarme acumulada em uma sequência indefinida de lotes.

## 5. Ameaças à validade

### Validade interna

- A intensidade máxima do *concept drift* foi calibrada contra o mesmo campeão e a mesma
  partição usados na demonstração para preservar degradação monotônica.
- Os meses reutilizam as mesmas linhas transformadas; portanto, as amostras são pareadas,
  enquanto o KS de duas amostras adota a formulação usual de amostras independentes.
- O teste é reutilizado para métricas finais, simulação, seleção do proxy, calibração do
  alarme e auditoria de equidade. Os resultados posteriores são exploratórios e podem
  refletir adaptação ao conjunto.

### Validade de construto

- PSI e KS medem deslocamento de distribuição, não perda de utilidade do modelo.
- A taxa de aprovação depende do modelo e do limiar de decisão, além da população.
- A regra dos 4/5 e a igualdade de oportunidade representam noções distintas de equidade;
  nenhuma delas, isoladamente, estabelece licitude ou ausência de discriminação.
- O rótulo é disponibilizado imediatamente pelo simulador, enquanto o desfecho de crédito
  real possui atraso de maturação.

### Validade externa

- O dataset é norte-americano, de 2011, e não representa automaticamente outra época,
  instituição, produto ou jurisdição.
- Registros sem renda declarada são excluídos, limitando a população-alvo do modelo.
- Não há coorte de produção real nem validação temporal *out-of-time*.

### Validade de conclusão

- Há uma única trajetória principal, com seis pontos mensais e uma semente base.
- As estimativas principais não possuem intervalos de confiança por reamostragem.
- A correlação perfeita de Spearman em seis pontos monotônicos demonstra concordância de
  ordem nessa trajetória, não robustez a outros mecanismos de drift.
- O falso alarme de 1% é definido por janela reamostrada sob independência; não equivale à
  taxa de incidentes de uma série temporal com janelas sobrepostas.

## 6. Hierarquia das conclusões

As conclusões devem ser lidas nesta ordem:

1. **Demonstrado por teste:** o contrato, os pipelines, os relatórios e os alertas executam
   conforme a especificação do repositório.
2. **Demonstrado no experimento:** o campeão degrada e os sinais se comportam como medido
   sob o processo gerador sintético definido.
3. **Hipótese para validação externa:** o mesmo monitoramento detectaria degradação com
   antecedência em uma carteira real.
4. **Não demonstrado:** impacto causal no mundo real, adequação do limiar econômico,
   ausência de discriminação e conformidade regulatória de uma implantação concreta.

## 7. Extensões confirmatórias recomendadas

Para transformar a prova de conceito em evidência confirmatória:

1. reservar um segundo *holdout* exclusivamente para validar monitoramento e alarmes;
2. executar múltiplas sementes e mecanismos de drift não usados no desenvolvimento;
3. publicar intervalos de confiança e sensibilidade dos limiares;
4. substituir ou complementar o KS por inferência que preserve o pareamento;
5. avaliar erro sequencial, sazonalidade e autocorrelação;
6. realizar validação temporal e externa sobre uma carteira compatível com o uso pretendido;
7. escolher o limiar de decisão por custo, calibração e restrições de equidade.

## 8. Referências metodológicas

- Benjamini, Y.; Hochberg, Y. *Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society, 1995.
- Gama, J. et al. *A Survey on Concept Drift Adaptation*. ACM Computing Surveys, 2014.
- Rabanser, S.; Günnemann, S.; Lipton, Z. *Failing Loudly: An Empirical Study of Methods
  for Detecting Dataset Shift*. NeurIPS, 2019.
- Sculley, D. et al. *Hidden Technical Debt in Machine Learning Systems*. NeurIPS, 2015.
- NIST. *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, 2023.
