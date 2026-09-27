"""PSI e KS medidos à mão — o número que alguém pode conferir, não o que uma biblioteca
devolve pronto.

O Evidently já calcula estas duas coisas, mas entrega um relatório HTML, não uma função
que a Etapa 3 possa chamar e um humano possa auditar linha a linha. E o preset padrão do
Evidently usa distância de Wasserstein, que o enunciado do projeto não menciona — usar o
que o enunciado pede (PSI e KS) exige escrever as duas contas.

**A descoberta que molda este módulo:** binning por quantil com um número fixo de bins
(o padrão de qualquer implementação ingênua, inclusive a deste projeto antes desta
medição) não falha em variável discreta e zero-inflada — ela mente silenciosamente.
Medido nas dez features da Referência real (117.917 linhas, `ref[col].nunique()`):

    RevolvingUtilizationOfUnsecuredLines   distintos=101384
    age                                    distintos=    82
    NumberOfTime30-59DaysPastDueNotWorse   distintos=    14
    DebtRatio                              distintos=107998
    MonthlyIncome                          distintos= 13569
    NumberOfOpenCreditLinesAndLoans        distintos=    58
    NumberOfTimes90DaysLate                distintos=    17
    NumberRealEstateLoansOrLines            distintos=    28
    NumberOfTime60-89DaysPastDueNotWorse    distintos=    10
    NumberOfDependents                      distintos=    13

Cinco features (10 a 28 valores distintos — três delas 83-95% concentradas num único
valor: as duas colunas de atraso mais graves e `NumberOfTime30-59DaysPastDueNotWorse`)
colapsam os 10 cortes de quantil pedidos a 1-4 cortes efetivos (`np.unique` depois de
`np.quantile`) — medido rodando `bins_por_quantil` sobre cada uma. As outras cinco (58 e
acima) preservam os 10 cortes pedidos sem colapso.

Cardinalidade separa essas dez features corretamente, mas é uma *proxy*, não a causa: o
mecanismo real é concentração de massa, não contagem de valores distintos. A prova de que
a proxy quebra já mora no próprio projeto: `DebtRatio` tem 107.998 valores distintos — bem
acima de qualquer limiar de cardinalidade razoável — e o simulador de Produção da Etapa 2
(`credito.data.simulate.aplicar_drift_de_divida`) o satura (clip) em 10,0, criando uma
massa pontual que a Referência não tem. Alta cardinalidade, massa concentrada: é
exatamente o par que faz uma cardinalidade única errar, cardinalidade alta demais para
qualquer limiar de "poucos valores" mas concentrada o bastante para colapsar cortes de
quantil do mesmo jeito que a variável de atraso colapsa.

Por isso `psi()` não decide a estratégia de binning pela cardinalidade da Referência — ela
mede o colapso diretamente: calcula `bins_por_quantil(referencia, bins=bins)` e verifica
se o número de bins que sobrou depois da deduplicação (`np.unique`) é menor que o número
pedido. Um cardinalidade recebe algum sinal, mas o próprio ato de pedir `bins` cortes e
não recebê-los de volta *é* a medição de que massa demais se concentra em menos pontos do
que os quantis pedidos — não uma extrapolação da cardinalidade para outras variáveis que
ninguém mediu.

Para `NumberOfTime30-59DaysPastDueNotWorse` especificamente: entre o mês 0 e o mês 6 do
simulador de Produção da Etapa 2 (`credito.data.simulate.simular_producao`, seed=42,
sobre a Referência real), seus zeros caem de 83,1% para 70,6% e a média mais que dobra —
deslocamento real e grande. Medido: PSI por quantil ≈ 0,090 (abaixo de 0,10, "estável"
pela convenção do setor) contra PSI por valor (um bin por valor distinto) ≈ 0,141 (banda
de atenção, 0,10-0,25). O binning por quantil não crasha — ele esconde o drift.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats

# PSI não tem log(0): uma proporção zero em qualquer bin faz a fórmula divergir
# (`log(0)` é `-inf`). A suavização substitui zero por um valor pequeno o bastante para
# não distorcer proporções reais (a menor proporção não nula de qualquer bin nas
# amostras deste projeto é da ordem de 1e-5, muito acima de 1e-6) mas grande o bastante
# para o `log` não estourar em ponto flutuante.
EPSILON_PADRAO = 1e-6


class ResultadoKS(NamedTuple):
    """Estatística e p-valor do teste de Kolmogorov-Smirnov, nomeados separadamente.

    PSI e KS apontam em sentidos opostos: no PSI, valor **maior** é mais drift; no
    p-valor do KS, valor **menor** é mais drift. Um chamador que recebesse uma tupla
    comum `(a, b)` teria que lembrar qual posição é qual toda vez — e errar essa ordem
    não gera exceção, gera um alarme de drift invertido que passa despercebido até
    alguém desconfiar do painel. Nomear os campos move essa decisão do comentário (que
    ninguém é obrigado a ler) para o tipo (que o autocomplete mostra).
    """

    estatistica: float
    p_valor: float


def bins_por_quantil(referencia: np.ndarray, *, bins: int) -> np.ndarray:
    """Cortes de quantil calculados sobre ``referencia``, para reaplicar à Produção.

    Calcular os cortes na própria amostra que se está medindo (em vez de na Referência)
    vazaria a resposta: os cortes se ajustariam à atual e o PSI ficaria artificialmente
    pequeno para qualquer deslocamento, porque bins deslizantes absorvem o próprio
    deslocamento que deveriam medir.

    Quando a massa da Referência se concentra (ex.: 90% de zeros, ou os 10,0 exatos que
    o clip de ``DebtRatio`` produz), vários quantis coincidem e ``np.quantile`` devolve
    cortes repetidos — que fariam ``np.histogram`` recusar o array por não ser
    estritamente crescente. ``np.unique`` deduplica; ``psi`` usa o tamanho do resultado
    para decidir se essa concentração é forte o bastante para trocar de estratégia (ver
    ``psi``). Os dois extremos são abertos (±inf) para que um valor de Produção fora do
    intervalo observado na Referência caia no bin extremo em vez de ser descartado do
    histograma sem contagem — o que inflaria falsamente o PSI ao fazer a proporção da
    atual não somar 1.
    """
    referencia = np.asarray(referencia, dtype=float)
    cortes = np.unique(np.quantile(referencia, np.linspace(0.0, 1.0, bins + 1)))
    if len(cortes) < 2:
        # Referência sem variância (um só valor distinto): não há como definir mais de
        # um bin. Ver a limitação estrutural documentada em `psi`.
        return np.array([-np.inf, np.inf])
    cortes[0] = -np.inf
    cortes[-1] = np.inf
    return cortes


def bins_por_valor(referencia: np.ndarray) -> np.ndarray:
    """Um bin por valor distinto da Referência — a alternativa ao binning por quantil
    quando os cortes de quantil colapsam (ver ``psi``).

    Os cortes ficam nos pontos médios entre valores distintos consecutivos, o que faz
    cada valor observado na Referência cair no seu próprio bin de largura própria. Os
    extremos são abertos pela mesma razão de ``bins_por_quantil``: um valor de Produção
    maior ou menor que qualquer coisa vista na Referência (ex.: um novo patamar de
    atraso que a injeção de drift cria) precisa contar em algum bin, não desaparecer do
    histograma.

    Devolve um corte por valor distinto **da Referência**, não do parâmetro ``bins`` de
    ``psi`` — ver a nota sobre esse parâmetro no docstring de ``psi``.
    """
    valores = np.sort(np.unique(np.asarray(referencia, dtype=float)))
    if len(valores) < 2:
        return np.array([-np.inf, np.inf])
    pontos_medios = (valores[:-1] + valores[1:]) / 2
    return np.concatenate(([-np.inf], pontos_medios, [np.inf]))


def _proporcoes(amostra: np.ndarray, cortes: np.ndarray, epsilon: float) -> np.ndarray:
    contagem = np.histogram(amostra, bins=cortes)[0]
    proporcao = contagem / len(amostra)
    return np.clip(proporcao, epsilon, None)


def _psi_a_partir_dos_cortes(
    referencia: np.ndarray,
    atual: np.ndarray,
    cortes: np.ndarray,
    *,
    epsilon: float = EPSILON_PADRAO,
) -> float:
    p_ref = _proporcoes(np.asarray(referencia, dtype=float), cortes, epsilon)
    p_atual = _proporcoes(np.asarray(atual, dtype=float), cortes, epsilon)
    return float(np.sum((p_atual - p_ref) * np.log(p_atual / p_ref)))


def psi(
    referencia: np.ndarray, atual: np.ndarray, *, bins: int = 10, epsilon: float = EPSILON_PADRAO
) -> float:
    """Population Stability Index entre duas amostras da mesma variável.

    Convenção de risco de crédito: < 0,10 estável · 0,10-0,25 mudança moderada ·
    >= 0,25 mudança material. **Valor maior significa mais drift** — o oposto do
    p-valor do KS, que é a confusão mais fácil de cometer com estes dois juntos (ver
    ``ResultadoKS``).

    A estratégia de binning é escolhida medindo se o binning por quantil colapsa, não
    pela cardinalidade da Referência — cardinalidade é uma *proxy* que a variável de
    atraso e a de dívida deste projeto já provam que engana: a segunda tem mais de cem
    mil valores distintos e ainda assim colapsa quando o clip de contrato concentra
    massa num único ponto (ver o docstring do módulo). Colapso é definido de forma
    estrita: se ``bins_por_quantil(referencia, bins=bins)`` devolve **menos** cortes do
    que os ``bins`` pedidos — nem um a menos é tolerado —, ao menos um corte de quantil
    coincidiu com o vizinho, o que só acontece quando massa concentrada demais para o
    número de bins pedido. Uma regra por fração (só trocar se metade dos cortes
    colapsar, por exemplo) toleraria perder resolução justamente na região concentrada
    enquanto ainda chama isso de "quantil íntegro" — a regra estrita não tem essa lacuna:
    qualquer colapso, por menor que seja, já é evidência medida de que os bins pedidos
    não vieram, e a estratégia por valor nunca é pior (não depende de escala, só do que a
    Referência realmente contém).

    ``bins`` é o número de cortes de quantil pedidos; quando a variável colapsa e a
    estratégia muda para ``bins_por_valor``, esse parâmetro deixa de valer — o número de
    bins passa a ser o número de valores distintos da Referência, o que quer que seja,
    não o que ``bins`` pediu.

    PSI não enxerga um deslocamento de valor constante: quando a Referência não tem
    variância (um só valor distinto), tanto ``bins_por_quantil`` quanto
    ``bins_por_valor`` colapsam para um único bin, e a proporção de qualquer amostra
    nesse bin é sempre 1,0 — o PSI sai sempre 0,0 mesmo que a atual seja uma constante
    totalmente diferente. Não é acidente: é o limite matemático de uma divergência por
    proporção quando só existe uma categoria possível.
    """
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)
    cortes_quantil = bins_por_quantil(referencia, bins=bins)
    colapsou = (len(cortes_quantil) - 1) < bins
    cortes = bins_por_valor(referencia) if colapsou else cortes_quantil
    return _psi_a_partir_dos_cortes(referencia, atual, cortes, epsilon=epsilon)


def ks(referencia: np.ndarray, atual: np.ndarray) -> ResultadoKS:
    """Teste de Kolmogorov-Smirnov de duas amostras: devolve estatística e p-valor
    **nomeados**, nunca uma tupla posicional que o chamador precisa lembrar de decifrar.

    Convenção: p-valor < 0,05 rejeita a hipótese de que as duas amostras vêm da mesma
    distribuição — **valor menor significa mais drift**, o oposto do PSI.
    """
    resultado = stats.ks_2samp(np.asarray(referencia, dtype=float), np.asarray(atual, dtype=float))
    return ResultadoKS(estatistica=float(resultado.statistic), p_valor=float(resultado.pvalue))
