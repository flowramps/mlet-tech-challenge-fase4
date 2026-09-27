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
acima) preservam os 10 cortes pedidos sem colapso. `LIMIAR_CARDINALIDADE_DISCRETA` cai
no meio dessa lacuna medida, não num número redondo escolhido por aparência.

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

# Ver a medição completa no docstring do módulo: com bins=10, cardinalidade até 28
# colapsa os cortes de quantil pedidos; cardinalidade a partir de 58 preserva todos os
# 10. O limiar cai no meio dessa lacuna medida entre 28 e 58 — não é o "10" ou "20"
# redondos que pareceriam naturais escolhidos sem medir.
LIMIAR_CARDINALIDADE_DISCRETA = 50

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

    Quando a massa da Referência se concentra (ex.: 90% de zeros), vários quantis
    coincidem e ``np.quantile`` devolve cortes repetidos — que fariam ``np.histogram``
    recusar o array por não ser estritamente crescente. ``np.unique`` deduplica.
    Os dois extremos são abertos (±inf) para que um valor de Produção fora do intervalo
    observado na Referência (a massa exata em ``DebtRatio == 10,0`` que o clip do
    simulador cria, por exemplo) caia no bin extremo em vez de ser descartado do
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
    para variáveis de baixa cardinalidade.

    Os cortes ficam nos pontos médios entre valores distintos consecutivos, o que faz
    cada valor observado na Referência cair no seu próprio bin de largura própria. Os
    extremos são abertos pela mesma razão de ``bins_por_quantil``: um valor de Produção
    maior ou menor que qualquer coisa vista na Referência (ex.: um novo patamar de
    atraso que a injeção de drift cria) precisa contar em algum bin, não desaparecer do
    histograma.
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

    A estratégia de binning é escolhida pela cardinalidade da Referência, não é única
    para todas as variáveis: cardinalidade até ``LIMIAR_CARDINALIDADE_DISCRETA`` usa um
    bin por valor distinto (``bins_por_valor``); acima disso usa cortes de quantil
    (``bins_por_quantil``). A razão é medida, não estética — ver o docstring do módulo:
    sob quantil, a variável zero-inflada de baixa cardinalidade deste projeto mede PSI
    abaixo do limiar de "estável" para um deslocamento que, medido por valor, está na
    banda de atenção.
    """
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)
    cardinalidade = len(np.unique(referencia))
    cortes = (
        bins_por_valor(referencia)
        if cardinalidade <= LIMIAR_CARDINALIDADE_DISCRETA
        else bins_por_quantil(referencia, bins=bins)
    )
    return _psi_a_partir_dos_cortes(referencia, atual, cortes, epsilon=epsilon)


def ks(referencia: np.ndarray, atual: np.ndarray) -> ResultadoKS:
    """Teste de Kolmogorov-Smirnov de duas amostras: devolve estatística e p-valor
    **nomeados**, nunca uma tupla posicional que o chamador precisa lembrar de decifrar.

    Convenção: p-valor < 0,05 rejeita a hipótese de que as duas amostras vêm da mesma
    distribuição — **valor menor significa mais drift**, o oposto do PSI.
    """
    resultado = stats.ks_2samp(np.asarray(referencia, dtype=float), np.asarray(atual, dtype=float))
    return ResultadoKS(estatistica=float(resultado.statistic), p_valor=float(resultado.pvalue))
