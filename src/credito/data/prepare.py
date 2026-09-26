"""Construção do Dataset de Referência a partir do arquivo bruto.

A Referência é o dado que o modelo aprende e o retrato contra o qual todo lote futuro é
comparado. Por isso a limpeza aplica exatamente as mesmas regras que o contrato de dados
vai exigir na ingestão: um dado que seria bloqueado na entrada não pode ter ensinado o
modelo. Cada linha descartada é contabilizada por motivo — descarte silencioso é perda de
auditoria.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ALVO = "inadimplente"
_ALVO_ORIGINAL = "FinancialDistressNextTwoYears"

FEATURES: tuple[str, ...] = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)

COLUNAS_DE_ATRASO: tuple[str, ...] = (
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
)

IDADE_MINIMA = 18
IDADE_MAXIMA = 110

# 96 e 98 são códigos de ausência herdados da coleta original, não contagens de atraso.
# Deixá-los passar ensinaria o modelo que existe um cliente com 98 inadimplências de 90
# dias, e qualquer estatística de distribuição sairia distorcida por 269 registros.
ATRASO_MAXIMO_PLAUSIVEL = 20

# A razão dívida/renda plausível: entre os registros com renda declarada, o p95 medido é
# 1,1. O teto de 10 é uma ordem de grandeza acima disso — generoso o bastante para não
# reprovar caso atípico legítimo, apertado o bastante para barrar o defeito conhecido.
# Definida aqui, não em `contracts/rules.py`, porque a limpeza da Referência e o contrato
# de ingestão precisam da mesma constante — e só há uma direção de import possível entre
# os dois módulos sem criar um ciclo (rules.py já importa daqui).
DEBT_RATIO_MAXIMO = 10.0


def limpar(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica a política de qualidade e devolve os dados limpos e a contagem por motivo."""
    motivos: dict[str, int] = {}
    trabalho = frame.copy()

    if _ALVO_ORIGINAL in trabalho.columns:
        trabalho[ALVO] = (trabalho[_ALVO_ORIGINAL] == "Yes").astype(int)
        trabalho = trabalho.drop(columns=[_ALVO_ORIGINAL])

    # O dedup olha só para FEATURES, não para a linha inteira: o contrato de ingestão
    # nunca vê o alvo, então duas linhas idênticas em FEATURES mas com alvo diferente já
    # passariam pelo contrato como duplicata mesmo que a limpeza as tratasse como
    # observações distintas. E duas aplicações idênticas com desfechos contraditórios não
    # são duas observações — são um conflito de rótulo, que este dedup também resolve
    # (mantendo a primeira ocorrência, na mesma convenção do `Check` do contrato).
    antes = len(trabalho)
    trabalho = trabalho.drop_duplicates(subset=list(FEATURES))
    motivos["duplicata"] = antes - len(trabalho)

    def _descartar(mascara: pd.Series, rotulo: str) -> None:
        nonlocal trabalho
        motivos[rotulo] = int(mascara.sum())
        trabalho = trabalho.loc[~mascara]

    # As contagens abaixo são cumulativas, não independentes: cada `_descartar` opera
    # sobre o que sobrou do descarte anterior, então a soma dos motivos é o total real
    # de linhas perdidas, sem dupla contagem. Isso também explica por que
    # "dependentes_nulo" mede 0 no dataset real: no arquivo bruto, 100% das linhas com
    # `NumberOfDependents` nulo também têm `MonthlyIncome` nulo — a regra de renda nula
    # já as removeu antes desta rodar. Renda nula, dependentes nulo e `DebtRatio`
    # absurdo não são três defeitos, são um único evento de ingestão quebrado com três
    # sintomas. Um "0" aqui é evidência desse fato, não sinal de regra morta a remover.
    _descartar(trabalho["MonthlyIncome"].isna(), "renda_nula")
    _descartar(trabalho["NumberOfDependents"].isna(), "dependentes_nulo")
    _descartar(
        ~trabalho["age"].between(IDADE_MINIMA, IDADE_MAXIMA),
        "idade_invalida",
    )

    sentinela = pd.Series(False, index=trabalho.index)
    for coluna in COLUNAS_DE_ATRASO:
        sentinela |= trabalho[coluna] > ATRASO_MAXIMO_PLAUSIVEL
    _descartar(sentinela, "atraso_sentinela")

    # Rede de segurança para quando a renda vem preenchida mas o DebtRatio ainda assim é
    # implausível: entre as linhas que sobram depois dos descartes acima (renda já
    # presente), 1,75% ficam fora de [0, DEBT_RATIO_MAXIMO]. É o mesmo teto que o
    # contrato de ingestão aplica — a mesma constante importada, não reafirmada — para
    # que limpeza e contrato nunca divirjam sobre o que é uma razão de dívida aceitável.
    _descartar(
        ~trabalho["DebtRatio"].between(0.0, DEBT_RATIO_MAXIMO),
        "razao_divida_implausivel",
    )

    limpo = trabalho[[ALVO, *FEATURES]].reset_index(drop=True)
    logger.info("referência com %d linhas; descartes: %s", len(limpo), motivos)
    return limpo, motivos


def separar(
    frame: pd.DataFrame,
    *,
    test_size: float,
    validation_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Separa em treino, validação e teste, estratificando pelo alvo.

    Com 6,94% de positivos, um corte aleatório sem estratificação pode variar a proporção
    da classe rara o bastante para mover a métrica mais que o próprio modelo.
    """
    resto, teste = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame[ALVO]
    )
    proporcao = validation_size / (1 - test_size)
    treino, validacao = train_test_split(
        resto, test_size=proporcao, random_state=seed, stratify=resto[ALVO]
    )
    return {
        "treino": treino.reset_index(drop=True),
        "validacao": validacao.reset_index(drop=True),
        "teste": teste.reset_index(drop=True),
    }
