"""Vocabulário do domínio: os nomes das colunas e os limites que definem um registro válido.

Este módulo não tem lógica e não importa nada do projeto. Ele existe para que política de
limpeza, contrato de dados e detecção de drift compartilhem uma definição só sem depender
uns dos outros: o detector de drift precisa saber quais são as features, e não tem nada a
ver com a política de descarte que constrói a Referência.
"""

from __future__ import annotations

ALVO = "inadimplente"
ALVO_ORIGINAL = "FinancialDistressNextTwoYears"

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

# As colunas que o contrato exige sem ter uma regra de negócio nomeada para elas: a única
# exigência é estrutural — existir, ter o tipo certo e não ser nula. Ficam declaradas aqui,
# junto das outras constantes compartilhadas, para que limpeza e contrato usem a mesma
# lista em vez de cada um manter a sua.
COLUNAS_SEM_REGRA_NOMEADA: tuple[str, ...] = (
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
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
