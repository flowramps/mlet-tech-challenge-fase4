"""O lote adulterado é a prova de que o contrato funciona. Ele precisa conter cada defeito
que as regras dizem interceptar — nem menos (a demo ficaria fraca) nem outros (a demo
ficaria confusa)."""

from __future__ import annotations

import pandas as pd

from credito.contracts.pandera_backend import construir_validador
from credito.data.adulterate import DEFEITOS, adulterar


def _base(n: int = 200) -> pd.DataFrame:
    # RevolvingUtilizationOfUnsecuredLines varia um pouco por linha — sem essa variação,
    # as `n` linhas ficam idênticas byte a byte e a própria regra `sem_duplicatas`
    # reprovaria a base antes de qualquer adulteração, o que não tem nada a ver com o
    # defeito que este teste quer isolar.
    return pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.3 + i * 1e-4 for i in range(n)],
            "age": [40] * n,
            "NumberOfTime30-59DaysPastDueNotWorse": [0] * n,
            "DebtRatio": [0.4] * n,
            "MonthlyIncome": [5400.0] * n,
            "NumberOfOpenCreditLinesAndLoans": [7] * n,
            "NumberOfTimes90DaysLate": [0] * n,
            "NumberRealEstateLoansOrLines": [1] * n,
            "NumberOfTime60-89DaysPastDueNotWorse": [0] * n,
            "NumberOfDependents": [0.0] * n,
        }
    )


def test_base_limpa_passa_antes_da_adulteracao():
    assert construir_validador().validar(_base()).valido is True


def test_lote_adulterado_e_reprovado():
    sujo = adulterar(_base(), seed=42)

    assert construir_validador().validar(sujo).valido is False


def test_cada_defeito_aparece_no_relatorio():
    resultado = construir_validador().validar(adulterar(_base(), seed=42))
    regras_violadas = {violacao.regra for violacao in resultado.violacoes}

    for defeito in DEFEITOS:
        assert defeito in regras_violadas, f"o defeito {defeito} não foi interceptado"


def test_adulteracao_e_reprodutivel():
    assert adulterar(_base(), seed=42).equals(adulterar(_base(), seed=42))


def test_defeitos_atingem_linhas_disjuntas():
    # Uma linha com dois defeitos esconderia se alguma regra deixou de disparar — cada
    # violação reportada tem que apontar para um conjunto de índices que não se repete
    # entre as seis regras.
    resultado = construir_validador().validar(adulterar(_base(), seed=42))

    todos_indices: list[int] = []
    for violacao in resultado.violacoes:
        todos_indices.extend(violacao.indices)

    assert len(todos_indices) == len(set(todos_indices))


def test_maioria_das_linhas_continua_boa():
    # Um lote 100% podre não prova nada: o interessante é o contrato barrar a ingestão
    # inteira por causa de uma minoria corrompida, que é como a falha chega na vida real.
    sujo = adulterar(_base(), seed=42, por_defeito=3)
    resultado = construir_validador().validar(sujo)

    assert resultado.linhas_reprovadas < len(sujo) * 0.2
