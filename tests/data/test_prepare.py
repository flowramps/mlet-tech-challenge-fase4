"""A Referência é o dado que o modelo aprende. A política de limpeza precisa ser
explícita e auditável: cada linha descartada tem um motivo contabilizado."""

from __future__ import annotations

import pandas as pd

from credito.data.prepare import ALVO, FEATURES, limpar, separar


def _linha(**ajustes) -> dict:
    base = {
        "FinancialDistressNextTwoYears": "No",
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5400.0,
        "NumberOfOpenCreditLinesAndLoans": 7,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0.0,
    }
    base.update(ajustes)
    return base


def test_alvo_vira_inteiro_binario():
    frame = pd.DataFrame([_linha(), _linha(FinancialDistressNextTwoYears="Yes")])

    limpo, _ = limpar(frame)

    assert sorted(limpo[ALVO].unique()) == [0, 1]
    assert limpo[ALVO].dtype.kind == "i"


def test_descarta_renda_nula():
    frame = pd.DataFrame([_linha(), _linha(MonthlyIncome=None)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["renda_nula"] == 1


def test_descarta_menores_de_18():
    frame = pd.DataFrame([_linha(), _linha(age=0), _linha(age=17)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["idade_invalida"] == 2


def test_descarta_duplicatas_exatas():
    frame = pd.DataFrame([_linha(), _linha(), _linha(age=50)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 2
    assert motivos["duplicata"] == 1


def test_descarta_sentinelas_de_atraso():
    # 96 e 98 são códigos de "não disponível" no dataset de origem, não contagens: um
    # cliente com 98 atrasos de 90 dias não existe. São tipo-válidos e semanticamente
    # impossíveis — exatamente o que o contrato precisa interceptar.
    frame = pd.DataFrame(
        [
            _linha(),
            _linha(**{"NumberOfTimes90DaysLate": 98}),
            _linha(**{"NumberOfTime30-59DaysPastDueNotWorse": 96}),
        ]
    )

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["atraso_sentinela"] == 2


def test_descarta_dependentes_nulo():
    frame = pd.DataFrame([_linha(), _linha(NumberOfDependents=None)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["dependentes_nulo"] == 1


def test_referencia_sai_com_as_features_esperadas():
    limpo, _ = limpar(pd.DataFrame([_linha()]))

    assert list(limpo.columns) == [ALVO, *FEATURES]


def test_separacao_e_estratificada_e_reprodutivel():
    frame = pd.DataFrame(
        [_linha(age=30 + i) for i in range(80)]
        + [_linha(age=30 + i, FinancialDistressNextTwoYears="Yes") for i in range(20)]
    )
    limpo, _ = limpar(frame)

    particoes = separar(limpo, test_size=0.2, validation_size=0.2, seed=42)
    de_novo = separar(limpo, test_size=0.2, validation_size=0.2, seed=42)

    assert set(particoes) == {"treino", "validacao", "teste"}
    assert len(particoes["treino"]) + len(particoes["validacao"]) + len(particoes["teste"]) == 100
    # A proporção de positivos precisa sobreviver ao corte: com 20% de positivos, uma
    # partição que perdesse a estratificação poderia sair sem nenhum.
    for nome, parte in particoes.items():
        assert 0.1 <= parte[ALVO].mean() <= 0.3, nome
    assert particoes["teste"].equals(de_novo["teste"])
