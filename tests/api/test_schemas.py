"""`ScoreRequest` precisa espelhar `credito.schema.FEATURES` campo a campo, nunca uma
lista solta — é o que faz o corte que `para_registro` devolve casar exatamente com o que
`modelo.predict_proba` espera receber."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from credito.api.schemas import ScoreRequest
from credito.schema import (
    ATRASO_MAXIMO_PLAUSIVEL,
    DEBT_RATIO_MAXIMO,
    FEATURES,
    IDADE_MAXIMA,
    IDADE_MINIMA,
)

_PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2.0,
}


def test_aliases_cobrem_exatamente_features():
    # Mutação que este teste pega: alguém acrescenta ou renomeia uma feature em
    # `credito.schema.FEATURES` sem espelhar o campo aqui — o corte de `para_registro`
    # ficaria incompleto ou levantaria `KeyError` sem que nenhum outro teste percebesse.
    aliases = {campo.alias or nome for nome, campo in ScoreRequest.model_fields.items()}
    assert aliases == set(FEATURES)


def test_para_registro_devolve_exatamente_as_chaves_de_features_na_ordem_certa():
    registro = ScoreRequest(**_PAYLOAD).para_registro()

    assert list(registro.keys()) == list(FEATURES)
    assert registro["DebtRatio"] == 0.2
    assert registro["age"] == 45


def test_para_registro_segue_a_ordem_de_features_nao_a_ordem_de_declaracao(monkeypatch):
    """Sem isso, `para_registro` poderia devolver a ordem de declaração dos campos do
    Pydantic — que hoje coincide com `FEATURES`, mas coincidir não é a mesma garantia que
    depender de `FEATURES` de fato. Trocando o `FEATURES` que o módulo importou por uma
    ordem invertida, só uma implementação que releia `FEATURES` a cada chamada segue a
    troca; uma que apenas devolvesse `self.model_dump(by_alias=True)` continuaria na
    ordem de declaração e este teste cairia.
    """
    import credito.api.schemas as schemas_module

    ordem_invertida = tuple(reversed(FEATURES))
    monkeypatch.setattr(schemas_module, "FEATURES", ordem_invertida)

    registro = ScoreRequest(**_PAYLOAD).para_registro()

    assert list(registro.keys()) == list(ordem_invertida)


def test_score_request_aceita_construcao_por_alias_hifenizado():
    requisicao = ScoreRequest.model_validate(_PAYLOAD)

    assert requisicao.number_of_time_30_59_days_past_due_not_worse == 0
    assert requisicao.debt_ratio == 0.2


def test_identificador_enviado_a_mais_e_descartado_antes_de_chegar_ao_modelo():
    # Minimização de dado (LGPD, art. 6º, III) como propriedade do código, não como
    # promessa sobre o cliente: o sistema de origem pode, por engano, mandar CPF e nome
    # junto com as features. Eles têm que morrer na validação — não chegar ao modelo, ao
    # log nem à métrica. Mutação que este teste pega: `extra="allow"` no `model_config`
    # faria o CPF sobreviver no objeto validado.
    com_identificacao = {**_PAYLOAD, "cpf": "123.456.789-00", "nome": "Fulana de Tal"}

    requisicao = ScoreRequest.model_validate(com_identificacao)

    sobreviventes = set(requisicao.model_dump(by_alias=True))
    assert sobreviventes == set(FEATURES)
    assert set(requisicao.para_registro()) == set(FEATURES)


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("age", IDADE_MINIMA - 1),
        ("age", IDADE_MAXIMA + 1),
        ("DebtRatio", -0.01),
        ("DebtRatio", DEBT_RATIO_MAXIMO + 0.01),
        ("MonthlyIncome", -0.01),
        ("NumberOfDependents", -1),
        ("NumberOfTime30-59DaysPastDueNotWorse", -1),
        ("NumberOfTime30-59DaysPastDueNotWorse", ATRASO_MAXIMO_PLAUSIVEL + 1),
        ("NumberOfTimes90DaysLate", -1),
        ("NumberOfTimes90DaysLate", ATRASO_MAXIMO_PLAUSIVEL + 1),
        ("NumberOfTime60-89DaysPastDueNotWorse", -1),
        ("NumberOfTime60-89DaysPastDueNotWorse", ATRASO_MAXIMO_PLAUSIVEL + 1),
    ],
)
def test_score_request_rejeita_valor_fora_do_contrato(campo, valor):
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({**_PAYLOAD, campo: valor})


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("age", IDADE_MINIMA),
        ("age", IDADE_MAXIMA),
        ("DebtRatio", 0),
        ("DebtRatio", DEBT_RATIO_MAXIMO),
        ("MonthlyIncome", 0),
        ("NumberOfDependents", 0),
        ("NumberOfTime30-59DaysPastDueNotWorse", 0),
        ("NumberOfTime30-59DaysPastDueNotWorse", ATRASO_MAXIMO_PLAUSIVEL),
        ("NumberOfTimes90DaysLate", 0),
        ("NumberOfTimes90DaysLate", ATRASO_MAXIMO_PLAUSIVEL),
        ("NumberOfTime60-89DaysPastDueNotWorse", 0),
        ("NumberOfTime60-89DaysPastDueNotWorse", ATRASO_MAXIMO_PLAUSIVEL),
    ],
)
def test_score_request_aceita_limites_inclusivos_do_contrato(campo, valor):
    requisicao = ScoreRequest.model_validate({**_PAYLOAD, campo: valor})

    assert requisicao.para_registro()[campo] == valor
