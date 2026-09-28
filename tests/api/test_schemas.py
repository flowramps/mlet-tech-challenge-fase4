"""`ScoreRequest` precisa espelhar `credito.schema.FEATURES` campo a campo, nunca uma
lista solta — é o que faz o corte que `para_registro` devolve casar exatamente com o que
`modelo.predict_proba` espera receber."""

from __future__ import annotations

from credito.api.schemas import ScoreRequest
from credito.schema import FEATURES

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
