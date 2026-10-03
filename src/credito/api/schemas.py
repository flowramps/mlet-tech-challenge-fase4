"""Contratos de entrada e saída da API de scoring.

`ScoreRequest` tem um campo por `credito.schema.FEATURES` — a mesma lista que o contrato
de dados e o modelo esperam, nunca uma cópia solta. Três dessas features têm hífen no
nome (`NumberOfTime30-59DaysPastDueNotWorse` e as duas colunas irmãs de atraso), que não é
identificador Python válido; por isso o nome do campo troca o hífen por underscore e o
``alias`` carrega o nome original — que é o nome que a API recebe no corpo da requisição,
o mesmo que o dataset e o contrato já usam. `test_schemas.py` tranca que o conjunto de
aliases nunca fique fora de sincronia com `FEATURES`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from credito.schema import FEATURES


class ScoreRequest(BaseModel):
    """Um registro de crédito a pontuar."""

    # `extra="ignore"` é o default do Pydantic, declarado aqui porque é decisão de
    # privacidade, não acidente de biblioteca: um identificador enviado a mais (CPF, nome)
    # é descartado na validação e nunca chega ao modelo, ao log nem à métrica — a
    # minimização de dado como propriedade do código (ver `docs/governanca.md`).
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    revolving_utilization_of_unsecured_lines: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines",
        description="Saldo em aberto nas linhas de crédito rotativo, sobre o limite total.",
    )
    age: int = Field(description="Idade do cliente, em anos completos.")
    number_of_time_30_59_days_past_due_not_worse: int = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Quantidade de atrasos de 30 a 59 dias nos últimos dois anos.",
    )
    debt_ratio: float = Field(
        alias="DebtRatio", description="Razão entre dívida mensal e renda mensal."
    )
    monthly_income: float = Field(alias="MonthlyIncome", description="Renda mensal declarada.")
    number_of_open_credit_lines_and_loans: int = Field(
        alias="NumberOfOpenCreditLinesAndLoans",
        description="Quantidade de linhas de crédito e empréstimos abertos.",
    )
    number_of_times_90_days_late: int = Field(
        alias="NumberOfTimes90DaysLate",
        description="Quantidade de atrasos de 90 dias ou mais nos últimos dois anos.",
    )
    number_real_estate_loans_or_lines: int = Field(
        alias="NumberRealEstateLoansOrLines",
        description="Quantidade de empréstimos imobiliários e linhas de crédito hipotecárias.",
    )
    number_of_time_60_89_days_past_due_not_worse: int = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Quantidade de atrasos de 60 a 89 dias nos últimos dois anos.",
    )
    number_of_dependents: float = Field(
        alias="NumberOfDependents", description="Quantidade de dependentes do cliente."
    )

    def para_registro(self) -> dict[str, float]:
        """Devolve um dict com exatamente as chaves de `FEATURES`, na ordem de `FEATURES`
        — o formato que `modelo.predict_proba` espera (o mesmo recorte que
        `credito.monitoring.proxies._probabilidades` já faz sobre um lote inteiro, aqui
        para um único registro). Ler por `by_alias=True` é o que devolve os nomes
        originais do dataset (`"DebtRatio"`, não `"debt_ratio"`), que é a chave que
        `FEATURES` de fato contém.
        """
        dados = self.model_dump(by_alias=True)
        return {feature: dados[feature] for feature in FEATURES}


class ScoreResponse(BaseModel):
    """Resultado da pontuação de crédito para um registro."""

    model_config = ConfigDict(protected_namespaces=())

    id_decisao: str = Field(
        description=(
            "Identificador desta decisão no registro de auditoria. O sistema de origem guarda "
            "o vínculo com o cliente; é por ele que uma revisão (LGPD, art. 20) reconstrói "
            "a decisão."
        )
    )
    probabilidade_inadimplencia: float = Field(
        description="Probabilidade que o modelo atribui ao cliente ser inadimplente."
    )
    aprovado: bool = Field(
        description=(
            "Decisão de aprovação: verdadeiro quando a probabilidade fica abaixo do "
            "limiar de decisão."
        )
    )
    limiar: float = Field(description="Limiar de decisão usado nesta resposta.")
    candidato: str = Field(description="Tipo do modelo campeão que respondeu.")
    inference_ms: float = Field(description="Tempo de inferência do modelo, em ms.")


class HealthResponse(BaseModel):
    """Estado do serviço e do modelo carregado."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    candidato: str
