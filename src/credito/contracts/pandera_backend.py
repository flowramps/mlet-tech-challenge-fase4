"""Execução do contrato com Pandera.

Os nomes de regra usados neste módulo (``_REGRA_POR_COLUNA`` e
``_REGRA_POR_CHECAGEM_DE_LOTE``) são literais escritos à mão, não derivados de
``rules.REGRAS`` — as duas listas existem separadas porque ``Regra`` carrega descrição e
motivo em prosa, feitos para leitura humana, e transformá-la numa camada de geração de
schema adicionaria indireção sem necessidade real. O que garante que as duas listas não
se afastem é um teste (``test_nomes_de_regra_sao_subconjunto_de_regras``), não a
estrutura do código: ele falha assim que um nome aqui deixar de existir em ``REGRAS``.

``lazy=True`` é essencial: sem ele o Pandera para na primeira falha, e quem está corrigindo
um lote precisa ver todos os problemas de uma vez.

Uma observação empírica (Pandera 0.33.1) mudou o desenho do conversor abaixo em relação a
uma abordagem ingênua: o ``name=`` passado a um ``Check`` de **coluna** não aparece em
``failure_cases["check"]`` — essa coluna traz a representação genérica do check
(``"in_range(18, 110)"``, ``"not_nullable"``), nunca o nome customizado. Só o ``name=`` de
um ``Check`` de **DataFrame inteiro** (o de duplicata) sobrevive ali. Por isso o
mapeamento de nome de regra para violação de coluna é feito por **nome de coluna**
(``_REGRA_POR_COLUNA``), não por texto do check; o mapeamento por nome de check
(``_REGRA_POR_CHECAGEM_DE_LOTE``) é reservado para as checagens de lote inteiro, onde o
nome realmente aparece — e ainda assim fica explícito aqui, em vez de reaproveitar
``failure_cases["check"]`` diretamente, para que uma mudança de versão do Pandera que pare
de preservar esse nome quebre um teste em vez de renomear a regra em silêncio.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from credito.contracts.base import ValidationResult, Validator, Violacao

# As cinco constantes compartilhadas moram em `data.prepare` e vêm de lá, todas pela mesma
# porta. `DEBT_RATIO_MAXIMO` já chegou aqui reexportada por `contracts.rules`: o valor era
# o mesmo, mas a rota extra sugeria que a constante pertencesse ao módulo de regras, e
# quem fosse alterá-la procuraria no lugar errado.
from credito.data.prepare import (
    ATRASO_MAXIMO_PLAUSIVEL,
    COLUNAS_DE_ATRASO,
    DEBT_RATIO_MAXIMO,
    IDADE_MAXIMA,
    IDADE_MINIMA,
)

_NOME_CHECAGEM_DUPLICATA = "sem_duplicatas"
_CHECAGEM_COLUNA_AUSENTE = "column_in_dataframe"

# Três colunas do schema (RevolvingUtilizationOfUnsecuredLines,
# NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines) têm exigência de tipo e
# de não-nulo, mas nenhuma regra de negócio nomeada em REGRAS. Uma falha nelas é real —
# não pode virar `regra=<nome da coluna>`, um identificador de coluna disfarçado de nome
# de regra, inconsistente com todo outro valor de `regra` do sistema. Este sentinela
# nomeia o caso: "restrição estrutural falhou numa coluna sem regra de negócio".
_REGRA_CAMPO_INVALIDO = "campo_invalido"

# Uma regra por coluna: nesse schema nenhuma coluna carrega duas regras diferentes, então
# o nome de coluna já identifica a regra sem ambiguidade — o que salva o conversor de
# precisar decifrar o texto do check (ver docstring do módulo). As colunas que ficam de
# fora deste mapa são exatamente as de `COLUNAS_SEM_REGRA_NOMEADA`, e um teste cobra essa
# correspondência: a limpeza da Referência espelha as exigências do contrato coluna a
# coluna, e uma coluna que mudasse de categoria aqui sem mudar lá abriria a divergência.
_REGRA_POR_COLUNA: dict[str, str] = {
    "MonthlyIncome": "renda_nao_nula",
    "age": "idade_plausivel",
    "DebtRatio": "razao_divida_plausivel",
    "NumberOfDependents": "dependentes_nao_nulo",
    **dict.fromkeys(COLUNAS_DE_ATRASO, "atraso_plausivel"),
}

# Mapeamento explícito e pinado por teste: mesmo o nome de uma checagem de lote inteiro
# sobrevivendo hoje em ``failure_cases["check"]`` é comportamento observado desta versão
# do Pandera, não uma garantia da API. Passar pelo dicionário em vez de usar o texto do
# check diretamente é o que torna essa dependência visível e testável.
_REGRA_POR_CHECAGEM_DE_LOTE: dict[str, str] = {
    _NOME_CHECAGEM_DUPLICATA: "sem_duplicatas",
}


def _coluna_de_atraso() -> pa.Column:
    return pa.Column(
        int,
        checks=pa.Check.in_range(0, ATRASO_MAXIMO_PLAUSIVEL, name="atraso_plausivel"),
        nullable=False,
        coerce=True,
    )


def _schema() -> pa.DataFrameSchema:
    colunas: dict[str, pa.Column] = {
        "RevolvingUtilizationOfUnsecuredLines": pa.Column(float, nullable=False, coerce=True),
        "age": pa.Column(
            int,
            checks=pa.Check.in_range(IDADE_MINIMA, IDADE_MAXIMA, name="idade_plausivel"),
            nullable=False,
            coerce=True,
        ),
        "DebtRatio": pa.Column(
            float,
            checks=pa.Check.in_range(0.0, DEBT_RATIO_MAXIMO, name="razao_divida_plausivel"),
            nullable=False,
            coerce=True,
        ),
        "MonthlyIncome": pa.Column(
            float,
            checks=pa.Check.ge(0.0, name="renda_nao_nula"),
            nullable=False,
            coerce=True,
        ),
        "NumberOfOpenCreditLinesAndLoans": pa.Column(int, nullable=False, coerce=True),
        "NumberRealEstateLoansOrLines": pa.Column(int, nullable=False, coerce=True),
        "NumberOfDependents": pa.Column(
            float,
            checks=pa.Check.ge(0.0, name="dependentes_nao_nulo"),
            nullable=False,
            coerce=True,
        ),
    }
    for coluna in COLUNAS_DE_ATRASO:
        colunas[coluna] = _coluna_de_atraso()

    nomes_do_contrato = tuple(colunas)

    def _sem_duplicatas(frame: pd.DataFrame) -> pd.Series:
        """Duplicata é avaliada só sobre as colunas do contrato, nunca sobre o lote inteiro.

        Deduplicar o frame inteiro faz a regra se desligar sozinha diante de qualquer
        coluna extra única por linha — e a predição, que o lote de produção carrega, é
        única por linha praticamente por construção. O resultado seria uma regra que
        continua no schema, continua aparecendo na documentação e não reprova mais nada,
        sem nada ficar vermelho. Restringir às colunas do contrato também alinha esta
        checagem ao dedup da limpeza da Referência, que já olha só para as ``FEATURES``.
        """
        presentes = [nome for nome in nomes_do_contrato if nome in frame.columns]
        return ~frame[presentes].duplicated()

    return pa.DataFrameSchema(
        colunas,
        # `strict=False` porque o lote de produção carrega a predição e a coluna de alvo
        # quando existe; o contrato exige presença, não exclusividade.
        strict=False,
        unique_column_names=True,
        # A regra de duplicata compara linhas, não valores de uma coluna: por isso é um
        # `Check` de DataFrame. O recorte de colunas que ela usa está em `_sem_duplicatas`.
        checks=pa.Check(
            _sem_duplicatas,
            name=_NOME_CHECAGEM_DUPLICATA,
            element_wise=False,
        ),
    )


class PanderaValidator:
    """Implementação de :class:`credito.contracts.base.Validator` sobre Pandera."""

    def __init__(self) -> None:
        self._schema = _schema()

    def validar(self, frame: pd.DataFrame) -> ValidationResult:
        try:
            self._schema.validate(frame, lazy=True)
        except pa.errors.SchemaErrors as erros:
            return ValidationResult(
                total=len(frame), violacoes=self._converter(erros, total=len(frame))
            )
        return ValidationResult(total=len(frame), violacoes=())

    @staticmethod
    def _converter(erros: pa.errors.SchemaErrors, *, total: int) -> tuple[Violacao, ...]:
        relatorio = erros.failure_cases
        violacoes: list[Violacao] = []

        estrutural = relatorio[relatorio["check"] == _CHECAGEM_COLUNA_AUSENTE]
        for coluna_ausente in estrutural["failure_case"].unique():
            # Coluna ausente é um problema de contrato — nenhuma linha individual pode
            # ser apontada como culpada, então o piso defensável é "todo o lote".
            violacoes.append(
                Violacao(regra="coluna_ausente", coluna=str(coluna_ausente), linhas=total)
            )

        por_coluna = relatorio[relatorio["schema_context"] == "Column"]
        for coluna, grupo in por_coluna.groupby("column", sort=True):
            nome_coluna = str(coluna)
            regra = _REGRA_POR_COLUNA.get(nome_coluna, _REGRA_CAMPO_INVALIDO)
            indices = tuple(sorted({int(valor) for valor in grupo["index"].dropna()}))
            violacoes.append(
                Violacao(regra=regra, coluna=nome_coluna, linhas=len(indices), indices=indices)
            )

        por_lote = relatorio[
            (relatorio["schema_context"] == "DataFrameSchema")
            & (relatorio["check"] != _CHECAGEM_COLUNA_AUSENTE)
        ]
        for checagem, grupo in por_lote.groupby("check", sort=True):
            nome_checagem = str(checagem)
            regra = _REGRA_POR_CHECAGEM_DE_LOTE.get(nome_checagem, _REGRA_CAMPO_INVALIDO)
            indices = tuple(sorted({int(valor) for valor in grupo["index"].dropna()}))
            violacoes.append(
                Violacao(regra=regra, coluna="*", linhas=len(indices), indices=indices)
            )

        return tuple(violacoes)


def construir_validador() -> Validator:
    """Ponto único de construção — quem valida não importa a implementação."""
    return PanderaValidator()
