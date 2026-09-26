"""O contrato precisa aprovar dado bom, reprovar cada defeito conhecido, e — o ponto
central da fase — aprovar dado deslocado, porque drift não é invalidez."""

from __future__ import annotations

import pandas as pd
import pytest

from credito.contracts.base import ContratoViolado
from credito.contracts.pandera_backend import (
    _REGRA_POR_CHECAGEM_DE_LOTE,
    _REGRA_POR_COLUNA,
    construir_validador,
)
from credito.contracts.rules import REGRAS


def _lote(n: int = 5, **ajustes) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.1 + 0.05 * i for i in range(n)],
            "age": [30 + i for i in range(n)],
            "NumberOfTime30-59DaysPastDueNotWorse": [0] * n,
            "DebtRatio": [0.3 + 0.01 * i for i in range(n)],
            "MonthlyIncome": [5000.0 + 100 * i for i in range(n)],
            "NumberOfOpenCreditLinesAndLoans": [7] * n,
            "NumberOfTimes90DaysLate": [0] * n,
            "NumberRealEstateLoansOrLines": [1] * n,
            "NumberOfTime60-89DaysPastDueNotWorse": [0] * n,
            "NumberOfDependents": [0.0] * n,
        }
    )
    for coluna, valores in ajustes.items():
        base[coluna] = valores
    return base


def test_lote_limpo_passa():
    resultado = construir_validador().validar(_lote())

    assert resultado.valido is True
    assert resultado.total == 5


@pytest.mark.parametrize(
    ("ajuste", "regra"),
    [
        ({"MonthlyIncome": [5000.0, None, 5200.0, 5300.0, 5400.0]}, "renda_nao_nula"),
        ({"age": [30, 17, 32, 33, 34]}, "idade_plausivel"),
        ({"NumberOfTimes90DaysLate": [0, 98, 0, 0, 0]}, "atraso_plausivel"),
        ({"NumberOfDependents": [0.0, None, 1.0, 0.0, 2.0]}, "dependentes_nao_nulo"),
        ({"DebtRatio": [0.3, 1159.0, 0.4, 0.5, 0.6]}, "razao_divida_plausivel"),
    ],
)
def test_cada_defeito_conhecido_e_reprovado(ajuste, regra):
    resultado = construir_validador().validar(_lote(**ajuste))

    assert resultado.valido is False
    assert regra in {violacao.regra for violacao in resultado.violacoes}


def test_duplicata_e_reprovada():
    lote = _lote(3)
    com_duplicata = pd.concat([lote, lote.iloc[[0]]], ignore_index=True)

    resultado = construir_validador().validar(com_duplicata)

    violacoes_duplicata = [v for v in resultado.violacoes if v.regra == "sem_duplicatas"]
    # Pino o nome relatado, não apenas a presença: a regra de duplicata é a única do
    # contrato cujo nome sobrevive ao caminho de fallback do Pandera (o nome customizado
    # de um `Check` de coluna não aparece em `failure_cases["check"]` nesta versão — só o
    # de um `Check` de DataFrame inteiro aparece). Um refactor que troque esse caminho
    # sem querer trocaria o nome relatado aqui, e este teste teria que acusar.
    assert len(violacoes_duplicata) == 1
    assert violacoes_duplicata[0].coluna == "*"
    assert 3 in violacoes_duplicata[0].indices


def test_duplicata_e_reprovada_mesmo_com_coluna_extra_unica():
    # A regressão que mais importa aqui: com o dedup olhando o lote inteiro, qualquer
    # coluna única por linha desligava a regra em silêncio — e a predição, que o lote de
    # produção carrega, é única por linha por construção. O mesmo lote do teste acima, com
    # uma coluna de predição junto, precisa continuar reprovando.
    lote = _lote(3)
    com_duplicata = pd.concat([lote, lote.iloc[[0]]], ignore_index=True)
    com_duplicata["predicao"] = [0.11, 0.22, 0.33, 0.44]

    resultado = construir_validador().validar(com_duplicata)

    violacoes_duplicata = [v for v in resultado.violacoes if v.regra == "sem_duplicatas"]
    assert len(violacoes_duplicata) == 1
    assert violacoes_duplicata[0].indices == (3,)


def test_lote_com_coluna_extra_e_aceito():
    # O que `strict=False` compra, dito por um teste em vez de só por um comentário:
    # o lote de produção chega com predição e, quando existe, com o alvo. O contrato exige
    # presença das colunas que ele conhece, não exclusividade. Com `strict=True` este lote
    # seria recusado por trazer informação a mais — e a Etapa 2 não conseguiria validar
    # nada que já tivesse sido pontuado.
    lote = _lote()
    lote["predicao"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    lote["inadimplente"] = [0, 0, 1, 0, 0]

    assert construir_validador().validar(lote).valido is True


def test_coluna_repetida_e_reprovada():
    # O par do teste acima: `strict=False` aceita coluna a mais, `unique_column_names=True`
    # recusa coluna repetida. Sem a segunda flag, um `age` duplicado passa em silêncio e
    # qualquer leitura por nome de coluna passa a devolver um frame onde se esperava uma
    # série — verificado desligando a flag: o lote abaixo era aceito.
    lote = _lote()
    repetida = pd.concat([lote, lote[["age"]]], axis=1)

    assert construir_validador().validar(repetida).valido is False


def test_coluna_ausente_aponta_o_lote_inteiro():
    # `coluna_ausente` é a única violação do contrato sem índice de linha: não há linha
    # culpada quando o upstream deixou de mandar a coluna. O piso defensável é o lote
    # inteiro, e é disso que a docstring de `linhas_reprovadas` fala.
    lote = _lote()
    resultado = construir_validador().validar(lote.drop(columns=["MonthlyIncome"]))

    violacao = next(v for v in resultado.violacoes if v.regra == "coluna_ausente")
    assert violacao.indices == ()
    assert violacao.linhas == len(lote)
    assert resultado.linhas_reprovadas == len(lote)


def test_lote_com_drift_passa_no_contrato():
    # O ponto central da fase: renda 60% maior e endividamento em dobro deslocam a
    # distribuição sem tornar nenhum registro inválido. Se este teste ficasse vermelho, o
    # contrato estaria barrando drift — e o detector de drift da Etapa 2 nunca veria o lote.
    deslocado = _lote()
    deslocado["MonthlyIncome"] = deslocado["MonthlyIncome"] * 1.6
    deslocado["DebtRatio"] = deslocado["DebtRatio"] * 2

    assert construir_validador().validar(deslocado).valido is True


def test_resultado_reprovado_bloqueia_a_ingestao():
    resultado = construir_validador().validar(_lote(age=[30, 0, 32, 33, 34]))

    with pytest.raises(ContratoViolado, match="idade_plausivel"):
        resultado.erguer()


def test_coluna_ausente_e_reprovada():
    # Um upstream que deixa de mandar uma coluna é falha de contrato, não dado faltante.
    resultado = construir_validador().validar(_lote().drop(columns=["MonthlyIncome"]))

    assert resultado.valido is False


def test_varias_violacoes_sao_reportadas_juntas():
    # Reportar só a primeira obrigaria a corrigir e re-rodar em ciclos; quem opera precisa
    # ver o estrago inteiro de uma vez.
    resultado = construir_validador().validar(
        _lote(age=[30, 0, 32, 33, 34], MonthlyIncome=[5000.0, 5100.0, None, 5300.0, 5400.0])
    )

    assert len(resultado.violacoes) >= 2


def test_indices_das_violacoes_de_coluna_apontam_as_linhas_certas():
    # `linhas_reprovadas` é exata quando os índices vêm preenchidos — vale a pena fixar
    # que o conversor realmente carrega o índice que o Pandera devolve, não só a contagem.
    resultado = construir_validador().validar(_lote(age=[30, 17, 32, 33, 34]))

    violacao = next(v for v in resultado.violacoes if v.regra == "idade_plausivel")
    assert violacao.indices == (1,)
    assert resultado.linhas_reprovadas == 1


def test_nomes_de_regra_sao_subconjunto_de_regras():
    # `_REGRA_POR_COLUNA` e `_REGRA_POR_CHECAGEM_DE_LOTE` são literais escritos à mão,
    # não derivados de REGRAS (ver docstring do módulo) — este teste é o que impede as
    # duas listas de se afastarem em silêncio: se um nome mudar em `rules.py` e não aqui
    # (ou vice-versa), a rastreabilidade quebra e este teste tem que acusar.
    nomes_usados = set(_REGRA_POR_COLUNA.values()) | set(_REGRA_POR_CHECAGEM_DE_LOTE.values())
    nomes_declarados = {regra.nome for regra in REGRAS}

    assert nomes_usados <= nomes_declarados


def test_coluna_sem_regra_de_negocio_reporta_campo_invalido():
    # RevolvingUtilizationOfUnsecuredLines tem exigência de tipo e de não-nulo, mas
    # nenhuma regra de negócio nomeada em REGRAS. Sem este sentinela, a violação cairia
    # de volta no nome bruto da coluna — um identificador camelCase disfarçado de nome de
    # regra, o único jeito de `regra` fugir do vocabulário do resto do sistema.
    resultado = construir_validador().validar(
        _lote(RevolvingUtilizationOfUnsecuredLines=[0.1, None, 0.2, 0.3, 0.4])
    )

    violacao = next(
        v for v in resultado.violacoes if v.coluna == "RevolvingUtilizationOfUnsecuredLines"
    )
    assert violacao.regra == "campo_invalido"
