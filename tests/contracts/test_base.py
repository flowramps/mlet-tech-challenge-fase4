"""A fronteira do contrato existe para que as regras sejam declaradas uma vez e a
biblioteca que as executa seja trocável sem reescrevê-las."""

from __future__ import annotations

import pytest

from credito.contracts.base import ContratoViolado, ValidationResult, Violacao
from credito.contracts.rules import REGRAS


def test_resultado_sem_violacoes_e_valido():
    resultado = ValidationResult(total=10, violacoes=())

    assert resultado.valido is True
    assert resultado.linhas_reprovadas == 0


def test_resultado_com_violacao_nao_e_valido():
    resultado = ValidationResult(
        total=10, violacoes=(Violacao(regra="idade_minima", coluna="age", linhas=3),)
    )

    assert resultado.valido is False
    assert resultado.linhas_reprovadas == 3


def test_linhas_reprovadas_nao_soma_duas_vezes_a_mesma_linha():
    # Uma linha pode violar várias regras. Somar as contagens por regra superestimaria o
    # dano e faria o relatório dizer que 12 linhas falharam num lote de 10.
    resultado = ValidationResult(
        total=10,
        violacoes=(
            Violacao(regra="idade_minima", coluna="age", linhas=3, indices=(0, 1, 2)),
            Violacao(regra="renda_nao_nula", coluna="MonthlyIncome", linhas=2, indices=(2, 5)),
        ),
    )

    assert resultado.linhas_reprovadas == 4


def test_erguer_falha_quando_ha_violacao():
    resultado = ValidationResult(total=10, violacoes=(Violacao(regra="r", coluna="c", linhas=1),))

    with pytest.raises(ContratoViolado, match="r"):
        resultado.erguer()


def test_erguer_e_silencioso_quando_valido():
    ValidationResult(total=10, violacoes=()).erguer()


def test_existem_ao_menos_tres_regras_rigidas():
    # O enunciado exige no mínimo três. Este teste é o que impede que uma refatoração
    # apague uma regra sem ninguém perceber.
    assert len(REGRAS) >= 3
    assert len({regra.nome for regra in REGRAS}) == len(REGRAS), "nomes de regra duplicados"
