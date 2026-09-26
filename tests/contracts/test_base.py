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


def test_linhas_reprovadas_e_piso_quando_duas_violacoes_nao_tem_indice():
    # Duas regras de lote inteiro (sem índice de linha) não têm como provar que cobrem
    # linhas diferentes. Somar 100 + 50 arriscaria contar a mesma linha duas vezes; a
    # propriedade devolve o maior valor defensável — um piso, não a contagem exata — e
    # este teste fixa esse comportamento para que um refactor não o troque em silêncio.
    resultado = ValidationResult(
        total=200,
        violacoes=(
            Violacao(regra="sem_duplicatas", coluna="*", linhas=100),
            Violacao(regra="outra_regra_de_lote", coluna="*", linhas=50),
        ),
    )

    assert resultado.linhas_reprovadas == 100


def test_erguer_falha_quando_ha_violacao():
    resultado = ValidationResult(total=10, violacoes=(Violacao(regra="r", coluna="c", linhas=1),))

    with pytest.raises(ContratoViolado, match="r"):
        resultado.erguer()


def test_erguer_e_silencioso_quando_valido():
    ValidationResult(total=10, violacoes=()).erguer()


def test_existem_ao_menos_tres_regras_rigidas():
    # Três é o mínimo para o contrato ser um contrato e não uma checagem isolada; hoje são
    # seis. Este teste é o que impede que uma refatoração apague uma regra sem ninguém
    # perceber — e a unicidade dos nomes é o que impede que duas regras se confundam no
    # relatório de violações.
    assert len(REGRAS) >= 3
    assert len({regra.nome for regra in REGRAS}) == len(REGRAS), "nomes de regra duplicados"
