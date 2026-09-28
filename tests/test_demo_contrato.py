"""A demonstração do contrato é o que o README manda rodar para ver o bloqueio acontecer.

Ela é código de apresentação, e por isso mesmo precisa terminar sempre numa mensagem e num
código de saída — um traceback numa demonstração é a pior forma possível de explicar o que
o contrato faz.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from credito.contracts.base import ValidationResult, Violacao
from credito.schema import ALVO, FEATURES

_CAMINHO = Path(__file__).resolve().parents[1] / "scripts" / "demo_contrato.py"


@pytest.fixture
def demo(monkeypatch):
    """Carrega o script pelo caminho, como `make demo-contrato` faz.

    `scripts/` não é pacote importável: carregar por caminho é o que reproduz a execução
    real em vez de testar um módulo que ninguém roda desse jeito.
    """
    especificacao = importlib.util.spec_from_file_location("demo_contrato", _CAMINHO)
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)

    # 60 linhas distintas: `adulterar` exige 30 e a distinção entre elas mantém a âncora da
    # duplicata sendo o único defeito da cópia.
    referencia = pd.DataFrame(
        {
            ALVO: [0] * 60,
            "RevolvingUtilizationOfUnsecuredLines": [0.1 + 0.001 * i for i in range(60)],
            "age": [30 + i % 40 for i in range(60)],
            "NumberOfTime30-59DaysPastDueNotWorse": [0] * 60,
            "DebtRatio": [0.3] * 60,
            "MonthlyIncome": [5000.0] * 60,
            "NumberOfOpenCreditLinesAndLoans": [7] * 60,
            "NumberOfTimes90DaysLate": [0] * 60,
            "NumberRealEstateLoansOrLines": [1] * 60,
            "NumberOfTime60-89DaysPastDueNotWorse": [0] * 60,
            "NumberOfDependents": [0.0] * 60,
        }
    )[[ALVO, *FEATURES]]

    monkeypatch.setattr(modulo, "ler_arff", lambda _: referencia)
    monkeypatch.setattr(modulo, "limpar", lambda frame: (frame, {}))
    return modulo


def test_demonstracao_bloqueia_o_lote_adulterado(demo):
    # O desfecho que a demonstração existe para mostrar: lote limpo aceito, lote adulterado
    # recusado, saída 0 porque o contrato fez o que devia.
    assert demo.main() == 0


def test_referencia_reprovada_sai_com_mensagem_e_nao_com_traceback(demo, monkeypatch):
    # O desfecho que a demonstração não espera. Ele era o único caminho sem tratamento:
    # `erguer()` sobre o lote limpo escapava como traceback justamente quando havia algo
    # de verdade a relatar — a divergência entre a limpeza e o contrato. Verificado
    # removendo o `try` da demonstração: este teste vira erro em vez de falha.
    class _SempreRecusa:
        def validar(self, frame):
            return ValidationResult(
                total=len(frame),
                violacoes=(Violacao(regra="renda_nao_nula", coluna="MonthlyIncome", linhas=1),),
            )

    monkeypatch.setattr(demo, "construir_validador", lambda: _SempreRecusa())

    assert demo.main() == 1
