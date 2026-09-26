"""Com 6,94% de positivos, um classificador que responde sempre "adimplente" acerta 93%
das vezes e não serve para nada. O tratamento de desbalanceamento não é refinamento: é o
que separa um modelo de uma constante."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credito.data.prepare import ALVO, FEATURES
from credito.model.train import TIPOS_DE_MODELO, carregar_modelo, salvar_modelo, treinar


@pytest.fixture
def desbalanceado() -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    n = 600
    positivo = gerador.random(n) < 0.07
    dados = {
        "RevolvingUtilizationOfUnsecuredLines": gerador.random(n),
        "age": gerador.integers(18, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": gerador.integers(0, 3, n),
        "DebtRatio": gerador.random(n),
        # A renda carrega o sinal: quem é positivo ganha menos.
        "MonthlyIncome": np.where(positivo, 2000.0, 6000.0) + gerador.normal(0, 300, n),
        "NumberOfOpenCreditLinesAndLoans": gerador.integers(1, 15, n),
        "NumberOfTimes90DaysLate": gerador.integers(0, 2, n),
        "NumberRealEstateLoansOrLines": gerador.integers(0, 4, n),
        "NumberOfTime60-89DaysPastDueNotWorse": gerador.integers(0, 2, n),
        "NumberOfDependents": gerador.integers(0, 4, n).astype(float),
        ALVO: positivo.astype(int),
    }
    return pd.DataFrame(dados)[[ALVO, *FEATURES]]


@pytest.mark.parametrize("tipo", TIPOS_DE_MODELO)
def test_treina_e_prediz_probabilidade(desbalanceado, tipo):
    modelo = treinar(desbalanceado, tipo, seed=42)

    probabilidades = modelo.predict_proba(desbalanceado[list(FEATURES)])[:, 1]

    assert probabilidades.shape == (len(desbalanceado),)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


@pytest.mark.parametrize("tipo", TIPOS_DE_MODELO)
def test_nao_colapsa_na_classe_majoritaria(desbalanceado, tipo):
    # Sem tratamento de desbalanceamento, ambos os modelos preveriam 0 para tudo.
    modelo = treinar(desbalanceado, tipo, seed=42)

    previsto = modelo.predict(desbalanceado[list(FEATURES)])

    assert previsto.sum() > 0, "o modelo prevê a classe majoritária para todo mundo"


@pytest.mark.parametrize("tipo", TIPOS_DE_MODELO)
def test_treino_e_reprodutivel(desbalanceado, tipo):
    um = treinar(desbalanceado, tipo, seed=42).predict_proba(desbalanceado[list(FEATURES)])
    outro = treinar(desbalanceado, tipo, seed=42).predict_proba(desbalanceado[list(FEATURES)])

    np.testing.assert_allclose(um, outro)


def test_tipo_desconhecido_falha(desbalanceado):
    with pytest.raises(ValueError, match="desconhecido"):
        treinar(desbalanceado, "inexistente", seed=42)


def test_salvar_e_carregar_preserva_predicoes_e_metadados(desbalanceado, tmp_path):
    modelo = treinar(desbalanceado, "regressao_logistica", seed=42)
    caminho = tmp_path / "m.joblib"

    salvar_modelo(modelo, caminho, {"tipo": "regressao_logistica", "auc_pr": 0.42})
    recarregado, metadados = carregar_modelo(caminho)

    np.testing.assert_allclose(
        modelo.predict_proba(desbalanceado[list(FEATURES)]),
        recarregado.predict_proba(desbalanceado[list(FEATURES)]),
    )
    assert metadados["tipo"] == "regressao_logistica"


def test_modelo_tolera_nulo_residual(desbalanceado):
    # A Referência não tem nulo, mas um lote de produção validado pode trazer coluna
    # numérica com ausência residual. O pipeline precisa não estourar.
    com_nulo = desbalanceado.copy()
    com_nulo.loc[0, "MonthlyIncome"] = np.nan

    modelo = treinar(desbalanceado, "regressao_logistica", seed=42)

    assert modelo.predict_proba(com_nulo[list(FEATURES)]).shape[0] == len(com_nulo)
