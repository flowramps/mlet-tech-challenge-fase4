"""Com 6,94% de positivos, um classificador que responde sempre "adimplente" acerta 93%
das vezes e não serve para nada. O tratamento de desbalanceamento não é refinamento: é o
que separa um modelo de uma constante."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credito.model.train import TIPOS_DE_MODELO, carregar_modelo, salvar_modelo, treinar
from credito.schema import ALVO, FEATURES


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


@pytest.fixture
def desbalanceado_com_sobreposicao() -> pd.DataFrame:
    """Fixture dedicada ao teste de colapso.

    Em `desbalanceado`, a renda separa as classes por ~13 desvios-padrão — qualquer
    classificador, com ou sem peso, acerta 100% sobre ela (verificado manualmente: um
    modelo treinado sem `class_weight`/`scale_pos_weight` também identifica todos os
    positivos ali, porque a fixture é trivial demais para exercitar o tratamento de
    desbalanceamento). Crédito real não tem sinal isolado tão limpo, então aqui a
    diferença de médias fica abaixo de 1 desvio-padrão — sobreposição real, do tipo que
    só um modelo com tratamento explícito consegue explorar.
    """
    gerador = np.random.default_rng(42)
    n = 8000
    positivo = gerador.random(n) < 0.07
    dados = {
        "RevolvingUtilizationOfUnsecuredLines": gerador.random(n),
        "age": gerador.integers(18, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": gerador.integers(0, 3, n),
        "DebtRatio": gerador.random(n),
        "MonthlyIncome": np.where(positivo, 3900.0, 5100.0) + gerador.normal(0, 1300, n),
        "NumberOfOpenCreditLinesAndLoans": gerador.integers(1, 15, n),
        "NumberOfTimes90DaysLate": gerador.integers(0, 2, n),
        "NumberRealEstateLoansOrLines": gerador.integers(0, 4, n),
        "NumberOfTime60-89DaysPastDueNotWorse": gerador.integers(0, 2, n),
        "NumberOfDependents": gerador.integers(0, 4, n).astype(float),
        ALVO: positivo.astype(int),
    }
    return pd.DataFrame(dados)[[ALVO, *FEATURES]]


def _montar_sem_peso(tipo: str, seed: int) -> Pipeline:
    """Réplica do candidato de `treinar`, mas sem nenhum tratamento de desbalanceamento.

    É o controle do teste de colapso: mesma arquitetura e mesmos hiperparâmetros do
    candidato de produção, com a única diferença sendo o peso da classe. Se o candidato
    de produção não superar este controle, o peso não estava fazendo nada.
    """
    if tipo == "regressao_logistica":
        return Pipeline(
            [
                ("imputacao", SimpleImputer(strategy="median")),
                ("escala", StandardScaler()),
                ("classificador", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    return Pipeline(
        [
            (
                "classificador",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="aucpr",
                    random_state=seed,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            )
        ]
    )


def _recall_positivo(modelo: Pipeline, avaliacao: pd.DataFrame) -> float:
    previsto = modelo.predict(avaliacao[list(FEATURES)])
    verdadeiro = avaliacao[ALVO].to_numpy()
    return float(((previsto == 1) & (verdadeiro == 1)).sum() / verdadeiro.sum())


@pytest.mark.parametrize("tipo", TIPOS_DE_MODELO)
def test_treina_e_prediz_probabilidade(desbalanceado, tipo):
    modelo = treinar(desbalanceado, tipo, seed=42)

    probabilidades = modelo.predict_proba(desbalanceado[list(FEATURES)])[:, 1]

    assert probabilidades.shape == (len(desbalanceado),)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


@pytest.mark.parametrize("tipo", TIPOS_DE_MODELO)
def test_nao_colapsa_na_classe_majoritaria(desbalanceado_com_sobreposicao, tipo):
    # Comparação contra um controle idêntico, mas sem peso, treinado e avaliado sobre o
    # mesmo corte treino/teste (avaliar em dados nunca vistos importa: um XGBoost com
    # capacidade suficiente memoriza o próprio treino independente do peso, o que
    # mascararia justamente o efeito que este teste precisa provar). Se o tratamento de
    # desbalanceamento não fizer diferença, o candidato de produção não recupera mais
    # positivos do que o controle sem peso.
    treino, avaliacao = train_test_split(
        desbalanceado_com_sobreposicao,
        test_size=0.3,
        random_state=42,
        stratify=desbalanceado_com_sobreposicao[ALVO],
    )

    modelo = treinar(treino, tipo, seed=42)
    controle = _montar_sem_peso(tipo, seed=42)
    controle.fit(treino[list(FEATURES)], treino[ALVO])

    recall_tratado = _recall_positivo(modelo, avaliacao)
    recall_controle = _recall_positivo(controle, avaliacao)

    assert recall_tratado > recall_controle + 0.1, (
        "o tratamento de desbalanceamento não recuperou mais inadimplentes que o "
        "controle sem peso"
    )


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
