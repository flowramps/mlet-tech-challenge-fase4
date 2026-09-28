from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credito.api.main import create_app
from credito.schema import FEATURES

METADADOS = {"candidato": "xgboost", "auc_pr": 0.4147, "recall_positivo": 0.71, "seed": 42}


def _payload(**overrides: float) -> dict[str, float]:
    """Um registro válido de crédito, com todos os campos de `FEATURES` preenchidos."""
    base = {
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
    base.update(overrides)
    return base


class _ModeloPorRevolving:
    """Devolve `RevolvingUtilizationOfUnsecuredLines` diretamente como probabilidade —
    controlável pelo próprio corpo da requisição, sem precisar de um modelo real
    treinado. O mesmo papel que `_ModeloPorFeature` cumpre em
    `tests/monitoring/test_proxies.py`.

    Confere que recebe só `FEATURES`, na ordem certa — a mesma prova de recorte que aquele
    teste já faz, aqui na fronteira da API.
    """

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        assert list(entradas.columns) == list(
            FEATURES
        ), f"esperava receber exatamente FEATURES, recebeu {list(entradas.columns)!r}"
        p = entradas["RevolvingUtilizationOfUnsecuredLines"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


@pytest.fixture
def client():
    # O `with` é obrigatório: sem ele o TestClient não dispara o lifespan e o modelo
    # nunca chega em app.state.
    with TestClient(create_app(_ModeloPorRevolving(), METADADOS)) as cliente:
        yield cliente


def test_health_reporta_status_e_candidato(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["candidato"] == "xgboost"


def test_score_devolve_probabilidade_do_modelo(client):
    resposta = client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.73))
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["probabilidade_inadimplencia"] == pytest.approx(0.73)
    assert corpo["candidato"] == "xgboost"


def test_score_aprova_abaixo_do_limiar_e_recusa_acima(client):
    aprovado = client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.1)).json()
    recusado = client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.9)).json()

    assert aprovado["aprovado"] is True
    assert recusado["aprovado"] is False
    assert aprovado["limiar"] == pytest.approx(0.5)


def test_score_no_limiar_exato_nao_e_aprovado():
    """Fecha a lacuna que os dois casos distantes (0,1 e 0,9) deixam passar: a convenção
    de `taxa_de_aprovacao` é `probabilidade < limiar` (estrito) — no limiar exato, a
    decisão é recusar. Uma implementação com `<=` aprovaria aqui, e este teste é o único
    que pega essa troca de operador."""
    with TestClient(create_app(_ModeloPorRevolving(), METADADOS)) as cliente:
        corpo = cliente.post(
            "/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.5)
        ).json()
    assert corpo["aprovado"] is False


def test_score_reporta_tempo_de_inferencia(client):
    corpo = client.post("/score", json=_payload()).json()
    assert corpo["inference_ms"] >= 0.0


def test_score_rejeita_corpo_sem_campo_obrigatorio(client):
    incompleto = _payload()
    del incompleto["DebtRatio"]
    resposta = client.post("/score", json=incompleto)
    assert resposta.status_code == 422


def test_score_rejeita_corpo_vazio(client):
    assert client.post("/score", json={}).status_code == 422


def test_openapi_documenta_as_tres_rotas(client):
    caminhos = client.get("/openapi.json").json()["paths"]
    assert "/health" in caminhos
    assert "/score" in caminhos
    assert "/metrics" in caminhos


def test_startup_registra_o_candidato_carregado(caplog):
    """Sem isso não há como saber, pelo log, qual candidato subiu em produção."""
    with (
        caplog.at_level("INFO", logger="credito.api.main"),
        TestClient(create_app(_ModeloPorRevolving(), METADADOS)),
    ):
        pass

    assert any("xgboost" in registro.getMessage() for registro in caplog.records)
