from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credito.api.main import create_app
from credito.monitoring.proxies import LIMIAR_PADRAO, taxa_de_aprovacao
from credito.schema import FEATURES

METADADOS = {"candidato": "xgboost", "auc_pr": 0.4147, "recall_positivo": 0.71, "seed": 42}


def _payload(**overrides: float) -> dict[str, float]:
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
    """Mesmo dublê de `tests/api/test_main.py`: devolve
    `RevolvingUtilizationOfUnsecuredLines` como probabilidade, controlável pelo corpo da
    requisição."""

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        assert list(entradas.columns) == list(
            FEATURES
        ), f"esperava receber exatamente FEATURES, recebeu {list(entradas.columns)!r}"
        p = entradas["RevolvingUtilizationOfUnsecuredLines"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


@pytest.fixture
def client():
    with TestClient(create_app(_ModeloPorRevolving(), METADADOS)) as cliente:
        yield cliente


def test_metrics_responde_no_formato_prometheus(client):
    resposta = client.get("/metrics")
    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/plain")


def test_score_incrementa_o_contador_por_rota_e_status(client):
    client.post("/score", json=_payload())
    client.post("/score", json=_payload())
    corpo = client.get("/metrics").text
    assert 'credito_http_requests_total{method="POST",route="/score",status="200"} 2.0' in corpo


def test_erro_de_validacao_conta_com_o_status_422(client):
    """Sem contar os 4xx não existe painel de taxa de erro que mereça o nome."""
    client.post("/score", json={})
    corpo = client.get("/metrics").text
    assert 'credito_http_requests_total{method="POST",route="/score",status="422"} 1.0' in corpo


def test_duracao_da_requisicao_e_observada_por_rota(client):
    client.post("/score", json=_payload())
    corpo = client.get("/metrics").text
    assert 'credito_http_request_duration_seconds_count{method="POST",route="/score"} 1.0' in corpo


def test_inferencia_e_observada_separada_do_http(client):
    client.post("/score", json=_payload())
    corpo = client.get("/metrics").text
    assert "credito_inference_duration_seconds_count 1.0" in corpo


def test_distribuicao_do_score_e_observada(client):
    client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.42))
    corpo = client.get("/metrics").text
    assert "credito_score_distribution_count 1.0" in corpo
    assert "credito_score_distribution_sum 0.42" in corpo


def test_taxa_de_aprovacao_bate_com_credito_monitoring_proxies(client):
    """A garantia de reuso: a métrica não pode ser uma segunda implementação do corte de
    limiar — precisa bater EXATAMENTE com uma chamada direta a
    `credito.monitoring.proxies.taxa_de_aprovacao` sobre as mesmas probabilidades, na
    mesma ordem."""
    probabilidades = [0.1, 0.6, 0.2, 0.9, 0.3]
    for p in probabilidades:
        client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=p))

    corpo = client.get("/metrics").text
    esperado = taxa_de_aprovacao(np.array(probabilidades), limiar=LIMIAR_PADRAO)
    assert f"credito_taxa_de_aprovacao {esperado}" in corpo


def test_taxa_de_aprovacao_chama_de_fato_credito_monitoring_proxies(client, monkeypatch):
    """A garantia forte de reuso, além da comparação numérica de
    `test_taxa_de_aprovacao_bate_com_credito_monitoring_proxies` (que uma reimplementação
    friamente equivalente — por exemplo, `<=` em vez de `<` — passaria sem chamar a
    função de verdade, porque nenhuma das probabilidades testadas cai exatamente no
    limiar). Substitui o nome importado em `credito.api.metrics` por um dublê que
    devolve um valor sentinela; se o gauge refletir o sentinela, a chamada passou de
    fato pela função importada, não por uma segunda implementação do mesmo cálculo.
    """
    import credito.api.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "taxa_de_aprovacao", lambda *args, **kwargs: 0.1234)

    client.post("/score", json=_payload())
    corpo = client.get("/metrics").text

    assert "credito_taxa_de_aprovacao 0.1234" in corpo


def test_taxa_de_aprovacao_no_limiar_exato_conta_como_aprovado(client):
    """Fecha a lacuna que a comparação numérica sozinha deixa passar: com uma
    probabilidade EXATAMENTE no limiar, `<` (a convenção de `taxa_de_aprovacao`) aprova e
    `<=` também aprovaria — mas qualquer implementação que usasse `>` ou `>=` invertida
    seria pega aqui."""
    client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=LIMIAR_PADRAO))
    corpo = client.get("/metrics").text
    esperado = taxa_de_aprovacao(np.array([LIMIAR_PADRAO]), limiar=LIMIAR_PADRAO)
    assert f"credito_taxa_de_aprovacao {esperado}" in corpo


def test_taxa_de_aprovacao_atualiza_a_cada_novo_score(client):
    """Mutação que este teste pega: calcular a taxa uma vez (no primeiro score) e nunca
    recalcular depois — o gauge ficaria travado em 1.0 mesmo depois que uma nova
    requisição fosse recusada."""
    for _ in range(3):
        client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.1))
    assert "credito_taxa_de_aprovacao 1.0" in client.get("/metrics").text

    client.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.9))
    corpo = client.get("/metrics").text
    assert "credito_taxa_de_aprovacao 0.75" in corpo


def test_model_info_expoe_candidato_e_metricas_de_qualidade(client):
    """No Grafana dá para saber qual modelo respondia em cada janela de tempo sem sair
    do painel."""
    corpo = client.get("/metrics").text
    assert "credito_model_info" in corpo
    assert 'candidato="xgboost"' in corpo
    assert 'auc_pr="0.4147"' in corpo


def test_scrape_do_metrics_nao_conta_como_trafego(client):
    """O coletor raspa em intervalo fixo: se o scrape contasse, o painel de total de
    requisições subiria sozinho com a API sem nenhum cliente real."""
    client.get("/metrics")
    corpo = client.get("/metrics").text
    assert 'route="/metrics"' not in corpo


def test_rota_nao_declarada_nao_vaza_o_caminho_bruto_como_rotulo(client):
    """A garantia central de cardinalidade: um scanner varrendo URLs aleatórias não pode
    criar uma série nova por URL. Uma rota inexistente usa o rótulo fixo `sem_rota`,
    nunca o caminho bruto da requisição."""
    resposta = client.get("/rota-que-nao-existe-e2f9c1")
    assert resposta.status_code == 404

    corpo = client.get("/metrics").text
    assert "rota-que-nao-existe-e2f9c1" not in corpo
    assert 'route="sem_rota"' in corpo


def test_cada_aplicacao_tem_registro_proprio():
    """A suíte cria várias apps no mesmo processo; um registry global colidiria na
    segunda criação e vazaria contadores de um teste para o outro."""
    with TestClient(create_app(_ModeloPorRevolving(), METADADOS)) as primeira:
        primeira.post("/score", json=_payload())

    with TestClient(create_app(_ModeloPorRevolving(), METADADOS)) as segunda:
        corpo = segunda.get("/metrics").text

    assert 'route="/score"' not in corpo
