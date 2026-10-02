"""O cliente MLflow registra UMA execução do pipeline de monitoramento como um run —
parâmetros, métricas (com série temporal por mês quando aplicável) e artefatos.

Cada teste sobe um backend SQLite isolado dentro de `tmp_path` (nunca o backend de
arquivo, que o MLflow 3.16 pôs em modo de manutenção — ver o docstring de
`credito.tracking.mlflow_client`) e nunca reaproveita o `mlruns/` do repositório: os dois
testes de artefato conferem isso apontando `artifact_location` para dentro de `tmp_path` e
verificando que o `artifact_uri` do run devolvido de fato começa por lá.
"""

from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

from credito.config import Settings
from credito.drift.base import DriftDeFeature, DriftReport, Severidade, classificar
from credito.drift.gate import avaliar_gate
from credito.tracking.mlflow_client import (
    parametros_da_execucao,
    passo_do_lote,
    registrar_execucao,
)


def _feature(nome: str, psi: float, ks_p_valor: float = 0.5) -> DriftDeFeature:
    return DriftDeFeature(
        feature=nome, psi_divergencia=psi, ks_p_valor=ks_p_valor, severidade=classificar(psi)
    )


def _uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/mlflow.db"


# --- passo_do_lote: a ponte entre o nome do lote ("mes_XX") e o `step` do MLflow --------


@pytest.mark.parametrize(
    ("lote", "esperado"),
    [("mes_00", 0), ("mes_01", 1), ("mes_06", 6), ("mes_12", 12)],
)
def test_passo_do_lote_le_o_numero_do_mes(lote, esperado):
    assert passo_do_lote(lote) == esperado


@pytest.mark.parametrize("lote", ["lote_1", "mes_x", "mes01", "MES_01", ""])
def test_passo_do_lote_rejeita_nome_fora_do_padrao(lote):
    with pytest.raises(ValueError, match="mes_"):
        passo_do_lote(lote)


# --- parametros_da_execucao: os limiares e a semente, nunca as métricas medidas --------


def test_parametros_da_execucao_extrai_semente_janela_e_limiares_do_settings():
    settings = Settings()

    parametros = parametros_da_execucao(settings, seed=42, meses=6)

    assert parametros == {
        "semente": 42,
        "meses": 6,
        "psi_atencao": settings.psi_atencao,
        "psi_critico": settings.psi_critico,
        "min_auc_pr": settings.min_auc_pr,
        "min_recall_positivo": settings.min_recall_positivo,
    }


# --- registrar_execucao: parâmetros viram parâmetro, nunca métrica ----------------------


def test_registrar_execucao_grava_parametros_como_parametro_mlflow(tmp_path):
    gate = avaliar_gate([DriftReport(lote="mes_01", features=(_feature("age", 0.01),))])

    run_id = registrar_execucao(
        tracking_uri=_uri(tmp_path),
        experimento="teste-parametros",
        parametros={"semente": 42, "meses": 6, "psi_atencao": 0.10},
        metricas_do_campeao_por_lote={},
        drift_por_lote=[],
        proxies_por_lote={},
        gate=gate,
        artifact_location=str(tmp_path / "artefatos"),
    )

    client = MlflowClient(tracking_uri=_uri(tmp_path))
    r = client.get_run(run_id)

    assert r.data.params["semente"] == "42"
    assert r.data.params["meses"] == "6"
    assert r.data.params["psi_atencao"] == "0.1"
    # o alfa de Benjamini-Hochberg do próprio gate também é parâmetro — vem do `gate`,
    # nunca duplicado à mão pelo chamador (ver o docstring do módulo).
    assert r.data.params["gate.alfa"] == "0.05"
    # nenhum parâmetro vaza para o lado de métrica.
    assert "semente" not in r.data.metrics


# --- registrar_execucao: PSI por feature é série temporal, não um metric por mês --------


def test_registrar_execucao_grava_psi_por_feature_como_serie_temporal_com_roundtrip(tmp_path):
    drift_por_lote = [
        DriftReport(lote="mes_01", features=(_feature("MonthlyIncome", 0.01),)),
        DriftReport(lote="mes_03", features=(_feature("MonthlyIncome", 0.05),)),
        DriftReport(lote="mes_06", features=(_feature("MonthlyIncome", 0.26),)),
    ]
    gate = avaliar_gate(drift_por_lote)

    run_id = registrar_execucao(
        tracking_uri=_uri(tmp_path),
        experimento="teste-psi",
        parametros={},
        metricas_do_campeao_por_lote={},
        drift_por_lote=drift_por_lote,
        proxies_por_lote={},
        gate=gate,
        artifact_location=str(tmp_path / "artefatos"),
    )

    client = MlflowClient(tracking_uri=_uri(tmp_path))

    historico = client.get_metric_history(run_id, "drift.psi.MonthlyIncome")
    pontos = sorted((ponto.step, ponto.value) for ponto in historico)
    assert pontos == [(1, 0.01), (3, 0.05), (6, 0.26)]

    # `.data.metrics` (via `get_run`) só devolve o último valor por nome — a série
    # completa exige `get_metric_history` (confirmado rodando a biblioteca real, ver o
    # docstring do módulo). Este teste prova as duas metades da mesma afirmação: o
    # histórico completo sobrevive, e o atalho de conveniência não o substitui.
    r = client.get_run(run_id)
    assert r.data.metrics["drift.psi.MonthlyIncome"] == 0.26


# --- registrar_execucao: métricas do campeão por lote também são série temporal ---------


def test_registrar_execucao_grava_metricas_do_campeao_por_lote_com_step(tmp_path):
    gate = avaliar_gate([DriftReport(lote="mes_01", features=(_feature("age", 0.01),))])

    run_id = registrar_execucao(
        tracking_uri=_uri(tmp_path),
        experimento="teste-campeao",
        parametros={},
        metricas_do_campeao_por_lote={
            "mes_01": {"auc_pr": 0.3716},
            "mes_02": {"auc_pr": 0.3502},
        },
        drift_por_lote=[],
        proxies_por_lote={},
        gate=gate,
        artifact_location=str(tmp_path / "artefatos"),
    )

    client = MlflowClient(tracking_uri=_uri(tmp_path))
    historico = client.get_metric_history(run_id, "campeao.auc_pr")
    pontos = sorted((ponto.step, ponto.value) for ponto in historico)
    assert pontos == [(1, 0.3716), (2, 0.3502)]


# --- registrar_execucao: os proxies sem rótulo também são série temporal ----------------


def test_registrar_execucao_grava_proxies_por_lote_com_step(tmp_path):
    gate = avaliar_gate([DriftReport(lote="mes_01", features=(_feature("age", 0.01),))])

    run_id = registrar_execucao(
        tracking_uri=_uri(tmp_path),
        experimento="teste-proxies",
        parametros={},
        metricas_do_campeao_por_lote={},
        drift_por_lote=[],
        proxies_por_lote={
            "mes_01": {"confianca_media": 0.91, "taxa_de_aprovacao": 0.88},
            "mes_02": {"confianca_media": 0.84, "taxa_de_aprovacao": 0.80},
        },
        gate=gate,
        artifact_location=str(tmp_path / "artefatos"),
    )

    client = MlflowClient(tracking_uri=_uri(tmp_path))

    confianca = sorted(
        (p.step, p.value) for p in client.get_metric_history(run_id, "proxy.confianca_media")
    )
    aprovacao = sorted(
        (p.step, p.value) for p in client.get_metric_history(run_id, "proxy.taxa_de_aprovacao")
    )
    assert confianca == [(1, 0.91), (2, 0.84)]
    assert aprovacao == [(1, 0.88), (2, 0.80)]


# --- registrar_execucao: o veredito do gate é tag (busca) e métrica (numérico) ----------


def test_registrar_execucao_grava_veredito_do_gate_como_tag_e_metrica_numerica(tmp_path):
    # Dois lotes com PSI acima de `psi_critico` (0,25) garantem severidade CRITICO sem
    # depender de nenhuma calibração de KS/Benjamini-Hochberg — o próprio PSI já satura a
    # banda mais alta em `classificar` (ver `drift/base.py`).
    drift_por_lote = [DriftReport(lote="mes_01", features=(_feature("DebtRatio", 0.30, 0.001),))]
    gate = avaliar_gate(drift_por_lote)
    assert gate.severidade is Severidade.CRITICO  # a fixture faz o que a asserção espera

    run_id = registrar_execucao(
        tracking_uri=_uri(tmp_path),
        experimento="teste-gate",
        parametros={},
        metricas_do_campeao_por_lote={},
        drift_por_lote=drift_por_lote,
        proxies_por_lote={},
        gate=gate,
        artifact_location=str(tmp_path / "artefatos"),
    )

    client = MlflowClient(tracking_uri=_uri(tmp_path))
    r = client.get_run(run_id)

    assert r.data.tags["gate.veredito"] == "CRITICO"
    assert r.data.metrics["gate.severidade"] == float(Severidade.CRITICO.value)


# --- registrar_execucao: artefatos (HTML do Evidently, monitoramento.json) --------------


def test_registrar_execucao_grava_artefatos_dentro_do_tmp_path(tmp_path):
    gate = avaliar_gate([DriftReport(lote="mes_01", features=(_feature("age", 0.01),))])

    html = tmp_path / "drift_mes_01.html"
    html.write_text("<html></html>", encoding="utf-8")
    relatorio = tmp_path / "monitoramento.json"
    relatorio.write_text("{}", encoding="utf-8")

    run_id = registrar_execucao(
        tracking_uri=_uri(tmp_path),
        experimento="teste-artefatos",
        parametros={},
        metricas_do_campeao_por_lote={},
        drift_por_lote=[],
        proxies_por_lote={},
        gate=gate,
        artefatos=[html, relatorio],
        artifact_location=str(tmp_path / "artefatos"),
    )

    client = MlflowClient(tracking_uri=_uri(tmp_path))
    r = client.get_run(run_id)

    nomes = {info.path for info in client.list_artifacts(run_id)}
    assert nomes == {"drift_mes_01.html", "monitoramento.json"}
    # Sem `artifact_location` explícito, o MLflow grava em `./mlruns` relativo ao
    # diretório de trabalho do processo — medido rodando a biblioteca real (ver o
    # docstring do módulo) — e um teste que não fixasse isso vazaria arquivo para dentro
    # do repositório. Confirma que o artefato de fato ficou dentro de `tmp_path`.
    assert r.info.artifact_uri.startswith(str(tmp_path))


# --- registrar_execucao: nome de métrica nunca sai de um conjunto fechado ---------------


def test_registrar_execucao_rejeita_feature_fora_de_credito_schema_features(tmp_path):
    drift_por_lote = [DriftReport(lote="mes_01", features=(_feature("renda_do_cliente", 0.01),))]
    gate = avaliar_gate(drift_por_lote)

    with pytest.raises(ValueError, match="FEATURES"):
        registrar_execucao(
            tracking_uri=_uri(tmp_path),
            experimento="teste-feature-invalida",
            parametros={},
            metricas_do_campeao_por_lote={},
            drift_por_lote=drift_por_lote,
            proxies_por_lote={},
            gate=gate,
            artifact_location=str(tmp_path / "artefatos"),
        )

    # A validação roda ANTES de abrir o run — nenhum run parcial fica para trás.
    client = MlflowClient(tracking_uri=_uri(tmp_path))
    experimento = client.get_experiment_by_name("teste-feature-invalida")
    assert experimento is None


def test_registrar_execucao_rejeita_proxy_fora_do_conjunto_conhecido(tmp_path):
    gate = avaliar_gate([DriftReport(lote="mes_01", features=(_feature("age", 0.01),))])

    with pytest.raises(ValueError, match="sinais_do_lote"):
        registrar_execucao(
            tracking_uri=_uri(tmp_path),
            experimento="teste-proxy-invalido",
            parametros={},
            metricas_do_campeao_por_lote={},
            drift_por_lote=[],
            proxies_por_lote={"mes_01": {"renda_media_do_lote": 500.0}},
            gate=gate,
            artifact_location=str(tmp_path / "artefatos"),
        )

    client = MlflowClient(tracking_uri=_uri(tmp_path))
    experimento = client.get_experiment_by_name("teste-proxy-invalido")
    assert experimento is None


def test_registrar_execucao_rejeita_metrica_de_campeao_fora_de_metricas_globais(tmp_path):
    gate = avaliar_gate([DriftReport(lote="mes_01", features=(_feature("age", 0.01),))])

    with pytest.raises(ValueError, match="avaliar"):
        registrar_execucao(
            tracking_uri=_uri(tmp_path),
            experimento="teste-metrica-campeao-invalida",
            parametros={},
            metricas_do_campeao_por_lote={
                "mes_01": {"nome_completamente_livre_do_usuario_123": 0.5}
            },
            drift_por_lote=[],
            proxies_por_lote={},
            gate=gate,
            artifact_location=str(tmp_path / "artefatos"),
        )

    # A mesma garantia dos dois testes acima: a validação roda ANTES de abrir o run —
    # nenhum run parcial fica para trás.
    client = MlflowClient(tracking_uri=_uri(tmp_path))
    experimento = client.get_experiment_by_name("teste-metrica-campeao-invalida")
    assert experimento is None
