"""``EvidentlyDetector`` tem duas responsabilidades e este arquivo testa as duas:

1. Gerar o relatório HTML — o entregável nominal do enunciado —, e
2. Devolver um ``DriftReport`` que **concorda em veredicto** com ``drift/statistics.py``.

O Evidently faz o próprio binning internamente (não expõe os cortes), então o PSI que ele
devolve e o PSI de ``credito.drift.statistics.psi`` **não batem em valor** — ver o
docstring de ``credito.drift.evidently_backend`` sobre o mecanismo. O que os testes abaixo
comparam é o veredicto (``Severidade``) nos dois extremos inequívocos: dados idênticos (os
dois têm que dizer ESTAVEL) e deslocamento de 20 unidades numa variável cujo suporte real
é ``[0, 1]`` (os dois têm que dizer CRITICO). Fora desses dois extremos as duas
implementações podem divergir até em banda de severidade — o docstring do módulo de
produção traz o caso medido, em variável de cauda longa —, e nenhum teste aqui pisa nessa
faixa.

Nenhum teste toca rede: o fixture ``_bloqueia_rede`` troca ``socket.socket.connect`` por
uma função que derruba o teste — se o Evidently (ou a telemetria opcional que ele carrega,
``iterative-telemetry``) tentasse abrir uma conexão real durante ``Report.run()`` ou
``save_html()``, o teste falharia por isso, não passaria em silêncio.
"""

from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest

from credito.config import get_settings
from credito.drift.base import DriftDetector, DriftReport, Severidade
from credito.drift.evidently_backend import EvidentlyDetector, _extrair_metricas, construir_detector
from credito.drift.statistics import ks, psi
from credito.schema import FEATURES

N = 2000
SEED = 11


@pytest.fixture(autouse=True)
def _bloqueia_rede(monkeypatch):
    def _recusa(*_args, **_kwargs):
        raise AssertionError(
            "teste de drift tentou abrir uma conexão de rede — Evidently roda só localmente"
        )

    monkeypatch.setattr(socket.socket, "connect", _recusa)
    monkeypatch.setattr(socket.socket, "connect_ex", _recusa)


def _dataframe_base(n: int, seed: int) -> pd.DataFrame:
    """Um DataFrame sintético com todas as ``FEATURES`` do projeto, plausível o bastante
    (contagens em faixas pequenas, ``DebtRatio``/utilização em ``[0, 1]``) sem depender do
    arquivo real do dataset — nenhum teste deste módulo pode tocar disco de dado nem rede."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0.0, 1.0, n),
            "age": rng.normal(45.0, 12.0, n),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 3, n).astype(float),
            "DebtRatio": rng.uniform(0.0, 1.0, n),
            "MonthlyIncome": rng.normal(5000.0, 1500.0, n),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 15, n).astype(float),
            "NumberOfTimes90DaysLate": rng.integers(0, 2, n).astype(float),
            "NumberRealEstateLoansOrLines": rng.integers(0, 4, n).astype(float),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 2, n).astype(float),
            "NumberOfDependents": rng.integers(0, 4, n).astype(float),
        }
    )


def _relatorio_proprio(referencia: pd.DataFrame, atual: pd.DataFrame, *, lote: str) -> DriftReport:
    """Monta o ``DriftReport`` de referência usando ``drift/statistics.py`` diretamente —
    o "gabarito" contra o qual o veredicto do Evidently é comparado nos testes abaixo."""
    from credito.drift.base import DriftDeFeature, classificar

    features = []
    for feature in FEATURES:
        psi_valor = psi(referencia[feature].to_numpy(), atual[feature].to_numpy())
        _, p_valor = ks(referencia[feature].to_numpy(), atual[feature].to_numpy())
        features.append(
            DriftDeFeature(
                feature=feature,
                psi_divergencia=psi_valor,
                ks_p_valor=p_valor,
                severidade=classificar(psi_valor),
            )
        )
    return DriftReport(lote=lote, features=tuple(features))


# --- construir_detector: fábrica que devolve algo que satisfaz DriftDetector -------------


def test_construir_detector_usa_o_diretorio_de_relatorios_da_configuracao():
    detector = construir_detector()

    assert isinstance(detector, EvidentlyDetector)
    assert detector.diretorio_relatorios == get_settings().reports_dir


def test_evidently_detector_satisfaz_o_protocolo_driftdetector():
    # `DriftDetector` (drift/base.py) não é `@runtime_checkable` — `isinstance` contra ele
    # ergueria `TypeError`, não `False`. A checagem estrutural que resta, e que de fato
    # importa aqui, é a assinatura: um `detectar(referencia, atual, *, lote=...)` chamável
    # com os tipos que o Protocol declara.
    detector: DriftDetector = EvidentlyDetector(diretorio_relatorios=get_settings().reports_dir)

    assert callable(detector.detectar)


# --- Veredicto: dados idênticos --------------------------------------------------------


def test_dados_identicos_evidently_e_estatistica_propria_concordam_em_estavel(tmp_path):
    referencia = _dataframe_base(N, SEED)
    atual = referencia.copy()

    detector = EvidentlyDetector(diretorio_relatorios=tmp_path)
    relatorio_evidently = detector.detectar(referencia, atual, lote="mes_00")
    relatorio_proprio = _relatorio_proprio(referencia, atual, lote="mes_00")

    assert relatorio_evidently.severidade_maxima is Severidade.ESTAVEL
    assert relatorio_proprio.severidade_maxima is Severidade.ESTAVEL


# --- Veredicto: deslocamento forte e inequívoco -----------------------------------------


def test_deslocamento_forte_evidently_e_estatistica_propria_concordam_em_critico(tmp_path):
    # DebtRatio mora em [0, 1] nesta amostra sintética; deslocar tudo em +20 tira a atual
    # inteira do suporte da referência — o mesmo tipo de deslocamento de 5 desvios-padrão
    # que test_statistics.py usa para o mesmo propósito (PSI >= 0,25 sem ambiguidade).
    # As outras nove features ficam idênticas, isoladando DebtRatio como a única fonte de
    # severidade — o que a asserção sobre `features_em_drift` abaixo confirma.
    referencia = _dataframe_base(N, SEED)
    atual = referencia.copy()
    atual["DebtRatio"] = atual["DebtRatio"] + 20.0

    detector = EvidentlyDetector(diretorio_relatorios=tmp_path)
    relatorio_evidently = detector.detectar(referencia, atual, lote="mes_06")
    relatorio_proprio = _relatorio_proprio(referencia, atual, lote="mes_06")

    assert relatorio_evidently.severidade_maxima is Severidade.CRITICO
    assert relatorio_proprio.severidade_maxima is Severidade.CRITICO

    nomes_em_drift_evidently = {f.feature for f in relatorio_evidently.features_em_drift}
    nomes_em_drift_proprio = {f.feature for f in relatorio_proprio.features_em_drift}
    assert "DebtRatio" in nomes_em_drift_evidently
    assert "DebtRatio" in nomes_em_drift_proprio


def test_relatorio_devolvido_carrega_o_lote_passado(tmp_path):
    referencia = _dataframe_base(200, SEED)
    atual = referencia.copy()
    detector = EvidentlyDetector(diretorio_relatorios=tmp_path)

    relatorio = detector.detectar(referencia, atual, lote="mes_03")

    assert relatorio.lote == "mes_03"


# --- O entregável nominal: o HTML aponta a variável degradada --------------------------


def test_gera_html_no_diretorio_configurado_apontando_a_variavel_em_drift(tmp_path):
    referencia = _dataframe_base(N, SEED)
    atual = referencia.copy()
    atual["DebtRatio"] = atual["DebtRatio"] + 20.0

    detector = EvidentlyDetector(diretorio_relatorios=tmp_path)
    detector.detectar(referencia, atual, lote="mes_06")

    caminho_html = tmp_path / "drift_mes_06.html"
    assert caminho_html.exists()
    assert caminho_html.stat().st_size > 0

    conteudo = caminho_html.read_text(encoding="utf-8")
    assert "DebtRatio" in conteudo


def test_colunas_fora_de_features_nao_aparecem_no_relatorio_nem_no_html(tmp_path):
    # Medido diretamente contra o Evidently: uma coluna não declarada em
    # `numerical_columns` (ex.: o alvo, se o chamador passasse o DataFrame inteiro) ainda
    # assim é autodetectada por `DataDefinition` e ganha seu próprio `ValueDrift` no
    # preset — não é ignorada por omissão. `EvidentlyDetector.detectar` restringe
    # `referencia`/`atual` a `colunas = list(FEATURES)` antes de montar o `Dataset`
    # exatamente por isso: sem essa restrição, uma coluna estranha ao vocabulário de
    # features apareceria monitorada no HTML (mesmo que `_extrair_metricas`, por só
    # iterar as `FEATURES` pedidas, já impeça essa coluna de entrar no `DriftReport`).
    referencia = _dataframe_base(N, SEED)
    atual = referencia.copy()
    referencia["inadimplente"] = 0
    atual["inadimplente"] = 1

    detector = EvidentlyDetector(diretorio_relatorios=tmp_path)
    relatorio = detector.detectar(referencia, atual, lote="mes_02")

    assert {f.feature for f in relatorio.features} == set(FEATURES)
    conteudo = (tmp_path / "drift_mes_02.html").read_text(encoding="utf-8")
    assert "inadimplente" not in conteudo


def test_cria_o_diretorio_de_relatorios_quando_nao_existe(tmp_path):
    diretorio = tmp_path / "ainda_nao_existe"
    referencia = _dataframe_base(200, SEED)
    atual = referencia.copy()

    detector = EvidentlyDetector(diretorio_relatorios=diretorio)
    detector.detectar(referencia, atual, lote="mes_01")

    assert (diretorio / "drift_mes_01.html").exists()


# --- _extrair_metricas: o parser do dict() do Evidently, isolado do Report.run() -------


def _metrica_value_drift(coluna: str, metodo: str, valor) -> dict:
    return {
        "id": "x",
        "metric_name": f"ValueDrift(column={coluna},method={metodo})",
        "config": {"type": "evidently:metric_v2:ValueDrift", "column": coluna, "method": metodo},
        "value": valor,
    }


def _metrica_drifted_columns_count() -> dict:
    return {
        "id": "y",
        "metric_name": "DriftedColumnsCount(drift_share=0.5,method=psi)",
        "config": {
            "type": "evidently:metric_v2:DriftedColumnsCount",
            "drift_share": 0.5,
            "method": "psi",
        },
        "value": {"count": 1.0, "share": 0.5},
    }


def test_extrair_metricas_junta_psi_e_ks_por_feature_sem_trocar_metodo():
    resultado = {
        "metrics": [
            _metrica_drifted_columns_count(),
            _metrica_value_drift("age", "psi", 0.03),
            _metrica_value_drift("age", "ks", 0.9),
        ],
        "tests": [],
    }

    coletado = _extrair_metricas(resultado, ("age",))

    assert coletado == {"age": {"psi": 0.03, "ks": 0.9}}


def test_extrair_metricas_ignora_metricas_de_outros_tipos():
    # DriftedColumnsCount tem "value" como dict, não float — se o parser tentasse tratar
    # todo metrics[] como ValueDrift, estouraria aqui em vez de ignorar corretamente.
    resultado = {
        "metrics": [
            _metrica_drifted_columns_count(),
            _metrica_value_drift("age", "psi", 0.03),
            _metrica_value_drift("age", "ks", 0.9),
        ],
        "tests": [],
    }

    coletado = _extrair_metricas(resultado, ("age",))

    assert "count" not in coletado.get("age", {})


def test_extrair_metricas_nao_deixa_metrica_de_outro_tipo_sobrescrever_o_valor_real():
    # A entrada intrusa vem *depois* da ValueDrift legítima e tem column="age",
    # method="psi" — mesma coluna, mesmo método, tipo diferente. Se o filtro por
    # `config["type"]` fosse removido, o dict comprehension processaria as duas em ordem
    # e a intrusa (999.0) sobrescreveria o valor real (0.03) por último. A versão anterior
    # deste teste (`test_extrair_metricas_ignora_metricas_de_outros_tipos`) não pegava essa
    # troca: como `DriftedColumnsCount` não carrega "column", o check de coluna já a
    # descartava sozinho, e o teste passava mesmo com o filtro de tipo apagado — achado do
    # sweep de mutação.
    resultado = {
        "metrics": [
            _metrica_value_drift("age", "psi", 0.03),
            {
                "id": "z",
                "metric_name": "OutroTipo(column=age,method=psi)",
                "config": {
                    "type": "evidently:metric_v2:OutroTipo",
                    "column": "age",
                    "method": "psi",
                },
                "value": 999.0,
            },
            _metrica_value_drift("age", "ks", 0.9),
        ],
        "tests": [],
    }

    coletado = _extrair_metricas(resultado, ("age",))

    assert coletado == {"age": {"psi": 0.03, "ks": 0.9}}


def test_extrair_metricas_reclama_de_feature_sem_metrica_devolvida():
    resultado = {"metrics": [_metrica_value_drift("age", "psi", 0.03)], "tests": []}

    with pytest.raises(ValueError, match="DebtRatio"):
        _extrair_metricas(resultado, ("age", "DebtRatio"))


def test_extrair_metricas_reclama_de_feature_com_so_metade_do_par():
    # "age" tem psi mas não ks — o parser não pode devolver silenciosamente um par
    # incompleto que quebraria mais adiante dentro de DriftDeFeature.
    resultado = {"metrics": [_metrica_value_drift("age", "psi", 0.03)], "tests": []}

    with pytest.raises(ValueError, match="age"):
        _extrair_metricas(resultado, ("age",))


def test_extrair_metricas_reclama_de_valor_nao_numerico():
    resultado = {
        "metrics": [
            _metrica_value_drift("age", "psi", {"inesperado": True}),
            _metrica_value_drift("age", "ks", 0.9),
        ],
        "tests": [],
    }

    with pytest.raises(ValueError, match="age"):
        _extrair_metricas(resultado, ("age",))
