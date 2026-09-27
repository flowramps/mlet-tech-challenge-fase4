"""Acurácia é inútil aqui: prever "adimplente" para todo mundo dá 93%. As métricas que
importam são AUC-PR e o recall da classe positiva — o inadimplente que passa é o erro caro.

O recorte por faixa etária existe porque idade é atributo protegido: sem medir por grupo,
não há como afirmar nada sobre viés."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from credito.model.evaluate import (
    FAIXAS_ETARIAS,
    avaliar,
    avaliar_por_faixa_etaria,
    salvar_metricas,
)
from credito.schema import ALVO, FEATURES


class _ModeloFixo:
    """Devolve a probabilidade que lhe mandarem — isola a avaliação do treino."""

    def __init__(self, probabilidades: np.ndarray) -> None:
        self._probabilidades = probabilidades

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        recorte = self._probabilidades[: len(entradas)]
        return np.column_stack([1 - recorte, recorte])


def _frame(alvo: list[int], idades: list[int] | None = None) -> pd.DataFrame:
    n = len(alvo)
    dados = {coluna: [0.5] * n for coluna in FEATURES}
    dados["age"] = idades if idades is not None else [40] * n
    dados[ALVO] = alvo
    return pd.DataFrame(dados)[[ALVO, *FEATURES]]


def test_modelo_perfeito_tem_auc_pr_um():
    frame = _frame([0, 0, 1, 1])
    modelo = _ModeloFixo(np.array([0.01, 0.02, 0.98, 0.99]))

    metricas = avaliar(modelo, frame)

    assert metricas["auc_pr"] == pytest.approx(1.0)
    assert metricas["auc_roc"] == pytest.approx(1.0)
    assert metricas["recall_positivo"] == pytest.approx(1.0)


def test_modelo_que_nunca_acusa_tem_recall_zero():
    # É este o modelo que a acurácia elogiaria.
    frame = _frame([0] * 9 + [1])
    modelo = _ModeloFixo(np.full(10, 0.01))

    metricas = avaliar(modelo, frame)

    assert metricas["recall_positivo"] == pytest.approx(0.0)
    assert metricas["acuracia"] == pytest.approx(0.9)


def test_metricas_esperadas_estao_presentes():
    metricas = avaliar(_ModeloFixo(np.array([0.2, 0.8])), _frame([0, 1]))

    for chave in ("auc_pr", "auc_roc", "recall_positivo", "precisao_positiva", "acuracia", "n"):
        assert chave in metricas


def test_limiar_desloca_o_recall():
    frame = _frame([0, 1, 1])
    modelo = _ModeloFixo(np.array([0.1, 0.4, 0.6]))

    assert avaliar(modelo, frame, limiar=0.5)["recall_positivo"] == pytest.approx(0.5)
    assert avaliar(modelo, frame, limiar=0.3)["recall_positivo"] == pytest.approx(1.0)


def test_recorte_etario_cobre_as_faixas_presentes():
    frame = _frame([0, 1, 0, 1], idades=[22, 35, 50, 70])
    modelo = _ModeloFixo(np.array([0.1, 0.9, 0.2, 0.8]))

    por_faixa = avaliar_por_faixa_etaria(modelo, frame)

    assert set(por_faixa) <= set(FAIXAS_ETARIAS)
    for faixa in por_faixa.values():
        assert "taxa_de_aprovacao" in faixa
        assert "n" in faixa


def test_faixa_sem_amostra_e_omitida():
    frame = _frame([0, 1], idades=[35, 36])
    modelo = _ModeloFixo(np.array([0.1, 0.9]))

    por_faixa = avaliar_por_faixa_etaria(modelo, frame)

    assert all(faixa["n"] > 0 for faixa in por_faixa.values())
    # As duas idades caem em "26-40": as outras três faixas não têm nenhuma linha e não
    # podem aparecer no resultado como se tivessem sido medidas.
    assert set(por_faixa) == {"26-40"}


def test_faixa_sem_positivo_nao_reporta_recall():
    # Sem nenhum inadimplente na faixa, recall é indefinido — reportar 0.0 seria dizer
    # que o modelo captura "nenhum dos zero" quando na verdade não há nada a capturar.
    frame = _frame([0, 0], idades=[22, 23])
    modelo = _ModeloFixo(np.array([0.1, 0.2]))

    por_faixa = avaliar_por_faixa_etaria(modelo, frame)

    assert "recall_positivo" not in por_faixa["18-25"]


def test_salvar_metricas_grava_json_legivel(tmp_path):
    caminho = tmp_path / "m.json"

    salvar_metricas({"auc_pr": 0.42}, caminho)

    assert json.loads(caminho.read_text(encoding="utf-8"))["auc_pr"] == 0.42
