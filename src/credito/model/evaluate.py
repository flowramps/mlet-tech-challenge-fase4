"""Avaliação do modelo de risco, com o recorte por grupo que a análise de viés exige.

A escolha das métricas é consequência do desbalanceamento: com 6,94% de positivos, a
acurácia premia o modelo que nunca acusa ninguém. ``auc_pr`` mede a qualidade do
ordenamento onde a classe rara importa, e ``recall_positivo`` é a métrica de negócio —
o inadimplente aprovado custa o valor do empréstimo, o adimplente recusado custa a margem.
``auc_roc`` fica como figura secundária no model card: infla com o desbalanceamento e por
isso não é a métrica que decide nada aqui.

``avaliar_por_faixa_etaria`` existe porque idade é atributo protegido: afirmar que o
modelo não discrimina sem medir por grupo é afirmação sem evidência.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credito.data.prepare import ALVO, FEATURES

logger = logging.getLogger(__name__)

# Faixas do crédito ao consumidor: o jovem sem histórico e o idoso próximo da
# aposentadoria são os dois grupos em que um modelo de risco costuma ser mais severo.
FAIXAS_ETARIAS: dict[str, tuple[int, int]] = {
    "18-25": (18, 25),
    "26-40": (26, 40),
    "41-60": (41, 60),
    "61+": (61, 200),
}


def _probabilidades(modelo: Any, frame: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(frame[list(FEATURES)])[:, 1]


def avaliar(modelo: Any, frame: pd.DataFrame, *, limiar: float = 0.5) -> dict[str, float]:
    """Calcula as métricas globais do modelo sobre ``frame``."""
    verdadeiro = frame[ALVO].to_numpy()
    probabilidade = _probabilidades(modelo, frame)
    previsto = (probabilidade >= limiar).astype(int)

    metricas = {
        "auc_pr": float(average_precision_score(verdadeiro, probabilidade)),
        "auc_roc": float(roc_auc_score(verdadeiro, probabilidade)),
        "recall_positivo": float(recall_score(verdadeiro, previsto, zero_division=0)),
        "precisao_positiva": float(precision_score(verdadeiro, previsto, zero_division=0)),
        "acuracia": float(accuracy_score(verdadeiro, previsto)),
        "taxa_de_positivos": float(verdadeiro.mean()),
        "limiar": float(limiar),
        "n": int(len(frame)),
    }
    logger.info(
        "auc_pr=%.4f auc_roc=%.4f recall+=%.4f",
        metricas["auc_pr"],
        metricas["auc_roc"],
        metricas["recall_positivo"],
    )
    return metricas


def avaliar_por_faixa_etaria(
    modelo: Any, frame: pd.DataFrame, *, limiar: float = 0.5
) -> dict[str, dict[str, float]]:
    """Mede o comportamento do modelo dentro de cada faixa etária.

    ``taxa_de_aprovacao`` é a fração de clientes da faixa que o modelo **não** classifica
    como risco. Comparar essa taxa entre faixas é a base do *disparate impact*, que a
    documentação de governança usa para discutir viés com número em vez de adjetivo.

    Uma faixa sem nenhuma linha é omitida do resultado — reportar ``n=0`` como se fosse
    uma medição seria inventar um dado. Dentro de uma faixa presente, ``recall_positivo``
    só aparece se houver ao menos um inadimplente de verdade: sem positivo, a métrica é
    indefinida, e emitir 0.0 nesse caso sugeriria uma falha que não existe.
    """
    probabilidade = _probabilidades(modelo, frame)
    previsto = (probabilidade >= limiar).astype(int)
    trabalho = frame.assign(_previsto=previsto, _probabilidade=probabilidade)

    resultado: dict[str, dict[str, float]] = {}
    for nome, (minimo, maximo) in FAIXAS_ETARIAS.items():
        grupo = trabalho[trabalho["age"].between(minimo, maximo)]
        if grupo.empty:
            continue
        verdadeiro = grupo[ALVO].to_numpy()
        entrada = {
            "n": int(len(grupo)),
            "taxa_de_positivos_real": float(verdadeiro.mean()),
            "taxa_de_recusa": float(grupo["_previsto"].mean()),
            "taxa_de_aprovacao": float(1 - grupo["_previsto"].mean()),
            "probabilidade_media": float(grupo["_probabilidade"].mean()),
        }
        # Recall só é definível se a faixa tem algum positivo de verdade.
        if verdadeiro.sum() > 0:
            entrada["recall_positivo"] = float(
                recall_score(verdadeiro, grupo["_previsto"], zero_division=0)
            )
        resultado[nome] = entrada
    return resultado


def salvar_metricas(metricas: dict[str, Any], caminho: Path) -> None:
    """Grava as métricas em JSON legível por humano e por máquina."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
