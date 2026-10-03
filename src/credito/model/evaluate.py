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

from credito.schema import ALVO, FEATURES

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


# As chaves que `avaliar` de fato devolve — conjunto fechado, não uma lista solta: é o que
# `credito.tracking.mlflow_client` importa para validar nome de métrica `campeao.*` antes
# de abrir um run do MLflow, a mesma disciplina de cardinalidade controlada que já vale
# para `credito.schema.FEATURES` (nome de métrica nunca vem de entrada não controlada).
# `tests/model/test_evaluate.py::test_metricas_globais_cobre_exatamente_as_chaves_de_avaliar`
# tranca que esta tupla nunca fique defasada do que `avaliar` de fato devolve.
METRICAS_GLOBAIS: tuple[str, ...] = (
    "auc_pr",
    "auc_roc",
    "recall_positivo",
    "precisao_positiva",
    "acuracia",
    "taxa_de_positivos",
    "limiar",
    "n",
)


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


# Regra dos quatro quintos (Uniform Guidelines on Employee Selection Procedures, EUA,
# 1978): a taxa de seleção de um grupo abaixo de 80% da do grupo mais selecionado é
# tratada como evidência de impacto adverso. É convenção regulatória, não medição deste
# dado — o mesmo status que `psi_atencao`/`psi_critico` declaram em `config.py`.
LIMIAR_QUATRO_QUINTOS = 0.8


def razao_impacto_adverso(por_faixa: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Regra dos 4/5 sobre a taxa de aprovação de cada faixa, contra a mais aprovada.

    Recebe o que `avaliar_por_faixa_etaria` devolve. Mede **desigualdade de desfecho**:
    quanto menos uma faixa é aprovada que a faixa mais aprovada. Não desconta diferença
    de risco real entre as faixas — e é exatamente por isso que esta métrica sozinha não
    basta (ver `diferenca_de_oportunidade`).
    """
    if not por_faixa:
        raise ValueError("nenhuma faixa para comparar")
    referencia = max(por_faixa, key=lambda nome: por_faixa[nome]["taxa_de_aprovacao"])
    taxa_referencia = por_faixa[referencia]["taxa_de_aprovacao"]
    if taxa_referencia <= 0:
        raise ValueError("nenhuma faixa aprova ninguém: a razão é indefinida")

    faixas = {}
    for nome, faixa in por_faixa.items():
        razao = faixa["taxa_de_aprovacao"] / taxa_referencia
        faixas[nome] = {"razao": razao, "impacto_adverso": bool(razao < LIMIAR_QUATRO_QUINTOS)}
    return {"referencia": referencia, "limiar": LIMIAR_QUATRO_QUINTOS, "faixas": faixas}


def diferenca_de_oportunidade(por_faixa: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Distância entre o recall de cada faixa e o maior recall entre as faixas.

    Mede **desigualdade de erro**: entre os que de fato inadimpliram, quantos o modelo
    deixa passar em cada grupo. Ao contrário da regra dos 4/5, condiciona no desfecho
    real — então uma faixa pode ser a mais aprovada e, ao mesmo tempo, aquela em que o
    modelo menos enxerga o risco.

    Faixa sem `recall_positivo` (nenhum inadimplente real nela) fica fora: a métrica é
    indefinida ali, e zerá-la faria a faixa parecer a mais prejudicada. Não há corte
    regulatório consolidado para esta diferença, então nenhum é declarado aqui.
    """
    com_recall = {nome: f for nome, f in por_faixa.items() if "recall_positivo" in f}
    if not com_recall:
        raise ValueError("nenhuma faixa tem recall definido")
    referencia = max(com_recall, key=lambda nome: com_recall[nome]["recall_positivo"])
    recall_referencia = com_recall[referencia]["recall_positivo"]

    faixas = {
        nome: {
            "recall_positivo": faixa["recall_positivo"],
            "diferenca": recall_referencia - faixa["recall_positivo"],
        }
        for nome, faixa in com_recall.items()
    }
    return {"referencia": referencia, "faixas": faixas}


def salvar_metricas(metricas: dict[str, Any], caminho: Path) -> None:
    """Grava as métricas em JSON legível por humano e por máquina."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
