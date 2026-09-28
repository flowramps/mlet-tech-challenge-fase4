"""Treino dos candidatos a modelo de risco de crédito.

Dois candidatos, não um: a comparação entre uma linha de base linear e um ensemble de
árvores transforma a escolha do modelo em evidência em vez de preferência, e o custo é uma
chamada a mais no pipeline.

Ambos recebem tratamento explícito de desbalanceamento. Com 6,94% de positivos, um
classificador sem esse tratamento converge para prever a classe majoritária para todo
mundo: acerta 93% e não identifica nenhum inadimplente — que é a única coisa que se quer
dele.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credito.schema import ALVO, FEATURES

logger = logging.getLogger(__name__)

TIPOS_DE_MODELO: tuple[str, ...] = ("regressao_logistica", "xgboost")


def _peso_da_classe_positiva(alvo: pd.Series) -> float:
    positivos = int(alvo.sum())
    negativos = len(alvo) - positivos
    if positivos == 0:
        raise ValueError("o conjunto de treino não tem nenhum positivo")
    return negativos / positivos


def treinar(frame: pd.DataFrame, tipo: str, *, seed: int) -> Pipeline:
    """Treina um candidato e devolve o pipeline completo, pronto para servir."""
    if tipo not in TIPOS_DE_MODELO:
        raise ValueError(f"tipo de modelo desconhecido: {tipo}")

    entradas = frame[list(FEATURES)]
    alvo = frame[ALVO]

    if tipo == "regressao_logistica":
        # A imputação e a padronização viajam dentro do pipeline: quem serve o modelo não
        # pode precisar lembrar de repetir o pré-processamento, senão a inferência
        # diverge do treino em silêncio.
        modelo = Pipeline(
            [
                ("imputacao", SimpleImputer(strategy="median")),
                ("escala", StandardScaler()),
                (
                    "classificador",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        # O XGBoost lida com ausência nativamente, então não leva imputação; a escala é
        # irrelevante para árvores.
        modelo = Pipeline(
            [
                (
                    "classificador",
                    XGBClassifier(
                        # Padrões conservadores de baseline, não valores otimizados: a
                        # comparação com a regressão logística só é evidência se o
                        # ensemble entrar como candidato honesto, não como vencedor
                        # ajustado a dedo. n_estimators/max_depth moderados evitam
                        # memorizar o treino; subsample/colsample_bytree abaixo de 1
                        # dão alguma resistência a overfitting sem exigir busca de
                        # hiperparâmetros nesta etapa.
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=_peso_da_classe_positiva(alvo),
                        eval_metric="aucpr",
                        random_state=seed,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                )
            ]
        )

    modelo.fit(entradas, alvo)
    logger.info("candidato %s treinado sobre %d linhas", tipo, len(frame))
    return modelo


def salvar_modelo(modelo: Pipeline, caminho: Path, metadados: dict[str, Any]) -> None:
    """Grava o pipeline junto dos metadados que descrevem como ele foi obtido."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "metadados": metadados}, caminho)


def carregar_modelo(caminho: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Lê o pipeline e seus metadados."""
    pacote = joblib.load(Path(caminho))
    return pacote["modelo"], pacote["metadados"]
