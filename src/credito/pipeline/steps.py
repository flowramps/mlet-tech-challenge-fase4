"""Etapas do pipeline de treino.

Cada etapa recebe e devolve apenas tipos serializáveis (``str``, ``float``, ``dict``).
Isso não é preciosismo: é o que permite encadeá-las como tarefas de um orquestrador, onde
o valor trafega serializado e um ``Path`` não sobreviveria, e ao mesmo tempo testá-las sem
orquestrador nenhum instalado. Artefatos grandes vão para disco; só os caminhos trafegam.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Candidato reprovado num piso absoluto de qualidade.

    Sinaliza que algo está errado com o treino: o modelo não é utilizável. O run do
    pipeline deve **falhar** — alguém precisa investigar.
    """


# O sufixo `Error` que a N818 pede é justamente o que esta classe não pode ter: ela
# sinaliza um desfecho normal do pipeline, e o nome é metade da distinção que este módulo
# faz questão de manter. Regra dispensada de propósito, não por descuido.
class ModelNotPromoted(RuntimeError):  # noqa: N818
    """Candidato utilizável, porém não melhor que o modelo já em produção.

    Deliberadamente **não** é subclasse de :class:`QualityGateError`: é um desfecho normal,
    não um defeito. Sobre um dataset estático o retreino reproduz o incumbente, então esta
    é a saída esperada de toda execução periódica depois da primeira — tratá-la como falha
    deixaria o pipeline vermelho para sempre. A orquestração converte isto em *skip*.
    """


def motivos_de_piso_absoluto(
    candidato: dict[str, float],
    *,
    min_auc_pr: float,
    min_recall_positivo: float,
) -> list[str]:
    """Critérios de piso absoluto que o candidato não cumpriu.

    Separada de :func:`motivos_de_regressao` porque as duas listas levam a desfechos
    opostos — esta vira :class:`QualityGateError`, a outra vira :class:`ModelNotPromoted` —
    e essa decisão precisa ser tomada sobre **qual lista** trouxe o motivo, nunca sobre o
    texto dele. Um critério futuro de drift cuja frase também dissesse "abaixo do piso"
    transformaria um desfecho normal numa falha do run, e nada quebraria para avisar.
    """
    motivos: list[str] = []

    if candidato["auc_pr"] < min_auc_pr:
        motivos.append(f"auc_pr {candidato['auc_pr']:.4f} abaixo do piso {min_auc_pr:.4f}")
    if candidato["recall_positivo"] < min_recall_positivo:
        motivos.append(
            f"recall_positivo {candidato['recall_positivo']:.4f} abaixo do piso "
            f"{min_recall_positivo:.4f}"
        )

    return motivos


def motivos_de_regressao(
    candidato: dict[str, float],
    incumbente: dict[str, float] | None,
) -> list[str]:
    """Critérios de não regressão frente ao modelo em produção que o candidato não cumpriu.

    Sem incumbente não há com o que regredir, e a lista sai vazia — é o caso da primeira
    execução. Reprovar aqui não é defeito: o candidato é utilizável, só não é melhor.
    """
    if incumbente is None:
        return []

    motivos: list[str] = []

    if candidato["auc_pr"] <= incumbente["auc_pr"]:
        motivos.append(
            f"auc_pr {candidato['auc_pr']:.4f} não supera o modelo em produção "
            f"({incumbente['auc_pr']:.4f})"
        )
    if candidato["recall_positivo"] < incumbente["recall_positivo"]:
        motivos.append(
            f"recall_positivo {candidato['recall_positivo']:.4f} regride em relação ao "
            f"modelo em produção ({incumbente['recall_positivo']:.4f})"
        )

    return motivos


def motivos_de_reprovacao(
    candidato: dict[str, float],
    incumbente: dict[str, float] | None,
    *,
    min_auc_pr: float,
    min_recall_positivo: float,
) -> list[str]:
    """Lista, em texto, cada critério do gate que o candidato não cumpriu.

    A concatenação das duas listas, na ordem em que o histórico as registra. Quem precisa
    **decidir** entre falhar e pular chama as duas separadamente; esta existe para quem só
    precisa do relato completo, como o registro de auditoria.
    """
    return [
        *motivos_de_piso_absoluto(
            candidato, min_auc_pr=min_auc_pr, min_recall_positivo=min_recall_positivo
        ),
        *motivos_de_regressao(candidato, incumbente),
    ]


def deve_promover(
    candidato: dict[str, float],
    incumbente: dict[str, float] | None,
    *,
    min_auc_pr: float,
    min_recall_positivo: float,
) -> bool:
    """Decide se o candidato substitui o modelo em produção.

    Dois pisos absolutos — AUC-PR e recall da classe positiva — e, quando já existe um
    incumbente, dois critérios de não regressão. O recall positivo é a métrica de negócio:
    aprovar um inadimplente custa o valor emprestado, recusar um bom pagador custa a
    margem. AUC-PR sozinho não enxerga essa assimetria.
    """
    return not motivos_de_reprovacao(
        candidato,
        incumbente,
        min_auc_pr=min_auc_pr,
        min_recall_positivo=min_recall_positivo,
    )


def registrar_historico(
    caminho: Path | str,
    *,
    candidato: str,
    metricas: dict[str, Any],
    promovido: bool,
    motivos: list[str],
) -> None:
    """Acrescenta uma linha ao histórico de execuções, em JSON Lines.

    Uma linha por execução, nunca reescrita: cada run só acrescenta, e uma escrita
    interrompida derruba a última linha em vez do arquivo inteiro. É o que permite
    responder "por que o gate vem reprovando?" lendo, não adivinhando.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    registro = {
        "carimbo": datetime.now(UTC).isoformat(),
        "candidato": candidato,
        "metricas": metricas,
        "promovido": promovido,
        "motivos": motivos,
    }
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
