"""Log de execução dos pipelines no MLflow — sucesso e falha, tarefa a tarefa, com volume.

`credito.tracking.mlflow_client.registrar_execucao` grava o **resultado** de um
monitoramento que deu certo. Isto é outra coisa: o **log da execução**, aberto no início do
pipeline, para que uma falha no meio — um lote reprovado pelo contrato, um piso do gate
violado — deixe um run `FAILED` com a causa, em vez de nenhum rastro. É o que permite
responder no MLflow, sem ler stdout de ninguém: quando o pipeline rodou, se terminou, em
que tarefa parou, quantas linhas processou e quantas recusou.

**Falha contra "nada a fazer".** Uma exceção que o pipeline usa como desfecho normal (o
candidato não supera o incumbente, `ModelNotPromoted`) entra em `desfechos_normais`: o run
termina `FINISHED`, com o desfecho declarado. Só o resto vira `FAILED` — o mesmo princípio
que separa `QualityGateError` de `ModelNotPromoted` no gate.

**Nomes de conjunto fechado.** Tarefa, desfecho e pipeline vêm de conjuntos declarados
aqui; volume, de um formato validado — a mesma disciplina de cardinalidade das outras
famílias de métrica do projeto.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import mlflow

from credito.tracking.mlflow_client import preparar_experimento

PIPELINES = frozenset({"treino", "monitoramento"})
TAREFAS = frozenset(
    {
        # treino
        "ingestao",
        "contrato",
        "treino",
        "avaliacao",
        "gate",
        # monitoramento (reaproveita ingestao e contrato)
        "simulacao",
        "predicao",
        "deteccao_de_drift",
    }
)
DESFECHOS = frozenset({"promovido", "nao_promovido", "concluido", "falha"})
_FORMATO_DE_VOLUME = re.compile(r"^[a-z][a-z_]*(\.[a-z][a-z_]*)?$")
_FORMATO_DE_GRUPO = re.compile(r"^(teste|validacao\.[a-z][a-z_]*)$")
# Mensagem de falha vira tag: o MLflow limita o tamanho do valor, e o relatório completo de
# um contrato reprovado já sai no log e na exceção.
_LIMITE_DA_MENSAGEM = 500


class Execucao:
    """O run aberto de uma execução de pipeline."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def tarefa(
        self,
        nome: str,
        *,
        sucesso: bool = True,
        passo: int | None = None,
        volume: Mapping[str, float] | None = None,
    ) -> None:
        """Status da tarefa (1 = sucesso, 0 = falha) e o volume que ela processou.

        `passo` é o mês do lote, quando a tarefa roda uma vez por lote — o mesmo `step` que
        `mlflow_client` usa, para que a série se leia junto com a de drift.
        """
        if nome not in TAREFAS:
            raise ValueError(f"tarefa fora do conjunto declarado: {nome!r}")
        volume = dict(volume or {})
        malformados = [chave for chave in volume if not _FORMATO_DE_VOLUME.match(chave)]
        if malformados:
            raise ValueError(f"nome de volume fora do formato: {malformados!r}")
        passo = passo if passo is not None else 0
        mlflow.log_metric(f"tarefa.{nome}", 1.0 if sucesso else 0.0, step=passo)
        for chave, valor in volume.items():
            mlflow.log_metric(f"volume.{chave}", float(valor), step=passo)

    def parametros(self, valores: Mapping[str, int | float | str]) -> None:
        """Entradas que não mudam durante o run: semente, partições, pisos."""
        for nome, valor in valores.items():
            if not _FORMATO_DE_VOLUME.match(nome):
                raise ValueError(f"nome de parâmetro fora do formato: {nome!r}")
            mlflow.log_param(nome, valor)

    def metricas(self, grupo: str, valores: Mapping[str, float]) -> None:
        """Métricas de modelo, sob um grupo do conjunto fechado (`validacao.<candidato>`,
        `teste`) — o ciclo de vida dos candidatos no mesmo run do log de execução."""
        if not _FORMATO_DE_GRUPO.match(grupo):
            raise ValueError(f"grupo de métrica fora do formato: {grupo!r}")
        for nome, valor in valores.items():
            if not _FORMATO_DE_VOLUME.match(nome):
                raise ValueError(f"nome de métrica fora do formato: {nome!r}")
            if isinstance(valor, bool | int | float):
                mlflow.log_metric(f"{grupo}.{nome}", float(valor))

    def artefato(self, caminho: str) -> None:
        mlflow.log_artifact(str(caminho))

    def concluir(self, desfecho: str) -> None:
        if desfecho not in DESFECHOS:
            raise ValueError(f"desfecho fora do conjunto declarado: {desfecho!r}")
        mlflow.set_tag("desfecho", desfecho)


@contextmanager
def execucao_rastreada(
    *,
    tracking_uri: str,
    experimento: str,
    artifact_location: str | None,
    pipeline: str,
    desfechos_normais: tuple[type[BaseException], ...] = (),
) -> Iterator[Execucao]:
    """Abre um run para a execução inteira; uma exceção o fecha `FAILED` com a causa."""
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline fora do conjunto declarado: {pipeline!r}")
    mlflow.set_tracking_uri(tracking_uri)
    preparar_experimento(experimento, artifact_location)
    run = mlflow.start_run()
    mlflow.set_tag("pipeline", pipeline)
    execucao = Execucao(run.info.run_id)
    try:
        yield execucao
    except desfechos_normais:
        mlflow.end_run(status="FINISHED")
        raise
    except BaseException as erro:
        mlflow.set_tag("desfecho", "falha")
        mlflow.set_tag("falha.tipo", type(erro).__name__)
        mlflow.set_tag("falha.mensagem", str(erro)[:_LIMITE_DA_MENSAGEM])
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")
