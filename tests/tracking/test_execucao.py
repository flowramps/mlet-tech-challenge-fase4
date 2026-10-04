"""Logs de execução do pipeline centralizados no MLflow — inclusive quando o pipeline falha.

Um registro que só existe quando tudo dá certo não é log de execução: é relatório de
sucesso. `execucao_rastreada` abre o run no início, e uma exceção no meio o deixa `FAILED`
com o tipo e a mensagem da falha, em vez de não deixar rastro nenhum.
"""

from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

from credito.tracking.execucao import DESFECHOS, TAREFAS, execucao_rastreada


class _FalhaError(RuntimeError):
    pass


def _abrir(tmp_path, **extra):
    return execucao_rastreada(
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        experimento="teste",
        artifact_location=str(tmp_path / "artefatos"),
        pipeline="monitoramento",
        **extra,
    )


def _run(tmp_path, run_id):
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db").get_run(run_id)


def test_execucao_bem_sucedida_termina_finished_com_desfecho(tmp_path):
    with _abrir(tmp_path) as execucao:
        execucao.tarefa("ingestao", volume={"linhas_brutas": 150_000})
        execucao.concluir("concluido")

    run = _run(tmp_path, execucao.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["pipeline"] == "monitoramento"
    assert run.data.tags["desfecho"] == "concluido"
    assert run.data.metrics["tarefa.ingestao"] == 1.0
    assert run.data.metrics["volume.linhas_brutas"] == 150_000


def test_falha_no_meio_deixa_o_run_failed_com_a_causa_e_reergue(tmp_path):
    # O caso que motivou o módulo: antes, um lote reprovado pelo contrato abortava o
    # monitoramento antes de qualquer registro — a falha não existia para o MLflow.
    with pytest.raises(_FalhaError, match="lote mes_03"), _abrir(tmp_path) as execucao:
        execucao.tarefa("ingestao")
        raise _FalhaError("lote mes_03 reprovou o contrato")

    run = _run(tmp_path, execucao.run_id)
    assert run.info.status == "FAILED"
    assert run.data.tags["desfecho"] == "falha"
    assert run.data.tags["falha.tipo"] == "_FalhaError"
    assert "mes_03" in run.data.tags["falha.mensagem"]
    assert run.data.metrics["tarefa.ingestao"] == 1.0


def test_tarefa_com_falha_registra_zero_no_passo_do_lote(tmp_path):
    with _abrir(tmp_path) as execucao:
        execucao.tarefa("contrato", passo=1)
        execucao.tarefa("contrato", sucesso=False, passo=3, volume={"linhas_reprovadas": 7})
        execucao.concluir("concluido")

    historico = MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db").get_metric_history(
        execucao.run_id, "tarefa.contrato"
    )
    assert [(p.step, p.value) for p in historico] == [(1, 1.0), (3, 0.0)]


def test_excecao_de_desfecho_normal_nao_vira_falha(tmp_path):
    # "Nada a promover" não é defeito: o retreino sobre dado estático reproduz o incumbente
    # toda semana. O pipeline declara o desfecho e o run termina FINISHED — um alarme de
    # falha disparando toda semana destruiria o valor do alarme.
    with pytest.raises(_FalhaError), _abrir(tmp_path, desfechos_normais=(_FalhaError,)) as execucao:
        execucao.concluir("nao_promovido")
        raise _FalhaError("candidato não supera o incumbente")

    run = _run(tmp_path, execucao.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["desfecho"] == "nao_promovido"
    assert "falha.tipo" not in run.data.tags


def test_tarefa_fora_do_conjunto_fechado_e_recusada_antes_de_registrar(tmp_path):
    with _abrir(tmp_path) as execucao, pytest.raises(ValueError, match="tarefa"):
        execucao.tarefa("qualquer coisa")


def test_volume_com_nome_malformado_e_recusado(tmp_path):
    with _abrir(tmp_path) as execucao, pytest.raises(ValueError, match="volume"):
        execucao.tarefa("ingestao", volume={"Linhas Brutas!": 1})


def test_desfecho_fora_do_conjunto_fechado_e_recusado(tmp_path):
    with _abrir(tmp_path) as execucao, pytest.raises(ValueError, match="desfecho"):
        execucao.concluir("talvez")


def test_conjuntos_fechados_cobrem_os_dois_pipelines():
    assert {"ingestao", "contrato", "treino", "avaliacao", "gate"} <= TAREFAS
    assert {"simulacao", "predicao", "deteccao_de_drift"} <= TAREFAS
    assert {"promovido", "nao_promovido", "concluido", "falha"} == DESFECHOS
