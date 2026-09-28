"""Registra uma execução do pipeline de monitoramento (`credito.pipeline.monitoring`) como
um run do MLflow — parâmetros, métricas e artefatos de uma única passada, lote a lote.

**Backend: SQLite, não o de arquivo.** Rodando `mlflow==3.16.1` contra um `file://` puro,
`mlflow.set_experiment(...)` levanta `MlflowException` avisando que o backend de arquivo
está em modo de manutenção e recomendando migrar para um backend de banco (medido
diretamente, `mlflow.__version__ == "3.16.1"`). Este módulo usa sempre
`sqlite:///<caminho>/mlflow.db` — um arquivo só, mais simples de manter fora do git que uma
árvore de diretórios, e o caminho que a própria mensagem de erro do MLflow recomenda.

**`artifact_location` precisa ser explícito quando o teste não pode tocar o diretório de
trabalho do processo.** Medido rodando a biblioteca real: sem informar `artifact_location`
ao criar o experimento, o MLflow grava artefato em `./mlruns/<id_do_experimento>/...`
relativo ao `cwd` do processo — **mesmo** com o backend de tracking apontando para um
SQLite dentro de `tmp_path`. As duas coisas são independentes: o SQLite guarda metadado
(parâmetro, métrica, tag), o `artifact_location` decide onde o arquivo de artefato é
gravado, e um não implica o outro. Por isso `registrar_execucao` expõe `artifact_location`
como parâmetro explícito em vez de assumir que apontar o backend de tracking para
`tmp_path` já isola os artefatos — os testes deste módulo (`tests/tracking/
test_mlflow_client.py`) fixam os dois.

**Parâmetro vs métrica vs tag — decisão, não acidente:**

- **Parâmetro** (`mlflow.log_param`): uma entrada que NÃO muda durante o run — a semente
  da simulação, a quantidade de meses da janela, os limiares de PSI/AUC-PR/recall, o
  `alfa` de Benjamini-Hochberg que o próprio `GateDeDrift` carrega.
- **Métrica** (`mlflow.log_metric`): um número MEDIDO. Usa `step` sempre que existe um
  valor por mês — PSI por feature, as métricas do campeão por lote, os proxies sem rótulo
  de `credito.monitoring.proxies.sinais_do_lote`. Confirmado rodando a biblioteca real
  (`mlflow==3.16.1`): uma métrica logada em vários `step` guarda o histórico inteiro —
  `MlflowClient.get_metric_history` devolve todos os pontos, na ordem em que foram
  logados; `MlflowClient.get_run(...).data.metrics` devolve só o último valor por nome, e
  não é onde a série completa se lê. O veredito do gate é logado sem `step`: é um único
  número por run inteiro, não uma série.
- **Tag** (`mlflow.set_tag`): um rótulo pensado para busca entre runs — o veredito do gate
  como string (`CRITICO`/`ATENCAO`/`ESTAVEL`), para uma consulta do tipo "todo run
  CRÍTICO" no MLflow.

**Nome de métrica é ponto-e-nível, nunca string livre.** `drift.psi.MonthlyIncome`, nunca
"PSI da renda" — a mesma disciplina de cardinalidade controlada que manteve o middleware
Prometheus de uma etapa anterior deste projeto sem explodir: um rótulo de métrica só pode
vir de um conjunto FECHADO, nunca de entrada não controlada. Isto vale para as **três**
famílias de métrica que este módulo grava, não só duas: `credito.schema.FEATURES` (para
`drift.psi.*`), as três chaves que `sinais_do_lote` de fato devolve (para `proxy.*`), e
`credito.model.evaluate.METRICAS_GLOBAIS` — as chaves que `avaliar` de fato devolve (para
`campeao.*`). `registrar_execucao` valida as três contra esses conjuntos e levanta
`ValueError` antes de abrir qualquer run se algo fugir deles, para que um nome de feature,
de proxy ou de métrica de campeão nunca vazado vire rótulo de métrica sem ninguém notar.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from credito.config import Settings
from credito.drift.base import DriftReport
from credito.drift.gate import GateDeDrift
from credito.model.evaluate import METRICAS_GLOBAIS
from credito.schema import FEATURES

# `credito.pipeline.monitoring.executar_monitoramento` nomeia cada lote simulado
# `f"mes_{mes:02d}"` (`mes` de 1 a `meses`); este módulo só entende esse padrão porque é o
# único que o resto do projeto de fato produz — um nome fora dele é bug do chamador, não
# um formato alternativo a suportar.
_PADRAO_LOTE = re.compile(r"^mes_(\d+)$")

# As três chaves que `credito.monitoring.proxies.sinais_do_lote` de fato devolve (lidas do
# código-fonte daquele módulo, não inventadas aqui) — o conjunto fechado que autoriza um
# nome de métrica `proxy.*`.
_PROXIES_CONHECIDOS = frozenset({"confianca_media", "taxa_de_aprovacao", "psi_do_score"})

_FEATURES_CONHECIDAS = frozenset(FEATURES)

# Importado de `credito.model.evaluate`, não restatado aqui — o conjunto fechado que
# autoriza um nome de métrica `campeao.*`, a mesma disciplina que `_FEATURES_CONHECIDAS` e
# `_PROXIES_CONHECIDOS` já aplicam. `test_metricas_globais_cobre_exatamente_as_chaves_de_
# avaliar` (em `tests/model/test_evaluate.py`) tranca que `METRICAS_GLOBAIS` nunca fique
# defasado do que `avaliar` de fato devolve.
_METRICAS_CAMPEAO_CONHECIDAS = frozenset(METRICAS_GLOBAIS)


def passo_do_lote(lote: str) -> int:
    """Extrai o número do mês do nome do lote (`"mes_06"` -> `6`) — o `step` que toda
    métrica em série temporal deste módulo usa.

    Levanta `ValueError` para qualquer nome fora do padrão `mes_<dígitos>`: um nome de
    lote inesperado aqui não tem "melhor esforço" — silenciosamente inventar um `step`
    (por exemplo, a ordem de chegada) descreveria uma cronologia que ninguém mediu.
    """
    casamento = _PADRAO_LOTE.fullmatch(lote)
    if casamento is None:
        raise ValueError(
            f"lote {lote!r} fora do padrão 'mes_<dígitos>' que "
            "credito.pipeline.monitoring produz — não há step a extrair dele"
        )
    return int(casamento.group(1))


def parametros_da_execucao(settings: Settings, *, seed: int, meses: int) -> dict[str, int | float]:
    """Monta o dicionário de parâmetros (entradas que não mudam durante o run) a partir de
    `Settings` — a semente da simulação, a janela em meses, e os quatro limiares que o
    gate de drift e o gate de promoção usam para decidir. Nenhum destes é recalculado
    aqui: todos vêm de `settings`, a mesma fonte que o resto do pipeline já lê.
    """
    return {
        "semente": seed,
        "meses": meses,
        "psi_atencao": settings.psi_atencao,
        "psi_critico": settings.psi_critico,
        "min_auc_pr": settings.min_auc_pr,
        "min_recall_positivo": settings.min_recall_positivo,
    }


def _validar_metricas_campeao(
    metricas_do_campeao_por_lote: Mapping[str, Mapping[str, float]],
) -> None:
    for lote, metricas in metricas_do_campeao_por_lote.items():
        passo_do_lote(lote)
        for nome in metricas:
            if nome not in _METRICAS_CAMPEAO_CONHECIDAS:
                raise ValueError(
                    f"métrica de campeão {nome!r} (lote {lote!r}) não está entre as "
                    f"chaves que credito.model.evaluate.avaliar devolve "
                    f"({sorted(_METRICAS_CAMPEAO_CONHECIDAS)}) — nome de métrica de "
                    "campeão só pode vir desse conjunto fechado"
                )


def _validar_features(drift_por_lote: Sequence[DriftReport]) -> None:
    for relatorio in drift_por_lote:
        passo_do_lote(relatorio.lote)
        for feature in relatorio.features:
            if feature.feature not in _FEATURES_CONHECIDAS:
                raise ValueError(
                    f"feature {feature.feature!r} (lote {relatorio.lote!r}) não está em "
                    "credito.schema.FEATURES — nome de métrica de drift só pode vir "
                    "desse conjunto fechado"
                )


def _validar_proxies(proxies_por_lote: Mapping[str, Mapping[str, float]]) -> None:
    for lote, proxies in proxies_por_lote.items():
        passo_do_lote(lote)
        for nome in proxies:
            if nome not in _PROXIES_CONHECIDOS:
                raise ValueError(
                    f"proxy {nome!r} (lote {lote!r}) não está entre as chaves que "
                    f"credito.monitoring.proxies.sinais_do_lote devolve "
                    f"({sorted(_PROXIES_CONHECIDOS)}) — nome de métrica de proxy só pode "
                    "vir desse conjunto fechado"
                )


def _preparar_experimento(nome: str, artifact_location: str | None) -> None:
    """Cria o experimento com `artifact_location` explícito na primeira vez que `nome` é
    visto; reusa o existente (e o `artifact_location` já fixado nele) nas vezes
    seguintes — o MLflow não permite trocar o `artifact_location` de um experimento já
    criado, e não há por que tentar.
    """
    client = MlflowClient()
    if client.get_experiment_by_name(nome) is None:
        mlflow.create_experiment(nome, artifact_location=artifact_location)
    mlflow.set_experiment(nome)


def registrar_execucao(
    *,
    tracking_uri: str,
    experimento: str,
    parametros: Mapping[str, int | float | str],
    metricas_do_campeao_por_lote: Mapping[str, Mapping[str, float]],
    drift_por_lote: Sequence[DriftReport],
    proxies_por_lote: Mapping[str, Mapping[str, float]],
    gate: GateDeDrift,
    artefatos: Sequence[Path | str] = (),
    artifact_location: str | None = None,
    nome_da_execucao: str | None = None,
) -> str:
    """Registra uma execução completa do pipeline de monitoramento como um run do MLflow
    e devolve o `run_id`.

    Valida nomes de métrica de campeão, de feature e de proxy ANTES de abrir o run (e
    antes mesmo de apontar o `tracking_uri`): um nome fora do conjunto fechado
    correspondente é erro do chamador, e levantar depois de já ter criado um run deixaria
    um run parcial (só parâmetros, sem métrica) no backend — pior que falhar cedo sem
    escrever nada.
    """
    _validar_metricas_campeao(metricas_do_campeao_por_lote)
    _validar_features(drift_por_lote)
    _validar_proxies(proxies_por_lote)

    mlflow.set_tracking_uri(tracking_uri)
    _preparar_experimento(experimento, artifact_location)

    with mlflow.start_run(run_name=nome_da_execucao) as run:
        for nome, valor in parametros.items():
            mlflow.log_param(nome, valor)
        # O alfa de correção vem do próprio `gate`, nunca repetido à mão pelo chamador —
        # duas fontes para o mesmo número divergiriam silenciosamente no dia em que
        # alguém mudasse uma sem lembrar da outra.
        mlflow.log_param("gate.alfa", gate.alfa)

        for lote, metricas in metricas_do_campeao_por_lote.items():
            passo = passo_do_lote(lote)
            for nome, valor in metricas.items():
                mlflow.log_metric(f"campeao.{nome}", float(valor), step=passo)

        for relatorio in drift_por_lote:
            passo = passo_do_lote(relatorio.lote)
            for feature in relatorio.features:
                mlflow.log_metric(
                    f"drift.psi.{feature.feature}", float(feature.psi_divergencia), step=passo
                )

        for lote, proxies in proxies_por_lote.items():
            passo = passo_do_lote(lote)
            for nome, valor in proxies.items():
                mlflow.log_metric(f"proxy.{nome}", float(valor), step=passo)

        # Veredito do gate: tag para busca ("todo run CRITICO"), métrica para consulta
        # numérica (comparar, agregar, plotar a severidade ao longo de vários runs) — ver
        # o docstring do módulo sobre por que os dois, não um só.
        mlflow.set_tag("gate.veredito", gate.severidade.name)
        mlflow.log_metric("gate.severidade", float(gate.severidade.value))

        for artefato in artefatos:
            mlflow.log_artifact(str(artefato))

        return run.info.run_id
