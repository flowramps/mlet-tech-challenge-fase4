"""A orquestração compõe peças já testadas isoladamente (contrato, simulação, PSI/KS
próprio, Evidently, gate, degradação, atribuição causal) — o que estes testes cobrem é o
ENCADEAMENTO: a ordem certa, o desfecho certo quando um lote reprova o contrato, e que a
consolidação aponta exatamente as features que `credito.data.simulate` de fato desloca.

Nenhum teste toca rede nem o arquivo real do dataset: o dataset é sintético, escrito em
`tmp_path`, e a conexão de rede é bloqueada do mesmo jeito que
`tests/drift/test_evidently_backend.py` já faz — se o Evidently tentasse abrir uma conexão
real, o teste falharia por isso, não passaria em silêncio.
"""

from __future__ import annotations

import json
import socket

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from credito.contracts.base import ContratoViolado, ValidationResult, Violacao
from credito.data.simulate import VARIAVEIS_COM_DRIFT
from credito.drift.base import DriftDeFeature, DriftReport, Severidade
from credito.model.train import salvar_modelo, treinar
from credito.pipeline.monitoring import (
    _CANAL_POR_FEATURE,
    ReferenciaInvalida,
    _narrativa_causal_versus_psi,
    executar_monitoramento,
    main,
)
from credito.schema import ALVO, FEATURES

# N=2000 não é arbitrário: a partição de teste que resulta (test_size=0,2 do padrão de
# `config.py`, ~400 linhas) precisa ser grande o bastante para que o p-valor de KS da
# injeção de atraso sobreviva à correção de Benjamini-Hochberg no mês 6 — com N menor
# (medido: 1200, partição de 240 linhas) o PSI já cruza 0,10 mas o KS não é significativo
# o bastante após a correção, e a severidade efetiva cai para ESTAVEL por desenho do gate
# (`drift/gate.py`), não por falha deste teste. Medido com este N e este seed: PSI≈0,131,
# p-valor de KS≈0,005, sobrevive à correção com folga.
N = 2000
SEED = 7


@pytest.fixture(autouse=True)
def _bloqueia_rede(monkeypatch):
    def _recusa(*_args, **_kwargs):
        raise AssertionError("o monitoramento tentou abrir uma conexão de rede")

    monkeypatch.setattr(socket.socket, "connect", _recusa)
    monkeypatch.setattr(socket.socket, "connect_ex", _recusa)


def _referencia_sintetica(n: int, *, seed: int) -> pd.DataFrame:
    """Uma Referência plausível e sem duplicata: variação contínua por linha em toda
    ``FEATURES``, o mesmo mecanismo que ``tests/data/test_simulate.py`` já usa para que
    nenhuma linha colida antes de qualquer drift."""
    gerador = np.random.default_rng(seed)
    positivo = gerador.random(n) < 0.15
    # Sinal real e ruidoso: LogisticRegression precisa de algo para aprender, mas não tão
    # limpo a ponto de separar as classes perfeitamente e o piso absoluto do treino nunca
    # ser exercitável — não é o que este módulo testa, mas evita AUC-PR degenerado (1,0 ou
    # indefinido) que quebraria `avaliar` mais adiante.
    frame = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": gerador.uniform(0.01, 1.0, n),
            "age": gerador.integers(18, 90, n),
            "NumberOfTime30-59DaysPastDueNotWorse": np.where(
                positivo, gerador.integers(1, 4, n), gerador.integers(0, 2, n)
            ),
            "DebtRatio": gerador.uniform(0.02, 1.2, n),
            "MonthlyIncome": np.maximum(
                np.where(positivo, 2500.0, 5000.0) + gerador.normal(0, 1500, n), 200.0
            ),
            "NumberOfOpenCreditLinesAndLoans": gerador.integers(1, 20, n),
            "NumberOfTimes90DaysLate": gerador.integers(0, 2, n),
            "NumberRealEstateLoansOrLines": gerador.integers(0, 4, n),
            "NumberOfTime60-89DaysPastDueNotWorse": gerador.integers(0, 2, n),
            "NumberOfDependents": gerador.integers(0, 5, n).astype(float),
        }
    )
    frame[ALVO] = positivo.astype(int)
    return frame[[ALVO, *FEATURES]]


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    """Monta uma Referência pequena em disco, treina e publica um campeão simples, e
    aponta toda a configuração para `tmp_path` — o mesmo padrão de
    `tests/pipeline/test_training.py::ambiente`, adaptado: aqui o alvo do fixture é ter um
    modelo publicado para `executar_monitoramento` carregar, não exercitar o próprio
    treino."""
    from credito.config import get_settings

    referencia = _referencia_sintetica(N, seed=SEED)

    bruto = tmp_path / "raw" / "d.arff"
    bruto.parent.mkdir(parents=True, exist_ok=True)
    cabecalho = ["@RELATION t", "@ATTRIBUTE FinancialDistressNextTwoYears {No, Yes}"]
    cabecalho += [f"@ATTRIBUTE {coluna} REAL" for coluna in FEATURES]
    cabecalho.append("@DATA")
    saida = referencia.copy()
    saida["FinancialDistressNextTwoYears"] = np.where(saida[ALVO] == 1, "Yes", "No")
    corpo = saida[["FinancialDistressNextTwoYears", *FEATURES]].to_csv(index=False, header=False)
    bruto.write_text("\n".join(cabecalho) + "\n" + corpo, encoding="utf-8")

    get_settings.cache_clear()
    monkeypatch.setenv("CREDITO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CREDITO_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CREDITO_METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("CREDITO_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("CREDITO_MLRUNS_DIR", str(tmp_path / "mlruns"))
    monkeypatch.setenv("CREDITO_DATASET_FILENAME", "d.arff")
    settings = get_settings()

    modelo = treinar(referencia, "regressao_logistica", seed=42)
    salvar_modelo(
        modelo,
        settings.model_path,
        {"candidato": "regressao_logistica", "auc_pr": 0.5, "recall_positivo": 0.5},
    )

    yield tmp_path
    get_settings.cache_clear()


def test_ponta_a_ponta_produz_seis_relatorios_html(ambiente):
    resultado = executar_monitoramento(meses=6, seed=123)

    caminho_relatorios = ambiente / "reports"
    htmls = sorted(caminho_relatorios.glob("drift_mes_*.html"))
    assert [caminho.name for caminho in htmls] == [f"drift_mes_{m:02d}.html" for m in range(1, 7)]

    assert len(resultado["lotes"]) == 6
    assert resultado["gate"].severidade in (
        Severidade.ESTAVEL,
        Severidade.ATENCAO,
        Severidade.CRITICO,
    )


def test_ponta_a_ponta_grava_o_resumo_consolidado_em_json(ambiente):
    executar_monitoramento(meses=6, seed=123)

    caminho = ambiente / "reports" / "monitoramento.json"
    assert caminho.exists()
    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    assert "gate" in conteudo
    assert "degradacao_por_lote" in conteudo
    assert "atribuicao_causal_por_lote" in conteudo
    assert len(conteudo["lotes"]) == 6


def test_ponta_a_ponta_registra_run_no_mlflow_com_parametros_metricas_e_artefatos(ambiente):
    """`executar_monitoramento` liga ao MLflow de verdade — não só constrói o
    JSON consolidado. Sem este teste, remover a chamada a `registrar_execucao` deixaria a
    suíte inteira verde: nenhum outro teste desta classe consulta o backend MLflow."""
    from credito.config import get_settings

    resultado = executar_monitoramento(meses=6, seed=123)

    assert resultado["mlflow_run_id"]

    settings = get_settings()
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    run = client.get_run(resultado["mlflow_run_id"])

    assert run.data.params["semente"] == "123"
    assert run.data.params["meses"] == "6"

    # PSI por feature é série temporal (um valor por mês, step=mes) — get_metric_history
    # devolve todos os pontos; r.data.metrics só devolveria o último (ver o docstring de
    # credito.tracking.mlflow_client).
    historico_psi = {
        nome: client.get_metric_history(run.info.run_id, nome)
        for nome in run.data.metrics
        if nome.startswith("drift.psi.")
    }
    assert historico_psi, "nenhuma métrica drift.psi.* foi registrada"
    algum_historico = next(iter(historico_psi.values()))
    assert len(algum_historico) == 6, "psi por feature deveria ter um ponto por mês"

    # As três famílias de proxy sem rótulo (`credito.monitoring.proxies`) também precisam
    # estar no run — é o que conecta o alarme desta etapa ao rastreamento.
    nomes_de_metrica = set(run.data.metrics)
    assert any(nome.startswith("proxy.taxa_de_aprovacao") for nome in nomes_de_metrica)
    assert any(nome.startswith("proxy.psi_do_score") for nome in nomes_de_metrica)
    assert any(nome.startswith("proxy.confianca_media") for nome in nomes_de_metrica)

    artefatos = {artefato.path for artefato in client.list_artifacts(run.info.run_id)}
    assert "monitoramento.json" in artefatos
    assert any(nome.startswith("drift_mes_") for nome in artefatos)


def test_json_consolidado_poe_a_consequencia_antes_da_tabela_de_psi(ambiente):
    """A ordem das chaves do JSON é uma afirmação feita em três documentos — o docstring
    de `credito.pipeline.monitoring`, a seção de monitoramento do README e
    `docs/model_card.md` dizem todos que o consolidado grava a consequência (degradação e
    atribuição causal) ANTES da tabela de PSI/KS, que mora dentro de `lotes`.

    `json.dumps` preserva a ordem de inserção do dict e `json.loads` a devolve — então a
    ordem é observável por quem lê o arquivo, humano ou máquina, e é testável aqui. Sem
    este teste a afirmação já tinha ficado falsa uma vez sem nada notar: `lotes` vinha
    primeiro.
    """
    executar_monitoramento(meses=6, seed=123)

    conteudo = json.loads((ambiente / "reports" / "monitoramento.json").read_text(encoding="utf-8"))
    chaves = list(conteudo)

    assert chaves.index("degradacao_por_lote") < chaves.index("lotes")
    assert chaves.index("atribuicao_causal_por_lote") < chaves.index("lotes")


def test_atribuicao_causal_usa_a_mesma_janela_dos_lotes_monitorados(ambiente):
    """Regressão do acoplamento que faltava: `executar_monitoramento` expõe `meses` e o
    usa para gerar os lotes, mas `atribuir_degradacao` tem o seu próprio default (`MESES`,
    6). Sem repassar `meses`, uma janela diferente de 6 produziria um bloco
    `atribuicao_causal_por_lote` calculado com `k = mes / 6` enquanto os lotes medidos
    foram gerados com `k = mes / meses` — a decomposição descreveria um processo gerador
    que nunca produziu os lotes do relatório, sem nenhum aviso.

    Uma janela de 3 meses é o caso mínimo que separa as duas leituras (todo teste desta
    suíte usava `meses=6`, o único valor em que o default coincide com a janela).
    """
    resultado = executar_monitoramento(meses=3, seed=123)

    atribuicao = resultado["atribuicao_causal_por_lote"]
    assert set(atribuicao) == {"mes_01", "mes_02", "mes_03"}
    for mes, bloco in enumerate(atribuicao.values(), start=1):
        assert bloco["mes"] == mes
        assert bloco["meses"] == 3

    # E a prova de que a janela mudou o número, não só o metadado: o cenário "todos" do
    # último mês tem que bater com a métrica que o lote de fato monitorado produziu — o
    # mesmo `avaliar` já guardado em `metricas_campeao`.
    assert resultado["atribuicao_causal_por_lote"]["mes_03"]["metricas_por_cenario"][
        "todos"
    ] == pytest.approx(resultado["lotes"]["mes_03"]["metricas_campeao"]["auc_roc"])


def test_consolidacao_identifica_as_features_corretas(ambiente):
    """As três variáveis que `credito.data.simulate` de fato desloca
    (`VARIAVEIS_COM_DRIFT`) precisam cruzar para atenção em algum lote da janela; as sete
    que o simulador nunca toca precisam permanecer ESTAVEL do início ao fim — a
    consolidação não pode acusar (nem deixar de acusar) a feature errada."""
    resultado = executar_monitoramento(meses=6, seed=123)

    cruzamentos = {c.feature: c for c in resultado["gate"].cruzamentos}

    for feature in VARIAVEIS_COM_DRIFT:
        assert cruzamentos[feature].primeiro_lote_atencao is not None, feature

    intocadas = set(FEATURES) - set(VARIAVEIS_COM_DRIFT)
    for feature in intocadas:
        assert cruzamentos[feature].primeiro_lote_atencao is None, feature
        assert cruzamentos[feature].severidade_final is Severidade.ESTAVEL, feature


def test_lote_invalido_falha_o_run_com_contrato_violado(ambiente, monkeypatch):
    """Regressão do requisito central desta camada: se a simulação produzisse um lote que o
    contrato reprova, o run tem de parar com `ContratoViolado` — nunca chegar ao detector
    de drift fingindo que aquilo foi só um alerta.

    O validador falso aprova as duas primeiras chamadas (a Referência inteira, e o lote
    `mes_01`) e reprova a partir da terceira (`mes_02`) — o mesmo ponto da janela em que o
    defeito real (colisão de quase-gêmeos na injeção de atraso, corrigido em
    `credito.data.simulate`) foi reproduzido contra a Referência real antes desta correção.
    """

    class _FalhaAPartirDoTerceiroLote:
        def __init__(self):
            self.chamadas = 0

        def validar(self, frame):
            self.chamadas += 1
            if self.chamadas >= 3:
                return ValidationResult(
                    total=len(frame),
                    violacoes=(
                        Violacao(regra="sem_duplicatas", coluna="*", linhas=1, indices=(0,)),
                    ),
                )
            return ValidationResult(total=len(frame), violacoes=())

    monkeypatch.setattr(
        "credito.pipeline.monitoring.construir_validador", lambda: _FalhaAPartirDoTerceiroLote()
    )

    with pytest.raises(ContratoViolado):
        executar_monitoramento(meses=6, seed=123)


def test_referencia_invalida_e_distinta_do_lote_invalido(ambiente, monkeypatch):
    """Regressão de revisão: se a própria Referência reprovar — antes de qualquer
    simulação —, a causa é o arquivo real, nunca o gerador de produção simulada. Sem
    `ReferenciaInvalida`, esta causa e a do teste acima (um lote SIMULADO reprovando)
    levantariam o mesmo `ContratoViolado` genérico, e `main()` não teria como saber qual
    mensagem escrever — acabaria sempre culpando o gerador, mesmo quando o defeito é no
    arquivo real.
    """

    class _SempreRecusa:
        def validar(self, frame):
            return ValidationResult(
                total=len(frame),
                violacoes=(
                    Violacao(
                        regra="renda_nao_nula", coluna="MonthlyIncome", linhas=1, indices=(0,)
                    ),
                ),
            )

    monkeypatch.setattr("credito.pipeline.monitoring.construir_validador", lambda: _SempreRecusa())

    with pytest.raises(ReferenciaInvalida):
        executar_monitoramento(meses=6, seed=123)


def test_main_reporta_contrato_violado_como_defeito_do_gerador(ambiente, monkeypatch, caplog):
    import logging

    def _interrompe(**_kwargs):
        raise ContratoViolado("1 de 100 linha(s) reprovadas — sem_duplicatas (*)")

    monkeypatch.setattr("credito.pipeline.monitoring.executar_monitoramento", _interrompe)

    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as saida:
        main()

    assert saida.value.code == 1
    # A mensagem precisa apontar o gerador, não o detector — é a distinção central desta
    # camada (drift não é invalidez; um lote inválido é defeito de quem o produziu).
    assert "gerador" in caplog.text
    assert "não drift" in caplog.text or "nao drift" in caplog.text


def test_main_reporta_referencia_invalida_apontando_o_arquivo_nao_o_gerador(
    ambiente, monkeypatch, caplog
):
    """O par exato do teste acima, para a outra causa: a mensagem tem que apontar o
    arquivo real, e não pode mencionar o gerador — as duas causas precisam continuar
    distinguíveis depois que `main()` traduz a exceção para o log, não só antes.
    """
    import logging

    def _interrompe(**_kwargs):
        raise ReferenciaInvalida("5 de 100 linha(s) reprovadas — renda_nao_nula (MonthlyIncome)")

    monkeypatch.setattr("credito.pipeline.monitoring.executar_monitoramento", _interrompe)

    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as saida:
        main()

    assert saida.value.code == 1
    assert "arquivo" in caplog.text
    assert "gerador" not in caplog.text


def test_canal_por_feature_cobre_exatamente_as_variaveis_com_drift():
    assert set(_CANAL_POR_FEATURE) == set(VARIAVEIS_COM_DRIFT)


def _relatorio(feature_top_psi: str, psi_top: float) -> DriftReport:
    return DriftReport(
        lote="mes_06",
        features=(
            DriftDeFeature(
                feature=feature_top_psi,
                psi_divergencia=psi_top,
                ks_p_valor=0.001,
                severidade=Severidade.CRITICO if psi_top >= 0.25 else Severidade.ATENCAO,
            ),
            DriftDeFeature(
                feature="age", psi_divergencia=0.01, ks_p_valor=0.9, severidade=Severidade.ESTAVEL
            ),
        ),
    )


def test_narrativa_aponta_quando_concept_drift_supera_a_feature_de_maior_psi():
    relatorio = _relatorio("DebtRatio", 0.30)
    atribuicao = {
        "metrica": "auc_roc",
        "degradacao_por_cenario": {
            "renda": 0.0,
            "divida": 0.003,
            "atraso": 0.0,
            "concept_drift": 0.05,
            "nenhum_drift": 0.0,
            "todos": 0.06,
        },
        "efeito_conjunto": 0.06,
    }

    narrativa = _narrativa_causal_versus_psi(relatorio, atribuicao)

    assert "DebtRatio" in narrativa
    assert "não é a causa mais provável" in narrativa


def test_narrativa_nao_alerta_quando_a_feature_de_maior_psi_de_fato_domina():
    relatorio = _relatorio("DebtRatio", 0.30)
    atribuicao = {
        "metrica": "auc_roc",
        "degradacao_por_cenario": {
            "renda": 0.0,
            "divida": 0.05,
            "atraso": 0.0,
            "concept_drift": 0.001,
            "nenhum_drift": 0.0,
            "todos": 0.06,
        },
        "efeito_conjunto": 0.06,
    }

    narrativa = _narrativa_causal_versus_psi(relatorio, atribuicao)

    assert "não é a causa mais provável" not in narrativa


def test_narrativa_distingue_sem_canal_isolado_de_degradacao_nao_mensuravel():
    """Regressão de revisão: duas razões diferentes podem impedir o cálculo da fração, e
    a narrativa não pode usar a mesma frase para as duas. Aqui a feature de maior PSI
    (`NumberOfDependents`) é uma das sete que `credito.data.simulate` nunca desloca — não
    tem canal causal isolado —, mas a degradação do lote está perfeitamente mensurada
    (`efeito_conjunto=0.05 > 0`). Dizer "degradação não mensurável" seria falso.
    """
    relatorio = _relatorio("NumberOfDependents", 0.30)
    atribuicao = {
        "metrica": "auc_roc",
        "degradacao_por_cenario": {
            "renda": 0.0,
            "divida": 0.0,
            "atraso": 0.0,
            "concept_drift": 0.05,
            "nenhum_drift": 0.0,
            "todos": 0.05,
        },
        "efeito_conjunto": 0.05,
    }

    narrativa = _narrativa_causal_versus_psi(relatorio, atribuicao)

    assert "não tem canal isolado" in narrativa
    assert "não mensurável" not in narrativa


def test_narrativa_relata_degradacao_nao_mensuravel_quando_efeito_conjunto_e_zero():
    """A outra metade do par: canal existe (`DebtRatio` mapeia para `divida`), mas o
    efeito conjunto medido no lote não é positivo — aí sim a mensagem correta é
    "degradação não mensurável", e não a de canal ausente."""
    relatorio = _relatorio("DebtRatio", 0.30)
    atribuicao = {
        "metrica": "auc_roc",
        "degradacao_por_cenario": {
            "renda": 0.0,
            "divida": 0.0,
            "atraso": 0.0,
            "concept_drift": 0.0,
            "nenhum_drift": 0.0,
            "todos": 0.0,
        },
        "efeito_conjunto": 0.0,
    }

    narrativa = _narrativa_causal_versus_psi(relatorio, atribuicao)

    assert "não mensurável" in narrativa
    assert "não tem canal isolado" not in narrativa


def test_lote_reprovado_deixa_run_failed_com_a_tarefa_e_o_mes_que_pararam(ambiente, monkeypatch):
    """Antes, um lote reprovado pelo contrato abortava o monitoramento antes de qualquer
    registro: a falha só existia no stdout. Agora o run é aberto no início e fica `FAILED`,
    com o contrato em 0 exatamente no mês que reprovou e os anteriores registrados."""
    from credito.config import get_settings

    class _ReprovaOSegundoLote:
        def __init__(self):
            self.chamadas = 0

        def validar(self, frame):
            self.chamadas += 1  # 1 = Referência, 2 = mes_01, 3 = mes_02
            if self.chamadas == 3:
                return ValidationResult(
                    total=len(frame),
                    violacoes=(Violacao(regra="sem_duplicatas", coluna="*", linhas=4),),
                )
            return ValidationResult(total=len(frame), violacoes=())

    monkeypatch.setattr(
        "credito.pipeline.monitoring.construir_validador", lambda: _ReprovaOSegundoLote()
    )
    with pytest.raises(ContratoViolado):
        executar_monitoramento(meses=6, seed=123)

    settings = get_settings()
    cliente = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    experimento = cliente.get_experiment_by_name(settings.mlflow_experimento)
    (run,) = cliente.search_runs([experimento.experiment_id])
    assert run.info.status == "FAILED"
    assert run.data.tags["falha.tipo"] == "ContratoViolado"
    contrato = cliente.get_metric_history(run.info.run_id, "tarefa.contrato")
    # passo 0 = a Referência; 1 = mes_01 aprovado; 2 = mes_02 reprovado — e nada depois.
    assert [(p.step, p.value) for p in contrato] == [(0, 1.0), (1, 1.0), (2, 0.0)]
    reprovadas = cliente.get_metric_history(run.info.run_id, "volume.linhas_reprovadas")
    assert (2, 4.0) in [(p.step, p.value) for p in reprovadas]


def test_monitoramento_registra_volume_por_lote_e_desfecho(ambiente):
    from credito.config import get_settings

    resultado = executar_monitoramento(meses=6, seed=123)

    cliente = MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)
    run = cliente.get_run(resultado["mlflow_run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["pipeline"] == "monitoramento"
    assert run.data.tags["desfecho"] == "concluido"
    volume = cliente.get_metric_history(run.info.run_id, "volume.linhas_lote")
    assert [p.step for p in volume] == [1, 2, 3, 4, 5, 6]
    for tarefa in ("ingestao", "simulacao", "predicao", "deteccao_de_drift", "gate"):
        assert f"tarefa.{tarefa}" in run.data.metrics, tarefa
