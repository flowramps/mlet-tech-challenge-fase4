"""O pipeline ponta a ponta precisa de um teste que não dependa de rede nem de 120 mil
linhas: ele valida o encadeamento e os desfechos, não a qualidade do modelo."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from credito.contracts.base import ContratoViolado, ValidationResult, Violacao
from credito.model.train import carregar_modelo
from credito.pipeline.steps import ModelNotPromoted, QualityGateError
from credito.pipeline.training import executar_pipeline, main


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    """Monta um dataset pequeno em disco e aponta a configuração para o tmp_path."""
    from credito.config import get_settings
    from credito.schema import FEATURES

    gerador = np.random.default_rng(42)
    n = 800
    positivo = gerador.random(n) < 0.07
    dados = {coluna: gerador.random(n) for coluna in FEATURES}
    dados["age"] = gerador.integers(18, 80, n)
    # Sinal presente mas ruidoso, de propósito: com desvio pequeno as classes se separam
    # perfeitamente, o AUC-PR vira 1,0 e o teste do piso absoluto perde o sentido — não
    # existiria piso ao mesmo tempo alcançável e violável. Já o corte em 100 não é
    # cosmético: o contrato exige MonthlyIncome >= 0, e sem ele a cauda esquerda do ruído
    # geraria renda negativa; o pipeline morreria no contrato antes de chegar ao gate, que
    # é justamente o que estes testes precisam exercitar.
    dados["MonthlyIncome"] = np.maximum(
        np.where(positivo, 2000.0, 6000.0) + gerador.normal(0, 2500, n), 100.0
    )
    for coluna in (
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
    ):
        dados[coluna] = gerador.integers(0, 3, n)
    dados["DebtRatio"] = gerador.random(n)
    dados["NumberOfDependents"] = gerador.integers(0, 3, n).astype(float)
    frame = pd.DataFrame(dados)
    frame["FinancialDistressNextTwoYears"] = np.where(positivo, "Yes", "No")

    bruto = tmp_path / "raw" / "d.arff"
    bruto.parent.mkdir(parents=True, exist_ok=True)
    cabecalho = ["@RELATION t"]
    colunas = ["FinancialDistressNextTwoYears", *FEATURES]
    cabecalho.append("@ATTRIBUTE FinancialDistressNextTwoYears {No, Yes}")
    cabecalho += [f"@ATTRIBUTE {coluna} REAL" for coluna in FEATURES]
    cabecalho.append("@DATA")
    corpo = frame[colunas].to_csv(index=False, header=False)
    bruto.write_text("\n".join(cabecalho) + "\n" + corpo, encoding="utf-8")

    get_settings.cache_clear()
    monkeypatch.setenv("CREDITO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CREDITO_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CREDITO_METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("CREDITO_DATASET_FILENAME", "d.arff")
    # Pisos baixos de propósito: o que estes testes exercitam é o encadeamento, e um piso
    # calibrado sobre o dado real reprovaria este dado sintético por motivo irrelevante.
    monkeypatch.setenv("CREDITO_MIN_AUC_PR", "0.01")
    monkeypatch.setenv("CREDITO_MIN_RECALL_POSITIVO", "0.01")
    yield tmp_path
    get_settings.cache_clear()


def test_primeira_execucao_promove_e_grava_artefatos(ambiente):
    resultado = executar_pipeline()

    assert resultado["promovido"] is True
    assert (ambiente / "models" / "model.joblib").exists()
    assert (ambiente / "metrics" / "metrics.json").exists()

    metricas = json.loads((ambiente / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert "auc_pr" in metricas
    assert "por_faixa_etaria" in metricas


def test_metrics_json_publica_as_duas_metricas_de_equidade_sobre_o_mesmo_recorte(ambiente):
    # As duas métricas precisam sair do MESMO `por_faixa_etaria` que o arquivo publica —
    # calculadas sobre outro recorte, o model card citaria uma razão que não bate com a
    # tabela de faixas ao lado dela.
    executar_pipeline()

    metricas = json.loads((ambiente / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    equidade = metricas["equidade"]
    por_faixa = metricas["por_faixa_etaria"]

    assert set(equidade["quatro_quintos"]["faixas"]) == set(por_faixa)
    referencia = equidade["quatro_quintos"]["referencia"]
    for nome, faixa in equidade["quatro_quintos"]["faixas"].items():
        esperado = por_faixa[nome]["taxa_de_aprovacao"] / por_faixa[referencia]["taxa_de_aprovacao"]
        assert faixa["razao"] == esperado
    assert "referencia" in equidade["oportunidade"]


def test_segunda_execucao_nao_falha_quando_empata(ambiente):
    # Esta é a regressão mais importante da etapa: sobre dado estático, o retreino
    # reproduz o incumbente. Se isso levantasse QualityGateError, todo pipeline periódico
    # ficaria vermelho da segunda execução em diante.
    executar_pipeline()

    with pytest.raises(ModelNotPromoted):
        executar_pipeline()


def test_modelo_publicado_nao_e_tocado_quando_nao_promove(ambiente):
    executar_pipeline()
    modelo = ambiente / "models" / "model.joblib"
    antes = modelo.read_bytes()

    with pytest.raises(ModelNotPromoted):
        executar_pipeline()

    assert modelo.read_bytes() == antes


def test_historico_registra_as_duas_execucoes(ambiente):
    executar_pipeline()
    with pytest.raises(ModelNotPromoted):
        executar_pipeline()

    linhas = (
        (ambiente / "metrics" / "training_history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(linhas) == 2
    assert json.loads(linhas[0])["promovido"] is True
    assert json.loads(linhas[1])["promovido"] is False
    assert json.loads(linhas[1])["motivos"]


def test_metricas_descrevem_o_modelo_publicado(ambiente):
    # metrics.json não pode descrever um candidato recusado enquanto outro modelo serve.
    resultado = executar_pipeline()
    metricas = json.loads((ambiente / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    _, metadados = carregar_modelo(ambiente / "models" / "model.joblib")

    assert metricas["candidato"] == resultado["candidato"]
    assert metadados["candidato"] == metricas["candidato"]
    assert metadados["auc_pr"] == pytest.approx(metricas["auc_pr"])


def test_metricas_nao_sao_reescritas_por_candidato_recusado(ambiente):
    # O arquivo de métricas descreve quem está servindo. Gravá-lo antes do gate faria ele
    # descrever um candidato que nunca foi publicado — e ninguém notaria lendo o JSON.
    executar_pipeline()
    metricas = ambiente / "metrics" / "metrics.json"
    antes = metricas.read_bytes()

    with pytest.raises(ModelNotPromoted):
        executar_pipeline()

    assert metricas.read_bytes() == antes


def test_contrato_reprovado_interrompe_antes_de_treinar(ambiente, monkeypatch):
    """O contrato é um bloqueio, não um aviso registrado em log.

    Sem este teste, apagar a linha que valida a Referência não quebraria nada — verificado
    removendo-a: a suíte inteira continuava verde. Como a limpeza e o contrato são a mesma
    política, nenhum dado realista reprova aqui; o jeito honesto de exercitar o bloqueio é
    substituir o validador por um que recusa e exigir que o pipeline pare.
    """

    class _SempreRecusa:
        def validar(self, frame):
            return ValidationResult(
                total=len(frame),
                violacoes=(Violacao(regra="renda_nao_nula", coluna="MonthlyIncome", linhas=1),),
            )

    monkeypatch.setattr("credito.pipeline.training.construir_validador", lambda: _SempreRecusa())

    with pytest.raises(ContratoViolado):
        executar_pipeline()

    # Nada de modelo treinado sobre dado reprovado, e nada publicado.
    assert not (ambiente / "models" / "model.joblib").exists()
    assert not (ambiente / "metrics" / "metrics.json").exists()


def test_motivo_de_regressao_com_texto_de_piso_continua_sendo_skip(ambiente, monkeypatch):
    """A escolha entre falhar e pular lê qual lista trouxe o motivo, não o texto dele.

    A versão anterior decidia procurando "abaixo do piso" na prosa dos motivos, que o gate
    escreve para humano ler no histórico. A Etapa 2 acrescenta critérios de drift, e
    "estabilidade abaixo do piso" é uma frase natural para um deles: pela busca em texto,
    uma não promoção rotineira viraria falha do run e todo pipeline periódico ficaria
    vermelho — sem nada quebrar para avisar. Verificado restaurando a busca por texto:
    este teste fica vermelho e nenhum outro se mexe.
    """
    monkeypatch.setattr(
        "credito.pipeline.training.motivos_de_regressao",
        lambda candidato, incumbente: (
            ["estabilidade 0.1200 abaixo do piso 0.2000"] if incumbente is not None else []
        ),
    )

    executar_pipeline()

    with pytest.raises(ModelNotPromoted):
        executar_pipeline()


def test_piso_violado_falha_o_run_e_nao_publica(ambiente, monkeypatch):
    # O outro desfecho do gate: piso absoluto é defeito, o run tem de falhar e nada pode
    # ser publicado. Um piso de 0,99 em AUC-PR é inalcançável sobre este dado.
    from credito.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CREDITO_MIN_AUC_PR", "0.99")

    with pytest.raises(QualityGateError):
        executar_pipeline()

    assert not (ambiente / "models" / "model.joblib").exists()
    assert not (ambiente / "metrics" / "metrics.json").exists()
    # A execução reprovada ainda precisa deixar rastro: o histórico é o que responde
    # "por que o gate vem reprovando?".
    linhas = (
        (ambiente / "metrics" / "training_history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(linhas) == 1
    assert json.loads(linhas[0])["promovido"] is False


@pytest.mark.parametrize(
    ("erro", "codigo_de_saida"),
    [
        (ModelNotPromoted("auc_pr 0.3716 não supera o modelo em produção (0.3716)"), None),
        (QualityGateError("auc_pr 0.3716 abaixo do piso 0.9900"), 1),
        (
            ContratoViolado("5 de 500 linha(s) reprovadas — renda_nao_nula (MonthlyIncome)"),
            1,
        ),
    ],
)
def test_main_converte_cada_interrupcao_numa_linha_legivel(
    monkeypatch, caplog, erro, codigo_de_saida
):
    """Nenhuma das três formas de o pipeline parar pode escapar como traceback.

    `ContratoViolado` era a que faltava, e era a que mais custava: a mensagem dela carrega
    o relatório de violações inteiro — exatamente o que a camada de contrato existe para
    produzir — e um traceback o enterraria no meio da pilha. Verificado removendo o
    `except ContratoViolado` de `main`: só o caso do contrato fica vermelho.
    """

    def _interrompe(**_):
        raise erro

    monkeypatch.setattr("credito.pipeline.training.executar_pipeline", _interrompe)

    with caplog.at_level(logging.INFO):
        if codigo_de_saida is None:
            main()
        else:
            with pytest.raises(SystemExit) as saida:
                main()
            assert saida.value.code == codigo_de_saida

    assert str(erro) in caplog.text
