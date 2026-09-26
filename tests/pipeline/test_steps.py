"""O gate tem dois desfechos, e a diferença entre eles é o que mantém o pipeline
periódico legível: piso absoluto violado é defeito e precisa **falhar**; candidato válido
que não supera o incumbente é desfecho normal e precisa **pular**.

Se os dois fossem a mesma exceção, o retreino sobre dado estático deixaria o pipeline
vermelho para sempre e o alarme perderia o sentido."""

from __future__ import annotations

import json

from credito.pipeline.steps import (
    ModelNotPromoted,
    QualityGateError,
    deve_promover,
    motivos_de_reprovacao,
    registrar_historico,
)

BONS = {"auc_pr": 0.40, "recall_positivo": 0.70}
PISOS = {"min_auc_pr": 0.30, "min_recall_positivo": 0.60}


def test_candidato_bom_sem_incumbente_e_promovido():
    assert deve_promover(BONS, None, **PISOS) is True


def test_candidato_abaixo_do_piso_absoluto_nao_e_promovido():
    ruim = {"auc_pr": 0.10, "recall_positivo": 0.70}

    assert deve_promover(ruim, None, **PISOS) is False
    assert "auc_pr" in " ".join(motivos_de_reprovacao(ruim, None, **PISOS))


def test_piso_da_metrica_de_negocio_tambem_reprova():
    ruim = {"auc_pr": 0.40, "recall_positivo": 0.20}

    assert deve_promover(ruim, None, **PISOS) is False
    assert "recall_positivo" in " ".join(motivos_de_reprovacao(ruim, None, **PISOS))


def test_empate_com_o_incumbente_nao_promove():
    # Sobre dado estático o retreino reproduz o incumbente. Este é o caso mais comum de
    # todos, e ele não é um erro.
    assert deve_promover(BONS, dict(BONS), **PISOS) is False


def test_candidato_melhor_supera_o_incumbente():
    melhor = {"auc_pr": 0.45, "recall_positivo": 0.70}

    assert deve_promover(melhor, BONS, **PISOS) is True


def test_ganho_tecnico_com_perda_de_negocio_nao_promove():
    # AUC-PR melhor, recall pior: tecnicamente superior, pior para o negócio. O gate
    # precisa enxergar essa assimetria — AUC-PR sozinho não enxerga.
    trocado = {"auc_pr": 0.50, "recall_positivo": 0.55}

    assert deve_promover(trocado, BONS, **PISOS) is False


def test_as_duas_excecoes_sao_independentes():
    # Herança faria um `except QualityGateError` engolir as duas e apagar a distinção
    # inteira. A independência é o ponto, e por isso é testada.
    assert not issubclass(ModelNotPromoted, QualityGateError)
    assert not issubclass(QualityGateError, ModelNotPromoted)


def test_historico_e_append_only(tmp_path):
    caminho = tmp_path / "historico.jsonl"

    registrar_historico(caminho, candidato="xgboost", metricas=BONS, promovido=True, motivos=[])
    registrar_historico(
        caminho,
        candidato="xgboost",
        metricas=BONS,
        promovido=False,
        motivos=["não supera o incumbente"],
    )

    linhas = caminho.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 2
    assert json.loads(linhas[0])["promovido"] is True
    assert json.loads(linhas[1])["motivos"] == ["não supera o incumbente"]


def test_historico_registra_a_recusa_com_motivo(tmp_path):
    # Sem o motivo gravado, "por que o gate vem reprovando?" vira adivinhação.
    caminho = tmp_path / "h.jsonl"
    ruim = {"auc_pr": 0.10, "recall_positivo": 0.70}

    registrar_historico(
        caminho,
        candidato="regressao_logistica",
        metricas=ruim,
        promovido=False,
        motivos=motivos_de_reprovacao(ruim, None, **PISOS),
    )

    registro = json.loads(caminho.read_text(encoding="utf-8").strip())
    assert registro["motivos"]
    assert "carimbo" in registro
