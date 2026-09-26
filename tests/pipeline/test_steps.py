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
    motivos_de_piso_absoluto,
    motivos_de_regressao,
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
    # precisa enxergar essa assimetria — AUC-PR sozinho não enxerga. O recall aqui fica
    # acima do piso absoluto (0.60) e abaixo apenas do incumbente (0.70): se caísse sob o
    # piso, a reprovação seria explicada pelo piso, não pela regressão frente ao incumbente,
    # e o teste deixaria de isolar o critério que quer provar.
    trocado = {"auc_pr": 0.50, "recall_positivo": 0.65}

    assert deve_promover(trocado, BONS, **PISOS) is False


def test_reprovacao_por_regressao_nao_aparece_na_lista_de_piso():
    # O ponto da separação: um candidato que só não supera o incumbente não pode deixar
    # nenhum motivo na lista de piso absoluto, porque é ela que o pipeline converte em
    # falha do run. Se as duas listas se misturassem, o retreino sobre dado estático —
    # o caso mais comum de todos — passaria a falhar.
    piso = {"min_auc_pr": 0.30, "min_recall_positivo": 0.60}

    assert motivos_de_piso_absoluto(BONS, **piso) == []
    assert motivos_de_regressao(BONS, dict(BONS)) != []


def test_reprovacao_por_piso_nao_aparece_na_lista_de_regressao():
    # O simétrico: um candidato inutilizável reprova no piso mesmo sem incumbente nenhum,
    # e a lista de regressão fica vazia — não há com o que comparar na primeira execução.
    ruim = {"auc_pr": 0.10, "recall_positivo": 0.20}

    assert motivos_de_piso_absoluto(ruim, **PISOS) != []
    assert motivos_de_regressao(ruim, None) == []


def test_motivos_de_reprovacao_e_a_concatenacao_das_duas_listas():
    # `deve_promover` e o pipeline decidem sobre listas diferentes — um sobre a
    # concatenação, o outro sobre as duas partes. Este teste é o que garante que as duas
    # leituras não podem discordar: um critério acrescentado a só uma das partes e
    # esquecido na composição sairia do relato de auditoria sem sair do desfecho.
    trocado = {"auc_pr": 0.20, "recall_positivo": 0.65}

    assert motivos_de_reprovacao(trocado, BONS, **PISOS) == [
        *motivos_de_piso_absoluto(trocado, **PISOS),
        *motivos_de_regressao(trocado, BONS),
    ]
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
