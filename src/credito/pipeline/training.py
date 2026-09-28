"""Execução local do pipeline de treino, ponta a ponta.

A ordem é a mesma que a orquestração usará: ingestão, contrato, preparo, treino dos
candidatos, avaliação, gate e publicação. O contrato roda **antes** do treino porque essa
é a única ordem que faz sentido — um modelo treinado sobre dado reprovado já nasceu
comprometido.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from credito.config import get_settings
from credito.contracts.base import ContratoViolado
from credito.contracts.pandera_backend import construir_validador
from credito.data.arff import ler_arff
from credito.data.download import baixar_dataset
from credito.data.prepare import limpar, separar
from credito.model.evaluate import avaliar, avaliar_por_faixa_etaria, salvar_metricas
from credito.model.train import (
    TIPOS_DE_MODELO,
    carregar_modelo,
    salvar_modelo,
    treinar,
)
from credito.pipeline.steps import (
    ModelNotPromoted,
    QualityGateError,
    motivos_de_piso_absoluto,
    motivos_de_regressao,
    registrar_historico,
)
from credito.schema import ALVO, FEATURES

logger = logging.getLogger(__name__)


def executar_pipeline(*, force_download: bool = False) -> dict[str, Any]:
    """Roda o pipeline completo e devolve o resumo da execução."""
    settings = get_settings()

    if not settings.dataset_path.exists() or force_download:
        baixar_dataset(
            settings.dataset_path,
            settings.dataset_url,
            settings.dataset_md5,
            force=force_download,
        )

    bruto = ler_arff(settings.dataset_path)
    referencia, descartes = limpar(bruto)
    logger.info("referência: %d linhas | descartes: %s", len(referencia), descartes)

    # A Referência tem de passar no próprio contrato: se a limpeza e as regras
    # divergissem, o modelo aprenderia sobre dado que a ingestão recusaria.
    construir_validador().validar(referencia[list(FEATURES)]).erguer()

    particoes = separar(
        referencia,
        test_size=settings.test_size,
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )

    avaliacoes: dict[str, dict[str, float]] = {}
    modelos = {}
    for tipo in TIPOS_DE_MODELO:
        modelos[tipo] = treinar(particoes["treino"], tipo, seed=settings.random_seed)
        # O campeão é escolhido na validação, nunca no teste: escolher no teste
        # transformaria o conjunto de teste em parte do treino e a métrica publicada
        # viraria otimista por construção.
        avaliacoes[tipo] = avaliar(modelos[tipo], particoes["validacao"])

    campeao = max(avaliacoes, key=lambda tipo: avaliacoes[tipo]["auc_pr"])
    logger.info("campeão na validação: %s (auc_pr %.4f)", campeao, avaliacoes[campeao]["auc_pr"])

    metricas_teste = avaliar(modelos[campeao], particoes["teste"])
    por_faixa = avaliar_por_faixa_etaria(modelos[campeao], particoes["teste"])

    incumbente = None
    if settings.model_path.exists():
        _, metadados = carregar_modelo(settings.model_path)
        incumbente = {
            "auc_pr": metadados["auc_pr"],
            "recall_positivo": metadados["recall_positivo"],
        }

    # As duas listas vêm separadas da origem porque é delas que sai o desfecho: piso
    # absoluto falha o run, regressão frente ao incumbente vira skip. `promovido` é
    # `not motivos` por construção, em vez de uma segunda avaliação dos mesmos critérios
    # que pudesse discordar da primeira.
    piso_violado = motivos_de_piso_absoluto(
        metricas_teste,
        min_auc_pr=settings.min_auc_pr,
        min_recall_positivo=settings.min_recall_positivo,
    )
    regressao = motivos_de_regressao(metricas_teste, incumbente)
    motivos = [*piso_violado, *regressao]
    promovido = not motivos

    # O histórico é gravado antes de qualquer desfecho: uma execução reprovada é
    # exatamente a que alguém vai querer auditar depois, e ela não pode sumir do registro
    # por ter terminado em exceção.
    registrar_historico(
        settings.metrics_dir / "training_history.jsonl",
        candidato=campeao,
        metricas=metricas_teste,
        promovido=promovido,
        motivos=motivos,
    )

    if not promovido:
        # A ordem importa: o piso absoluto é defeito e precisa falhar o run; a não
        # superação do incumbente é desfecho normal e precisa virar skip. A escolha lê
        # qual lista trouxe o motivo, não o texto dele — buscar uma marca em prosa
        # devolveria a decisão a quem escreve a frase do critério, e um critério de drift
        # que dissesse "abaixo do piso" passaria a pintar de vermelho todo run periódico.
        if piso_violado:
            raise QualityGateError("; ".join(piso_violado))
        raise ModelNotPromoted("; ".join(regressao))

    # metrics.json só é gravado na promoção: descrever um candidato recusado enquanto
    # outro modelo está servindo torna o arquivo enganoso.
    salvar_modelo(
        modelos[campeao],
        settings.model_path,
        {
            "candidato": campeao,
            "auc_pr": metricas_teste["auc_pr"],
            "recall_positivo": metricas_teste["recall_positivo"],
            "seed": settings.random_seed,
            "features": list(FEATURES),
            "alvo": ALVO,
        },
    )
    salvar_metricas(
        {
            "candidato": campeao,
            **metricas_teste,
            "validacao": avaliacoes,
            "por_faixa_etaria": por_faixa,
            "referencia": {"linhas": len(referencia), "descartes": descartes},
        },
        settings.metrics_dir / "metrics.json",
    )

    return {
        "candidato": campeao,
        "promovido": True,
        "metricas": metricas_teste,
        "por_faixa_etaria": por_faixa,
    }


def main() -> None:
    """Ponto de entrada do ``make train``.

    Os três desfechos de interrupção do pipeline saem daqui como uma linha legível, nunca
    como traceback: quem lê o log de um run agendado precisa da causa, e um traceback
    esconde a causa no meio da pilha. ``ContratoViolado`` é o que tem mais a perder nisso —
    a mensagem carrega o relatório inteiro de violações, que é justamente o que a camada de
    contrato existe para produzir.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        resumo = executar_pipeline()
    except ModelNotPromoted as motivo:
        # Desfecho normal: sai com 0 para que o agendador não acuse falha a cada retreino
        # sobre dado estático.
        logger.info("nada a promover: %s", motivo)
        return
    except QualityGateError as erro:
        # Defeito: sai com código diferente de zero, mas com uma linha legível em vez de
        # um traceback — a causa já está no histórico e no texto do motivo.
        logger.error("gate de qualidade reprovou o candidato: %s", erro)
        sys.exit(1)
    except ContratoViolado as erro:
        # O dado foi recusado na porta e nada chegou a ser treinado. Também é defeito e
        # também falha o run, mas a causa está no upstream, não no modelo — por isso a
        # mensagem separa os dois casos em vez de dizer só "o run falhou".
        logger.error("contrato de dados reprovou a Referência: %s", erro)
        sys.exit(1)
    logger.info("promovido: %s", resumo["candidato"])


if __name__ == "__main__":
    main()
