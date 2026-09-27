"""Orquestração do monitoramento: simula a passagem do tempo sobre a partição de teste da
Referência e roda o pipeline de drift, lote a lote, até o veredicto consolidado.

Separado de `training.py` de propósito: treino e monitoramento têm cadências diferentes —
treino roda quando um novo candidato pode substituir o campeão; monitoramento roda
periodicamente sobre o modelo já publicado — e falham por motivos diferentes. O gate de
`pipeline.steps` decide se um MODELO é bom o bastante para publicar; o gate de
`drift.gate` decide se os DADOS que o modelo publicado está vendo ainda se parecem com o
que ele aprendeu. Misturar os dois módulos misturaria dois desfechos que precisam ficar
legíveis separadamente.

**A ordem por lote** é a mesma do enunciado da tarefa: contrato -> predição do campeão ->
PSI/KS próprios -> Evidently (HTML). O contrato roda em TODO lote e precisa passar — é a
demonstração viva de que drift não é invalidez (ver `contracts/base.py`): um lote
deslocado continua sendo dado válido, e só o gate de drift, nunca o contrato, decide se o
deslocamento preocupa. Quando um lote reprova o contrato, o run falha com
`ContratoViolado`, e a causa está no GERADOR (`credito.data.simulate`), nunca no
detector — `main()` traduz essa distinção para quem lê o log, o mesmo padrão que
`pipeline.training.main` já usa para separar as causas de parada.

**A amostra que a simulação desloca é a partição de teste da Referência** — nunca a
Referência inteira. O campeão publicado já viu as linhas de treino; medir degradação
sobre elas seria uma avaliação vazada, otimista por construção, e é a mesma partição
contra a qual `credito.data.simulate` já calibra `_TAXA_MAXIMA_DE_INVERSAO` e contra a
qual `scripts/verificar_degradacao.py` mede a monotonicidade.

**A consolidação de PSI/KS (`drift.gate.avaliar_gate`) só enxerga o quanto a distribuição
de entrada se moveu, nunca o quanto isso custou ao campeão** (ver `drift/gate.py`). Esta
orquestração é o único lugar do projeto com acesso simultâneo ao campeão publicado E ao
processo gerador controlável — por isso é o único lugar capaz de compor as duas leituras
lado a lado: a degradação real (`degradacao_por_lote`) e a decomposição causal por
intervenção (`atribuir_degradacao`) entram no relatório consolidado ANTES do veredicto de
PSI/KS, não depois. Colocar a tabela de PSI primeiro deixaria o olho do leitor ancorar em
"maior número" antes de aprender que PSI alto não é o mesmo que causa mais provável — é
exatamente a armadilha que `_NOTA_DIAGNOSTICO`, em `drift/gate.py`, avisa sem números por
não ter este acesso; `_narrativa_causal_versus_psi`, abaixo, é o lugar do projeto que tem
e usa.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from credito.config import get_settings
from credito.contracts.base import ContratoViolado
from credito.contracts.pandera_backend import construir_validador
from credito.data.arff import ler_arff
from credito.data.prepare import limpar, separar
from credito.data.simulate import MESES, simular_producao
from credito.drift.base import DriftDeFeature, DriftReport, Severidade, classificar
from credito.drift.calibration import degradacao_por_lote
from credito.drift.causal import atribuir_degradacao
from credito.drift.evidently_backend import construir_detector
from credito.drift.gate import CruzamentoDeFeature, GateDeDrift, avaliar_gate
from credito.drift.statistics import ks, psi
from credito.model.evaluate import avaliar
from credito.model.train import carregar_modelo
from credito.schema import FEATURES

logger = logging.getLogger(__name__)

# Mapeia cada variável que `credito.data.simulate` de fato desloca (`VARIAVEIS_COM_DRIFT`)
# ao canal correspondente na atribuição causal por intervenção (`credito.drift.causal`),
# para narrar qual fração da degradação medida uma feature de PSI alto de fato explica.
# `test_canal_por_feature_cobre_exatamente_as_variaveis_com_drift` trava que este mapa
# nunca fique defasado de `VARIAVEIS_COM_DRIFT` — o mesmo tipo de guarda que
# `contracts/pandera_backend.py` já usa entre `_REGRA_POR_COLUNA` e `rules.REGRAS`.
_CANAL_POR_FEATURE: dict[str, str] = {
    "MonthlyIncome": "renda",
    "DebtRatio": "divida",
    "NumberOfTime30-59DaysPastDueNotWorse": "atraso",
}


def _relatorio_proprio(lote: str, referencia: Any, atual: Any) -> DriftReport:
    """PSI e KS calculados com `drift/statistics.py` — a conta auditável à mão que
    alimenta o gate. Independente do Evidently, que roda à parte só para produzir o HTML
    (ver o docstring de `drift/evidently_backend.py` sobre por que os dois números não
    precisam bater)."""
    features = tuple(
        DriftDeFeature(
            feature=feature,
            psi_divergencia=(
                valor_psi := psi(referencia[feature].to_numpy(), atual[feature].to_numpy())
            ),
            ks_p_valor=ks(referencia[feature].to_numpy(), atual[feature].to_numpy()).p_valor,
            severidade=classificar(valor_psi),
        )
        for feature in FEATURES
    )
    return DriftReport(lote=lote, features=features)


def _severidade_para_texto(severidade: Severidade) -> str:
    return severidade.name


def _cruzamento_para_dict(cruzamento: CruzamentoDeFeature) -> dict[str, Any]:
    return {
        "feature": cruzamento.feature,
        "primeiro_lote_atencao": cruzamento.primeiro_lote_atencao,
        "primeiro_lote_critico": cruzamento.primeiro_lote_critico,
        "severidade_final": _severidade_para_texto(cruzamento.severidade_final),
    }


def _gate_para_dict(gate: GateDeDrift) -> dict[str, Any]:
    return {
        "severidade": _severidade_para_texto(gate.severidade),
        "alfa": gate.alfa,
        "recomenda_retreino": gate.recomenda_retreino,
        "mensagem": gate.mensagem,
        "resumo": gate.resumo,
        "cruzamentos": [_cruzamento_para_dict(cruzamento) for cruzamento in gate.cruzamentos],
    }


def _drift_report_para_dict(relatorio: DriftReport) -> dict[str, Any]:
    return {
        "lote": relatorio.lote,
        "severidade_maxima": _severidade_para_texto(relatorio.severidade_maxima),
        "features": [
            {
                "feature": feature.feature,
                "psi_divergencia": feature.psi_divergencia,
                "ks_p_valor": feature.ks_p_valor,
                "severidade": _severidade_para_texto(feature.severidade),
            }
            for feature in relatorio.features
        ],
    }


def _narrativa_causal_versus_psi(
    relatorio_final: DriftReport, atribuicao_final: dict[str, Any]
) -> str:
    """Amarra, no último lote da janela, o veredicto de PSI ao custo real medido pela
    atribuição causal — o texto que esta orquestração tem condição de escrever (tem acesso
    ao campeão publicado e ao gerador controlável) e que `drift/gate.py` deliberadamente
    não escreve (não tem, ver `_NOTA_DIAGNOSTICO` naquele módulo). Todo número aqui é
    medido nesta mesma execução — nunca citado de memória de uma execução anterior.
    """
    maior_psi = max(relatorio_final.features, key=lambda feature: feature.psi_divergencia)
    canal = _CANAL_POR_FEATURE.get(maior_psi.feature)

    linhas = [
        f"Lote {relatorio_final.lote}: a feature de maior PSI é {maior_psi.feature} "
        f"(psi={maior_psi.psi_divergencia:.4f})."
    ]

    degradacao = atribuicao_final["degradacao_por_cenario"]
    efeito_conjunto = atribuicao_final["efeito_conjunto"]
    # Uma fração só faz sentido sobre degradação de fato medida (efeito_conjunto > 0) e
    # sobre uma feature que a atribuição causal sabe decompor — as sete features que
    # `credito.data.simulate` nunca desloca não têm canal isolado nenhum.
    if canal is not None and efeito_conjunto > 0:
        fracao_da_feature = degradacao[canal] / efeito_conjunto
        fracao_do_concept = degradacao["concept_drift"] / efeito_conjunto
        linhas.append(
            f"Na atribuição causal por intervenção do mesmo lote, o canal {canal!r} (o "
            f"que {maior_psi.feature} desloca) responde por {fracao_da_feature:.1%} da "
            f"queda de {atribuicao_final['metrica']} medida contra o campeão publicado; "
            f"o concept drift isolado responde por {fracao_do_concept:.1%}."
        )
        if fracao_do_concept > fracao_da_feature:
            linhas.append(
                "A feature de maior PSI não é a causa mais provável da degradação medida "
                "neste lote — PSI mede onde a distribuição se moveu, não quanto isso "
                "custou (ver credito.drift.causal.atribuir_degradacao)."
            )
    else:
        linhas.append(
            "Degradação não mensurável neste lote (efeito_conjunto <= 0) — a comparação "
            "entre PSI e causa fica sem base neste ponto da janela."
        )

    linhas.append(
        "Nota: PSI e KS são diagnóstico — onde a distribuição de entrada se moveu. A "
        "atribuição causal por intervenção é o custo real medido contra o campeão "
        "publicado. Os dois não precisam apontar para a mesma variável."
    )
    return "\n".join(linhas)


def _gravar_relatorio_consolidado(diretorio: Path, resultado: dict[str, Any]) -> Path:
    """Grava o resumo consolidado (gate, degradação, atribuição causal e narrativa) em
    JSON ao lado dos HTML por lote — o mesmo padrão de `model.evaluate.salvar_metricas`
    para o treino. `/reports/` é git-ignored: o arquivo é reproduzido a cada run, não um
    artefato versionado.
    """
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / "monitoramento.json"

    serializavel = {
        "candidato": resultado["candidato"],
        "amostra_linhas": resultado["amostra_linhas"],
        "lotes": {
            nome: {
                "linhas": dados["linhas"],
                "metricas_campeao": dados["metricas_campeao"],
                "drift_proprio": _drift_report_para_dict(dados["drift_proprio"]),
                "drift_evidently": _drift_report_para_dict(dados["drift_evidently"]),
                "relatorio_html": str(dados["relatorio_html"]),
            }
            for nome, dados in resultado["lotes"].items()
        },
        "degradacao_por_lote": resultado["degradacao_por_lote"],
        "atribuicao_causal_por_lote": resultado["atribuicao_causal_por_lote"],
        "gate": _gate_para_dict(resultado["gate"]),
        "narrativa": resultado["narrativa"],
    }
    caminho.write_text(
        json.dumps(serializavel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return caminho


def executar_monitoramento(*, meses: int = MESES, seed: int) -> dict[str, Any]:
    """Roda o pipeline de drift completo, lote a lote, e devolve o veredicto consolidado.

    Levanta `ContratoViolado` se a própria Referência, ou algum dos `meses` lotes
    simulados, reprovar o contrato de dados. O mês 0 (a amostra sem nenhum drift) não
    entra no laço de `meses` lotes por definição: ele é idêntico à amostra que já foi
    validada antes de simular. Um lote reprovado aqui é um defeito do GERADOR
    (`credito.data.simulate`), nunca do detector de drift — `main()` traduz essa
    distinção para quem lê o log.
    """
    settings = get_settings()
    validador = construir_validador()
    detector = construir_detector()

    referencia, _ = limpar(ler_arff(settings.dataset_path))
    # A Referência precisa passar no próprio contrato antes de qualquer coisa — o mesmo
    # bloqueio que `pipeline.training.executar_pipeline` já aplica.
    validador.validar(referencia[list(FEATURES)]).erguer()

    particoes = separar(
        referencia,
        test_size=settings.test_size,
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )
    amostra = particoes["teste"]

    modelo, metadados = carregar_modelo(settings.model_path)

    lotes = simular_producao(amostra, meses=meses, seed=seed)
    nomes_dos_lotes = [f"mes_{mes:02d}" for mes in range(1, meses + 1)]

    relatorios_proprios: list[DriftReport] = []
    relatorio_por_lote: dict[str, dict[str, Any]] = {}

    for nome in nomes_dos_lotes:
        lote = lotes[nome]

        # Contrato primeiro: se este lote reprovar, o run para aqui — nenhuma predição,
        # nenhum PSI/KS e nenhum HTML são calculados sobre dado que o próprio contrato de
        # ingestão rejeitaria (drift não é invalidez; isto não é um lote com drift, é um
        # defeito do gerador).
        validador.validar(lote[list(FEATURES)]).erguer()

        metricas_do_campeao = avaliar(modelo, lote)
        relatorio_proprio = _relatorio_proprio(nome, amostra, lote)
        relatorios_proprios.append(relatorio_proprio)
        relatorio_evidently = detector.detectar(amostra, lote, lote=nome)

        relatorio_por_lote[nome] = {
            "linhas": int(len(lote)),
            "metricas_campeao": metricas_do_campeao,
            "drift_proprio": relatorio_proprio,
            "drift_evidently": relatorio_evidently,
            "relatorio_html": settings.reports_dir / f"drift_{nome}.html",
        }

    gate = avaliar_gate(relatorios_proprios)

    lotes_simulados = {nome: lotes[nome] for nome in nomes_dos_lotes}
    degradacao = degradacao_por_lote(modelo, lotes_simulados)

    atribuicao_causal = {
        f"mes_{mes:02d}": atribuir_degradacao(amostra, modelo, mes=mes, seed=seed)
        for mes in range(1, meses + 1)
    }

    narrativa = _narrativa_causal_versus_psi(
        relatorios_proprios[-1], atribuicao_causal[f"mes_{meses:02d}"]
    )

    resultado: dict[str, Any] = {
        "candidato": metadados.get("candidato") if isinstance(metadados, dict) else None,
        "amostra_linhas": int(len(amostra)),
        "lotes": relatorio_por_lote,
        "degradacao_por_lote": degradacao,
        "atribuicao_causal_por_lote": atribuicao_causal,
        "gate": gate,
        "narrativa": narrativa,
    }

    _gravar_relatorio_consolidado(settings.reports_dir, resultado)

    return resultado


def main() -> None:
    """Ponto de entrada do `make monitor`.

    `ContratoViolado` aqui tem uma causa diferente da que `pipeline.training.main` trata:
    lá, é a Referência real que reprova (defeito na limpeza ou no dado bruto); aqui pode
    ser um LOTE SIMULADO que reprova — a simulação (`credito.data.simulate`) produzindo
    dado que o próprio contrato rejeitaria, o oposto do que a tarefa pede ("drift não é
    invalidez"). O log separa as duas causas para quem lê não confundir "o dado real está
    ruim" com "o gerador de produção simulada tem um defeito".
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    try:
        resultado = executar_monitoramento(seed=settings.random_seed)
    except ContratoViolado as erro:
        logger.error(
            "a simulação de produção gerou um lote que reprova o contrato de dados — "
            "defeito no gerador (credito.data.simulate), não drift: %s",
            erro,
        )
        sys.exit(1)
    logger.info("gate de drift: %s", resultado["gate"].mensagem)


if __name__ == "__main__":
    main()
