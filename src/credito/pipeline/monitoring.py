"""Orquestração do monitoramento: simula a passagem do tempo sobre a partição de teste da
Referência e roda o pipeline de drift, lote a lote, até o veredicto consolidado.

Separado de `training.py` de propósito: treino e monitoramento têm cadências diferentes —
treino roda quando um novo candidato pode substituir o campeão; monitoramento roda
periodicamente sobre o modelo já publicado — e falham por motivos diferentes. O gate de
`pipeline.steps` decide se um MODELO é bom o bastante para publicar; o gate de
`drift.gate` decide se os DADOS que o modelo publicado está vendo ainda se parecem com o
que ele aprendeu. Misturar os dois módulos misturaria dois desfechos que precisam ficar
legíveis separadamente.

**A ordem por lote** é deliberada: contrato -> predição do campeão ->
PSI/KS próprios -> Evidently (HTML). O contrato roda em TODO lote e precisa passar — é a
demonstração viva de que drift não é invalidez (ver `contracts/base.py`): um lote
deslocado continua sendo dado válido, e só o gate de drift, nunca o contrato, decide se o
deslocamento preocupa.

**Duas causas diferentes levantam `ContratoViolado` aqui, e `main()` só consegue apontar a
certa porque o código as distingue por TIPO, nunca por texto** (o mesmo princípio que
`pipeline.steps` já aplica entre `QualityGateError` e `ModelNotPromoted`). Se a própria
Referência reprovar — antes de qualquer simulação —, a causa é o arquivo real
(desatualizado ou corrompido): `ReferenciaInvalida` (abaixo, subclasse de
`ContratoViolado`) carrega essa origem. Se um LOTE SIMULADO reprovar, a causa é o GERADOR
(`credito.data.simulate`) produzindo dado que o próprio contrato rejeitaria — o oposto do
princípio desta camada ("drift não é invalidez") —, e continua sendo um `ContratoViolado`
comum.
`main()` tem um `except` para cada uma, na ordem certa (a subclasse primeiro), e cada um
loga a causa que de fato se aplica.

**A amostra que a simulação desloca é a partição de teste da Referência** — nunca a
Referência inteira. O campeão publicado já viu as linhas de treino; medir degradação
sobre elas seria uma avaliação vazada, otimista por construção, e é a mesma partição
contra a qual `credito.data.simulate` já calibra `_TAXA_MAXIMA_DE_INVERSAO` e contra a
qual `scripts/verificar_degradacao.py` mede a monotonicidade.

**A consolidação de PSI/KS (`drift.gate.avaliar_gate`) só enxerga o quanto a distribuição
de entrada se moveu, nunca o quanto isso custou ao campeão** (ver `drift/gate.py`). Esta
orquestração é o único lugar do projeto com acesso simultâneo ao campeão publicado E ao
processo gerador controlável — por isso é o único lugar capaz de compor as duas leituras
lado a lado: a degradação real (a mesma conta de `credito.drift.calibration.
degradacao_por_lote` — `avaliar(modelo, lote)` —, reaproveitada das métricas já calculadas
por lote em vez de recalculada) e a decomposição causal por intervenção
(`atribuir_degradacao`) entram no relatório consolidado ANTES do veredicto de
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
from credito.drift.causal import atribuir_degradacao
from credito.drift.evidently_backend import construir_detector
from credito.drift.gate import CruzamentoDeFeature, GateDeDrift, avaliar_gate
from credito.drift.statistics import ks, psi
from credito.model.evaluate import avaliar
from credito.model.train import carregar_modelo
from credito.monitoring.proxies import sinais_do_lote
from credito.schema import FEATURES
from credito.tracking.execucao import Execucao, execucao_rastreada
from credito.tracking.mlflow_client import (
    parametros_da_execucao,
    passo_do_lote,
    registrar_no_run_ativo,
)

logger = logging.getLogger(__name__)


class ReferenciaInvalida(ContratoViolado):
    """A própria Referência (antes de qualquer simulação) reprovou o contrato de dados —
    causa real: arquivo desatualizado ou corrompido, nunca o simulador de produção
    (`credito.data.simulate`), que ainda nem rodou neste ponto do pipeline.

    Subclasse de `ContratoViolado`, não um tipo à parte: um chamador que só soubesse lidar
    com `ContratoViolado` continua funcionando sem precisar conhecer esta subclasse. A
    razão de existir é permitir que `main()` distinga as duas causas pelo TIPO da exceção
    — nunca remontando a mesma armadilha que `pipeline.steps` já documenta (rotear por
    substring de mensagem, que classifica errado assim que duas causas escrevem frases
    parecidas).
    """


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
    # Duas razões distintas podem impedir a fração de fazer sentido, e a mensagem precisa
    # dizer qual das duas é a de fato — uma mensagem única para as duas afirmaria "não
    # mensurável" mesmo quando a degradação era perfeitamente mensurável e o único
    # problema era a feature não ter canal isolado (as sete que `credito.data.simulate`
    # nunca desloca).
    if canal is None:
        linhas.append(
            f"{maior_psi.feature} não tem canal isolado na atribuição causal por "
            "intervenção — a comparação entre PSI e causa não se aplica a ela."
        )
    elif efeito_conjunto <= 0:
        linhas.append(
            "Degradação não mensurável neste lote (efeito_conjunto <= 0) — a comparação "
            "entre PSI e causa fica sem base neste ponto da janela."
        )
    else:
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

    # A ORDEM DAS CHAVES É PARTE DO RELATÓRIO, não detalhe de serialização: `json.dumps`
    # preserva a ordem de inserção do dict, e é essa ordem que o leitor humano encontra ao
    # abrir o arquivo. `degradacao_por_lote` e `atribuicao_causal_por_lote` vêm ANTES de
    # `lotes` (que carrega as tabelas de PSI/KS) pelo mesmo motivo que o docstring deste
    # módulo e a seção de monitoramento do README dão: quem lê a tabela de PSI primeiro
    # ancora em "maior número" e conclui que a feature de PSI mais alto é a causa mais
    # provável — que nesta execução ela não é. Inverter as duas aqui tornaria falsa uma
    # afirmação feita em três documentos (este módulo, o README e `docs/model_card.md`).
    serializavel = {
        "candidato": resultado["candidato"],
        "amostra_linhas": resultado["amostra_linhas"],
        "degradacao_por_lote": resultado["degradacao_por_lote"],
        "atribuicao_causal_por_lote": resultado["atribuicao_causal_por_lote"],
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
        "gate": _gate_para_dict(resultado["gate"]),
        "narrativa": resultado["narrativa"],
    }
    caminho.write_text(
        json.dumps(serializavel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return caminho


def executar_monitoramento(*, meses: int = MESES, seed: int) -> dict[str, Any]:
    """Roda o pipeline de drift completo, lote a lote, e devolve o veredicto consolidado.

    Levanta `ReferenciaInvalida` se a própria Referência reprovar o contrato (arquivo real
    desatualizado ou corrompido), e `ContratoViolado` (o tipo base, não a subclasse) se
    algum dos `meses` lotes SIMULADOS reprovar — aí a causa é o gerador
    (`credito.data.simulate`), nunca o dado real. O mês 0 (a amostra sem nenhum drift) não
    entra no laço de `meses` lotes por definição: ele é idêntico à amostra que já foi
    validada antes de simular. `main()` tem um `except` para cada uma dessas duas causas
    (ver o docstring do módulo).
    """
    settings = get_settings()
    # O run do MLflow abre aqui, antes de qualquer tarefa: um lote reprovado no meio deixa
    # o run `FAILED` com a causa e o mês, em vez de nenhum rastro (`credito.tracking.execucao`).
    with execucao_rastreada(
        tracking_uri=settings.mlflow_tracking_uri,
        experimento=settings.mlflow_experimento,
        artifact_location=settings.mlflow_artifact_location,
        pipeline="monitoramento",
    ) as execucao:
        resultado = _executar(settings, execucao, meses=meses, seed=seed)
        execucao.concluir("concluido")
        return resultado


def _executar(settings: Any, execucao: Execucao, *, meses: int, seed: int) -> dict[str, Any]:
    validador = construir_validador()
    detector = construir_detector()

    bruto = ler_arff(settings.dataset_path)
    referencia, _ = limpar(bruto)
    execucao.tarefa(
        "ingestao", volume={"linhas_brutas": len(bruto), "linhas_referencia": len(referencia)}
    )
    # A Referência precisa passar no próprio contrato antes de qualquer coisa — o mesmo
    # bloqueio que `pipeline.training.executar_pipeline` já aplica. Reaproveita a mensagem
    # que `erguer()` já monta (nunca duplica esse formato) e só troca o TIPO da exceção,
    # para que `main()` consiga apontar o arquivo real como causa, não o gerador.
    verificacao = validador.validar(referencia[list(FEATURES)])
    execucao.tarefa(
        "contrato",
        sucesso=verificacao.valido,
        passo=0,
        volume={"linhas_reprovadas": verificacao.linhas_reprovadas},
    )
    try:
        verificacao.erguer()
    except ContratoViolado as erro:
        raise ReferenciaInvalida(str(erro)) from erro

    particoes = separar(
        referencia,
        test_size=settings.test_size,
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )
    amostra = particoes["teste"]

    modelo, metadados = carregar_modelo(settings.model_path)

    lotes = simular_producao(amostra, meses=meses, seed=seed)
    execucao.tarefa("simulacao", volume={"linhas_amostra": len(amostra)})
    nomes_dos_lotes = [f"mes_{mes:02d}" for mes in range(1, meses + 1)]

    relatorios_proprios: list[DriftReport] = []
    relatorio_por_lote: dict[str, dict[str, Any]] = {}
    proxies_por_lote: dict[str, dict[str, float]] = {}

    for nome in nomes_dos_lotes:
        lote = lotes[nome]

        # Contrato primeiro: se este lote reprovar, o run para aqui — nenhuma predição,
        # nenhum PSI/KS e nenhum HTML são calculados sobre dado que o próprio contrato de
        # ingestão rejeitaria (drift não é invalidez; isto não é um lote com drift, é um
        # defeito do gerador).
        mes = passo_do_lote(nome)
        verificacao = validador.validar(lote[list(FEATURES)])
        execucao.tarefa(
            "contrato",
            sucesso=verificacao.valido,
            passo=mes,
            volume={"linhas_lote": len(lote), "linhas_reprovadas": verificacao.linhas_reprovadas},
        )
        verificacao.erguer()

        metricas_do_campeao = avaliar(modelo, lote)
        execucao.tarefa("predicao", passo=mes)
        relatorio_proprio = _relatorio_proprio(nome, amostra, lote)
        relatorios_proprios.append(relatorio_proprio)
        relatorio_evidently = detector.detectar(amostra, lote, lote=nome)
        execucao.tarefa("deteccao_de_drift", passo=mes)
        # `referencia=amostra`: os sinais sem rótulo (`credito.monitoring.proxies`) usam a
        # mesma partição de teste que o resto desta orquestração já usa como linha de base —
        # nunca uma segunda Referência calculada à parte.
        proxies_por_lote[nome] = sinais_do_lote(modelo, lote, referencia=amostra)

        relatorio_por_lote[nome] = {
            "linhas": int(len(lote)),
            "metricas_campeao": metricas_do_campeao,
            "drift_proprio": relatorio_proprio,
            "drift_evidently": relatorio_evidently,
            "relatorio_html": settings.reports_dir / f"drift_{nome}.html",
        }

    gate = avaliar_gate(relatorios_proprios)
    execucao.tarefa("gate")

    # A degradação real por lote já foi calculada dentro do laço acima (`metricas_do_
    # campeao`, a mesma chamada a `avaliar` que `credito.drift.calibration.
    # degradacao_por_lote` faria de novo sobre o mesmo modelo e o mesmo lote) — reaproveitar
    # evita uma segunda passagem redundante e garante que as duas leituras não possam
    # divergir por terem sido calculadas duas vezes.
    degradacao = {nome: dados["metricas_campeao"] for nome, dados in relatorio_por_lote.items()}

    # `meses` vai junto: sem ele, `atribuir_degradacao` cairia no default `MESES` e
    # calcularia `k = mes / MESES` enquanto os lotes acima foram gerados com
    # `k = mes / meses` — a decomposição descreveria um processo gerador que não produziu
    # nenhum dos lotes monitorados, em silêncio (ver o docstring de `drift/causal.py`).
    atribuicao_causal = {
        f"mes_{mes:02d}": atribuir_degradacao(amostra, modelo, mes=mes, seed=seed, meses=meses)
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

    caminho_consolidado = _gravar_relatorio_consolidado(settings.reports_dir, resultado)

    artefatos = [dados["relatorio_html"] for dados in relatorio_por_lote.values()]
    artefatos.append(caminho_consolidado)

    registrar_no_run_ativo(
        parametros=parametros_da_execucao(settings, seed=seed, meses=meses),
        metricas_do_campeao_por_lote=degradacao,
        drift_por_lote=relatorios_proprios,
        proxies_por_lote=proxies_por_lote,
        gate=gate,
        artefatos=artefatos,
    )
    resultado["mlflow_run_id"] = execucao.run_id

    return resultado


def main() -> None:
    """Ponto de entrada do `make monitor`.

    Duas causas diferentes de `ContratoViolado` merecem duas mensagens diferentes (ver o
    docstring do módulo), e a distinção é feita pelo TIPO da exceção, nunca pelo texto: o
    `except ReferenciaInvalida` precisa vir ANTES do `except ContratoViolado` porque
    `ReferenciaInvalida` é subclasse — na ordem inversa, o `except` mais genérico capturaria
    as duas causas e a distinção deixaria de existir na prática, mesmo com os dois blocos
    escritos.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    try:
        resultado = executar_monitoramento(seed=settings.random_seed)
    except ReferenciaInvalida as erro:
        logger.error(
            "a Referência real reprovou o contrato de dados antes de qualquer "
            "simulação — arquivo desatualizado ou corrompido, não um defeito do "
            "simulador (credito.data.simulate): %s",
            erro,
        )
        sys.exit(1)
    except ContratoViolado as erro:
        logger.error(
            "a simulação de produção gerou um lote que reprova o contrato de dados — "
            "defeito no gerador (credito.data.simulate), não drift: %s",
            erro,
        )
        sys.exit(1)
    logger.info("gate de drift: %s", resultado["gate"].mensagem)
    logger.info(
        "execução registrada no MLflow: experimento=%s run_id=%s",
        settings.mlflow_experimento,
        resultado["mlflow_run_id"],
    )


if __name__ == "__main__":
    main()
