"""Consolida os `DriftReport` de vários lotes numa única decisão e recomendação — a
última fronteira antes de um humano decidir se retreina o campeão.

**A distinção que este módulo herda de `drift/base.py` e estende:** drift nunca é falha
de pipeline. O contrato (`contracts/base.py`) bloqueia a ingestão de dado inválido, e essa
decisão já aconteceu antes deste módulo existir — um lote que chega até aqui já passou no
contrato. Este gate responde a uma pergunta diferente ("este lote se parece com o que o
modelo aprendeu, e o suficiente para preocupar?"), e "sim, preocupa" é um alerta a ler, não
uma exceção a propagar. `avaliar_gate` nunca ergue exceção pela severidade do resultado —
só pela forma de entrada estar quebrada (lista vazia, ver abaixo).

**O risco herdado da Etapa 1 que este módulo não reintroduz:** um gate de qualidade
anterior roteava o desfecho do treino (falhar o run vs. pular o run) buscando uma
substring dentro de frases de auditoria em português — removido na revisão final porque
uma razão de drift redigida do mesmo jeito que uma razão de piso absoluto seria
classificada errado, sem nada acusar. Aqui a severidade é sempre `Severidade`, e todo
roteamento deste módulo (`_MENSAGENS`, `recomenda_retreino`, o `if significativo` dentro de
`avaliar_gate`) decide a partir do enum ou do booleano de `benjamini_hochberg` — nunca de
uma comparação de texto.

**Por que Benjamini-Hochberg roda antes da severidade valer alguma coisa, não depois.**
`DriftDeFeature.severidade` (Tarefa 4) já é travada por `classificar(psi_divergencia)` e
por isso este módulo não pode — nem deveria — reescrever essa severidade. O que ele faz é
computar uma severidade EFETIVA por par lote x feature: a severidade original quando o teste
de KS da mesma feature, dentro do PRÓPRIO lote, sobrevive à correção de múltiplos testes,
rebaixada para `ESTAVEL` quando não sobrevive. PSI mede o TAMANHO do deslocamento; KS testa
se ele é estatisticamente distinguível de acaso amostral, e rodar dez testes por lote a
alfa=0,05 cada, sem correção, infla a chance de ao menos um falso positivo POR LOTE bem
acima de 0,05 (e, ao longo de seis lotes — sessenta comparações no total —, produz em
expectativa ~3 falsos positivos na janela inteira) — relatar isso como "três features
sofreram drift" seria o erro estatístico elementar que `benjamini_hochberg`
(`drift/calibration.py`) existe para evitar.

**A família de correção é o LOTE, não a janela inteira que este gate recebe.** Cada lote
roda sua própria chamada de `benjamini_hochberg`, com sua própria família (só as features
medidas NAQUELE lote), independente dos demais — não uma família única com todos os
`relatorios` achatados juntos. A razão não é só de forma: é inferencial. Um monitor de
produção decide um lote de cada vez, e a decisão sobre o lote 1 só pode usar o que existia
quando o lote 1 chegou — corrigir sobre a janela inteira usaria o p-valor do lote 6 (que
ainda não existe no momento em que o lote 1 precisa de uma decisão) para classificar o
lote 1, um vazamento de informação do futuro para o passado que uma correção por lote não
comete. Pooling também é estritamente mais conservador sempre que os lotes adicionais só
contribuem ruído (p-valores altos): aumentar o tamanho da família sem aumentar o número de
sinais verdadeiros só encolhe o limiar de cada rank, nunca o alarga — o que esconderia
justamente o drift silencioso e precoce que é a narrativa central desta etapa, exatamente
onde a narrativa mais precisa que ele apareça.

A severidade original só pode ser REBAIXADA pela correção, nunca promovida: uma feature
com PSI baixo (`ESTAVEL` por `classificar`) e um p-valor de KS minúsculo continua
`ESTAVEL` — o p-valor testa se a distribuição mudou, não decide POR SI que ela mudou o
bastante para preocupar; quem decide isso é o PSI, e um KS significativo sobre uma
divergência pequena não é raro (basta lote grande o bastante).

**O mês em que cada feature cruzou cada limiar.** Uma feature pode cruzar para atenção,
recuperar e cruzar de novo — a decisão explícita deste módulo é que `primeiro_lote_atencao`
e `primeiro_lote_critico` registram sempre o PRIMEIRO cruzamento, nunca o mais recente; o
estado mais recente é o que `severidade_final` responde (ver `CruzamentoDeFeature` e
`_primeiro_lote_em`). E a severidade agregada do gate (`GateDeDrift.severidade`) é o
máximo entre TODOS os lotes da janela, não só do último: um episódio crítico que se
recuperou antes do fim da janela continua tendo acontecido, e esconder isso atrás do
estado final do último lote seria a mesma armadilha que `DriftReport.severidade_maxima`
(Tarefa 4) já resolve dentro de um único lote — aqui só se estende a mesma regra a vários.

**Drift de feature é diagnóstico, não o alarme.** PSI e KS medem o deslocamento da
distribuição de entrada (P(X)); não medem, por si, quanto esse deslocamento custa ao
campeão. A atribuição causal por intervenção (`credito.drift.causal.atribuir_degradacao`)
existe precisamente para medir esse custo — isolando quanto cada variável desloca o
desempenho quando as demais ficam fixas —, e mediu, contra o campeão real desta etapa, que
a feature de maior PSI não é a que mais move a degradação (a decomposição está registrada
no relatório dessa tarefa, não repetida aqui). Um leitor que visse só a tabela de PSI seria
levado a concluir o oposto. `GateDeDrift.resumo` carrega essa ressalva qualitativa
embutida (ver `_NOTA_DIAGNOSTICO`), para que a advertência viaje com o relatório mesmo
para quem nunca abriu este docstring — deliberadamente SEM os números: este gate não tem
acesso ao campeão publicado nem ao dado real (nenhum teste desta suíte toca rede ou
dataset), e citar aqui um número que só `atribuir_degradacao`/`degradacao_por_lote` têm
como medir seria o gate afirmando uma medição que ele nunca fez — e que um retreino do
campeão tornaria silenciosamente obsoleta, sem que nada nesta suíte notasse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from credito.drift.base import DriftReport, Severidade
from credito.drift.calibration import benjamini_hochberg

_MENSAGENS: dict[Severidade, str] = {
    Severidade.ESTAVEL: "estável — nada a fazer",
    Severidade.ATENCAO: "alerta — observar, sem ação automática",
    Severidade.CRITICO: "alerta crítico — recomenda-se avaliar retreino do modelo",
}

# Deliberadamente sem números: um valor fixo aqui (ex.: "renda responde por 0,8% da
# degradação") apodreceria a cada retreino do campeão, cada mudança na região de concept
# drift ou cada refresh do dataset, e nenhum teste desta suíte notaria a divergência —
# pior ainda, o gate estaria emitindo, no seu próprio relatório, uma medição que ele nunca
# fez (não tem acesso ao campeão nem ao dado real). Os números reais moram em
# `credito.drift.causal.atribuir_degradacao` e em
# `credito.drift.calibration.degradacao_por_lote` — quem compõe o relatório final com os
# dois lado a lado é a orquestração, não este gate.
_NOTA_DIAGNOSTICO = (
    "Nota: PSI e KS medem o deslocamento da distribuição de entrada (P(X)), não o custo "
    "desse deslocamento para o campeão. Este relatório é diagnóstico: aponta ONDE a "
    "distribuição se moveu, não QUANTO isso custou ao modelo — e a feature de PSI mais "
    "alto não é, por isso, a causa mais provável da degradação. Quem mede o custo real é "
    "credito.drift.causal.atribuir_degradacao (decomposição por variável, por "
    "intervenção) e credito.drift.calibration.degradacao_por_lote (a consequência "
    "agregada por lote)."
)


@dataclass(frozen=True)
class CruzamentoDeFeature:
    """O mês em que uma feature entrou em cada banda de severidade, pela primeira vez —
    e o estado em que a janela termina.

    `primeiro_lote_atencao`/`primeiro_lote_critico` guardam sempre o PRIMEIRO cruzamento:
    se a feature recupera e cruza de novo, este campo não muda para o segundo episódio
    (ver `_primeiro_lote_em`, e o docstring do módulo sobre por que essa é a leitura
    escolhida). `severidade_final` é o único campo que responde "e agora, no último lote
    da janela?" — os dois juntos respondem "quando começou" e "onde está hoje" sem
    confundir uma pergunta com a outra.

    Um `None` em qualquer um dos dois campos de mês significa "nunca cruzou esta banda
    dentro da janela avaliada" — não "cruzou no lote zero" nem qualquer outro valor que
    pudesse ser confundido com um rótulo de lote real.
    """

    feature: str
    primeiro_lote_atencao: str | None
    primeiro_lote_critico: str | None
    severidade_final: Severidade


def _narrar_cruzamento(cruzamento: CruzamentoDeFeature) -> str | None:
    """Uma linha de narrativa por feature que já cruzou para atenção — features que
    nunca deixaram `ESTAVEL` não aparecem, para o resumo não afogar o que importa em
    features que não disseram nada a respeito de drift."""
    if cruzamento.primeiro_lote_atencao is None:
        return None
    if cruzamento.primeiro_lote_critico is not None:
        estado = f"tornou-se crítica em {cruzamento.primeiro_lote_critico}"
    elif cruzamento.severidade_final is Severidade.ESTAVEL:
        estado = "recuperou depois, sem nunca ter ficado crítica"
    else:
        estado = "nunca ficou crítica"
    return (
        f"{cruzamento.feature}: entrou em atenção em {cruzamento.primeiro_lote_atencao} "
        f"— {estado}"
    )


@dataclass(frozen=True)
class GateDeDrift:
    """O desfecho consolidado de `avaliar_gate`: uma `Severidade` (o veredito, sempre
    lido como enum — ver o docstring do módulo sobre por que nunca é texto), o `alfa`
    usado para corrigir os p-valores de KS de cada lote (a correção roda por lote, não
    sobre a janela inteira — ver o docstring do módulo), e um `CruzamentoDeFeature` por
    feature vista em qualquer lote da janela."""

    severidade: Severidade
    alfa: float
    cruzamentos: tuple[CruzamentoDeFeature, ...]

    @property
    def recomenda_retreino(self) -> bool:
        """Só `True` quando a severidade é `CRITICO` — a única banda do brief que carrega
        recomendação explícita de retreino. `ATENCAO` é "observar", não "agir"."""
        return self.severidade is Severidade.CRITICO

    @property
    def mensagem(self) -> str:
        """Uma linha por veredito, lida do enum (`_MENSAGENS`), nunca escrita a partir de
        comparação de texto."""
        return _MENSAGENS[self.severidade]

    @property
    def resumo(self) -> str:
        """A narrativa que transforma a tabela de severidades em texto — o que o brief
        pede explicitamente ("é o que transforma o relatório em narrativa"). Sempre
        termina com `_NOTA_DIAGNOSTICO`, para que a ressalva "PSI alto não é o mesmo que
        causa mais provável" viaje junto com o relatório, não só com quem leu o
        docstring do módulo."""
        linhas = [self.mensagem]
        for cruzamento in self.cruzamentos:
            texto = _narrar_cruzamento(cruzamento)
            if texto is not None:
                linhas.append(texto)
        linhas.append(_NOTA_DIAGNOSTICO)
        return "\n".join(linhas)


def _primeiro_lote_em(historico: list[tuple[str, Severidade]], patamar: Severidade) -> str | None:
    """Primeiro lote, na ordem de chegada em `historico`, cuja severidade efetiva atinge
    ou supera `patamar`. Percorrer em ordem e devolver no primeiro achado é o que garante
    a decisão do módulo: uma recuperação e um recruzamento posteriores nunca sobrescrevem
    este valor, porque a busca já parou no primeiro."""
    for lote, severidade in historico:
        if severidade.value >= patamar.value:
            return lote
    return None


def avaliar_gate(relatorios: Sequence[DriftReport], *, alfa: float = 0.05) -> GateDeDrift:
    """Consolida `relatorios` (um `DriftReport` por lote, em ordem cronológica) numa
    única `GateDeDrift`.

    A ordem de `relatorios` importa e não é verificada: este módulo confia que o chamador
    passa os lotes na ordem em que aconteceram, a mesma confiança que
    `credito.drift.calibration.distribuicao_nula_psi` deposita em quem monta a série que
    lhe entrega. Sem essa ordem, "primeiro cruzamento" não tem sentido — inverter a
    entrada inverteria qual cruzamento conta como "primeiro".

    Levanta `ValueError` só quando `relatorios` está vazio — não há lote nenhum para
    consolidar, o mesmo tipo de estado inválido que `DriftReport.__post_init__` já
    rejeita um nível abaixo (um relatório sem nenhuma feature). Qualquer severidade de
    resultado, incluindo `CRITICO`, volta como valor lido — nunca como exceção (ver o
    docstring do módulo).
    """
    if not relatorios:
        raise ValueError(
            "avaliar_gate precisa de ao menos um DriftReport — nenhum lote para consolidar"
        )

    historico_por_feature: dict[str, list[tuple[str, Severidade]]] = {}
    severidades_efetivas: list[Severidade] = []
    for relatorio in relatorios:
        # A família de Benjamini-Hochberg é ESTE lote — só as features medidas nele —,
        # não os `relatorios` inteiros achatados juntos (ver o docstring do módulo sobre
        # por que corrigir pela janela inteira vazaria informação de lotes futuros para
        # a decisão de um lote passado).
        p_valores = np.array([feature.ks_p_valor for feature in relatorio.features])
        significativos = benjamini_hochberg(p_valores, alfa=alfa)
        for feature, significativo in zip(relatorio.features, significativos, strict=True):
            # A correção só pode rebaixar: a severidade original já é o teto que
            # `classificar` sustenta a partir do PSI (ver o docstring do módulo).
            efetiva = feature.severidade if significativo else Severidade.ESTAVEL
            historico_por_feature.setdefault(feature.feature, []).append((relatorio.lote, efetiva))
            severidades_efetivas.append(efetiva)

    cruzamentos = tuple(
        CruzamentoDeFeature(
            feature=nome,
            primeiro_lote_atencao=_primeiro_lote_em(historico, Severidade.ATENCAO),
            primeiro_lote_critico=_primeiro_lote_em(historico, Severidade.CRITICO),
            severidade_final=historico[-1][1],
        )
        for nome, historico in historico_por_feature.items()
    )

    severidade = max(severidades_efetivas, key=lambda s: s.value)

    return GateDeDrift(severidade=severidade, alfa=alfa, cruzamentos=cruzamentos)
