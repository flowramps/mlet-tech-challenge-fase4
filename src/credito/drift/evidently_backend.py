"""Backend Evidently: gera o relatório HTML — o entregável visual desta camada — e devolve
um ``DriftReport`` que corrobora a estatística própria de ``drift/statistics.py``.

**Por que existir ao lado de ``drift/statistics.py`` em vez de substituí-lo:** o PSI e o KS
próprios são a conta que um humano confere linha a linha; o Evidently não expõe os cortes
de bin que usa internamente, então não dá para auditar o número da mesma forma — mas ele
produz o HTML que esta camada precisa entregar, e nenhum código deste projeto reimplementa
geração de relatório visual. Os dois convivem atrás da mesma interface (``DriftDetector``,
``drift/base.py``) porque respondem à mesma pergunta por caminhos independentes.

**A armadilha que o preset do Evidently esconde:** ``DataDriftPreset`` sem argumento usa
distância de Wasserstein por padrão — medido rodando o preset vazio, ele devolve
``method='Wasserstein distance (normed)'``, que não é nenhuma das duas medidas que este
projeto adota. Todo ponto de construção de métrica aqui passa ``method="psi"`` ou
``method="ks"`` explicitamente; nenhum caminho depende do default da biblioteca.

**Por que os dois PSI não batem em valor:** ``credito.drift.statistics.psi`` decide a
estratégia de binning medindo colapso de quantil (ver o docstring daquele módulo);
o Evidently faz o próprio binning, com sua própria regra, e não expõe os cortes escolhidos
para comparação. Exigir que os dois números coincidam seria testar um acidente de
implementação, não uma propriedade real — o que os testes deste módulo exigem em vez disso
é que os dois cheguem à mesma ``Severidade`` nos casos inequívocos (dados idênticos:
ESTAVEL nos dois; deslocamento de várias ordens de grandeza fora do suporte observado:
CRITICO nos dois). Nenhuma asserção de igualdade de valor existe aqui, porque seria uma
asserção que mente sobre o que a matemática garante.

**E a divergência não fica confinada a uma zona cinzenta perto dos limiares.** Medido no
mês 6 da simulação de Produção sobre a partição de teste real (23.584 linhas): em
``DebtRatio`` os dois valores são próximos (0,5046 próprio contra 0,4226 do Evidently) e em
``NumberOfTime30-59DaysPastDueNotWorse`` são idênticos (0,1442), mas em ``MonthlyIncome``
saem 0,2634 contra 0,0120 — CRITICO de um lado, ESTAVEL do outro, três bandas de distância.
O mecanismo foi reproduzido, não suposto: recalculando o PSI fora da biblioteca com a regra
de ``evidently.legacy.calculations.stattests.utils.get_binned_data``, os três valores saem
iguais aos que o relatório publica, até o último dígito. Essa regra usa, para coluna
numérica com mais de 20 valores distintos, bins de LARGURA IGUAL pela fórmula de Sturges
sobre o intervalo combinado de referência e atual — e ``MonthlyIncome`` tem mediana 5.416 e
máximo 699.530, uma cauda que joga 99,80% da referência e 99,65% do lote dentro do primeiro
bin de 57.608 de largura. A inflação de 40% da renda move 0,15 ponto percentual de massa
entre bins, e o PSI sai perto de zero. É a mesma armadilha que ``drift/statistics.py``
documenta ao escolher binning por quantil, observada aqui de fora. Vale notar que a
docstring da própria ``get_binned_data`` descreve a função como "split variable into n
buckets based on reference quantiles" — descrição que vale para o caminho de baixa
cardinalidade, não para o numérico de alta cardinalidade que ``MonthlyIncome`` percorre.

Isso não é defeito do Evidently, e não é motivo para removê-lo: ele produz o relatório
visual, e os dois concordam em
veredito nas outras duas variáveis deslocadas. É, sim, a razão de o gate consumir o número
próprio: em variável de cauda longa, o número que um humano pode conferir corte a corte e o
número da biblioteca discordam em banda, não em casa decimal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

from credito.config import get_settings
from credito.drift.base import DriftDeFeature, DriftDetector, DriftReport, classificar
from credito.schema import FEATURES

_METODO_PSI = "psi"
_METODO_KS = "ks"
_TIPO_VALUE_DRIFT = "evidently:metric_v2:ValueDrift"


def _extrair_metricas(resultado: Mapping, features: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Isola do ``dict()`` do Evidently só as métricas ``ValueDrift`` de PSI e KS, uma por
    feature — a parte da resposta que ``EvidentlyDetector.detectar`` de fato consome.

    Filtra por ``config["type"]`` em vez de tentar interpretar toda entrada de
    ``metrics[]`` como ``ValueDrift``: ``DriftedColumnsCount`` (que o preset também
    devolve) carrega ``value`` como um dict (``{"count": ..., "share": ...}``), não um
    float, e um parser que assumisse float em toda métrica quebraria nela sem necessidade
    — ela não é o que este módulo precisa ler.

    Duas falhas são decididas aqui, não acidentais: uma feature pedida sem nenhuma métrica
    devolvida (typo de coluna, ou uma versão futura do Evidently que renomeie o tipo) e uma
    feature com só metade do par PSI/KS (o Evidently mudou o parâmetro de um dos dois e o
    outro sumiu) — ambas erguem ``ValueError`` nomeando a feature, em vez de deixar
    ``DriftDeFeature`` receber um argumento ausente mais adiante e falhar com um erro que
    não aponta a causa. Um valor que não seja ``int``/``float`` onde uma métrica de
    ``ValueDrift`` era esperada (o formato do ``value`` mudou) também é reportado assim,
    pelo mesmo motivo.
    """
    coletado: dict[str, dict[str, float]] = {feature: {} for feature in features}
    for metrica in resultado.get("metrics", []):
        config = metrica.get("config", {})
        if config.get("type") != _TIPO_VALUE_DRIFT:
            continue
        coluna = config.get("column")
        metodo = config.get("method")
        if coluna not in coletado or metodo not in (_METODO_PSI, _METODO_KS):
            continue
        valor = metrica.get("value")
        if not isinstance(valor, int | float):
            raise ValueError(
                f"Evidently devolveu um valor não numérico para a feature {coluna!r} "
                f"({metodo}): {valor!r} — o formato de ValueDrift mudou?"
            )
        coletado[coluna][metodo] = float(valor)

    faltando = [
        feature
        for feature, valores in coletado.items()
        if _METODO_PSI not in valores or _METODO_KS not in valores
    ]
    if faltando:
        raise ValueError(
            f"Evidently não devolveu psi e ks completos para: {faltando!r} — "
            "verificar se as métricas ValueDrift(method=psi) e ValueDrift(method=ks) "
            "foram de fato incluídas no Report"
        )
    return coletado


@dataclass(frozen=True)
class EvidentlyDetector:
    """``DriftDetector`` que delega o cálculo ao Evidently — ver o docstring do módulo
    para a relação com ``drift/statistics.py``.

    ``diretorio_relatorios`` é injetado, não lido de ``get_settings()`` dentro de
    ``detectar`` — a mesma razão de ``baixar_dataset`` receber ``transporte`` por
    parâmetro: um teste que precisasse ler variável de ambiente ou criar um diretório real
    para verificar o HTML gerado seria um teste mais frágil e mais lento do que um que
    recebe ``tmp_path`` diretamente.
    """

    diretorio_relatorios: Path

    def detectar(self, referencia: pd.DataFrame, atual: pd.DataFrame, *, lote: str) -> DriftReport:
        """Roda o Evidently sobre ``FEATURES`` (nunca o alvo — drift é uma pergunta sobre
        o que o modelo recebe como entrada, não sobre o rótulo), grava o HTML nomeado pelo
        ``lote`` em ``diretorio_relatorios`` e devolve o ``DriftReport`` equivalente.

        As colunas são restritas a ``FEATURES`` explicitamente (``referencia[colunas]``) em
        vez de passar o DataFrame inteiro: um chamador que incluísse o alvo ou uma coluna
        de identificação no frame não deveria fazer essa coluna aparecer como "feature em
        drift" no relatório. Isso não é só higiene de relatório: uma coluna presente no
        DataFrame mas ausente de ``numerical_columns`` é autodetectada pelo Evidently do
        mesmo jeito, e medido contra o dataset real (117.917 linhas) remover esta restrição
        faz ``Report.run()`` **não terminar em 120 s** — a coluna de alvo, quase binária,
        aparentemente força um caminho de inferência muito mais caro. Um refactor futuro
        que resolvesse o vazamento de outra forma (descartando o alvo mais cedo no
        pipeline, por exemplo) e por isso removesse esta linha não estaria só reabrindo um
        risco de relatório: estaria enviando um job de produção que trava.
        """
        colunas = list(FEATURES)
        definicao = DataDefinition(numerical_columns=colunas)
        dataset_referencia = Dataset.from_pandas(referencia[colunas], data_definition=definicao)
        dataset_atual = Dataset.from_pandas(atual[colunas], data_definition=definicao)

        # DataDriftPreset(method="psi") devolve DriftedColumnsCount + um ValueDrift(psi)
        # por coluna; ele não tem um parâmetro para pedir os dois métodos ao mesmo tempo
        # (medido: rodar com method="ks" troca o psi pelo ks em vez de somar), então o KS
        # por feature é adicionado à parte, um ValueDrift(method="ks") explícito por
        # coluna — daí "ks" nunca fica implícito nem cai no default de Wasserstein.
        metricas = [DataDriftPreset(method=_METODO_PSI)]
        metricas += [ValueDrift(column=feature, method=_METODO_KS) for feature in FEATURES]

        relatorio = Report(metrics=metricas)
        execucao = relatorio.run(current_data=dataset_atual, reference_data=dataset_referencia)

        self.diretorio_relatorios.mkdir(parents=True, exist_ok=True)
        caminho_html = self.diretorio_relatorios / f"drift_{lote}.html"
        execucao.save_html(str(caminho_html))

        coletado = _extrair_metricas(execucao.dict(), FEATURES)
        features_do_lote = tuple(
            DriftDeFeature(
                feature=feature,
                psi_divergencia=valores[_METODO_PSI],
                ks_p_valor=valores[_METODO_KS],
                severidade=classificar(valores[_METODO_PSI]),
            )
            for feature, valores in coletado.items()
        )
        return DriftReport(lote=lote, features=features_do_lote)


def construir_detector() -> DriftDetector:
    """Fábrica que resolve ``diretorio_relatorios`` a partir de ``get_settings()`` — o
    ponto único onde este módulo lê configuração, para que ``EvidentlyDetector`` em si
    permaneça testável sem depender de variável de ambiente nenhuma."""
    return EvidentlyDetector(diretorio_relatorios=get_settings().reports_dir)
