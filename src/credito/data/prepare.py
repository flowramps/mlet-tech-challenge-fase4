"""Construção do Dataset de Referência a partir do arquivo bruto.

A Referência é o dado que o modelo aprende e o retrato contra o qual todo lote futuro é
comparado. Por isso a limpeza aplica exatamente as mesmas regras que o contrato de dados
vai exigir na ingestão: um dado que seria bloqueado na entrada não pode ter ensinado o
modelo. Cada linha descartada é contabilizada por motivo — descarte silencioso é perda de
auditoria.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from credito.schema import (
    ALVO,
    ALVO_ORIGINAL,
    ATRASO_MAXIMO_PLAUSIVEL,
    COLUNAS_DE_ATRASO,
    COLUNAS_SEM_REGRA_NOMEADA,
    DEBT_RATIO_MAXIMO,
    FEATURES,
    IDADE_MAXIMA,
    IDADE_MINIMA,
)

logger = logging.getLogger(__name__)


def limpar(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica a política de qualidade e devolve os dados limpos e a contagem por motivo."""
    motivos: dict[str, int] = {}
    trabalho = frame.copy()

    if ALVO_ORIGINAL in trabalho.columns:
        trabalho[ALVO] = (trabalho[ALVO_ORIGINAL] == "Yes").astype(int)
        trabalho = trabalho.drop(columns=[ALVO_ORIGINAL])

    # O dedup olha só para FEATURES, não para a linha inteira: o contrato de ingestão
    # nunca vê o alvo, então duas linhas idênticas em FEATURES mas com alvo diferente já
    # passariam pelo contrato como duplicata mesmo que a limpeza as tratasse como
    # observações distintas. E duas aplicações idênticas com desfechos contraditórios não
    # são duas observações — são um conflito de rótulo, que este dedup também resolve
    # (mantendo a primeira ocorrência, na mesma convenção do `Check` do contrato).
    antes = len(trabalho)
    trabalho = trabalho.drop_duplicates(subset=list(FEATURES))
    motivos["duplicata"] = antes - len(trabalho)

    def _descartar(mascara: pd.Series, rotulo: str) -> None:
        nonlocal trabalho
        motivos[rotulo] = int(mascara.sum())
        trabalho = trabalho.loc[~mascara]

    # Cada máscara abaixo é a negação de uma exigência do contrato de ingestão, uma a uma,
    # sem sobrar nenhuma. Qualquer exigência do contrato sem espelho aqui produz uma linha
    # que sobrevive à limpeza e reprova na validação da própria Referência — foi o que
    # quase derrubou o pipeline uma vez, e é por isso que a forma é `~<exigência>` em vez
    # de uma lista de defeitos conhecidos. As comparações usam `ge`/`between`, que
    # devolvem False para nulo: "não satisfaz a exigência" já inclui o valor ausente, sem
    # um `isna()` separado que alguém possa esquecer de acrescentar.
    #
    # As contagens são cumulativas, não independentes: cada `_descartar` opera
    # sobre o que sobrou do descarte anterior, então a soma dos motivos é o total real
    # de linhas perdidas, sem dupla contagem. Isso também explica por que
    # "dependentes_nulo" mede 0 no dataset real: no arquivo bruto, 100% das linhas com
    # `NumberOfDependents` nulo também têm `MonthlyIncome` nulo — a regra de renda nula
    # já as removeu antes desta rodar. Renda nula, dependentes nulo e `DebtRatio`
    # absurdo não são três defeitos, são um único evento de ingestão quebrado com três
    # sintomas. Um "0" aqui é evidência desse fato, não sinal de regra morta a remover.
    _descartar(~trabalho["MonthlyIncome"].ge(0.0), "renda_nula")
    _descartar(~trabalho["NumberOfDependents"].ge(0.0), "dependentes_nulo")
    _descartar(
        ~trabalho["age"].between(IDADE_MINIMA, IDADE_MAXIMA),
        "idade_invalida",
    )

    sentinela = pd.Series(False, index=trabalho.index)
    for coluna in COLUNAS_DE_ATRASO:
        sentinela |= ~trabalho[coluna].between(0, ATRASO_MAXIMO_PLAUSIVEL)
    _descartar(sentinela, "atraso_sentinela")

    # Rede de segurança para quando a renda vem preenchida mas o DebtRatio ainda assim é
    # implausível: entre as linhas que sobram depois dos descartes acima (renda já
    # presente), 1,75% ficam fora de [0, DEBT_RATIO_MAXIMO]. É o mesmo teto que o
    # contrato de ingestão aplica — a mesma constante importada, não reafirmada — para
    # que limpeza e contrato nunca divirjam sobre o que é uma razão de dívida aceitável.
    _descartar(
        ~trabalho["DebtRatio"].between(0.0, DEBT_RATIO_MAXIMO),
        "razao_divida_implausivel",
    )

    # As três colunas restantes do contrato não têm regra de negócio nomeada, só a
    # exigência estrutural de não serem nulas. O contrato relata uma falha nelas como
    # `campo_invalido`, e o motivo do descarte usa o mesmo nome para que os dois lados
    # falem de um defeito só. Mede 0 no arquivo real — nenhuma das três chega nula —, o
    # que é evidência de que o dado está íntegro nessas colunas, não regra sobrando.
    obrigatoria_nula = pd.Series(False, index=trabalho.index)
    for coluna in COLUNAS_SEM_REGRA_NOMEADA:
        obrigatoria_nula |= trabalho[coluna].isna()
    _descartar(obrigatoria_nula, "campo_invalido")

    limpo = trabalho[[ALVO, *FEATURES]].reset_index(drop=True)
    logger.info("referência com %d linhas; descartes: %s", len(limpo), motivos)
    return limpo, motivos


def separar(
    frame: pd.DataFrame,
    *,
    test_size: float,
    validation_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Separa em treino, validação e teste, estratificando pelo alvo.

    Com 6,94% de positivos, um corte aleatório sem estratificação pode variar a proporção
    da classe rara o bastante para mover a métrica mais que o próprio modelo.
    """
    resto, teste = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame[ALVO]
    )
    proporcao = validation_size / (1 - test_size)
    treino, validacao = train_test_split(
        resto, test_size=proporcao, random_state=seed, stratify=resto[ALVO]
    )
    return {
        "treino": treino.reset_index(drop=True),
        "validacao": validacao.reset_index(drop=True),
        "teste": teste.reset_index(drop=True),
    }
