"""Guarda de regressão para a propriedade central da Etapa 2: o campeão publicado precisa
degradar de forma monotônica contra os seis lotes simulados — não melhorar, não oscilar.

Roda com `make verificar-degradacao`. Carrega o campeão real (`models/model.joblib`) e a
Referência real (`data/raw/GiveMeSomeCredit.arff`), separa a mesma partição de teste que o
treino usou (`separar()`, os mesmos `test_size`/`validation_size`/`random_seed` de
`config.py`), simula os seis meses sobre ela e mede AUC-PR e recall da classe positiva
lote a lote com `degradacao_por_lote`. Sai com código de erro se a queda deixar de ser
monotônica em qualquer um dos dois.

**E verifica a outra metade da calibração**: o comentário de `_TAXA_MAXIMA_DE_INVERSAO`,
em `credito.data.simulate`, justifica o valor 0,25 citando um contraexperimento — com 0,5,
a AUC-PR para de cair e VOLTA A SUBIR, porque a taxa de positivos do lote (o piso
estrutural da métrica) sobe mais rápido do que a discriminação cai. Esses números eram
citados sem nenhuma forma de reproduzi-los por comando, e apodreceram em silêncio quando o
gerador mudou. `_contraexperimento` os regenera na mesma execução e falha se a reversão
deixar de acontecer — caso em que o comentário é que está velho, não o código.

Existe porque a suíte automatizada não pode tocar nem o dataset nem o modelo publicado
(critério do projeto: nenhum teste toca rede ou arquivo real), mas as duas propriedades
foram calibradas contra este campeão específico — um retreino do campeão, uma mudança no
limiar da região de risco emergente ou uma atualização do dataset podem quebrá-las em
silêncio, sem que nenhum teste unitário perceba. Este script é o que as torna verificáveis
sob demanda, e o que uma futura esteira de CI chamaria antes de publicar o README com a
progressão como resultado.

A saída impressa é a mesma tabela que o relatório desta correção reproduz — rodar este
script é também como regenerar os números do README e do comentário de
`_TAXA_MAXIMA_DE_INVERSAO`, não só validar a monotonicidade.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import pairwise
from typing import Any

import pandas as pd

import credito.data.simulate as simulate
from credito.config import get_settings
from credito.data.arff import ler_arff
from credito.data.prepare import limpar, separar
from credito.data.simulate import simular_producao
from credito.drift.calibration import degradacao_por_lote
from credito.model.train import carregar_modelo
from credito.schema import ALVO


def _checar_monotonicidade(nome_da_metrica: str, valores: list[float]) -> list[str]:
    """Devolve uma violação por par consecutivo que sobe em vez de cair — vazio se a
    sequência inteira é não crescente."""
    violacoes = []
    for mes, (anterior, atual) in enumerate(pairwise(valores)):
        if atual > anterior:
            violacoes.append(
                f"{nome_da_metrica}: mes_{mes:02d}={anterior:.4f} -> "
                f"mes_{mes + 1:02d}={atual:.4f} (subiu)"
            )
    return violacoes


def _e_nao_crescente(valores: list[float]) -> bool:
    """A mesma checagem de `_checar_monotonicidade`, lida como predicado — o
    contraexperimento abaixo precisa afirmar tanto "esta série cai" quanto "esta NÃO
    cai", e `if not _checar_monotonicidade(...)` inverteria o sentido duas vezes na
    mesma linha."""
    return not _checar_monotonicidade("", valores)


@contextmanager
def _taxa_de_inversao(valor: float) -> Iterator[None]:
    """Troca temporariamente `_TAXA_MAXIMA_DE_INVERSAO` pelo valor do contraexperimento.

    Mexer num nome privado de outro módulo é deliberado e está contido aqui: o
    contraexperimento É "e se a constante fosse outra", e não há (nem deve haver) um
    parâmetro público para isso — o valor calibrado é uma decisão do módulo, não uma opção
    de chamada. O `finally` restaura sempre, para que a tabela principal impressa acima e
    qualquer código que rode depois nunca vejam o valor do experimento.
    """
    original = simulate._TAXA_MAXIMA_DE_INVERSAO
    simulate._TAXA_MAXIMA_DE_INVERSAO = valor
    try:
        yield
    finally:
        simulate._TAXA_MAXIMA_DE_INVERSAO = original


def _contraexperimento(
    modelo: Any, teste: pd.DataFrame, *, seed: int, taxa: float = 0.5
) -> list[str]:
    """Reproduz o contraexperimento que justifica o valor de `_TAXA_MAXIMA_DE_INVERSAO`.

    Com `taxa` no dobro do calibrado, a AUC-PR tem que PARAR de cair em algum ponto da
    janela e voltar a subir — é isso que mostra que o piso estrutural da métrica (a taxa
    de positivos do lote) passou a dominar a perda de ranking, e é por isso que a constante
    não pode ser maior. A AUC-ROC, insensível à prevalência, continua caindo. Devolve as
    violações encontradas: vazio quando o contraexperimento reproduz o que o comentário de
    `credito.data.simulate` afirma.
    """
    with _taxa_de_inversao(taxa):
        lotes = simular_producao(teste, meses=6, seed=seed)
        resultado = degradacao_por_lote(modelo, lotes)

    auc_pr = [metricas["auc_pr"] for metricas in resultado.values()]
    auc_roc = [metricas["auc_roc"] for metricas in resultado.values()]
    taxa_de_positivos = [metricas["taxa_de_positivos"] for metricas in resultado.values()]

    print()
    # O valor calibrado é lido do módulo, não escrito à mão: "o dobro do calibrado" viraria
    # legenda falsa no instante em que a constante ou `taxa` mudassem.
    print(
        f"contraexperimento com _TAXA_MAXIMA_DE_INVERSAO = {taxa} "
        f"(calibrado: {simulate._TAXA_MAXIMA_DE_INVERSAO}):"
    )
    print(f"{'lote':8s} {'auc_pr':>8s} {'auc_roc':>8s} {'taxa_pos':>9s}")
    for nome, metricas in resultado.items():
        print(
            f"{nome:8s} {metricas['auc_pr']:8.4f} {metricas['auc_roc']:8.4f} "
            f"{metricas['taxa_de_positivos']:9.4f}"
        )
    mes_minimo = min(range(len(auc_pr)), key=lambda indice: auc_pr[indice])
    print(
        f"  auc_pr: mínimo no mês {mes_minimo} ({auc_pr[mes_minimo]:.4f}), "
        f"mês 6 {auc_pr[-1]:.4f} | auc_roc: {auc_roc[0]:.4f} -> {auc_roc[-1]:.4f} | "
        f"taxa de positivos: {taxa_de_positivos[0]:.2%} -> {taxa_de_positivos[-1]:.2%}"
    )

    violacoes = []
    if _e_nao_crescente(auc_pr):
        violacoes.append(
            f"contraexperimento com taxa={taxa}: a AUC-PR caiu monotonicamente e NÃO "
            "reverteu — o comentário de _TAXA_MAXIMA_DE_INVERSAO em credito.data.simulate "
            "afirma que reverte, e os números que ele cita estão velhos"
        )
    if not _e_nao_crescente(auc_roc):
        violacoes.append(
            f"contraexperimento com taxa={taxa}: a AUC-ROC deixou de cair monotonicamente "
            "— o mesmo comentário afirma que ela cai mês a mês sem nunca inverter"
        )
    if min(auc_roc) <= 0.5:
        violacoes.append(
            f"contraexperimento com taxa={taxa}: a AUC-ROC chegou a {min(auc_roc):.4f}, "
            "não acima de 0,5 — o ranking teria se tornado anticorrelacionado, o oposto "
            "do que o mesmo comentário afirma"
        )
    return violacoes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()

    referencia, _ = limpar(ler_arff(settings.dataset_path))
    particoes = separar(
        referencia,
        test_size=settings.test_size,
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )
    teste = particoes["teste"]

    modelo, metadados = carregar_modelo(settings.model_path)
    candidato = metadados.get("candidato", metadados) if isinstance(metadados, dict) else metadados

    lotes = simular_producao(teste, meses=6, seed=settings.random_seed)
    concordancia_mes_00 = float((lotes["mes_00"][ALVO].to_numpy() == teste[ALVO].to_numpy()).mean())

    resultado = degradacao_por_lote(modelo, lotes)

    print(f"campeão: {candidato} | partição de teste: {len(teste)} linhas")
    print(f"concordância do rótulo no mês 0 com a partição de teste: {concordancia_mes_00:.4f}")
    print()
    print(f"{'lote':8s} {'auc_pr':>8s} {'auc_roc':>8s} {'recall+':>8s} {'taxa_pos':>9s}")
    for nome, metricas in resultado.items():
        print(
            f"{nome:8s} {metricas['auc_pr']:8.4f} {metricas['auc_roc']:8.4f} "
            f"{metricas['recall_positivo']:8.4f} {metricas['taxa_de_positivos']:9.4f}"
        )

    auc_pr = [metricas["auc_pr"] for metricas in resultado.values()]
    recall_positivo = [metricas["recall_positivo"] for metricas in resultado.values()]

    violacoes = _checar_monotonicidade("auc_pr", auc_pr) + _checar_monotonicidade(
        "recall_positivo", recall_positivo
    )
    violacoes += _contraexperimento(modelo, teste, seed=settings.random_seed)

    print()
    if violacoes:
        print("CALIBRAÇÃO DE _TAXA_MAXIMA_DE_INVERSAO QUEBRADA:")
        for violacao in violacoes:
            print(f"  - {violacao}")
        return 1

    print(
        "monotonicidade preservada nos seis meses (AUC-PR e recall+ só caem) e "
        "contraexperimento com 0,5 reverte, como o comentário da constante afirma"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
