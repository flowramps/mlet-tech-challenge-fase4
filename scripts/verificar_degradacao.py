"""Guarda de regressão para a propriedade central da Etapa 2: o campeão publicado precisa
degradar de forma monotônica contra os seis lotes simulados — não melhorar, não oscilar.

Roda com `make verificar-degradacao`. Carrega o campeão real (`models/model.joblib`) e a
Referência real (`data/raw/GiveMeSomeCredit.arff`), separa a mesma partição de teste que o
treino usou (`separar()`, os mesmos `test_size`/`validation_size`/`random_seed` de
`config.py`), simula os seis meses sobre ela e mede AUC-PR e recall da classe positiva
lote a lote com `degradacao_por_lote`. Sai com código de erro se a queda deixar de ser
monotônica em qualquer um dos dois.

Existe porque a suíte automatizada não pode tocar nem o dataset nem o modelo publicado
(critério do projeto: nenhum teste toca rede ou arquivo real), mas a monotonicidade foi
calibrada contra este campeão específico (`_TAXA_MAXIMA_DE_INVERSAO`, em
`credito.data.simulate`) — um retreino do campeão, uma mudança no limiar da região de
risco emergente ou uma atualização do dataset podem quebrá-la em silêncio, sem que nenhum
teste unitário perceba. Este script é o que torna a propriedade verificável sob demanda, e
o que uma futura esteira de CI chamaria antes de publicar o README com a progressão como
resultado.

A saída impressa é a mesma tabela que o relatório desta correção reproduz — rodar este
script é também como regenerar os números do README, não só validar a monotonicidade.
"""

from __future__ import annotations

import logging
import sys
from itertools import pairwise

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

    print()
    if violacoes:
        print("MONOTONICIDADE QUEBRADA:")
        for violacao in violacoes:
            print(f"  - {violacao}")
        return 1

    print("monotonicidade preservada nos seis meses (AUC-PR e recall+ só caem)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
