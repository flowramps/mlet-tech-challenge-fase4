"""Calibração do alarme ao vivo: a taxa de aprovação numa janela das últimas N decisões.

A validação de proxies (`credito.monitoring.validacao_de_proxy`) mediu que a taxa de
aprovação acompanha a degradação real — mas mediu sobre lotes mensais de 23.584 decisões.
A API expõe a mesma taxa numa janela de `credito.api.metrics.JANELA_TAXA_DE_APROVACAO`
decisões, muito menor e, portanto, mais ruidosa. Um limiar herdado da escala mensal não
diz nada sobre essa janela; este módulo mede o que diz:

- **A nula** (`nula_da_taxa_de_aprovacao`): sob tráfego estável — decisões sorteadas da
  mesma população, sem drift —, quanto a taxa de uma janela oscila só por acaso.
- **O limiar** (`limiar_inferior`): a cauda de baixo da nula, na taxa de falso alarme
  escolhida.
- **O poder** (`poder_do_alarme`): em cada mês simulado de drift, que fração das janelas
  cai abaixo do limiar. É a resposta a "quanto o alarme antecipa", em vez de suposição.

Toda taxa é calculada por `credito.monitoring.proxies.taxa_de_aprovacao`, a mesma função
que a API usa — nunca uma segunda implementação do corte.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from credito.monitoring.proxies import LIMIAR_PADRAO, taxa_de_aprovacao


def _taxas_em_janelas(
    probabilidades: np.ndarray, *, janela: int, n_amostras: int, seed: int, limiar: float
) -> np.ndarray:
    """Sorteia `n_amostras` janelas de `janela` decisões e devolve a taxa de cada uma.

    Com reposição: em produção as decisões chegam uma a uma da população, e a janela da API
    é uma sequência dessas chegadas, não um recorte sem repetição de um lote fixo.
    """
    gerador = np.random.default_rng(seed)
    populacao = np.asarray(probabilidades, dtype=float)
    indices = gerador.integers(0, len(populacao), size=(n_amostras, janela))
    return np.array([taxa_de_aprovacao(populacao[linha], limiar=limiar) for linha in indices])


def nula_da_taxa_de_aprovacao(
    probabilidades: np.ndarray,
    *,
    janela: int,
    n_amostras: int,
    seed: int,
    limiar: float = LIMIAR_PADRAO,
) -> np.ndarray:
    """Taxa de aprovação de `n_amostras` janelas sorteadas de uma população sem drift."""
    return _taxas_em_janelas(
        probabilidades, janela=janela, n_amostras=n_amostras, seed=seed, limiar=limiar
    )


def limiar_inferior(nula: np.ndarray, *, taxa_de_falso_alarme: float) -> float:
    """O ponto que a nula só cruza para baixo na `taxa_de_falso_alarme` declarada."""
    if not 0 < taxa_de_falso_alarme < 1:
        raise ValueError(f"taxa de falso alarme fora de (0, 1): {taxa_de_falso_alarme}")
    return float(np.percentile(np.asarray(nula, dtype=float), taxa_de_falso_alarme * 100))


def poder_do_alarme(
    probabilidades_por_lote: Mapping[str, np.ndarray],
    *,
    limiar_de_alarme: float,
    janela: int,
    n_amostras: int,
    seed: int,
    limiar: float = LIMIAR_PADRAO,
) -> dict[str, float]:
    """Fração das janelas de cada lote estritamente abaixo de `limiar_de_alarme`."""
    poder: dict[str, float] = {}
    for nome, probabilidades in probabilidades_por_lote.items():
        taxas = _taxas_em_janelas(
            probabilidades, janela=janela, n_amostras=n_amostras, seed=seed, limiar=limiar
        )
        poder[nome] = float(np.mean(taxas < limiar_de_alarme))
    return poder
