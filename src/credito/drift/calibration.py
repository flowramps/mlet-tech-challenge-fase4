"""Rigor estatístico que o PSI/KS por si só não dá: uma nula medida, uma correção de
múltiplos testes e a consequência sobre o modelo — as três lacunas que a maioria das
implementações de drift deixa em aberto (ver `barra-especialista.md`).

**1. A nula do PSI.** "PSI >= 0,25 é mudança material" é convenção de indústria — não uma
medição deste dado. PSI tem distribuição amostral: com um número fixo de bins e um lote de
tamanho finito, duas fatias aleatórias da MESMA população (sem nenhum drift verdadeiro)
produzem PSI maior que zero só por acaso de amostragem. `distribuicao_nula_psi` mede essa
distribuição por reamostragem, e `limiar_empirico` lê um percentil dela — o que transforma
"0,25 é convenção" em "neste dado, com estes bins e este tamanho de lote, a nula chega a X
no percentil Y". A calibração contra a Referência real (não incluída nos testes desta
suíte, que não tocam o arquivo do dataset) está no relatório da tarefa.

**2. Múltiplos testes.** Rodar KS em várias features por lote, cada uma a alfa=0,05, infla
a chance de falso positivo por lote muito acima de 0,05 — é o problema clássico de
comparações múltiplas. `benjamini_hochberg` controla a fração esperada de falsas
descobertas entre os p-valores rejeitados, preservando mais poder que Bonferroni (que
controla a família inteira, ao custo de rejeitar menos quando há positivos verdadeiros).

**3. Degradação.** Um deslocamento de distribuição não é, por si, um incidente: o que
importa é se o modelo publicado piora. `degradacao_por_lote` aplica o campeão publicado
(sem retreinar) a cada lote e devolve as métricas de `credito.model.evaluate.avaliar` — o
número que liga "a distribuição mudou" a "e daí".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credito.drift.statistics import psi
from credito.model.evaluate import avaliar


def distribuicao_nula_psi(
    serie: np.ndarray,
    *,
    n_amostras: int,
    tamanho_lote: int,
    bins: int,
    seed: int,
) -> np.ndarray:
    """PSI entre a Referência e uma reamostra sua, repetido `n_amostras` vezes — a
    distribuição de PSI que o acaso produz quando não há drift nenhum.

    Em produção, `psi()` sempre recebe a Referência inteira e fixa como primeiro
    argumento (o lado que define os cortes de bin) e um lote como segundo argumento (o
    lado que varia de mês a mês). Para calibrar exatamente o que o detector real vai ver,
    esta função preserva essa assimetria: `serie` inteira continua sendo o lado que
    define os bins em toda repetição — nunca é ela mesma reamostrada —, e só o lado
    "atual" é sorteado, do tamanho `tamanho_lote`. Reamostrar os dois lados a cada
    repetição calibraria um cenário que o detector nunca encontra (bins recalculados a
    cada mês), não o que ele de fato mede.

    A reamostragem é sem reposição quando `tamanho_lote <= len(serie)` (um recorte
    aleatório de verdade, sem repetir linha) e cai para bootstrap com reposição só
    quando o lote pedido excede o tamanho da própria série — o único jeito de não
    travar nesse caso, sem mudar o comportamento no caso comum.

    Chama `psi()` diretamente, com os mesmos `bins` pedidos aqui — não uma cópia da
    lógica de corte. Isso é o que faz esta nula calibrar exatamente o roteamento por
    colapso medido que `psi()` já decide sozinha (quantil ou por valor, ver
    `drift/statistics.py`): um comportamento calculado de outro jeito calibraria uma
    conta diferente da que o detector realmente usa.
    """
    serie = np.asarray(serie, dtype=float)
    gerador = np.random.default_rng(seed)
    com_reposicao = tamanho_lote > len(serie)

    valores = np.empty(n_amostras, dtype=float)
    for indice in range(n_amostras):
        amostra = gerador.choice(serie, size=tamanho_lote, replace=com_reposicao)
        valores[indice] = psi(serie, amostra, bins=bins)
    return valores


def limiar_empirico(nula: np.ndarray, *, percentil: float) -> float:
    """Lê um percentil da distribuição nula — o limiar que a reamostragem, não a
    convenção, sustenta para este dado, este número de bins e este tamanho de lote."""
    return float(np.percentile(np.asarray(nula, dtype=float), percentil))


def benjamini_hochberg(p_valores: np.ndarray, *, alfa: float = 0.05) -> np.ndarray:
    """Controla a taxa de falsas descobertas (FDR) numa família de testes de hipótese.

    Ordena os p-valores, encontra o maior rank `k` cujo p-valor não excede
    `k/m * alfa` (`m` é o tamanho da família) e rejeita toda hipótese cujo p-valor não
    excede o p-valor de corte encontrado nesse rank. A comparação final é feita pelo
    *valor* de corte (`p <= p_de_corte`), a forma mais direta de aplicar essa definição
    sem precisar re-mapear posições do array ordenado de volta aos índices originais —
    `p <= p_de_corte` já é, por si, invariante a qualquer permutação da entrada.
    Empates nunca ficam divididos entre rejeitado/não-rejeitado, mas isso não é uma
    escolha de implementação que evita um bug: como o limiar cresce estritamente com o
    rank e os p-valores estão ordenados, dois p-valores idênticos nunca podem cair um de
    cada lado da fronteira de elegibilidade (se o de rank menor é elegível, o de rank
    maior — mesmo valor, limiar maior — é elegível também, por definição); uma
    implementação que rejeitasse por posição de rank em vez de por valor chegaria ao
    mesmo resultado. Medido rodando um sweep de mutação sobre esta função (trocar a
    linha abaixo por uma versão posicional e reexecutar a suíte): a suíte inteira
    continua verde, confirmando que as duas formas são equivalentes, não que uma
    corrige a outra.

    Bonferroni (`p <= alfa/m` para cada hipótese) controla a chance de qualquer falso
    positivo na família inteira; BH controla algo mais fraco (a fração esperada de
    falsos entre os rejeitados) e por isso rejeita pelo menos tantas hipóteses quanto
    Bonferroni sempre — nunca menos. A prova é direta a partir de como o corte é
    definido: o rank 1 já usa o limiar `alfa/m`, que é exatamente o limiar de
    Bonferroni; qualquer p-valor que passasse em Bonferroni já teria feito o rank 1
    (o menor da família ordenada) passar em BH, e o p-valor de corte de BH nunca fica
    abaixo do que o rank 1 sustenta.
    """
    p = np.asarray(p_valores, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)

    ordem = np.argsort(p)
    p_ordenado = p[ordem]
    rank = np.arange(1, m + 1)
    limiar_por_rank = rank / m * alfa

    elegiveis = np.flatnonzero(p_ordenado <= limiar_por_rank)
    if elegiveis.size == 0:
        return np.zeros(m, dtype=bool)

    p_de_corte = p_ordenado[elegiveis.max()]
    return p <= p_de_corte


def degradacao_por_lote(modelo: Any, lotes: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """Aplica o campeão publicado (nunca retreinado) a cada lote e devolve as métricas de
    `credito.model.evaluate.avaliar` — AUC-PR, recall da classe positiva e AUC-ROC entre
    elas.

    É este número, não o PSI ou o KS de nenhuma feature isolada, que prova "degradação
    silenciosa": um deslocamento de distribuição que não move a AUC-PR do campeão é
    diagnóstico sem consequência; um que a derruba é o incidente. Delega inteiramente a
    `avaliar` — a mesma conta que o resto do projeto usa para medir o campeão no teste da
    Referência — em vez de recalcular as métricas aqui, para que as duas leituras nunca
    divirjam por implementação duplicada.
    """
    return {nome_do_lote: avaliar(modelo, lote) for nome_do_lote, lote in lotes.items()}
