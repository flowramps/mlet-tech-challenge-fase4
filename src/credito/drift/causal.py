"""Atribuição causal por intervenção: qual variável deslocada é responsável pela
degradação do campeão — respondida por `do(·)`, não por correlação.

Em dado observacional, quando renda, endividamento e atraso se deslocam juntos, nada
distingue qual dos três causou a queda de desempenho — os três se movem na mesma janela de
tempo, e atribuir a qualquer um deles seria coincidência temporal travestida de causa. A
resposta por intervenção é diferente: fixar as demais variáveis exatamente como estavam e
mover só uma, medindo a queda que essa única mudança produz. É o que este módulo faz, um
cenário por vez — e só é possível aqui porque a Produção é simulada e o processo gerador é
conhecido e controlável; a condição e a limitação que ela implica (em produção real não há
intervenção disponível — ver o módulo docstring de `atribuir_degradacao`) são a mesma
moeda.

Cada cenário chama diretamente as transformações públicas de `credito.data.simulate`
(nunca as reimplementa): `_gerar_lote` compõe as quatro na mesma ordem que
`simular_producao` usa, ligando (intensidade > 0) só a transformação da variável que o
cenário isola — as demais entram com intensidade zero, que cada uma já documenta como
no-op ou identidade. É essa composição idêntica, com as mesmas sementes por mês, que faz o
cenário "todos" reproduzir bit a bit o lote que `simular_producao` geraria para o mesmo
mês — condição sem a qual a atribuição mediria um processo gerador diferente do que foi de
fato simulado, e a análise perderia validade (ver o docstring de `VARIAVEIS_COM_DRIFT` em
`simulate.py`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from credito.data.simulate import (
    MESES,
    aplicar_concept_drift,
    aplicar_drift_de_atraso,
    aplicar_drift_de_divida,
    aplicar_drift_de_renda,
)
from credito.model.evaluate import avaliar

# AUC-ROC, não AUC-PR: AUC-ROC mede só a ordenação (é insensível à prevalência, por
# construção — não depende da taxa de positivos do lote), enquanto o piso estrutural da
# AUC-PR é a própria taxa de positivos do lote (ver `simulate.py`, `_TAXA_MAXIMA_DE_INVERSAO`).
# Decompor AUC-PR atribuiria parte do movimento à prevalência crescente do lote, não só ao
# deslocamento de cada variável — o que misturaria dois efeitos distintos numa única conta.
# Alternativa equivalente seria decompor "lift acima do piso" (auc_pr - taxa_de_positivos),
# mas essa métrica derivada não está entre as que `avaliar()` já devolve; usar AUC-ROC
# reaproveita a mesma função testada em todo o resto do projeto, sem introduzir uma conta
# nova.
METRICA = "auc_roc"

# Cada entrada diz, para o cenário, se a transformação correspondente entra com a
# intensidade do mês (`True`) ou com intensidade zero, ou seja, sem efeito (`False`). A
# ordem das chaves é a mesma ordem da tabela do método: nenhum drift, cada variável
# isolada, concept drift isolado, todas juntas.
_CENARIOS: dict[str, tuple[bool, bool, bool, bool]] = {
    "nenhum_drift": (False, False, False, False),
    "renda": (True, False, False, False),
    "divida": (False, True, False, False),
    "atraso": (False, False, True, False),
    "concept_drift": (False, False, False, True),
    "todos": (True, True, True, True),
}

# Os quatro cenários que isolam uma única transformação — a soma das degradações destes é
# o que se compara contra o efeito do cenário "todos" para obter o termo de interação.
_ISOLADOS: tuple[str, ...] = ("renda", "divida", "atraso", "concept_drift")


def _gerar_lote(
    referencia: pd.DataFrame,
    *,
    k: float,
    liga_renda: bool,
    liga_divida: bool,
    liga_atraso: bool,
    liga_concept: bool,
    seed_atraso: int,
    seed_concept: int,
) -> pd.DataFrame:
    """Aplica, na mesma ordem que `simular_producao` usa, só as transformações ligadas
    neste cenário — as demais recebem intensidade zero, que cada uma documenta como sem
    efeito (`aplicar_drift_de_renda`/`aplicar_drift_de_divida` multiplicam por 1;
    `aplicar_drift_de_atraso`/`aplicar_concept_drift` não sorteiam nada). Fixar tudo e
    mover só uma é o próprio desenho da intervenção: nenhuma outra variável muda porque a
    função que a moveria roda com intensidade zero, não porque foi pulada.
    """
    lote = referencia.copy()
    lote = aplicar_drift_de_renda(lote, k if liga_renda else 0.0)
    lote = aplicar_drift_de_divida(lote, k if liga_divida else 0.0)
    lote = aplicar_drift_de_atraso(lote, k if liga_atraso else 0.0, seed=seed_atraso)
    lote = aplicar_concept_drift(lote, k if liga_concept else 0.0, seed=seed_concept)
    return lote


def atribuir_degradacao(
    referencia: pd.DataFrame, modelo: Any, *, mes: int, seed: int
) -> dict[str, Any]:
    """Decompõe a degradação do campeão em `mes` nos quatro canais de drift que
    `credito.data.simulate` expõe, mais o termo de interação entre eles.

    Para cada cenário da tabela do método, gera o lote correspondente (`_gerar_lote`,
    delegando inteiramente às transformações de `simulate.py`) e mede `METRICA` com
    `avaliar` — a mesma função que o resto do projeto usa para medir o campeão, nunca
    recalculada aqui. A degradação de cada cenário é a queda de `METRICA` contra o
    cenário "nenhum drift" (a linha de base do mês, sem nenhuma transformação): todo
    cenário isolado é comparado contra essa MESMA referência, nunca uns contra os outros,
    porque só assim os quatro efeitos isolados e o conjunto ficam na mesma escala.

    A soma dos quatro efeitos isolados não bate, em geral, com o efeito do cenário
    "todos" — a diferença (`termo_de_interacao`) é o que as variáveis produzem agindo
    juntas que nenhuma delas produz sozinha. Reportá-la é parte da análise, não um resto
    a esconder: atribuir 100% da queda à soma das partes isoladas seria mais simples de
    escrever e menos honesto do que é o número.

    `mes` fixa a intensidade (`k = mes / MESES`, a mesma fórmula de `simular_producao`) e
    as sementes por transformação estocástica (`seed + 2*mes` para o atraso, `seed +
    2*mes + 1` para o concept drift) — os mesmos valores que `simular_producao` usaria
    para esse mês, e por isso o cenário "todos" reproduz, linha a linha, o lote que a
    simulação de produção geraria para `mes` com a mesma `seed`.
    """
    k = mes / MESES
    seed_atraso = seed + 2 * mes
    seed_concept = seed + 2 * mes + 1

    metricas: dict[str, float] = {}
    for nome, (liga_renda, liga_divida, liga_atraso, liga_concept) in _CENARIOS.items():
        lote = _gerar_lote(
            referencia,
            k=k,
            liga_renda=liga_renda,
            liga_divida=liga_divida,
            liga_atraso=liga_atraso,
            liga_concept=liga_concept,
            seed_atraso=seed_atraso,
            seed_concept=seed_concept,
        )
        metricas[nome] = avaliar(modelo, lote)[METRICA]

    linha_de_base = metricas["nenhum_drift"]
    degradacao = {nome: linha_de_base - valor for nome, valor in metricas.items()}

    soma_dos_efeitos_isolados = sum(degradacao[nome] for nome in _ISOLADOS)
    efeito_conjunto = degradacao["todos"]
    termo_de_interacao = efeito_conjunto - soma_dos_efeitos_isolados

    return {
        "metrica": METRICA,
        "mes": mes,
        "seed": seed,
        "metricas_por_cenario": metricas,
        "degradacao_por_cenario": degradacao,
        "soma_dos_efeitos_isolados": soma_dos_efeitos_isolados,
        "efeito_conjunto": efeito_conjunto,
        "termo_de_interacao": termo_de_interacao,
    }
