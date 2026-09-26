"""Geração de um lote com defeitos injetados, para exercitar o contrato.

Os defeitos não são inventados: cada um reproduz um problema **medido** no arquivo bruto
original. O lote adulterado é, na prática, uma amostra de como o dado chegaria se a
limpeza da ingestão não existisse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFEITOS: tuple[str, ...] = (
    "renda_nao_nula",
    "idade_plausivel",
    "sem_duplicatas",
    "atraso_plausivel",
    "dependentes_nao_nulo",
    "razao_divida_plausivel",
)


def adulterar(frame: pd.DataFrame, *, seed: int, por_defeito: int = 5) -> pd.DataFrame:
    """Injeta ``por_defeito`` ocorrências de cada defeito conhecido.

    Os índices atingidos por cada defeito são disjuntos: uma linha com dois defeitos
    dificultaria ler o relatório e esconderia se alguma regra deixou de disparar. Para
    isso, as primeiras ``por_defeito`` linhas ficam reservadas como âncora da duplicata e
    de fora do sorteio dos outros cinco defeitos — sem essa reserva, um sorteio azarado
    poderia recair sobre a própria âncora e a cópia levaria dois defeitos de uma vez.
    """
    sujo = frame.copy().reset_index(drop=True)
    gerador = np.random.default_rng(seed)

    necessario = por_defeito * (len(DEFEITOS) - 1)
    if len(sujo) < necessario + por_defeito:
        raise ValueError(f"lote pequeno demais: precisa de ao menos {necessario + por_defeito}")

    ancora = np.arange(por_defeito)
    pool = np.arange(por_defeito, len(sujo))
    sorteados = gerador.choice(pool, size=necessario, replace=False)
    fatias = np.split(sorteados, len(DEFEITOS) - 1)
    (renda, idade, atraso, dependentes, razao) = fatias

    # Renda ausente — 19,8% do arquivo bruto chega assim.
    sujo.loc[renda, "MonthlyIncome"] = np.nan

    # Idade impossível — o bruto traz um registro com 0.
    sujo.loc[idade, "age"] = 0

    # Sentinela de ausência disfarçada de contagem — 269 linhas do bruto usam 96/98.
    sujo.loc[atraso, "NumberOfTimes90DaysLate"] = 98

    # Dependentes ausentes — 3.924 linhas do bruto (2,6%) chegam assim.
    sujo.loc[dependentes, "NumberOfDependents"] = np.nan

    # Dívida bruta no lugar da razão — o que acontece quando a renda falta.
    sujo.loc[razao, "DebtRatio"] = 1159.0

    # Duplicatas: 646 linhas do bruto repetem outra nas colunas do contrato. Entram como
    # cópia da âncora reservada, que
    # continua limpa — a linha nova viola só "sem_duplicatas", nunca mais nada.
    copias = sujo.loc[ancora].copy()
    return pd.concat([sujo, copias], ignore_index=True)
