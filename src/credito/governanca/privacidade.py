"""Risco de reidentificação por combinação de atributos — a pergunta que "o dataset não tem
nome nem CPF" não responde.

A LGPD só tira um dado do regime de dado pessoal quando a anonimização não pode ser
revertida "com esforços razoáveis" (art. 12). Sem identificador direto, uma linha ainda
pode ser a única com aquela idade, aquela renda e aquele número de dependentes — e quem
conhece esses três fatos sobre alguém acha a linha inteira, inclusive o histórico de
atraso e o rótulo de inadimplência. `risco_de_reidentificacao` mede isso como
k-anonimato: o tamanho do menor grupo de linhas indistinguíveis entre si.

`generalizar` é a mitigação medida: troca valor exato por faixa. Não é parte do pipeline
de treino — o modelo continua treinando sobre o valor exato. É a transformação a aplicar
antes de qualquer conservação ou compartilhamento da Referência para fim que não seja o
treino (ver `docs/governanca.md`).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from credito.model.evaluate import FAIXAS_ETARIAS

# Convenção da literatura de k-anonimato, não medição deste dado: grupos com menos de 5
# linhas indistinguíveis são tratados como expostos.
K_MINIMO = 5

# Quantos grupos de renda a generalização usa. Quintil, não decil: medido contra a
# Referência real, o decil ainda deixa 25 linhas em grupo menor que K_MINIMO, o quintil 12
# (ver `scripts/auditar_privacidade.py`).
QUANTIS_DE_RENDA = 5

# Dependentes acima disto viram um único grupo "3 ou mais" — a cauda é onde as
# combinações raras se concentram.
TETO_DE_DEPENDENTES = 3


def risco_de_reidentificacao(
    frame: pd.DataFrame, quase_identificadores: Sequence[str], *, k: int = K_MINIMO
) -> dict[str, int]:
    """Quantas linhas ficam sozinhas, ou em grupo menor que `k`, quando o frame é agrupado
    pelos `quase_identificadores`."""
    if not quase_identificadores:
        raise ValueError("informe ao menos um quase-identificador")
    colunas = list(quase_identificadores)
    tamanho = frame.groupby(colunas, dropna=False)[colunas[0]].transform("size")
    return {
        "n": int(len(frame)),
        "unicas": int((tamanho == 1).sum()),
        "abaixo_de_k": int((tamanho < k).sum()),
        "k_minimo": int(tamanho.min()),
    }


def _faixa_etaria(idade: float) -> str:
    for nome, (minimo, maximo) in FAIXAS_ETARIAS.items():
        if minimo <= idade <= maximo:
            return nome
    # O contrato de dados já reprova idade fora do domínio; chegar aqui é defeito a montante,
    # e uma faixa inventada esconderia exatamente a linha mais rara do frame.
    raise ValueError(f"idade {idade} fora de todas as faixas etárias")


def generalizar(frame: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia com idade em faixa, renda em quintil e dependentes limitados."""
    generalizado = frame.copy()
    generalizado["age"] = frame["age"].map(_faixa_etaria)
    generalizado["MonthlyIncome"] = pd.qcut(
        frame["MonthlyIncome"], QUANTIS_DE_RENDA, labels=False, duplicates="drop"
    )
    generalizado["NumberOfDependents"] = frame["NumberOfDependents"].clip(upper=TETO_DE_DEPENDENTES)
    return generalizado
