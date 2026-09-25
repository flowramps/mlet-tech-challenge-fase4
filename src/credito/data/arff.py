"""Leitor de ARFF — o formato em que o OpenML publica o dataset.

Não usamos ``scipy.io.arff.loadarff``: ele devolve um array estruturado do numpy com
atributos nominais em ``bytes``, o que exigiria uma camada de conversão maior que este
leitor inteiro. Depois da linha ``@DATA`` o arquivo é um CSV comum, então a leitura
aproveita o parser do pandas e só precisa descobrir os nomes das colunas no cabeçalho.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_ATRIBUTO = re.compile(r"^@attribute\s+(\S+)\s+", re.IGNORECASE)


def ler_arff(caminho: Path, *, marcas_de_ausente: tuple[str, ...] = ("?",)) -> pd.DataFrame:
    """Lê um arquivo ARFF e devolve os dados como ``DataFrame``."""
    colunas: list[str] = []
    linhas_de_cabecalho = 0

    with Path(caminho).open("r", encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            if linha.lower().startswith("@data"):
                linhas_de_cabecalho = numero
                break
            casamento = _ATRIBUTO.match(linha.strip())
            if casamento:
                colunas.append(casamento.group(1))

    if not linhas_de_cabecalho:
        raise ValueError(f"{caminho} não contém a seção @DATA")

    return pd.read_csv(
        caminho,
        skiprows=linhas_de_cabecalho,
        header=None,
        names=colunas,
        na_values=list(marcas_de_ausente),
    )
