"""Demonstração do contrato de dados bloqueando a ingestão de um lote defeituoso.

Roda com `make demo-contrato`. Imprime os dois desfechos lado a lado: o lote limpo é
aceito, o lote adulterado é recusado com o relatório de todas as violações.
"""

from __future__ import annotations

import logging
import sys

from credito.config import get_settings
from credito.contracts.base import ContratoViolado
from credito.contracts.pandera_backend import construir_validador
from credito.data.adulterate import adulterar
from credito.data.arff import ler_arff
from credito.data.prepare import limpar
from credito.schema import FEATURES


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()

    referencia, _ = limpar(ler_arff(settings.dataset_path))
    amostra = referencia[list(FEATURES)].head(500)
    validador = construir_validador()

    print("\n=== Lote da Referência (limpo) ===")
    limpo = validador.validar(amostra)
    print(f"linhas: {limpo.total} | válido: {limpo.valido}")
    try:
        limpo.erguer()
    except ContratoViolado as erro:
        # Este é o desfecho que a demonstração não espera: a Referência reprovando no
        # próprio contrato significa que limpeza e contrato divergiram. Vale mais do que
        # um traceback — é o relatório de violações que diz qual regra abriu a divergência.
        print(f"\nERRO: a Referência não passa no próprio contrato: {erro}")
        return 1
    print("ingestão liberada")

    print("\n=== Lote adulterado ===")
    sujo = adulterar(amostra, seed=settings.random_seed)
    resultado = validador.validar(sujo)
    print(f"linhas: {resultado.total} | válido: {resultado.valido}")
    for violacao in resultado.violacoes:
        print(f"  - {violacao}")

    try:
        resultado.erguer()
    except ContratoViolado as erro:
        print(f"\nINGESTÃO BLOQUEADA: {erro}")
        return 0

    print("\nERRO: o lote adulterado passou — o contrato não está fazendo seu trabalho")
    return 1


if __name__ == "__main__":
    sys.exit(main())
