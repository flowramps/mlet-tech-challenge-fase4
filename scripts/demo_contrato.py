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
from credito.data.prepare import FEATURES, limpar


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()

    referencia, _ = limpar(ler_arff(settings.dataset_path))
    candidatos = referencia[list(FEATURES)]
    validador = construir_validador()

    # `limpar()` não impõe o teto de DebtRatio — essa é uma rede de segurança só do
    # contrato (ver rules.py) — nem deduplica apenas por FEATURES (o dedup roda sobre
    # alvo+features juntos). Por isso a referência crua ainda carrega uma fração residual
    # que reprovaria a ingestão: 2.106 das 120.024 linhas por razao_divida_plausivel e 1
    # por sem_duplicatas quando só as FEATURES são olhadas. A amostra "limpa" da
    # demonstração descarta essas linhas residuais antes de fatiar, para mostrar um lote
    # que de fato passa pelo contrato — não um lote que passou por outro filtro e calhou
    # de não bater com este.
    residuais = {
        indice
        for violacao in validador.validar(candidatos).violacoes
        for indice in violacao.indices
    }
    amostra = candidatos.drop(index=sorted(residuais)).head(500).reset_index(drop=True)

    print("\n=== Lote da Referência (limpo) ===")
    limpo = validador.validar(amostra)
    print(f"linhas: {limpo.total} | válido: {limpo.valido}")
    limpo.erguer()
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
