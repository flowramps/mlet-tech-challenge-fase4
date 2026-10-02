"""Mede o risco de reidentificação da Referência real e o efeito da generalização.

Roda com `make auditar-privacidade`. Carrega a Referência limpa (`data/raw/...`), mede
k-anonimato em combinações crescentes de quase-identificadores — idade, renda e
dependentes são exatamente os atributos que um empregador, um parente ou um vizinho
conhecem sobre alguém — e repete a medição depois de `generalizar`.

É a origem dos números da seção de proteção de dados de `docs/governanca.md`. Existe
como script separado da suíte pela mesma razão de `scripts/validar_proxies.py`: nenhum
teste toca o arquivo real do dataset.
"""

from __future__ import annotations

import logging

from credito.config import get_settings
from credito.data.arff import ler_arff
from credito.data.prepare import limpar
from credito.governanca.privacidade import K_MINIMO, generalizar, risco_de_reidentificacao

_COMBINACOES = (
    ["age"],
    ["age", "NumberOfDependents"],
    ["age", "MonthlyIncome"],
    ["age", "MonthlyIncome", "NumberOfDependents"],
)


def _linha(rotulo: str, risco: dict[str, int]) -> str:
    n = risco["n"]
    return (
        f"{rotulo:42s} {risco['unicas']:7d} {risco['unicas'] / n:8.2%} "
        f"{risco['abaixo_de_k']:8d} {risco['abaixo_de_k'] / n:8.2%} {risco['k_minimo']:6d}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    referencia, _ = limpar(ler_arff(settings.dataset_path))

    print(f"Referência: {len(referencia)} linhas | k-anonimato com k = {K_MINIMO}")
    cabecalho = f"{'quase-identificadores':42s} {'únicas':>7s} {'%':>8s}"
    print(f"{cabecalho} {'< k':>8s} {'%':>8s} {'k mín':>6s}")

    print("valores exatos (como o modelo os recebe):")
    for colunas in _COMBINACOES:
        print(_linha(" + ".join(colunas), risco_de_reidentificacao(referencia, colunas)))

    generalizado = generalizar(referencia)
    print("generalizado (faixa etária, quintil de renda, dependentes até 3):")
    for colunas in _COMBINACOES:
        print(_linha(" + ".join(colunas), risco_de_reidentificacao(generalizado, colunas)))


if __name__ == "__main__":
    main()
