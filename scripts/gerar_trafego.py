"""Gera tráfego real contra a API de scoring (`credito.api`) para que os painéis do Grafana
tenham o que mostrar — sem tráfego, o painel de taxa de erro e os de latência saem vazios.

Roda com `make traffic`, contra `docker compose up -d` já de pé. Alterna requisições
válidas (payload aleatório dentro do domínio plausível de cada feature) e, a cada décima,
uma inválida (falta `MonthlyIncome`) — é o que dá ao painel de tráfego e erro um `422` real
para mostrar, sem nunca produzir um `5xx` de propósito (a API responde `422` a payload
malformado; um `5xx` seria bug, não entrada ruim, ver `docs/monitoring_plan.md`).

Não faz parte da suíte automatizada pelo mesmo motivo que os demais scripts de
`scripts/`: toca rede de verdade (a API precisa estar no ar).
"""

from __future__ import annotations

import argparse
import logging
import random

import httpx

logger = logging.getLogger(__name__)

_PAYLOAD_BASE = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2.0,
}

# A cada quantas requisições uma é deliberadamente inválida — mesma cadência que o
# docstring do módulo e o plano de monitoramento já declaram.
_CADENCIA_INVALIDA = 10


def _payload_valido(gerador: random.Random) -> dict[str, float | int]:
    payload = dict(_PAYLOAD_BASE)
    payload["RevolvingUtilizationOfUnsecuredLines"] = gerador.uniform(0.0, 1.2)
    payload["age"] = gerador.randint(21, 80)
    payload["DebtRatio"] = gerador.uniform(0.0, 2.0)
    payload["MonthlyIncome"] = gerador.uniform(500.0, 20000.0)
    payload["NumberOfOpenCreditLinesAndLoans"] = gerador.randint(0, 20)
    payload["NumberOfDependents"] = float(gerador.randint(0, 5))
    return payload


def _payload_invalido() -> dict[str, float | int]:
    payload = dict(_PAYLOAD_BASE)
    del payload["MonthlyIncome"]
    return payload


def gerar_trafego(*, base_url: str, n: int, seed: int, cliente: httpx.Client) -> dict[int, int]:
    """Envia `n` requisições a `{base_url}/score` e devolve a contagem por status HTTP."""
    gerador = random.Random(seed)
    contagem: dict[int, int] = {}

    for indice in range(n):
        invalida = (indice % _CADENCIA_INVALIDA) == (_CADENCIA_INVALIDA - 1)
        payload = _payload_invalido() if invalida else _payload_valido(gerador)
        resposta = cliente.post(f"{base_url}/score", json=payload, timeout=5.0)
        contagem[resposta.status_code] = contagem.get(resposta.status_code, 0) + 1

    return contagem


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx loga cada requisição em INFO — 600 linhas de eco que escondem a única que
    # importa (a contagem final por status).
    logging.getLogger("httpx").setLevel(logging.WARNING)

    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--base-url", default="http://localhost:8000")
    analisador.add_argument("--n", type=int, default=600)
    analisador.add_argument("--seed", type=int, default=7)
    argumentos = analisador.parse_args()

    with httpx.Client() as cliente:
        resposta_saude = cliente.get(f"{argumentos.base_url}/health", timeout=5.0)
        resposta_saude.raise_for_status()
        logger.info("API saudável: %s", resposta_saude.json())

        contagem = gerar_trafego(
            base_url=argumentos.base_url, n=argumentos.n, seed=argumentos.seed, cliente=cliente
        )

    logger.info("%d requisições enviadas, contagem por status: %s", argumentos.n, contagem)


if __name__ == "__main__":
    main()
