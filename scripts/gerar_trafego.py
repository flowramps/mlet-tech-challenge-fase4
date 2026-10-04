"""Gera tráfego real contra a API de scoring (`credito.api`) para que os painéis do Grafana
tenham o que mostrar — sem tráfego, o painel de taxa de erro e os de latência saem vazios.

Roda com `make traffic`, contra `docker compose up -d` já de pé. Alterna requisições
válidas (payload aleatório dentro do domínio plausível de cada feature) e, a cada décima,
uma inválida (falta `MonthlyIncome`) — é o que dá ao painel de tráfego e erro um `422` real
para mostrar, sem nunca produzir um `5xx` de propósito (a API responde `422` a payload
malformado; um `5xx` seria bug, não entrada ruim, ver `docs/monitoring_plan.md`).

Com `--lote mes_06` (ou qualquer `mes_00` … `mes_06`), em vez de payloads aleatórios envia
linhas reais daquele mês da simulação de Produção (`credito.data.simulate`): é o tráfego que
carrega o drift de verdade, e o que faz a taxa de aprovação ao vivo cair até o alerta
calibrado disparar (`docker/prometheus/alertas.yml`). Exige o dataset (`make data`). Sem
inválidas nesse modo: o ponto é medir a decisão, não o erro.

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


def _enviar(cliente: httpx.Client, url: str, payload: dict) -> int | str:
    """Status HTTP da resposta, ou `"timeout"`: um timeout é um resultado a contar, não um
    motivo para abandonar a rodada. Ele aparece de verdade — a gravação do registro de
    decisão com `fsync` trava por segundos quando o host está escrevendo muito em disco
    (medido, ver `docs/monitoring_plan.md`) — e antes derrubava o script no meio."""
    try:
        return cliente.post(url, json=payload, timeout=5.0).status_code
    except httpx.TimeoutException:
        return "timeout"


def gerar_trafego(
    *, base_url: str, n: int, seed: int, cliente: httpx.Client
) -> dict[int | str, int]:
    """Envia `n` requisições a `{base_url}/score` e devolve a contagem por status HTTP."""
    gerador = random.Random(seed)
    contagem: dict[int | str, int] = {}

    for indice in range(n):
        invalida = (indice % _CADENCIA_INVALIDA) == (_CADENCIA_INVALIDA - 1)
        payload = _payload_invalido() if invalida else _payload_valido(gerador)
        resultado = _enviar(cliente, f"{base_url}/score", payload)
        contagem[resultado] = contagem.get(resultado, 0) + 1

    return contagem


def _payloads_do_lote(nome: str, n: int, seed: int) -> list[dict[str, float]]:
    """`n` linhas sorteadas de um mês da simulação, no formato do corpo de `/score`.

    Importa o pipeline só aqui: o modo aleatório não precisa do dataset nem das dependências
    do grupo `pipeline`.
    """
    from credito.config import get_settings
    from credito.data.arff import ler_arff
    from credito.data.prepare import limpar, separar
    from credito.data.simulate import MESES, simular_producao
    from credito.schema import FEATURES

    settings = get_settings()
    referencia, _ = limpar(ler_arff(settings.dataset_path))
    teste = separar(
        referencia,
        test_size=settings.test_size,
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )["teste"]
    lote = simular_producao(teste, meses=MESES, seed=settings.random_seed)[nome]
    amostra = lote[list(FEATURES)].sample(n=n, replace=n > len(lote), random_state=seed)
    return [{k: float(v) for k, v in linha.items()} for linha in amostra.to_dict("records")]


def enviar_lote(
    *, base_url: str, payloads: list[dict[str, float]], cliente: httpx.Client
) -> dict[int | str, int]:
    """Envia cada payload a `{base_url}/score` e devolve a contagem por status HTTP."""
    contagem: dict[int | str, int] = {}
    for payload in payloads:
        resultado = _enviar(cliente, f"{base_url}/score", payload)
        contagem[resultado] = contagem.get(resultado, 0) + 1
    return contagem


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx loga cada requisição em INFO — milhares de linhas de eco que escondem a única que
    # importa (a contagem final por status).
    logging.getLogger("httpx").setLevel(logging.WARNING)

    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--base-url", default="http://localhost:8000")
    # 2.500 por padrão: o suficiente para encher a janela da taxa de aprovação
    # (`credito.api.metrics.JANELA_TAXA_DE_APROVACAO`, 2.000) e a regra de alerta avaliar.
    analisador.add_argument("--n", type=int, default=2_500)
    analisador.add_argument("--seed", type=int, default=7)
    analisador.add_argument(
        "--lote", default=None, help="mes_00 … mes_06: envia linhas reais daquele mês simulado"
    )
    argumentos = analisador.parse_args()

    with httpx.Client() as cliente:
        resposta_saude = cliente.get(f"{argumentos.base_url}/health", timeout=5.0)
        resposta_saude.raise_for_status()
        logger.info("API saudável: %s", resposta_saude.json())

        if argumentos.lote:
            payloads = _payloads_do_lote(argumentos.lote, argumentos.n, argumentos.seed)
            contagem = enviar_lote(base_url=argumentos.base_url, payloads=payloads, cliente=cliente)
        else:
            contagem = gerar_trafego(
                base_url=argumentos.base_url, n=argumentos.n, seed=argumentos.seed, cliente=cliente
            )

    logger.info("%d requisições enviadas, contagem por status: %s", argumentos.n, contagem)


if __name__ == "__main__":
    main()
