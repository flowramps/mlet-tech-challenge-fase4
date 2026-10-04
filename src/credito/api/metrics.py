"""Instrumentação Prometheus da API de scoring.

**Rótulo de rota é a rota declarada, nunca o caminho bruto.** Um scanner varrendo URLs
aleatórias criaria uma série temporal nova por URL e esgotaria a memória do processo —
por isso o rótulo lê `request.scope["route"]` (o objeto que o roteador do FastAPI já
casou), nunca `request.url.path`. Uma requisição sem rota casada (404) usa o rótulo fixo
``"sem_rota"``, o mesmo padrão que o middleware equivalente de uma etapa anterior deste
projeto já usava.

**`/metrics` não conta como tráfego da aplicação.** O scrape do Prometheus roda em
intervalo fixo mesmo sem nenhuma requisição real de scoring; contá-lo somaria uma
requisição fantasma a cada scrape.

**Latência de inferência (só a chamada a `predict_proba`) é medida separada da latência
HTTP (a requisição inteira).** É o que permite atribuir uma lentidão ao modelo ou ao
servidor — a montagem do payload, a validação do Pydantic e a serialização da resposta
entram na segunda métrica, nunca na primeira.

**A taxa de aprovação exposta aqui reusa `credito.monitoring.proxies.taxa_de_aprovacao`
sobre uma janela deslizante das últimas requisições — nunca uma segunda implementação do
corte de limiar.** É a métrica que a Etapa 3 mediu como o proxy sem rótulo mais forte
(passo médio 0,0079 contra degradação real) e que o alarme de produção lê.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    ProcessCollector,
    generate_latest,
)

from credito.monitoring.proxies import LIMIAR_PADRAO, taxa_de_aprovacao

# Medido em duas rodadas. `/health` não toca o modelo, então a primeira medição — TestClient
# contra esta app, 50 requisições cronometradas com `time.perf_counter` — continua valendo:
# p50 0,80 ms, p99 1,05 ms. `/score` foi remedido contra o campeão real, com a pilha de pé
# (540 requisições, cliente sequencial, lidas do próprio histograma no Prometheus — ver
# `docs/monitoring_plan.md`): p50 6,46 ms, p95 10,7 ms, p99 22,2 ms. Os buckets vão de
# 0,5 ms (abaixo do p50 de `/health`) a 250 ms (cerca de 11x o p99 de `/score`), para que
# tanto uma resposta local quanto uma degradação real de rede/CPU caiam num bucket
# intermediário em vez de estourar a régua.
HTTP_BUCKETS = (0.0005, 0.00075, 0.001, 0.0015, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)

# A primeira medição desta constante usou um dublê de modelo (sem custo real de
# árvore/regressão) e calibrou buckets até 1 ms — valor que o campeão real (XGBoost, 300
# árvores) estourava em toda requisição assim que a stack subiu de verdade (docker compose): as
# 540 chamadas reais caíam inteiras no bucket +Inf, tornando o Histogram inútil. Remedido
# com `Modelo.predict_proba` real, 540 chamadas, payload variado: p50 3,515 ms, p90
# 7,696 ms, p95 9,665 ms, p99 13,442 ms, máximo 17,367 ms (medição registrada em
# `docs/monitoring_plan.md`). Os buckets abaixo cobrem essa faixa com granularidade fina
# e vão até 150 ms — cerca de 9x acima do máximo medido, a mesma folga proporcional que a
# primeira calibração já usava sobre o próprio máximo dela.
INFERENCE_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.1,
    0.15,
)

# `predict_proba` devolve sempre um valor em [0, 1] — não é medição, é o domínio
# matemático de uma probabilidade; por isso os buckets cobrem o intervalo inteiro em
# passos iguais de 0,05, sem precisar de nenhuma medição prévia.
SCORE_BUCKETS = tuple(round(i * 0.05, 2) for i in range(21))

# Tamanho da janela deslizante que alimenta a taxa de aprovação, e o limiar do alarme sobre
# ela — os dois escolhidos por medição (`make calibrar-alarme`, que também confere que estes
# valores ainda batem com o campeão publicado). Sorteando janelas sem drift do mês 0 da
# simulação e aplicando o limiar a cada mês de drift, com 1% de falso alarme por janela:
#
#   janela   desvio sem drift   limiar   fração das janelas que disparam, mês 1 … mês 6
#      500         1,80 p.p.    0,7560   5,7%  13,9%  25,4%  38,7%  54,5%   68,1%
#    2.000         0,89 p.p.    0,7755  14,9%  48,9%  79,5%  94,1%  99,3%  100,0%
#    5.000         0,56 p.p.    0,7828  35,0%  89,8%  99,5% 100,0% 100,0%  100,0%
#
# A janela era 500, por convenção — e o ruído dela (1,8 p.p.) é maior que a queda de um mês
# inteiro de degradação (0,79 p.p.): mesmo no mês 6, um terço das janelas não disparava.
# 2.000 é a menor janela medida que dispara na maioria das janelas a partir do mês 3. A de
# 5.000 detecta antes, ao custo de 2,5x mais decisões para encher; a escolha certa depende do
# volume de tráfego, e a tabela fica aqui para quem operar decidir pelo próprio volume.
JANELA_TAXA_DE_APROVACAO = 2_000
LIMIAR_DO_ALARME = 0.7755


@dataclass(frozen=True)
class Metrics:
    """Instrumentos da aplicação, agrupados para viajar em `app.state`."""

    requests_total: Counter
    request_duration: Histogram
    inference_duration: Histogram
    score_distribution: Histogram
    taxa_de_aprovacao_atual: Gauge
    amostras_na_janela: Gauge
    model_info: Info
    _janela: deque[float] = field(default_factory=lambda: deque(maxlen=JANELA_TAXA_DE_APROVACAO))

    def registrar_score(self, probabilidade: float) -> None:
        """Alimenta a distribuição do score e recalcula a taxa de aprovação sobre a janela
        deslizante — sempre via `credito.monitoring.proxies.taxa_de_aprovacao`, sobre o
        mesmo `LIMIAR_PADRAO` que o módulo já declara, nunca uma segunda implementação do
        corte de limiar dentro da API.
        """
        self.score_distribution.observe(probabilidade)
        self._janela.append(probabilidade)
        taxa = taxa_de_aprovacao(np.asarray(self._janela, dtype=float), limiar=LIMIAR_PADRAO)
        self.taxa_de_aprovacao_atual.set(taxa)
        self.amostras_na_janela.set(len(self._janela))


def instrument(application: FastAPI, registry: CollectorRegistry) -> Metrics:
    """Instala o middleware de medição HTTP e a rota `/metrics`.

    Um `CollectorRegistry` próprio por aplicação, nunca o global da biblioteca: a suíte de
    testes cria várias apps no mesmo processo, e no registry global a segunda criação
    colidiria com a primeira, vazando contadores de um teste para o outro.
    """
    # Saúde do processo (CPU, memória residente, descritores, hora de início — um reinício
    # aparece como salto nessa última): a estabilidade da infraestrutura, no mesmo /metrics.
    ProcessCollector(registry=registry)

    metrics = Metrics(
        requests_total=Counter(
            "credito_http_requests_total",
            "Total de requisições HTTP, por método, rota e status da resposta.",
            labelnames=("method", "route", "status"),
            registry=registry,
        ),
        request_duration=Histogram(
            "credito_http_request_duration_seconds",
            "Duração da requisição HTTP, do recebimento ao envio da resposta.",
            labelnames=("method", "route"),
            buckets=HTTP_BUCKETS,
            registry=registry,
        ),
        inference_duration=Histogram(
            "credito_inference_duration_seconds",
            "Duração da inferência do modelo campeão, sem HTTP.",
            buckets=INFERENCE_BUCKETS,
            registry=registry,
        ),
        score_distribution=Histogram(
            "credito_score_distribution",
            "Distribuição da probabilidade de inadimplência devolvida pelo modelo.",
            buckets=SCORE_BUCKETS,
            registry=registry,
        ),
        taxa_de_aprovacao_atual=Gauge(
            "credito_taxa_de_aprovacao",
            "Fração aprovada nas últimas requisições de score — o proxy sem rótulo mais "
            "forte medido contra degradação real, calculado por "
            "credito.monitoring.proxies.taxa_de_aprovacao sobre uma janela deslizante.",
            registry=registry,
        ),
        amostras_na_janela=Gauge(
            "credito_taxa_de_aprovacao_amostras",
            "Quantas decisões a janela da taxa de aprovação já tem. A regra de alerta só "
            "avalia com a janela cheia: o limiar foi calibrado para janelas de "
            "JANELA_TAXA_DE_APROVACAO decisões, e uma janela incompleta é ruído.",
            registry=registry,
        ),
        model_info=Info(
            "credito_model",
            "Metadados do modelo campeão servido.",
            registry=registry,
        ),
    )

    @application.middleware("http")
    async def medir_requisicao(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # O scrape do Prometheus não é tráfego de scoring: medi-lo somaria uma requisição
        # a cada intervalo de scrape mesmo com a API sem nenhum cliente real.
        if request.url.path == "/metrics":
            return await call_next(request)

        inicio = time.perf_counter()
        status = 500  # se o handler estourar antes de responder, o cliente recebe 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            rota = request.scope.get("route")
            caminho = rota.path if rota is not None else "sem_rota"
            metrics.requests_total.labels(
                method=request.method, route=caminho, status=str(status)
            ).inc()
            metrics.request_duration.labels(method=request.method, route=caminho).observe(
                time.perf_counter() - inicio
            )

    @application.get("/metrics", tags=["operação"], summary="Métricas no formato Prometheus")
    def exposicao() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return metrics
