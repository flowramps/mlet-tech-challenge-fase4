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
    generate_latest,
)

from credito.monitoring.proxies import LIMIAR_PADRAO, taxa_de_aprovacao

# Medido localmente: TestClient contra esta mesma app, modelo-dublê (sem custo real de
# árvore/regressão), 200 requisições a /score e 50 a /health, cada uma cronometrada com
# `time.perf_counter` ao redor da chamada do cliente de teste (script de medição
# descartado após o uso, resultado registrado no relatório da tarefa). /score: p50 1,45
# ms, p90 1,96 ms, p99 2,39 ms, máximo 2,54 ms. /health (sem inferência nem validação de
# payload): p50 0,80 ms, p99 1,05 ms, máximo 1,10 ms. Os buckets abaixo vão de 0,5 ms
# (abaixo do p50 mais rápido medido, /health) a 250 ms (duas ordens de grandeza acima do
# p99 mais lento medido, /score), para que tanto uma resposta local quanto uma
# degradação real de rede/CPU ainda caiam num bucket intermediário em vez de estourar a
# régua.
HTTP_BUCKETS = (0.0005, 0.00075, 0.001, 0.0015, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)

# A primeira medição desta constante usou um dublê de modelo (sem custo real de
# árvore/regressão) e calibrou buckets até 1 ms — valor que o campeão real (XGBoost, 300
# árvores) estourava em toda requisição assim que a stack subiu de verdade (Task 5): as
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

# Tamanho da janela deslizante que alimenta a taxa de aprovação: convenção operacional
# (quantas observações recentes resumem "agora"), não medição deste dataset — o mesmo
# status que `credito.config.Settings.psi_atencao`/`psi_critico` já declaram para os
# próprios cortes. Grande o bastante para não oscilar a cada requisição isolada, limitado
# para não crescer sem fim num processo de vida longa.
JANELA_TAXA_DE_APROVACAO = 500


@dataclass(frozen=True)
class Metrics:
    """Instrumentos da aplicação, agrupados para viajar em `app.state`."""

    requests_total: Counter
    request_duration: Histogram
    inference_duration: Histogram
    score_distribution: Histogram
    taxa_de_aprovacao_atual: Gauge
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


def instrument(application: FastAPI, registry: CollectorRegistry) -> Metrics:
    """Instala o middleware de medição HTTP e a rota `/metrics`.

    Um `CollectorRegistry` próprio por aplicação, nunca o global da biblioteca: a suíte de
    testes cria várias apps no mesmo processo, e no registry global a segunda criação
    colidiria com a primeira, vazando contadores de um teste para o outro.
    """
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
            "forte medido na Etapa 3 contra degradação real, calculado por "
            "credito.monitoring.proxies.taxa_de_aprovacao sobre uma janela deslizante.",
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
