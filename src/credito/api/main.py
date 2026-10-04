"""API REST de scoring de risco de crédito.

Serve o campeão publicado por `credito.pipeline.training` (`carregar_modelo`) atrás de
três rotas: `/health` (liveness e identificação do modelo), `/score` (a decisão de
crédito) e `/metrics` (instrumentação Prometheus, ver `credito.api.metrics`).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CollectorRegistry

from credito.api.metrics import instrument
from credito.api.schemas import HealthResponse, ScoreRequest, ScoreResponse
from credito.config import get_settings
from credito.governanca.decisoes import (
    Decisao,
    RegistroDeDecisoes,
    RegistroJsonl,
    sha256_do_arquivo,
)
from credito.model.train import carregar_modelo
from credito.monitoring.proxies import LIMIAR_PADRAO
from credito.schema import FEATURES

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _configure_logging() -> None:
    """Instala um handler de raiz, se ainda não houver.

    O uvicorn só configura os loggers dele, então mensagens da aplicação cairiam num
    root sem handler e sumiriam. Como a API é o processo raiz, configurar o logging é
    responsabilidade dela.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _frame_de_registro(registro: dict[str, float]) -> pd.DataFrame:
    """Monta o `DataFrame` de uma linha só, com exatamente as colunas de `FEATURES` na
    ordem certa — o mesmo recorte que `credito.monitoring.proxies._probabilidades` faz
    sobre um lote inteiro. Fica fora do histograma de inferência de propósito: é
    montagem de payload, não a chamada ao modelo — a mesma separação que faz a métrica
    permitir atribuir uma lentidão ao modelo ou ao servidor.
    """
    return pd.DataFrame([registro], columns=list(FEATURES))


def create_app(
    modelo: Any | None = None,
    metadados: dict[str, Any] | None = None,
    *,
    registro: RegistroDeDecisoes | None = None,
) -> FastAPI:
    """Monta a aplicação.

    Recebendo `modelo`/`metadados` prontos, a API fica testável sem tocar disco nem
    treinar nada; sem eles, o campeão publicado é carregado uma única vez no startup, via
    `credito.model.train.carregar_modelo`, e não a cada requisição. Sem `registro`, as
    decisões vão para o arquivo de `Settings.decisoes_path`.
    """
    _configure_logging()
    registro_de_decisoes = registro or RegistroJsonl(get_settings().decisoes_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if modelo is not None:
            application.state.modelo = modelo
            application.state.metadados = metadados or {}
            application.state.modelo_sha256 = str(
                application.state.metadados.get("modelo_sha256", "desconhecido")
            )
        else:
            settings = get_settings()
            modelo_carregado, metadados_carregados = carregar_modelo(settings.model_path)
            application.state.modelo = modelo_carregado
            application.state.metadados = metadados_carregados
            application.state.modelo_sha256 = sha256_do_arquivo(settings.model_path)

        candidato = application.state.metadados.get("candidato", "desconhecido")
        # O mesmo registro que alimenta /metrics: no Grafana dá para saber qual modelo
        # respondia em cada janela de tempo sem sair do dashboard.
        metricas.model_info.info(
            {
                "candidato": str(candidato),
                "auc_pr": str(application.state.metadados.get("auc_pr", "")),
                "recall_positivo": str(application.state.metadados.get("recall_positivo", "")),
                "seed": str(application.state.metadados.get("seed", "")),
            }
        )
        logger.info("modelo carregado: candidato=%s", candidato)
        yield

    application = FastAPI(
        title="API de Scoring de Risco de Crédito",
        description=(
            "Serve o modelo campeão de risco de crédito e expõe métricas Prometheus para "
            "monitoramento de degradação sem rótulo."
        ),
        version=version("credito"),
        lifespan=lifespan,
    )

    # Registry próprio por aplicação — ver o docstring de `instrument` sobre por quê.
    metricas = instrument(application, CollectorRegistry())
    application.state.metrics = metricas

    @application.get("/health", response_model=HealthResponse, tags=["operação"])
    def health(request: Request) -> HealthResponse:
        """Liveness do serviço e identificação do modelo em uso."""
        candidato = request.app.state.metadados.get("candidato", "desconhecido")
        return HealthResponse(status="ok", candidato=str(candidato))

    @application.post("/score", response_model=ScoreResponse, tags=["scoring"])
    def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
        """Pontua um registro de crédito e devolve a decisão de aprovação."""
        modelo_servido: Any = request.app.state.modelo
        metadados_servidos: dict[str, Any] = request.app.state.metadados
        candidato = str(metadados_servidos.get("candidato", "desconhecido"))

        frame = _frame_de_registro(payload.para_registro())

        # Só a chamada ao modelo entra no relógio: montagem do payload, validação do
        # Pydantic e serialização da resposta ficam de fora, porque são custo do
        # servidor, não do modelo — e é exatamente essa separação que permite atribuir
        # uma lentidão a um lado ou a outro.
        inicio = time.perf_counter()
        probabilidade = float(modelo_servido.predict_proba(frame)[0, 1])
        decorrido = time.perf_counter() - inicio

        # A mesma medição vai para o histograma de inferência: é o que permite atribuir
        # uma lentidão ao modelo em vez de ao servidor HTTP.
        request.app.state.metrics.inference_duration.observe(decorrido)

        # Decisão por registro: mesma convenção de `taxa_de_aprovacao`
        # (`probabilidade < limiar`), reimplementada aqui como comparação de uma linha
        # porque responde uma pergunta diferente (uma decisão individual, não a fração
        # agregada de um lote) — o mesmo motivo que `credito.monitoring.proxies.
        # _probabilidades` documenta para não importar entre módulos que respondem
        # perguntas diferentes. A métrica agregada exposta em /metrics, essa sim, nunca
        # reimplementa: vem de `registrar_score`, que chama
        # `credito.monitoring.proxies.taxa_de_aprovacao` de verdade.
        aprovado = probabilidade < LIMIAR_PADRAO

        # Sem registro, sem decisão: a decisão só é emitida depois de gravada. Uma falha de
        # disco vira 503, nunca uma aprovação ou recusa que ninguém conseguiria reconstruir
        # numa revisão (LGPD, art. 20) — o mesmo princípio do histórico de treino, gravado
        # antes de qualquer desfecho.
        decisao = Decisao(
            id_decisao=str(uuid.uuid4()),
            registrada_em=datetime.now(UTC).isoformat(),
            features=payload.para_registro(),
            probabilidade_inadimplencia=probabilidade,
            aprovado=aprovado,
            limiar=LIMIAR_PADRAO,
            candidato=candidato,
            modelo_sha256=request.app.state.modelo_sha256,
        )
        try:
            registro_de_decisoes.registrar(decisao)
        except OSError as erro:
            logger.error("decisão não registrada, nenhuma decisão emitida: %s", erro)
            raise HTTPException(
                status_code=503,
                detail="decisão não registrada; nenhuma decisão foi emitida",
            ) from erro

        # Só a decisão emitida entra no alarme ao vivo.
        request.app.state.metrics.registrar_score(probabilidade)

        return ScoreResponse(
            id_decisao=decisao.id_decisao,
            probabilidade_inadimplencia=probabilidade,
            aprovado=aprovado,
            limiar=LIMIAR_PADRAO,
            candidato=candidato,
            inference_ms=round(decorrido * 1_000, 3),
        )

    return application


app = create_app()


def main() -> None:
    """Ponto de entrada alternativo a `uvicorn credito.api.main:app`, para `poetry run
    python -m credito.api.main`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
