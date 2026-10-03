# Build em dois estágios: o estágio de dependências carrega Poetry e ferramentas de
# compilação; o estágio final leva só o virtualenv pronto e o código, o que reduz a
# superfície da imagem e o tempo de subida do container.

FROM python:3.12-slim AS builder

# 2.3.2: a mesma versão que .github/workflows/ci.yml pino para o CI (env.POETRY_VERSION) —
# consistência entre o que testa o projeto e o que sobe em produção.
ENV POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

# --only main: a imagem não carrega pytest nem ruff (grupo `dev`), nem MLflow, Evidently e
# Pandera (grupo `pipeline`) — só o que credito.api precisa para responder requisições.
# `tests/test_ferramental.py` trava que a API nunca passe a importar algo de fora de `main`.
RUN poetry install --only main --no-interaction


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Usuário sem privilégio: um comprometimento do processo de inferência não vira root
# dentro do container.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /app/.venv /app/.venv
COPY src ./src

# O campeão publicado por `make train` (models/model.joblib, 755 KB — fora do git, gerado
# pelo treino; o job de build da CI treina antes de construir) é copiado direto, sem repetir
# o treino no build. `Settings.model_path`
# (src/credito/config.py) resolve a partir de `PROJECT_ROOT = Path(__file__).resolve().
# parents[2]`: com o código em /app/src/credito/config.py, isso é /app — o mesmo WORKDIR
# desta imagem — então o default (models/model.joblib) já aponta para o que a linha abaixo
# copia, sem precisar declarar CREDITO_MODELS_DIR no ambiente.
COPY models/ ./models/

# Sem `chown`: código, virtualenv e modelo ficam de root, legíveis pelo `appuser` e não
# graváveis por ele. O processo só lê (com PYTHONDONTWRITEBYTECODE nem `.pyc` ele grava), e
# um processo comprometido não consegue reescrever o próprio código nem trocar o modelo.
# Um `RUN chown -R` aqui também duplicava o virtualenv inteiro numa camada nova — medido:
# 1,65 GB a mais na imagem e minutos a mais em todo build.
USER appuser

EXPOSE 8000

# `start-period` folgado o suficiente para o carregamento do modelo no lifespan (leitura de
# disco + desserialização joblib de um XGBoost com 300 árvores): um healthcheck que dispara
# antes disso derruba o container em loop de reinício antes do primeiro /health responder.
# `HealthResponse` (src/credito/api/schemas.py) devolve `{"status": "ok", "candidato": ...}`
# com 200 — o teste abaixo confere só o status HTTP, que já é suficiente para liveness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "credito.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
