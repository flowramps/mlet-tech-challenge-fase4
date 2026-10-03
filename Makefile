# Interface do projeto: o README manda rodar `make X` e `make help` lista tudo.

.PHONY: help install lint format test data train monitor demo-contrato verificar-degradacao \
	validar-proxies auditar-privacidade consultar-decisao expurgar-decisoes reproduzir mlflow-up api observabilidade-up observabilidade-down traffic

help:             ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:          ## Instala dependências e hooks
	poetry install
	poetry run pre-commit install

lint:             ## Verifica estilo e formatação
	poetry run ruff check .
	poetry run ruff format --check .

format:           ## Aplica formatação
	poetry run ruff check --fix .
	poetry run ruff format .

test:             ## Roda a suíte com cobertura
	poetry run pytest -ra --cov=credito --cov-report=term-missing

data:             ## Baixa o dataset público
	poetry run python -m credito.data.download

train:            ## Treina os candidatos, avalia e promove o campeão
	poetry run python -m credito.pipeline.training

monitor:          ## Simula a produção e gera os relatórios de drift
	poetry run python -m credito.pipeline.monitoring

demo-contrato:    ## Demonstra o contrato bloqueando um lote defeituoso
	poetry run python scripts/demo_contrato.py

verificar-degradacao: ## Confere a degradação monotônica do campeão e a calibração que a sustenta
	poetry run python scripts/verificar_degradacao.py

validar-proxies:  ## Mede quanto cada proxy sem rótulo antecipa a degradação real do campeão
	poetry run python scripts/validar_proxies.py

auditar-privacidade: ## Mede o risco de reidentificação da Referência e o efeito da generalização
	poetry run python scripts/auditar_privacidade.py

consultar-decisao: ## Reconstrói uma decisão para revisão: make consultar-decisao ID=<id_decisao>
	poetry run python scripts/consultar_decisao.py --id $(ID)

expurgar-decisoes: ## Remove do registro as decisões que passaram do prazo de retenção
	poetry run python scripts/expurgar_decisoes.py

reproduzir:       ## Regenera, do zero, todo número que o README publica (dado, treino, drift, proxies, privacidade)
	$(MAKE) data
	$(MAKE) train
	$(MAKE) monitor
	$(MAKE) verificar-degradacao
	$(MAKE) validar-proxies
	$(MAKE) auditar-privacidade

mlflow-up:        ## Sobe a UI do MLflow contra o mesmo SQLite que `make monitor` grava
	MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1 \
		poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

api:              ## Sobe a API de scoring localmente (fora do Docker), para desenvolvimento
	poetry run uvicorn credito.api.main:app --host 0.0.0.0 --port 8000 --reload

observabilidade-up: ## Sobe a pilha completa (API + Prometheus + Grafana) via Docker Compose
	docker compose up -d --build

observabilidade-down: ## Derruba a pilha subida por `make observabilidade-up`
	docker compose down

traffic:          ## Gera tráfego real contra a API para os painéis terem o que mostrar
	poetry run python scripts/gerar_trafego.py
