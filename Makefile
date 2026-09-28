# Interface do projeto: o README manda rodar `make X` e `make help` lista tudo.

.PHONY: help install lint format test data train monitor demo-contrato verificar-degradacao

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
