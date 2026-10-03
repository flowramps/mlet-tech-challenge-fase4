"""Toda aplicação criada nos testes da API grava decisões em `tmp_path`, nunca no
diretório `decisoes/` do repositório: o registro de decisão guarda dado pessoal, e um
teste que o deixasse no disco real seria exatamente o vazamento que ele existe para evitar."""

from __future__ import annotations

import pytest

from credito.config import get_settings


@pytest.fixture(autouse=True)
def _decisoes_em_tmp(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("CREDITO_DECISOES_DIR", str(tmp_path / "decisoes"))
    yield tmp_path / "decisoes"
    get_settings.cache_clear()
