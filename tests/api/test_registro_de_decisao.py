"""A API registra cada decisão antes de devolvê-la — o que dá objeto ao direito de revisão
(LGPD, art. 20). Sem registro, sem decisão: se a gravação falha, a requisição falha, em vez
de emitir uma aprovação ou recusa que ninguém conseguiria reconstruir depois."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from credito.api.main import create_app
from credito.governanca.decisoes import RegistroJsonl
from credito.schema import FEATURES

METADADOS = {"candidato": "xgboost", "modelo_sha256": "sha-do-campeao"}


def _payload(**overrides: float) -> dict[str, float]:
    base = {
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
    base.update(overrides)
    return base


class _ModeloPorRevolving:
    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        p = entradas["RevolvingUtilizationOfUnsecuredLines"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


class _RegistroQueFalha:
    def registrar(self, decisao) -> None:
        raise OSError("disco cheio")


def test_score_devolve_um_id_de_decisao_unico_por_chamada(tmp_path):
    registro = RegistroJsonl(tmp_path / "d.jsonl")
    with TestClient(create_app(_ModeloPorRevolving(), METADADOS, registro=registro)) as cliente:
        ids = [cliente.post("/score", json=_payload()).json()["id_decisao"] for _ in range(3)]

    assert len(set(ids)) == 3
    assert all(uuid.UUID(i).version == 4 for i in ids)


def test_decisao_registrada_e_a_mesma_que_a_resposta_devolveu(tmp_path):
    registro = RegistroJsonl(tmp_path / "d.jsonl")
    with TestClient(create_app(_ModeloPorRevolving(), METADADOS, registro=registro)) as cliente:
        resposta = cliente.post("/score", json=_payload(RevolvingUtilizationOfUnsecuredLines=0.7))
    corpo = resposta.json()

    decisao = registro.buscar(corpo["id_decisao"])

    assert decisao is not None
    assert decisao.probabilidade_inadimplencia == corpo["probabilidade_inadimplencia"]
    assert decisao.aprovado is corpo["aprovado"] is False
    assert decisao.limiar == corpo["limiar"]
    assert decisao.candidato == "xgboost"
    assert decisao.modelo_sha256 == "sha-do-campeao"
    # O vetor gravado é exatamente o que o modelo recebeu: as dez FEATURES, nada mais.
    assert list(decisao.features) == list(FEATURES)
    assert decisao.features["RevolvingUtilizationOfUnsecuredLines"] == 0.7


def test_identificador_enviado_a_mais_nao_chega_ao_registro(tmp_path):
    # O registro guarda dado pessoal (o vetor de features), mas nenhum identificador: o
    # CPF que o sistema de origem mande por engano morre na validação, antes do registro.
    caminho = tmp_path / "d.jsonl"
    with TestClient(
        create_app(_ModeloPorRevolving(), METADADOS, registro=RegistroJsonl(caminho))
    ) as cliente:
        cliente.post("/score", json={**_payload(), "cpf": "123.456.789-00", "nome": "Fulana"})

    conteudo = caminho.read_text(encoding="utf-8")
    assert "123.456.789-00" not in conteudo
    assert "Fulana" not in conteudo
    assert set(json.loads(conteudo)["features"]) == set(FEATURES)


def test_sem_registro_nao_ha_decisao(tmp_path):
    with TestClient(
        create_app(_ModeloPorRevolving(), METADADOS, registro=_RegistroQueFalha())
    ) as cliente:
        resposta = cliente.post("/score", json=_payload())
        metricas = cliente.get("/metrics").text

    assert resposta.status_code == 503
    assert "aprovado" not in resposta.json()
    # A decisão não emitida também não entra na taxa de aprovação ao vivo: o alarme não
    # pode contar uma decisão que, para o titular, nunca existiu.
    assert "credito_score_distribution_count 0.0" in metricas


def test_sem_registro_injetado_a_api_grava_no_diretorio_configurado(_decisoes_em_tmp):
    with TestClient(create_app(_ModeloPorRevolving(), METADADOS)) as cliente:
        id_decisao = cliente.post("/score", json=_payload()).json()["id_decisao"]

    caminho = _decisoes_em_tmp / "decisoes.jsonl"
    assert RegistroJsonl(caminho).buscar(id_decisao) is not None
