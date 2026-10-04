"""O gerador de tráfego é a ferramenta da demonstração do alarme — e foi ele que escondeu
um problema real duas vezes: um único timeout o derrubava com traceback no meio da rodada,
e a causa (gravação síncrona do registro de decisão travando sob pressão de disco no host)
passou como "erro sem explicação". Um timeout é um resultado a contar, não um motivo para
parar de medir.

Sem rede: `httpx.MockTransport` responde no lugar da API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

_CAMINHO = Path(__file__).resolve().parents[1] / "scripts" / "gerar_trafego.py"


@pytest.fixture
def trafego():
    especificacao = importlib.util.spec_from_file_location("gerar_trafego", _CAMINHO)
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


def _cliente_que_expira_na(chamada_lenta: int) -> httpx.Client:
    chamadas = {"n": 0}

    def responder(requisicao: httpx.Request) -> httpx.Response:
        chamadas["n"] += 1
        if chamadas["n"] == chamada_lenta:
            raise httpx.ReadTimeout("timed out", request=requisicao)
        return httpx.Response(200, json={"aprovado": True})

    return httpx.Client(transport=httpx.MockTransport(responder))


def test_timeout_no_meio_da_rodada_e_contado_e_a_rodada_continua(trafego):
    payloads = [{"age": 40}] * 10

    contagem = trafego.enviar_lote(
        base_url="http://api", payloads=payloads, cliente=_cliente_que_expira_na(4)
    )

    assert contagem == {200: 9, "timeout": 1}


def test_modo_aleatorio_tambem_conta_timeout_em_vez_de_parar(trafego):
    contagem = trafego.gerar_trafego(
        base_url="http://api", n=20, seed=7, cliente=_cliente_que_expira_na(5)
    )

    assert contagem["timeout"] == 1
    assert sum(contagem.values()) == 20
