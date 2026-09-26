"""O download precisa ser cacheado e verificar integridade — um arquivo truncado que
passasse silenciosamente contaminaria todo o resto do pipeline."""

from __future__ import annotations

import hashlib

import httpx
import pytest

from credito.data.download import baixar_dataset

CONTEUDO = b"@RELATION t\n@ATTRIBUTE a INTEGER\n@DATA\n1\n"
MD5 = hashlib.md5(CONTEUDO).hexdigest()


def _transporte(conteudo: bytes, contador: list[int]) -> httpx.MockTransport:
    def responder(request: httpx.Request) -> httpx.Response:
        contador.append(1)
        return httpx.Response(200, content=conteudo)

    return httpx.MockTransport(responder)


def test_baixa_e_grava_no_destino(tmp_path):
    chamadas: list[int] = []
    destino = tmp_path / "raw" / "d.arff"

    resultado = baixar_dataset(
        destino, "https://exemplo/d.arff", MD5, transporte=_transporte(CONTEUDO, chamadas)
    )

    assert resultado == destino
    assert destino.read_bytes() == CONTEUDO
    assert len(chamadas) == 1


def test_segunda_chamada_usa_o_cache(tmp_path):
    chamadas: list[int] = []
    destino = tmp_path / "d.arff"
    transporte = _transporte(CONTEUDO, chamadas)

    baixar_dataset(destino, "https://exemplo/d.arff", MD5, transporte=transporte)
    baixar_dataset(destino, "https://exemplo/d.arff", MD5, transporte=transporte)

    assert len(chamadas) == 1, "o cache válido não deveria disparar um segundo download"


def test_force_refaz_o_download(tmp_path):
    chamadas: list[int] = []
    destino = tmp_path / "d.arff"
    transporte = _transporte(CONTEUDO, chamadas)

    baixar_dataset(destino, "https://exemplo/d.arff", MD5, transporte=transporte)
    baixar_dataset(destino, "https://exemplo/d.arff", MD5, force=True, transporte=transporte)

    assert len(chamadas) == 2


def test_md5_divergente_apaga_o_arquivo_e_falha(tmp_path):
    chamadas: list[int] = []
    destino = tmp_path / "d.arff"

    with pytest.raises(ValueError, match="md5"):
        baixar_dataset(
            destino, "https://exemplo/d.arff", "0" * 32, transporte=_transporte(CONTEUDO, chamadas)
        )

    # Deixar o arquivo corrompido em disco seria pior que não ter arquivo: a execução
    # seguinte o trataria como cache.
    assert not destino.exists()


def test_cache_corrompido_e_rebaixado(tmp_path):
    chamadas: list[int] = []
    destino = tmp_path / "d.arff"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(b"lixo")

    baixar_dataset(
        destino, "https://exemplo/d.arff", MD5, transporte=_transporte(CONTEUDO, chamadas)
    )

    assert destino.read_bytes() == CONTEUDO
    assert len(chamadas) == 1
