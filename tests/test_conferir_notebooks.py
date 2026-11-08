"""A conferência dos notebooks só serve se falhar quando um número muda e só então.

Os dois erros possíveis custam caro de jeitos diferentes: comparar coisa que muda a cada
execução (horário, imagem, endereço de memória) deixaria a CI vermelha para sempre e
ensinaria todo mundo a ignorá-la; deixar de comparar uma saída de texto deixaria passar
justamente o número velho que a conferência existe para pegar.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_CAMINHO = Path(__file__).resolve().parents[1] / "scripts" / "conferir_notebooks.py"


@pytest.fixture(scope="module")
def conferir():
    """Carrega o script pelo caminho, como `make conferir-notebooks` faz."""
    especificacao = importlib.util.spec_from_file_location("conferir_notebooks", _CAMINHO)
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


def _notebook(*saidas, metadata=None):
    return {
        "cells": [
            {"cell_type": "markdown", "source": "# título", "metadata": {}},
            {"cell_type": "code", "outputs": list(saidas), "metadata": metadata or {}},
        ]
    }


def _texto(valor):
    return {"output_type": "stream", "name": "stdout", "text": valor}


def test_o_mesmo_texto_nao_gera_diferenca(conferir):
    nb = _notebook(_texto("AUC-PR 0.3716\n"))
    assert conferir.diferencas(nb, nb, "x") == []


def test_um_numero_diferente_gera_diferenca(conferir):
    salvo = _notebook(_texto("AUC-PR 0.3716\n"))
    novo = _notebook(_texto("AUC-PR 0.3698\n"))
    diff = conferir.diferencas(salvo, novo, "x")
    assert any("0.3716" in linha for linha in diff)
    assert any("0.3698" in linha for linha in diff)


def test_tabela_exibida_entra_na_comparacao(conferir):
    def tabela(valor):
        return {"output_type": "execute_result", "data": {"text/plain": f"aprovação {valor}"}}

    assert conferir.diferencas(_notebook(tabela("0.65")), _notebook(tabela("0.67")), "x")


def test_horario_de_execucao_nao_conta(conferir):
    salvo = _notebook(_texto("ok\n"), metadata={"execution": {"iopub.status.busy": "10:00"}})
    novo = _notebook(_texto("ok\n"), metadata={"execution": {"iopub.status.busy": "11:30"}})
    assert conferir.diferencas(salvo, novo, "x") == []


def test_imagem_nao_conta(conferir):
    def figura(bytes_png):
        return {
            "output_type": "display_data",
            "data": {"image/png": bytes_png, "text/plain": "<Figure size 800x400 with 1 Axes>"},
        }

    assert conferir.diferencas(_notebook(figura("AAAA")), _notebook(figura("BBBB")), "x") == []


def test_endereco_de_memoria_nao_conta(conferir):
    def styler(endereco):
        texto = f"<pandas.io.formats.style.Styler at {endereco}>"
        return {"output_type": "display_data", "data": {"text/plain": texto}}

    salvo, novo = _notebook(styler("0x25c1c0bdeb0")), _notebook(styler("0x7f00aa11"))
    assert conferir.diferencas(salvo, novo, "x") == []


def test_aviso_em_stderr_nao_conta(conferir):
    def aviso(texto):
        return {"output_type": "stream", "name": "stderr", "text": texto}

    salvo = _notebook(_texto("ok\n"), aviso("UserWarning: /home/a/lib.py\n"))
    novo = _notebook(_texto("ok\n"), aviso("UserWarning: C:\\b\\lib.py\n"))
    assert conferir.diferencas(salvo, novo, "x") == []


def test_linha_em_branco_e_espaco_no_fim_nao_contam(conferir):
    salvo = _notebook(_texto("qui-quadrado = 67.9\n"), _texto("\n"))
    novo = _notebook(_texto("qui-quadrado = 67.9   \n"))
    assert conferir.diferencas(salvo, novo, "x") == []


def test_erro_na_reexecucao_aparece_na_diferenca(conferir):
    erro = {"output_type": "error", "ename": "FileNotFoundError", "evalue": "model.joblib"}
    diff = conferir.diferencas(_notebook(_texto("ok\n")), _notebook(erro), "x")
    assert any("FileNotFoundError" in linha for linha in diff)


def test_o_padrao_deixa_de_fora_so_a_analise_causal(conferir):
    existentes = {p.stem for p in (_CAMINHO.parents[1] / "notebooks").glob("*.ipynb")}
    assert set(conferir.PADRAO) == existentes - {"02_analise_causal"}
