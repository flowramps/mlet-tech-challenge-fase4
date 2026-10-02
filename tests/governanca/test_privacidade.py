"""Ausência de nome e CPF não é anonimização: combinação de atributos comuns pode apontar
uma única pessoa. Estes testes travam a CONTA do risco de reidentificação — a medição
contra a Referência real mora em `scripts/auditar_privacidade.py`, porque nenhum teste
toca o arquivo do dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from credito.governanca.privacidade import K_MINIMO, generalizar, risco_de_reidentificacao


def _frame(linhas: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(linhas, columns=["age", "MonthlyIncome", "NumberOfDependents"])


def test_linha_sozinha_na_combinacao_conta_como_unica():
    frame = _frame([(30, 1000.0, 0.0), (30, 1000.0, 0.0), (45, 9000.0, 2.0)])

    risco = risco_de_reidentificacao(frame, ["age", "MonthlyIncome"])

    assert risco["n"] == 3
    assert risco["unicas"] == 1
    assert risco["k_minimo"] == 1


def test_grupo_com_exatamente_k_linhas_nao_conta_como_abaixo_de_k():
    # "abaixo de k" é estritamente menor: um grupo com K_MINIMO linhas já satisfaz
    # k-anonimato. Um `<=` contaria como exposta uma linha que a definição protege.
    frame = _frame([(30, 1000.0, 0.0)] * K_MINIMO + [(40, 2000.0, 1.0)] * (K_MINIMO - 1))

    risco = risco_de_reidentificacao(frame, ["age", "MonthlyIncome"])

    assert risco["abaixo_de_k"] == K_MINIMO - 1
    assert risco["k_minimo"] == K_MINIMO - 1


def test_mais_atributos_nunca_diminuem_o_numero_de_linhas_unicas():
    # Propriedade da partição: acrescentar coluna só refina os grupos. Se a contagem de
    # únicas caísse, a conta estaria agrupando pelo conjunto errado.
    frame = _frame([(30, 1000.0, 0.0), (30, 1000.0, 1.0), (30, 2000.0, 0.0), (30, 2000.0, 0.0)])

    menos = risco_de_reidentificacao(frame, ["age"])
    mais = risco_de_reidentificacao(frame, ["age", "MonthlyIncome", "NumberOfDependents"])

    assert mais["unicas"] >= menos["unicas"]
    assert menos["unicas"] == 0
    assert mais["unicas"] == 2


def test_risco_recusa_lista_vazia_de_quase_identificadores():
    with pytest.raises(ValueError, match="quase-identificador"):
        risco_de_reidentificacao(_frame([(30, 1000.0, 0.0)]), [])


def test_generalizar_troca_valor_exato_por_faixa_e_limita_dependentes():
    frame = _frame([(22, 800.0, 0.0), (35, 3000.0, 5.0), (50, 6000.0, 3.0), (70, 20000.0, 1.0)])

    generalizado = generalizar(frame)

    assert list(generalizado["age"]) == ["18-25", "26-40", "41-60", "61+"]
    # dependentes acima de 3 viram o mesmo grupo que 3: "3 ou mais"
    assert list(generalizado["NumberOfDependents"]) == [0.0, 3.0, 3.0, 1.0]
    # renda vira quintil (0..4), nunca o valor em reais
    assert set(generalizado["MonthlyIncome"]) <= set(range(5))
    assert generalizado["MonthlyIncome"].is_monotonic_increasing


def test_generalizar_nao_altera_o_frame_de_entrada():
    frame = _frame([(22, 800.0, 0.0), (35, 3000.0, 5.0)])
    copia = frame.copy()

    generalizar(frame)

    pd.testing.assert_frame_equal(frame, copia)
