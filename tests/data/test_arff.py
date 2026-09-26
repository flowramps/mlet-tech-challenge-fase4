"""O OpenML publica o dataset em ARFF; o leitor precisa dar um DataFrame utilizável."""

from __future__ import annotations

import pytest

from credito.data.arff import ler_arff

ARFF_MINIMO = """% comentário que deve ser ignorado
@RELATION teste

@ATTRIBUTE alvo {No, Yes}
@ATTRIBUTE idade INTEGER
@ATTRIBUTE renda REAL

@DATA
No,57,10121.0
Yes,41,?
No,48,6000.0
"""


def test_le_colunas_e_linhas(tmp_path):
    caminho = tmp_path / "teste.arff"
    caminho.write_text(ARFF_MINIMO, encoding="utf-8")

    frame = ler_arff(caminho)

    assert list(frame.columns) == ["alvo", "idade", "renda"]
    assert len(frame) == 3


def test_interrogacao_vira_nulo(tmp_path):
    caminho = tmp_path / "teste.arff"
    caminho.write_text(ARFF_MINIMO, encoding="utf-8")

    frame = ler_arff(caminho)

    # `?` é a marca de ausente do ARFF. Se ela chegasse como a string "?", a coluna
    # inteira viraria texto e toda estatística numérica sairia errada em silêncio.
    assert frame["renda"].isna().sum() == 1
    assert frame["renda"].dtype.kind == "f"


def test_nome_de_coluna_com_hifen_e_preservado(tmp_path):
    caminho = tmp_path / "hifen.arff"
    caminho.write_text(
        "@RELATION t\n@ATTRIBUTE NumberOfTime30-59Days INTEGER\n@DATA\n3\n", encoding="utf-8"
    )

    assert list(ler_arff(caminho).columns) == ["NumberOfTime30-59Days"]


def test_arquivo_sem_secao_de_dados_falha(tmp_path):
    caminho = tmp_path / "vazio.arff"
    caminho.write_text("@RELATION t\n@ATTRIBUTE a INTEGER\n", encoding="utf-8")

    with pytest.raises(ValueError, match="@DATA"):
        ler_arff(caminho)
