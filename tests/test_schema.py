"""`schema.py` é o vocabulário do domínio: só nomes de coluna e limiares, sem lógica."""

from __future__ import annotations

import ast
from pathlib import Path

import credito.schema
from credito.schema import ALVO, FEATURES


def test_schema_nao_importa_nada_do_projeto():
    """schema.py é folha na árvore de dependências.

    Se ele importar de data/, contracts/ ou model/, volta a ser um módulo de política
    disfarçado de vocabulário e o acoplamento que esta separação removeu reaparece.
    """
    arvore = ast.parse(Path(credito.schema.__file__).read_text(encoding="utf-8"))
    importados = [
        no.module for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom) and no.module
    ]
    assert not [m for m in importados if m.startswith("credito")]


def test_features_tem_dez_colunas_e_nao_inclui_o_alvo():
    assert len(FEATURES) == 10
    assert ALVO not in FEATURES
