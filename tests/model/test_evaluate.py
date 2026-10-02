"""Acurácia é inútil aqui: prever "adimplente" para todo mundo dá 93%. As métricas que
importam são AUC-PR e o recall da classe positiva — o inadimplente que passa é o erro caro.

O recorte por faixa etária existe porque idade é atributo protegido: sem medir por grupo,
não há como afirmar nada sobre viés."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from credito.model.evaluate import (
    FAIXAS_ETARIAS,
    LIMIAR_QUATRO_QUINTOS,
    METRICAS_GLOBAIS,
    avaliar,
    avaliar_por_faixa_etaria,
    diferenca_de_oportunidade,
    razao_impacto_adverso,
    salvar_metricas,
)
from credito.schema import ALVO, FEATURES


class _ModeloFixo:
    """Devolve a probabilidade que lhe mandarem — isola a avaliação do treino."""

    def __init__(self, probabilidades: np.ndarray) -> None:
        self._probabilidades = probabilidades

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        recorte = self._probabilidades[: len(entradas)]
        return np.column_stack([1 - recorte, recorte])


def _frame(alvo: list[int], idades: list[int] | None = None) -> pd.DataFrame:
    n = len(alvo)
    dados = {coluna: [0.5] * n for coluna in FEATURES}
    dados["age"] = idades if idades is not None else [40] * n
    dados[ALVO] = alvo
    return pd.DataFrame(dados)[[ALVO, *FEATURES]]


def test_modelo_perfeito_tem_auc_pr_um():
    frame = _frame([0, 0, 1, 1])
    modelo = _ModeloFixo(np.array([0.01, 0.02, 0.98, 0.99]))

    metricas = avaliar(modelo, frame)

    assert metricas["auc_pr"] == pytest.approx(1.0)
    assert metricas["auc_roc"] == pytest.approx(1.0)
    assert metricas["recall_positivo"] == pytest.approx(1.0)


def test_modelo_que_nunca_acusa_tem_recall_zero():
    # É este o modelo que a acurácia elogiaria.
    frame = _frame([0] * 9 + [1])
    modelo = _ModeloFixo(np.full(10, 0.01))

    metricas = avaliar(modelo, frame)

    assert metricas["recall_positivo"] == pytest.approx(0.0)
    assert metricas["acuracia"] == pytest.approx(0.9)


def test_metricas_esperadas_estao_presentes():
    metricas = avaliar(_ModeloFixo(np.array([0.2, 0.8])), _frame([0, 1]))

    for chave in ("auc_pr", "auc_roc", "recall_positivo", "precisao_positiva", "acuracia", "n"):
        assert chave in metricas


def test_limiar_desloca_o_recall():
    frame = _frame([0, 1, 1])
    modelo = _ModeloFixo(np.array([0.1, 0.4, 0.6]))

    assert avaliar(modelo, frame, limiar=0.5)["recall_positivo"] == pytest.approx(0.5)
    assert avaliar(modelo, frame, limiar=0.3)["recall_positivo"] == pytest.approx(1.0)


def test_recorte_etario_cobre_as_faixas_presentes():
    frame = _frame([0, 1, 0, 1], idades=[22, 35, 50, 70])
    modelo = _ModeloFixo(np.array([0.1, 0.9, 0.2, 0.8]))

    por_faixa = avaliar_por_faixa_etaria(modelo, frame)

    assert set(por_faixa) <= set(FAIXAS_ETARIAS)
    for faixa in por_faixa.values():
        assert "taxa_de_aprovacao" in faixa
        assert "n" in faixa


def test_faixa_sem_amostra_e_omitida():
    frame = _frame([0, 1], idades=[35, 36])
    modelo = _ModeloFixo(np.array([0.1, 0.9]))

    por_faixa = avaliar_por_faixa_etaria(modelo, frame)

    assert all(faixa["n"] > 0 for faixa in por_faixa.values())
    # As duas idades caem em "26-40": as outras três faixas não têm nenhuma linha e não
    # podem aparecer no resultado como se tivessem sido medidas.
    assert set(por_faixa) == {"26-40"}


def test_faixa_sem_positivo_nao_reporta_recall():
    # Sem nenhum inadimplente na faixa, recall é indefinido — reportar 0.0 seria dizer
    # que o modelo captura "nenhum dos zero" quando na verdade não há nada a capturar.
    frame = _frame([0, 0], idades=[22, 23])
    modelo = _ModeloFixo(np.array([0.1, 0.2]))

    por_faixa = avaliar_por_faixa_etaria(modelo, frame)

    assert "recall_positivo" not in por_faixa["18-25"]


def test_metricas_globais_cobre_exatamente_as_chaves_de_avaliar():
    # `credito.tracking.mlflow_client` importa `METRICAS_GLOBAIS` como o conjunto fechado
    # que autoriza um nome de métrica `campeao.*` — este teste tranca que a tupla nunca
    # fique defasada do que `avaliar` de fato devolve (a mesma disciplina de
    # `test_canal_por_feature_cobre_exatamente_as_variaveis_com_drift`, em
    # `tests/pipeline/test_monitoring.py`).
    metricas = avaliar(_ModeloFixo(np.array([0.2, 0.8])), _frame([0, 1]))

    assert set(metricas) == set(METRICAS_GLOBAIS)


def test_salvar_metricas_grava_json_legivel(tmp_path):
    caminho = tmp_path / "m.json"

    salvar_metricas({"auc_pr": 0.42}, caminho)

    assert json.loads(caminho.read_text(encoding="utf-8"))["auc_pr"] == 0.42


# --- equidade: duas métricas formais, que podem discordar entre si ---------------------
#
# Os fixtures são dicts no formato que `avaliar_por_faixa_etaria` devolve, montados à mão:
# as duas funções são puras sobre esse resultado, e montar o dict é o único jeito de pôr
# uma faixa EXATAMENTE na fronteira dos 4/5 sem depender de arredondamento de modelo.


def _faixa(aprovacao: float, recall: float | None = None) -> dict[str, float]:
    entrada = {"n": 100, "taxa_de_aprovacao": aprovacao}
    if recall is not None:
        entrada["recall_positivo"] = recall
    return entrada


def test_quatro_quintos_compara_cada_faixa_contra_a_mais_aprovada():
    resultado = razao_impacto_adverso({"jovem": _faixa(0.5), "idoso": _faixa(1.0)})

    assert resultado["referencia"] == "idoso"
    assert resultado["faixas"]["jovem"]["razao"] == pytest.approx(0.5)
    assert resultado["faixas"]["idoso"]["razao"] == pytest.approx(1.0)


def test_quatro_quintos_exatamente_na_fronteira_nao_e_impacto_adverso():
    # A regra é "menos de quatro quintos": 0,80 cravado passa. Um `<=` no lugar do `<`
    # inverteria o veredito exatamente aqui, e só aqui — por isso o teste mora na
    # fronteira, não longe dela.
    resultado = razao_impacto_adverso({"a": _faixa(0.8), "b": _faixa(1.0)})

    assert LIMIAR_QUATRO_QUINTOS == 0.8
    assert resultado["faixas"]["a"]["razao"] == 0.8
    assert resultado["faixas"]["a"]["impacto_adverso"] is False


def test_quatro_quintos_logo_abaixo_da_fronteira_e_impacto_adverso():
    resultado = razao_impacto_adverso({"a": _faixa(0.79), "b": _faixa(1.0)})

    assert resultado["faixas"]["a"]["impacto_adverso"] is True
    assert resultado["faixas"]["b"]["impacto_adverso"] is False


def test_quatro_quintos_recusa_resultado_vazio():
    with pytest.raises(ValueError, match="faixa"):
        razao_impacto_adverso({})


def test_quatro_quintos_recusa_quando_ninguem_e_aprovado():
    # Sem nenhuma aprovação em lugar nenhum a razão é 0/0: devolver 0 ou 1 inventaria um
    # veredito sobre um caso em que não há o que comparar.
    with pytest.raises(ValueError, match="aprova"):
        razao_impacto_adverso({"a": _faixa(0.0), "b": _faixa(0.0)})


def test_oportunidade_mede_a_distancia_ate_o_maior_recall():
    resultado = diferenca_de_oportunidade(
        {"jovem": _faixa(0.6, recall=0.8), "idoso": _faixa(0.9, recall=0.5)}
    )

    assert resultado["referencia"] == "jovem"
    assert resultado["faixas"]["idoso"]["diferenca"] == pytest.approx(0.3)
    assert resultado["faixas"]["jovem"]["diferenca"] == pytest.approx(0.0)


def test_oportunidade_omite_faixa_sem_recall_em_vez_de_zerar():
    # `avaliar_por_faixa_etaria` não emite recall numa faixa sem inadimplente real: a
    # métrica é indefinida ali. Tratar a ausência como 0,0 faria essa faixa parecer a
    # mais prejudicada da tabela — o oposto de "não há o que medir".
    resultado = diferenca_de_oportunidade(
        {"sem_positivo": _faixa(0.9), "a": _faixa(0.6, recall=0.8), "b": _faixa(0.7, recall=0.7)}
    )

    assert "sem_positivo" not in resultado["faixas"]
    assert set(resultado["faixas"]) == {"a", "b"}


def test_oportunidade_recusa_quando_nenhuma_faixa_tem_recall():
    with pytest.raises(ValueError, match="recall"):
        diferenca_de_oportunidade({"a": _faixa(0.9), "b": _faixa(0.8)})


def test_as_duas_metricas_podem_apontar_grupos_opostos():
    # O caso que justifica reportar as duas, não uma: o grupo mais aprovado (referência
    # da regra dos 4/5) é o mesmo em que o modelo menos enxerga o inadimplente. Uma só
    # métrica de equidade daria o veredito de uma direção e esconderia a outra.
    por_faixa = {"jovem": _faixa(0.65, recall=0.79), "idoso": _faixa(0.93, recall=0.50)}

    quatro_quintos = razao_impacto_adverso(por_faixa)
    oportunidade = diferenca_de_oportunidade(por_faixa)

    assert quatro_quintos["faixas"]["jovem"]["impacto_adverso"] is True
    assert oportunidade["referencia"] == "jovem"
    assert oportunidade["faixas"]["idoso"]["diferenca"] > 0
