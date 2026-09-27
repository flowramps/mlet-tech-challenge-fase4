"""A Produção é simulada, não coletada: por isso pode ser conferida em dois sentidos ao
mesmo tempo. Data drift e concept drift precisam ser independentes um do outro — um teste
liga só um dos dois e mede que o outro não se move — e todo lote, por mais deslocado que
fique, continua sendo dado válido: quem reprova drift é o detector, nunca o contrato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credito.contracts.pandera_backend import construir_validador
from credito.data.simulate import (
    VARIAVEIS_COM_DRIFT,
    aplicar_concept_drift,
    aplicar_drift_de_atraso,
    aplicar_drift_de_divida,
    aplicar_drift_de_renda,
    simular_producao,
)
from credito.schema import ALVO, FEATURES

N = 800


@pytest.fixture
def amostra() -> pd.DataFrame:
    # Sintética, sem tocar rede nem o arquivo do dataset. A variação contínua por linha
    # (Revolving, DebtRatio, MonthlyIncome) evita que duas linhas colidam em FEATURES e
    # disparem a regra de duplicata do próprio contrato antes de qualquer drift.
    gerador = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": gerador.uniform(0.01, 1.0, N),
            "age": gerador.integers(18, 90, N),
            "NumberOfTime30-59DaysPastDueNotWorse": gerador.integers(0, 3, N),
            "DebtRatio": gerador.uniform(0.02, 1.2, N),
            "MonthlyIncome": gerador.uniform(1500.0, 12000.0, N),
            "NumberOfOpenCreditLinesAndLoans": gerador.integers(1, 20, N),
            "NumberOfTimes90DaysLate": gerador.integers(0, 2, N),
            "NumberRealEstateLoansOrLines": gerador.integers(0, 4, N),
            "NumberOfTime60-89DaysPastDueNotWorse": gerador.integers(0, 2, N),
            "NumberOfDependents": gerador.integers(0, 5, N).astype(float),
        }
    )
    # Uma linha propositalmente perto do teto do contrato (DEBT_RATIO_MAXIMO=10,0): é o
    # que exercita o clip de `aplicar_drift_de_divida` dentro do pipeline inteiro, não só
    # na unidade isolada.
    frame.loc[0, "DebtRatio"] = 9.8
    frame[ALVO] = (gerador.random(N) < 0.0694).astype(int)
    return frame[[ALVO, *FEATURES]]


# --- As quatro transformações, uma a uma -----------------------------------------------


def test_aplicar_drift_de_renda_escala_pela_intensidade():
    frame = pd.DataFrame({"MonthlyIncome": [1000.0, 2000.0]})

    resultado = aplicar_drift_de_renda(frame, 0.5)

    # Coeficiente do desenho: 40% no mês 6 (k=1); em k=0,5 só metade do efeito age.
    assert resultado["MonthlyIncome"].tolist() == [1200.0, 2400.0]


def test_aplicar_drift_de_renda_nao_toca_outras_colunas():
    frame = pd.DataFrame({"MonthlyIncome": [1000.0], "DebtRatio": [0.5]})

    resultado = aplicar_drift_de_renda(frame, 1.0)

    assert resultado["DebtRatio"].tolist() == [0.5]


def test_aplicar_drift_de_divida_escala_pela_intensidade():
    frame = pd.DataFrame({"DebtRatio": [1.0]})

    resultado = aplicar_drift_de_divida(frame, 1.0)

    assert resultado["DebtRatio"].iloc[0] == pytest.approx(1.8)


def test_aplicar_drift_de_divida_limita_no_teto_do_contrato():
    # Medido no arquivo real (117.917 linhas): 9 já chegam exatamente no teto de 10,0.
    # Qualquer fator de inflação >1 as empurraria para fora do contrato sem o clip.
    frame = pd.DataFrame({"DebtRatio": [9.5, 10.0]})

    resultado = aplicar_drift_de_divida(frame, 1.0)

    assert (resultado["DebtRatio"] <= 10.0).all()
    assert resultado["DebtRatio"].iloc[1] == 10.0


def test_aplicar_drift_de_atraso_injeta_fracao_esperada():
    frame = pd.DataFrame({"NumberOfTime30-59DaysPastDueNotWorse": [0] * 1000})

    resultado = aplicar_drift_de_atraso(frame, 1.0, seed=1)

    afetadas = int((resultado["NumberOfTime30-59DaysPastDueNotWorse"] != 0).sum())
    # Fração literal do desenho: 15% das linhas no mês 6 (k=1).
    assert afetadas == 150


def test_aplicar_drift_de_atraso_e_proporcional_a_intensidade():
    frame = pd.DataFrame({"NumberOfTime30-59DaysPastDueNotWorse": [0] * 1000})

    resultado = aplicar_drift_de_atraso(frame, 0.5, seed=1)

    afetadas = int((resultado["NumberOfTime30-59DaysPastDueNotWorse"] != 0).sum())
    assert afetadas == 75


def test_aplicar_drift_de_atraso_nao_ultrapassa_o_teto_plausivel():
    # Linha já perto do teto de atraso plausível (20): o incremento sorteado não pode
    # empurrá-la para fora do contrato.
    frame = pd.DataFrame({"NumberOfTime30-59DaysPastDueNotWorse": [19] * 100})

    resultado = aplicar_drift_de_atraso(frame, 1.0, seed=1)

    assert (resultado["NumberOfTime30-59DaysPastDueNotWorse"] <= 20).all()


def test_aplicar_concept_drift_sorteia_o_alvo_nao_copia():
    # Mesmo DebtRatio, mesma intensidade — se o rótulo fosse copiado, cada bloco de
    # linhas idênticas sairia inteiro 0 ou inteiro 1. O sorteio produz os dois valores.
    frame = pd.DataFrame({"DebtRatio": [0.3] * 500, ALVO: [0] * 500})

    resultado = aplicar_concept_drift(frame, 0.0, seed=1)

    assert set(resultado[ALVO].unique()) == {0, 1}


def test_aplicar_concept_drift_sobe_com_a_intensidade():
    frame = pd.DataFrame({"DebtRatio": [0.3] * 2000, ALVO: [0] * 2000})

    taxa_k0 = aplicar_concept_drift(frame, 0.0, seed=1)[ALVO].mean()
    taxa_k1 = aplicar_concept_drift(frame, 1.0, seed=1)[ALVO].mean()

    assert taxa_k1 > taxa_k0


def test_aplicar_concept_drift_nao_toca_features():
    frame = pd.DataFrame({"DebtRatio": [0.3, 0.9], "MonthlyIncome": [3000.0, 5000.0], ALVO: [0, 1]})

    resultado = aplicar_concept_drift(frame, 1.0, seed=1)

    assert resultado["DebtRatio"].tolist() == [0.3, 0.9]
    assert resultado["MonthlyIncome"].tolist() == [3000.0, 5000.0]


class _ModeloFalso:
    """Duplo de teste: prevê 0,5 para todo mundo, só para provar que a coluna é gravada."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n = len(x)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


def test_modelo_opcional_adiciona_coluna_de_previsao(amostra):
    lotes = simular_producao(amostra, meses=1, seed=42, modelo=_ModeloFalso())

    for lote in lotes.values():
        assert (lote["previsao"] == 0.6).all()


def test_sem_modelo_nao_ha_coluna_de_previsao(amostra):
    lotes = simular_producao(amostra, meses=1, seed=42)

    for lote in lotes.values():
        assert "previsao" not in lote.columns


# --- A composição: simular_producao -----------------------------------------------------


def test_produz_um_lote_por_mes_mais_o_mes_zero(amostra):
    lotes = simular_producao(amostra, meses=6, seed=42)

    assert set(lotes) == {f"mes_{m:02d}" for m in range(7)}


def test_mes_zero_e_a_referencia(amostra):
    lotes = simular_producao(amostra, meses=6, seed=42)

    pd.testing.assert_frame_equal(lotes["mes_00"][list(FEATURES)], amostra[list(FEATURES)])


def test_intensidade_e_monotonica(amostra):
    """A renda mediana precisa subir a cada mês, senão não há 'passagem do tempo'."""
    lotes = simular_producao(amostra, meses=6, seed=42)

    medianas = [lotes[f"mes_{m:02d}"]["MonthlyIncome"].median() for m in range(7)]

    assert medianas == sorted(medianas)
    assert medianas[-1] > medianas[0] * 1.3


def test_todo_lote_passa_no_contrato(amostra):
    """Drift não é invalidez. Se um lote for reprovado, o detector nunca o vê."""
    validador = construir_validador()

    for nome, lote in simular_producao(amostra, meses=6, seed=42).items():
        resultado = validador.validar(lote[list(FEATURES)])
        assert resultado.valido, (nome, resultado.violacoes)


def test_concept_drift_muda_a_taxa_de_positivos(amostra):
    lotes = simular_producao(amostra, meses=6, seed=42)

    taxa_inicial = lotes["mes_00"][ALVO].mean()
    taxa_final = lotes["mes_06"][ALVO].mean()

    assert taxa_final > taxa_inicial * 1.5


def test_data_drift_sozinho_nao_move_o_rotulo(amostra):
    lote = aplicar_drift_de_renda(amostra, 1.0)
    lote = aplicar_drift_de_divida(lote, 1.0)
    lote = aplicar_drift_de_atraso(lote, 1.0, seed=42)

    pd.testing.assert_series_equal(lote[ALVO], amostra[ALVO])


def test_concept_drift_sozinho_nao_move_as_features(amostra):
    lote = aplicar_concept_drift(amostra, 1.0, seed=42)

    pd.testing.assert_frame_equal(lote[list(FEATURES)], amostra[list(FEATURES)])


def test_simulacao_e_reprodutivel(amostra):
    primeira = simular_producao(amostra, meses=6, seed=42)
    segunda = simular_producao(amostra, meses=6, seed=42)

    for nome in primeira:
        pd.testing.assert_frame_equal(primeira[nome], segunda[nome])


def test_colunas_sem_drift_ficam_intactas(amostra):
    colunas_fixas = [coluna for coluna in FEATURES if coluna not in VARIAVEIS_COM_DRIFT]
    lotes = simular_producao(amostra, meses=6, seed=42)

    for lote in lotes.values():
        pd.testing.assert_frame_equal(lote[colunas_fixas], amostra[colunas_fixas])


def test_variaveis_com_drift_e_subconjunto_das_features():
    assert set(VARIAVEIS_COM_DRIFT) <= set(FEATURES)
    assert len(VARIAVEIS_COM_DRIFT) >= 2
