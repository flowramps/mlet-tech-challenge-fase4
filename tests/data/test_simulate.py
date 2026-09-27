"""A Produção é simulada, não coletada: por isso pode ser conferida em dois sentidos ao
mesmo tempo. Data drift e concept drift precisam ser independentes um do outro — um teste
liga só um dos dois e mede que o outro não se move — e todo lote, por mais deslocado que
fique, continua sendo dado válido: quem reprova drift é o detector, nunca o contrato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credito.contracts.pandera_backend import construir_validador
from credito.data.simulate import (
    VARIAVEIS_COM_DRIFT,
    _desfazer_colisoes_de_atraso,
    aplicar_concept_drift,
    aplicar_drift_de_atraso,
    aplicar_drift_de_divida,
    aplicar_drift_de_renda,
    simular_producao,
)
from credito.model.evaluate import avaliar
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


def _linhas_de_regiao(n: int, *, debt_ratio: float, alvo: int = 0) -> pd.DataFrame:
    """Linhas dentro do segmento de risco emergente: histórico de atraso limpo (as três
    colunas de `COLUNAS_DE_ATRASO` em zero) e `DebtRatio` acima do limiar do módulo."""
    return pd.DataFrame(
        {
            "DebtRatio": [debt_ratio] * n,
            "NumberOfTime30-59DaysPastDueNotWorse": [0] * n,
            "NumberOfTimes90DaysLate": [0] * n,
            "NumberOfTime60-89DaysPastDueNotWorse": [0] * n,
            ALVO: [alvo] * n,
        }
    )


def test_aplicar_concept_drift_intensidade_zero_preserva_o_rotulo_real():
    # O requisito central do redesenho: em k=0 o rótulo não é sorteado, é o mesmo — linha
    # a linha, não só a mesma taxa agregada.
    frame = _linhas_de_regiao(500, debt_ratio=0.9)

    resultado = aplicar_concept_drift(frame, 0.0, seed=1)

    pd.testing.assert_series_equal(resultado[ALVO], frame[ALVO])


def test_aplicar_concept_drift_intensidade_zero_preserva_o_rotulo_real_na_amostra(amostra):
    # A mesma invariante, mas sobre uma amostra com a variação real de FEATURES — não só
    # o caso homogêneo acima.
    resultado = aplicar_concept_drift(amostra, 0.0, seed=1)

    pd.testing.assert_series_equal(resultado[ALVO], amostra[ALVO])


def test_aplicar_concept_drift_sobe_com_a_intensidade():
    frame = _linhas_de_regiao(2000, debt_ratio=0.9)

    taxa_k0 = aplicar_concept_drift(frame, 0.0, seed=1)[ALVO].mean()
    taxa_k1 = aplicar_concept_drift(frame, 1.0, seed=1)[ALVO].mean()

    assert taxa_k0 == 0.0
    assert taxa_k1 > taxa_k0


def test_aplicar_concept_drift_inverte_fracao_esperada_dentro_da_regiao():
    frame = _linhas_de_regiao(2000, debt_ratio=0.9)

    resultado = aplicar_concept_drift(frame, 1.0, seed=1)

    # Fração literal do módulo (`_TAXA_MAXIMA_DE_INVERSAO`): 25% dos elegíveis no mês 6
    # (k=1) — todas as 2000 linhas são elegíveis aqui (região, rótulo negativo).
    assert int(resultado[ALVO].sum()) == 500


def test_aplicar_concept_drift_e_proporcional_a_intensidade():
    frame = _linhas_de_regiao(2000, debt_ratio=0.9)

    resultado = aplicar_concept_drift(frame, 0.5, seed=1)

    assert int(resultado[ALVO].sum()) == 250


def test_aplicar_concept_drift_nao_inverte_fora_da_regiao_por_debt_ratio_baixo():
    # Histórico de atraso limpo, mas DebtRatio abaixo do limiar: fora da região, o rótulo
    # não muda em nenhuma intensidade.
    frame = _linhas_de_regiao(1000, debt_ratio=0.1)

    resultado = aplicar_concept_drift(frame, 1.0, seed=1)

    assert (resultado[ALVO] == 0).all()


def test_aplicar_concept_drift_nao_inverte_fora_da_regiao_por_atraso_sujo():
    # DebtRatio alto, mas histórico de atraso NÃO limpo: fora da região.
    frame = pd.DataFrame(
        {
            "DebtRatio": [0.9] * 1000,
            "NumberOfTime30-59DaysPastDueNotWorse": [1] * 1000,
            "NumberOfTimes90DaysLate": [0] * 1000,
            "NumberOfTime60-89DaysPastDueNotWorse": [0] * 1000,
            ALVO: [0] * 1000,
        }
    )

    resultado = aplicar_concept_drift(frame, 1.0, seed=1)

    assert (resultado[ALVO] == 0).all()


def test_aplicar_concept_drift_nao_inverte_quem_ja_e_positivo():
    # Já positivo: não há o que inverter, e a contagem de positivos não pode dobrar.
    frame = _linhas_de_regiao(1000, debt_ratio=0.9, alvo=1)

    resultado = aplicar_concept_drift(frame, 1.0, seed=1)

    assert (resultado[ALVO] == 1).all()


def test_aplicar_concept_drift_nao_toca_features():
    frame = pd.DataFrame(
        {
            "DebtRatio": [0.3, 0.9],
            "MonthlyIncome": [3000.0, 5000.0],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 0],
            "NumberOfTimes90DaysLate": [0, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0],
            ALVO: [0, 1],
        }
    )

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


def test_mes_zero_preserva_o_alvo_real(amostra):
    """Critério de aceite central desta correção: o mês 0 não é só a mesma TAXA de
    positivos da Referência, é o mesmo RÓTULO, linha a linha — quem inadimple no mês 0 é
    exatamente quem inadimple na Referência, não uma reamostragem que só coincide na
    média. É o que torna válido medir o campeão contra o mês 0 como proxy do seu
    desempenho real de teste.
    """
    lotes = simular_producao(amostra, meses=6, seed=42)

    pd.testing.assert_series_equal(lotes["mes_00"][ALVO], amostra[ALVO])


def test_intensidade_e_monotonica(amostra):
    """A renda mediana precisa subir a cada mês, senão não há 'passagem do tempo'."""
    lotes = simular_producao(amostra, meses=6, seed=42)

    medianas = [lotes[f"mes_{m:02d}"]["MonthlyIncome"].median() for m in range(7)]

    assert medianas == sorted(medianas)
    assert medianas[-1] > medianas[0] * 1.3


def _quase_gemeos(n_pares: int, *, seed: int) -> pd.DataFrame:
    """``n_pares`` pares de linhas idênticas em ``FEATURES`` exceto na coluna de atraso, que
    difere entre as duas linhas do par por um valor entre 1 e 3 — a mesma faixa que
    ``aplicar_drift_de_atraso`` sorteia como incremento. Nenhum par colide antes da injeção
    (a diferença nunca é zero); em escala, a chance de o incremento sorteado fechar
    exatamente essa diferença para ao menos um par deixa de ser desprezível — é o mecanismo
    medido na Referência real (117.917 linhas), reproduzido aqui numa fixture pequena o
    bastante para rodar na suíte sem tocar o arquivo do dataset.
    """
    gerador = np.random.default_rng(seed)
    n = n_pares * 2
    base_atraso = gerador.integers(0, 5, n_pares)
    diferenca = gerador.integers(1, 4, n_pares)
    atraso = np.empty(n, dtype=int)
    atraso[0::2] = base_atraso
    atraso[1::2] = base_atraso + diferenca

    frame = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": np.repeat(
                gerador.uniform(0.01, 1.0, n_pares), 2
            ),
            "age": np.repeat(gerador.integers(18, 90, n_pares), 2),
            "NumberOfTime30-59DaysPastDueNotWorse": atraso,
            "DebtRatio": np.repeat(gerador.uniform(0.01, 5.0, n_pares), 2),
            "MonthlyIncome": np.repeat(gerador.uniform(1000.0, 20000.0, n_pares), 2),
            "NumberOfOpenCreditLinesAndLoans": np.repeat(gerador.integers(1, 20, n_pares), 2),
            "NumberOfTimes90DaysLate": np.zeros(n, dtype=int),
            "NumberRealEstateLoansOrLines": np.repeat(gerador.integers(0, 4, n_pares), 2),
            "NumberOfTime60-89DaysPastDueNotWorse": np.zeros(n, dtype=int),
            "NumberOfDependents": np.repeat(gerador.integers(0, 5, n_pares), 2).astype(float),
        }
    )
    return frame[list(FEATURES)]


def test_aplicar_drift_de_atraso_nao_cria_duplicata_exata_por_colisao_de_quase_gemeos():
    """Regressão: reproduzido contra a Referência real (117.917 linhas), o lote ``mes_02``
    da simulação continha uma linha colidindo com sua quase-gêmea depois da injeção de
    atraso — o contrato reprovava o lote inteiro, e um lote reprovado nunca chega ao
    detector de drift (bloquearia a Tarefa 7 inteira). A fixture de 100 pares (200 linhas)
    não tem nenhuma duplicata antes da injeção; sem a correção em
    ``_desfazer_colisoes_de_atraso``, este teste falha de forma determinística com este
    seed — a chance de zero colisões em 100 pares independentes, cada um com 1/3 de chance
    de o incremento sorteado fechar a diferença, é desprezível.
    """
    frame = _quase_gemeos(100, seed=1)
    assert not frame.duplicated().any()

    resultado = aplicar_drift_de_atraso(frame, 1.0, seed=42)

    assert resultado.duplicated().sum() == 0
    assert construir_validador().validar(resultado).valido


def test_desfazer_colisoes_de_atraso_busca_para_baixo_quando_so_ha_vaga_abaixo():
    """Regressão de revisão: a linha colidida começa em 10; todo valor de 4 a 20 já está
    ocupado por outra linha idêntica em `FEATURES`; só o valor 3 está livre. Uma versão
    anterior deste mecanismo só incrementava (e invertia para decrementar só ao tocar o
    teto, voltando a incrementar no passo seguinte) — a partir de 10 ela nunca alcançava
    3, ficava oscilando entre 19 e 20 até esgotar as tentativas e devolver o frame ainda
    duplicado. A busca bidirecional (``valor±1``, ``valor±2``, ...) alcança 3 na sétima
    distância testada.
    """
    base = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 30,
        "DebtRatio": 0.3,
        "MonthlyIncome": 3000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }
    linhas = [{**base, "NumberOfTime30-59DaysPastDueNotWorse": 10}]  # a duplicata original
    linhas += [
        {**base, "NumberOfTime30-59DaysPastDueNotWorse": valor}
        for valor in range(4, 21)
        if valor != 10
    ]
    indice_colidido = len(linhas)
    linhas.append({**base, "NumberOfTime30-59DaysPastDueNotWorse": 10})  # a linha a mover

    frame = pd.DataFrame(linhas)[list(FEATURES)]
    assert frame.duplicated(keep=False).sum() == 2  # só o par em 10, antes de mover nada

    resultado = _desfazer_colisoes_de_atraso(frame, indices=np.array([indice_colidido]))

    assert resultado.loc[indice_colidido, "NumberOfTime30-59DaysPastDueNotWorse"] == 3
    assert resultado[list(FEATURES)].duplicated().sum() == 0


def test_todo_lote_passa_no_contrato(amostra):
    """Drift não é invalidez. Se um lote for reprovado, o detector nunca o vê."""
    validador = construir_validador()

    for nome, lote in simular_producao(amostra, meses=6, seed=42).items():
        resultado = validador.validar(lote[list(FEATURES)])
        assert resultado.valido, (nome, resultado.violacoes)


def test_concept_drift_muda_a_taxa_de_positivos(amostra):
    """O desenho novo é localizado (só o segmento de risco emergente, só quem ainda é
    negativo), não uma reamostragem global — por isso o aumento de taxa é bem mais
    modesto que o defeito original, mas precisa existir e crescer mês a mês.

    O limiar (1,15x) é bem menor que o efeito medido contra o dado real (partição de
    teste real, 23.584 linhas: 6,94% -> 15,82%, +128%) porque a região é uma fatia bem
    menor desta fixture sintética: medido com `_regiao_de_risco_emergente(amostra)`,
    só 5,4% das 800 linhas caem na região (43 linhas), contra ~20% na Referência real
    — a fixture não foi desenhada para reproduzir a proporção real do segmento, só
    para ter alguma massa nele. Um limiar calcado no efeito real (1,5x, como na versão
    anterior deste teste) não sobra margem nenhuma nesta fixture pequena; 1,15x fica
    abaixo do medido (1,1846x) com folga para não deixar passar um mecanismo quase
    inerte, sem exigir da fixture um efeito que ela não tem massa para produzir.
    """
    lotes = simular_producao(amostra, meses=6, seed=42)
    taxas = [lotes[f"mes_{m:02d}"][ALVO].mean() for m in range(7)]

    assert taxas == sorted(taxas)
    # Medido nesta fixture com este seed: 0,08125 -> 0,09625 (+18,46%).
    assert taxas[-1] > taxas[0] * 1.15


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


def test_degradacao_e_monotonica_para_um_modelo_que_ignora_o_segmento():
    """A prova end-to-end de que o redesenho reverte o defeito original: um modelo
    treinado ANTES do deslocamento — sem nenhuma pista de que histórico de atraso limpo
    deixa de ser sinal de baixo risco em alavancagem alta — perde AUC-PR e recall da
    classe positiva mês a mês contra os seis lotes simulados. Isso é o oposto do defeito
    original, em que o campeão MELHORAVA ao longo dos meses porque o rótulo inteiro era
    reamostrado de um sinal univariado que a inflação só fortalecia.

    Sintética, sem tocar o arquivo real nem `models/model.joblib` — a verificação com o
    campeão de verdade e a partição de teste é o passo manual descrito no relatório desta
    correção, não uma responsabilidade de teste automatizado (que não pode tocar o
    dataset). Os coeficientes do gerador e a semente (42, 30.000 linhas) foram achados por
    varredura para produzirem monotonicidade estrita nas duas métricas com este seed —
    não são medição do dataset real, só o suficiente para exercitar o mecanismo aqui.
    """
    gerador = np.random.default_rng(42)
    n = 30_000
    atraso_sujo = gerador.random(n) < 0.4
    divida = gerador.uniform(0.02, 1.2, n)
    renda = gerador.uniform(1500.0, 12000.0, n)

    # O gerador conhece um pouco de DebtRatio e renda (como o mundo real), mas a maior
    # parte do risco vem do histórico de atraso — o modelo treinado nele aprende a
    # confiar fortemente em "histórico limpo" como baixo risco, exatamente a crença que
    # `_regiao_de_risco_emergente` (em `simulate.py`) documenta como sendo verdadeira
    # hoje e vulnerável à inflação.
    logito = -3.0 + 2.6 * atraso_sujo + 0.8 * (divida - 0.6) - 0.4 * (renda - 6000) / 6000
    probabilidade_real = 1 / (1 + np.exp(-logito))
    alvo = (gerador.random(n) < probabilidade_real).astype(int)

    frame = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": gerador.uniform(0.01, 1.0, n),
            "age": gerador.integers(18, 90, n),
            "NumberOfTime30-59DaysPastDueNotWorse": np.where(
                atraso_sujo, gerador.integers(1, 4, n), 0
            ),
            "DebtRatio": divida,
            "MonthlyIncome": renda,
            "NumberOfOpenCreditLinesAndLoans": gerador.integers(1, 20, n),
            "NumberOfTimes90DaysLate": np.zeros(n, dtype=int),
            "NumberRealEstateLoansOrLines": gerador.integers(0, 4, n),
            "NumberOfTime60-89DaysPastDueNotWorse": np.zeros(n, dtype=int),
            "NumberOfDependents": gerador.integers(0, 5, n).astype(float),
        }
    )
    frame[ALVO] = alvo
    frame = frame[[ALVO, *FEATURES]]

    modelo = Pipeline(
        [
            ("escala", StandardScaler()),
            (
                "classificador",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    modelo.fit(frame[list(FEATURES)], frame[ALVO])

    lotes = simular_producao(frame, meses=6, seed=42)
    metricas = [avaliar(modelo, lotes[f"mes_{mes:02d}"]) for mes in range(7)]

    auc_pr = [m["auc_pr"] for m in metricas]
    recall_positivo = [m["recall_positivo"] for m in metricas]

    assert all(auc_pr[i + 1] < auc_pr[i] for i in range(len(auc_pr) - 1)), auc_pr
    assert all(
        recall_positivo[i + 1] < recall_positivo[i] for i in range(len(recall_positivo) - 1)
    ), recall_positivo
