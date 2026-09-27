"""Atribuição causal por intervenção: a diferença entre "a renda mudou e o desempenho caiu,
logo a renda causou" (correlação — três variáveis se deslocam juntas, nada distingue qual
causou o quê) e "fixei tudo, movi só a renda, e medi a queda" (intervenção — o `do(·)` de
verdade). Isso só é possível porque a Produção é simulada: controlamos o processo gerador,
então podemos aplicar o drift a uma variável de cada vez.

Nenhum teste toca rede nem o arquivo real do dataset — só frames sintéticos e modelos
falsos, seguindo a mesma convenção de `tests/drift/test_calibration.py` e
`tests/data/test_simulate.py`. A verificação contra o campeão real e a partição de teste
real é medição separada, reportada na task, não suíte automatizada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credito.data.simulate import MESES, simular_producao
from credito.drift.causal import METRICA, atribuir_degradacao
from credito.model.evaluate import avaliar
from credito.schema import ALVO, FEATURES

SEED = 7
N = 2_000


@pytest.fixture
def amostra() -> pd.DataFrame:
    """Frame sintético com todas as `FEATURES` e o alvo, respeitando os tetos do contrato
    (`DebtRatio` <= 10, atraso <= 20) — para que o cenário "nenhum drift" (k=0 em todas as
    transformações, inclusive o clip de `aplicar_drift_de_divida`) não seja alterado pelo
    próprio clip mesmo sem nenhuma inflação real.
    """
    gerador = np.random.default_rng(SEED)
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
    frame[ALVO] = (gerador.random(N) < 0.0694).astype(int)
    return frame[[ALVO, *FEATURES]]


class _ModeloSensivelAUmaVariavel:
    """Duplo de teste: prevê como função só de UMA feature (`coluna`), ignorando as
    outras nove por completo — o instrumento mais direto para provar que o drift na
    variável que o modelo usa degrada mais que o drift numa que ele ignora, sem depender
    de treinar um modelo real e torcer para o coeficiente sair perto de zero.
    """

    def __init__(self, coluna: str) -> None:
        self._coluna = coluna

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        valor = x[self._coluna].to_numpy(dtype=float)
        z = (valor - valor.mean()) / (valor.std() + 1e-9)
        positivo = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - positivo, positivo])


def _rotulo_correlacionado_com_atraso(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Devolve uma cópia de `frame` cujo alvo é sorteado a partir de uma probabilidade que
    é função só de `NumberOfTime30-59DaysPastDueNotWorse` — para que um modelo que lê essa
    mesma coluna acerte a ordenação bem acima do acaso, sem depender de nenhum ajuste real.
    """
    resultado = frame.copy()
    gerador = np.random.default_rng(seed)
    valor = resultado["NumberOfTime30-59DaysPastDueNotWorse"].to_numpy(dtype=float)
    z = (valor - valor.mean()) / (valor.std() + 1e-9)
    probabilidade = 1.0 / (1.0 + np.exp(-z))
    resultado[ALVO] = (gerador.random(len(resultado)) < probabilidade).astype(int)
    return resultado


# --- Estrutura do resultado --------------------------------------------------------------


def test_decompoe_auc_roc_nao_auc_pr(amostra):
    # Decisão de design central do módulo (ver o docstring de `METRICA` em causal.py):
    # AUC-ROC é insensível à prevalência do lote, ao contrário de AUC-PR, cujo piso
    # estrutural é a própria taxa de positivos. Comparado contra a string literal
    # "auc_roc", não contra a constante `METRICA` importada — um teste que só recomputa o
    # esperado com a própria `METRICA` se autoajusta a qualquer valor que ela tenha e
    # nunca discrimina uma troca de métrica.
    modelo = _ModeloSensivelAUmaVariavel("NumberOfTime30-59DaysPastDueNotWorse")
    frame = _rotulo_correlacionado_com_atraso(amostra, seed=SEED)

    resultado = atribuir_degradacao(frame, modelo, mes=MESES, seed=SEED)

    assert resultado["metrica"] == "auc_roc"
    esperado = avaliar(modelo, frame)["auc_roc"]
    assert resultado["metricas_por_cenario"]["nenhum_drift"] == pytest.approx(esperado)


def test_devolve_as_chaves_esperadas(amostra):
    modelo = _ModeloSensivelAUmaVariavel("DebtRatio")

    resultado = atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)

    assert resultado["metrica"] == METRICA
    assert resultado["mes"] == MESES
    assert set(resultado["metricas_por_cenario"]) == {
        "nenhum_drift",
        "renda",
        "divida",
        "atraso",
        "concept_drift",
        "todos",
    }
    assert set(resultado["degradacao_por_cenario"]) == set(resultado["metricas_por_cenario"])
    assert "soma_dos_efeitos_isolados" in resultado
    assert "efeito_conjunto" in resultado
    assert "termo_de_interacao" in resultado


def test_interacao_fecha_a_conta_algebrica(amostra):
    # Identidade que define o termo de interação: efeito_conjunto = soma_dos_isolados +
    # interacao, por construção — não uma medição, uma tautologia aritmética que a
    # implementação tem que respeitar sempre.
    modelo = _ModeloSensivelAUmaVariavel("DebtRatio")

    resultado = atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)

    soma = resultado["soma_dos_efeitos_isolados"] + resultado["termo_de_interacao"]
    assert soma == pytest.approx(resultado["efeito_conjunto"])


def test_soma_dos_isolados_e_a_soma_dos_quatro_canais_nomeados(amostra):
    # Complementa o teste acima, que sozinho é tautológico: como `termo_de_interacao` é
    # sempre definido como o resto (`efeito_conjunto - soma`), a identidade fecha para
    # QUALQUER definição de `soma`, inclusive uma que esqueça um canal — não prova que os
    # quatro canais certos entraram na soma. Este teste prova isso de forma direta,
    # comparando contra a soma explícita das quatro degradações isoladas nomeadas.
    frame = _rotulo_correlacionado_com_atraso(amostra, seed=SEED)
    modelo = _ModeloSensivelAUmaVariavel("NumberOfTime30-59DaysPastDueNotWorse")

    resultado = atribuir_degradacao(frame, modelo, mes=MESES, seed=SEED)
    degradacao = resultado["degradacao_por_cenario"]

    soma_esperada = (
        degradacao["renda"]
        + degradacao["divida"]
        + degradacao["atraso"]
        + degradacao["concept_drift"]
    )
    assert resultado["soma_dos_efeitos_isolados"] == pytest.approx(soma_esperada)


# --- O cenário sem drift ------------------------------------------------------------------


def test_nenhum_drift_degrada_aproximadamente_zero(amostra):
    modelo = _ModeloSensivelAUmaVariavel("DebtRatio")

    resultado = atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)

    assert resultado["degradacao_por_cenario"]["nenhum_drift"] == pytest.approx(0.0, abs=1e-9)


def test_nenhum_drift_produz_o_mesmo_lote_da_referencia(amostra):
    # O cenário "nenhum drift" tem que medir o modelo sobre o MESMO dado, não uma cópia
    # que só parece igual: a métrica sobre `amostra` direto tem que bater exatamente com a
    # métrica do cenário "nenhum_drift".
    modelo = _ModeloSensivelAUmaVariavel("MonthlyIncome")

    resultado = atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)
    esperado = avaliar(modelo, amostra)[METRICA]

    assert resultado["metricas_por_cenario"]["nenhum_drift"] == pytest.approx(esperado)


# --- Isolar a variável certa: uma que o modelo usa, uma que ele ignora -------------------


def test_variavel_ignorada_degrada_menos_que_variavel_usada(amostra):
    # Modelo lê só o atraso (a variável que `aplicar_drift_de_atraso` desloca) e ignora
    # por completo a renda (a variável que `aplicar_drift_de_renda` desloca). O rótulo é
    # construído para correlacionar com atraso, então a linha de base do modelo é
    # informativa (não fica em ~0,5 por acaso) antes de qualquer drift.
    frame = _rotulo_correlacionado_com_atraso(amostra, seed=SEED)
    modelo = _ModeloSensivelAUmaVariavel("NumberOfTime30-59DaysPastDueNotWorse")

    resultado = atribuir_degradacao(frame, modelo, mes=MESES, seed=SEED)

    degradacao = resultado["degradacao_por_cenario"]
    # Renda não entra na previsão: deslocar renda não muda nenhuma previsão, logo a
    # ordenação (AUC-ROC) fica idêntica e a degradação é EXATAMENTE zero, não só menor.
    assert degradacao["renda"] == pytest.approx(0.0, abs=1e-9)
    assert degradacao["atraso"] > degradacao["renda"]


def test_variavel_ignorada_nao_move_a_previsao_em_nenhuma_linha(amostra):
    # Prova mais direta da mesma propriedade acima, sem depender de AUC-ROC: se o modelo
    # ignora a coluna, a mudança de distribuição dessa coluna não pode mudar nem uma
    # previsão individual.
    modelo = _ModeloSensivelAUmaVariavel("NumberOfTime30-59DaysPastDueNotWorse")

    resultado = atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)

    assert resultado["metricas_por_cenario"]["renda"] == pytest.approx(
        resultado["metricas_por_cenario"]["nenhum_drift"]
    )


# --- O cenário conjunto ---------------------------------------------------------------


def test_todos_degrada_pelo_menos_tanto_quanto_o_maior_isolado(amostra):
    """A propriedade esperada da decomposição — a soma dos efeitos isolados não bate com o
    conjunto, mas o
    conjunto pelo menos iguala o maior isolado — vale aqui porque DUAS transformações
    diferentes atingem a mesma variável que o modelo usa: `aplicar_drift_de_atraso`
    (injeção não monotônica, muda a ordenação) e `aplicar_concept_drift` (inverte rótulo
    justo na região de atraso limpo, que o modelo pontua como baixo risco). Empilhar as
    duas não pode discriminar MENOS do que qualquer uma isolada porque nenhuma cancela a
    outra — não é uma lei geral (o termo de interação pode ser negativo em outros
    modelos; a medição contra o campeão real, reportada à parte, verifica se isso se
    sustenta lá).
    """
    frame = _rotulo_correlacionado_com_atraso(amostra, seed=SEED)
    modelo = _ModeloSensivelAUmaVariavel("NumberOfTime30-59DaysPastDueNotWorse")

    resultado = atribuir_degradacao(frame, modelo, mes=MESES, seed=SEED)

    isolados = [
        resultado["degradacao_por_cenario"][nome]
        for nome in ("renda", "divida", "atraso", "concept_drift")
    ]
    assert resultado["efeito_conjunto"] >= max(isolados)


def test_cenario_todos_reproduz_o_lote_de_simular_producao(amostra):
    # Prova de não-reimplementação: o cenário "todos" tem que usar exatamente a mesma
    # composição (mesma ordem, mesmas sementes por mês) que `simular_producao` usaria para
    # o mesmo mês — senão a atribuição mediria um processo gerador diferente do simulado.
    modelo = _ModeloSensivelAUmaVariavel("DebtRatio")
    mes = MESES

    resultado = atribuir_degradacao(amostra, modelo, mes=mes, seed=SEED)

    lotes = simular_producao(amostra, seed=SEED)
    esperado = avaliar(modelo, lotes[f"mes_{mes:02d}"])[METRICA]

    assert resultado["metricas_por_cenario"]["todos"] == pytest.approx(esperado)


def test_cenario_todos_em_mes_intermediario_reproduz_simular_producao(amostra):
    modelo = _ModeloSensivelAUmaVariavel("DebtRatio")
    mes = 3

    resultado = atribuir_degradacao(amostra, modelo, mes=mes, seed=SEED)

    lotes = simular_producao(amostra, seed=SEED)
    esperado = avaliar(modelo, lotes[f"mes_{mes:02d}"])[METRICA]

    assert resultado["metricas_por_cenario"]["todos"] == pytest.approx(esperado)


# --- Reprodutibilidade ---------------------------------------------------------------


def test_reprodutivel_pela_mesma_semente(amostra):
    modelo = _ModeloSensivelAUmaVariavel("NumberOfTime30-59DaysPastDueNotWorse")

    primeiro = atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)
    segundo = atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)

    assert primeiro["metricas_por_cenario"] == pytest.approx(segundo["metricas_por_cenario"])
    assert primeiro["degradacao_por_cenario"] == pytest.approx(segundo["degradacao_por_cenario"])
    assert primeiro["termo_de_interacao"] == pytest.approx(segundo["termo_de_interacao"])


def test_sementes_diferentes_podem_mudar_os_cenarios_estocasticos(amostra):
    # O oposto do teste acima: os cenários que sorteiam algo (atraso, concept_drift) não
    # podem ser invariantes à semente, senão a "reprodutibilidade" seria só a ausência de
    # aleatoriedade nenhuma, não determinismo controlado por semente.
    modelo = _ModeloSensivelAUmaVariavel("NumberOfTime30-59DaysPastDueNotWorse")

    primeiro = atribuir_degradacao(amostra, modelo, mes=MESES, seed=1)
    segundo = atribuir_degradacao(amostra, modelo, mes=MESES, seed=2)

    assert primeiro["metricas_por_cenario"]["atraso"] != pytest.approx(
        segundo["metricas_por_cenario"]["atraso"]
    )


# --- Chama as transformações reais, não uma reimplementação ------------------------------


def _espiar(monkeypatch, modulo, nome: str, chamadas: list[float]) -> None:
    original = getattr(modulo, nome)

    def _fn(frame, intensidade, **kwargs):
        chamadas.append(intensidade)
        return original(frame, intensidade, **kwargs)

    monkeypatch.setattr(modulo, nome, _fn)


def test_cada_cenario_liga_a_transformacao_certa_e_so_ela(amostra, monkeypatch):
    import credito.drift.causal as causal

    chamadas_renda: list[float] = []
    chamadas_divida: list[float] = []
    chamadas_atraso: list[float] = []
    chamadas_concept: list[float] = []
    _espiar(monkeypatch, causal, "aplicar_drift_de_renda", chamadas_renda)
    _espiar(monkeypatch, causal, "aplicar_drift_de_divida", chamadas_divida)
    _espiar(monkeypatch, causal, "aplicar_drift_de_atraso", chamadas_atraso)
    _espiar(monkeypatch, causal, "aplicar_concept_drift", chamadas_concept)

    modelo = _ModeloSensivelAUmaVariavel("DebtRatio")
    k = MESES / MESES  # 1.0, a intensidade do último mês

    atribuir_degradacao(amostra, modelo, mes=MESES, seed=SEED)

    # Ordem dos cenários: nenhum_drift, renda, divida, atraso, concept_drift, todos —
    # cada função é chamada uma vez por cenário, ligada (k) só quando é a variável do
    # próprio cenário ou o cenário "todos".
    assert chamadas_renda == [0.0, k, 0.0, 0.0, 0.0, k]
    assert chamadas_divida == [0.0, 0.0, k, 0.0, 0.0, k]
    assert chamadas_atraso == [0.0, 0.0, 0.0, k, 0.0, k]
    assert chamadas_concept == [0.0, 0.0, 0.0, 0.0, k, k]
