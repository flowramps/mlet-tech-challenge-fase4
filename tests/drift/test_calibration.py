"""A nula do PSI, a correção de múltiplos testes e a degradação do campeão — as três
formas de rigor estatístico que separam "medimos um PSI de 0,12" de "sabemos se 0,12 é
sinal ou ruído amostral neste dado". Nenhum teste toca rede nem o arquivo real do
dataset: a calibração contra a Referência real é medição separada, publicada no README,
não suíte automatizada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credito.drift.calibration import (
    benjamini_hochberg,
    degradacao_por_lote,
    distribuicao_nula_psi,
    limiar_empirico,
)
from credito.schema import ALVO, FEATURES

SEED = 13


# --- distribuicao_nula_psi: quanto PSI o acaso produz sem nenhum drift -------------------


def test_devolve_um_valor_por_amostra():
    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 20_000)

    nula = distribuicao_nula_psi(serie, n_amostras=50, tamanho_lote=2_000, bins=10, seed=SEED)

    assert isinstance(nula, np.ndarray)
    assert nula.shape == (50,)


def test_nula_e_nao_negativa_e_finita():
    # PSI é uma soma de termos (p_atual - p_ref) * log(p_atual / p_ref) com p_atual, p_ref
    # > 0 — a divergência é sempre >= 0 (é uma forma de divergência de Kullback-Leibler
    # simetrizada), nunca negativa nem infinita para amostras finitas.
    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 20_000)

    nula = distribuicao_nula_psi(serie, n_amostras=200, tamanho_lote=2_000, bins=10, seed=SEED)

    assert np.all(np.isfinite(nula))
    assert np.all(nula >= 0.0)


def test_mediana_da_nula_fica_proxima_de_zero():
    # A nula é PSI entre a mesma população e uma reamostra sua — sem nenhum drift
    # verdadeiro, a mediana precisa ficar bem abaixo do limiar de atenção da convenção
    # (0,10): medido com este gerador/semente, tamanho_lote=4000 sobre 20000 linhas, a
    # mediana de 300 reamostras fica em 0,00171 — a asserção usa 0,03 como margem larga o
    # bastante para não ser sensível a mudança de semente, mas estrita o bastante para
    # provar que a nula não se confunde com a banda "moderada" da convenção.
    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 20_000)

    nula = distribuicao_nula_psi(serie, n_amostras=300, tamanho_lote=4_000, bins=10, seed=SEED)

    assert np.median(nula) < 0.03


def test_percentil_99_cresce_quando_o_lote_encolhe():
    """O teste que dá valor à função: lote pequeno produz mais PSI espúrio.

    Medido com este gerador/semente (300 reamostras, bins=10, mesma série de 20.000
    linhas): p99 com tamanho_lote=500 é 0,04238; com tamanho_lote=8.000 é 0,00141 — cerca
    de 30x menor. Se a implementação ignorasse ``tamanho_lote`` (por exemplo, sempre
    reamostrando do tamanho da própria série), os dois p99 sairiam iguais e este teste
    haveria de falhar — é o mutante mais direto que existe para esta função.
    """
    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 20_000)

    nula_lote_pequeno = distribuicao_nula_psi(
        serie, n_amostras=300, tamanho_lote=500, bins=10, seed=SEED
    )
    nula_lote_grande = distribuicao_nula_psi(
        serie, n_amostras=300, tamanho_lote=8_000, bins=10, seed=SEED
    )

    p99_pequeno = limiar_empirico(nula_lote_pequeno, percentil=99)
    p99_grande = limiar_empirico(nula_lote_grande, percentil=99)

    assert p99_pequeno > p99_grande


def test_semente_e_reprodutivel():
    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 5_000)

    primeira = distribuicao_nula_psi(serie, n_amostras=40, tamanho_lote=500, bins=10, seed=99)
    segunda = distribuicao_nula_psi(serie, n_amostras=40, tamanho_lote=500, bins=10, seed=99)

    np.testing.assert_array_equal(primeira, segunda)


def test_lote_maior_que_a_serie_reamostra_com_reposicao_sem_travar():
    # Uma série pequena (menor que o tamanho de lote pedido) não pode travar a função:
    # sem reposição, `np.random.Generator.choice` ergueria erro por pedir mais itens do
    # que a população tem. A nula ainda é uma medição válida — só passa a ser bootstrap
    # (com reposição) em vez de subamostragem sem reposição.
    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 50)

    nula = distribuicao_nula_psi(serie, n_amostras=20, tamanho_lote=200, bins=5, seed=SEED)

    assert np.all(np.isfinite(nula))


def test_reamostragem_usa_o_psi_do_modulo_de_estatisticas(monkeypatch):
    # A nula só calibra o que o detector real mede se usar exatamente a mesma função —
    # com o mesmo roteamento de binning por colapso medido (ver `drift/statistics.py`).
    # Reimplementar o binning aqui, mesmo que pareça equivalente, calibraria uma conta
    # diferente da que o detector aplica. Este teste prova que a função importada
    # (`credito.drift.calibration.psi`) é chamada uma vez por amostra, com os `bins`
    # pedidos — não uma cópia local da lógica de corte.
    import credito.drift.calibration as calibracao

    chamadas: list[tuple[int, int, int]] = []

    def _psi_espiao(referencia, atual, *, bins):
        chamadas.append((len(referencia), len(atual), bins))
        return 0.0

    monkeypatch.setattr(calibracao, "psi", _psi_espiao)

    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 1_000)

    resultado = distribuicao_nula_psi(serie, n_amostras=7, tamanho_lote=100, bins=8, seed=SEED)

    assert len(chamadas) == 7
    # `referencia` é sempre a série inteira (1000), nunca reamostrada; `atual` é sempre
    # do tamanho de lote pedido (100); `bins` é sempre o pedido (8), não um valor
    # hardcoded — os três são o que prova que a chamada delega em vez de reimplementar.
    assert all(
        tamanho_referencia == 1_000 and tamanho_atual == 100 and bins_usado == 8
        for tamanho_referencia, tamanho_atual, bins_usado in chamadas
    )
    assert np.all(resultado == 0.0)


def test_reamostragem_sem_reposicao_nao_repete_linha_quando_lote_cabe_na_serie(monkeypatch):
    # Complementa o teste acima: quando `tamanho_lote <= len(serie)`, a amostra tem que
    # ser um recorte de verdade (sem repetir linha), não um bootstrap disfarçado — do
    # contrário a nula mediria a variância do bootstrap (que difere da variância de uma
    # subamostragem sem reposição) mesmo no caso comum, onde não há necessidade de
    # bootstrap nenhum.
    import credito.drift.calibration as calibracao

    amostras_recebidas: list[np.ndarray] = []

    def _psi_espiao(referencia, atual, *, bins):
        amostras_recebidas.append(np.asarray(atual))
        return 0.0

    monkeypatch.setattr(calibracao, "psi", _psi_espiao)

    gerador = np.random.default_rng(SEED)
    serie = gerador.normal(0.0, 1.0, 1_000)  # valores praticamente todos distintos

    distribuicao_nula_psi(serie, n_amostras=10, tamanho_lote=200, bins=8, seed=SEED)

    for amostra in amostras_recebidas:
        assert len(np.unique(amostra)) == len(amostra)


# --- limiar_empirico: ler um percentil da nula como limiar -------------------------------


def test_limiar_empirico_percentil_conhecido():
    nula = np.arange(0.0, 101.0)  # 0, 1, ..., 100

    assert limiar_empirico(nula, percentil=50) == pytest.approx(50.0)
    assert limiar_empirico(nula, percentil=99) == pytest.approx(99.0)


def test_limiar_empirico_devolve_float_nativo():
    nula = np.array([0.01, 0.02, 0.03])

    resultado = limiar_empirico(nula, percentil=95)

    assert isinstance(resultado, float)


# --- benjamini_hochberg: controle de falsa descoberta, não Bonferroni -------------------


def test_todos_os_p_valores_altos_nao_rejeita_nenhum():
    p_valores = np.array([0.8, 0.9, 0.5, 0.99, 0.6, 0.7, 0.85, 0.55, 0.65, 0.95])

    rejeitados = benjamini_hochberg(p_valores, alfa=0.05)

    assert isinstance(rejeitados, np.ndarray)
    assert rejeitados.dtype == bool
    assert not np.any(rejeitados)


def test_um_p_valor_minusculo_entre_muitos_altos_rejeita_exatamente_um():
    p_valores = np.array([0.8, 0.9, 0.0001, 0.99, 0.6, 0.7, 0.85, 0.55, 0.65, 0.95])

    rejeitados = benjamini_hochberg(p_valores, alfa=0.05)

    assert rejeitados.sum() == 1
    assert rejeitados[2]  # a posição do p-valor minúsculo


def test_bh_rejeita_ao_menos_tantos_quanto_bonferroni():
    """Propriedade matemática do método, não escolha de dado: para qualquer família de
    p-valores, o conjunto que Benjamini-Hochberg rejeita contém o conjunto que Bonferroni
    rejeitaria (``p <= alfa/m``). Roda várias famílias aleatórias — inclusive com
    p-valores minúsculos misturados a altos, e com todos altos — porque uma propriedade
    matemática que só se verifica num caso escolhido a dedo não está de fato provada."""
    alfa = 0.05
    gerador = np.random.default_rng(SEED)

    for _ in range(200):
        m = gerador.integers(2, 40)
        p_valores = gerador.uniform(0.0, 1.0, m)
        # Injeta alguns p-valores minúsculos com probabilidade alta, para não testar só
        # o caso trivial "nada rejeitado por nenhum dos dois".
        if gerador.random() < 0.7:
            n_minusculos = gerador.integers(1, max(2, m // 3))
            indices = gerador.choice(m, size=n_minusculos, replace=False)
            p_valores[indices] = gerador.uniform(0.0, 0.001, n_minusculos)

        bonferroni = p_valores <= (alfa / m)
        bh = benjamini_hochberg(p_valores, alfa=alfa)

        assert bh.sum() >= bonferroni.sum()
        assert np.all(bh[bonferroni])  # todo rejeitado por Bonferroni também é por BH


def test_ordem_de_entrada_nao_importa():
    gerador = np.random.default_rng(SEED)
    p_valores = gerador.uniform(0.0, 1.0, 15)
    p_valores[3] = 0.0002
    p_valores[9] = 0.0004

    permutacao = gerador.permutation(len(p_valores))
    embaralhado = p_valores[permutacao]

    rejeitados_original = benjamini_hochberg(p_valores, alfa=0.05)
    rejeitados_embaralhado = benjamini_hochberg(embaralhado, alfa=0.05)

    # Desfazer a permutação do resultado embaralhado tem que bater com o original.
    desfeito = np.empty_like(rejeitados_embaralhado)
    desfeito[permutacao] = rejeitados_embaralhado
    np.testing.assert_array_equal(rejeitados_original, desfeito)


def test_p_valor_exatamente_no_limiar_do_proprio_rank_e_rejeitado():
    # A definição de BH usa "<=", não "<": um p-valor exatamente igual ao limiar do seu
    # rank (`rank/m * alfa`) é rejeitado, não descartado por uma casa decimal de
    # distância. Com m=1, o limiar do único rank é `1/1 * 0,05 = 0,05` — o p-valor abaixo
    # cai exatamente nele, isolando a fronteira sem depender de nenhum outro p-valor.
    p_valores = np.array([0.05])

    rejeitados = benjamini_hochberg(p_valores, alfa=0.05)

    assert rejeitados[0]


def test_p_valor_exatamente_zero_e_sempre_rejeitado():
    p_valores = np.array([0.0, 0.5, 0.6, 0.7, 0.8])

    rejeitados = benjamini_hochberg(p_valores, alfa=0.05)

    assert rejeitados[0]


def test_p_valor_exatamente_um_nunca_e_rejeitado_com_alfa_convencional():
    p_valores = np.array([1.0, 0.001, 0.002, 0.003])

    rejeitados = benjamini_hochberg(p_valores, alfa=0.05)

    assert not rejeitados[0]


def test_empates_no_p_valor_de_corte_sao_tratados_igualmente():
    # Dois p-valores idênticos, ambos abaixo do limiar de corte, têm que sair rejeitados
    # junto — nunca um sim e o outro não. (Um sweep de mutação confirmou que uma
    # implementação por posição de rank chega ao mesmo resultado aqui — ver o docstring
    # de `benjamini_hochberg`; este teste documenta a garantia em si, não distingue as
    # duas formas de implementar.)
    p_valores = np.array([0.001, 0.001, 0.9, 0.95, 0.85, 0.8])

    rejeitados = benjamini_hochberg(p_valores, alfa=0.05)

    assert rejeitados[0] == rejeitados[1] == True  # noqa: E712


def test_alfa_default_e_005():
    # p=0,07 cai exatamente entre 0,05 e 0,10 (m=1, o limiar do único rank é o próprio
    # alfa) — um teste que usasse só p-valores muito baixos ou muito altos não notaria
    # se o default fosse outro alfa plausível como 0,10; medido: com este p, o default
    # errado (0,10) rejeitaria e o correto (0,05) não, então a comparação discrimina.
    p_valores = np.array([0.07])

    com_default = benjamini_hochberg(p_valores)
    com_explicito = benjamini_hochberg(p_valores, alfa=0.05)

    np.testing.assert_array_equal(com_default, com_explicito)
    assert not com_default[0]


# --- degradacao_por_lote: a consequência que o drift de feature só sugere ---------------


class _ModeloEspiao:
    """Devolve uma probabilidade fixa por linha e proíbe retreino — o campeão publicado
    é aplicado, nunca reajustado ao lote que está sendo medido."""

    def __init__(self, probabilidade_positiva: np.ndarray) -> None:
        self._probabilidade_positiva = probabilidade_positiva

    def fit(self, *_args, **_kwargs):
        raise AssertionError(
            "degradacao_por_lote não pode retreinar o campeão publicado — só avaliar"
        )

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        recorte = self._probabilidade_positiva[: len(entradas)]
        return np.column_stack([1 - recorte, recorte])


def _lote(alvo: list[int]) -> pd.DataFrame:
    n = len(alvo)
    dados = {coluna: [0.5] * n for coluna in FEATURES}
    dados[ALVO] = alvo
    return pd.DataFrame(dados)[[ALVO, *FEATURES]]


def test_degradacao_por_lote_devolve_uma_entrada_por_lote():
    lotes = {
        "mes_00": _lote([0, 0, 1, 1]),
        "mes_06": _lote([0, 1, 1, 1]),
    }
    modelo = _ModeloEspiao(np.array([0.01, 0.02, 0.9, 0.95]))

    resultado = degradacao_por_lote(modelo, lotes)

    assert set(resultado.keys()) == {"mes_00", "mes_06"}


def test_degradacao_por_lote_reporta_auc_pr_recall_e_auc_roc():
    lotes = {"mes_00": _lote([0, 0, 1, 1])}
    modelo = _ModeloEspiao(np.array([0.01, 0.02, 0.98, 0.99]))

    resultado = degradacao_por_lote(modelo, lotes)

    metricas = resultado["mes_00"]
    assert metricas["auc_pr"] == pytest.approx(1.0)
    assert metricas["auc_roc"] == pytest.approx(1.0)
    assert metricas["recall_positivo"] == pytest.approx(1.0)


def test_degradacao_por_lote_nunca_retreina_o_campeao():
    # `_ModeloEspiao.fit` ergue AssertionError — se `degradacao_por_lote` chamasse
    # `.fit` em algum lote, o teste falharia por isso, não passaria em silêncio.
    lotes = {
        "mes_00": _lote([0, 0, 1, 1]),
        "mes_03": _lote([1, 0, 1, 0]),
        "mes_06": _lote([1, 1, 0, 1]),
    }
    modelo = _ModeloEspiao(np.array([0.2, 0.3, 0.7, 0.8]))

    degradacao_por_lote(modelo, lotes)  # não deve levantar


def test_degradacao_por_lote_usa_avaliar_do_modulo_de_evaluate(monkeypatch):
    # Prova que a função delega a `model.evaluate.avaliar` em vez de reimplementar as
    # métricas — a mesma conta que o resto do projeto usa para medir o campeão.
    import credito.drift.calibration as calibracao

    chamadas: list[tuple[object, pd.DataFrame]] = []

    def _avaliar_espiao(modelo, frame, **_kwargs):
        chamadas.append((modelo, frame))
        return {"auc_pr": 0.42}

    monkeypatch.setattr(calibracao, "avaliar", _avaliar_espiao)

    lotes = {"mes_00": _lote([0, 1]), "mes_01": _lote([1, 0])}
    modelo = _ModeloEspiao(np.array([0.5, 0.5]))

    resultado = degradacao_por_lote(modelo, lotes)

    assert len(chamadas) == 2
    assert resultado == {"mes_00": {"auc_pr": 0.42}, "mes_01": {"auc_pr": 0.42}}


def test_degradacao_por_lote_degrada_junto_com_o_deslocamento_do_rotulo():
    # Réplica pequena e determinística do sinal que esta camada existe para provar: o mesmo
    # modelo (mesmas probabilidades fixas por posição) mede AUC-PR alta quando o rótulo
    # concorda com a ordem das probabilidades e AUC-PR baixa quando o rótulo se
    # deslocou o bastante para inverter essa ordem — a consequência que o drift de
    # feature sozinho não mostra.
    probabilidades = np.array([0.1, 0.2, 0.8, 0.9])
    modelo = _ModeloEspiao(probabilidades)
    lotes = {
        "concorda": _lote([0, 0, 1, 1]),
        "inverte": _lote([1, 1, 0, 0]),
    }

    resultado = degradacao_por_lote(modelo, lotes)

    assert resultado["concorda"]["auc_pr"] > resultado["inverte"]["auc_pr"]
