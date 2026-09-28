"""Quanto cada proxy sem rótulo antecipa a degradação real — a validação central da Etapa
3. Só é possível medir aqui, com dado sintético, porque a simulação real (Etapa 2) tem
rótulo; a verificação contra o campeão real e a partição de teste real é medição separada,
reportada na task, não suíte automatizada (mesma convenção de `tests/drift/test_causal.py`
e `tests/drift/test_calibration.py`: nenhum teste toca rede nem o arquivo real).

Concordância de ORDEM (Spearman) e força do SINAL (magnitude) respondem perguntas
diferentes — a revisão desta etapa mediu, no próprio run real, dois proxies exatamente
empatados em `rho=±1,0` com magnitudes que discordam por medida (`variacao_relativa`
favorece um, `passo_medio_absoluto` favorece o outro). Os testes abaixo cobrem as duas
leituras separadamente, e cobrem o portão de significância (p-valor exato por permutação,
não o intervalo de confiança fechado — que satura perto de `|rho|=1` independente de `n`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from credito.drift.calibration import degradacao_por_lote
from credito.model.evaluate import avaliar
from credito.monitoring.proxies import sinais_do_lote
from credito.monitoring.validacao_de_proxy import (
    ALFA_SIGNIFICANCIA,
    MagnitudeDoProxy,
    ResultadoCorrelacao,
    correlacao_spearman,
    correlacionar_proxies,
    escolher_alarme,
    magnitude_do_proxy,
    p_valor_exato_spearman,
    proxies_no_topo,
)
from credito.schema import ALVO, FEATURES

SEED = 11


class _ModeloPorScore:
    """Devolve `RevolvingUtilizationOfUnsecuredLines` diretamente como probabilidade — a
    mesma convenção de `_ModeloPorFeature` em `tests/monitoring/test_proxies.py`. Um modelo
    que reage ao conteúdo do frame (em vez de devolver um array fixo por posição) é o que
    garante que `psi_do_score` mede uma distribuição de fato diferente por lote, não o
    mesmo array reaproveitado.
    """

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        p = entradas["RevolvingUtilizationOfUnsecuredLines"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def _lote_sintetico(
    gerador: np.random.Generator,
    *,
    n_pos: int,
    n_neg: int,
    media_pos: float,
    media_neg: float,
    dp: float,
) -> pd.DataFrame:
    """Um lote com separação de classes controlada pela média e pelo desvio-padrão de cada
    grupo — sem esses dois parâmetros aproximando as médias (`media_pos` descendo,
    `media_neg` subindo) e alargando o desvio, a discriminação do "modelo" (que só ecoa a
    própria coluna de score) nunca cairia, e não haveria degradação real para correlacionar
    contra os proxies.
    """
    scores = np.clip(
        np.concatenate(
            [gerador.normal(media_pos, dp, n_pos), gerador.normal(media_neg, dp, n_neg)]
        ),
        0.001,
        0.999,
    )
    n = n_pos + n_neg
    dados = {coluna: [0.5] * n for coluna in FEATURES}
    dados["RevolvingUtilizationOfUnsecuredLines"] = scores
    dados[ALVO] = [1] * n_pos + [0] * n_neg
    return pd.DataFrame(dados)


def _seis_lotes_degradando() -> dict[str, pd.DataFrame]:
    """`mes_00` a `mes_06`: separação de classes decaindo mês a mês (médias convergindo,
    desvio-padrão crescendo) — construído para que a degradação real (AUC-ROC e lift acima
    do piso do "modelo" `_ModeloPorScore`) caia de forma aproximadamente monotônica, o
    suficiente para dar variação real às seis correlações sem depender de nenhum número
    exato.
    """
    gerador = np.random.default_rng(SEED)
    n_pos, n_neg = 40, 160  # 20% de prevalência
    lotes: dict[str, pd.DataFrame] = {}
    for mes in range(7):
        k = mes / 6
        lotes[f"mes_{mes:02d}"] = _lote_sintetico(
            gerador,
            n_pos=n_pos,
            n_neg=n_neg,
            media_pos=0.85 - 0.35 * k,
            media_neg=0.15 + 0.35 * k,
            dp=0.12 + 0.10 * k,
        )
    return lotes


@pytest.fixture
def lotes() -> dict[str, pd.DataFrame]:
    return _seis_lotes_degradando()


@pytest.fixture
def modelo() -> _ModeloPorScore:
    return _ModeloPorScore()


# --- p_valor_exato_spearman: exato por permutação completa, não a aproximação assintótica


def test_p_valor_exato_spearman_n4_perfeito_bate_2_sobre_24():
    # Só a identidade e a reversão total, entre as 4!=24 reordenações possíveis, produzem
    # |rho|=1 — o número que a revisão desta etapa citou de forma independente (0,0833)
    # como o p exato que um rho=1,0 perfeito sustenta com apenas 4 pontos.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x.copy()

    assert p_valor_exato_spearman(x, y) == pytest.approx(2 / 24)


def test_p_valor_exato_spearman_n6_perfeito_bate_2_sobre_720():
    x = np.arange(6.0)
    y = x.copy()

    assert p_valor_exato_spearman(x, y) == pytest.approx(2 / 720)


def test_p_valor_exato_spearman_e_menor_quanto_mais_extremo_o_rho():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_perfeito = x.copy()
    y_moderado = np.array([2.0, 1.0, 4.0, 3.0, 5.0])  # rho menor, não perfeito

    p_perfeito = p_valor_exato_spearman(x, y_perfeito)
    p_moderado = p_valor_exato_spearman(x, y_moderado)

    assert p_perfeito < p_moderado


def test_p_valor_exato_spearman_serie_constante_e_nan():
    x = np.array([5.0] * 6)
    y = np.arange(6.0)

    assert math.isnan(p_valor_exato_spearman(x, y))


def test_p_valor_exato_spearman_no_limite_exato_ainda_calcula(monkeypatch):
    # Pino da borda: n == _N_MAXIMO_PARA_PERMUTACAO_EXATA precisa CALCULAR pela enumeração
    # exata (não é "acima do limite", que troca para Monte Carlo). Uma comparação `>=` em
    # vez de `>` empurraria este caso para a amostragem calada — só um teste que fixa n
    # exatamente igual ao limite, ao lado do teste abaixo que fixa n exatamente um a mais,
    # pega a troca de operador.
    import credito.monitoring.validacao_de_proxy as modulo

    monkeypatch.setattr(modulo, "_N_MAXIMO_PARA_PERMUTACAO_EXATA", 4)
    x = np.array([1.0, 2.0, 3.0, 4.0])  # n=4 == limite artificial
    y = x.copy()

    assert p_valor_exato_spearman(x, y) == pytest.approx(2 / 24)


# --- acima do limite exato: Monte Carlo, nunca nan (ver o docstring do módulo) ----------


def test_p_valor_exato_spearman_acima_do_limite_e_significativo_para_correlacao_perfeita():
    # A regressão que a revisão pediu para fechar: `n=10` já é maior que
    # `_N_MAXIMO_PARA_PERMUTACAO_EXATA` (9) com a constante real do módulo (sem
    # monkeypatch). Uma versão anterior devolvia `nan` aqui, e `_significativos` descartava
    # `nan` como "não significativo" — o mesmo texto de um achado genuíno de ausência de
    # correlação. Isso teria feito um `MESES` maior em `credito.data.simulate` inverter a
    # conclusão da etapa em silêncio.
    x = np.arange(10.0)
    y = x.copy()  # rho = 1,0 exato, n=10

    p = p_valor_exato_spearman(x, y)

    assert not math.isnan(p)
    assert p < ALFA_SIGNIFICANCIA


def test_p_valor_exato_spearman_monte_carlo_bate_o_exato_dentro_da_margem(monkeypatch):
    # n=6 tem resposta exata conhecida (2/720=0,002778); forçar o corte para baixo de 6
    # empurra o MESMO caso para a amostragem de Monte Carlo — a estimativa precisa
    # convergir para perto do valor exato, dentro de uma margem generosa frente ao
    # erro-padrão teórico (~0,00017 em p=0,0028, K=100.000).
    import credito.monitoring.validacao_de_proxy as modulo

    monkeypatch.setattr(modulo, "_N_MAXIMO_PARA_PERMUTACAO_EXATA", 5)
    x = np.arange(6.0)
    y = x.copy()

    estimado = p_valor_exato_spearman(x, y)

    assert estimado == pytest.approx(2 / 720, abs=0.01)


def test_p_valor_exato_spearman_monte_carlo_e_deterministico(monkeypatch):
    # A mesma entrada precisa devolver o MESMO p-valor em toda chamada — uma amostragem sem
    # semente fixa tornaria correlacao_spearman não determinística acima do corte exato,
    # inaceitável para um número que entra num relatório de risco de crédito.
    import credito.monitoring.validacao_de_proxy as modulo

    monkeypatch.setattr(modulo, "_N_MAXIMO_PARA_PERMUTACAO_EXATA", 5)
    x = np.arange(6.0)
    gerador = np.random.default_rng(3)
    y = gerador.normal(size=6)

    primeiro = p_valor_exato_spearman(x, y)
    segundo = p_valor_exato_spearman(x, y)

    assert primeiro == segundo


def test_p_valor_exato_spearman_monte_carlo_correcao_evita_p_zero():
    # n=10 com rho perfeito é um evento raríssimo sob a nula (2 em 10!=3.628.800
    # reordenações) — bem abaixo da resolução de K=100.000 amostras, então a contagem
    # crua quase certamente seria zero. Sem a correção +1/+1 (ver o docstring da função),
    # isso devolveria p=0,0 — indistinguível de "impossível sob a nula", quando só
    # significa "não amostrado". Com a correção, o menor p possível é 1/(K+1) > 0.
    import credito.monitoring.validacao_de_proxy as modulo

    x = np.arange(10.0)
    y = x.copy()

    p = p_valor_exato_spearman(x, y)

    assert p > 0.0
    assert p >= 1 / (modulo._N_AMOSTRAS_MONTE_CARLO + 1) - 1e-12


# --- magnitude_do_proxy: a força do sinal, separada da concordância de ordem ------------


def test_magnitude_do_proxy_calcula_variacao_e_passo_medio():
    serie = np.array([0.10, 0.12, 0.15, 0.20])

    magnitude = magnitude_do_proxy(serie)

    assert magnitude.valor_inicial == pytest.approx(0.10)
    assert magnitude.valor_final == pytest.approx(0.20)
    assert magnitude.variacao_absoluta == pytest.approx(0.10)
    assert magnitude.variacao_relativa == pytest.approx(1.0)  # dobrou: +100%
    # passos: 0,02 / 0,03 / 0,05 -> média 0,0333...
    assert magnitude.passo_medio_absoluto == pytest.approx((0.02 + 0.03 + 0.05) / 3)


def test_magnitude_do_proxy_valor_inicial_zero_variacao_relativa_e_nan():
    serie = np.array([0.0, 0.01, 0.02])

    magnitude = magnitude_do_proxy(serie)

    assert math.isnan(magnitude.variacao_relativa)
    assert magnitude.variacao_absoluta == pytest.approx(0.02)  # o resto continua definido


def test_magnitude_do_proxy_um_unico_ponto_levanta_erro():
    with pytest.raises(ValueError):
        magnitude_do_proxy(np.array([0.5]))


def test_magnitude_do_proxy_passo_medio_nao_relativa_desempatam_diferente():
    # O caso que motivou a correção: um proxy que sobe muito em TERMOS RELATIVOS a partir
    # de uma base minúscula (psi_do_score no run real) pode ter um passo médio absoluto
    # menor que um proxy que sobe pouco em termos relativos a partir de uma base maior
    # (taxa_de_aprovacao no mesmo run) — as duas medidas de magnitude podem discordar sobre
    # qual proxy "se move mais", e é exatamente por isso que elas são medidas separadas.
    base_quase_zero = np.array([0.0013, 0.0048, 0.0107, 0.0142, 0.0220, 0.0291])
    base_grande = np.array([0.7850, 0.7753, 0.7673, 0.7606, 0.7529, 0.7455])

    mag_pequena = magnitude_do_proxy(base_quase_zero)
    mag_grande = magnitude_do_proxy(base_grande)

    assert abs(mag_pequena.variacao_relativa) > abs(mag_grande.variacao_relativa)
    assert mag_pequena.passo_medio_absoluto < mag_grande.passo_medio_absoluto


# --- correlacao_spearman: o intervalo, o uso de postos, e o caso degenerado -------------


def test_correlacao_spearman_e_de_fato_spearman_nao_pearson():
    # Relação monotônica NÃO-linear (cúbica): postos batem perfeitamente (Spearman = 1,0),
    # mas os valores não crescem linearmente (Pearson < 1,0). Se a implementação trocasse
    # `stats.spearmanr` por `stats.pearsonr` por engano, este teste é o único que nota —
    # os dois métodos discordam exatamente neste tipo de série.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = x**3

    resultado = correlacao_spearman(x, y)
    pearson_rho, _ = stats.pearsonr(x, y)

    assert resultado.rho == pytest.approx(1.0)
    assert pearson_rho < 0.99  # a relação não é linear: Pearson não chega a 1,0
    assert resultado.rho != pytest.approx(pearson_rho, abs=1e-3)


def test_correlacao_spearman_bate_bit_a_bit_com_scipy():
    gerador = np.random.default_rng(SEED)
    x = gerador.normal(size=6)
    y = gerador.normal(size=6)

    resultado = correlacao_spearman(x, y)
    rho_esperado, p_esperado = stats.spearmanr(x, y)

    assert resultado.rho == rho_esperado
    assert resultado.p_valor == p_esperado
    assert resultado.n == 6


def test_correlacao_spearman_p_valor_exato_delega_para_p_valor_exato_spearman():
    gerador = np.random.default_rng(SEED)
    x = gerador.normal(size=6)
    y = gerador.normal(size=6)

    resultado = correlacao_spearman(x, y)

    assert resultado.p_valor_exato == p_valor_exato_spearman(x, y)


def test_correlacao_spearman_intervalo_bate_com_formula_de_bonett_wright():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])  # rho != ±1, n=6

    resultado = correlacao_spearman(x, y)

    rho_esperado, _ = stats.spearmanr(x, y)
    z = np.arctanh(rho_esperado)
    erro_padrao = np.sqrt(1.06 / (6 - 3))
    baixo_esperado = np.tanh(z - 1.959963984540054 * erro_padrao)
    alto_esperado = np.tanh(z + 1.959963984540054 * erro_padrao)

    assert resultado.intervalo_confianca[0] == pytest.approx(baixo_esperado)
    assert resultado.intervalo_confianca[1] == pytest.approx(alto_esperado)
    # O ponto fica estritamente dentro do próprio intervalo que o cerca.
    assert resultado.intervalo_confianca[0] < resultado.rho < resultado.intervalo_confianca[1]


def test_correlacao_spearman_rho_perfeito_nao_produz_intervalo_infinito():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = x.copy()  # rho = 1,0 exato

    resultado = correlacao_spearman(x, y)

    assert resultado.rho == pytest.approx(1.0)
    assert math.isfinite(resultado.intervalo_confianca[0])
    assert math.isfinite(resultado.intervalo_confianca[1])


def test_correlacao_spearman_clip_do_rho_perfeito_usa_epsilon_de_1e_9():
    # Pino do valor exato do clip (1e-9), não só "é finito" — a mutação que a revisão
    # descreveu (clip para 1e-2) muda o intervalo de [0,99999999; 1,0] para [0,90; 1,0],
    # e só um teste que recalcula o valor esperado de forma independente (não importando a
    # constante do módulo) pega isso.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = x.copy()  # rho = 1,0 exato, n=6

    resultado = correlacao_spearman(x, y)

    epsilon_esperado = 1e-9
    z_esperado = np.arctanh(1 - epsilon_esperado)
    erro_padrao_esperado = np.sqrt(1.06 / (6 - 3))
    baixo_esperado = np.tanh(z_esperado - 1.959963984540054 * erro_padrao_esperado)
    alto_esperado = np.tanh(z_esperado + 1.959963984540054 * erro_padrao_esperado)

    assert resultado.intervalo_confianca[0] == pytest.approx(baixo_esperado, abs=1e-12)
    assert resultado.intervalo_confianca[1] == pytest.approx(alto_esperado, abs=1e-12)
    # A propriedade concreta que um epsilon 1e4 vezes maior (1e-5) já quebraria: o limite
    # inferior fica a menos de 1e-6 de distância de 1,0 — bem mais apertado do que
    # qualquer clip grosseiro produziria.
    assert resultado.intervalo_confianca[0] > 1.0 - 1e-6


def test_correlacao_spearman_serie_constante_e_nan_sem_lancar_warning():
    # `pyproject.toml` trata warning como erro: se a implementação chamasse
    # `scipy.stats.spearmanr` direto sobre uma série constante, o próprio
    # `ConstantInputWarning` já quebraria este teste antes de qualquer assert — a ausência
    # de exceção aqui já é parte da prova, não só os valores abaixo.
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    resultado = correlacao_spearman(x, y)

    assert math.isnan(resultado.rho)
    assert math.isnan(resultado.p_valor_exato)
    assert math.isnan(resultado.intervalo_confianca[0])
    assert math.isnan(resultado.intervalo_confianca[1])


def test_correlacao_spearman_amostra_pequena_demais_para_intervalo():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 1.0, 2.0])

    resultado = correlacao_spearman(x, y)

    assert math.isfinite(resultado.rho)  # rho continua definido
    assert math.isnan(resultado.intervalo_confianca[0])
    assert math.isnan(resultado.intervalo_confianca[1])


# --- correlacionar_proxies: a composição contra dado sintético controlado ---------------


def test_correlacionar_proxies_usa_a_chave_mais_antiga_como_referencia(lotes, modelo):
    resultado = correlacionar_proxies(lotes, modelo)

    assert resultado["referencia"] == "mes_00"
    assert resultado["lotes"] == [f"mes_{mes:02d}" for mes in range(1, 7)]


def test_correlacionar_proxies_menos_de_dois_lotes_levanta_erro(modelo):
    with pytest.raises(ValueError):
        correlacionar_proxies({"mes_00": pd.DataFrame({c: [0.5] for c in FEATURES})}, modelo)


def test_proxies_por_lote_bate_com_sinais_do_lote_chamado_direto(lotes, modelo):
    resultado = correlacionar_proxies(lotes, modelo)

    esperado = sinais_do_lote(modelo, lotes["mes_03"], referencia=lotes["mes_00"])
    assert resultado["proxies_por_lote"]["mes_03"] == esperado


def test_degradacao_por_lote_bate_com_avaliar_e_degradacao_por_lote_diretos(lotes, modelo):
    resultado = correlacionar_proxies(lotes, modelo)

    metricas_referencia = avaliar(modelo, lotes["mes_00"])
    metricas_mes_04 = avaliar(modelo, lotes["mes_04"])
    lift_referencia = metricas_referencia["auc_pr"] - metricas_referencia["taxa_de_positivos"]
    lift_mes_04 = metricas_mes_04["auc_pr"] - metricas_mes_04["taxa_de_positivos"]

    degradacao = resultado["degradacao_por_lote"]["mes_04"]
    assert degradacao["auc_roc"] == metricas_mes_04["auc_roc"]
    assert degradacao["lift_acima_do_piso"] == pytest.approx(lift_mes_04)
    assert degradacao["degradacao_auc_roc"] == pytest.approx(
        metricas_referencia["auc_roc"] - metricas_mes_04["auc_roc"]
    )
    assert degradacao["degradacao_lift_acima_do_piso"] == pytest.approx(
        lift_referencia - lift_mes_04
    )

    # E reaproveita degradacao_por_lote (nunca recalcula avaliar por fora dela): as métricas
    # cruas do lote batem também com essa chamada direta.
    esperado_bruto = degradacao_por_lote(modelo, lotes)["mes_04"]
    for chave, valor in esperado_bruto.items():
        assert degradacao[chave] == valor


def test_degradacao_real_e_de_fato_monotonicamente_pior_no_dado_sintetico(lotes, modelo):
    # Confere que o fixture faz o que o docstring dele promete: sem isso, os testes de
    # correlação abaixo estariam medindo ruído, não sinal.
    resultado = correlacionar_proxies(lotes, modelo)
    degradacao = resultado["degradacao_por_lote"]

    assert degradacao["mes_06"]["auc_roc"] < degradacao["mes_01"]["auc_roc"]
    assert degradacao["mes_06"]["degradacao_auc_roc"] > degradacao["mes_01"]["degradacao_auc_roc"]


def test_correlacoes_delegam_a_correlacao_spearman_bit_a_bit(lotes, modelo):
    resultado = correlacionar_proxies(lotes, modelo)

    nomes_dos_lotes = resultado["lotes"]
    serie_psi = np.array(
        [resultado["proxies_por_lote"][nome]["psi_do_score"] for nome in nomes_dos_lotes]
    )
    serie_degradacao = np.array(
        [resultado["degradacao_por_lote"][nome]["degradacao_auc_roc"] for nome in nomes_dos_lotes]
    )
    esperado = correlacao_spearman(serie_psi, serie_degradacao)

    assert resultado["correlacoes"]["psi_do_score"]["degradacao_auc_roc"] == esperado


def test_correlacoes_cobre_os_tres_proxies_e_os_dois_campos_de_degradacao(lotes, modelo):
    resultado = correlacionar_proxies(lotes, modelo)

    assert set(resultado["correlacoes"]) == {"psi_do_score", "confianca_media", "taxa_de_aprovacao"}
    for correlacoes_do_proxy in resultado["correlacoes"].values():
        assert set(correlacoes_do_proxy) == {"degradacao_auc_roc", "degradacao_lift_acima_do_piso"}
        for valor in correlacoes_do_proxy.values():
            assert isinstance(valor, ResultadoCorrelacao)


def test_correlacionar_proxies_magnitudes_bate_com_magnitude_do_proxy_direto(lotes, modelo):
    resultado = correlacionar_proxies(lotes, modelo)

    nomes_dos_lotes = resultado["lotes"]
    serie_confianca = np.array(
        [resultado["proxies_por_lote"][nome]["confianca_media"] for nome in nomes_dos_lotes]
    )
    esperado = magnitude_do_proxy(serie_confianca)

    assert resultado["magnitudes"]["confianca_media"] == esperado
    assert set(resultado["magnitudes"]) == {"psi_do_score", "confianca_media", "taxa_de_aprovacao"}
    for magnitude in resultado["magnitudes"].values():
        assert isinstance(magnitude, MagnitudeDoProxy)


def test_correlacionar_proxies_repassa_limiar_para_taxa_de_aprovacao(lotes, modelo):
    padrao = correlacionar_proxies(lotes, modelo)
    limiar_baixo = correlacionar_proxies(lotes, modelo, limiar=0.01)

    # limiar 0,01: quase ninguém tem score abaixo disso — a taxa de aprovação cai
    # ESTRITAMENTE abaixo do que o limiar padrão (0,5) produz em todo lote com scores
    # tipicamente acima de 0,01. Desigualdade estrita (não `<=`): um `limiar` ignorado por
    # engano (sempre comparando contra 0,5) deixaria as duas rodadas idênticas, e só `<`
    # nota a diferença — `<=` aceitaria esse bug calado, como aceitou antes desta correção.
    for nome in padrao["lotes"]:
        assert (
            limiar_baixo["proxies_por_lote"][nome]["taxa_de_aprovacao"]
            < padrao["proxies_por_lote"][nome]["taxa_de_aprovacao"]
        )


def test_correlacionar_proxies_repassa_bins_para_psi_do_score(lotes, modelo):
    padrao = correlacionar_proxies(lotes, modelo)
    poucos_bins = correlacionar_proxies(lotes, modelo, bins=2)

    algum_lote = padrao["lotes"][-1]
    assert poucos_bins["proxies_por_lote"][algum_lote]["psi_do_score"] != pytest.approx(
        padrao["proxies_por_lote"][algum_lote]["psi_do_score"]
    )


# --- _significativos / proxies_no_topo: o portão é o p-valor EXATO, não o intervalo -----


def _resultado(rho: float, p_valor_exato: float) -> ResultadoCorrelacao:
    """Fixture de `ResultadoCorrelacao` para os testes de significância e desempate — só os
    dois campos que `_significativos`/`proxies_no_topo`/`escolher_alarme` de fato leem
    (`rho`, `p_valor_exato`) variam; os demais são valores neutros sem papel na decisão.
    """
    return ResultadoCorrelacao(
        rho=rho, p_valor=0.5, p_valor_exato=p_valor_exato, intervalo_confianca=(0.0, 0.0), n=6
    )


def test_significancia_usa_p_valor_exato_nao_o_intervalo_fechado():
    # O caso que a revisão mediu: n=4 com rho perfeito produz um intervalo de confiança
    # fechado que NÃO cruza zero (o clip perto de |rho|=1 satura, ver
    # test_correlacao_spearman_clip_...), mas o p-valor exato (2/4!=0,0833) não é
    # significativo a 5%. Um portão baseado no intervalo aceitaria este proxy; o portão
    # correto (p_valor_exato) rejeita.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x.copy()
    resultado_n4 = correlacao_spearman(x, y)

    assert resultado_n4.p_valor_exato == pytest.approx(2 / 24)
    assert resultado_n4.p_valor_exato > ALFA_SIGNIFICANCIA  # não deveria passar no portão

    correlacoes = {"psi_do_score": {"degradacao_auc_roc": resultado_n4}}
    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == ()


def test_proxies_no_topo_devolve_todo_mundo_empatado_no_maior_rho_absoluto():
    # psi_do_score (+1,0) e taxa_de_aprovacao (-1,0) têm o MESMO |rho| e os dois são
    # significativos — um empate real, do tipo que a simulação monotônica desta etapa de
    # fato produz. confianca_media fica de fora por não ser significativo.
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(1.0, 0.01)},
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(-1.0, 0.01)},
        "confianca_media": {"degradacao_auc_roc": _resultado(0.5, 0.30)},  # não significativo
    }

    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == (
        "psi_do_score",
        "taxa_de_aprovacao",
    )


def test_proxies_no_topo_sem_empate_devolve_um_unico_nome():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.90, 0.01)},
        "confianca_media": {"degradacao_auc_roc": _resultado(0.50, 0.01)},
    }

    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == ("psi_do_score",)


def test_proxies_no_topo_vazio_quando_nenhum_e_significativo():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.99, 0.20)},  # p_valor_exato alto
    }

    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == ()


def test_proxies_no_topo_ignora_p_valor_exato_nan():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(float("nan"), float("nan"))},
    }

    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == ()


# --- escolher_alarme: agora desempata por magnitude, nunca por acidente alfabético ------


def _magnitude(passo_medio_absoluto: float) -> MagnitudeDoProxy:
    return MagnitudeDoProxy(
        valor_inicial=0.5,
        valor_final=0.5,
        variacao_absoluta=0.0,
        variacao_relativa=0.0,
        passo_medio_absoluto=passo_medio_absoluto,
    )


def test_escolher_alarme_sem_empate_devolve_o_unico_do_topo():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.90, 0.01)},
        "confianca_media": {"degradacao_auc_roc": _resultado(0.50, 0.01)},
    }
    magnitudes = {"psi_do_score": _magnitude(0.01), "confianca_media": _magnitude(0.01)}

    assert escolher_alarme(correlacoes, magnitudes, campo="degradacao_auc_roc") == "psi_do_score"


def test_escolher_alarme_devolve_none_quando_nenhum_e_significativo():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.40, 0.50)},
    }
    magnitudes = {"psi_do_score": _magnitude(0.01)}

    assert escolher_alarme(correlacoes, magnitudes, campo="degradacao_auc_roc") is None


def test_escolher_alarme_desempata_empate_de_rho_pela_maior_magnitude():
    # O caso que a revisão apontou como defeito: dois proxies empatados em |rho|=1,0
    # significativo. A versão anterior devolvia o primeiro em ordem alfabética
    # ("confianca_media") mesmo sendo, neste exemplo, o de MENOR magnitude — exatamente o
    # padrão medido no run real (confianca_media tem o menor passo médio dos três).
    correlacoes = {
        "confianca_media": {"degradacao_auc_roc": _resultado(-1.0, 0.003)},
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(-1.0, 0.003)},
    }
    magnitudes = {
        "confianca_media": _magnitude(0.0033),
        "taxa_de_aprovacao": _magnitude(0.0079),
    }

    assert (
        escolher_alarme(correlacoes, magnitudes, campo="degradacao_auc_roc") == "taxa_de_aprovacao"
    )


def test_escolher_alarme_desempata_por_passo_medio_mesmo_quando_variacao_relativa_discorda():
    # O ponto do Minor que a revisão levantou: as duas medidas de MagnitudeDoProxy podem
    # apontar para vencedores DIFERENTES (exatamente o que o run real mediu — ver o
    # relatório da task). Este teste usa MagnitudeDoProxy completo, não o atalho
    # `_magnitude()` (que zera variacao_relativa em toda entrada e não seria capaz de
    # expor esta divergência), com os números reais do run: psi_do_score vence em variação
    # relativa (+2080%) mas perde em passo médio absoluto (0,0056 < 0,0079) para
    # taxa_de_aprovacao. escolher_alarme precisa seguir passo_medio_absoluto.
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(1.0, 0.003)},
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(-1.0, 0.003)},
    }
    magnitudes = {
        "psi_do_score": MagnitudeDoProxy(
            valor_inicial=0.0013,
            valor_final=0.0291,
            variacao_absoluta=0.0278,
            variacao_relativa=20.8,  # vence em variação relativa
            passo_medio_absoluto=0.0056,
        ),
        "taxa_de_aprovacao": MagnitudeDoProxy(
            valor_inicial=0.7850,
            valor_final=0.7455,
            variacao_absoluta=-0.0394,
            variacao_relativa=-0.05,
            passo_medio_absoluto=0.0079,  # vence em passo médio absoluto
        ),
    }

    assert (
        escolher_alarme(correlacoes, magnitudes, campo="degradacao_auc_roc") == "taxa_de_aprovacao"
    )


def test_escolher_alarme_recusa_quando_magnitude_tambem_empata():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(1.0, 0.003)},
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(-1.0, 0.003)},
    }
    magnitudes = {
        "psi_do_score": _magnitude(0.0056),
        "taxa_de_aprovacao": _magnitude(0.0056),  # empate exato de magnitude também
    }

    assert escolher_alarme(correlacoes, magnitudes, campo="degradacao_auc_roc") is None


def test_escolher_alarme_le_o_campo_pedido_nao_sempre_o_mesmo():
    correlacoes = {
        "psi_do_score": {
            "degradacao_auc_roc": _resultado(0.10, 0.80),  # não significativo
            "degradacao_lift_acima_do_piso": _resultado(0.85, 0.01),  # significativo
        },
    }
    magnitudes = {"psi_do_score": _magnitude(0.01)}

    assert escolher_alarme(correlacoes, magnitudes, campo="degradacao_auc_roc") is None
    assert (
        escolher_alarme(correlacoes, magnitudes, campo="degradacao_lift_acima_do_piso")
        == "psi_do_score"
    )


def test_escolher_alarme_no_run_sintetico_usa_magnitudes_de_correlacionar_proxies(lotes, modelo):
    # Teste de integração: escolher_alarme alimentado pelo próprio retorno de
    # correlacionar_proxies (correlações E magnitudes), não por fixtures isoladas — garante
    # que os dois dicts têm as mesmas chaves e que a função não quebra com o formato real.
    # A asserção é falseável: se houver alarme, ele precisa ser EXATAMENTE o proxy de maior
    # passo_medio_absoluto entre os empatados em proxies_no_topo — recalculado aqui de
    # forma independente, não só "é uma chave válida de magnitudes" (que qualquer proxy,
    # certo ou errado, sempre satisfaria).
    resultado = correlacionar_proxies(lotes, modelo)

    campo = "degradacao_auc_roc"
    topo = proxies_no_topo(resultado["correlacoes"], campo=campo)
    alarme = escolher_alarme(resultado["correlacoes"], resultado["magnitudes"], campo=campo)

    if not topo:
        assert alarme is None
    else:
        esperado = max(topo, key=lambda nome: resultado["magnitudes"][nome].passo_medio_absoluto)
        maior_passo = resultado["magnitudes"][esperado].passo_medio_absoluto
        empatados_na_magnitude = [
            nome
            for nome in topo
            if resultado["magnitudes"][nome].passo_medio_absoluto == maior_passo
        ]
        if len(empatados_na_magnitude) > 1:
            assert alarme is None
        else:
            assert alarme == esperado
