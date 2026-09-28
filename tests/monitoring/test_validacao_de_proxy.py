"""Quanto cada proxy sem rótulo antecipa a degradação real — a validação central da Etapa
3. Só é possível medir aqui, com dado sintético, porque a simulação real (Etapa 2) tem
rótulo; a verificação contra o campeão real e a partição de teste real é medição separada,
reportada na task, não suíte automatizada (mesma convenção de `tests/drift/test_causal.py`
e `tests/drift/test_calibration.py`: nenhum teste toca rede nem o arquivo real).
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
    ResultadoCorrelacao,
    correlacao_spearman,
    correlacionar_proxies,
    escolher_alarme,
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


def test_correlacao_spearman_serie_constante_e_nan_sem_lancar_warning():
    # `pyproject.toml` trata warning como erro: se a implementação chamasse
    # `scipy.stats.spearmanr` direto sobre uma série constante, o próprio
    # `ConstantInputWarning` já quebraria este teste antes de qualquer assert — a ausência
    # de exceção aqui já é parte da prova, não só os valores abaixo.
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    resultado = correlacao_spearman(x, y)

    assert math.isnan(resultado.rho)
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


def test_correlacionar_proxies_repassa_limiar_para_taxa_de_aprovacao(lotes, modelo):
    padrao = correlacionar_proxies(lotes, modelo)
    limiar_baixo = correlacionar_proxies(lotes, modelo, limiar=0.01)

    # limiar 0,01: quase ninguém tem score abaixo disso — a taxa de aprovação cai bem
    # abaixo do que o limiar padrão (0,5) produz em qualquer lote com scores tipicamente
    # acima de 0,01.
    for nome in padrao["lotes"]:
        assert (
            limiar_baixo["proxies_por_lote"][nome]["taxa_de_aprovacao"]
            <= padrao["proxies_por_lote"][nome]["taxa_de_aprovacao"]
        )


def test_correlacionar_proxies_repassa_bins_para_psi_do_score(lotes, modelo):
    padrao = correlacionar_proxies(lotes, modelo)
    poucos_bins = correlacionar_proxies(lotes, modelo, bins=2)

    algum_lote = padrao["lotes"][-1]
    assert poucos_bins["proxies_por_lote"][algum_lote]["psi_do_score"] != pytest.approx(
        padrao["proxies_por_lote"][algum_lote]["psi_do_score"]
    )


# --- escolher_alarme: a recomendação honesta, inclusive quando ela é "nenhum proxy" -----


def _resultado(rho: float, intervalo: tuple[float, float]) -> ResultadoCorrelacao:
    return ResultadoCorrelacao(rho=rho, p_valor=0.5, intervalo_confianca=intervalo, n=6)


def test_escolher_alarme_pega_o_maior_rho_absoluto_entre_os_significativos():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.90, (0.30, 0.99))},
        "confianca_media": {"degradacao_auc_roc": _resultado(-0.60, (-0.95, 0.50))},  # cruza 0
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(0.70, (0.10, 0.95))},
    }

    assert escolher_alarme(correlacoes, campo="degradacao_auc_roc") == "psi_do_score"


def test_escolher_alarme_ignora_intervalo_que_cruza_zero_mesmo_com_rho_maior():
    # taxa_de_aprovacao tem |rho| maior, mas o intervalo cruza zero — psi_do_score, com
    # |rho| menor mas intervalo que não cruza zero, é o único elegível.
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.55, (0.05, 0.85))},
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(0.80, (-0.10, 0.99))},
    }

    assert escolher_alarme(correlacoes, campo="degradacao_auc_roc") == "psi_do_score"


def test_escolher_alarme_devolve_none_quando_nenhum_intervalo_exclui_zero():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.40, (-0.40, 0.90))},
        "confianca_media": {"degradacao_auc_roc": _resultado(-0.30, (-0.85, 0.45))},
    }

    assert escolher_alarme(correlacoes, campo="degradacao_auc_roc") is None


def test_escolher_alarme_ignora_correlacao_nan():
    correlacoes = {
        "psi_do_score": {
            "degradacao_auc_roc": _resultado(float("nan"), (float("nan"), float("nan")))
        },
    }

    assert escolher_alarme(correlacoes, campo="degradacao_auc_roc") is None


def test_escolher_alarme_le_o_campo_pedido_nao_sempre_o_mesmo():
    correlacoes = {
        "psi_do_score": {
            "degradacao_auc_roc": _resultado(0.10, (-0.50, 0.60)),  # cruza 0
            "degradacao_lift_acima_do_piso": _resultado(0.85, (0.40, 0.97)),  # não cruza
        },
    }

    assert escolher_alarme(correlacoes, campo="degradacao_auc_roc") is None
    assert escolher_alarme(correlacoes, campo="degradacao_lift_acima_do_piso") == "psi_do_score"


# --- proxies_no_topo: o empate que escolher_alarme, sozinho, esconderia -----------------


def test_proxies_no_topo_devolve_todo_mundo_empatado_no_maior_rho_absoluto():
    # psi_do_score (+1,0) e taxa_de_aprovacao (-1,0) têm o MESMO |rho| — um empate real,
    # do tipo que a simulação monotônica desta etapa de fato produz (ver o docstring de
    # `proxies_no_topo`). confianca_media fica de fora por ter |rho| menor.
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(1.0, (0.90, 1.0))},
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(-1.0, (-1.0, -0.90))},
        "confianca_media": {"degradacao_auc_roc": _resultado(0.5, (0.05, 0.85))},
    }

    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == (
        "psi_do_score",
        "taxa_de_aprovacao",
    )


def test_proxies_no_topo_sem_empate_devolve_um_unico_nome():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.90, (0.30, 0.99))},
        "confianca_media": {"degradacao_auc_roc": _resultado(0.50, (0.05, 0.85))},
    }

    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == ("psi_do_score",)


def test_proxies_no_topo_vazio_quando_nenhum_e_significativo():
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(0.40, (-0.40, 0.90))},
    }

    assert proxies_no_topo(correlacoes, campo="degradacao_auc_roc") == ()


def test_escolher_alarme_em_caso_de_empate_e_o_primeiro_de_proxies_no_topo():
    # A promessa explícita do docstring de escolher_alarme: em empate, é o primeiro
    # elemento alfabético de proxies_no_topo — não uma escolha independente por mérito.
    correlacoes = {
        "psi_do_score": {"degradacao_auc_roc": _resultado(1.0, (0.90, 1.0))},
        "taxa_de_aprovacao": {"degradacao_auc_roc": _resultado(-1.0, (-1.0, -0.90))},
    }

    topo = proxies_no_topo(correlacoes, campo="degradacao_auc_roc")
    assert len(topo) > 1  # a premissa do teste: precisa ser mesmo um empate
    assert escolher_alarme(correlacoes, campo="degradacao_auc_roc") == topo[0]
