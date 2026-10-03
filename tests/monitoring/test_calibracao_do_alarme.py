"""O alarme de produção lê a taxa de aprovação numa janela das últimas N decisões. Um
limiar para ele só é defensável depois de medir duas coisas: quanto essa taxa oscila sem
drift nenhum (a nula), e que fração das janelas o limiar pega quando há drift (o poder).
Estes testes travam as contas; a medição contra o campeão real mora em
`scripts/calibrar_alarme.py`."""

from __future__ import annotations

import numpy as np
import pytest

from credito.monitoring.calibracao_do_alarme import (
    limiar_inferior,
    nula_da_taxa_de_aprovacao,
    poder_do_alarme,
)


def test_nula_e_reproduzivel_pela_semente():
    probabilidades = np.linspace(0.0, 1.0, 1000)

    a = nula_da_taxa_de_aprovacao(probabilidades, janela=100, n_amostras=50, seed=7)
    b = nula_da_taxa_de_aprovacao(probabilidades, janela=100, n_amostras=50, seed=7)

    np.testing.assert_array_equal(a, b)
    assert a.shape == (50,)


def test_nula_se_centra_na_taxa_de_aprovacao_da_populacao():
    # 30% das probabilidades abaixo do limiar de 0,5: a média das janelas tem que ficar
    # em 0,30 — é a taxa sem drift nenhum.
    probabilidades = np.r_[np.full(300, 0.2), np.full(700, 0.8)]

    nula = nula_da_taxa_de_aprovacao(probabilidades, janela=200, n_amostras=4000, seed=1)

    assert nula.mean() == pytest.approx(0.30, abs=0.005)


def test_janela_maior_oscila_menos():
    # A razão de existir desta calibração: o ruído da taxa numa janela cai com o tamanho
    # dela (desvio de uma proporção ~ 1/raiz(n)). Se não caísse, a amostragem estaria
    # ignorando o tamanho da janela.
    probabilidades = np.r_[np.full(300, 0.2), np.full(700, 0.8)]

    pequena = nula_da_taxa_de_aprovacao(probabilidades, janela=50, n_amostras=4000, seed=1)
    grande = nula_da_taxa_de_aprovacao(probabilidades, janela=800, n_amostras=4000, seed=1)

    assert grande.std() < pequena.std() / 3


def test_limiar_inferior_le_a_cauda_de_baixo():
    nula = np.arange(1, 101) / 100  # 0,01 … 1,00

    assert limiar_inferior(nula, taxa_de_falso_alarme=0.05) == pytest.approx(np.percentile(nula, 5))


def test_limiar_inferior_recusa_taxa_fora_de_zero_e_um():
    with pytest.raises(ValueError, match="falso alarme"):
        limiar_inferior(np.ones(10), taxa_de_falso_alarme=0.0)


def test_poder_e_a_fracao_de_janelas_estritamente_abaixo_do_limiar():
    # Janela exatamente no limiar não dispara: o alarme é "abaixo de", e o limiar foi
    # escolhido como o ponto que a nula só cruza na taxa de falso alarme declarada.
    todas_aprovadas = np.zeros(1000)  # taxa 1,0 em toda janela
    nenhuma_aprovada = np.ones(1000)  # taxa 0,0 em toda janela

    poder = poder_do_alarme(
        {"estavel": todas_aprovadas, "colapso": nenhuma_aprovada},
        limiar_de_alarme=0.5,
        janela=100,
        n_amostras=200,
        seed=3,
    )

    assert poder == {"estavel": 0.0, "colapso": 1.0}


def test_poder_na_propria_nula_fica_perto_da_taxa_de_falso_alarme():
    # Calibração coerente: aplicar o limiar à mesma população que o gerou dispara na taxa
    # de falso alarme escolhida — nem muito mais (limiar frouxo), nem zero (limiar inútil).
    rng = np.random.default_rng(0)
    populacao = rng.uniform(0, 1, 20_000)
    nula = nula_da_taxa_de_aprovacao(populacao, janela=500, n_amostras=4000, seed=1)
    limiar = limiar_inferior(nula, taxa_de_falso_alarme=0.05)

    poder = poder_do_alarme(
        {"mesma": populacao}, limiar_de_alarme=limiar, janela=500, n_amostras=4000, seed=2
    )

    assert 0.02 < poder["mesma"] < 0.08


def test_janela_exatamente_no_limiar_de_alarme_nao_dispara():
    # Toda janela desta população tem taxa 1,0 — exatamente o limiar. Estritamente abaixo:
    # nenhuma dispara. Um `<=` dispararia todas, e só um fixture em cima da fronteira vê
    # essa diferença (o teste anterior, com taxas 0 e 1 e limiar 0,5, não vê).
    poder = poder_do_alarme(
        {"no_limiar": np.zeros(1000)}, limiar_de_alarme=1.0, janela=100, n_amostras=200, seed=3
    )

    assert poder == {"no_limiar": 0.0}
