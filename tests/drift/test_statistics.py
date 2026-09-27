"""PSI e KS precisam ser conferíveis na mão — por isso são medidos aqui contra
distribuições sintéticas simples, nunca contra o arquivo do dataset. O teste central desta
suíte (`test_variavel_discreta_zero_inflada_...`) reproduz uma descoberta feita medindo o
Dataset de Produção simulado na Etapa 2: `NumberOfTime30-59DaysPastDueNotWorse` tem 14
valores distintos e 83,1% de zeros na Referência real; entre o mês 0 e o mês 6 da
simulação seus zeros caem para 70,9% e a média mais que dobra — deslocamento real e
grande. Sob binning por quantil (o padrão ingênuo), esse deslocamento mede PSI ≈ 0,090,
abaixo do limiar de 0,10 que a convenção do setor chama de "estável". Sob binning por
valor (um bin por valor distinto), o mesmo par de amostras mede PSI ≈ 0,14, na banda de
atenção. A causa é mecânica: 83% de massa num único valor consome a maior parte dos 10
cortes de quantil pedidos, que colapsam (via `np.unique`) a só 2 ou 3 cortes efetivos —
sem erro, sem exceção, só um número que mente por padrão.
"""

from __future__ import annotations

import numpy as np
import pytest

from credito.drift.statistics import (
    LIMIAR_CARDINALIDADE_DISCRETA,
    ResultadoKS,
    _psi_a_partir_dos_cortes,
    bins_por_quantil,
    bins_por_valor,
    ks,
    psi,
)

SEED = 7


# --- psi: os casos que a convenção de risco de crédito precisa ver corretos ------------


def test_distribuicoes_identicas_psi_proximo_de_zero_e_p_valor_alto():
    gerador = np.random.default_rng(SEED)
    referencia = gerador.normal(0.0, 1.0, 5000)
    atual = referencia.copy()

    assert psi(referencia, atual) == pytest.approx(0.0, abs=1e-9)
    _, p_valor = ks(referencia, atual)
    assert p_valor == pytest.approx(1.0)


def test_deslocamento_forte_psi_acima_do_limiar_e_p_valor_baixo():
    # Convenção do setor: PSI >= 0,25 é mudança material; p-valor < 0,05 é a mesma
    # conclusão do lado do KS. Um deslocamento de 5 desvios-padrão não deixa dúvida.
    gerador = np.random.default_rng(SEED)
    referencia = gerador.normal(0.0, 1.0, 5000)
    atual = gerador.normal(5.0, 1.0, 5000)

    assert psi(referencia, atual) > 0.25

    resultado = ks(referencia, atual)
    assert resultado.p_valor < 0.05


def test_psi_e_simetrico_o_bastante_para_deslocamentos_pequenos():
    gerador = np.random.default_rng(SEED)
    referencia = gerador.normal(0.0, 1.0, 5000)
    atual = gerador.normal(0.1, 1.0, 5000)

    direto = psi(referencia, atual)
    invertido = psi(atual, referencia)

    assert direto == pytest.approx(invertido, rel=0.2)


def test_bin_vazio_nao_diverge():
    # Referência cobre 0-100 em 10 bins de quantil; a atual inteira cai no primeiro bin.
    # Sem suavização, os bins vazios da atual geram log(0) e o PSI diverge (inf/nan).
    gerador = np.random.default_rng(SEED)
    referencia = gerador.uniform(0.0, 100.0, 5000)
    atual = gerador.uniform(0.0, 10.0, 5000)

    resultado = psi(referencia, atual)

    assert np.isfinite(resultado)
    assert resultado > 0.0


def test_cauda_longa_nao_achata_o_psi():
    # Um outlier extremo (100.000) ao lado de uma massa concentrada em 0-1: bins de
    # largura igual jogariam quase tudo num único bin e esconderiam qualquer
    # deslocamento do grosso da massa. Bins por quantil não têm esse problema porque os
    # cortes vêm das proporções da amostra, não da escala dos valores.
    gerador = np.random.default_rng(SEED)
    referencia = np.concatenate([gerador.uniform(0.0, 1.0, 4999), [100_000.0]])
    atual = np.concatenate([gerador.uniform(0.5, 1.5, 4999), [100_000.0]])

    resultado = psi(referencia, atual)

    assert resultado > 0.05


def test_valor_fora_do_intervalo_da_referencia_nao_desaparece_do_psi():
    # Réplica do artefato que o clip de `DebtRatio` cria no simulador de Produção da
    # Etapa 2: 20% da atual senta exatamente num ponto que a Referência nunca alcança.
    # Se os cortes de quantil não fossem abertos nas pontas, `np.histogram` descartaria
    # essa fatia (nem conta, nem soma 1 nas proporções da atual) e o PSI mediria a fatia
    # que sobrou como se a massa excedente nunca tivesse existido — o oposto de
    # "auditável". Medido: com pontas abertas, PSI ≈ 0,25 (material); com pontas
    # fechadas nos valores observados da Referência, PSI cairia para ≈ 0,05 (estável) e
    # a soma das proporções da atual ficaria abaixo de 1.
    gerador = np.random.default_rng(SEED)
    referencia = gerador.uniform(0.0, 1.0, 5000)
    atual = np.concatenate([gerador.uniform(0.0, 1.0, 4000), np.full(1000, 10.0)])

    assert psi(referencia, atual) > 0.10


def test_bins_por_quantil_deduplica_cortes_repetidos():
    # Massa concentrada (90% no mesmo valor) faz vários quantis coincidirem — replica a
    # concentração de DebtRatio na Referência real. `np.quantile` cru devolveria cortes
    # repetidos, e `np.histogram` recusa bins não estritamente crescentes.
    gerador = np.random.default_rng(SEED)
    referencia = np.concatenate([np.zeros(9000), gerador.uniform(0.01, 5.0, 1000)])

    cortes = bins_por_quantil(referencia, bins=10)

    assert np.all(np.diff(cortes) > 0)
    # Não crasha nem produz log(0) sem suavização.
    atual = np.concatenate([np.zeros(8000), gerador.uniform(0.01, 5.0, 2000)])
    assert np.isfinite(psi(referencia, atual))


def test_lote_atual_menor_que_o_numero_de_bins_nao_crasha():
    # Um lote de produção pode ser menor que os 10 bins pedidos — os cortes vêm só da
    # Referência, então o tamanho da atual nunca deveria travar o histograma.
    gerador = np.random.default_rng(SEED)
    referencia = gerador.normal(0.0, 1.0, 5000)
    atual = gerador.normal(0.0, 1.0, 4)

    assert np.isfinite(psi(referencia, atual, bins=10))


def test_variavel_constante_na_referencia_nao_crasha():
    # Uma variável sem variância na Referência tem cardinalidade 1: PSI não tem como
    # enxergar um deslocamento de valor constante com um único bin (a proporção em
    # ambos os lados é sempre 1,0) — é uma limitação estrutural da estatística, decidida
    # e testada aqui, não um acidente.
    referencia = np.zeros(500)
    atual = np.full(500, 99.0)

    assert psi(referencia, atual) == pytest.approx(0.0, abs=1e-9)


# --- A descoberta central: cardinalidade decide o binning, não um padrão único ---------


def test_cardinalidade_medida_faz_o_limiar_cair_na_lacuna_real():
    # Medido nas dez features da Referência real (117.917 linhas): as quatro colunas de
    # atraso e NumberOfDependents ficam entre 10 e 17 valores distintos,
    # NumberRealEstateLoansOrLines soma 28 — e o próximo salto é para 58
    # (NumberOfOpenCreditLinesAndLoans). O limiar cai no meio dessa lacuna medida.
    assert 28 < LIMIAR_CARDINALIDADE_DISCRETA < 58


def test_variavel_discreta_zero_inflada_dispara_o_binning_por_valor():
    """O teste que pega exatamente a descoberta desta tarefa.

    Réplica sintética da variável real: 14 valores distintos, 83,1% de zeros na
    referência (medido: `ref["NumberOfTime30-59DaysPastDueNotWorse"].nunique()` e
    `.eq(0).mean()` sobre a Referência real de 117.917 linhas), caindo para 70,9% na
    atual com a média mais que dobrando — o mesmo padrão medido entre o mês 0 e o mês 6
    do simulador de produção da Etapa 2. Sob 10 cortes de quantil, 83% de massa num só
    valor colapsa a maioria dos cortes pedidos (`bins_por_quantil` devolve só 2 cortes
    efetivos aqui) e o PSI sai abaixo de 0,10. Sob um bin por valor distinto, o mesmo
    par de amostras mede acima de 0,10. Se um refactor futuro trocar o roteamento de
    `psi()` para sempre usar quantil, este teste é o único que nota — a asserção de
    `psi_valor` volta a bater com `psi_quantil` e ambas caem abaixo de 0,10.
    """
    gerador = np.random.default_rng(SEED)
    valores = np.arange(14)

    def pesos(fracao_zero: float, decaimento: float) -> np.ndarray:
        cauda = decaimento ** np.arange(1, 14)
        cauda = cauda / cauda.sum() * (1 - fracao_zero)
        return np.concatenate(([fracao_zero], cauda))

    referencia = gerador.choice(valores, size=20_000, p=pesos(0.831, 0.55))
    atual = gerador.choice(valores, size=20_000, p=pesos(0.709, 0.80))

    # As proporções sintéticas reproduzem a medição real dentro de casas decimais.
    assert referencia.mean() > 0 and (referencia == 0).mean() == pytest.approx(0.831, abs=0.01)
    assert atual.mean() > referencia.mean() * 2

    # O suporte tem 14 valores possíveis (0 a 13); o mais raro (peso ~1,2 em 20.000
    # sorteios) pode não aparecer em toda seed — o que importa para o roteamento é que a
    # cardinalidade observada fique bem abaixo do limiar, não o valor exato.
    cardinalidade_observada = len(np.unique(referencia))
    assert 10 <= cardinalidade_observada <= 14
    assert cardinalidade_observada <= LIMIAR_CARDINALIDADE_DISCRETA

    psi_quantil = _psi_a_partir_dos_cortes(referencia, atual, bins_por_quantil(referencia, bins=10))
    psi_valor = psi(referencia, atual)  # roteia por cardinalidade: 14 <= limiar, usa bins_por_valor

    assert psi_quantil < 0.10
    assert psi_valor >= 0.10


def test_bins_por_valor_um_bin_por_valor_distinto():
    referencia = np.array([0.0, 1.0, 1.0, 3.0, 3.0, 3.0])

    cortes = bins_por_valor(referencia)

    # Três valores distintos (0, 1, 3) => quatro cortes (três bins fechados, extremos
    # abertos para capturar qualquer valor da atual fora do intervalo da Referência).
    # Os cortes internos são os pontos médios entre valores consecutivos — (0+1)/2 e
    # (1+3)/2 — não os próprios valores distintos, senão um valor exatamente igual a um
    # corte cairia no bin vizinho errado.
    assert len(cortes) == 4
    assert cortes[0] == -np.inf
    assert cortes[1] == pytest.approx(0.5)
    assert cortes[2] == pytest.approx(2.0)
    assert cortes[-1] == np.inf
    assert np.all(np.diff(cortes) > 0)


# --- ks: a direção que não pode ser confundida com a do PSI -----------------------------


def test_ks_devolve_estatistica_e_p_valor_nomeados():
    gerador = np.random.default_rng(SEED)
    referencia = gerador.normal(0.0, 1.0, 1000)
    atual = gerador.normal(0.0, 1.0, 1000)

    resultado = ks(referencia, atual)

    assert isinstance(resultado, ResultadoKS)
    assert isinstance(resultado, tuple)
    assert resultado.estatistica == resultado[0]
    assert resultado.p_valor == resultado[1]
