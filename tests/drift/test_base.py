"""A fronteira do drift existe para que a classificação de severidade seja declarada uma
vez e a biblioteca que calcula PSI e KS fique atrás da interface — e para que "drift" nunca
vire um jeito de derrubar o run, a distinção que este módulo preserva da Etapa 1."""

from __future__ import annotations

from credito.config import Settings
from credito.drift.base import (
    DriftDeFeature,
    DriftReport,
    Severidade,
    classificar,
)

# Os limiares testados aqui são os defaults de `Settings` (config.py), não números
# reinventados no teste: se `psi_atencao`/`psi_critico` mudarem em config.py sem que
# ninguém atualize este arquivo, este teste é o que denuncia a divergência.
_ATENCAO = Settings().psi_atencao
_CRITICO = Settings().psi_critico


# --- classificar: a fronteira exata das duas bandas -------------------------------------


def test_classificar_abaixo_do_limiar_de_atencao_e_estavel():
    assert classificar(_ATENCAO - 0.01) is Severidade.ESTAVEL


def test_classificar_no_limiar_de_atencao_e_atencao():
    # Fronteira inclusiva: `psi == psi_atencao` já conta como ATENCAO, não ESTAVEL — a
    # mesma convenção que `drift/statistics.py` documenta ("0,10-0,25 mudança moderada").
    # Uma comparação `>` no lugar de `>=` deixaria passar batido exatamente o valor que a
    # convenção do setor usa como corte.
    assert classificar(_ATENCAO) is Severidade.ATENCAO


def test_classificar_entre_os_limiares_e_atencao():
    assert classificar((_ATENCAO + _CRITICO) / 2) is Severidade.ATENCAO


def test_classificar_no_limiar_critico_e_critico():
    # Mesma fronteira inclusiva do lado de cima: `psi == psi_critico` já é CRITICO, não
    # ATENCAO.
    assert classificar(_CRITICO) is Severidade.CRITICO


def test_classificar_acima_do_critico_e_critico():
    assert classificar(_CRITICO + 0.10) is Severidade.CRITICO


# --- DriftReport.severidade_maxima -------------------------------------------------------


def test_severidade_maxima_de_relatorio_vazio_e_estavel():
    # Decidido, não acidental: nenhuma feature medida não é evidência de deslocamento, e o
    # piso seguro é "nada indica drift" — não uma exceção por iterar uma sequência vazia.
    relatorio = DriftReport(lote="mes_00", features=())

    assert relatorio.severidade_maxima is Severidade.ESTAVEL


def test_severidade_maxima_e_a_pior_entre_as_features():
    # A lista termina em ESTAVEL de propósito: uma implementação que devolvesse a
    # severidade da última feature (em vez do máximo de verdade) passaria despercebida se
    # o caso mais grave estivesse por último.
    relatorio = DriftReport(
        lote="mes_06",
        features=(
            DriftDeFeature(
                feature="DebtRatio",
                psi_divergencia=0.50,
                ks_p_valor=0.0,
                severidade=Severidade.CRITICO,
            ),
            DriftDeFeature(
                feature="MonthlyIncome",
                psi_divergencia=0.15,
                ks_p_valor=0.01,
                severidade=Severidade.ATENCAO,
            ),
            DriftDeFeature(
                feature="age", psi_divergencia=0.01, ks_p_valor=0.9, severidade=Severidade.ESTAVEL
            ),
        ),
    )

    assert relatorio.severidade_maxima is Severidade.CRITICO


def test_severidade_maxima_com_todas_estaveis_e_estavel():
    relatorio = DriftReport(
        lote="mes_01",
        features=(
            DriftDeFeature(
                feature="age", psi_divergencia=0.01, ks_p_valor=0.9, severidade=Severidade.ESTAVEL
            ),
        ),
    )

    assert relatorio.severidade_maxima is Severidade.ESTAVEL


# --- DriftReport.features_em_drift -------------------------------------------------------


def test_features_em_drift_filtra_as_estaveis():
    debt_ratio = DriftDeFeature(
        feature="DebtRatio", psi_divergencia=0.50, ks_p_valor=0.0, severidade=Severidade.CRITICO
    )
    renda = DriftDeFeature(
        feature="MonthlyIncome",
        psi_divergencia=0.15,
        ks_p_valor=0.01,
        severidade=Severidade.ATENCAO,
    )
    idade = DriftDeFeature(
        feature="age", psi_divergencia=0.01, ks_p_valor=0.9, severidade=Severidade.ESTAVEL
    )
    relatorio = DriftReport(lote="mes_06", features=(debt_ratio, renda, idade))

    assert relatorio.features_em_drift == (debt_ratio, renda)


def test_features_em_drift_vazio_quando_tudo_estavel():
    relatorio = DriftReport(
        lote="mes_00",
        features=(
            DriftDeFeature(
                feature="age", psi_divergencia=0.0, ks_p_valor=1.0, severidade=Severidade.ESTAVEL
            ),
        ),
    )

    assert relatorio.features_em_drift == ()


# --- DriftDeFeature: os dois campos numéricos não podem se confundir ---------------------


def test_driftdefeature_guarda_psi_e_ks_p_valor_por_nome_nao_por_posicao():
    # Réplica em miniatura da razão de `ResultadoKS` ser NamedTuple: um chamador que
    # trocasse os dois argumentos posicionalmente receberia um objeto que não crasha em
    # lugar nenhum, só mede errado. Nomear os parâmetros na construção é a defesa; este
    # teste fixa que os nomes existem e guardam o valor certo.
    feature = DriftDeFeature(
        feature="MonthlyIncome",
        psi_divergencia=0.30,
        ks_p_valor=0.02,
        severidade=Severidade.CRITICO,
    )

    assert feature.psi_divergencia == 0.30
    assert feature.ks_p_valor == 0.02
