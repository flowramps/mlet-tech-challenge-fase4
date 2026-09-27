"""A fronteira do drift existe para que a classificação de severidade seja declarada uma
vez e a biblioteca que calcula PSI e KS fique atrás da interface — e para que "drift" nunca
vire um jeito de derrubar o run, a distinção que este módulo preserva da Etapa 1."""

from __future__ import annotations

import pytest

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


def test_relatorio_sem_features_e_rejeitado_na_construcao():
    # Ruling do revisor: um relatório com zero features não é um estado plausível — é
    # sinal de detector que travou, foi mal configurado, ou perdeu toda feature no
    # caminho. Devolver ESTAVEL nesse caso (a implementação original, por analogia mal
    # aplicada a `ValidationResult.valido`) deixaria esse bug se disfarçar de "tudo
    # saudável" — o único jeito de falhar que uma fronteira de drift não pode ter.
    # Rejeitar na construção dissolve a pergunta em vez de arriscar respondê-la errado.
    with pytest.raises(ValueError, match="nenhuma feature"):
        DriftReport(lote="mes_00", features=())


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
    # lugar nenhum, só mede errado. Nomear os parâmetros na construção evita a troca de
    # *posição*; não evita, por si só, que alguém escreva o valor errado no campo certo
    # (`ks_p_valor=resultado.estatistica`, por exemplo) — essa defesa mais forte é o
    # `__post_init__` testado abaixo. Este teste fixa só a parte que o nome garante: os
    # dois campos guardam exatamente o valor passado com o nome certo.
    feature = DriftDeFeature(
        feature="MonthlyIncome",
        psi_divergencia=0.30,
        ks_p_valor=0.02,
        severidade=Severidade.CRITICO,
    )

    assert feature.psi_divergencia == 0.30
    assert feature.ks_p_valor == 0.02


# --- DriftDeFeature.__post_init__: a dataclass fecha sobre a própria regra ---------------


def test_driftdefeature_rejeita_severidade_que_nao_bate_com_classificar():
    # A lacuna que o Important 1 do revisor apontou: nada, além desta validação, impede
    # que `severidade` e `psi_divergencia` se contradigam. 0,01 classifica como ESTAVEL
    # (bem abaixo de psi_atencao); rotulá-lo CRITICO tem que ser rejeitado na construção,
    # não silenciosamente aceito e propagado para `severidade_maxima`.
    with pytest.raises(ValueError, match="severidade"):
        DriftDeFeature(
            feature="DebtRatio", psi_divergencia=0.01, ks_p_valor=0.9, severidade=Severidade.CRITICO
        )


def test_driftdefeature_aceita_severidade_consistente_com_classificar():
    # As três severidades continuam construtíveis quando bate com `classificar` — a
    # validação não pode reprovar o caso correto, só o inconsistente.
    for psi, severidade in (
        (0.01, Severidade.ESTAVEL),
        (0.15, Severidade.ATENCAO),
        (0.50, Severidade.CRITICO),
    ):
        DriftDeFeature(feature="x", psi_divergencia=psi, ks_p_valor=0.5, severidade=severidade)


def test_driftdefeature_rejeita_ks_p_valor_acima_de_um():
    # Um p-valor > 1 é impossível por definição — aceitar um é aceitar um bug já na
    # entrada (ver o docstring da classe sobre o que este check pega e o que não pega).
    with pytest.raises(ValueError, match="ks_p_valor"):
        DriftDeFeature(
            feature="age", psi_divergencia=0.01, ks_p_valor=1.5, severidade=Severidade.ESTAVEL
        )


def test_driftdefeature_rejeita_ks_p_valor_negativo():
    with pytest.raises(ValueError, match="ks_p_valor"):
        DriftDeFeature(
            feature="age", psi_divergencia=0.01, ks_p_valor=-0.1, severidade=Severidade.ESTAVEL
        )


def test_driftdefeature_aceita_ks_p_valor_nos_extremos_do_intervalo():
    # 0,0 e 1,0 são valores de p-valor válidos (KS entre distribuições idênticas ou
    # totalmente disjuntas) — o check é `[0, 1]` fechado, não aberto.
    DriftDeFeature(
        feature="age", psi_divergencia=0.01, ks_p_valor=0.0, severidade=Severidade.ESTAVEL
    )
    DriftDeFeature(
        feature="age", psi_divergencia=0.01, ks_p_valor=1.0, severidade=Severidade.ESTAVEL
    )
