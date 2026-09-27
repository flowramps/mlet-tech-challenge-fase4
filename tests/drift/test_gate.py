"""O gate consolida os `DriftReport` de vários lotes numa única decisão — e a decisão
nunca é falha: drift alerta, quem decide retreinar é uma pessoa (ver o docstring de
`credito.drift.gate`). Todo p-valor de KS que entra num teste aqui passa por
Benjamini-Hochberg antes de a severidade valer alguma coisa — é essa ordem, não a
convenção de PSI sozinha, que os testes de "correção derruba falso positivo" e "correção
preserva sinal forte" verificam.
"""

from __future__ import annotations

import re

import pytest

from credito.drift.base import DriftDeFeature, DriftReport, Severidade, classificar
from credito.drift.gate import CruzamentoDeFeature, GateDeDrift, avaliar_gate

# --- construção de fixtures: sempre via classificar(), nunca severidade digitada à mão --
# (a mesma regra que DriftDeFeature.__post_init__ já impõe — ver drift/base.py)


def _feature(nome: str, psi: float, ks_p_valor: float) -> DriftDeFeature:
    return DriftDeFeature(
        feature=nome, psi_divergencia=psi, ks_p_valor=ks_p_valor, severidade=classificar(psi)
    )


def _lote(nome: str, *features: DriftDeFeature) -> DriftReport:
    return DriftReport(lote=nome, features=features)


# --- avaliar_gate: entrada vazia é bug do chamador, não "lote sem drift" ----------------


def test_avaliar_gate_rejeita_lista_vazia():
    with pytest.raises(ValueError, match="DriftReport"):
        avaliar_gate([])


# --- o desfecho de três bandas ------------------------------------------------------------


def test_gate_estavel_quando_nenhuma_feature_cruza_atencao():
    relatorios = [
        _lote("mes_00", _feature("age", 0.01, 0.90), _feature("DebtRatio", 0.02, 0.80)),
        _lote("mes_03", _feature("age", 0.03, 0.70), _feature("DebtRatio", 0.04, 0.60)),
    ]

    gate = avaliar_gate(relatorios)

    assert gate.severidade is Severidade.ESTAVEL
    assert gate.recomenda_retreino is False


def test_gate_alerta_quando_alguma_feature_fica_entre_bandas():
    relatorios = [
        _lote("mes_00", _feature("age", 0.01, 0.90)),
        _lote("mes_03", _feature("age", 0.15, 0.001)),
    ]

    gate = avaliar_gate(relatorios)

    assert gate.severidade is Severidade.ATENCAO
    assert gate.recomenda_retreino is False


def test_gate_alerta_critico_quando_alguma_feature_atinge_critico():
    relatorios = [
        _lote("mes_00", _feature("DebtRatio", 0.02, 0.90)),
        _lote("mes_06", _feature("DebtRatio", 0.50, 0.0001)),
    ]

    gate = avaliar_gate(relatorios)

    assert gate.severidade is Severidade.CRITICO
    assert gate.recomenda_retreino is True


def test_gate_nunca_ergue_excecao_mesmo_com_severidade_critica():
    # A distinção herdada da Etapa 1: drift nunca é uma falha de pipeline (isso é trabalho
    # do contrato, que bloqueia mais cedo). Mesmo a pior severidade tem que voltar como um
    # valor lido, não como uma exceção.
    relatorios = [_lote("mes_06", _feature("DebtRatio", 0.90, 0.0))]

    gate = avaliar_gate(relatorios)  # não deve levantar

    assert gate.severidade is Severidade.CRITICO


# --- severidade do gate é o MÁXIMO de toda a janela, não só do último lote --------------


def test_severidade_do_gate_considera_toda_a_janela_nao_so_o_ultimo_lote():
    # DebtRatio ficou crítico no mes_01 e SE RECUPEROU até o mes_03 — o veredito do gate
    # ainda precisa acusar que a janela teve um episódio crítico. Reportar só o estado do
    # último lote esconderia esse episódio.
    relatorios = [
        _lote("mes_00", _feature("DebtRatio", 0.02, 0.90)),
        _lote("mes_01", _feature("DebtRatio", 0.50, 0.0001)),
        _lote("mes_03", _feature("DebtRatio", 0.02, 0.90)),
    ]

    gate = avaliar_gate(relatorios)

    assert gate.severidade is Severidade.CRITICO


# --- Benjamini-Hochberg efetivamente muda o veredito, não só aparece no relatório -------


def _familia_com_um_pico_isolado(p_valor_do_pico: float) -> list[DriftReport]:
    """Quatro lotes de três features cada: dentro do lote `mes_00`, duas combinações
    estáveis e sem sinal algum (p-valor alto) mais uma com PSI crítico cujo p-valor é o
    parâmetro do teste. A correção roda por lote (ver `drift/gate.py`), então a família
    relevante para o pico é só o seu próprio lote (m=3), não os quatro lotes juntos."""
    relatorios = []
    for indice_lote in range(4):
        nome_lote = f"mes_{indice_lote:02d}"
        if indice_lote == 0:
            pico = _feature("DebtRatio", 0.50, p_valor_do_pico)
        else:
            pico = _feature("DebtRatio", 0.01, 0.90)
        ruido_a = _feature("age", 0.01, 0.5 + indice_lote * 0.01)
        ruido_b = _feature("MonthlyIncome", 0.01, 0.6 + indice_lote * 0.01)
        relatorios.append(_lote(nome_lote, pico, ruido_a, ruido_b))
    return relatorios


def test_bh_derruba_severidade_de_pico_isolado_que_nao_sobrevive_a_correcao():
    # p=0,04 é "significativo" pelo teste de KS não corrigido (0,04 < 0,05) — é
    # exatamente o falso positivo que a correção existe para pegar quando há outras duas
    # comparações no mesmo lote. Sem a correção (rank por rank, alfa=0,05 aplicado a cada
    # p-valor isoladamente) esta feature ficaria CRITICO; com BH sobre a família do
    # PRÓPRIO lote (m=3: DebtRatio 0,04, age ~0,5, MonthlyIncome ~0,6), o rank 1 (o
    # próprio p=0,04, o menor do lote) exige p <= 1/3 * 0,05 = 0,016667 para ser
    # rejeitado, e 0,04 fica acima disso — nada é rejeitado neste lote.
    relatorios = _familia_com_um_pico_isolado(0.04)

    gate = avaliar_gate(relatorios)

    debt_ratio = next(c for c in gate.cruzamentos if c.feature == "DebtRatio")
    assert debt_ratio.primeiro_lote_critico is None
    assert debt_ratio.primeiro_lote_atencao is None
    assert debt_ratio.severidade_final is Severidade.ESTAVEL
    assert gate.severidade is Severidade.ESTAVEL


def test_bh_preserva_sinal_forte_mesmo_apos_correcao():
    # Mesmo lote de três features, mas o pico tem p-valor minúsculo (1e-8): mesmo com o
    # limiar do próprio lote (rank 1: p <= 0,016667, m=3), 1e-8 passa com folga. A
    # correção reduz poder, não elimina sinal genuíno.
    relatorios = _familia_com_um_pico_isolado(1e-8)

    gate = avaliar_gate(relatorios)

    debt_ratio = next(c for c in gate.cruzamentos if c.feature == "DebtRatio")
    assert debt_ratio.primeiro_lote_critico == "mes_00"
    assert gate.severidade is Severidade.CRITICO


def test_correcao_e_por_lote_nao_pela_janela_inteira():
    """A propriedade que a revisão exigiu tornar explícita: a família de
    Benjamini-Hochberg é o LOTE, não a janela inteira que o gate recebe — decisão
    inferencial, não só de forma. Um monitor real decide um lote de cada vez, e a
    decisão sobre o lote 1 só pode usar o que existia quando o lote 1 chegou.

    `DebtRatio` no `mes_00` tem PSI crítico (0,50) e p=0,04. Sozinho no seu próprio lote
    (m=1, nenhuma outra feature medida nesse lote), o limiar do único rank é o próprio
    alfa (0,05) — 0,04 <= 0,05 é rejeitado sem ressalva, e a severidade permanece
    CRITICO. Cinco lotes SEGUINTES, cada um com dez features estáveis e sem sinal
    (p-valor alto), não têm nenhum efeito sobre essa decisão, porque a correção do
    `mes_00` nunca vê os p-valores de lotes que ainda não existiam quando `mes_00` foi
    classificado.

    Sob correção AGRUPADA (pooled — a implementação anterior a esta revisão), esses
    mesmos cinquenta p-valores de ruído entrariam na família do `mes_00`, o `m` subiria
    de 1 para 51, o limiar do rank 1 encolheria de 0,05 para 1/51*0,05 ≈ 0,00098, e
    0,04 > 0,00098 rebaixaria `DebtRatio` para ESTAVEL — usando informação de cinco
    lotes que, na linha do tempo real, ainda não tinham acontecido. Este teste fixa a
    implementação correta (por lote); reverter `avaliar_gate` para a correção agrupada faz
    este teste ficar vermelho, que é o que o torna discriminante.
    """
    lote_com_sinal = _lote("mes_00", _feature("DebtRatio", 0.50, 0.04))
    lotes_de_ruido = [
        _lote(
            f"mes_{indice:02d}",
            *[
                _feature(f"ruido_{indice}_{posicao}", 0.01, 0.5 + posicao * 0.01)
                for posicao in range(10)
            ],
        )
        for indice in range(1, 6)
    ]

    gate = avaliar_gate([lote_com_sinal, *lotes_de_ruido])

    debt_ratio = next(c for c in gate.cruzamentos if c.feature == "DebtRatio")
    assert debt_ratio.severidade_final is Severidade.CRITICO
    assert debt_ratio.primeiro_lote_critico == "mes_00"
    assert gate.severidade is Severidade.CRITICO


def test_severidade_efetiva_nunca_promove_acima_do_que_o_psi_sustenta():
    # Um p-valor de KS minúsculo não pode, sozinho, promover uma feature cujo PSI é
    # baixo — severidade nasce de `classificar(psi)` (`drift/base.py`) e a correção só pode
    # rebaixar, nunca elevar. Família com sinal forte de KS (p=1e-9) só em `age`, cujo
    # PSI é 0,01 (ESTAVEL por construção).
    relatorios = [
        _lote(
            "mes_00",
            _feature("age", 0.01, 1e-9),
            _feature("DebtRatio", 0.01, 0.9),
            _feature("MonthlyIncome", 0.01, 0.8),
        ),
    ]

    gate = avaliar_gate(relatorios)

    assert gate.severidade is Severidade.ESTAVEL
    idade = next(c for c in gate.cruzamentos if c.feature == "age")
    assert idade.primeiro_lote_atencao is None


def test_alfa_default_do_gate_e_005():
    # Réplica, um nível acima, do mesmo teste de `benjamini_hochberg`: com m=1, o limiar
    # do único rank é o próprio alfa. p=0,07 fica exatamente entre 0,05 (default) e um
    # alfa mais frouxo — discrimina os dois.
    relatorios = [_lote("mes_00", _feature("DebtRatio", 0.50, 0.07))]

    com_default = avaliar_gate(relatorios)
    com_alfa_explicito_005 = avaliar_gate(relatorios, alfa=0.05)
    com_alfa_frouxo = avaliar_gate(relatorios, alfa=0.10)

    assert com_default.severidade is com_alfa_explicito_005.severidade is Severidade.ESTAVEL
    assert com_alfa_frouxo.severidade is Severidade.CRITICO


# --- cruzamentos: o mês em que cada feature cruzou cada limiar --------------------------


def test_primeiro_cruzamento_de_atencao_e_o_primeiro_no_tempo():
    relatorios = [
        _lote("mes_00", _feature("MonthlyIncome", 0.01, 0.9)),
        _lote("mes_03", _feature("MonthlyIncome", 0.15, 0.001)),
        _lote("mes_06", _feature("MonthlyIncome", 0.20, 0.0001)),
    ]

    gate = avaliar_gate(relatorios)

    renda = next(c for c in gate.cruzamentos if c.feature == "MonthlyIncome")
    assert renda.primeiro_lote_atencao == "mes_03"
    assert renda.primeiro_lote_critico is None


def test_primeiro_lote_critico_pode_ser_posterior_ao_primeiro_de_atencao():
    relatorios = [
        _lote("mes_00", _feature("DebtRatio", 0.01, 0.9)),
        _lote("mes_02", _feature("DebtRatio", 0.15, 0.001)),
        _lote("mes_05", _feature("DebtRatio", 0.30, 0.0001)),
    ]

    gate = avaliar_gate(relatorios)

    divida = next(c for c in gate.cruzamentos if c.feature == "DebtRatio")
    assert divida.primeiro_lote_atencao == "mes_02"
    assert divida.primeiro_lote_critico == "mes_05"


def test_cruza_recupera_e_cruza_de_novo_reporta_o_primeiro_cruzamento_nao_o_ultimo():
    # A decisão explícita que este módulo toma: cruzar, recuperar e cruzar de novo NÃO
    # empurra `primeiro_lote_atencao` para o segundo cruzamento — o campo é sobre
    # QUANDO a feature entrou na banda pela primeira vez, não sobre o estado mais
    # recente (que é o que `severidade_final` responde).
    relatorios = [
        _lote("mes_00", _feature("NumberOfTimes90DaysLate", 0.15, 0.001)),  # cruza
        _lote("mes_02", _feature("NumberOfTimes90DaysLate", 0.02, 0.9)),  # recupera
        _lote("mes_05", _feature("NumberOfTimes90DaysLate", 0.18, 0.0005)),  # cruza de novo
    ]

    gate = avaliar_gate(relatorios)

    atraso = next(c for c in gate.cruzamentos if c.feature == "NumberOfTimes90DaysLate")
    assert atraso.primeiro_lote_atencao == "mes_00"
    assert atraso.severidade_final is Severidade.ATENCAO


def test_cruza_e_recupera_ate_o_fim_severidade_final_reflete_a_recuperacao():
    relatorios = [
        _lote("mes_00", _feature("NumberOfTimes90DaysLate", 0.15, 0.001)),  # cruza
        _lote("mes_05", _feature("NumberOfTimes90DaysLate", 0.02, 0.9)),  # recupera e fica
    ]

    gate = avaliar_gate(relatorios)

    atraso = next(c for c in gate.cruzamentos if c.feature == "NumberOfTimes90DaysLate")
    assert atraso.primeiro_lote_atencao == "mes_00"
    assert atraso.severidade_final is Severidade.ESTAVEL


def test_feature_que_nunca_cruza_tem_ambos_os_campos_none():
    relatorios = [_lote("mes_00", _feature("age", 0.01, 0.9))]

    gate = avaliar_gate(relatorios)

    idade = next(c for c in gate.cruzamentos if c.feature == "age")
    assert idade.primeiro_lote_atencao is None
    assert idade.primeiro_lote_critico is None
    assert idade.severidade_final is Severidade.ESTAVEL


def test_cruzamentos_cobrem_todas_as_features_vistas_nao_so_as_em_drift():
    relatorios = [_lote("mes_00", _feature("age", 0.01, 0.9), _feature("DebtRatio", 0.30, 0.0001))]

    gate = avaliar_gate(relatorios)

    assert {c.feature for c in gate.cruzamentos} == {"age", "DebtRatio"}


# --- mensagem / recomenda_retreino: roteados pelo enum, nunca por texto -----------------


@pytest.mark.parametrize(
    ("severidade", "palavra_chave"),
    [
        (Severidade.ESTAVEL, "estável"),
        (Severidade.ATENCAO, "alerta"),
        (Severidade.CRITICO, "crítico"),
    ],
)
def test_mensagem_corresponde_a_severidade(severidade, palavra_chave):
    gate = GateDeDrift(severidade=severidade, alfa=0.05, cruzamentos=())
    assert palavra_chave in gate.mensagem


def test_recomenda_retreino_somente_quando_critico():
    assert GateDeDrift(severidade=Severidade.CRITICO, alfa=0.05, cruzamentos=()).recomenda_retreino
    assert not GateDeDrift(
        severidade=Severidade.ATENCAO, alfa=0.05, cruzamentos=()
    ).recomenda_retreino
    assert not GateDeDrift(
        severidade=Severidade.ESTAVEL, alfa=0.05, cruzamentos=()
    ).recomenda_retreino


# --- resumo: a narrativa que transforma a tabela em texto, sem inverter o alarme --------


def test_resumo_inclui_a_mensagem_do_veredito():
    gate = GateDeDrift(severidade=Severidade.CRITICO, alfa=0.05, cruzamentos=())
    assert gate.mensagem in gate.resumo


def test_resumo_lista_feature_que_cruzou_e_omite_a_que_nunca_cruzou():
    relatorios = [
        _lote(
            "mes_03",
            _feature("MonthlyIncome", 0.15, 0.001),
            _feature("age", 0.01, 0.9),
        )
    ]

    gate = avaliar_gate(relatorios)

    assert "MonthlyIncome" in gate.resumo
    assert "age" not in gate.resumo


def test_resumo_diz_nunca_ficou_critica_quando_so_atencao():
    relatorios = [_lote("mes_03", _feature("MonthlyIncome", 0.15, 0.001))]
    gate = avaliar_gate(relatorios)
    assert "nunca ficou crítica" in gate.resumo


def test_resumo_diz_tornou_se_critica_quando_atinge_critico():
    relatorios = [_lote("mes_03", _feature("DebtRatio", 0.30, 0.0001))]
    gate = avaliar_gate(relatorios)
    assert "tornou-se crítica" in gate.resumo


def test_resumo_diz_recuperou_quando_volta_a_estavel_depois_de_atencao():
    relatorios = [
        _lote("mes_00", _feature("MonthlyIncome", 0.15, 0.001)),
        _lote("mes_03", _feature("MonthlyIncome", 0.01, 0.9)),
    ]
    gate = avaliar_gate(relatorios)
    assert "recuperou" in gate.resumo


def test_resumo_traz_a_nota_diagnostica_sem_citar_numeros_do_campeao():
    # A armadilha que este texto existe para evitar: um leitor que visse só a tabela de
    # PSI concluiria que a feature mais barulhenta é o problema mais urgente. Mas o gate
    # não tem acesso ao campeão publicado nem ao dado real (nenhum teste desta suíte
    # toca rede ou dataset) — citar aqui um número específico da atribuição causal seria
    # o gate afirmando uma medição que ele nunca fez, e um retreino do campeão tornaria
    # esse número obsoleto sem que nada nesta suíte notasse. O aviso é qualitativo:
    # aponta para onde a medição real mora, sem repeti-la.
    gate = GateDeDrift(severidade=Severidade.ESTAVEL, alfa=0.05, cruzamentos=())

    resumo = gate.resumo
    assert "diagnóstico" in resumo.lower()
    assert "atribuir_degradacao" in resumo
    assert "degradacao_por_lote" in resumo
    # Nenhum número específico da decomposição causal pode vazar para o texto emitido —
    # regressão explícita: o gate não tem como medir esses números e não pode citá-los.
    # Os quatro literais são os da execução ATUAL publicada no README e no model card
    # (`DebtRatio` PSI 0,5046, `MonthlyIncome` PSI 0,2634, as duas juntas respondendo por
    # 3,1% da degradação e o concept drift por 48,0%). Guardar contra valores já
    # superseded — como "0,268"/"0,503", de uma execução anterior — deixaria a guarda
    # verde justamente quando o vazamento fosse dos números certos.
    assert "3,1%" not in resumo
    assert "48,0%" not in resumo
    assert "0,2634" not in resumo
    assert "0,5046" not in resumo
    # E a guarda que não apodrece na próxima execução: o resumo não contém NENHUM número
    # decimal, em nenhum formato. Os quatro literais acima travam a regressão concreta;
    # esta linha trava a classe inteira, inclusive os números da próxima medição, que
    # ninguém vai lembrar de acrescentar aqui.
    assert re.search(r"\d[.,]\d", resumo) is None


# --- CruzamentoDeFeature é um dataclass simples, sem regra escondida --------------------


def test_cruzamento_de_feature_e_imutavel():
    cruzamento = CruzamentoDeFeature(
        feature="age",
        primeiro_lote_atencao=None,
        primeiro_lote_critico=None,
        severidade_final=Severidade.ESTAVEL,
    )
    with pytest.raises(AttributeError):
        cruzamento.feature = "outra"  # type: ignore[misc]
