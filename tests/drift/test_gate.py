"""O gate consolida os `DriftReport` de vários lotes numa única decisão — e a decisão
nunca é falha: drift alerta, quem decide retreinar é uma pessoa (ver o docstring de
`credito.drift.gate`). Todo p-valor de KS que entra num teste aqui passa por
Benjamini-Hochberg antes de a severidade valer alguma coisa — é essa ordem, não a
convenção de PSI sozinha, que os testes de "correção derruba falso positivo" e "correção
preserva sinal forte" verificam.
"""

from __future__ import annotations

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


# --- o desfecho de três bandas do brief --------------------------------------------------


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
    """Quatro lotes de três features cada (m=12): onze combinações estáveis e sem sinal
    algum (p-valor alto), e uma única combinação com PSI crítico cujo p-valor é o
    parâmetro do teste — para variar só essa peça e observar o efeito da correção."""
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
    # exatamente o falso positivo que a correção existe para pegar quando há outras 11
    # comparações na mesma família. Sem a correção (rank por rank, alfa=0,05 aplicado a
    # cada p-valor isoladamente) esta feature ficaria CRITICO; com BH sobre a família de
    # 12, o rank 1 (o próprio p=0,04, o menor da família) exige p <= 1/12 * 0,05 =
    # 0,004166 para ser rejeitado, e 0,04 fica muito acima disso — nada é rejeitado.
    relatorios = _familia_com_um_pico_isolado(0.04)

    gate = avaliar_gate(relatorios)

    debt_ratio = next(c for c in gate.cruzamentos if c.feature == "DebtRatio")
    assert debt_ratio.primeiro_lote_critico is None
    assert debt_ratio.primeiro_lote_atencao is None
    assert debt_ratio.severidade_final is Severidade.ESTAVEL
    assert gate.severidade is Severidade.ESTAVEL


def test_bh_preserva_sinal_forte_mesmo_apos_correcao():
    # Mesma família de 12, mas o pico tem p-valor minúsculo (1e-8): mesmo com o limiar
    # mais rigoroso da família inteira (rank 1: p <= 0,004166), 1e-8 passa com folga. A
    # correção reduz poder, não elimina sinal genuíno.
    relatorios = _familia_com_um_pico_isolado(1e-8)

    gate = avaliar_gate(relatorios)

    debt_ratio = next(c for c in gate.cruzamentos if c.feature == "DebtRatio")
    assert debt_ratio.primeiro_lote_critico == "mes_00"
    assert gate.severidade is Severidade.CRITICO


def test_severidade_efetiva_nunca_promove_acima_do_que_o_psi_sustenta():
    # Um p-valor de KS minúsculo não pode, sozinho, promover uma feature cujo PSI é
    # baixo — severidade nasce de `classificar(psi)` (Tarefa 4) e a correção só pode
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


def test_resumo_traz_a_nota_de_diagnostico_com_os_numeros_medidos_da_atribuicao_causal():
    # A armadilha que este texto existe para evitar: um leitor que visse só a tabela de
    # PSI concluiria que a feature mais barulhenta é o problema mais urgente. A
    # atribuição causal por intervenção mediu o oposto — as duas features de PSI mais
    # alto do projeto respondem por 3,1% da queda de AUC-ROC; o concept drift, que PSI
    # de feature não enxerga, responde por 48,0% sozinho.
    gate = GateDeDrift(severidade=Severidade.ESTAVEL, alfa=0.05, cruzamentos=())

    assert "3,1%" in gate.resumo
    assert "48,0%" in gate.resumo
    assert "diagnóstico" in gate.resumo.lower()


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
