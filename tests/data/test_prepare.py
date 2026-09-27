"""A Referência é o dado que o modelo aprende. A política de limpeza precisa ser
explícita e auditável: cada linha descartada tem um motivo contabilizado."""

from __future__ import annotations

import pandas as pd

from credito.contracts.pandera_backend import _REGRA_POR_COLUNA, _schema, construir_validador
from credito.data.prepare import limpar, separar
from credito.schema import ALVO, COLUNAS_SEM_REGRA_NOMEADA, FEATURES


def _linha(**ajustes) -> dict:
    base = {
        "FinancialDistressNextTwoYears": "No",
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5400.0,
        "NumberOfOpenCreditLinesAndLoans": 7,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0.0,
    }
    base.update(ajustes)
    return base


def test_alvo_vira_inteiro_binario():
    # age varia entre as duas linhas para que elas não sejam duplicata uma da outra em
    # FEATURES — o que este teste quer isolar é a conversão do alvo, não o dedup.
    frame = pd.DataFrame([_linha(), _linha(age=50, FinancialDistressNextTwoYears="Yes")])

    limpo, _ = limpar(frame)

    assert sorted(limpo[ALVO].unique()) == [0, 1]
    assert limpo[ALVO].dtype.kind == "i"


def test_descarta_renda_nula():
    frame = pd.DataFrame([_linha(), _linha(MonthlyIncome=None)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["renda_nula"] == 1


def test_descarta_menores_de_18():
    frame = pd.DataFrame([_linha(), _linha(age=0), _linha(age=17)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["idade_invalida"] == 2


def test_descarta_duplicatas_exatas():
    frame = pd.DataFrame([_linha(), _linha(), _linha(age=50)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 2
    assert motivos["duplicata"] == 1


def test_descarta_sentinelas_de_atraso():
    # 96 e 98 são códigos de "não disponível" no dataset de origem, não contagens: um
    # cliente com 98 atrasos de 90 dias não existe. São tipo-válidos e semanticamente
    # impossíveis — exatamente o que o contrato precisa interceptar.
    frame = pd.DataFrame(
        [
            _linha(),
            _linha(**{"NumberOfTimes90DaysLate": 98}),
            _linha(**{"NumberOfTime30-59DaysPastDueNotWorse": 96}),
        ]
    )

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["atraso_sentinela"] == 2


def test_descarta_dependentes_nulo():
    frame = pd.DataFrame([_linha(), _linha(NumberOfDependents=None)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["dependentes_nulo"] == 1


def test_descarta_razao_divida_implausivel():
    # Rede de segurança para quando a renda vem preenchida mas o DebtRatio ainda assim é
    # implausível — o mesmo teto que o contrato de ingestão aplica.
    frame = pd.DataFrame([_linha(), _linha(DebtRatio=15.0)])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["razao_divida_implausivel"] == 1


def test_descarta_duplicata_por_features_mesmo_com_alvo_diferente():
    # Duas aplicações idênticas em FEATURES com desfechos diferentes não são duas
    # observações — são um conflito de rótulo. O dedup por FEATURES resolve isso mantendo
    # a primeira ocorrência, a mesma convenção do `Check` de duplicata do contrato.
    frame = pd.DataFrame([_linha(), _linha(FinancialDistressNextTwoYears="Yes")])

    limpo, motivos = limpar(frame)

    assert len(limpo) == 1
    assert motivos["duplicata"] == 1
    assert limpo[ALVO].iloc[0] == 0


def test_referencia_limpa_passa_no_proprio_contrato():
    # A invariante que sustenta o projeto: dado que seria bloqueado na porta não pode ter
    # ensinado o modelo. Não é um teste de escala — roda numa amostra sintética que
    # carrega uma instância de **cada exigência do contrato**, e verifica que o que sobra
    # passa no contrato de ingestão de verdade.
    #
    # A distinção importa: numa versão anterior o frame trazia só os defeitos que
    # `limpar()` já sabia tratar, então ele passava por construção e não podia acusar
    # divergência nenhuma. As linhas marcadas abaixo são justamente as que a limpeza
    # deixava passar e o contrato reprovava — cada uma põe o teste vermelho se o espelho
    # correspondente sumir de `limpar()`.
    frame = pd.DataFrame(
        [
            _linha(),
            _linha(),
            _linha(FinancialDistressNextTwoYears="Yes"),
            _linha(MonthlyIncome=None),
            _linha(NumberOfDependents=None),
            _linha(age=15),
            _linha(age=120),
            _linha(**{"NumberOfTimes90DaysLate": 98}),
            _linha(DebtRatio=15.0),
            _linha(DebtRatio=-1.0),
            # A partir daqui, as seis que a limpeza não espelhava.
            _linha(MonthlyIncome=-1.0),
            _linha(NumberOfDependents=-1.0),
            _linha(**{"NumberOfTimes90DaysLate": -1}),
            _linha(RevolvingUtilizationOfUnsecuredLines=None),
            _linha(NumberOfOpenCreditLinesAndLoans=None),
            _linha(NumberRealEstateLoansOrLines=None),
        ]
    )

    limpo, _ = limpar(frame)

    assert construir_validador().validar(limpo[list(FEATURES)]).valido is True
    # Sobram só as linhas boas: as três primeiras colapsam em uma pelo dedup em FEATURES,
    # e todas as outras são descartes. Sem esta contagem o teste continuaria verde se
    # `limpar()` passasse a descartar o lote inteiro.
    assert len(limpo) == 1


def test_limpeza_cobre_todas_as_colunas_do_contrato():
    # O espelho entre limpeza e contrato é verificado acima caso a caso; este teste cobra
    # a outra metade, estrutural: toda coluna que o contrato exige precisa estar coberta
    # por uma regra nomeada ou pela lista das que só têm exigência estrutural. Uma coluna
    # nova no schema sem entrada em nenhum dos dois lados cai aqui, antes de virar uma
    # divergência que só o arquivo real revelaria.
    do_schema = set(_schema().columns)

    assert do_schema == set(FEATURES)
    assert do_schema == set(_REGRA_POR_COLUNA) | set(COLUNAS_SEM_REGRA_NOMEADA)
    assert not set(_REGRA_POR_COLUNA) & set(COLUNAS_SEM_REGRA_NOMEADA)


def test_referencia_sai_com_as_features_esperadas():
    limpo, _ = limpar(pd.DataFrame([_linha()]))

    assert list(limpo.columns) == [ALVO, *FEATURES]


def test_separacao_e_estratificada_e_reprodutivel():
    # RevolvingUtilizationOfUnsecuredLines varia por grupo e por linha para que nenhuma
    # linha do grupo positivo seja duplicata em FEATURES de uma do grupo negativo — as
    # faixas de age dos dois grupos se sobrepõem (30..49), e sem essa variação o dedup
    # por FEATURES colapsaria as 20 linhas positivas nas 20 negativas de mesma idade.
    frame = pd.DataFrame(
        [_linha(age=30 + i, RevolvingUtilizationOfUnsecuredLines=0.1 + i * 1e-4) for i in range(80)]
        + [
            _linha(
                age=30 + i,
                RevolvingUtilizationOfUnsecuredLines=0.5 + i * 1e-4,
                FinancialDistressNextTwoYears="Yes",
            )
            for i in range(20)
        ]
    )
    limpo, _ = limpar(frame)

    particoes = separar(limpo, test_size=0.2, validation_size=0.2, seed=42)
    de_novo = separar(limpo, test_size=0.2, validation_size=0.2, seed=42)

    assert set(particoes) == {"treino", "validacao", "teste"}
    assert len(particoes["treino"]) + len(particoes["validacao"]) + len(particoes["teste"]) == 100
    # A proporção de positivos precisa sobreviver ao corte: com 20% de positivos, uma
    # partição que perdesse a estratificação poderia sair sem nenhum.
    for nome, parte in particoes.items():
        assert 0.1 <= parte[ALVO].mean() <= 0.3, nome
    assert particoes["teste"].equals(de_novo["teste"])
