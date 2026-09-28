"""Os quatro sinais que um monitor de produção consegue calcular sem rótulo.

`test_sinais_do_lote_funciona_sem_a_coluna_alvo` e
`test_modulo_nao_cita_a_coluna_alvo_em_lugar_nenhum` são o par que prova a garantia
central do módulo (ver `credito.monitoring.proxies`): o primeiro prova em tempo de
execução que um lote sem `ALVO` não quebra nada; o segundo prova estaticamente que nenhuma
função do módulo sequer importou ou citou o nome da coluna — as duas provas juntas cobrem
tanto "não precisou" quanto "não teria como", que uma prova sozinha não cobriria.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from credito.monitoring import proxies
from credito.monitoring.proxies import (
    confianca_media,
    psi_do_score,
    sinais_do_lote,
    taxa_de_aprovacao,
)
from credito.schema import ALVO, FEATURES

SEED = 11


class _ModeloFixo:
    """Devolve a probabilidade que lhe mandarem — isola os sinais do treino real, o
    mesmo papel que `_ModeloFixo` já cumpre em `tests/model/test_evaluate.py`.

    Confere que só recebe `FEATURES`, na ordem de `FEATURES` — nem a mais (a coluna de
    rótulo incluída por engano), nem a menos. Um modelo real treinado sobre `FEATURES`
    reage a isso (scikit-learn recusa nomes de coluna que não batem com o treino); um
    "fake" que apenas contasse linhas, como uma primeira versão deste fixture fazia,
    aceitaria `lote` inteiro sem reclamar e nenhum teste notaria se `_probabilidades`
    parasse de filtrar as colunas — verificado removendo o filtro em
    `credito.monitoring.proxies._probabilidades` e vendo a suíte inteira continuar verde.
    """

    def __init__(self, probabilidades: np.ndarray) -> None:
        self._probabilidades = np.asarray(probabilidades, dtype=float)

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        assert list(entradas.columns) == list(
            FEATURES
        ), f"esperava receber exatamente FEATURES, recebeu {list(entradas.columns)!r}"
        recorte = self._probabilidades[: len(entradas)]
        return np.column_stack([1 - recorte, recorte])


def _lote(n: int, *, com_alvo: bool = True) -> pd.DataFrame:
    dados = {coluna: [0.5] * n for coluna in FEATURES}
    if com_alvo:
        dados[ALVO] = [0] * n
    return pd.DataFrame(dados)


# --- psi_do_score: reuso de credito.drift.statistics.psi, não reimplementação ----------


def test_psi_do_score_distribuicoes_iguais_e_proximo_de_zero():
    gerador = np.random.default_rng(SEED)
    referencia = gerador.uniform(0.0, 1.0, 3000)

    assert psi_do_score(referencia, referencia.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_do_score_desloca_com_a_distribuicao():
    gerador = np.random.default_rng(SEED)
    referencia = gerador.beta(2, 8, 3000)  # score concentrado perto de zero
    atual = gerador.beta(8, 2, 3000)  # concentrado perto de um — deslocamento forte

    assert psi_do_score(referencia, atual) > 0.25


def test_psi_do_score_bate_bit_a_bit_com_psi_direto():
    # A garantia de reuso: chamar psi_do_score não pode ser uma segunda implementação da
    # mesma conta. Se alguém reescrever a fórmula por dentro em vez de importar `psi`,
    # este teste é o único que nota — os dois caminhos deixam de bater exatamente.
    gerador = np.random.default_rng(SEED)
    referencia = gerador.normal(0.4, 0.2, 2000)
    atual = gerador.normal(0.5, 0.2, 2000)

    from credito.drift.statistics import psi

    assert psi_do_score(referencia, atual) == psi(referencia, atual)


def test_psi_do_score_repassa_bins_para_psi():
    # `bins` precisa mesmo chegar até `psi()`, não só existir na assinatura: com poucos
    # bins o histograma fica mais grosso e o PSI medido muda de valor. Comparar contra a
    # chamada direta a `psi(..., bins=3)` é o que pega uma implementação que aceita
    # `bins` mas ignora o argumento por dentro (ex.: sempre delega com o default 10).
    gerador = np.random.default_rng(SEED)
    referencia = gerador.normal(0.4, 0.2, 2000)
    atual = gerador.normal(0.5, 0.2, 2000)

    from credito.drift.statistics import psi

    assert psi_do_score(referencia, atual, bins=3) == psi(referencia, atual, bins=3)
    assert psi_do_score(referencia, atual, bins=3) != psi_do_score(referencia, atual, bins=10)


def test_psi_do_score_score_zero_inflado_usa_o_mesmo_roteamento_por_colapso():
    # Um score bem calibrado sobre uma população majoritariamente de baixo risco se
    # concentra perto de zero — a mesma forma que `bins_por_quantil` colapsa em
    # `NumberOfTime30-59DaysPastDueNotWorse` (ver `drift/statistics.py`). Reusar `psi`
    # (em vez de reimplementar) é o que garante que o score herda esse mesmo roteamento
    # por colapso medido, sem que este módulo precise saber que ele existe.
    gerador = np.random.default_rng(SEED)
    referencia = np.concatenate([np.full(9000, 0.01), gerador.uniform(0.02, 0.3, 1000)])
    atual = np.concatenate([np.full(7000, 0.01), gerador.uniform(0.02, 0.3, 3000)])

    resultado = psi_do_score(referencia, atual)

    assert np.isfinite(resultado)
    assert resultado > 0.10


# --- confianca_media: distância média até a fronteira de decisão 0,5 -------------------


def test_confianca_media_scores_extremos_e_alta():
    # max(p, 1-p) para cada um: 0,99 / 0,98 / 0,98 / 0,99 — média 0,985.
    probabilidades = np.array([0.01, 0.02, 0.98, 0.99])

    assert confianca_media(probabilidades) == pytest.approx(0.985)


def test_confianca_media_scores_na_fronteira_e_minima():
    probabilidades = np.array([0.5, 0.5, 0.5])

    assert confianca_media(probabilidades) == pytest.approx(0.5)


def test_confianca_media_e_simetrica_entre_probabilidade_e_complementar():
    # Mutação que este teste pega: usar `probabilidades.mean()` em vez de
    # `max(p, 1-p).mean()` deixaria de ser simétrico — 0,1 e 0,9 são a mesma confiança
    # (a mesma distância da fronteira), mas médias diretas dão 0,1 e 0,9.
    baixo = confianca_media(np.array([0.1, 0.1, 0.1]))
    alto = confianca_media(np.array([0.9, 0.9, 0.9]))

    assert baixo == pytest.approx(alto)
    assert baixo == pytest.approx(0.9)


def test_confianca_media_lote_vazio_leva_a_erro():
    # Uma média sobre zero linhas não é um número, é um `nan` disfarçado de medição —
    # melhor a falha explícita do que um sinal silenciosamente inválido no relatório.
    with pytest.raises(ValueError):
        confianca_media(np.array([]))


# --- taxa_de_aprovacao: a decisão agregada, na mesma convenção de model.evaluate -------


def test_taxa_de_aprovacao_conta_fracao_abaixo_do_limiar():
    probabilidades = np.array([0.1, 0.2, 0.6, 0.9])

    assert taxa_de_aprovacao(probabilidades, limiar=0.5) == pytest.approx(0.5)


def test_taxa_de_aprovacao_limiar_desloca_a_taxa():
    # Mutação que este teste pega: ignorar `limiar` e comparar sempre contra 0,5.
    probabilidades = np.array([0.2, 0.4, 0.6])

    assert taxa_de_aprovacao(probabilidades, limiar=0.5) == pytest.approx(2 / 3)
    assert taxa_de_aprovacao(probabilidades, limiar=0.3) == pytest.approx(1 / 3)


def test_taxa_de_aprovacao_todo_mundo_abaixo_do_limiar_e_um():
    assert taxa_de_aprovacao(np.array([0.01, 0.02, 0.03]), limiar=0.5) == pytest.approx(1.0)


def test_taxa_de_aprovacao_todo_mundo_acima_do_limiar_e_zero():
    assert taxa_de_aprovacao(np.array([0.9, 0.95, 0.99]), limiar=0.5) == pytest.approx(0.0)


def test_taxa_de_aprovacao_lote_vazio_leva_a_erro():
    with pytest.raises(ValueError):
        taxa_de_aprovacao(np.array([]), limiar=0.5)


# --- sinais_do_lote: a composição, e a garantia de não precisar do rótulo --------------


def test_sinais_do_lote_traz_confianca_e_taxa_de_aprovacao():
    lote = _lote(4)
    modelo = _ModeloFixo(np.array([0.1, 0.2, 0.8, 0.9]))

    sinais = sinais_do_lote(modelo, lote)

    assert sinais["confianca_media"] == pytest.approx(
        confianca_media(np.array([0.1, 0.2, 0.8, 0.9]))
    )
    assert sinais["taxa_de_aprovacao"] == pytest.approx(0.5)


def test_sinais_do_lote_funciona_sem_a_coluna_alvo():
    lote = _lote(3, com_alvo=False)
    assert ALVO not in lote.columns
    modelo = _ModeloFixo(np.array([0.2, 0.5, 0.8]))

    sinais = sinais_do_lote(modelo, lote)

    assert set(sinais) == {"confianca_media", "taxa_de_aprovacao"}


def test_sinais_do_lote_usa_o_limiar_informado():
    lote = _lote(3, com_alvo=False)
    modelo = _ModeloFixo(np.array([0.2, 0.4, 0.6]))

    sinais_padrao = sinais_do_lote(modelo, lote)
    sinais_com_limiar_baixo = sinais_do_lote(modelo, lote, limiar=0.1)

    # limiar padrão 0,5: 0,2 e 0,4 aprovam, 0,6 não — 2/3. Com limiar 0,1, ninguém aprova.
    assert sinais_padrao["taxa_de_aprovacao"] == pytest.approx(2 / 3)
    assert sinais_com_limiar_baixo["taxa_de_aprovacao"] == pytest.approx(0.0)


class _ModeloPorFeature:
    """Devolve o valor de uma feature diretamente como probabilidade — ao contrário de
    `_ModeloFixo`, que ignora o conteúdo do frame e só olha o número de linhas.

    `_ModeloFixo` não serviria para o teste abaixo: com `referencia` e `atual` do mesmo
    tamanho, `_ModeloFixo` devolveria o EXATO mesmo array fatiado para as duas chamadas, e
    um `sinais_do_lote` que comparasse a distribuição consigo mesma por engano (em vez de
    contra `referencia`) passaria despercebido — verificado: essa mutação (trocar
    `probabilidades_referencia` por `probabilidades` na chamada a `psi_do_score` dentro de
    `sinais_do_lote`) não derruba nenhum teste com `_ModeloFixo`. Este modelo reage ao
    conteúdo de cada frame, então `referencia` e `atual` só produzem scores diferentes se
    `sinais_do_lote` de fato os previu separadamente.
    """

    def __init__(self, coluna: str = "RevolvingUtilizationOfUnsecuredLines") -> None:
        self._coluna = coluna

    def predict_proba(self, entradas: pd.DataFrame) -> np.ndarray:
        p = entradas[self._coluna].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def test_sinais_do_lote_com_referencia_inclui_psi_do_score():
    coluna = "RevolvingUtilizationOfUnsecuredLines"
    gerador = np.random.default_rng(SEED)
    referencia = _lote(2000, com_alvo=False)
    referencia[coluna] = gerador.beta(1.5, 6, 2000)  # score baixo, cauda assimétrica
    atual = _lote(2000, com_alvo=False)
    atual[coluna] = gerador.beta(6, 1.5, 2000)  # score alto, cauda assimétrica no espelho
    modelo = _ModeloPorFeature(coluna)

    sinais = sinais_do_lote(modelo, atual, referencia=referencia, limiar=0.5)

    from credito.drift.statistics import psi

    esperado = psi(referencia[coluna].to_numpy(), atual[coluna].to_numpy())
    invertido = psi(atual[coluna].to_numpy(), referencia[coluna].to_numpy())
    # As duas betas espelhadas (assimétricas, não uma reflexão exata uma da outra no
    # sentido que o binning por quantil enxerga) garantem que psi(ref, atual) e
    # psi(atual, ref) não batem — condição necessária para que a igualdade exata abaixo
    # também pegue os argumentos trocados, não só a autocomparação.
    assert esperado != pytest.approx(invertido)
    # Igualdade exata (não só "presente e finito"): prova que `sinais_do_lote` de fato
    # previu `referencia` e `atual` separadamente e passou os dois scores, na ordem
    # certa, para `psi_do_score` — pega tanto autocomparação quanto argumentos trocados.
    assert sinais["psi_do_score"] == pytest.approx(esperado)
    assert sinais["psi_do_score"] > 0.25


def test_sinais_do_lote_sem_referencia_nao_inclui_psi_do_score():
    lote = _lote(3, com_alvo=False)
    modelo = _ModeloFixo(np.array([0.2, 0.4, 0.6]))

    sinais = sinais_do_lote(modelo, lote)

    assert "psi_do_score" not in sinais


def test_sinais_do_lote_nao_le_features_extras_como_o_alvo():
    # Um lote que carrega ALVO precisa dar o mesmo resultado de um lote sem ALVO — a
    # prova, em execução, de que a coluna é ignorada, não apenas ausente por acaso.
    modelo = _ModeloFixo(np.array([0.3, 0.6, 0.9]))
    com_alvo = _lote(3, com_alvo=True)
    com_alvo[ALVO] = [1, 0, 1]  # rótulos que, se lidos, mudariam qualquer sinal agregado
    sem_alvo = com_alvo.drop(columns=[ALVO])

    assert sinais_do_lote(modelo, com_alvo) == sinais_do_lote(modelo, sem_alvo)


def test_modulo_nao_cita_a_coluna_alvo_em_lugar_nenhum():
    # A garantia mecânica que o brief pede: nenhuma função deste módulo lê, importa ou
    # sequer menciona a coluna de rótulo — nem pelo nome da constante, nem pelo valor
    # literal que ela carrega.
    codigo_fonte = inspect.getsource(proxies)

    assert "ALVO" not in codigo_fonte
    assert ALVO not in codigo_fonte
