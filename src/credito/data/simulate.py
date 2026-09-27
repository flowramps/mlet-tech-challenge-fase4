"""Simulação do Dataset de Produção: a passagem do tempo como uma sequência de lotes.

O enunciado pede "simular a passagem do tempo" — este módulo transforma essa frase em
seis lotes mensais com intensidade progressiva, mais o mês 0 (a própria Referência), para
que um gráfico mostre a degradação se instalando e não apenas dois estados, "antes" e
"depois".

Duas coisas diferentes acontecem, e ficam em funções separadas de propósito:

- **Data drift** muda ``P(X)`` — a distribuição das entradas. É o que
  ``aplicar_drift_de_renda``, ``aplicar_drift_de_divida`` e ``aplicar_drift_de_atraso``
  fazem, cada uma isolada numa única coluna.
- **Concept drift** muda ``P(y|X)`` — a relação entre entrada e desfecho. É o que
  ``aplicar_concept_drift`` faz: em intensidade zero o rótulo é **exatamente** o da
  Referência, linha a linha; acima de zero, uma fração crescente dos rótulos negativos
  de um segmento específico (histórico de atraso limpo, mas alavancagem alta — ver
  ``_regiao_de_risco_emergente``) vira positiva, porque sob inflação esse segmento deixa
  de se comportar como o passado ensinou.

  A primeira versão deste módulo recalculava o rótulo inteiro, em todo mês, inclusive no
  mês 0, a partir de um modelo logístico univariado sobre ``DebtRatio`` calibrado para
  reproduzir a taxa agregada de positivos da Referência (6,94%). Isso preservava a taxa
  **marginal**, não a condicional ``P(y|X)`` — e por isso o campeão, treinado nas dez
  features reais, media AUC-PR 0,0782 contra o mês 0 (medido contra a partição de teste
  real, 23.584 linhas), quando o desempenho histórico é 0,3716: o rótulo sintético não
  tinha relação com o que o campeão aprendeu, mesmo antes de qualquer drift. O desenho
  atual corrige isso na raiz: fora do segmento afetado, e em qualquer segmento quando
  ``intensidade == 0``, o rótulo nunca é tocado.

As quatro funções são expostas separadamente — não só compostas dentro de
``simular_producao`` — porque uma etapa posterior faz atribuição causal por
**intervenção**: aplica o drift a uma variável de cada vez, mantendo as outras fixas, e
mede o quanto cada uma isoladamente degrada o modelo. Isso só é uma medida causal válida
se a intervenção usar exatamente a mesma transformação que gerou os dados simulados — se
a atribuição precisasse reimplementar a lógica, mediria um processo gerador diferente do
que foi de fato simulado, e a análise inteira perderia validade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credito.schema import (
    ALVO,
    ATRASO_MAXIMO_PLAUSIVEL,
    COLUNAS_DE_ATRASO,
    DEBT_RATIO_MAXIMO,
    FEATURES,
)

MESES: int = 6

# O enunciado exige alterar a distribuição de pelo menos duas variáveis importantes.
# Entregamos três, cobrindo os dois tipos de drift que o enunciado também exige.
VARIAVEIS_COM_DRIFT: tuple[str, ...] = (
    "MonthlyIncome",
    "DebtRatio",
    "NumberOfTime30-59DaysPastDueNotWorse",
)

_COLUNA_ATRASO = "NumberOfTime30-59DaysPastDueNotWorse"

# Coeficientes de intensidade máxima (k=1, mês MESES) de cada transformação de data
# drift — literais do desenho da fase: inflação nominal de 40% na renda, injeção de novo
# perfil de inadimplência em 15% das linhas.
_COEFICIENTE_RENDA = 0.40
_FRACAO_MAXIMA_ATRASO = 0.15

# O coeficiente de dívida também é literal do desenho (80%), mas a Referência real
# (117.917 linhas) já tem 9 linhas exatamente no teto do contrato, DEBT_RATIO_MAXIMO=10,0
# — medido com `ref.DebtRatio.max()` e `(ref.DebtRatio == 10.0).sum()`. Qualquer fator de
# inflação maior que 1 empurraria essas linhas para fora do contrato em qualquer amostra
# que as inclua, então o coeficiente sozinho não basta: `aplicar_drift_de_divida` também
# limita (clip) o resultado ao próprio teto do contrato — não a um teto novo, o mesmo que
# a Referência já respeita. Medido sobre a Referência inteira com este coeficiente
# (`(ref.DebtRatio > DEBT_RATIO_MAXIMO / 1.8).sum()`): 200 das 117.917 linhas (0,17%)
# precisam do clip — a imensa maioria do lote recebe a inflação de 80% inteira, sem
# distorção perceptível na distribuição (o PSI do mês 6 é o mesmo com ou sem o clip,
# porque o bin de topo do PSI por quantil é aberto). O clip, no entanto, cria um empate
# exato em 10,0 que a Referência real não tem nessa proporção — um artefato distribucional
# à parte da inflação legítima, inofensivo sob binning por quantil mas visível em qualquer
# diagnóstico de valores distintos ou de forma de cauda.
_COEFICIENTE_DIVIDA = 0.80

# O segmento de risco emergente: histórico de atraso limpo (as três colunas de
# `COLUNAS_DE_ATRASO` em zero) e alavancagem (`DebtRatio`) acima do próprio p75 desse
# subgrupo. Medido na Referência real (117.917 linhas): histórico limpo é 78,8% da
# Referência; dentro dele, o quartil superior de DebtRatio (>= 0,458121, o p75 do próprio
# subgrupo — `ref[atraso_limpo].DebtRatio.quantile(0.75)`) já tem taxa de positivos
# 4,86% contra 2,32% no resto do mesmo subgrupo — o dobro, dentro de um segmento que hoje
# é mais seguro que a média geral (6,94%). É o "novo perfil de cliente" que o enunciado
# pede: um grupo que o histórico ensinou a tratar como baixo risco, e cuja alavancagem
# crescente (o mesmo eixo que `aplicar_drift_de_divida` infla) é o sinal, já presente nos
# dados reais, de que essa crença deixa de valer sob inflação.
_LIMIAR_DEBT_RATIO_REGIAO = 0.458121

# Fração máxima (k=1) dos elegíveis (região ∩ rótulo negativo) que vira positiva.
# Calibrado rodando o campeão publicado (`models/model.joblib`) contra os seis meses
# simulados na partição de teste real (23.584 linhas, mesmo corte de `separar()`):
# valores entre 0,15 e 0,35 produzem AUC-PR e recall da classe positiva
# monotonicamente decrescentes mês a mês; acima disso a taxa de positivos do segmento
# infla o bastante para a AUC-PR voltar a subir (mais positivos fáceis de ordenar no
# topo), o oposto de degradação. 0,25 fica no meio dessa faixa medida, com folga dos
# dois lados.
_TAXA_MAXIMA_DE_INVERSAO = 0.25


def _regiao_de_risco_emergente(frame: pd.DataFrame) -> pd.Series:
    """O segmento cujo ``P(y|X)`` a inflação desloca: ver a procedência medida em
    `_LIMIAR_DEBT_RATIO_REGIAO`. Definido só a partir de `FEATURES` — nunca da previsão
    de nenhum modelo — para que a mudança seja descobrível por qualquer campeão que
    olhasse para esses dados, não construída contra as previsões de um em particular.
    """
    atraso_limpo = (frame[list(COLUNAS_DE_ATRASO)] == 0).all(axis=1)
    return atraso_limpo & (frame["DebtRatio"] >= _LIMIAR_DEBT_RATIO_REGIAO)


def aplicar_drift_de_renda(frame: pd.DataFrame, intensidade: float) -> pd.DataFrame:
    """Renda nominal sobe com a inflação — muda ``P(X)``, nunca o rótulo.

    ``intensidade`` é ``k = mes / MESES``: em k=0 o fator é 1 (sem efeito), em k=1 a
    renda sobe 40%, o coeficiente literal do desenho desta fase.
    """
    resultado = frame.copy()
    resultado["MonthlyIncome"] = resultado["MonthlyIncome"] * (1 + _COEFICIENTE_RENDA * intensidade)
    return resultado


def aplicar_drift_de_divida(frame: pd.DataFrame, intensidade: float) -> pd.DataFrame:
    """Endividamento cresce mais rápido que a renda — muda ``P(X)``, nunca o rótulo.

    O resultado é limitado ao teto do próprio contrato de dados (``DEBT_RATIO_MAXIMO``):
    ver a justificativa medida na constante ``_COEFICIENTE_DIVIDA`` acima. Sem o clip,
    o mesmo lote que este módulo produz reprovaria no contrato de ingestão — e um lote
    reprovado nunca chega ao detector de drift, o que anula o argumento central da fase.
    """
    resultado = frame.copy()
    fator = 1 + _COEFICIENTE_DIVIDA * intensidade
    resultado["DebtRatio"] = (resultado["DebtRatio"] * fator).clip(upper=DEBT_RATIO_MAXIMO)
    return resultado


def aplicar_drift_de_atraso(frame: pd.DataFrame, intensidade: float, *, seed: int) -> pd.DataFrame:
    """Injeta um novo perfil de inadimplência em ``0,15 * intensidade`` das linhas.

    Muda ``P(X)``, nunca o rótulo. O incremento (1 a 3 atrasos) e o clip no teto de
    plausibilidade (``ATRASO_MAXIMO_PLAUSIVEL``) espelham a mesma regra que o contrato de
    ingestão já aplica à Referência — o clip aqui é defensivo: no arquivo real, o valor
    máximo desta coluna é 13, longe do teto de 20, mas a amostra recebida por esta função
    não tem essa garantia.
    """
    resultado = frame.copy()
    n = len(resultado)
    quantidade = int(round(_FRACAO_MAXIMA_ATRASO * intensidade * n))

    if quantidade > 0:
        gerador = np.random.default_rng(seed)
        indices = gerador.choice(n, size=quantidade, replace=False)
        incremento = gerador.integers(1, 4, size=quantidade)

        valores = resultado[_COLUNA_ATRASO].to_numpy().copy()
        valores[indices] = valores[indices] + incremento
        resultado[_COLUNA_ATRASO] = np.clip(valores, 0, ATRASO_MAXIMO_PLAUSIVEL)

    return resultado


def aplicar_concept_drift(frame: pd.DataFrame, intensidade: float, *, seed: int) -> pd.DataFrame:
    """Muda ``P(y|X)`` dentro do segmento de risco emergente — nunca as features, e nunca
    nada fora do segmento.

    Em ``intensidade == 0`` o rótulo devolvido é **idêntico** ao recebido: nenhuma linha é
    sorteada, nenhuma é tocada. Acima de zero, uma fração ``_TAXA_MAXIMA_DE_INVERSAO *
    intensidade`` dos elegíveis — linhas dentro de `_regiao_de_risco_emergente` cujo
    rótulo ainda é negativo — vira positiva, sorteada sem reposição com `seed`. É uma
    inversão, não um sorteio a partir de uma probabilidade recalculada do zero: uma linha
    fora do segmento, ou já positiva, sai exatamente como entrou, em qualquer
    intensidade — a mudança de crença é sobre esse segmento, não sobre o dataset inteiro.
    """
    resultado = frame.copy()
    if intensidade <= 0:
        return resultado

    candidatos = _regiao_de_risco_emergente(resultado) & (resultado[ALVO] == 0)
    elegiveis = np.flatnonzero(candidatos.to_numpy())
    quantidade = int(round(_TAXA_MAXIMA_DE_INVERSAO * intensidade * len(elegiveis)))

    if quantidade > 0:
        gerador = np.random.default_rng(seed)
        selecionados = gerador.choice(elegiveis, size=quantidade, replace=False)
        alvo = resultado[ALVO].to_numpy().copy()
        alvo[selecionados] = 1
        resultado[ALVO] = alvo

    return resultado


def simular_producao(
    referencia: pd.DataFrame,
    *,
    meses: int = MESES,
    seed: int,
    modelo: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Compõe as quatro transformações num lote por mês, do mês 0 (a Referência) ao mês
    ``meses``, com intensidade progressiva ``k = mes / meses``.

    O mês 0 sai idêntico à Referência em FEATURES **e** no rótulo: multiplicar por
    ``1 + coef*0`` é multiplicar por 1, injetar em ``0,15*0`` das linhas é injetar em zero
    linhas, e ``aplicar_concept_drift`` devolve o rótulo intocado quando ``intensidade``
    é zero. Isso é o que torna o mês 0 um marco confiável de "sem drift" — inclusive para
    quem for medir o campeão contra ele —, não um lote especial tratado à parte.

    ``modelo``, quando informado, recebe a mesma interface de ``model.evaluate``
    (``predict_proba`` sobre ``FEATURES``) e grava a probabilidade prevista numa coluna
    ``previsao`` — conveniência para quem for medir degradação lote a lote sem reabrir o
    modelo a cada vez.
    """
    lotes: dict[str, pd.DataFrame] = {}

    for mes in range(meses + 1):
        k = mes / meses
        # Sementes distintas por mês e por transformação estocástica: duas transformações
        # do mesmo mês não podem compartilhar o mesmo sorteio, senão a injeção de atraso e
        # o sorteio do rótulo ficariam correlacionados por acidente de implementação, não
        # por nenhuma razão do domínio.
        lote = referencia.copy()
        lote = aplicar_drift_de_renda(lote, k)
        lote = aplicar_drift_de_divida(lote, k)
        lote = aplicar_drift_de_atraso(lote, k, seed=seed + 2 * mes)
        lote = aplicar_concept_drift(lote, k, seed=seed + 2 * mes + 1)

        if modelo is not None:
            lote["previsao"] = modelo.predict_proba(lote[list(FEATURES)])[:, 1]

        lotes[f"mes_{mes:02d}"] = lote

    return lotes
