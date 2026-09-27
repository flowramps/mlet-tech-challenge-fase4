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
  ``aplicar_concept_drift`` faz: o rótulo do lote de produção é **recalculado**, não
  copiado da Referência, porque sob inflação o mesmo ``DebtRatio`` passa a implicar mais
  risco do que implicava antes.

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

from credito.schema import ALVO, ATRASO_MAXIMO_PLAUSIVEL, DEBT_RATIO_MAXIMO, FEATURES

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

# Padronização de DebtRatio usada em `aplicar_concept_drift`: média e desvio medidos na
# Referência real (117.917 linhas), `ref.DebtRatio.mean()` e `.std()`. É fixa — não
# recalculada por lote — porque o que muda com a inflação é o que um DebtRatio já
# conhecido *significa* em termos de risco, não a régua que mede esse DebtRatio. Uma
# padronização recalculada por lote apagaria o próprio deslocamento de `P(X)` que o data
# drift produziu, o que tornaria concept e data drift artificialmente independentes por
# construção em vez de por medição.
_MEDIA_DIVIDA_REFERENCIA = 0.374921
_DESVIO_DIVIDA_REFERENCIA = 0.483042

INCLINACAO_DIVIDA = 1.0

# Calibrado (Passo 5) por bisseção contra a Referência real (117.917 linhas): é o valor
# que faz a média de sigmoid(LOGITO_BASE + INCLINACAO_DIVIDA * divida_padronizada), sobre
# o DebtRatio real da Referência em k=0, reproduzir a taxa de positivos medida na própria
# Referência, 6,9422%. Valor obtido: -2,845598.
LOGITO_BASE = -2.845598

# Faz a taxa de positivos subir ao longo dos meses. Medido rodando `simular_producao`
# sobre uma amostra de 20.000 linhas da Referência real (Passo 5): a taxa de positivos
# sai de 6,89% no mês 0 para 24,14% no mês 6 — um aumento visível sem ser uma inversão
# implausível da base de risco.
DESLOCAMENTO = 1.0


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
    """Recalcula o rótulo — muda ``P(y|X)``, nunca as features.

    Sob inflação, o mesmo ``DebtRatio`` passa a implicar mais risco: o logito soma um
    termo fixo (``LOGITO_BASE``), um termo proporcional ao DebtRatio padronizado
    (``INCLINACAO_DIVIDA``) e um deslocamento que cresce com a intensidade
    (``DESLOCAMENTO * intensidade``). O rótulo é **sorteado** a partir dessa
    probabilidade, não copiado — duas linhas com o mesmo DebtRatio e a mesma intensidade
    podem sair com rótulos diferentes, exatamente como duas pessoas com o mesmo perfil de
    risco não têm necessariamente o mesmo desfecho.
    """
    resultado = frame.copy()
    divida_padronizada = (
        resultado["DebtRatio"].to_numpy() - _MEDIA_DIVIDA_REFERENCIA
    ) / _DESVIO_DIVIDA_REFERENCIA
    logito = LOGITO_BASE + INCLINACAO_DIVIDA * divida_padronizada + DESLOCAMENTO * intensidade
    probabilidade = 1 / (1 + np.exp(-logito))

    gerador = np.random.default_rng(seed)
    resultado[ALVO] = (gerador.random(len(resultado)) < probabilidade).astype(int)
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

    O mês 0 sai idêntico à Referência em FEATURES: multiplicar por ``1 + coef*0`` é
    multiplicar por 1, e injetar em ``0,15*0`` das linhas é injetar em zero linhas. Isso é
    o que torna o mês 0 um marco confiável de "sem drift", não um lote especial tratado à
    parte.

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
