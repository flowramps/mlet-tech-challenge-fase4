"""Os sinais que um serviço de produção consegue calcular sem nunca ver o rótulo.

**Por que este módulo existe separado de `drift/`.** A Etapa 2 mediu duas coisas que
juntas decidem como um alarme de produção precisa ser construído. Primeiro
(`credito.drift.causal.atribuir_degradacao`, números publicados no README): as duas
features de maior PSI — `MonthlyIncome` (0,2634) e `DebtRatio` (0,5046) — causam juntas
3,1% da degradação medida contra o campeão; o concept drift isolado, que nenhum PSI de
feature enxerga porque `P(X)` não se move nele, causa 48,0%. Um painel que dispara no PSI
mais alto aponta consistentemente para a variável errada. Segundo: a simulação da Etapa 2
tem rótulo porque nós controlamos o gerador — um sistema real não sabe quem inadimpliu até
meses depois da decisão de crédito, e é exatamente nesse intervalo que a degradação
silenciosa acontece. `drift/` mede o que só é mensurável em laboratório (PSI de feature
contra uma Referência, atribuição causal contra o rótulo real). Este módulo mede só o que
seria mensurável em produção — as entradas que o modelo recebeu e o que ele devolveu — e
por isso não pode importar a coluna de rótulo em lugar nenhum: fazer isso apagaria a
distinção que o módulo existe para provar.

Os quatro sinais, na mesma tabela de candidatos que a Etapa 3 se propõe a validar
(`a-ideia-central.md`): deslocamento da distribuição de score (`psi_do_score`), queda da
confiança média (`confianca_media`), deriva da taxa de aprovação agregada
(`taxa_de_aprovacao`) e a composição das três para um lote (`sinais_do_lote`). Nenhum
decide sozinho se é o melhor proxy da degradação real — essa correlação é medida à parte,
lote a lote, contra a Etapa 2; aqui só o cálculo de cada sinal é responsabilidade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credito.drift.statistics import psi
from credito.schema import FEATURES

# Mesmo padrão de `credito.model.evaluate.avaliar`: 0,5 é a fronteira de decisão binária
# usual, não um valor calibrado neste dado — convenção, não medição.
LIMIAR_PADRAO = 0.5


def _probabilidades(modelo: Any, lote: pd.DataFrame) -> np.ndarray:
    """Extrai a probabilidade da classe positiva, selecionando só `FEATURES`.

    Selecionar exatamente `FEATURES` (nunca `lote` inteiro) é o que faz este módulo não
    precisar saber se a coluna de rótulo, ou qualquer outra coluna, está presente no
    `lote` — o mesmo recorte que `credito.model.evaluate._probabilidades` já faz,
    reproduzido aqui em vez de importado porque os dois módulos respondem a perguntas
    diferentes (ver o docstring do módulo) e não deveriam depender um do outro para uma
    função de duas linhas.
    """
    return modelo.predict_proba(lote[list(FEATURES)])[:, 1]


def psi_do_score(referencia: np.ndarray, atual: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index entre duas amostras da distribuição de **score**.

    Delega inteiramente a `credito.drift.statistics.psi` — não recalcula nada. Um score
    de risco de crédito bem calibrado se concentra perto de zero (a maioria dos clientes
    é de baixo risco), a mesma forma de massa concentrada que fez `psi()` deixar de
    decidir o binning pela cardinalidade da Referência e passar a medir o colapso do
    binning por quantil diretamente (ver o docstring de `drift/statistics.py`): um score
    zero-inflado colapsaria os mesmos cortes de quantil que
    `NumberOfTime30-59DaysPastDueNotWorse` colapsa, pela mesma razão mecânica. Reusar
    `psi()` significa herdar esse roteamento sem precisar decidir de novo se o score
    colapsa — `psi()` já mede isso.
    """
    return psi(referencia, atual, bins=bins)


def confianca_media(probabilidades: np.ndarray) -> float:
    """Confiança média do lote: a distância média entre o score e a fronteira de decisão
    0,5, dobrada para o intervalo [0,5, 1,0].

    Usar `max(p, 1-p)` em vez da probabilidade bruta é o que torna a métrica simétrica: um
    cliente com `p=0,1` (o modelo está confiante de que ele é bom pagador) e outro com
    `p=0,9` (o modelo está igualmente confiante de que ele é mau pagador) representam a
    mesma confiança do modelo, não confianças opostas — só a média bruta de `p` os trataria
    como extremos diferentes. A fórmula (`max(p, 1-p)`), não a convenção de "0,5 é a
    fronteira", é medição: para qualquer `p` em [0,1], `max(p, 1-p)` é sempre >= 0,5, e vale
    exatamente 1,0 quando o modelo está seguro (`p` em {0,1}) e 0,5 quando está indiferente
    (`p=0,5`) — o mesmo range que a fronteira de decisão usual (0,5, também convenção, não
    medição) exige.

    Levanta `ValueError` num lote vazio: a média de um array vazio é `nan`, e um `nan`
    dentro do relatório de monitoramento pareceria um sinal calculado em vez do que
    realmente é — ausência de dado.
    """
    probabilidades = np.asarray(probabilidades, dtype=float)
    if probabilidades.size == 0:
        raise ValueError("confianca_media não está definida para um lote vazio")
    return float(np.mean(np.maximum(probabilidades, 1 - probabilidades)))


def taxa_de_aprovacao(probabilidades: np.ndarray, *, limiar: float) -> float:
    """Fração do lote que o modelo aprova — a mesma convenção de
    `credito.model.evaluate.avaliar_por_faixa_etaria`: aprovado é quem o modelo **não**
    classifica como risco, isto é, `probabilidade < limiar`.

    É a decisão agregada, não a distribuição: dois lotes podem ter a mesma
    `taxa_de_aprovacao` com confianças médias bem diferentes (um deles perto da fronteira
    em cada caso, o outro longe dela) — por isso esta métrica e `confianca_media`
    respondem perguntas diferentes e nenhuma substitui a outra.

    Levanta `ValueError` num lote vazio, pela mesma razão de `confianca_media`.
    """
    probabilidades = np.asarray(probabilidades, dtype=float)
    if probabilidades.size == 0:
        raise ValueError("taxa_de_aprovacao não está definida para um lote vazio")
    return float(np.mean(probabilidades < limiar))


def sinais_do_lote(
    modelo: Any,
    lote: pd.DataFrame,
    *,
    referencia: pd.DataFrame | None = None,
    limiar: float = LIMIAR_PADRAO,
    bins: int = 10,
) -> dict[str, float]:
    """Os sinais observáveis de um lote, calculados só a partir de `FEATURES` e do que o
    `modelo` devolve — nunca da coluna de rótulo (ver
    `tests/monitoring/test_proxies.py::test_pacote_monitoring_nao_cita_a_coluna_alvo_em_lugar_nenhum`,
    que confere isso estaticamente para este módulo e para todo módulo que ele importar
    de dentro de `credito.monitoring`, e `test_sinais_do_lote_funciona_sem_a_coluna_alvo`
    / `test_sinais_do_lote_nao_le_features_extras_como_o_alvo`, que conferem em
    execução).

    `confianca_media` e `taxa_de_aprovacao` só precisam do `lote`: sempre presentes.
    `psi_do_score` precisa de uma segunda distribuição de score para comparar — não existe
    "deslocamento" sem um "a partir de quê" —, e por isso só entra no resultado quando
    `referencia` é informada. Um chamador que só tem o lote corrente (ex.: o primeiro lote
    que um monitor recém-publicado processa, antes de ter uma Referência de score
    calibrada) continua recebendo os outros dois sinais em vez de um erro. `bins` só
    afeta `psi_do_score` (repassado sem alteração) — as outras duas métricas não têm
    binning nenhum.
    """
    probabilidades = _probabilidades(modelo, lote)
    sinais: dict[str, float] = {
        "confianca_media": confianca_media(probabilidades),
        "taxa_de_aprovacao": taxa_de_aprovacao(probabilidades, limiar=limiar),
    }
    if referencia is not None:
        probabilidades_referencia = _probabilidades(modelo, referencia)
        sinais["psi_do_score"] = psi_do_score(probabilidades_referencia, probabilidades, bins=bins)
    return sinais
