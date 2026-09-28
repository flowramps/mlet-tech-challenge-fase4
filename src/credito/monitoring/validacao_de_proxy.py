"""Quanto cada proxy sem rótulo antecipa a degradação real — o entregável central da Etapa 3.

A Etapa 2 provou, por intervenção causal, que PSI de feature mede 3,1% da degradação do
campeão, enquanto concept drift (invisível a qualquer PSI sobre `P(X)`) mede 48% (ver
`credito.drift.causal`, números publicados no README). Disso segue que um painel que
dispara no PSI de feature mais alto aponta consistentemente para a causa errada — mas isso
não diz se ALGUM sinal observável sem rótulo (`credito.monitoring.proxies`) rastreia a
degradação real. Só é possível responder essa pergunta porque a simulação da Etapa 2 *tem*
rótulo: nenhum sistema em produção consegue rodar esta validação, porque não teria a
degradação real para comparar (ver `a-ideia-central.md`).

**O método.** Para cada lote simulado, computa-se cada proxy de `monitoring.proxies` e a
degradação real medida contra o campeão publicado (`credito.drift.calibration.
degradacao_por_lote`, que delega a `model.evaluate.avaliar` — nunca recalculada aqui). A
correlação entre as duas séries usa **Spearman, não Pearson**: o que importa é se o proxy
ordena os lotes na mesma ordem que a degradação real, não se a relação é linear — um proxy
pode se mover de forma monotônica e não-linear com a degradação e ainda ser um alarme
perfeito; Pearson subestimaria essa relação por medir só a parte linear dela.

**Duas leituras de degradação real, não uma.** AUC-ROC é insensível à prevalência do lote
por construção — não precisa de nenhum ajuste. AUC-PR precisa: o piso estrutural da AUC-PR
é a própria taxa de positivos do lote, e essa taxa se desloca junto com o drift simulado —
"lift acima do piso" (`auc_pr - taxa_de_positivos`) é o que sobra depois de descontar esse
piso móvel (a mesma leitura que o README da Etapa 2 documenta na seção "A sutileza do piso
de prevalência"). As duas entram na correlação; nenhuma decide sozinha o alarme.

**A incerteza é reportada, não escondida.** Seis lotes é uma amostra pequena para qualquer
correlação — o intervalo de confiança de `correlacao_spearman` usa o z de Fisher com o
ajuste de erro-padrão de Bonett & Wright (2000) para Spearman (`SE = sqrt(1,06/(n-3))`, 6%
maior que o SE de Pearson para refletir a variância extra de correlacionar postos). Com
n=6, `n-3=3` — o intervalo existe, mas é largo; reportar só o ponto (`rho`) sem o intervalo
sugeriria uma precisão que seis observações não sustentam.

**Este módulo tem acesso ao rótulo e é o único lugar do pacote `monitoring` que pode ter**
— ao contrário de `proxies.py`, cujo próprio propósito é medir só o que um serviço de
produção real veria. A validação exige comparar contra a verdade, então comparar contra a
verdade é exatamente o trabalho deste módulo, não um vazamento dele.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from credito.drift.calibration import degradacao_por_lote
from credito.monitoring.proxies import LIMIAR_PADRAO, sinais_do_lote

# Campos de degradação real presentes em `degradacao_por_lote` (todos derivados de
# `model.evaluate.avaliar`, mais o lift acima do piso computado aqui) contra os quais cada
# proxy é correlacionado — ver o docstring do módulo sobre por que as duas entram.
CAMPOS_DE_DEGRADACAO: tuple[str, ...] = ("degradacao_auc_roc", "degradacao_lift_acima_do_piso")

# Ajuste do erro-padrão do z de Fisher para Spearman em vez de Pearson (Bonett & Wright,
# 2000, "Sample size requirements for estimating Pearson, Kendall and Spearman
# correlations"): o SE de Pearson (`1/sqrt(n-3)`) subestima a variância de uma correlação
# de POSTOS; multiplicar por 1,06 é a correção fechada que o método propõe, sem exigir
# reamostragem.
_AJUSTE_BONETT_WRIGHT = 1.06

# Com n<=3, `n-3<=0` e o erro-padrão do z de Fisher (que divide por `n-3`) diverge ou fica
# indefinido — não há intervalo assintótico possível abaixo deste tamanho de amostra,
# só o rho pontual.
_N_MINIMO_PARA_INTERVALO = 4

_Z_95 = 1.959963984540054  # stats.norm.ppf(0.975) — os dois lados de 95% de confiança.


class ResultadoCorrelacao(NamedTuple):
    """Rho de Spearman, p-valor e intervalo de confiança, nomeados — a mesma razão de
    `ResultadoKS` em `drift/statistics.py`: uma tupla posicional obrigaria quem lê a
    decorar a ordem, e trocar `rho` pelo limite inferior do intervalo por engano não geraria
    erro nenhum, só um número errado silencioso num relatório de risco de crédito.
    """

    rho: float
    p_valor: float
    intervalo_confianca: tuple[float, float]
    n: int


def correlacao_spearman(x: np.ndarray, y: np.ndarray) -> ResultadoCorrelacao:
    """Spearman entre duas séries, com intervalo de confiança de 95% pelo z de Fisher
    ajustado (ver `_AJUSTE_BONETT_WRIGHT`).

    Uma série constante (variância zero) deixa a correlação matematicamente indefinida —
    `scipy.stats.spearmanr` devolve `nan` e emite `ConstantInputWarning` para avisar disso.
    Este projeto trata warning como erro (`pyproject.toml`, `filterwarnings = ["error"]`);
    chamar `spearmanr` sobre uma série constante quebraria a suíte por um `nan` que já era
    o resultado matematicamente correto. A checagem abaixo intercepta esse caso ANTES de
    chamar `spearmanr`, devolvendo o mesmo `nan` sem o aviso — não é uma forma de esconder
    o problema, é reconhecer que "sem variância, sem correlação definida" não é um erro de
    execução.

    Intervalo de confiança só é calculado com `n >= _N_MINIMO_PARA_INTERVALO` pontos (ver a
    constante) — abaixo disso o erro-padrão do z de Fisher divide por um número não
    positivo. `rho` fica sempre disponível quando definido; o intervalo é `(nan, nan)`
    nesse caso, nunca um valor inventado.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return ResultadoCorrelacao(
            rho=float("nan"),
            p_valor=float("nan"),
            intervalo_confianca=(float("nan"), float("nan")),
            n=n,
        )

    rho, p_valor = stats.spearmanr(x, y)
    rho = float(rho)

    if n < _N_MINIMO_PARA_INTERVALO:
        return ResultadoCorrelacao(
            rho=rho,
            p_valor=float(p_valor),
            intervalo_confianca=(float("nan"), float("nan")),
            n=n,
        )

    # `arctanh(±1)` diverge: um rho perfeito medido sobre seis pontos não significa que a
    # correlação populacional é exatamente 1 — significa que a amostra é pequena demais
    # para distinguir "1" de "muito perto de 1". Limitar rho a uma distância mínima de ±1
    # só para o cálculo do intervalo evita um intervalo infinito que esconderia essa
    # incerteza em vez de reportá-la.
    rho_para_z = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
    z = np.arctanh(rho_para_z)
    erro_padrao = np.sqrt(_AJUSTE_BONETT_WRIGHT / (n - 3))
    z_baixo, z_alto = z - _Z_95 * erro_padrao, z + _Z_95 * erro_padrao

    return ResultadoCorrelacao(
        rho=rho,
        p_valor=float(p_valor),
        intervalo_confianca=(float(np.tanh(z_baixo)), float(np.tanh(z_alto))),
        n=n,
    )


def correlacionar_proxies(
    lotes: dict[str, pd.DataFrame],
    modelo: Any,
    *,
    limiar: float = LIMIAR_PADRAO,
    bins: int = 10,
) -> dict[str, Any]:
    """Para cada lote de `lotes` (exceto o mais antigo, tratado como referência), calcula
    os proxies sem rótulo e a degradação real, e correlaciona as duas séries.

    `lotes` precisa ser o dicionário que `credito.data.simulate.simular_producao` produz —
    chaves `mes_00`, `mes_01`, ... — porque a ordenação usada para decidir qual é a
    referência é a ordem lexicográfica das chaves (`mes_00` < `mes_01` < ...), a mesma
    convenção que todo o resto do projeto já usa para nomear lote por mês. A chave
    lexicograficamente menor vira a referência: o proxy `psi_do_score` mede deslocamento
    *contra* alguma coisa, e a degradação real de cada lote é medida como queda a partir das
    métricas do campeão nessa mesma referência — sem uma referência não há "quanto
    antecipou", só um número solto por lote. Os `len(lotes) - 1` lotes restantes entram na
    correlação: passar o dicionário inteiro de seis meses de `simular_producao` (sete
    chaves, `mes_00` a `mes_06`) produz os seis pontos que o método desta etapa pede.

    Degradação real usa `credito.drift.calibration.degradacao_por_lote` — nunca recalcula
    `avaliar` aqui — e reporta duas leituras (ver o docstring do módulo sobre por que):
    `degradacao_auc_roc` (queda de AUC-ROC contra a referência, insensível à prevalência
    por construção) e `degradacao_lift_acima_do_piso` (queda de `auc_pr -
    taxa_de_positivos`, que desconta o piso estrutural da AUC-PR antes de medir a queda).

    Levanta `ValueError` se `lotes` tiver menos de dois lotes: sem um segundo lote além da
    referência não há o que correlacionar.
    """
    if len(lotes) < 2:
        raise ValueError(
            "correlacionar_proxies precisa de ao menos uma referência e um lote para "
            "correlacionar"
        )

    nomes_ordenados = sorted(lotes)
    nome_da_referencia = nomes_ordenados[0]
    referencia = lotes[nome_da_referencia]
    nomes_dos_lotes = nomes_ordenados[1:]

    metricas_por_lote = degradacao_por_lote(modelo, lotes)
    metricas_da_referencia = metricas_por_lote[nome_da_referencia]
    lift_da_referencia = (
        metricas_da_referencia["auc_pr"] - metricas_da_referencia["taxa_de_positivos"]
    )

    proxies_por_lote: dict[str, dict[str, float]] = {}
    degradacao: dict[str, dict[str, float]] = {}

    for nome in nomes_dos_lotes:
        proxies_por_lote[nome] = sinais_do_lote(
            modelo, lotes[nome], referencia=referencia, limiar=limiar, bins=bins
        )

        metricas = metricas_por_lote[nome]
        lift = metricas["auc_pr"] - metricas["taxa_de_positivos"]
        degradacao[nome] = {
            **metricas,
            "lift_acima_do_piso": lift,
            "degradacao_auc_roc": metricas_da_referencia["auc_roc"] - metricas["auc_roc"],
            "degradacao_lift_acima_do_piso": lift_da_referencia - lift,
        }

    nomes_dos_proxies = sorted(proxies_por_lote[nomes_dos_lotes[0]])
    correlacoes: dict[str, dict[str, ResultadoCorrelacao]] = {}
    for nome_proxy in nomes_dos_proxies:
        serie_proxy = np.array([proxies_por_lote[nome][nome_proxy] for nome in nomes_dos_lotes])
        correlacoes[nome_proxy] = {
            campo: correlacao_spearman(
                serie_proxy,
                np.array([degradacao[nome][campo] for nome in nomes_dos_lotes]),
            )
            for campo in CAMPOS_DE_DEGRADACAO
        }

    return {
        "referencia": nome_da_referencia,
        "lotes": nomes_dos_lotes,
        "proxies_por_lote": proxies_por_lote,
        "degradacao_por_lote": degradacao,
        "correlacoes": correlacoes,
    }


def _significativos(
    correlacoes: dict[str, dict[str, ResultadoCorrelacao]], *, campo: str
) -> dict[str, ResultadoCorrelacao]:
    """Os proxies cujo intervalo de confiança contra `campo` NÃO cruza zero — a mesma
    checagem que `escolher_alarme` e `proxies_no_topo` compartilham, isolada para que as
    duas nunca possam divergir sobre o que "significativo" significa aqui.

    Um `rho` grande cujo intervalo inclui zero não é evidência de que o proxy antecipa a
    degradação: com seis pontos, é perfeitamente compatível com "nenhuma correlação
    populacional". Filtrar pelo ponto (`rho`) sozinho, ignorando o intervalo, seria
    exatamente o erro que o brief desta etapa pede para não cometer — reportar o número
    desapontador (ou a ausência de alarme) em vez de maquiar a incerteza.
    """
    candidatos = {
        nome: resultado[campo]
        for nome, resultado in correlacoes.items()
        if not np.isnan(resultado[campo].rho)
    }
    return {
        nome: resultado
        for nome, resultado in candidatos.items()
        if not np.isnan(resultado.intervalo_confianca[0])
        and not (resultado.intervalo_confianca[0] <= 0 <= resultado.intervalo_confianca[1])
    }


def proxies_no_topo(
    correlacoes: dict[str, dict[str, ResultadoCorrelacao]],
    *,
    campo: str = "degradacao_auc_roc",
) -> tuple[str, ...]:
    """TODOS os proxies empatados no maior `|rho|` significativo contra `campo`, em ordem
    alfabética — não só um.

    Com seis pontos, um empate exato entre proxies não é incomum: quando o drift simulado é
    monotônico na intensidade (`k = mes/meses`, ver `credito.data.simulate`), qualquer sinal
    que também seja monotônico no tempo bate `rho = ±1,0` contra a degradação real — e mais
    de um proxy pode satisfazer essa condição ao mesmo tempo, sem que isso signifique que um
    é melhor que o outro. `escolher_alarme` devolve só um nome (o primeiro em ordem
    alfabética dentre os empatados — consequência de como `max` resolve empate, não mérito
    medido) — conveniente para quem precisa de uma única resposta, mas capaz de esconder o
    empate de quem lê só aquele nome. Esta função existe para que um relatório desta etapa
    nunca apresente um vencedor arbitrário como se fosse superioridade medida: ver
    `scripts/validar_proxies.py`, que imprime o conjunto inteiro, não só `escolher_alarme`.
    """
    significativos = _significativos(correlacoes, campo=campo)
    if not significativos:
        return ()
    maior = max(abs(resultado.rho) for resultado in significativos.values())
    return tuple(sorted(nome for nome, r in significativos.items() if abs(r.rho) == maior))


def escolher_alarme(
    correlacoes: dict[str, dict[str, ResultadoCorrelacao]],
    *,
    campo: str = "degradacao_auc_roc",
) -> str | None:
    """Escolhe UM proxy — o de maior `|rho|` contra `campo`, entre os que têm intervalo de
    confiança que NÃO cruza zero — e devolve `None` quando nenhum qualifica.

    Em caso de empate exato de `|rho|` entre dois ou mais proxies (ver `proxies_no_topo`
    sobre quando isso acontece), devolve o primeiro em ordem alfabética — uma escolha
    determinística, não uma escolha por mérito. Um chamador que precisa de uma única
    resposta (ex.: configurar um único alarme) usa esta função; um relatório que precisa
    ser honesto sobre a possibilidade de empate usa `proxies_no_topo`, que devolve o
    conjunto inteiro em vez de decidir por ele.

    `None` é uma resposta honesta e esperada, não uma falha da função: o achado "nenhum
    proxy antecipa a degradação com confiança" é tão válido quanto "este proxy antecipa" —
    ver `a-ideia-central.md`.
    """
    topo = proxies_no_topo(correlacoes, campo=campo)
    return topo[0] if topo else None
