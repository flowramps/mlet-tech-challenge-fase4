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

**Concordância de ORDEM não é força de SINAL, e as duas precisam aparecer separadas.**
Spearman com seis pontos monotônicos satura em `rho = ±1,0` para qualquer proxy que também
seja monotônico no tempo — o que prova que a ordem bate, não que o proxy se moveu o
bastante para valer como alarme. Um proxy pode empatar em `rho` com outro e ainda ter se
deslocado uma fração do tamanho: é para isso que `magnitude_do_proxy` existe, e é o que
desempata `escolher_alarme` quando `rho` sozinho não distingue (ver o docstring dela).

**A significância usa o p-valor EXATO por permutação, não o intervalo de confiança
fechado.** O p-valor assintótico que `scipy.stats.spearmanr` devolve diverge para `0,0`
exatamente quando `|rho|→1` (a estatística t usada por baixo não está definida no limite) —
o mesmo tipo de aproximação que já se sabe otimista demais em amostra pequena neste
projeto (`ks_2samp` caindo para o modo assintótico, documentado no filtro de warning de
`pyproject.toml`). E o intervalo de confiança fechado (`correlacao_spearman`, z de Fisher
ajustado) tem o problema oposto: perto de `|rho|=1` ele SEMPRE exclui zero, não importa
quão pequena seja a amostra — `n=4` com `rho` perfeito produz o mesmo intervalo
"significativo" que `n=30` produziria, porque o `clip` que evita `arctanh(±1)` divergir
satura antes de `tanh` conseguir refletir a diferença real de incerteza entre os dois `n`.
Nenhum dos dois serve de portão de significância sozinho. `p_valor_exato_spearman` resolve
isso enumerando as `n!` reordenações possíveis (720 para `n=6`, o caso real desta etapa) e
contando a fração tão extrema quanto a observada — o mesmo raciocínio de um teste exato de
permutação, sem aproximação nenhuma. Acima de `n=9` a enumeração completa deixa de ser
viável (`n!` cresce fatorialmente); a função troca para uma estimativa por amostragem de
Monte Carlo em vez de desistir e devolver `nan` — devolver `nan` ali já foi o comportamento
de uma versão anterior, e `nan` é descartado como "não significativo" pelo mesmo portão que
trataria um achado genuíno de ausência de correlação, então um `MESES` maior em
`credito.data.simulate` teria invertido a conclusão da etapa em silêncio (ver o docstring de
`p_valor_exato_spearman`). É esse p-valor — sempre um número de verdade, nunca `nan` por
causa do tamanho de `n` —, não o intervalo, que `_significativos` usa para decidir se um
proxy conta.

**Este módulo tem acesso ao rótulo e é o único lugar do pacote `monitoring` que pode ter**
— ao contrário de `proxies.py`, cujo próprio propósito é medir só o que um serviço de
produção real veria. A validação exige comparar contra a verdade, então comparar contra a
verdade é exatamente o trabalho deste módulo, não um vazamento dele.
"""

from __future__ import annotations

import itertools
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
# reamostragem. Mantido para diagnóstico e para a tabela publicada — NÃO decide mais
# significância (ver `_significativos`): perto de `|rho|=1` este intervalo satura e deixa
# de distinguir tamanhos de amostra diferentes, o defeito que motivou trocar o portão de
# significância pelo p-valor exato por permutação.
_AJUSTE_BONETT_WRIGHT = 1.06

# Com n<=3, `n-3<=0` e o erro-padrão do z de Fisher (que divide por `n-3`) diverge ou fica
# indefinido — não há intervalo assintótico possível abaixo deste tamanho de amostra,
# só o rho pontual.
_N_MINIMO_PARA_INTERVALO = 4

_Z_95 = 1.959963984540054  # stats.norm.ppf(0.975) — os dois lados de 95% de confiança.

# `n!` permutações completas: 720 para o caso real desta etapa (seis lotes). Até `n=9`
# (362.880 permutações) o cálculo vetorizado (ver `p_valor_exato_spearman`) roda em menos
# de um segundo; acima disso o custo cresce fatorialmente e a função troca para a
# estimativa por Monte Carlo (ver `_N_AMOSTRAS_MONTE_CARLO`) em vez de enumerar — nunca
# devolve `nan` por causa do tamanho da amostra. Devolver `nan` acima deste corte já foi o
# comportamento desta função numa versão anterior, e era ele mesmo um defeito do mesmo tipo
# que este módulo existe para fechar: `nan` é descartado como "não significativo" por
# `_significativos`, então um `MESES` maior em `credito.data.simulate` inverteria a
# conclusão inteira da etapa (de "três proxies antecipam" para "nenhum antecipa") com MAIS
# dado, e sem nenhum aviso de que o motivo foi o tamanho da amostra, não a ausência de
# correlação — medido forçando `n=10` com um `rho` perfeito na versão anterior.
_N_MAXIMO_PARA_PERMUTACAO_EXATA = 9

# Tamanho da amostra de Monte Carlo acima do corte exato. Cada permutação sorteada é
# independente das outras, então a estimativa é uma proporção binomial: seu erro-padrão é
# `sqrt(p(1-p)/K)`. Em `p=ALFA_SIGNIFICANCIA=0,05` (o ponto onde a precisão mais importa,
# perto da fronteira de decisão), `K=100.000` dá `SE ~ sqrt(0,05*0,95/100000) ~= 0,00069` —
# um intervalo de 95% de largura ~0,0027 em torno do p estimado, preciso o bastante para
# decidir o lado de `ALFA_SIGNIFICANCIA` na esmagadora maioria dos casos sem exigir minutos
# de cálculo. `100.000` é literal de projeto (não uma medição deste dado), na mesma
# categoria que `ALFA_SIGNIFICANCIA` abaixo.
_N_AMOSTRAS_MONTE_CARLO = 100_000

# Semente fixa para a amostragem de Monte Carlo — mesma convenção de
# `credito.config.Settings.random_seed` (42): sem ela, o mesmo par `(x, y)` devolveria um
# p-valor levemente diferente a cada chamada, o que tornaria `correlacao_spearman` não
# determinística acima de `_N_MAXIMO_PARA_PERMUTACAO_EXATA` — inaceitável para um número que
# entra num relatório de risco de crédito.
_SEMENTE_MONTE_CARLO = 42

# Corte de significância do p-valor exato por permutação — convenção estatística usual
# (0,05), não uma medição deste dado, na mesma categoria que `psi_atencao`/`psi_critico`
# em `config.py` são convenção declarada, não calibração.
ALFA_SIGNIFICANCIA = 0.05


class ResultadoCorrelacao(NamedTuple):
    """Rho de Spearman, os dois p-valores e o intervalo de confiança, nomeados — a mesma
    razão de `ResultadoKS` em `drift/statistics.py`: uma tupla posicional obrigaria quem lê
    a decorar a ordem, e trocar `rho` pelo limite inferior do intervalo por engano não
    geraria erro nenhum, só um número errado silencioso num relatório de risco de crédito.

    `p_valor` é o assintótico que `scipy.stats.spearmanr` devolve; `p_valor_exato` é o de
    `p_valor_exato_spearman` (por permutação completa) — o que decide significância neste
    módulo (ver `_significativos`). Os dois convivem no mesmo resultado de propósito: a
    diferença entre eles, quando existe, é ela mesma um diagnóstico (ver o docstring do
    módulo sobre por que o assintótico pode divergir para 0,0 sem estar errado por acaso).
    """

    rho: float
    p_valor: float
    p_valor_exato: float
    intervalo_confianca: tuple[float, float]
    n: int


def _matriz_de_permutacoes_exatas(postos_y: np.ndarray) -> np.ndarray:
    """Todas as `n!` reordenações de `postos_y`, uma por linha."""
    return np.array(list(itertools.permutations(postos_y)))


def _matriz_de_permutacoes_por_amostragem(postos_y: np.ndarray, *, n_amostras: int) -> np.ndarray:
    """`n_amostras` reordenações de `postos_y` sorteadas uniformemente ao acaso, uma por
    linha — a alternativa à enumeração completa quando `n!` é grande demais para enumerar
    (ver `_N_MAXIMO_PARA_PERMUTACAO_EXATA`).

    `argsort` de ruído uniforme independente por linha (`gerador.random((n_amostras, n))`)
    é o truque padrão para sortear permutações uniformes vetorizado: a ordem dos valores
    sorteados numa linha é ela mesma uniforme sobre as `n!` ordens possíveis, e `argsort`
    devolve os ÍNDICES dessa ordem — que aplicados a `postos_y` (indexação por fantasia,
    `postos_y[indices]`) produzem a reordenação. Evita um laço Python de
    `gerador.permutation` chamado `n_amostras` vezes, que domina o tempo de execução para
    `n_amostras=100.000` (medido: a versão vetorizada roda a amostragem inteira em ordem de
    dezenas de milissegundos).
    """
    gerador = np.random.default_rng(_SEMENTE_MONTE_CARLO)
    n = len(postos_y)
    indices = np.argsort(gerador.random((n_amostras, n)), axis=1)
    return postos_y[indices]


def p_valor_exato_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """P-valor de Spearman por permutação — exato por enumeração completa das `n!`
    reordenações de `y` contra a ordem fixa de `x` quando `n <=
    _N_MAXIMO_PARA_PERMUTACAO_EXATA`, estimado por Monte Carlo (`_N_AMOSTRAS_MONTE_CARLO`
    reordenações sorteadas ao acaso, ver `_matriz_de_permutacoes_por_amostragem`) acima
    disso — em ambos os casos, a fração de reordenações cujo `|rho|` é tão extremo quanto o
    observado, sem a aproximação assintótica que `scipy.stats.spearmanr` usa.

    Correlação de Spearman é a correlação de Pearson entre os POSTOS (`scipy.stats.
    rankdata`, empates pela média — a mesma convenção que `spearmanr` usa por baixo).
    Permutar `y` só reordena seus postos; o próprio conjunto de postos (e portanto a média
    e o desvio-padrão deles) não muda — só a covariância com os postos fixos de `x` muda a
    cada permutação. Por isso a função não recalcula `rankdata`/correlação do zero a cada
    reordenação: constrói a matriz de reordenações dos postos de `y` de uma vez e calcula
    todas as covariâncias com uma única multiplicação de matriz.

    **Por que Monte Carlo, e não `nan`, acima do corte exato.** Uma versão anterior desta
    função devolvia `nan` para `n > _N_MAXIMO_PARA_PERMUTACAO_EXATA`, e `_significativos`
    descarta `nan` como "não significativo" — o mesmo texto que um achado genuíno de
    "nenhuma correlação" produziria. Isso significava que aumentar `MESES` em
    `credito.data.simulate` (hoje 6, bem abaixo do corte) inverteria silenciosamente a
    conclusão inteira desta etapa assim que ultrapassasse `_N_MAXIMO_PARA_PERMUTACAO_EXATA`
    — com MAIS dado, não menos —, e nenhum número no relatório diria que o motivo foi o
    tamanho da amostra cruzar um corte de implementação, não a ausência de correlação. A
    estimativa por Monte Carlo fecha essa lacuna: sempre devolve um p-valor de verdade,
    dentro da margem de erro que `_N_AMOSTRAS_MONTE_CARLO` documenta.

    A contagem de reordenações "tão extremas quanto" usa a correção `+1` no numerador e no
    denominador quando a amostragem é por Monte Carlo (não quando é exata): sem ela, uma
    amostra aleatória que por acaso não incluísse nenhuma reordenação tão extrema quanto a
    observada devolveria um p-valor de exatamente `0,0` — que pareceria "impossível sob a
    hipótese nula" mesmo quando só significa "não amostrado" —, a mesma armadilha do
    p-valor assintótico que `scipy.stats.spearmanr` diverge para `0,0` perto de `|rho|=1`.
    A correção (Davison & Hinkley, 1997; North, Curtis & Sham, 2002) trata a própria
    observação como se fosse ela mesma uma reordenação válida da família amostrada, o que
    garante um p-valor sempre `> 0`. A enumeração exata não precisa da correção porque a
    identidade (a própria observação) já está entre as `n!` reordenações enumeradas, e
    conta por si só.

    Devolve `nan` só quando `x` ou `y` são constantes — a correlação não está definida
    nesse caso, mesma razão de `correlacao_spearman` —, nunca por causa do tamanho de `n`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")

    postos_x = stats.rankdata(x)
    postos_y = stats.rankdata(y)

    media_x, media_y = postos_x.mean(), postos_y.mean()
    desvio_x = postos_x.std()
    desvio_y = postos_y.std()  # invariante a qualquer permutação/amostragem de postos_y

    rho_observado = float(
        (np.mean(postos_x * postos_y) - media_x * media_y) / (desvio_x * desvio_y)
    )

    exato = n <= _N_MAXIMO_PARA_PERMUTACAO_EXATA
    matriz_de_permutacoes = (
        _matriz_de_permutacoes_exatas(postos_y)
        if exato
        else _matriz_de_permutacoes_por_amostragem(postos_y, n_amostras=_N_AMOSTRAS_MONTE_CARLO)
    )
    covariancias = matriz_de_permutacoes @ postos_x / n - media_x * media_y
    rhos_por_permutacao = covariancias / (desvio_x * desvio_y)

    # Tolerância de ponto flutuante: sem ela, a própria permutação identidade (que
    # reproduz `rho_observado` bit a bit em teoria) poderia ficar de fora da contagem por
    # erro de arredondamento na soma vetorizada, subestimando o p-valor.
    tao_extremos = np.abs(rhos_por_permutacao) >= abs(rho_observado) - 1e-9
    contagem = np.count_nonzero(tao_extremos)
    total = len(rhos_por_permutacao)

    if exato:
        return float(contagem / total)
    # Correção +1/+1 de Monte Carlo (ver o docstring acima): só se aplica à amostragem,
    # nunca à enumeração exata, onde a identidade já garante contagem >= 1 por construção.
    return float((contagem + 1) / (total + 1))


def correlacao_spearman(x: np.ndarray, y: np.ndarray) -> ResultadoCorrelacao:
    """Spearman entre duas séries, com os dois p-valores (ver `ResultadoCorrelacao`) e um
    intervalo de confiança de 95% pelo z de Fisher ajustado (ver `_AJUSTE_BONETT_WRIGHT`) —
    este último mantido como diagnóstico, não como portão de significância (ver o
    docstring do módulo).

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
            p_valor_exato=float("nan"),
            intervalo_confianca=(float("nan"), float("nan")),
            n=n,
        )

    rho, p_valor = stats.spearmanr(x, y)
    rho = float(rho)
    p_valor_exato = p_valor_exato_spearman(x, y)

    if n < _N_MINIMO_PARA_INTERVALO:
        return ResultadoCorrelacao(
            rho=rho,
            p_valor=float(p_valor),
            p_valor_exato=p_valor_exato,
            intervalo_confianca=(float("nan"), float("nan")),
            n=n,
        )

    # `arctanh(±1)` diverge: um rho perfeito medido sobre seis pontos não significa que a
    # correlação populacional é exatamente 1 — significa que a amostra é pequena demais
    # para distinguir "1" de "muito perto de 1". Limitar rho a uma distância mínima de ±1
    # só para o cálculo do intervalo evita um intervalo infinito que esconderia essa
    # incerteza em vez de reportá-la. O valor do clip (1e-9) é ele mesmo o que faz este
    # intervalo saturar perto de `|rho|=1` independente de `n` — por isso ele NÃO decide
    # significância (`_significativos` usa `p_valor_exato`), só evita a divergência.
    rho_para_z = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
    z = np.arctanh(rho_para_z)
    erro_padrao = np.sqrt(_AJUSTE_BONETT_WRIGHT / (n - 3))
    z_baixo, z_alto = z - _Z_95 * erro_padrao, z + _Z_95 * erro_padrao

    return ResultadoCorrelacao(
        rho=rho,
        p_valor=float(p_valor),
        p_valor_exato=p_valor_exato,
        intervalo_confianca=(float(np.tanh(z_baixo)), float(np.tanh(z_alto))),
        n=n,
    )


class MagnitudeDoProxy(NamedTuple):
    """A força do sinal — o que Spearman, por desenho, não mede. Dois proxies podem
    concordar perfeitamente com a ORDEM da degradação real (`rho=1,0`) e ainda ter se
    deslocado por frações completamente diferentes do próprio intervalo — um empate em
    `rho` não é um empate em quanto o proxy realmente se moveu.

    `variacao_relativa` compara o movimento total ao próprio valor de partida — mas é
    instável quando esse valor de partida está perto de zero (um proxy que sai de 0,001 e
    chega a 0,03 "cresce 30 vezes" sem que isso signifique que ele ficou operacionalmente
    perceptível: `psi_do_score` no regime desta etapa é exatamente esse caso, ver o
    relatório da task). `passo_medio_absoluto` — a média do módulo do passo mês a mês — não
    tem essa instabilidade específica (não depende de dividir por um valor de partida perto
    de zero) e é a medida que `escolher_alarme` usa para desempatar quando `rho` sozinho não
    distingue.

    **Isso não a torna livre de limitação.** `confianca_media` e `taxa_de_aprovacao` são
    frações de verdade, matematicamente presas a `[0, 1]`; `psi_do_score` é uma divergência
    (Population Stability Index) sem teto matemático — nada a impede de crescer bem além de
    1 sob deslocamento extremo. Comparar `passo_medio_absoluto` entre os três continua
    sendo comparar unidades diferentes, mesmo sem a instabilidade de dividir por zero: sob
    um regime de drift mais severo que o desta etapa, `psi_do_score` poderia vencer o
    desempate por magnitude por causa da unidade em que PSI tende a crescer, não porque o
    sinal seja de fato mais forte — o mesmo tipo de acidente que motivou trocar a escolha
    alfabética de `escolher_alarme` por um critério medido. Neste run isso não acontece
    (`psi_do_score` não passa de 0,03 em nenhum mês, ver o relatório da task), e
    `passo_medio_absoluto` continua sendo a melhor das duas medidas disponíveis aqui — mas
    uma calibração formal de limiar por proxy (a mesma tarefa que precisa definir um
    limiar operacional para `taxa_de_aprovacao`) precisaria normalizar cada proxy pela
    própria escala ou convenção antes de comparar magnitude ENTRE proxies de natureza
    diferente, não só ao longo do tempo do mesmo proxy.
    """

    valor_inicial: float
    valor_final: float
    variacao_absoluta: float
    variacao_relativa: float
    passo_medio_absoluto: float


def magnitude_do_proxy(serie: np.ndarray) -> MagnitudeDoProxy:
    """Mede a força do sinal de uma série de proxy ao longo dos lotes — ver
    `MagnitudeDoProxy` sobre por que isso é uma pergunta diferente de `correlacao_spearman`.

    `variacao_relativa` é `nan` quando `serie[0] == 0`: dividir por zero não é um erro de
    execução aqui, é a ausência de uma base contra a qual medir variação relativa — o mesmo
    princípio de `correlacao_spearman` devolver `nan` em vez de inventar um número.

    Levanta `ValueError` com menos de dois pontos: sem um segundo ponto não há passo
    nenhum para medir.
    """
    serie = np.asarray(serie, dtype=float)
    if len(serie) < 2:
        raise ValueError("magnitude_do_proxy precisa de ao menos dois pontos")

    valor_inicial = float(serie[0])
    valor_final = float(serie[-1])
    variacao_absoluta = valor_final - valor_inicial
    variacao_relativa = variacao_absoluta / valor_inicial if valor_inicial != 0 else float("nan")
    passo_medio_absoluto = float(np.mean(np.abs(np.diff(serie))))

    return MagnitudeDoProxy(
        valor_inicial=valor_inicial,
        valor_final=valor_final,
        variacao_absoluta=variacao_absoluta,
        variacao_relativa=variacao_relativa,
        passo_medio_absoluto=passo_medio_absoluto,
    )


def correlacionar_proxies(
    lotes: dict[str, pd.DataFrame],
    modelo: Any,
    *,
    limiar: float = LIMIAR_PADRAO,
    bins: int = 10,
) -> dict[str, Any]:
    """Para cada lote de `lotes` (exceto o mais antigo, tratado como referência), calcula
    os proxies sem rótulo e a degradação real, correlaciona as duas séries (concordância de
    ordem) e mede a magnitude de cada proxy (força do sinal) — ver o docstring do módulo
    sobre por que as duas leituras são necessárias e nenhuma substitui a outra.

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

    magnitudes: dict[str, MagnitudeDoProxy] = {
        nome_proxy: magnitude_do_proxy(
            np.array([proxies_por_lote[nome][nome_proxy] for nome in nomes_dos_lotes])
        )
        for nome_proxy in nomes_dos_proxies
    }

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
        "magnitudes": magnitudes,
        "correlacoes": correlacoes,
    }


def _significativos(
    correlacoes: dict[str, dict[str, ResultadoCorrelacao]], *, campo: str
) -> dict[str, ResultadoCorrelacao]:
    """Os proxies cujo p-valor EXATO por permutação contra `campo` fica abaixo de
    `ALFA_SIGNIFICANCIA` — a mesma checagem que `escolher_alarme` e `proxies_no_topo`
    compartilham, isolada para que as duas nunca possam divergir sobre o que
    "significativo" significa aqui.

    Usa `p_valor_exato`, não o intervalo de confiança fechado: perto de `|rho|=1` o
    intervalo satura e exclui zero para QUALQUER `n`, inclusive `n=4` com um `rho`
    perfeito cujo p-valor exato (`2/4! = 0,0833`) não é significativo a 5% — ver o
    docstring do módulo. Um `rho` grande cujo p-valor exato não é pequeno não é evidência
    de que o proxy antecipa a degradação: com poucos pontos, é perfeitamente compatível
    com "nenhuma correlação populacional". Filtrar pelo ponto (`rho`) sozinho, ignorando a
    significância, seria exatamente o erro que o brief desta etapa pede para não cometer —
    reportar o número desapontador (ou a ausência de alarme) em vez de maquiar a incerteza.
    """
    candidatos = {
        nome: resultado[campo]
        for nome, resultado in correlacoes.items()
        if not np.isnan(resultado[campo].rho)
    }
    return {
        nome: resultado
        for nome, resultado in candidatos.items()
        if not np.isnan(resultado.p_valor_exato) and resultado.p_valor_exato < ALFA_SIGNIFICANCIA
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
    é melhor que o outro EM CONCORDÂNCIA DE ORDEM. `escolher_alarme` resolve esse empate por
    magnitude (ver o docstring dela) em vez de decidir por ordem alfabética; esta função
    continua existindo para que um relatório desta etapa nunca apresente o conjunto inteiro
    empatado como se fosse um só nome: ver `scripts/validar_proxies.py`, que imprime o
    conjunto inteiro, não só `escolher_alarme`.
    """
    significativos = _significativos(correlacoes, campo=campo)
    if not significativos:
        return ()
    maior = max(abs(resultado.rho) for resultado in significativos.values())
    return tuple(sorted(nome for nome, r in significativos.items() if abs(r.rho) == maior))


def escolher_alarme(
    correlacoes: dict[str, dict[str, ResultadoCorrelacao]],
    magnitudes: dict[str, MagnitudeDoProxy],
    *,
    campo: str = "degradacao_auc_roc",
) -> str | None:
    """Escolhe UM proxy — entre os empatados no maior `|rho|` significativo contra `campo`
    (`proxies_no_topo`), o de maior `passo_medio_absoluto` (a magnitude do sinal, ver
    `MagnitudeDoProxy`) — e devolve `None` quando nenhum qualifica, OU quando o empate
    persiste mesmo depois do desempate por magnitude.

    Concordância de ordem (`rho`) sozinha não decide um alarme: `proxies_no_topo` pode
    devolver mais de um nome exatamente empatado, e escolher entre eles por ordem alfabética
    — o que uma versão anterior desta função fazia — apresentaria um acidente de iteração
    como se fosse mérito medido. Usar a magnitude do sinal para desempatar é uma decisão
    genuína: entre dois proxies que concordam igualmente bem com a ORDEM da degradação, o
    que se moveu mais por lote é o que um operador consegue efetivamente enxergar antes de
    disparar um alarme. Se a magnitude também empatar exatamente (raro com dado contínuo,
    mas possível por construção em teste), a função se recusa a decidir por acaso — devolve
    `None`, a mesma resposta honesta que `proxies_no_topo` já dá quando nenhum proxy é
    significativo.

    `None` é uma resposta honesta e esperada, não uma falha da função: o achado "nenhum
    proxy antecipa a degradação com confiança" (ou "o empate não se resolve nem por
    magnitude") é tão válido quanto "este proxy antecipa" — ver `a-ideia-central.md`.
    """
    topo = proxies_no_topo(correlacoes, campo=campo)
    if not topo:
        return None
    if len(topo) == 1:
        return topo[0]

    maior_magnitude = max(magnitudes[nome].passo_medio_absoluto for nome in topo)
    vencedores = [nome for nome in topo if magnitudes[nome].passo_medio_absoluto == maior_magnitude]
    if len(vencedores) != 1:
        return None
    return vencedores[0]
