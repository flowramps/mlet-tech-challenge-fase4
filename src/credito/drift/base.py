"""Fronteira da detecção de drift.

``DriftDetector`` é um Protocol: as regras de classificação (``classificar``, os cortes
declarados em ``config.py``) ficam declaradas uma vez, e a biblioteca que calcula PSI e KS —
hoje ``drift/statistics.py``, talvez Evidently mais adiante — fica atrás desta interface.
Mesma forma que ``contracts/base.py``, e pelo mesmo motivo.

**A distinção que este módulo preserva, continuação direta da Etapa 1:** o contrato
(``contracts/base.py``) responde "este dado é válido?", e reprovar é uma falha dura —
``ContratoViolado`` bloqueia a ingestão porque dado inválido não pode entrar. Drift responde
uma pergunta diferente, "este lote se parece com o que o modelo aprendeu?", e a resposta
"não" **não é uma falha**: dado deslocado continua sendo dado válido, legítimo o bastante
para ter passado no contrato, e decidir se o deslocamento justifica retreinar é uma decisão
humana, não automática. Por isso este módulo não declara nenhuma exceção — ao contrário de
``ContratoViolado``, um ``DriftReport`` nunca é erguido, só lido. O paralelo mais próximo na
Etapa 1 é a diferença entre ``QualityGateError`` (falha dura do pipeline de treino) e
``ModelNotPromoted`` (desfecho normal, não um defeito); aqui a distinção é ainda mais
direta, porque o lado "alerta" nem chega a ser uma exceção — é um dado que se resume e se
inspeciona.

**A armadilha de direção que este módulo institucionaliza:** PSI e KS medem drift em
sentidos opostos.

| Teste | O valor mede  | Limiares     | Direção            |
|-------|---------------|--------------|--------------------|
| PSI   | divergência   | 0,10 / 0,25  | **maior** = mais drift |
| KS    | p-valor       | 0,05         | **menor** = mais drift |

``DriftDeFeature`` carrega os dois lado a lado, e os nomes dos campos —
``psi_divergencia``, ``ks_p_valor`` — carregam a direção junto: um chamador que quisesse
trocar os dois, ou colar a estatística do KS onde o p-valor é esperado, erraria o nome, não
só a posição. A mesma razão de ``ResultadoKS`` ser um ``NamedTuple`` de campos nomeados em
``drift/statistics.py``, um nível acima.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pandas as pd

from credito.config import get_settings


class Severidade(Enum):
    """Nível de alerta de drift, do menos para o mais grave.

    Os valores inteiros existem só para permitir comparação por gravidade (``max`` em
    ``DriftReport.severidade_maxima``); não são um limiar quantificado como
    ``psi_atencao``/``psi_critico`` — são posição na escala, nada mais.
    """

    ESTAVEL = 0
    ATENCAO = 1
    CRITICO = 2


def classificar(psi: float) -> Severidade:
    """Classifica um PSI já calculado segundo os cortes de convenção em ``config.py``.

    Fronteiras inclusivas no limite inferior de cada banda — decidido, não acidental:
    ``psi == psi_atencao`` já conta como ``ATENCAO`` (não ``ESTAVEL``), e
    ``psi == psi_critico`` já conta como ``CRITICO`` (não ``ATENCAO``). É a mesma convenção
    que ``drift/statistics.py`` já documenta na docstring de ``psi()`` ("< 0,10 estável ·
    0,10-0,25 mudança moderada · >= 0,25 mudança material"); este módulo só a aplica.
    """
    settings = get_settings()
    if psi >= settings.psi_critico:
        return Severidade.CRITICO
    if psi >= settings.psi_atencao:
        return Severidade.ATENCAO
    return Severidade.ESTAVEL


@dataclass(frozen=True)
class DriftDeFeature:
    """PSI e KS de uma única feature, mais a severidade que ``classificar`` decidiu a
    partir do PSI.

    ``psi_divergencia`` e ``ks_p_valor`` apontam em sentidos opostos (ver o docstring do
    módulo) — os nomes carregam essa direção para que a confusão vire erro de digitação
    óbvio, não um alarme invertido que só alguém desconfiado do painel nota.
    """

    feature: str
    psi_divergencia: float
    ks_p_valor: float
    severidade: Severidade


@dataclass(frozen=True)
class DriftReport:
    """Desfecho da checagem de drift de um lote inteiro: uma ``DriftDeFeature`` por
    feature medida. Nunca é erguido como exceção — só existe para ser lido, resumido e
    inspecionado (ver o docstring do módulo)."""

    lote: str
    features: tuple[DriftDeFeature, ...]

    @property
    def severidade_maxima(self) -> Severidade:
        """A pior severidade entre as features do lote.

        Um relatório sem nenhuma feature devolve ``ESTAVEL`` — decidido, não uma exceção
        por tirar o máximo de uma sequência vazia. O raciocínio espelha
        ``ValidationResult.valido`` em ``contracts/base.py``: na ausência de qualquer
        feature medida não há evidência de deslocamento nenhuma, e o piso seguro para
        reportar é "nada indica drift".
        """
        if not self.features:
            return Severidade.ESTAVEL
        return max((feature.severidade for feature in self.features), key=lambda s: s.value)

    @property
    def features_em_drift(self) -> tuple[DriftDeFeature, ...]:
        """As features cuja severidade passou de ``ESTAVEL`` — o subconjunto que vale a
        pena mostrar num painel, na ordem em que chegaram em ``features``."""
        return tuple(
            feature for feature in self.features if feature.severidade is not Severidade.ESTAVEL
        )


class DriftDetector(Protocol):
    """Executor de detecção de drift. O relatório fala com isto, nunca com Evidently nem
    com ``drift/statistics.py`` diretamente — trocar a biblioteca que calcula PSI e KS não
    reescreve nenhuma regra de severidade nem nenhum teste de comportamento."""

    def detectar(
        self, referencia: pd.DataFrame, atual: pd.DataFrame, *, lote: str
    ) -> DriftReport: ...
