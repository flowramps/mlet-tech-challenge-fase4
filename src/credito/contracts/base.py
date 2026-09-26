"""Fronteira do contrato de dados.

O ``Validator`` é um Protocol: as regras são declaradas uma vez em ``rules.py``, e a
biblioteca que as executa fica atrás desta interface. Trocar Pandera por outra
implementação não reescreve nenhuma regra nem nenhum teste de comportamento.

A distinção que o módulo sustenta: contrato responde "este dado é **válido**?", e é uma
falha dura quando a resposta é não. Detecção de drift responde "este dado é **o mesmo de
antes**?", e um lote com drift passa aqui — dado deslocado continua sendo dado válido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


# N818 pede sufixo "Error"; o nome fica em português e descreve o desfecho ("contrato
# violado"), não a mecânica de exceção — a semântica do domínio vale mais que a
# convenção de nomenclatura aqui, e é uma escolha deliberada, não um descuido.
class ContratoViolado(RuntimeError):  # noqa: N818
    """Lote reprovado no contrato de dados. A ingestão deve ser **bloqueada**."""


@dataclass(frozen=True)
class Violacao:
    """Uma regra desrespeitada, com quantas linhas a desrespeitaram."""

    regra: str
    coluna: str
    linhas: int
    indices: tuple[int, ...] = field(default=())

    def __str__(self) -> str:
        return f"{self.regra} ({self.coluna}): {self.linhas} linha(s)"


@dataclass(frozen=True)
class ValidationResult:
    """Desfecho da validação de um lote."""

    total: int
    violacoes: tuple[Violacao, ...]

    @property
    def valido(self) -> bool:
        return not self.violacoes

    @property
    def linhas_reprovadas(self) -> int:
        """Linhas distintas reprovadas — uma linha que viola duas regras conta uma vez.

        A contagem é exata quando toda ``Violacao`` carrega ``indices``. Duplicata exata é
        avaliada sobre o lote inteiro mas ainda aponta as linhas repetidas; a única
        violação do contrato hoje que chega sem índice é ``coluna_ausente``, porque não há
        linha culpada quando o upstream deixou de mandar a coluna — o defeito é do lote, e
        ela chega com ``linhas`` igual ao total. Quando alguma violação chega assim, sem
        índice, o retorno vira um **piso**: o maior valor defensável sem inventar índice
        que a biblioteca de validação nunca forneceu. Duas violações sem índice não se
        somam por isso — não há como saber se cobrem linhas distintas ou as mesmas, e
        somar arriscaria contar a mesma linha duas vezes, o erro oposto que esta
        propriedade existe para evitar.
        """
        if not self.violacoes:
            return 0
        indices: set[int] = set()
        sem_indice = 0
        for violacao in self.violacoes:
            if violacao.indices:
                indices.update(violacao.indices)
            else:
                sem_indice = max(sem_indice, violacao.linhas)
        return max(len(indices), sem_indice)

    def erguer(self) -> None:
        """Converte um resultado reprovado em exceção."""
        if self.valido:
            return
        detalhe = "; ".join(str(violacao) for violacao in self.violacoes)
        raise ContratoViolado(
            f"{self.linhas_reprovadas} de {self.total} linha(s) reprovadas — {detalhe}"
        )


class Validator(Protocol):
    """Executor de contrato. A ingestão fala com isto, nunca com Pandera diretamente."""

    def validar(self, frame: pd.DataFrame) -> ValidationResult: ...
