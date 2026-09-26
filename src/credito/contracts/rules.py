"""As regras do contrato, declaradas uma única vez e independentes de biblioteca.

Cada regra nasceu de um defeito **medido** no dataset real, não de uma suposição. Os
números em cada comentário são a evidência de que a regra tem trabalho a fazer.
"""

from __future__ import annotations

from dataclasses import dataclass

from credito.data.prepare import (
    ATRASO_MAXIMO_PLAUSIVEL,
    COLUNAS_DE_ATRASO,
    DEBT_RATIO_MAXIMO,
    IDADE_MAXIMA,
    IDADE_MINIMA,
)


@dataclass(frozen=True)
class Regra:
    """Uma exigência do contrato, com o porquê registrado junto."""

    nome: str
    coluna: str
    descricao: str
    motivo: str


REGRAS: tuple[Regra, ...] = (
    Regra(
        nome="renda_nao_nula",
        coluna="MonthlyIncome",
        descricao="MonthlyIncome não pode ser nulo e precisa ser maior ou igual a zero",
        motivo=(
            "29.731 de 150.000 linhas do arquivo bruto (19,8%) vêm sem renda. Nelas, "
            "DebtRatio guarda a dívida bruta em vez da razão — 90% ficam acima de 10, "
            "contra 1,75% entre as que têm renda. Barrar a renda nula corrige os dois "
            "campos de uma vez, porque é um defeito só."
        ),
    ),
    Regra(
        nome="idade_plausivel",
        coluna="age",
        descricao=f"age entre {IDADE_MINIMA} e {IDADE_MAXIMA}",
        motivo=(
            "Concessão de crédito a menor de idade é vedada, e o intervalo de age "
            "observado no arquivo bruto vai de 0 a 109 — o mesmo levantamento que "
            "encontrou o registro com idade 0 encontrou o extremo superior. O teto de "
            "110 barra erro de digitação sem reprovar esse cliente idoso legítimo."
        ),
    ),
    Regra(
        nome="sem_duplicatas",
        coluna="*",
        descricao="o lote não pode conter linhas idênticas",
        motivo=(
            "609 linhas do arquivo bruto são duplicatas exatas. Duplicata infla o peso de "
            "um perfil no treino e distorce qualquer contagem de distribuição."
        ),
    ),
    Regra(
        nome="atraso_plausivel",
        coluna=", ".join(COLUNAS_DE_ATRASO),
        descricao=f"contadores de atraso menores ou iguais a {ATRASO_MAXIMO_PLAUSIVEL}",
        motivo=(
            "269 linhas trazem os códigos 96 e 98 nas três colunas de atraso. São marcas "
            "de ausência herdadas da coleta, não contagens: um cliente com 98 atrasos de "
            "90 dias não existe. O valor é do tipo certo e semanticamente impossível — "
            "exatamente o que um contrato precisa interceptar e um schema de tipos não pega."
        ),
    ),
    Regra(
        nome="dependentes_nao_nulo",
        coluna="NumberOfDependents",
        descricao="NumberOfDependents não pode ser nulo e precisa ser maior ou igual a zero",
        motivo=(
            "3.924 linhas do arquivo bruto (2,6%) vêm sem o número de dependentes. Das "
            "quais 100% também vêm sem MonthlyIncome — não são três defeitos "
            "independentes, é um único evento de ingestão quebrado com vários sintomas."
        ),
    ),
    Regra(
        nome="razao_divida_plausivel",
        coluna="DebtRatio",
        descricao=f"DebtRatio entre 0 e {DEBT_RATIO_MAXIMO}",
        motivo=(
            "Rede de segurança para o caso de a renda vir preenchida mas incorreta: entre "
            "os registros com renda válida, 1,75% ainda passam de 10."
        ),
    ),
)
