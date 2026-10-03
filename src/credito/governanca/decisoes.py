"""Registro de decisão de crédito — o que dá objeto ao direito de revisão.

A LGPD (art. 20) e a Lei do Cadastro Positivo (art. 5º, VI) garantem ao titular a revisão
de uma decisão tomada unicamente por meio automatizado. Revisar exige reconstruir: com que
entradas, com que modelo, com que limiar a decisão saiu. Sem registro, a API responde e
esquece, e o direito fica sem objeto.

**O que o registro guarda e o que não guarda.** Guarda o vetor de features — que é dado
pessoal: idade e renda exatas apontam sozinhas 41% das pessoas da Referência (ver
`credito.governanca.privacidade`) —, a probabilidade, a decisão, o limiar, o candidato e o
sha256 do arquivo do modelo. Não guarda identificador nenhum: o `id_decisao` é gerado aqui,
devolvido ao sistema de origem na resposta, e é a origem que liga esse id à pessoa. A
retenção é de `RETENCAO_ANOS`, com expurgo executável (`expurgar`).

**Append-only em JSONL**, o mesmo formato do histórico de treino: uma linha por decisão,
nada reescrito no caminho normal. O expurgo é a única operação que reescreve, e o faz por
arquivo temporário + `os.replace`, para que uma falha no meio nunca deixe o registro
truncado.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

# Prazo de retenção do registro de decisão. Decisão deste projeto, não número da lei: a lei
# dá o critério (fim da finalidade, LGPD art. 15) e os tetos (5 anos para informação
# negativa no CDC, art. 43, §1º); o prazo concreto está argumentado em
# `docs/governanca.md`, seção de retenção.
RETENCAO_ANOS = 5


@dataclass(frozen=True)
class Decisao:
    """Uma decisão de crédito, com tudo o que reconstruí-la exige."""

    id_decisao: str
    registrada_em: str  # ISO 8601, sempre com fuso
    features: dict[str, float]
    probabilidade_inadimplencia: float
    aprovado: bool
    limiar: float
    candidato: str
    modelo_sha256: str

    def __post_init__(self) -> None:
        if datetime.fromisoformat(self.registrada_em).tzinfo is None:
            raise ValueError(f"registrada_em sem fuso horário: {self.registrada_em!r}")

    @property
    def momento(self) -> datetime:
        return datetime.fromisoformat(self.registrada_em)


class RegistroDeDecisoes(Protocol):
    """O que a API precisa de um registro: gravar uma decisão, ou falhar alto."""

    def registrar(self, decisao: Decisao) -> None: ...


def sha256_do_arquivo(caminho: Path) -> str:
    """Impressão digital do arquivo do modelo: é o que liga uma decisão registrada ao
    modelo exato que a tomou, mesmo depois de um retreino trocar o campeão."""
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def limite_de_retencao(agora: datetime, *, anos: int = RETENCAO_ANOS) -> datetime:
    """O instante antes do qual uma decisão já venceu o prazo de retenção."""
    try:
        return agora.replace(year=agora.year - anos)
    except ValueError:
        # 29 de fevereiro num ano sem ele: o prazo vence no último dia de fevereiro.
        return agora.replace(year=agora.year - anos, day=28)


class RegistroJsonl:
    """Registro append-only em JSONL, seguro para escrita concorrente dentro do processo."""

    def __init__(self, caminho: Path) -> None:
        self.caminho = Path(caminho)
        self._trava = threading.Lock()

    def registrar(self, decisao: Decisao) -> None:
        """Grava a decisão e só retorna depois de o sistema operacional confirmar a
        escrita (`fsync`): quem chama usa o retorno como garantia de que a decisão é
        reconstruível, e uma linha perdida num buffer quebraria essa garantia."""
        linha = json.dumps(asdict(decisao), ensure_ascii=False) + "\n"
        with self._trava:
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            with self.caminho.open("a", encoding="utf-8") as arquivo:
                arquivo.write(linha)
                arquivo.flush()
                os.fsync(arquivo.fileno())

    def _ler(self) -> list[Decisao]:
        if not self.caminho.exists():
            return []
        with self.caminho.open(encoding="utf-8") as arquivo:
            return [Decisao(**json.loads(linha)) for linha in arquivo if linha.strip()]

    def buscar(self, id_decisao: str) -> Decisao | None:
        """A decisão com este id, ou `None` se não houver — nunca uma aproximação."""
        with self._trava:
            return next((d for d in self._ler() if d.id_decisao == id_decisao), None)

    def expurgar(self, *, antes_de: datetime) -> int:
        """Remove as decisões registradas estritamente antes de `antes_de` e devolve
        quantas saíram. Uma decisão exatamente no limite ainda está no prazo."""
        with self._trava:
            decisoes = self._ler()
            mantidas = [d for d in decisoes if d.momento >= antes_de]
            removidas = len(decisoes) - len(mantidas)
            if removidas == 0:
                return 0
            descritor, temporario = tempfile.mkstemp(
                dir=self.caminho.parent, prefix=".expurgo-", suffix=".jsonl"
            )
            with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
                for decisao in mantidas:
                    arquivo.write(json.dumps(asdict(decisao), ensure_ascii=False) + "\n")
                arquivo.flush()
                os.fsync(arquivo.fileno())
            os.replace(temporario, self.caminho)
            return removidas
