"""Obtenção do dataset de referência, com cache em disco e verificação de integridade."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_BLOCO = 1 << 20


def _md5(caminho: Path) -> str:
    # Uso de integridade, não criptográfico: o md5 aqui só responde "o arquivo em disco é
    # o mesmo que o publicado", e é o checksum que o OpenML divulga.
    digestor = hashlib.md5()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(_BLOCO), b""):
            digestor.update(bloco)
    return digestor.hexdigest()


def baixar_dataset(
    destino: Path,
    url: str,
    md5_esperado: str,
    *,
    force: bool = False,
    transporte: httpx.BaseTransport | None = None,
) -> Path:
    """Baixa o dataset para ``destino``, reaproveitando o cache quando ele está íntegro.

    ``transporte`` existe para os testes injetarem um ``MockTransport``: a suíte não pode
    depender de rede, e um teste que baixa 7 MB do OpenML a cada execução é um teste que
    logo é desligado.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    if destino.exists() and not force:
        if _md5(destino) == md5_esperado:
            logger.info("cache íntegro em %s", destino)
            return destino
        logger.warning("cache corrompido em %s — refazendo o download", destino)

    with (
        httpx.Client(transport=transporte, follow_redirects=True, timeout=120.0) as cliente,
        cliente.stream("GET", url) as resposta,
    ):
        resposta.raise_for_status()
        with destino.open("wb") as arquivo:
            for bloco in resposta.iter_bytes(_BLOCO):
                arquivo.write(bloco)

    obtido = _md5(destino)
    if obtido != md5_esperado:
        destino.unlink()
        raise ValueError(f"md5 divergente: esperado {md5_esperado}, obtido {obtido}")

    logger.info("dataset gravado em %s (%.1f MB)", destino, destino.stat().st_size / 1e6)
    return destino


def main() -> None:
    from credito.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    baixar_dataset(settings.dataset_path, settings.dataset_url, settings.dataset_md5)


if __name__ == "__main__":
    main()
