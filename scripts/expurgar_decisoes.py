"""Aplica a retenção do registro de decisão: remove o que passou do prazo.

Roda com `make expurgar-decisoes`. O prazo (`RETENCAO_ANOS`) e o porquê estão em
`credito.governanca.decisoes` e em `docs/governanca.md`. Feito para rodar periodicamente
(cron, job agendado): a LGPD manda eliminar o dado ao fim do tratamento (art. 16), e um
prazo escrito num documento que nada executa não elimina nada.
"""

from __future__ import annotations

from datetime import UTC, datetime

from credito.config import get_settings
from credito.governanca.decisoes import RETENCAO_ANOS, RegistroJsonl, limite_de_retencao


def main() -> None:
    settings = get_settings()
    limite = limite_de_retencao(datetime.now(UTC))
    removidas = RegistroJsonl(settings.decisoes_path).expurgar(antes_de=limite)
    print(
        f"retenção de {RETENCAO_ANOS} anos: {removidas} decisão(ões) anterior(es) a "
        f"{limite.date().isoformat()} removida(s) de {settings.decisoes_path}"
    )


if __name__ == "__main__":
    main()
