"""Reconstrói uma decisão de crédito para revisão (LGPD, art. 20; Lei nº 12.414, art. 5º, VI).

Roda com `make consultar-decisao ID=<id_decisao>`, no servidor que guarda o registro — de
propósito não existe rota HTTP para isto: o registro contém dado pessoal, e a API não tem
autenticação. Imprime a decisão registrada e, se o modelo publicado ainda é o mesmo que a
tomou (mesmo sha256), refaz a pontuação com as mesmas entradas e confere que o resultado
é idêntico. É essa conferência que torna a revisão uma reconstrução, e não uma leitura de
log: o revisor vê que a decisão saiu exatamente daquele modelo, com aquelas entradas.

Sai com código 1 se o id não existe, 2 se a reconstrução diverge.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import pandas as pd

from credito.config import get_settings
from credito.governanca.decisoes import RegistroJsonl, sha256_do_arquivo
from credito.model.train import carregar_modelo
from credito.schema import FEATURES


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--id", required=True, help="id_decisao devolvido pela API")
    argumentos = analisador.parse_args()
    settings = get_settings()

    decisao = RegistroJsonl(settings.decisoes_path).buscar(argumentos.id)
    if decisao is None:
        print(f"decisão {argumentos.id} não encontrada em {settings.decisoes_path}")
        return 1
    print(json.dumps(asdict(decisao), indent=2, ensure_ascii=False))

    sha_atual = sha256_do_arquivo(settings.model_path)
    if sha_atual != decisao.modelo_sha256:
        print(
            "\nO modelo publicado mudou desde a decisão (sha256 atual "
            f"{sha_atual[:12]}…, da decisão {decisao.modelo_sha256[:12]}…). A reconstrução "
            "exige o artefato antigo — recuperável pelo MLflow ou pelo histórico de treino."
        )
        return 0

    modelo, _ = carregar_modelo(settings.model_path)
    entrada = pd.DataFrame([decisao.features], columns=list(FEATURES))
    repontuada = float(modelo.predict_proba(entrada)[0, 1])
    reaprovada = repontuada < decisao.limiar
    identica = repontuada == decisao.probabilidade_inadimplencia and (
        reaprovada == decisao.aprovado
    )
    print(
        f"\nreconstrução com o mesmo modelo (sha256 {sha_atual[:12]}…): probabilidade "
        f"{repontuada!r}, {'aprovado' if reaprovada else 'recusado'} — "
        f"{'IDÊNTICA à registrada' if identica else 'DIVERGE da registrada'}"
    )
    return 0 if identica else 2


if __name__ == "__main__":
    sys.exit(main())
