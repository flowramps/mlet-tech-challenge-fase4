"""Roda o entregável central da Etapa 3 contra o campeão real: quanto cada proxy sem
rótulo (`credito.monitoring.proxies`) antecipa a degradação real medida contra o campeão
publicado, nos seis lotes simulados sobre a partição de teste real.

Roda com `make validar-proxies`. Carrega o campeão real (`models/model.joblib`) e a
Referência real (`data/raw/GiveMeSomeCredit.arff`), separa a mesma partição de teste que o
treino usou (`separar()`, os mesmos `test_size`/`validation_size`/`random_seed` de
`config.py` — a mesma partição contra a qual `scripts/verificar_degradacao.py` já mede a
degradação real), simula os seis meses sobre ela e chama
`credito.monitoring.validacao_de_proxy.correlacionar_proxies`.

Existe como script separado da suíte automatizada pela mesma razão de
`verificar_degradacao.py`: nenhum teste toca o dataset real nem o modelo publicado.
"""

from __future__ import annotations

import json
import logging

from credito.config import get_settings
from credito.data.arff import ler_arff
from credito.data.prepare import limpar, separar
from credito.data.simulate import MESES, simular_producao
from credito.model.train import carregar_modelo
from credito.monitoring.validacao_de_proxy import correlacionar_proxies, proxies_no_topo


def _imprimir_tabela(resultado: dict) -> None:
    proxies = sorted(resultado["correlacoes"])
    campos = ("degradacao_auc_roc", "degradacao_lift_acima_do_piso")

    print(f"referência: {resultado['referencia']} | lotes correlacionados: {resultado['lotes']}")
    print()
    print(f"{'proxy':18s} {'campo':28s} {'rho':>8s} {'ic 95%':>22s} {'p-valor':>9s} {'n':>3s}")
    for proxy in proxies:
        for campo in campos:
            r = resultado["correlacoes"][proxy][campo]
            ic = f"[{r.intervalo_confianca[0]:+.4f}, {r.intervalo_confianca[1]:+.4f}]"
            print(f"{proxy:18s} {campo:28s} {r.rho:+8.4f} {ic:>22s} {r.p_valor:9.4f} {r.n:3d}")

    print()
    for campo in campos:
        # `proxies_no_topo` (não `escolher_alarme`) porque um empate exato de |rho| — que
        # a própria simulação monotônica desta etapa pode produzir — precisa aparecer
        # inteiro no relatório, não resolvido para um único nome em silêncio.
        topo = proxies_no_topo(resultado["correlacoes"], campo=campo)
        if not topo:
            print(f"nenhum proxy significativo contra {campo}")
        elif len(topo) == 1:
            print(f"proxy recomendado contra {campo}: {topo[0]!r}")
        else:
            print(f"empate contra {campo} entre: {list(topo)!r} — mesmo |rho| nos {len(topo)}")

    print()
    print(f"{'lote':8s} {'auc_roc':>8s} {'lift_piso':>10s} {'deg_auc_roc':>12s} {'deg_lift':>9s}")
    for nome in resultado["lotes"]:
        d = resultado["degradacao_por_lote"][nome]
        print(
            f"{nome:8s} {d['auc_roc']:8.4f} {d['lift_acima_do_piso']:10.4f} "
            f"{d['degradacao_auc_roc']:12.4f} {d['degradacao_lift_acima_do_piso']:9.4f}"
        )


def _serializavel(resultado: dict) -> dict:
    """Troca cada `ResultadoCorrelacao` (NamedTuple) por dict — `json.dumps` não sabe
    serializar NamedTuple com os campos nomeados preservados por padrão."""
    return {
        **resultado,
        "correlacoes": {
            proxy: {campo: r._asdict() for campo, r in campos.items()}
            for proxy, campos in resultado["correlacoes"].items()
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()

    referencia, _ = limpar(ler_arff(settings.dataset_path))
    particoes = separar(
        referencia,
        test_size=settings.test_size,
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )
    teste = particoes["teste"]

    modelo, metadados = carregar_modelo(settings.model_path)
    candidato = metadados.get("candidato", metadados) if isinstance(metadados, dict) else metadados

    lotes = simular_producao(teste, meses=MESES, seed=settings.random_seed)
    resultado = correlacionar_proxies(lotes, modelo)

    print(f"campeão: {candidato} | partição de teste: {len(teste)} linhas")
    _imprimir_tabela(resultado)

    caminho = settings.reports_dir / "validacao_de_proxy.json"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(_serializavel(resultado), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"relatório gravado em {caminho}")


if __name__ == "__main__":
    main()
