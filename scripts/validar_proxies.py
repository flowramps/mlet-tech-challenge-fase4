"""Roda o entregável central da Etapa 3 contra o campeão real: quanto cada proxy sem
rótulo (`credito.monitoring.proxies`) antecipa a degradação real medida contra o campeão
publicado, nos seis lotes simulados sobre a partição de teste real.

Roda com `make validar-proxies`. Carrega o campeão real (`models/model.joblib`) e a
Referência real (`data/raw/GiveMeSomeCredit.arff`), separa a mesma partição de teste que o
treino usou (`separar()`, os mesmos `test_size`/`validation_size`/`random_seed` de
`config.py` — a mesma partição contra a qual `scripts/verificar_degradacao.py` já mede a
degradação real), simula os seis meses sobre ela e chama
`credito.monitoring.validacao_de_proxy.correlacionar_proxies`.

Imprime as duas leituras separadas que a etapa exige: concordância de ORDEM (Spearman, com
o p-valor exato por permutação que decide significância — não o intervalo de confiança
fechado, que satura perto de `|rho|=1` independente de `n`) e força do SINAL (magnitude,
`passo_medio_absoluto`) — um proxy pode empatar com outro na primeira leitura e perder na
segunda, e é a segunda que `escolher_alarme` usa para desempatar.

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
from credito.monitoring.validacao_de_proxy import (
    correlacionar_proxies,
    escolher_alarme,
    proxies_no_topo,
)

_CAMPOS_DE_DEGRADACAO = ("degradacao_auc_roc", "degradacao_lift_acima_do_piso")


def _imprimir_tabela_de_correlacao(resultado: dict) -> None:
    proxies = sorted(resultado["correlacoes"])

    print(f"referência: {resultado['referencia']} | lotes correlacionados: {resultado['lotes']}")
    print()
    print("concordância de ordem (Spearman) — o p-valor EXATO por permutação decide significância:")
    cabecalho = (
        f"{'proxy':18s} {'campo':28s} {'rho':>8s} {'p exato':>9s} {'p assint':>9s} {'n':>3s}"
    )
    print(cabecalho)
    for proxy in proxies:
        for campo in _CAMPOS_DE_DEGRADACAO:
            r = resultado["correlacoes"][proxy][campo]
            print(
                f"{proxy:18s} {campo:28s} {r.rho:+8.4f} {r.p_valor_exato:9.4f} "
                f"{r.p_valor:9.4f} {r.n:3d}"
            )


def _imprimir_tabela_de_magnitude(resultado: dict) -> None:
    print()
    print("força do sinal (magnitude) — o que a concordância de ordem, sozinha, não mede:")
    print(f"{'proxy':18s} {'inicial':>9s} {'final':>9s} {'var. rel.':>10s} {'passo médio':>12s}")
    for proxy, m in sorted(resultado["magnitudes"].items()):
        var_rel = (
            f"{m.variacao_relativa:+.1%}" if m.variacao_relativa == m.variacao_relativa else "n/d"
        )
        print(
            f"{proxy:18s} {m.valor_inicial:9.4f} {m.valor_final:9.4f} {var_rel:>10s} "
            f"{m.passo_medio_absoluto:12.4f}"
        )


def _imprimir_recomendacao(resultado: dict) -> None:
    print()
    for campo in _CAMPOS_DE_DEGRADACAO:
        topo = proxies_no_topo(resultado["correlacoes"], campo=campo)
        alarme = escolher_alarme(resultado["correlacoes"], resultado["magnitudes"], campo=campo)
        if not topo:
            print(f"nenhum proxy significativo (p exato < 0,05) contra {campo}")
        elif len(topo) == 1:
            print(f"proxy recomendado contra {campo}: {topo[0]!r}")
        elif alarme is not None:
            print(
                f"empate em concordância de ordem contra {campo} entre {list(topo)!r} — "
                f"desempatado por magnitude (passo médio absoluto): {alarme!r}"
            )
        else:
            print(
                f"empate contra {campo} entre {list(topo)!r}, e a magnitude também empata — "
                "sem recomendação única (ver o relatório da task)"
            )


def _imprimir_tabela_de_degradacao(resultado: dict) -> None:
    print()
    print(f"{'lote':8s} {'auc_roc':>8s} {'lift_piso':>10s} {'deg_auc_roc':>12s} {'deg_lift':>9s}")
    for nome in resultado["lotes"]:
        d = resultado["degradacao_por_lote"][nome]
        print(
            f"{nome:8s} {d['auc_roc']:8.4f} {d['lift_acima_do_piso']:10.4f} "
            f"{d['degradacao_auc_roc']:12.4f} {d['degradacao_lift_acima_do_piso']:9.4f}"
        )


def _serializavel(resultado: dict) -> dict:
    """Troca cada `ResultadoCorrelacao`/`MagnitudeDoProxy` (NamedTuple) por dict —
    `json.dumps` não sabe serializar NamedTuple com os campos nomeados preservados por
    padrão."""
    return {
        **resultado,
        "correlacoes": {
            proxy: {campo: r._asdict() for campo, r in campos.items()}
            for proxy, campos in resultado["correlacoes"].items()
        },
        "magnitudes": {
            proxy: magnitude._asdict() for proxy, magnitude in resultado["magnitudes"].items()
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
    _imprimir_tabela_de_correlacao(resultado)
    _imprimir_tabela_de_magnitude(resultado)
    _imprimir_recomendacao(resultado)
    _imprimir_tabela_de_degradacao(resultado)

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
