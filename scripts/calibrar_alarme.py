"""Calibra o alarme ao vivo contra o campeão real e grava `reports/calibracao_do_alarme.json`.

Roda com `make calibrar-alarme` (também parte de `make reproduzir`). Duas medições:

1. **Taxa de aprovação por janela.** Para cada tamanho de janela, a nula — sorteando
   janelas da partição de teste sem drift (o mês 0 da simulação) — dá o limiar na taxa de
   falso alarme declarada; depois, em cada mês simulado de drift, mede que fração das
   janelas cai abaixo dele. A janela que a API de fato usa
   (`credito.api.metrics.JANELA_TAXA_DE_APROVACAO`) entra junto com janelas maiores, para
   que a escolha do tamanho seja feita com número.
2. **Nula do PSI do score** — antes medida por um script descartado, agora reproduzível:
   PSI entre os scores da Referência completa e reamostras do tamanho de um lote mensal.

Existe como script separado da suíte pela mesma razão dos outros de `scripts/`: nenhum
teste toca o dataset real nem o modelo publicado.
"""

from __future__ import annotations

import json
import logging
import sys

from credito.api.metrics import JANELA_TAXA_DE_APROVACAO, LIMIAR_DO_ALARME
from credito.config import get_settings
from credito.data.arff import ler_arff
from credito.data.prepare import limpar, separar
from credito.data.simulate import MESES, simular_producao
from credito.drift.calibration import distribuicao_nula_psi, limiar_empirico
from credito.model.train import carregar_modelo
from credito.monitoring.calibracao_do_alarme import (
    limiar_inferior,
    nula_da_taxa_de_aprovacao,
    poder_do_alarme,
)
from credito.schema import FEATURES

# Convenções declaradas, não medições: a fração das janelas sem drift que pode disparar o
# alarme, e quantas janelas sortear. A semente é a do projeto.
TAXA_DE_FALSO_ALARME = 0.01
N_AMOSTRAS = 5_000
# A janela que a API usa, a anterior (500) e uma maior — para que a escolha seja feita com
# número. `main` falha se a janela da API sair desta lista.
JANELAS = (500, 2_000, 5_000)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    referencia, _ = limpar(ler_arff(settings.dataset_path))
    teste = separar(
        referencia,
        test_size=settings.test_size,
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )["teste"]
    modelo, _ = carregar_modelo(settings.model_path)

    lotes = simular_producao(teste, meses=MESES, seed=settings.random_seed)
    probabilidades = {
        nome: modelo.predict_proba(lote[list(FEATURES)])[:, 1] for nome, lote in lotes.items()
    }
    sem_drift = probabilidades["mes_00"]
    com_drift = {nome: p for nome, p in probabilidades.items() if nome != "mes_00"}

    resultado: dict = {
        "taxa_de_falso_alarme": TAXA_DE_FALSO_ALARME,
        "n_amostras": N_AMOSTRAS,
        "semente": settings.random_seed,
        "janelas": {},
    }
    print(f"taxa de falso alarme por janela: {TAXA_DE_FALSO_ALARME:.0%} | {N_AMOSTRAS} janelas")
    cabecalho = "".join(f"{nome:>8s}" for nome in com_drift)
    print(f"{'janela':>7s} {'média':>7s} {'desvio':>7s} {'limiar':>7s} | poder: {cabecalho}")
    for janela in JANELAS:
        nula = nula_da_taxa_de_aprovacao(
            sem_drift, janela=janela, n_amostras=N_AMOSTRAS, seed=settings.random_seed
        )
        limiar = limiar_inferior(nula, taxa_de_falso_alarme=TAXA_DE_FALSO_ALARME)
        poder = poder_do_alarme(
            com_drift,
            limiar_de_alarme=limiar,
            janela=janela,
            n_amostras=N_AMOSTRAS,
            seed=settings.random_seed + 1,
        )
        resultado["janelas"][str(janela)] = {
            "media_sem_drift": float(nula.mean()),
            "desvio_sem_drift": float(nula.std()),
            "limiar": limiar,
            "poder_por_mes": poder,
        }
        linha = "".join(f"{valor:8.1%}" for valor in poder.values())
        print(f"{janela:7d} {nula.mean():7.4f} {nula.std():7.4f} {limiar:7.4f} | {linha}")

    scores_referencia = modelo.predict_proba(referencia[list(FEATURES)])[:, 1]
    nula_psi = distribuicao_nula_psi(
        scores_referencia,
        n_amostras=N_AMOSTRAS,
        tamanho_lote=len(teste),
        bins=10,
        seed=settings.random_seed,
    )
    resultado["nula_psi_do_score"] = {
        "p95": limiar_empirico(nula_psi, percentil=95),
        "p99": limiar_empirico(nula_psi, percentil=99),
        "maximo": float(nula_psi.max()),
        "tamanho_lote": len(teste),
        "linhas_referencia": len(referencia),
    }
    psi = resultado["nula_psi_do_score"]
    print(
        f"\nnula do PSI do score (Referência {len(referencia)} linhas, lotes de {len(teste)}): "
        f"p95 {psi['p95']:.6f} · p99 {psi['p99']:.6f} · máximo {psi['maximo']:.6f}"
    )

    caminho = settings.reports_dir / "calibracao_do_alarme.json"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"\nrelatório gravado em {caminho}")

    # Guarda: o limiar publicado em `credito.api.metrics` (e na regra do Prometheus) tem de
    # ser o que a medição de hoje dá para a janela que a API usa. Um retreino do campeão
    # muda a distribuição de score e, com ela, o limiar — e só esta conferência percebe.
    if str(JANELA_TAXA_DE_APROVACAO) not in resultado["janelas"]:
        print(f"JANELA_TAXA_DE_APROVACAO = {JANELA_TAXA_DE_APROVACAO} não foi medida")
        return 1
    medido = round(resultado["janelas"][str(JANELA_TAXA_DE_APROVACAO)]["limiar"], 4)
    if medido != LIMIAR_DO_ALARME:
        print(
            f"LIMIAR_DO_ALARME publicado ({LIMIAR_DO_ALARME}) diverge do medido hoje ({medido}) "
            f"para a janela de {JANELA_TAXA_DE_APROVACAO}: atualize credito.api.metrics e "
            "docker/prometheus/alertas.yml"
        )
        return 1
    print(f"limiar publicado ({LIMIAR_DO_ALARME}) confere com o medido para a janela da API")
    return 0


if __name__ == "__main__":
    sys.exit(main())
