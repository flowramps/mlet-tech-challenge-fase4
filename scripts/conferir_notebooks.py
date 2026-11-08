"""Confere que as saídas salvas dos notebooks ainda batem com uma reexecução.

Roda com `make conferir-notebooks` e na CI. Os notebooks são versionados com as saídas — é o
que permite lê-los no GitHub —, e o README cita números tirados delas. Se o modelo ou o
código mudar e ninguém reexecutar os notebooks, esses números ficam velhos em silêncio: foi o
que aconteceu antes da 1.2.0, quando os notebooks tinham sido gerados com um modelo
retreinado em outro sistema operacional.

A comparação é sobre o **texto** das saídas — o que um notebook imprime e as tabelas que ele
mostra —, não sobre o arquivo inteiro. O arquivo carrega coisas que mudam a cada execução sem
que nenhum resultado mude: o horário de execução de cada célula, os bytes das imagens e o
endereço de memória que o pandas imprime para um `Styler`. Comparar o arquivo inteiro
falharia sempre; comparar o texto falha exatamente quando um número muda.

A reexecução vai para um diretório temporário: o notebook versionado nunca é reescrito por
esta conferência.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# A análise causal (02) fica de fora por padrão: leva cerca de 5 minutos, quase todos em
# bootstrap e repetições de cenário, e dobraria o tempo da CI. As outras quatro, juntas,
# levam cerca de 1,5 minuto.
PADRAO = ("01_eda", "03_baseline", "04_equidade", "05_explicabilidade")

# `<pandas.io.formats.style.Styler at 0x25c1c0bdeb0>`: o endereço muda a cada processo.
_ENDERECO = re.compile(r" at 0x[0-9a-fA-F]+")


def saidas_de_texto(notebook: dict) -> list[str]:
    """As linhas de texto que as células de código mostram, prefixadas pela célula.

    Entram o que a célula imprime (`stdout`), o texto do que ela exibe (`text/plain` de
    resultados e displays — inclusive tabelas) e erros. Ficam de fora imagens, HTML (que
    repete o `text/plain` das tabelas), `stderr` (avisos de biblioteca, que mudam com o
    ambiente sem mudar resultado) e todos os metadados. Linhas em branco e espaços no fim da
    linha também não contam: dependem de como a execução quebrou a saída em pedaços, não do
    resultado.
    """
    linhas: list[str] = []
    for indice, celula in enumerate(notebook.get("cells", [])):
        if celula.get("cell_type") != "code":
            continue
        for saida in celula.get("outputs", []):
            tipo = saida.get("output_type")
            if tipo == "stream" and saida.get("name") == "stdout":
                texto = "".join(saida.get("text", ""))
            elif tipo in ("execute_result", "display_data") and "text/plain" in saida.get(
                "data", {}
            ):
                texto = "".join(saida["data"]["text/plain"])
            elif tipo == "error":
                texto = f"ERRO {saida.get('ename')}: {saida.get('evalue')}"
            else:
                continue
            texto = _ENDERECO.sub(" at 0x…", texto)
            linhas.extend(
                f"[célula {indice}] {linha.rstrip()}"
                for linha in texto.splitlines()
                if linha.strip()
            )
    return linhas


def diferencas(salvo: dict, reexecutado: dict, nome: str) -> list[str]:
    """Diff unificado entre as saídas de texto; vazio quando batem."""
    return list(
        difflib.unified_diff(
            saidas_de_texto(salvo),
            saidas_de_texto(reexecutado),
            fromfile=f"{nome} (salvo)",
            tofile=f"{nome} (reexecutado)",
            lineterm="",
        )
    )


def reexecutar(caminho: Path, destino: Path) -> dict:
    """Executa uma cópia do notebook em `destino` e devolve o resultado lido."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=1800",
            "--output-dir",
            str(destino),
            str(caminho),
        ],
        check=True,
        cwd=RAIZ,
    )
    return json.loads((destino / caminho.name).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    # O console do Windows não é UTF-8 por padrão; sem isto, os acentos do relatório saem
    # quebrados. Em Linux (a CI) não muda nada.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebooks",
        nargs="*",
        default=list(PADRAO),
        help="nomes sem extensão, de notebooks/ (padrão: todos menos a análise causal)",
    )
    args = parser.parse_args(argv)

    divergentes = []
    with tempfile.TemporaryDirectory() as temporario:
        for nome in args.notebooks:
            caminho = RAIZ / "notebooks" / f"{nome}.ipynb"
            salvo = json.loads(caminho.read_text(encoding="utf-8"))
            diff = diferencas(salvo, reexecutar(caminho, Path(temporario)), nome)
            if diff:
                divergentes.append(nome)
                print(f"\n{nome}: as saídas reexecutadas divergem das salvas")
                print("\n".join(diff[:60]))
                if len(diff) > 60:
                    print(f"... e mais {len(diff) - 60} linhas de diferença")
            else:
                print(f"{nome}: saídas idênticas às salvas")

    if divergentes:
        print(
            f"\n{len(divergentes)} notebook(s) desatualizado(s): {', '.join(divergentes)}. "
            "Reexecute-os contra o modelo publicado (make eda baseline equidade "
            "explicabilidade, em Linux) e commite as saídas."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
